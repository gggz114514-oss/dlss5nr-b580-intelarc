// Isolated RE8 input-format candidate. The frozen product bridge remains
// unchanged; D3D12's BGRA8 SRV presents logical RGB channels to the pack CS.
#define NOMINMAX
#include "nr_texture_bridge_v1.h"
#include "nr_gpu_handoff_lease.h"
#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include "ze_api.h"
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <array>
#include <algorithm>
#include <limits>

using Microsoft::WRL::ComPtr;
namespace sx=sycl::ext::oneapi::experimental;
struct ContractError:std::runtime_error { using std::runtime_error::runtime_error; };
static void require(bool ok,const char* why) { if(!ok) throw ContractError(why); }
static void hr(HRESULT value,const char* why) {
    if(FAILED(value)) throw std::runtime_error(std::string(why)+": HRESULT="+std::to_string(uint32_t(value)));
}
static void message(char* dest,uint32_t size,const char* text) {
    if(dest && size) strncpy_s(dest,size,text,_TRUNCATE);
}
static std::string shader_source() {
    std::ostringstream text;
    // Canonical RGB8->float32 table: keep the exact numpy RGB8/255 boundary.
    // UNORM SRV loads are rounded back to their integer code before lookup.
    text<<"static const uint RGB8[256]={";
    for(unsigned i=0;i<256;++i) { float v=float(double(i)/255.0); uint32_t bits; memcpy(&bits,&v,4); if(i)text<<","; text<<bits<<"u"; }
    text<<R"(};
Texture2D<float4> Color : register(t0);
Texture2D<float2> Motion : register(t1);
ByteAddressBuffer Source : register(t2);
RWByteAddressBuffer RGB : register(u0);
RWByteAddressBuffer MV : register(u1);
RWTexture2D<float4> FixtureColor : register(u2);
RWTexture2D<float2> FixtureMotion : register(u3);
RWTexture2D<float4> Output : register(u4);
cbuffer Params : register(b0) { uint W; uint H; uint Seed; uint ColorIsRGB8; float MotionScaleX; float MotionScaleY; };
[numthreads(8,8,1)] void pack(uint3 p:SV_DispatchThreadID) {
 if(p.x>=W||p.y>=H)return; uint i=p.y*W+p.x; float3 c=Color.Load(int3(p.xy,0)).rgb;
 uint3 bits=asuint(c);
 if(ColorIsRGB8) { uint3 v=(uint3)round(saturate(c)*255.0f); bits=uint3(RGB8[v.x],RGB8[v.y],RGB8[v.z]); }
 RGB.Store3(i*12,bits);
 float2 motion=Motion.Load(int3(p.xy,0))*float2(MotionScaleX,MotionScaleY);
 MV.Store2(i*8,asuint(motion));
}
[numthreads(8,8,1)] void unpack(uint3 p:SV_DispatchThreadID) {
 if(p.x>=W||p.y>=H)return;
 Output[p.xy]=float4(asfloat(Source.Load3((p.y*W+p.x)*12)),1.0f);
}
[numthreads(8,8,1)] void gather(uint3 p:SV_DispatchThreadID) {
 if(p.x>=W||p.y>=H)return;
 RGB.Store3((p.y*W+p.x)*12,asuint(Color.Load(int3(p.xy,0)).rgb));
}
[numthreads(8,8,1)] void fixture(uint3 p:SV_DispatchThreadID) {
 if(p.x>=W||p.y>=H)return;
 uint3 code=uint3(p.x*3+p.y*5+Seed*17,p.x*7+p.y*11+Seed*23,p.x*13+p.y*19+Seed*29)&255;
 FixtureColor[p.xy]=float4(asfloat(uint3(RGB8[code.x],RGB8[code.y],RGB8[code.z])),1.0f);
 FixtureMotion[p.xy]=float2((int(p.x%17)-8)*0.125f+Seed*0.0625f,(int(p.y%19)-9)*0.125f-Seed*0.0625f);
}
)";
    return text.str();
}
static ComPtr<ID3DBlob> compile_shader(const std::string& source,const char* entry) {
    ComPtr<ID3DBlob> code,error;
    auto rc=D3DCompile(source.data(),source.size(),"nr-texture-v1",nullptr,nullptr,entry,"cs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3|D3DCOMPILE_IEEE_STRICTNESS,0,&code,&error);
    if(FAILED(rc)) throw std::runtime_error(error?static_cast<char*>(error->GetBufferPointer()):"D3DCompile failed");
    return code;
}
static LUID native_luid(sycl::queue& queue) {
    require(queue.get_backend()==sycl::backend::ext_oneapi_level_zero,"Expected Level Zero XPU queue");
    auto loader=GetModuleHandleW(L"ze_loader.dll"); require(loader!=nullptr,"Level Zero loader missing");
    auto get=reinterpret_cast<decltype(&zeDeviceGetProperties)>(GetProcAddress(loader,"zeDeviceGetProperties"));
    require(get!=nullptr,"Level Zero properties API missing");
    ze_device_luid_ext_properties_t luid{}; luid.stype=ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES;
    ze_device_properties_t props{}; props.stype=ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES; props.pNext=&luid;
    auto rc=get(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device()),&props);
    require(rc==ZE_RESULT_SUCCESS && props.vendorId==0x8086 && luid.nodeMask==1,"Unsupported Level Zero adapter/node mask");
    LUID result{}; static_assert(sizeof(result)==sizeof(luid.luid)); memcpy(&result,&luid.luid,sizeof(result));
    require(result.LowPart || result.HighPart,"Invalid adapter LUID"); return result;
}
static bool same_luid(LUID a,LUID b) { return a.LowPart==b.LowPart && a.HighPart==b.HighPart; }
static void barrier(ID3D12GraphicsCommandList* list,ID3D12Resource* resource,D3D12_RESOURCE_STATES from,D3D12_RESOURCE_STATES to) {
    if(from==to)return;
    D3D12_RESOURCE_BARRIER b{}; b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition={resource,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,from,to}; list->ResourceBarrier(1,&b);
}
struct Shared {
    sycl::queue* queue=nullptr;
    ComPtr<ID3D12Resource> resource;
    HANDLE handle=nullptr; sx::external_mem imported{}; void* ptr=nullptr; bool valid=false;
    uint64_t bytes=0;
    void init(ID3D12Device* device,sycl::queue* q,uint64_t length) {
        queue=q;bytes=length;
        D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;d.Width=bytes;d.Height=1;
        d.DepthOrArraySize=1;d.MipLevels=1;d.SampleDesc.Count=1;d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;d.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;heap.CreationNodeMask=heap.VisibleNodeMask=1;
        hr(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_SHARED,&d,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&resource)),"shared allocation");
        hr(device->CreateSharedHandle(resource.Get(),nullptr,GENERIC_ALL,nullptr,&handle),"shared NT handle");
        auto bytes_allocated=device->GetResourceAllocationInfo(0,1,&d).SizeInBytes;
        sx::external_mem_descriptor<sx::resource_win32_handle> desc{{handle},sx::external_mem_handle_type::win32_nt_dx12_resource,size_t(bytes_allocated)};
        imported=sx::import_external_memory(desc,*q);valid=true;
        ptr=sx::map_external_linear_memory(imported,0,bytes,*q);
        require(ptr!=nullptr,"null imported device pointer");
    }
    void cleanup() {
        if(ptr){sx::unmap_external_linear_memory(ptr,*queue);ptr=nullptr;}
        if(valid){sx::release_external_memory(imported,*queue);valid=false;}
    }
    ~Shared() {
        try {cleanup();}catch(...){}
        if(handle)CloseHandle(handle);
    }
};
struct HandoffCommands {
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    void init(ID3D12Device* device) {
        hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)),"handoff allocator");
        hr(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&list)),"handoff list");
        hr(list->Close(),"handoff initial close");
    }
    void begin() {
        hr(allocator->Reset(),"handoff allocator reset");
        hr(list->Reset(allocator.Get(),nullptr),"handoff list reset");
    }
};
struct HandoffImport {
    HANDLE handle=nullptr;
    sx::external_semaphore semaphore{};
    bool valid=false;
    void init(ID3D12Device* device,ID3D12Fence* fence,sycl::queue& queue) {
        require(!valid && !handle,"External semaphore lease still owned");
        hr(device->CreateSharedHandle(fence,nullptr,GENERIC_ALL,nullptr,&handle),"shared handoff fence handle");
        sx::external_semaphore_descriptor<sx::resource_win32_handle> desc{{handle},sx::external_semaphore_handle_type::win32_nt_dx12_fence};
        semaphore=sx::import_external_semaphore(desc,queue);valid=true;
    }
    // Only invoked after the exact lease proof, or while pristine/unsubmitted.
    void release(sycl::queue& queue) {
        if(valid){sx::release_external_semaphore(semaphore,queue);valid=false;}
        if(handle){CloseHandle(handle);handle=nullptr;}
    }
};
struct TextureBridge {
    // A queue copy shares Torch's implementation/context. It retains those
    // owners even if the Python stream wrapper goes away during quarantine.
    sycl::queue xpu_owner;
    sycl::queue* xpu;
    uint64_t borrowed_queue;
    DWORD owner_thread=GetCurrentThreadId();
    uint32_t width,height;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> ready,retirement;
    ComPtr<ID3D12DescriptorHeap> heap;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pack,unpack,gather,fixture;
    ComPtr<ID3D12Resource> output,color_hold,motion_hold,fixture_color,fixture_motion;
    ComPtr<ID3D12Resource> fixture_re8_color,fixture_re8_motion;
    Shared rgb,mv,result;
    HANDLE event=nullptr;
    uint32_t stride=0;
    uint64_t ready_value=0,retirement_value=0,pending_id=0,last_id=0;
    bool pending=false,history=false,output_ready=false,output_borrowed=false,failed=false;
    DXGI_FORMAT fixture_mv_format=DXGI_FORMAT_UNKNOWN;
    bool gpu_handoff=false,handoff_initialized=false,handoff_attempted=false;
    bool producer_fence_required=false;
    nr_handoff::Lease lease;
    HandoffCommands pack_commands,unpack_commands;
    ComPtr<ID3D12Fence> handoff_ready,handoff_backward,producer_hold;
    HandoffImport forward_import,backward_import;
    std::array<sycl::event,nr_handoff::Lease::event_capacity> handoff_events;
    uint64_t handoff_value=0,backward_value=0,work_epoch=0;
    uint64_t producer_value_hold=0;
    NR_TextureHandoffStats counters{};
    TextureBridge(sycl::queue* q,uint32_t w,uint32_t h):xpu_owner(*q),xpu(&xpu_owner),borrowed_queue(uint64_t(reinterpret_cast<uintptr_t>(q))),width(w),height(h){
        require(*xpu==*q && xpu->get_context()==q->get_context(),"Borrowed queue/context changed");
    }
    ~TextureBridge(){if(event)CloseHandle(event);}
    void init(ID3D12Device* external_device,ID3D12CommandQueue* external_queue) {
        require(xpu && width && height && width<=4096 && height<=4096,"Invalid XPU queue or geometry");
        require(bool(external_device)==bool(external_queue),"Pass both native device and queue");
        LUID wanted=native_luid(*xpu);
        if(external_device) {
            require(same_luid(wanted,external_device->GetAdapterLuid()),"XPU/native adapter mismatch");
            device=external_device;queue=external_queue;
            ComPtr<ID3D12Device> owned;hr(queue->GetDevice(IID_PPV_ARGS(&owned)),"queue device");
            require(owned.Get()==device.Get() && queue->GetDesc().Type==D3D12_COMMAND_LIST_TYPE_DIRECT,"Queue/device mismatch or unsupported type");
        } else {
            ComPtr<IDXGIFactory4> f;ComPtr<IDXGIAdapter1> a;
            hr(CreateDXGIFactory2(0,IID_PPV_ARGS(&f)),"DXGI factory");
            hr(f->EnumAdapterByLuid(wanted,IID_PPV_ARGS(&a)),"adapter LUID");
            hr(D3D12CreateDevice(a.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&device)),"D3D12 device");
            D3D12_COMMAND_QUEUE_DESC d{};d.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
            hr(device->CreateCommandQueue(&d,IID_PPV_ARGS(&queue)),"queue");
        }
        hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)),"allocator");
        hr(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&list)),"list");
        hr(list->Close(),"initial close");
        hr(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&ready)),"ready fence");
        event=CreateEventW(nullptr,FALSE,FALSE,nullptr);require(event!=nullptr,"fence event");
        rgb.init(device.Get(),xpu,uint64_t(width)*height*12);
        mv.init(device.Get(),xpu,uint64_t(width)*height*8);
        result.init(device.Get(),xpu,rgb.bytes);
        output=texture(DXGI_FORMAT_R32G32B32A32_FLOAT);
        D3D12_DESCRIPTOR_HEAP_DESC hd{};hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;hd.NumDescriptors=8;hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        hr(device->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap)),"descriptor heap");
        stride=device->GetDescriptorHandleIncrementSize(hd.Type);
        D3D12_DESCRIPTOR_RANGE ranges[2]{};
        ranges[0]={D3D12_DESCRIPTOR_RANGE_TYPE_SRV,3,0,0,0};ranges[1]={D3D12_DESCRIPTOR_RANGE_TYPE_UAV,5,0,0,0};
        D3D12_ROOT_PARAMETER params[3]{};
        for(int i=0;i<2;++i){params[i].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;params[i].DescriptorTable={1,&ranges[i]};}
        params[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;params[2].Constants={0,0,6};
        D3D12_ROOT_SIGNATURE_DESC rs{};rs.NumParameters=3;rs.pParameters=params;
        ComPtr<ID3DBlob> code,errors;hr(D3D12SerializeRootSignature(&rs,D3D_ROOT_SIGNATURE_VERSION_1,&code,&errors),"root serialization");
        hr(device->CreateRootSignature(0,code->GetBufferPointer(),code->GetBufferSize(),IID_PPV_ARGS(&root)),"root signature");
        auto source=shader_source();const char* names[]={"pack","unpack","gather","fixture"};
        ComPtr<ID3D12PipelineState>* psos[]={std::addressof(pack),std::addressof(unpack),std::addressof(gather),std::addressof(fixture)};
        for(int i=0;i<4;++i){auto blob=compile_shader(source,names[i]);D3D12_COMPUTE_PIPELINE_STATE_DESC p{};p.pRootSignature=root.Get();p.CS={blob->GetBufferPointer(),blob->GetBufferSize()};hr(device->CreateComputePipelineState(&p,IID_PPV_ARGS(psos[i]->GetAddressOf())),"compute PSO");}
        srv(nullptr,DXGI_FORMAT_R32G32B32A32_FLOAT,0);srv(nullptr,DXGI_FORMAT_R32G32_FLOAT,1);
        raw_srv(result.resource.Get(),result.bytes,2);raw_uav(rgb.resource.Get(),rgb.bytes,3);raw_uav(mv.resource.Get(),mv.bytes,4);
        uav(nullptr,DXGI_FORMAT_R8G8B8A8_UNORM,5);uav(nullptr,DXGI_FORMAT_R16G16_FLOAT,6);
        uav(output.Get(),DXGI_FORMAT_R32G32B32A32_FLOAT,7);
    }
    ComPtr<ID3D12Resource> texture(DXGI_FORMAT format) {
        D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;d.Width=width;d.Height=height;d.DepthOrArraySize=1;d.MipLevels=1;d.SampleDesc.Count=1;d.Format=format;d.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        D3D12_HEAP_PROPERTIES h{};h.Type=D3D12_HEAP_TYPE_DEFAULT;h.CreationNodeMask=h.VisibleNodeMask=1;
        ComPtr<ID3D12Resource> t;hr(device->CreateCommittedResource(&h,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,nullptr,IID_PPV_ARGS(&t)),"texture");return t;
    }
    ComPtr<ID3D12Resource> render_texture(DXGI_FORMAT format) {
        D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;d.Width=width;d.Height=height;
        d.DepthOrArraySize=1;d.MipLevels=1;d.SampleDesc.Count=1;d.Format=format;d.Flags=D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        D3D12_HEAP_PROPERTIES h{};h.Type=D3D12_HEAP_TYPE_DEFAULT;h.CreationNodeMask=h.VisibleNodeMask=1;
        ComPtr<ID3D12Resource> t;
        hr(device->CreateCommittedResource(&h,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,nullptr,IID_PPV_ARGS(&t)),"RE8 fixture render texture");
        return t;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE cpu(uint32_t i){auto h=heap->GetCPUDescriptorHandleForHeapStart();h.ptr+=size_t(i)*stride;return h;}
    D3D12_GPU_DESCRIPTOR_HANDLE gpu(uint32_t i){auto h=heap->GetGPUDescriptorHandleForHeapStart();h.ptr+=uint64_t(i)*stride;return h;}
    void srv(ID3D12Resource* t,DXGI_FORMAT fmt,uint32_t index){D3D12_SHADER_RESOURCE_VIEW_DESC d{};d.Format=fmt;d.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;d.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;d.Texture2D.MipLevels=1;device->CreateShaderResourceView(t,&d,cpu(index));}
    void uav(ID3D12Resource* t,DXGI_FORMAT fmt,uint32_t index){D3D12_UNORDERED_ACCESS_VIEW_DESC d{};d.Format=fmt;d.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;device->CreateUnorderedAccessView(t,nullptr,&d,cpu(index));}
    void raw_srv(ID3D12Resource* t,uint64_t length,uint32_t index){D3D12_SHADER_RESOURCE_VIEW_DESC d{};d.Format=DXGI_FORMAT_R32_TYPELESS;d.ViewDimension=D3D12_SRV_DIMENSION_BUFFER;d.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;d.Buffer.NumElements=UINT(length/4);d.Buffer.Flags=D3D12_BUFFER_SRV_FLAG_RAW;device->CreateShaderResourceView(t,&d,cpu(index));}
    void raw_uav(ID3D12Resource* t,uint64_t length,uint32_t index){D3D12_UNORDERED_ACCESS_VIEW_DESC d{};d.Format=DXGI_FORMAT_R32_TYPELESS;d.ViewDimension=D3D12_UAV_DIMENSION_BUFFER;d.Buffer.NumElements=UINT(length/4);d.Buffer.Flags=D3D12_BUFFER_UAV_FLAG_RAW;device->CreateUnorderedAccessView(t,nullptr,&d,cpu(index));}
    void healthy(bool check_owner=true){require(!failed,"Bridge execution failed; close isolated worker");if(gpu_handoff && check_owner)require(GetCurrentThreadId()==owner_thread,"GPU handoff requires its owning host thread");}
    void pointer(const void* p){require(p && sycl::get_pointer_type(p,xpu->get_context())==sycl::usm::alloc::device,"Expected device pointer in caller XPU context");}
    void validate_fence(ID3D12Fence* fence,uint64_t value){
        require(bool(fence)==bool(value),"Pass a nonzero fence/value pair, or two zeros");
        if(!fence)return;
        ComPtr<ID3D12Device> owner;hr(fence->GetDevice(IID_PPV_ARGS(&owner)),"fence device");
        require(owner.Get()==device.Get(),"Fence belongs to another D3D12 device");
        require(fence!=ready.Get() || value<=ready_value,"Cannot wait for an unsignaled bridge fence value");
    }
    void wait_gpu(ID3D12Fence* fence,uint64_t value){if(value){require(fence!=nullptr,"Missing producer/consumer fence");hr(queue->Wait(fence,value),"GPU dependency wait");}}
    void begin(){hr(allocator->Reset(),"allocator reset");hr(list->Reset(allocator.Get(),nullptr),"list reset");}
    void submit(){hr(list->Close(),"list close");ID3D12CommandList* lists[]={list.Get()};queue->ExecuteCommandLists(1,lists);hr(queue->Signal(ready.Get(),++ready_value),"signal ready");hr(ready->SetEventOnCompletion(ready_value,event),"ready event");if(WaitForSingleObject(event,10000)!=WAIT_OBJECT_0)throw std::runtime_error("D3D12 completion timeout");hr(device->GetDeviceRemovedReason(),"D3D12 device removed");}
    void dispatch_on(ID3D12GraphicsCommandList* list,ID3D12PipelineState* p,uint32_t seed=0,uint32_t color8=0,float scaleX=1.0f,float scaleY=1.0f){ID3D12DescriptorHeap* heaps[]={heap.Get()};list->SetDescriptorHeaps(1,heaps);list->SetComputeRootSignature(root.Get());list->SetComputeRootDescriptorTable(0,gpu(0));list->SetComputeRootDescriptorTable(1,gpu(3));uint32_t c[]={width,height,seed,color8,0,0};memcpy(&c[4],&scaleX,4);memcpy(&c[5],&scaleY,4);list->SetComputeRoot32BitConstants(2,6,c,0);list->SetPipelineState(p);list->Dispatch((width+7)/8,(height+7)/8,1);}
    void dispatch(ID3D12PipelineState* p,uint32_t seed=0,uint32_t color8=0,float scaleX=1.0f,float scaleY=1.0f){dispatch_on(list.Get(),p,seed,color8,scaleX,scaleY);}
    void validate_texture(ID3D12Resource* resource,bool motion){
        require(resource!=nullptr,"Missing input texture");auto d=resource->GetDesc();
        require(d.Dimension==D3D12_RESOURCE_DIMENSION_TEXTURE2D && d.Width==width && d.Height==height && d.DepthOrArraySize==1 && d.MipLevels==1 && d.SampleDesc.Count==1,"Texture geometry mismatch");
        require(motion?(d.Format==DXGI_FORMAT_R16G16_FLOAT || d.Format==DXGI_FORMAT_R32G32_FLOAT || d.Format==DXGI_FORMAT_R16G16B16A16_SNORM):(d.Format==DXGI_FORMAT_R8G8B8A8_UNORM || d.Format==DXGI_FORMAT_B8G8R8A8_UNORM || d.Format==DXGI_FORMAT_R32G32B32A32_FLOAT),"Unsupported texture format");
        ComPtr<ID3D12Device> owner;hr(resource->GetDevice(IID_PPV_ARGS(&owner)),"resource device");require(owner.Get()==device.Get(),"Texture belongs to another D3D12 device");
    }
    void init_handoff() {
        if(handoff_initialized)return;
        require(xpu->has_property<sycl::property::queue::in_order>(),"GPU handoff requires the exact in-order model queue");
        pack_commands.init(device.Get());unpack_commands.init(device.Get());
        hr(device->CreateFence(0,D3D12_FENCE_FLAG_SHARED,IID_PPV_ARGS(&handoff_ready)),"shared pack ready fence");
        hr(device->CreateFence(0,D3D12_FENCE_FLAG_SHARED,IID_PPV_ARGS(&handoff_backward)),"shared backward fence");
        backward_import.init(device.Get(),handoff_backward.Get(),*xpu);
        handoff_initialized=true;
    }
    uint64_t next_value(uint64_t value) {
        require(value<std::numeric_limits<uint64_t>::max()-1,"Fence timeline exhausted");return value+1;
    }
    void issue_event() {
        require(lease.issue_event(),"Handoff event capacity/state exhausted");++work_epoch;
    }
    void retain_event(const sycl::event& event) {
        // Fixed storage: no vector growth after a potentially submitted API.
        const auto i=lease.events_recorded;
        require(i<lease.events_issued && i<handoff_events.size(),"Handoff event accounting mismatch");
        handoff_events[i]=event;require(lease.record_event(),"Handoff event retention failed");
    }
    nr_handoff::Completion observe_completion() {
        nr_handoff::Completion c{};
        hr(device->GetDeviceRemovedReason(),"device removed during handoff retirement");c.device_ok=true;
        if(lease.pack_signaled)c.pack=handoff_ready->GetCompletedValue();
        if(lease.unpack_signaled)c.unpack=ready->GetCompletedValue();
        if(lease.consumer_registered)c.consumer=retirement->GetCompletedValue();
        if(c.pack==UINT64_MAX || c.unpack==UINT64_MAX || c.consumer==UINT64_MAX)
            throw std::runtime_error("Invalid/device-lost completion value; owners quarantined");
        c.events_complete=lease.events_issued==lease.events_recorded;
        for(uint32_t i=0;i<lease.events_recorded;++i)
            c.events_complete &= handoff_events[i].get_info<sycl::info::event::command_execution_status>()==sycl::info::event_command_status::complete;
        if(c.events_complete){xpu->throw_asynchronous();c.events_success=true;}
        return c;
    }
    bool poll_handoff_idle(bool check_owner=true) {
        healthy(check_owner);if(lease.idle())return true;
        const auto completed=observe_completion();
        if(!lease.safe(completed))return false;
        // All D3D12 lists, XPU operations and (if borrowed) the caller's exact
        // consumer fence succeeded. Only now may imports and bindings retire.
        forward_import.release(*xpu);++counters.forward_releases;
        require(lease.retire(completed),"Handoff retirement proof changed");
        for(auto& e:handoff_events)e=sycl::event{};
        color_hold.Reset();motion_hold.Reset();producer_hold.Reset();retirement.Reset();
        producer_value_hold=0;
        retirement_value=0;output_borrowed=false;output_ready=false;pending=false;
        ++counters.frames_retired;return true;
    }
    bool legacy_idle() {
        healthy();if(pending)return false;
        hr(device->GetDeviceRemovedReason(),"device removed during idle check");
        const auto done=ready->GetCompletedValue();
        if(done==UINT64_MAX)throw std::runtime_error("Invalid ready completion");if(done<ready_value)return false;
        if(output_borrowed) {
            if(!retirement_value || !retirement)return false;
            const auto retired=retirement->GetCompletedValue();
            if(retired==UINT64_MAX)throw std::runtime_error("Invalid consumer completion");if(retired<retirement_value)return false;
        }
        // Legacy work already completed synchronously; retrieve only errors.
        xpu->throw_asynchronous();return true;
    }
    void set_handoff(uint32_t enabled) {
        require(GetCurrentThreadId()==owner_thread,"GPU handoff setter requires its owning host thread");
        healthy();require(enabled<=1,"GPU handoff switch must be 0/1");
        if(!(gpu_handoff?poll_handoff_idle():legacy_idle())) {
            ++counters.busy_rejections;throw ContractError("GPU handoff setter requires fully retired idle");
        }
        if(enabled){handoff_attempted=true;init_handoff();}
        // A prior legacy output has now actually retired, so the old flags and
        // consumer owner can be cleared without a speculative GPU-wait reset.
        if(!gpu_handoff){retirement.Reset();retirement_value=0;output_borrowed=false;output_ready=false;}
        gpu_handoff=enabled!=0;
    }
    void transfer_owner(uint64_t previous,uint64_t model_queue,uint64_t game_queue,uint32_t* adopted) {
        require(adopted!=nullptr,"Null owner transfer result");*adopted=0;
        require(previous==uint64_t(owner_thread),"Owner transfer previous thread mismatch");
        require(model_queue==borrowed_queue && game_queue==uint64_t(reinterpret_cast<uintptr_t>(queue.Get())),
            "Owner transfer borrowed queue proof mismatch");
        require(!failed && !lease.poisoned,"Failed owner cannot transfer; retain all owners");
        // The caller serializes process/retire on the same adapter lock. This
        // is the ONLY explicit cross-thread entry: it proves actual consumer,
        // D3D12 and XPU event completion before replacing the native owner.
        // An in-flight result is not an error and leaves ownership untouched.
        if(!(gpu_handoff?poll_handoff_idle(false):legacy_idle()))return;
        owner_thread=GetCurrentThreadId();*adopted=1;
    }
    void prepare_handoff(ID3D12Resource* color,ID3D12Resource* motion,ID3D12Fence* producer,uint64_t value,uint64_t id,uint64_t previous,bool reset,void* color_ptr,void* motion_ptr) {
        healthy();require(!pending,"Previous NR frame not exported");
        require(reset || (history && previous==last_id && id>last_id),"First/discontinuous frame requires reset or matching previous frame ID");
        validate_texture(color,false);validate_texture(motion,true);pointer(color_ptr);pointer(motion_ptr);validate_fence(producer,value);
        require(value!=UINT64_MAX,"Invalid producer value");
        require(!producer_fence_required || (producer && value),"Bypassing prepared CPU wait requires the exact source producer fence/value");
        if(!poll_handoff_idle()){++counters.busy_rejections;throw ContractError("Previous frame is not actually retired; retry after consumer completion");}
        const auto signal_value=next_value(handoff_value);
        require(lease.begin(id),"Handoff lease unavailable");
        color_hold=color;motion_hold=motion;producer_hold=producer;producer_value_hold=value;
        pending_id=id;pending=true;output_ready=false;
        ++counters.frames_started;
        // v5 requirement: a distinct retained external import for EVERY pack
        // wait, even though all frames share this same underlying DX12 fence.
        forward_import.init(device.Get(),handoff_ready.Get(),*xpu);++counters.forward_imports;
        require(lease.reset_pack_allowed(),"Pack record still owned by a frame");pack_commands.begin();
        auto* commands=pack_commands.list.Get();
        srv(color_hold.Get(),color_hold->GetDesc().Format,0);srv(motion_hold.Get(),motion_hold->GetDesc().Format,1);
        barrier(commands,rgb.resource.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        barrier(commands,mv.resource.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        const bool snorm=motion_hold->GetDesc().Format==DXGI_FORMAT_R16G16B16A16_SNORM;
        dispatch_on(commands,pack.Get(),0,color_hold->GetDesc().Format==DXGI_FORMAT_R8G8B8A8_UNORM || color_hold->GetDesc().Format==DXGI_FORMAT_B8G8R8A8_UNORM,snorm?float(width)*0.5f:1.0f,snorm?-float(height)*0.5f:1.0f);
        barrier(commands,rgb.resource.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COMMON);
        barrier(commands,mv.resource.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COMMON);
        hr(commands->Close(),"pack list close");
        // Mark before APIs whose exceptions/failure may leave queued owners.
        ++work_epoch;lease.pack_possible=true;wait_gpu(producer,value);
        ID3D12CommandList* lists[]={commands};queue->ExecuteCommandLists(1,lists);++counters.pack_submits;
        hr(queue->Signal(handoff_ready.Get(),signal_value),"pack Signal before XPU relay");
        handoff_value=signal_value;lease.pack_value=signal_value;lease.pack_signaled=true;
        issue_event();const auto wait=xpu->ext_oneapi_wait_external_semaphore(forward_import.semaphore,signal_value);
        retain_event(wait);++counters.xpu_waits;
        issue_event();retain_event(xpu->memcpy(color_ptr,rgb.ptr,size_t(rgb.bytes),wait));
        issue_event();retain_event(xpu->memcpy(motion_ptr,mv.ptr,size_t(mv.bytes),wait));
    }
    void export_handoff(uint64_t id,const void* color) {
        healthy();require(pending && id==pending_id,"NR output frame ID mismatch or no pending frame");pointer(color);
        const auto signal_value=next_value(backward_value), output_value=next_value(ready_value);
        require(lease.reset_unpack_allowed(),"Unpack record still owned by a frame");unpack_commands.begin();
        auto* commands=unpack_commands.list.Get();
        barrier(commands,result.resource.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        barrier(commands,output.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatch_on(commands,unpack.Get());
        barrier(commands,output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        barrier(commands,result.resource.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COMMON);
        hr(commands->Close(),"unpack list close");
        // The exact in-order model queue orders this copy after NR, without a
        // host wait. The returned event and the input tensor remain owned.
        issue_event();const auto copied=xpu->memcpy(result.ptr,color,size_t(result.bytes));retain_event(copied);
        // Pack Execute+Signal was already submitted in prepare_handoff, before
        // this signal API which may itself block in the external runtime.
        issue_event();retain_event(xpu->ext_oneapi_signal_external_semaphore(backward_import.semaphore,signal_value,copied));
        backward_value=signal_value;++counters.xpu_signals;
        ++work_epoch;lease.unpack_possible=true;
        hr(queue->Wait(handoff_backward.Get(),signal_value),"unpack backward GPU Wait");
        ID3D12CommandList* lists[]={commands};queue->ExecuteCommandLists(1,lists);++counters.unpack_submits;
        hr(queue->Signal(ready.Get(),output_value),"async output ready Signal");
        ready_value=output_value;lease.unpack_value=output_value;lease.unpack_signaled=true;
        lease.phase=nr_handoff::Phase::exported;last_id=id;history=true;pending=false;output_ready=true;
    }
    void close_handoff() {
        require(GetCurrentThreadId()==owner_thread,"GPU handoff close requires its owning host thread");
        require(!failed && !lease.poisoned,"Failed bridge is quarantined; retain all owners until worker exit");
        require(!output_borrowed || retirement_value,"Cannot close while an output consumer is unregistered");
        if(lease.phase==nr_handoff::Phase::prepared && !lease.closing) {
            // A model may have submitted work using prepared inputs even when
            // export was never called. Explicit close captures that queue tail.
            issue_event();retain_event(xpu->ext_oneapi_submit_barrier());lease.closing=true;
        }
        const auto deadline=GetTickCount64()+10000;
        while(!poll_handoff_idle()) {
            if(GetTickCount64()>=deadline)throw std::runtime_error("Explicit close drain timeout; owners quarantined");
            Sleep(1); // explicit bounded shutdown only; never a frame handoff
        }
        backward_import.release(*xpu);forward_import.release(*xpu);
        rgb.cleanup();mv.cleanup();result.cleanup();
    }
    void handoff_stats(NR_TextureHandoffStats* stats) {
        require(GetCurrentThreadId()==owner_thread,"GPU handoff stats requires its owning host thread");
        require(stats && stats->abi_size==sizeof(*stats),"GPU handoff stats ABI size mismatch");
        auto s=counters;s.abi_size=sizeof(s);s.version=1;s.enabled=gpu_handoff;s.active=!lease.idle();
        s.poisoned=lease.poisoned || failed;s.phase=uint32_t(lease.phase);s.records=handoff_initialized?2:0;
        s.forward_live=forward_import.valid;s.pack_value=handoff_value;s.output_value=ready_value;s.backward_value=backward_value;
        s.retained_bytes=rgb.bytes+mv.bytes+result.bytes+uint64_t(width)*height*16;
        s.borrowed_queue=borrowed_queue;
        s.native_context=uint64_t(reinterpret_cast<uintptr_t>(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(xpu->get_context())));
        s.native_device=uint64_t(reinterpret_cast<uintptr_t>(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(xpu->get_device())));
        *stats=s;
    }
    void configure_handoff(const NR_TextureGpuHandoffConfig* config) {
        require(config && config->abi_size==sizeof(*config) && config->version==1 &&
            config->enabled<=1 && config->producer_fence_required<=1,"GPU handoff config ABI/value mismatch");
        require(config->borrowed_sycl_queue==borrowed_queue &&
            config->borrowed_d3d12_queue==uint64_t(reinterpret_cast<uintptr_t>(queue.Get())),"GPU handoff config borrowed queue proof mismatch");
        set_handoff(config->enabled);producer_fence_required=config->producer_fence_required!=0;
    }
    void handoff_info(NR_TextureGpuHandoffInfo* info) {
        require(GetCurrentThreadId()==owner_thread,"GPU handoff info requires its owning host thread");
        require(info && info->abi_size==sizeof(*info),"GPU handoff info ABI size mismatch");
        NR_TextureGpuHandoffInfo i{};i.abi_size=sizeof(i);i.version=1;i.enabled=gpu_handoff;
        i.poisoned=lease.poisoned || failed;i.healthy=!i.poisoned;
        i.reuse_safe_idle=i.healthy && (gpu_handoff?poll_handoff_idle():legacy_idle());
        i.active=gpu_handoff?!lease.idle():!i.reuse_safe_idle;
        i.queue_context_equal=1;i.in_order=xpu->has_property<sycl::property::queue::in_order>();
        i.native_luid_matched=1;i.max_active_frames=1;i.private_records=handoff_initialized?2:0;
        i.forward_import_per_wait=1;i.producer_cpu_wait_bypass_supported=i.in_order && i.queue_context_equal && i.native_luid_matched;
        i.producer_fence_required=producer_fence_required;
        i.producer_cpu_wait_bypass_configured=i.enabled && i.healthy && i.producer_fence_required && i.producer_cpu_wait_bypass_supported;
        i.borrowed_sycl_queue=borrowed_queue;
        i.native_context=uint64_t(reinterpret_cast<uintptr_t>(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(xpu->get_context())));
        i.native_device=uint64_t(reinterpret_cast<uintptr_t>(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(xpu->get_device())));
        i.d3d12_device=uint64_t(reinterpret_cast<uintptr_t>(device.Get()));i.d3d12_queue=uint64_t(reinterpret_cast<uintptr_t>(queue.Get()));
        i.active_frame=lease.id;i.retained_producer_fence=uint64_t(reinterpret_cast<uintptr_t>(producer_hold.Get()));i.retained_producer_value=producer_value_hold;
        *info=i;
    }
    void read_inputs(void* color_ptr,void* motion_ptr){
        pointer(color_ptr);pointer(motion_ptr);require(color_hold && motion_hold,"No retained input textures");
        srv(color_hold.Get(),color_hold->GetDesc().Format,0);srv(motion_hold.Get(),motion_hold->GetDesc().Format,1);
        begin();barrier(list.Get(),rgb.resource.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);barrier(list.Get(),mv.resource.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        const bool snorm=motion_hold->GetDesc().Format==DXGI_FORMAT_R16G16B16A16_SNORM;
        // RE8 probe measured (480,-270) at 960x540. Its SNORM velocity
        // half-source scaling at other sizes is the candidate to validate in
        // live motion. Other games/formats require a separate motion contract.
        dispatch(pack.Get(),0,color_hold->GetDesc().Format==DXGI_FORMAT_R8G8B8A8_UNORM || color_hold->GetDesc().Format==DXGI_FORMAT_B8G8R8A8_UNORM,snorm?float(width)*0.5f:1.0f,snorm?-float(height)*0.5f:1.0f);
        barrier(list.Get(),rgb.resource.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COMMON);barrier(list.Get(),mv.resource.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COMMON);submit();
        xpu->memcpy(color_ptr,rgb.ptr,size_t(rgb.bytes)).wait_and_throw();xpu->memcpy(motion_ptr,mv.ptr,size_t(mv.bytes)).wait_and_throw();
    }
    void prepare(ID3D12Resource* color,ID3D12Resource* motion,ID3D12Fence* producer,uint64_t value,uint64_t id,uint64_t previous,bool reset,void* color_ptr,void* motion_ptr){
        if(gpu_handoff){prepare_handoff(color,motion,producer,value,id,previous,reset,color_ptr,motion_ptr);return;}
        healthy();require(!pending,"Previous NR frame not exported");require(!output_borrowed || retirement_value,"Downstream consumer completion must be registered before reuse");
        require(reset || (history && previous==last_id && id>last_id),"First/discontinuous frame requires reset or matching previous frame ID");
        validate_texture(color,false);validate_texture(motion,true);pointer(color_ptr);pointer(motion_ptr);validate_fence(producer,value);
        xpu->wait_and_throw();
        wait_gpu(retirement.Get(),retirement_value);wait_gpu(producer,value);
        color_hold=color;motion_hold=motion;read_inputs(color_ptr,motion_ptr);
        retirement.Reset();retirement_value=0;output_borrowed=false;pending_id=id;pending=true;
    }
    void export_frame(uint64_t id,const void* color){
        if(gpu_handoff){export_handoff(id,color);return;}
        healthy();require(pending && id==pending_id,"NR output frame ID mismatch or no pending frame");pointer(color);
        xpu->wait_and_throw();xpu->memcpy(result.ptr,color,size_t(result.bytes)).wait_and_throw();
        begin();barrier(list.Get(),result.resource.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);barrier(list.Get(),output.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatch(unpack.Get());barrier(list.Get(),output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);barrier(list.Get(),result.resource.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COMMON);submit();
        last_id=id;history=true;pending=false;output_ready=true;
    }
    void make_fixture(uint32_t seed,uint32_t bits,void** color,void** motion){
        if(gpu_handoff)require(poll_handoff_idle(),"Cannot rebind fixture descriptors while a lease is active");
        healthy();require(!pending,"Cannot replace a pending source");require(bits==16 || bits==32,"Fixture motion format must be 16/32");require(color && motion,"Null fixture result");
        if(!fixture_color)fixture_color=texture(DXGI_FORMAT_R8G8B8A8_UNORM);
        auto format=bits==16?DXGI_FORMAT_R16G16_FLOAT:DXGI_FORMAT_R32G32_FLOAT;
        if(fixture_mv_format!=format){fixture_motion=texture(format);fixture_mv_format=format;}
        uav(fixture_color.Get(),DXGI_FORMAT_R8G8B8A8_UNORM,5);uav(fixture_motion.Get(),format,6);
        begin();barrier(list.Get(),fixture_color.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);barrier(list.Get(),fixture_motion.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatch(fixture.Get(),seed);barrier(list.Get(),fixture_color.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);barrier(list.Get(),fixture_motion.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);submit();*color=fixture_color.Get();*motion=fixture_motion.Get();
    }
    void make_re8_fixture(void** color,void** motion){
        if(gpu_handoff)require(poll_handoff_idle(),"Cannot rebind fixture descriptors while a lease is active");
        healthy();require(!pending && color && motion,"Cannot create RE8 fixture now");
        // Synthetic BGRA/SNORM fixture follows the selected bridge geometry.
        if(!fixture_re8_color)fixture_re8_color=render_texture(DXGI_FORMAT_B8G8R8A8_UNORM);
        if(!fixture_re8_motion)fixture_re8_motion=render_texture(DXGI_FORMAT_R16G16B16A16_SNORM);
        D3D12_DESCRIPTOR_HEAP_DESC desc{};desc.Type=D3D12_DESCRIPTOR_HEAP_TYPE_RTV;desc.NumDescriptors=2;
        ComPtr<ID3D12DescriptorHeap> rtvs;hr(device->CreateDescriptorHeap(&desc,IID_PPV_ARGS(&rtvs)),"RE8 fixture RTV heap");
        auto c=rtvs->GetCPUDescriptorHandleForHeapStart();
        auto m=c;m.ptr+=device->GetDescriptorHandleIncrementSize(desc.Type);
        device->CreateRenderTargetView(fixture_re8_color.Get(),nullptr,c);
        device->CreateRenderTargetView(fixture_re8_motion.Get(),nullptr,m);
        const float rgb[4]={.25f,.5f,.75f,1.0f};
        const float flow[4]={.5f,-.25f,0.0f,0.0f};
        begin();barrier(list.Get(),fixture_re8_color.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_RENDER_TARGET);
        barrier(list.Get(),fixture_re8_motion.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_RENDER_TARGET);
        list->ClearRenderTargetView(c,rgb,0,nullptr);list->ClearRenderTargetView(m,flow,0,nullptr);
        barrier(list.Get(),fixture_re8_color.Get(),D3D12_RESOURCE_STATE_RENDER_TARGET,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        barrier(list.Get(),fixture_re8_motion.Get(),D3D12_RESOURCE_STATE_RENDER_TARGET,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        submit();*color=fixture_re8_color.Get();*motion=fixture_re8_motion.Get();
    }
    void audit_output(void* target){
        require(!gpu_handoff,"GPU handoff probe must consume output with its returned ready fence");
        healthy();require(output_ready && !pending,"No completed output to audit");pointer(target);
        srv(output.Get(),DXGI_FORMAT_R32G32B32A32_FLOAT,0);begin();barrier(list.Get(),rgb.resource.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);dispatch(gather.Get());barrier(list.Get(),rgb.resource.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COMMON);submit();xpu->memcpy(target,rgb.ptr,size_t(rgb.bytes)).wait_and_throw();
    }
    void close(){
        if(handoff_attempted && failed)throw ContractError("GPU handoff failure quarantined all owners, including partial initialization; stop isolated worker");
        if(gpu_handoff){close_handoff();return;}
        require(!output_borrowed || retirement_value,"Cannot close while an output consumer is unregistered");
        wait_gpu(retirement.Get(),retirement_value);begin();submit();xpu->wait_and_throw();
        // The setter permits disabling only after exact retirement. These
        // optional imports therefore have no pending users in legacy close.
        backward_import.release(*xpu);forward_import.release(*xpu);
    }
};
template<class F> static int invoke(void* h,char* error,uint32_t size,F&& function){
    if(!h){message(error,size,"Null bridge");return 1;}auto* b=static_cast<TextureBridge*>(h);
    const auto before=b->work_epoch;
    try{function(*b);return 0;}
    catch(const ContractError& e){if(b->work_epoch!=before){b->failed=true;b->lease.poison();}message(error,size,e.what());return 1;}
    catch(const std::exception& e){b->failed=true;if(b->handoff_attempted)b->lease.poison();message(error,size,e.what());return 1;}
    catch(...){b->failed=true;b->lease.poison();message(error,size,"Unknown bridge exception; owners quarantined");return 1;}
}
void* nr_texture_create(void* q,void* d,void* queue,uint32_t w,uint32_t h,char* e,uint32_t n){try{require(q!=nullptr,"Invalid XPU queue");auto b=std::make_unique<TextureBridge>(static_cast<sycl::queue*>(q),w,h);b->init(static_cast<ID3D12Device*>(d),static_cast<ID3D12CommandQueue*>(queue));return b.release();}catch(const std::exception& x){message(e,n,x.what());return nullptr;}catch(...){message(e,n,"Unknown bridge creation exception");return nullptr;}}
int nr_texture_prepare(void* h,void* c,void* m,void* f,uint64_t v,uint64_t id,uint64_t prev,uint32_t reset,void* rgb,void* mv,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){b.prepare(static_cast<ID3D12Resource*>(c),static_cast<ID3D12Resource*>(m),static_cast<ID3D12Fence*>(f),v,id,prev,reset!=0,rgb,mv);});}
int nr_texture_export(void* h,uint64_t id,const void* rgb,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){b.export_frame(id,rgb);});}
int nr_texture_output(void* h,void** r,void** f,uint64_t* v,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){b.healthy();require(b.output_ready && !b.pending && !b.output_borrowed && r && f && v,"Output unavailable, already borrowed, or null return pointer");if(b.gpu_handoff)require(b.lease.borrow(),"Handoff output lease unavailable");*r=b.output.Get();*f=b.ready.Get();*v=b.ready_value;b.output_borrowed=true;});}
int nr_texture_retire(void* h,void* f,uint64_t v,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){b.healthy();require(b.output_borrowed && f && v && !b.retirement_value,"Missing/duplicate output retirement");auto* fence=static_cast<ID3D12Fence*>(f);b.validate_fence(fence,v);if(b.gpu_handoff){require(fence!=b.ready.Get() && fence!=b.handoff_ready.Get() && fence!=b.handoff_backward.Get(),"Use the caller consumer completion fence, not a bridge production fence");require(b.lease.register_consumer(v),"Invalid handoff consumer registration");++b.counters.consumer_registrations;}b.retirement=fence;b.retirement_value=v;});}
int nr_texture_close(void* h,char* e,uint32_t n){int rc=invoke(h,e,n,[](auto& b){b.close();});if(!rc)delete static_cast<TextureBridge*>(h);return rc;}
int nr_texture_fixture(void* h,uint32_t seed,uint32_t bits,void** c,void** m,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){b.make_fixture(seed,bits,c,m);});}
int nr_texture_fixture_re8(void* h,void** c,void** m,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){b.make_re8_fixture(c,m);});}
int nr_texture_audit_output(void* h,void* p,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){b.audit_output(p);});}
int nr_texture_audit_inputs(void* h,void* p,void* m,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){b.healthy();require(!b.gpu_handoff,"Use prepared tensors for GPU handoff input auditing");b.read_inputs(p,m);});}
int nr_texture_compile_check(char* e,uint32_t n){try{auto s=shader_source();for(auto name:{"pack","unpack","gather","fixture"})compile_shader(s,name);return 0;}catch(const std::exception& x){message(e,n,x.what());return 1;}}
int nr_texture_test_context(void* h,void** d,void** q,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){b.healthy();require(d && q,"Null context result");*d=b.device.Get();*q=b.queue.Get();});}
int nr_texture_set_gpu_handoff(void* h,uint32_t enabled,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){b.set_handoff(enabled);});}
int nr_texture_poll_idle(void* h,uint32_t* idle,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){require(idle!=nullptr,"Null idle result");*idle=(b.gpu_handoff?b.poll_handoff_idle():b.legacy_idle())?1u:0u;});}
int nr_texture_gpu_handoff_stats(void* h,NR_TextureHandoffStats* s,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){b.handoff_stats(s);});}
int nr_texture_configure_gpu_handoff(void* h,const NR_TextureGpuHandoffConfig* c,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){b.configure_handoff(c);});}
int nr_texture_gpu_handoff_info(void* h,NR_TextureGpuHandoffInfo* i,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){b.handoff_info(i);});}
int nr_texture_transfer_owner(void* h,uint64_t previous,uint64_t model_queue,uint64_t game_queue,uint32_t* adopted,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){b.transfer_owner(previous,model_queue,game_queue,adopted);});}
