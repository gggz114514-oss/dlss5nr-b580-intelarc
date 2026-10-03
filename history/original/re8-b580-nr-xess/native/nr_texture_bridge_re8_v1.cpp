// Isolated RE8 input-format candidate. The frozen product bridge remains
// unchanged; D3D12's BGRA8 SRV presents logical RGB channels to the pack CS.
#define NOMINMAX
#include "nr_texture_bridge_v1.h"
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
    ~Shared() {
        try {if(ptr)sx::unmap_external_linear_memory(ptr,*queue);}catch(...){}
        try {if(valid)sx::release_external_memory(imported,*queue);}catch(...){}
        if(handle)CloseHandle(handle);
    }
};
struct TextureBridge {
    sycl::queue* xpu;
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
    TextureBridge(sycl::queue* q,uint32_t w,uint32_t h):xpu(q),width(w),height(h){}
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
    void healthy(){require(!failed,"Bridge execution failed; close isolated worker");}
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
    void dispatch(ID3D12PipelineState* p,uint32_t seed=0,uint32_t color8=0,float scaleX=1.0f,float scaleY=1.0f){ID3D12DescriptorHeap* heaps[]={heap.Get()};list->SetDescriptorHeaps(1,heaps);list->SetComputeRootSignature(root.Get());list->SetComputeRootDescriptorTable(0,gpu(0));list->SetComputeRootDescriptorTable(1,gpu(3));uint32_t c[]={width,height,seed,color8,0,0};memcpy(&c[4],&scaleX,4);memcpy(&c[5],&scaleY,4);list->SetComputeRoot32BitConstants(2,6,c,0);list->SetPipelineState(p);list->Dispatch((width+7)/8,(height+7)/8,1);}
    void validate_texture(ID3D12Resource* resource,bool motion){
        require(resource!=nullptr,"Missing input texture");auto d=resource->GetDesc();
        require(d.Dimension==D3D12_RESOURCE_DIMENSION_TEXTURE2D && d.Width==width && d.Height==height && d.DepthOrArraySize==1 && d.MipLevels==1 && d.SampleDesc.Count==1,"Texture geometry mismatch");
        require(motion?(d.Format==DXGI_FORMAT_R16G16_FLOAT || d.Format==DXGI_FORMAT_R32G32_FLOAT || d.Format==DXGI_FORMAT_R16G16B16A16_SNORM):(d.Format==DXGI_FORMAT_R8G8B8A8_UNORM || d.Format==DXGI_FORMAT_B8G8R8A8_UNORM || d.Format==DXGI_FORMAT_R32G32B32A32_FLOAT),"Unsupported texture format");
        ComPtr<ID3D12Device> owner;hr(resource->GetDevice(IID_PPV_ARGS(&owner)),"resource device");require(owner.Get()==device.Get(),"Texture belongs to another D3D12 device");
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
        healthy();require(!pending,"Previous NR frame not exported");require(!output_borrowed || retirement_value,"Downstream consumer completion must be registered before reuse");
        require(reset || (history && previous==last_id && id>last_id),"First/discontinuous frame requires reset or matching previous frame ID");
        validate_texture(color,false);validate_texture(motion,true);pointer(color_ptr);pointer(motion_ptr);validate_fence(producer,value);
        xpu->wait_and_throw();
        wait_gpu(retirement.Get(),retirement_value);wait_gpu(producer,value);
        color_hold=color;motion_hold=motion;read_inputs(color_ptr,motion_ptr);
        retirement.Reset();retirement_value=0;output_borrowed=false;pending_id=id;pending=true;
    }
    void export_frame(uint64_t id,const void* color){
        healthy();require(pending && id==pending_id,"NR output frame ID mismatch or no pending frame");pointer(color);
        xpu->wait_and_throw();xpu->memcpy(result.ptr,color,size_t(result.bytes)).wait_and_throw();
        begin();barrier(list.Get(),result.resource.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);barrier(list.Get(),output.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatch(unpack.Get());barrier(list.Get(),output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);barrier(list.Get(),result.resource.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COMMON);submit();
        last_id=id;history=true;pending=false;output_ready=true;
    }
    void make_fixture(uint32_t seed,uint32_t bits,void** color,void** motion){
        healthy();require(!pending,"Cannot replace a pending source");require(bits==16 || bits==32,"Fixture motion format must be 16/32");require(color && motion,"Null fixture result");
        if(!fixture_color)fixture_color=texture(DXGI_FORMAT_R8G8B8A8_UNORM);
        auto format=bits==16?DXGI_FORMAT_R16G16_FLOAT:DXGI_FORMAT_R32G32_FLOAT;
        if(fixture_mv_format!=format){fixture_motion=texture(format);fixture_mv_format=format;}
        uav(fixture_color.Get(),DXGI_FORMAT_R8G8B8A8_UNORM,5);uav(fixture_motion.Get(),format,6);
        begin();barrier(list.Get(),fixture_color.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);barrier(list.Get(),fixture_motion.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatch(fixture.Get(),seed);barrier(list.Get(),fixture_color.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);barrier(list.Get(),fixture_motion.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);submit();*color=fixture_color.Get();*motion=fixture_motion.Get();
    }
    void make_re8_fixture(void** color,void** motion){
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
        healthy();require(output_ready && !pending,"No completed output to audit");pointer(target);
        srv(output.Get(),DXGI_FORMAT_R32G32B32A32_FLOAT,0);begin();barrier(list.Get(),rgb.resource.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);dispatch(gather.Get());barrier(list.Get(),rgb.resource.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COMMON);submit();xpu->memcpy(target,rgb.ptr,size_t(rgb.bytes)).wait_and_throw();
    }
    void close(){
        require(!output_borrowed || retirement_value,"Cannot close while an output consumer is unregistered");
        wait_gpu(retirement.Get(),retirement_value);begin();submit();xpu->wait_and_throw();
    }
};
template<class F> static int invoke(void* h,char* error,uint32_t size,F&& function){
    if(!h){message(error,size,"Null bridge");return 1;}auto* b=static_cast<TextureBridge*>(h);
    try{function(*b);return 0;}catch(const ContractError& e){message(error,size,e.what());return 1;}catch(const std::exception& e){b->failed=true;message(error,size,e.what());return 1;}
}
void* nr_texture_create(void* q,void* d,void* queue,uint32_t w,uint32_t h,char* e,uint32_t n){try{auto b=std::make_unique<TextureBridge>(static_cast<sycl::queue*>(q),w,h);b->init(static_cast<ID3D12Device*>(d),static_cast<ID3D12CommandQueue*>(queue));return b.release();}catch(const std::exception& x){message(e,n,x.what());return nullptr;}}
int nr_texture_prepare(void* h,void* c,void* m,void* f,uint64_t v,uint64_t id,uint64_t prev,uint32_t reset,void* rgb,void* mv,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){b.prepare(static_cast<ID3D12Resource*>(c),static_cast<ID3D12Resource*>(m),static_cast<ID3D12Fence*>(f),v,id,prev,reset!=0,rgb,mv);});}
int nr_texture_export(void* h,uint64_t id,const void* rgb,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){b.export_frame(id,rgb);});}
int nr_texture_output(void* h,void** r,void** f,uint64_t* v,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){b.healthy();require(b.output_ready && !b.pending && !b.output_borrowed && r && f && v,"Output unavailable, already borrowed, or null return pointer");*r=b.output.Get();*f=b.ready.Get();*v=b.ready_value;b.output_borrowed=true;});}
int nr_texture_retire(void* h,void* f,uint64_t v,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){b.healthy();require(b.output_borrowed && f && v && !b.retirement_value,"Missing/duplicate output retirement");auto* fence=static_cast<ID3D12Fence*>(f);b.validate_fence(fence,v);b.retirement=fence;b.retirement_value=v;});}
int nr_texture_close(void* h,char* e,uint32_t n){int rc=invoke(h,e,n,[](auto& b){b.close();});if(!rc)delete static_cast<TextureBridge*>(h);return rc;}
int nr_texture_fixture(void* h,uint32_t seed,uint32_t bits,void** c,void** m,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){b.make_fixture(seed,bits,c,m);});}
int nr_texture_fixture_re8(void* h,void** c,void** m,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){b.make_re8_fixture(c,m);});}
int nr_texture_audit_output(void* h,void* p,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){b.audit_output(p);});}
int nr_texture_audit_inputs(void* h,void* p,void* m,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){b.healthy();b.read_inputs(p,m);});}
int nr_texture_compile_check(char* e,uint32_t n){try{auto s=shader_source();for(auto name:{"pack","unpack","gather","fixture"})compile_shader(s,name);return 0;}catch(const std::exception& x){message(e,n,x.what());return 1;}}
int nr_texture_test_context(void* h,void** d,void** q,char* e,uint32_t n){return invoke(h,e,n,[&](auto& b){b.healthy();require(d && q,"Null context result");*d=b.device.Get();*q=b.queue.Get();});}
