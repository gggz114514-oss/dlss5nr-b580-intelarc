// CPU-built only by the sidecar. Luna runs this on an exclusive B580 window.
// This is an isolated HDR adapter byte probe with a synthetic NR consumer;
// it does not establish real XPU/NR retirement or live-game acceptance.
#include "nr_hdr_proxy.h"
#include <dxgi1_6.h>
#include <d3d12sdklayers.h>
#include <bcrypt.h>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

using Microsoft::WRL::ComPtr;
namespace {
constexpr UINT width=1280,height=720;
void check(bool okay,const char* text) {if(!okay) throw std::runtime_error(text);}
void hr(HRESULT value,const char* text) {check(SUCCEEDED(value),text);}
D3D12_RESOURCE_BARRIER transition(ID3D12Resource* r,D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition={r,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after};return b;
}
ComPtr<ID3D12Resource> texture(ID3D12Device* device,DXGI_FORMAT format) {
    D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;d.Width=width;d.Height=height;
    d.DepthOrArraySize=1;d.MipLevels=1;d.SampleDesc.Count=1;d.Format=format;d.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_HEAP_PROPERTIES h{};h.Type=D3D12_HEAP_TYPE_DEFAULT;
    ComPtr<ID3D12Resource> result;
    hr(device->CreateCommittedResource(&h,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,IID_PPV_ARGS(result.GetAddressOf())),"Create texture");return result;
}
uint64_t wait(ID3D12Device* device,ID3D12Fence* fence) {
    HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);check(event!=nullptr,"Create event");
    const auto registered=fence->SetEventOnCompletion(1,event);
    const auto result=SUCCEEDED(registered)?WaitForSingleObject(event,30000):WAIT_FAILED;
    CloseHandle(event);check(result==WAIT_OBJECT_0,"Fence registration/wait failed");
    const auto done=fence->GetCompletedValue();
    check(done>=1 && done!=UINT64_MAX && SUCCEEDED(device->GetDeviceRemovedReason()),"Exact fence/device completion failed");
    return done;
}
struct Capture {
    ComPtr<ID3D12Resource> buffer;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 row_bytes=0,size=0;
    void initialize(ID3D12Device* device,ID3D12Resource* resource) {
        auto desc=resource->GetDesc();UINT rows=0;
        device->GetCopyableFootprints(&desc,0,1,0,&footprint,&rows,&row_bytes,&size);
        check(rows==height && row_bytes && size,"Unexpected readback geometry");
        D3D12_RESOURCE_DESC b{};b.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;b.Width=size;b.Height=1;
        b.DepthOrArraySize=1;b.MipLevels=1;b.SampleDesc.Count=1;b.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_HEAP_PROPERTIES h{};h.Type=D3D12_HEAP_TYPE_READBACK;
        hr(device->CreateCommittedResource(&h,D3D12_HEAP_FLAG_NONE,&b,D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,IID_PPV_ARGS(buffer.GetAddressOf())),"Create readback");
    }
    void record(ID3D12GraphicsCommandList* list,ID3D12Resource* resource,D3D12_RESOURCE_STATES state) {
        auto barrier=transition(resource,state,D3D12_RESOURCE_STATE_COPY_SOURCE);list->ResourceBarrier(1,&barrier);
        D3D12_TEXTURE_COPY_LOCATION src{},dst{};src.pResource=resource;src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.pResource=buffer.Get();dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;dst.PlacedFootprint=footprint;
        list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
        barrier=transition(resource,D3D12_RESOURCE_STATE_COPY_SOURCE,state);list->ResourceBarrier(1,&barrier);
    }
    std::vector<uint8_t> bytes() const {
        void* mapped=nullptr;D3D12_RANGE range{0,SIZE_T(size)};hr(buffer->Map(0,&range,&mapped),"Map capture");
        std::vector<uint8_t> packed(size_t(row_bytes)*height);
        for(UINT y=0;y<height;++y) std::memcpy(packed.data()+size_t(row_bytes)*y,
            static_cast<const uint8_t*>(mapped)+footprint.Offset+size_t(footprint.Footprint.RowPitch)*y,size_t(row_bytes));
        D3D12_RANGE none{0,0};buffer->Unmap(0,&none);return packed;
    }
};
struct Frame {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Resource> source,motion,xess,nr;
    ComPtr<ID3D12DescriptorHeap> clear_heap;
    std::array<ComPtr<ID3D12CommandAllocator>,3> allocators;
    std::array<ComPtr<ID3D12GraphicsCommandList>,3> lists;
    ComPtr<ID3D12Fence> prepared,ready,completion;
    std::array<Capture,3> captures;
    nrb::HdrProxy hdr;
    bool exposed=false,completed=false;
    void drop_records() {for(auto& list:lists) list.Reset();}
};
using Pixels=std::array<std::vector<uint8_t>,3>;
struct GpuResult {
    Pixels pixels;
    ComPtr<IUnknown> proxy,motion,source,xess,nr;
};
GpuResult run(ID3D12Device* device,ID3D12CommandQueue* queue,const std::wstring& path,
           nrb::HdrProxyCache& cache,nrb::HdrProxyResourcePool& pool,bool pool_on,unsigned frame_index) {
    pool.set_enabled(pool_on);
    auto owned=std::make_unique<Frame>();auto& f=*owned;f.device=device;f.queue=queue;
    try {
        f.source=texture(device,DXGI_FORMAT_R11G11B10_FLOAT);f.motion=texture(device,DXGI_FORMAT_R16G16_FLOAT);
        f.xess=texture(device,DXGI_FORMAT_R11G11B10_FLOAT);f.nr=texture(device,DXGI_FORMAT_R32G32B32A32_FLOAT);
        const bool null_motion=frame_index%4==3,use_motion=frame_index%2==0;
        check(f.hdr.initialize(device,f.source.Get(),null_motion?nullptr:f.motion.Get(),f.xess.Get(),path,&cache,&pool),"HDR initialize");
        auto* proof=f.hdr.resource_proof();check(bool(proof)==pool_on,"Unexpected pool lease/miss in isolated sequential probe");
        for(size_t i=0;i<3;++i) {
            hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(f.allocators[i].GetAddressOf())),"Create allocator");
            hr(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,f.allocators[i].Get(),nullptr,
                IID_PPV_ARGS(f.lists[i].GetAddressOf())),"Create list");
        }
        hr(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(f.prepared.GetAddressOf())),"Prepare fence");
        hr(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(f.ready.GetAddressOf())),"Synthetic ready fence");
        hr(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(f.completion.GetAddressOf())),"Consumer fence");
        D3D12_DESCRIPTOR_HEAP_DESC h{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,3,D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,0};
        hr(device->CreateDescriptorHeap(&h,IID_PPV_ARGS(f.clear_heap.GetAddressOf())),"Clear heap");
        const auto stride=device->GetDescriptorHandleIncrementSize(h.Type);
        auto cpu=f.clear_heap->GetCPUDescriptorHandleForHeapStart();auto gpu=f.clear_heap->GetGPUDescriptorHandleForHeapStart();
        ID3D12Resource* clear_resources[]={f.source.Get(),f.motion.Get(),f.nr.Get()};
        const float values[3][4]={{0.25f+frame_index*0.25f,0.5f+frame_index,0.75f+frame_index*2.0f,1},
            {0.01f+frame_index*0.001f,-0.02f-frame_index*0.002f,0,0},
            {0.32f+frame_index*0.01f,0.48f-frame_index*0.01f,0.66f,1}};
        ID3D12DescriptorHeap* heaps[]={f.clear_heap.Get()};f.lists[0]->SetDescriptorHeaps(1,heaps);
        for(size_t i=0;i<3;++i) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC u{};u.Format=clear_resources[i]->GetDesc().Format;u.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;
            device->CreateUnorderedAccessView(clear_resources[i],nullptr,&u,cpu);
            f.lists[0]->ClearUnorderedAccessViewFloat(gpu,cpu,clear_resources[i],values[i],0,nullptr);
            cpu.ptr+=stride;gpu.ptr+=stride;
        }
        D3D12_RESOURCE_BARRIER ready[3]={transition(f.motion.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(f.nr.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(f.xess.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)};
        f.lists[0]->ResourceBarrier(3,ready);
        f.hdr.record_prepare(f.lists[0].Get(),f.source.Get(),use_motion,1280,720);
        ID3D12Resource* outputs[]={f.hdr.proxy_color(),f.hdr.nr_motion(),f.xess.Get()};
        const D3D12_RESOURCE_STATES states[]={D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE};
        for(size_t i=0;i<3;++i) {f.captures[i].initialize(device,outputs[i]);f.captures[i].record(f.lists[2].Get(),outputs[i],states[i]);}
        hr(f.lists[0]->Close(),"Close prepare");hr(f.lists[2]->Close(),"Close private capture/deferred");
        if(proof) {proof->private_records_closed();proof->prepare_submitted();}
        ID3D12CommandList* list=f.lists[0].Get();f.exposed=true;queue->ExecuteCommandLists(1,&list);
        hr(queue->Signal(f.prepared.Get(),1),"Prepare signal");wait(device,f.prepared.Get());
        if(proof) {proof->prepare_completed(true);proof->processor_begin();}
        // The synthetic consumer is already prepared by the clear above. This
        // ready fence and terminal proof must not be called real XPU evidence.
        hr(queue->Signal(f.ready.Get(),1),"Synthetic ready signal");hr(queue->Wait(f.ready.Get(),1),"Synthetic ready wait");
        if(proof) proof->processor_ready(true);
        check(f.hdr.record_composite(f.lists[1].Get(),f.source.Get(),f.nr.Get(),f.xess.Get()),"Record composite");
        hr(f.lists[1]->Close(),"Close composite");
        if(proof) {proof->composite_closed();proof->composite_submitted();}
        list=f.lists[1].Get();queue->ExecuteCommandLists(1,&list);
        if(proof) proof->deferred_submitted();list=f.lists[2].Get();queue->ExecuteCommandLists(1,&list);
        const auto signaled=queue->Signal(f.completion.Get(),1);if(proof) proof->completion_signal(SUCCEEDED(signaled));hr(signaled,"Consumer signal");
        const auto done=wait(device,f.completion.Get());f.completed=true;
        if(proof) {proof->consumer_completed(done,true);proof->processor_retirement(true);check(proof->terminal_safe(),"Synthetic terminal proof");}
        GpuResult result;for(size_t i=0;i<3;++i) result.pixels[i]=f.captures[i].bytes();
        hr(f.hdr.proxy_color()->QueryInterface(IID_PPV_ARGS(result.proxy.GetAddressOf())),"Proxy identity");
        hr(f.hdr.nr_motion()->QueryInterface(IID_PPV_ARGS(result.motion.GetAddressOf())),"NR motion identity");
        hr(f.source.As(&result.source),"Source identity");hr(f.xess.As(&result.xess),"XeSS identity");hr(f.nr.As(&result.nr),"NR identity");
        f.drop_records();check(f.hdr.finish_resource_lease(),"Lease finish after private record destruction");
        return result;
    } catch(...) {
        if(f.exposed && !f.completed) {
            f.hdr.quarantine_resource_lease();(void)owned.release(); // keep ALL possibly pending GPU refs
        } else f.drop_records();
        throw;
    }
}
std::string sha256(const std::vector<uint8_t>& bytes) {
    BCRYPT_ALG_HANDLE algorithm=nullptr;BCRYPT_HASH_HANDLE hash=nullptr;
    check(BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0)>=0,"SHA provider");
    DWORD size=0,written=0;check(BCryptGetProperty(algorithm,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&size),sizeof(size),&written,0)>=0,"SHA size");
    std::vector<uint8_t> object(size);std::array<uint8_t,32> digest{};
    check(BCryptCreateHash(algorithm,&hash,object.data(),size,nullptr,0,0)>=0,"SHA create");
    check(BCryptHashData(hash,const_cast<PUCHAR>(bytes.data()),ULONG(bytes.size()),0)>=0 &&
        BCryptFinishHash(hash,digest.data(),ULONG(digest.size()),0)>=0,"SHA digest");
    BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(algorithm,0);
    std::string text;const char hex[]="0123456789abcdef";
    for(auto value:digest) {text+=hex[value>>4];text+=hex[value&15];}return text;
}
void debug_clean(ID3D12InfoQueue* info) {
    for(UINT64 i=0;i<info->GetNumStoredMessagesAllowedByRetrievalFilter();++i) {
        SIZE_T size=0;hr(info->GetMessage(i,nullptr,&size),"Debug message size");std::vector<uint8_t> storage(size);
        auto* m=reinterpret_cast<D3D12_MESSAGE*>(storage.data());hr(info->GetMessage(i,m,&size),"Debug message");
        if(m->Severity==D3D12_MESSAGE_SEVERITY_ERROR || m->Severity==D3D12_MESSAGE_SEVERITY_CORRUPTION) {
            std::fprintf(stderr,"D3D12: %s\n",m->pDescription);throw std::runtime_error("D3D12 debug error");
        }
    }
}
}
int wmain(int argc,wchar_t** argv) {
    if(argc!=3) {std::fwprintf(stderr,L"usage: nr_hdr_resource_pool_gpu_probe.exe <shader.hlsl> <new-output-dir-on-D>\n");return 2;}
    try {
        const std::filesystem::path out=argv[2];
        check(!std::filesystem::exists(out),"Output directory must be fresh");
        check(out.is_absolute() && (out.root_name()==L"D:" || out.root_name()==L"d:"),"GPU receipt data belongs on D:");
        ComPtr<ID3D12Debug> debug;
        if(FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(debug.GetAddressOf())))) {std::puts("SKIP: D3D12 debug layer unavailable; no verified GPU run");return 77;}
        debug->EnableDebugLayer();ComPtr<IDXGIFactory6> factory;hr(CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf())),"DXGI factory");
        ComPtr<IDXGIAdapter1> adapter;
        for(UINT i=0;;++i) {ComPtr<IDXGIAdapter1> candidate;auto result=factory->EnumAdapters1(i,candidate.GetAddressOf());
            if(result==DXGI_ERROR_NOT_FOUND) break;hr(result,"Enumerate adapter");DXGI_ADAPTER_DESC1 d{};hr(candidate->GetDesc1(&d),"Adapter desc");
            if(!(d.Flags&DXGI_ADAPTER_FLAG_SOFTWARE) && std::wstring(d.Description).find(L"B580")!=std::wstring::npos) {adapter=candidate;break;}}
        if(!adapter) {std::puts("SKIP: Intel Arc B580 unavailable");return 77;}
        ComPtr<ID3D12Device> device;hr(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(device.GetAddressOf())),"B580 device");
        ComPtr<ID3D12InfoQueue> info;hr(device.As(&info),"Debug info queue");
        D3D12_COMMAND_QUEUE_DESC q{};q.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;ComPtr<ID3D12CommandQueue> queue;
        hr(device->CreateCommandQueue(&q,IID_PPV_ARGS(queue.GetAddressOf())),"Queue");
        nrb::HdrProxyCache cache;nrb::HdrProxyResourcePool pool;std::string records;
        const char* names[]={"proxy_rgba32f","motion_rg16f","composite_r11g11b10"};unsigned comparisons=0,same_resource_reuses=0,changed_bindings=0;
        std::array<ComPtr<IUnknown>,2> resident_proxy,resident_motion;
        for(bool cache_on:{false,true}) {
            cache.set_enabled(cache_on);
            for(unsigned frame=0;frame<8;++frame) {
                if(cache_on && frame==4) cache.invalidate_shader();
                const auto before=cache.stats();
                const auto reference=run(device.Get(),queue.Get(),argv[1],cache,pool,false,frame);
                const auto reference_stats=cache.stats();
                const auto pooled=run(device.Get(),queue.Get(),argv[1],cache,pool,true,frame);
                const auto after=cache.stats();
                check(cache_on?after.pipeline_hits==reference_stats.pipeline_hits+1:
                    after.uncached_initializations==reference_stats.uncached_initializations+1,
                    "Pool hit bypassed per-frame pipeline acquisition/toggle");
                check(after.shader_generation>=before.shader_generation,"Shader generation regressed");
                const auto off_again=run(device.Get(),queue.Get(),argv[1],cache,pool,false,frame);
                const auto key_index=frame%4==3?1:0;
                if(resident_proxy[key_index]) {
                    check(resident_proxy[key_index].Get()==pooled.proxy.Get() &&
                        resident_motion[key_index].Get()==pooled.motion.Get(),"Counters claimed reuse without the same canonical textures");
                    ++same_resource_reuses;
                } else {resident_proxy[key_index]=pooled.proxy;resident_motion[key_index]=pooled.motion;}
                check(reference.source.Get()!=pooled.source.Get() && reference.xess.Get()!=pooled.xess.Get() &&
                    reference.nr.Get()!=pooled.nr.Get() && off_again.source.Get()!=pooled.source.Get(),"Frame bindings did not change");
                check(off_again.proxy.Get()!=pooled.proxy.Get() && off_again.motion.Get()!=pooled.motion.Get(),"Toggle OFF reused pooled textures");
                ++changed_bindings;
                for(size_t channel=0;channel<3;++channel) {
                    check(reference.pixels[channel]==pooled.pixels[channel] && off_again.pixels[channel]==pooled.pixels[channel],
                        "Pool OFF/ON/OFF bytes differ");comparisons+=2;
                    records+=(records.empty()?"":",\n")+std::string("    {\"cache_on\":")+(cache_on?"true":"false")+
                        ",\"frame\":"+std::to_string(frame)+",\"channel\":\""+names[channel]+"\",\"bytes\":"+
                        std::to_string(pooled.pixels[channel].size())+",\"sha256\":\""+sha256(pooled.pixels[channel])+"\"}";
                }
                debug_clean(info.Get());
            }
        }
        const auto stats=pool.stats();check(stats.creates==2 && stats.hits==14 && stats.retire_returns==16 &&
            stats.resident_slots==2 && !stats.leased_slots && !stats.quarantined_slots &&
            stats.texture_bytes<=pool.max_texture_bytes && same_resource_reuses==14 && changed_bindings==16 && !stats.enabled,
            "Bounded reuse/retirement/OFF-restoration counters unexpected");
        check(pool.shutdown(),"Safe pool shutdown");debug_clean(info.Get());
        std::filesystem::create_directories(out);
        std::ofstream receipt(out/L"gpu-byte-receipt.json",std::ios::binary);
        receipt<<"{\n  \"status\":\"passed\",\n  \"scope\":\"isolated HDR adapter, synthetic consumer; real NR/XPU and game unverified\",\n"
            <<"  \"source_width\":1280,\"source_height\":720,\"byte_comparisons\":"<<comparisons
            <<",\n  \"pool_creates\":"<<stats.creates<<",\"pool_hits\":"<<stats.hits<<",\"pool_texture_bytes\":"<<stats.texture_bytes
            <<",\n  \"same_canonical_resource_reuses\":"<<same_resource_reuses<<",\"changed_binding_cases\":"<<changed_bindings
            <<",\"off_on_off_cases\":16,\"final_pool_enabled\":false"
            <<",\n  \"debug_errors\":0,\n  \"samples\":[\n"<<records<<"\n  ]\n}\n";
        check(bool(receipt),"Write GPU receipt");
        std::printf("PASS: %u exact byte comparisons; two slots, %llu hits; debug errors=0; synthetic consumer only\n",comparisons,stats.hits);
        return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
