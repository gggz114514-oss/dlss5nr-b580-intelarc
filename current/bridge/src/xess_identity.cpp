#include "xess_identity.h"
#include <mutex>
#include <vector>
#include <atomic>

namespace nrb {
namespace {
struct Pending {
    ID3D12CommandList* list;
    ID3D12Resource* texture;
    uint64_t fence_value;
};
std::mutex mutex;
std::vector<Pending> pending;
ID3D12CommandQueue* retirement_queue=nullptr;
ID3D12Fence* retirement_fence=nullptr;
uint64_t next_fence_value=0;
std::atomic<unsigned> recorded{0},completed{0};

void reap() {
    if(!retirement_fence) return;
    const auto done=retirement_fence->GetCompletedValue();
    for(size_t i=0;i<pending.size();) {
        if(pending[i].fence_value && pending[i].fence_value<=done) {
            pending[i].texture->Release();
            pending.erase(pending.begin()+i);
            completed.fetch_add(1,std::memory_order_relaxed);
        } else ++i;
    }
}
D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource,
                                  D3D12_RESOURCE_STATES before,
                                  D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource=resource;
    b.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore=before;
    b.Transition.StateAfter=after;
    return b;
}
}

ID3D12Resource* record_ngx_identity(ID3D12GraphicsCommandList* list,
                                    ID3D12Resource* source) {
    if(!list || !source) return nullptr;
    const auto desc=source->GetDesc();
    if(desc.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
       desc.Format!=DXGI_FORMAT_R11G11B10_FLOAT ||
       desc.DepthOrArraySize!=1 || desc.MipLevels!=1 ||
       desc.SampleDesc.Count!=1 || desc.Width!=1280 ||
       desc.Height!=720) return nullptr;
    std::lock_guard guard(mutex);
    reap();
    // An unsubmitted recording stays pinned. Saturation bypasses, never
    // recycles a texture the GPU might still read.
    if(pending.size()>=8) return nullptr;
    ID3D12Device *device=nullptr,*source_device=nullptr;
    if(FAILED(list->GetDevice(IID_PPV_ARGS(&device))) ||
       FAILED(source->GetDevice(IID_PPV_ARGS(&source_device)))) {
        if(device) device->Release();
        if(source_device) source_device->Release();
        return nullptr;
    }
    const bool same=device==source_device;
    source_device->Release();
    if(!same) {device->Release();return nullptr;}
    auto copy_desc=desc;
    copy_desc.Flags=D3D12_RESOURCE_FLAG_NONE;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type=D3D12_HEAP_TYPE_DEFAULT;
    ID3D12Resource* copy=nullptr;
    const HRESULT hr=device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,
        &copy_desc,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&copy));
    device->Release();
    if(FAILED(hr) || !copy) return nullptr;
    const auto before=transition(source,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                                 D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1,&before);
    list->CopyResource(copy,source);
    D3D12_RESOURCE_BARRIER after[2]={
        transition(source,D3D12_RESOURCE_STATE_COPY_SOURCE,
                   D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
        transition(copy,D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)};
    list->ResourceBarrier(2,after);
    pending.push_back({list,copy,0});
    recorded.fetch_add(1,std::memory_order_relaxed);
    return copy;
}

unsigned xess_identity_submitted(ID3D12CommandQueue* queue,UINT count,
                                 ID3D12CommandList* const* lists) {
    if(!queue || !count || !lists) return 0;
    std::lock_guard guard(mutex);
    reap();
    if(retirement_queue && retirement_queue!=queue) return 0;
    bool matched=false;
    for(auto& entry:pending) if(!entry.fence_value)
        for(UINT i=0;i<count;i++) if(entry.list==lists[i]) {
            matched=true;break;
        }
    if(!matched) return 0;
    if(!retirement_fence) {
        ID3D12Device* device=nullptr;
        if(FAILED(queue->GetDevice(IID_PPV_ARGS(&device)))) return 0;
        const HRESULT hr=device->CreateFence(0,D3D12_FENCE_FLAG_NONE,
                                              IID_PPV_ARGS(&retirement_fence));
        device->Release();
        if(FAILED(hr)) return 0;
        retirement_queue=queue;
        retirement_queue->AddRef();
    }
    const uint64_t value=next_fence_value+1;
    if(FAILED(queue->Signal(retirement_fence,value))) return 0;
    next_fence_value=value;
    unsigned submitted=0;
    for(auto& entry:pending) if(!entry.fence_value)
        for(UINT i=0;i<count;i++) if(entry.list==lists[i]) {
            entry.fence_value=value;++submitted;break;
        }
    return submitted;
}
unsigned xess_identity_recorded() {return recorded.load(std::memory_order_relaxed);}
unsigned xess_identity_completed() {return completed.load(std::memory_order_relaxed);}
}
