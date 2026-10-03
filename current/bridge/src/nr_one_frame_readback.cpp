#include "nr_one_frame_readback.h"
#include <windows.h>
#include <mutex>
#include <atomic>
#include <cstdio>

namespace nrb {
namespace {
std::mutex capture_mutex;
// 0 idle, 1 recorded, 2 submitted, 3 complete, -1 failed. A failed copy
// remains pinned until process exit rather than recycling in-flight memory.
int capture_state=0;
ID3D12CommandList* captured_list=nullptr;
uint64_t captured_generation=0;
ID3D12Resource* captured_color=nullptr;
ID3D12Resource* readback=nullptr;
ID3D12Fence* fence=nullptr;
HANDLE fence_event=nullptr;
UINT row_pitch=0;
UINT64 byte_count=0;
ReadbackLog logger=nullptr;
std::atomic<bool> armed_notice{false};

void fail(ReadbackLog log,const char* reason,HRESULT hr=S_OK) {
    capture_state=-1;
    char line[256];
    sprintf_s(line,"one_frame_color_readback failed: %s hr=0x%08X\r\n",
              reason,static_cast<unsigned>(hr));
    if(log) log(line);
}

DWORD WINAPI finish_capture(void*) {
    const HRESULT wait_hr=fence->SetEventOnCompletion(1,fence_event);
    if(FAILED(wait_hr) || WaitForSingleObject(fence_event,10000)!=WAIT_OBJECT_0) {
        std::lock_guard guard(capture_mutex);
        fail(logger,"fence completion wait",wait_hr);
        return 0;
    }
    void* mapped=nullptr;
    const D3D12_RANGE range{0,static_cast<SIZE_T>(byte_count)};
    const HRESULT map_hr=readback->Map(0,&range,&mapped);
    if(FAILED(map_hr) || !mapped) {
        std::lock_guard guard(capture_mutex);
        fail(logger,"readback map",map_hr);
        return 0;
    }
    wchar_t path[MAX_PATH]{};
    swprintf_s(path,L"D:\\Codex-NR-Experiments\\cyberpunk-opt\\dlss720-color-%lu.bin",
               GetCurrentProcessId());
    HANDLE output=CreateFileW(path,GENERIC_WRITE,0,nullptr,CREATE_NEW,
                              FILE_ATTRIBUTE_NORMAL,nullptr);
    bool saved=false;
    if(output!=INVALID_HANDLE_VALUE) {
        DWORD written=0;
        saved=byte_count<=MAXDWORD && WriteFile(output,mapped,
             static_cast<DWORD>(byte_count),&written,nullptr) &&
             written==byte_count && FlushFileBuffers(output);
        CloseHandle(output);
    }
    const D3D12_RANGE no_write{0,0};
    readback->Unmap(0,&no_write);
    {
        std::lock_guard guard(capture_mutex);
        if(saved) {
            capture_state=3;
            char line[384];
            sprintf_s(line,"one_frame_color_readback complete: pid=%lu width=1280 height=720 row_pitch=%u bytes=%llu path=D:\\Codex-NR-Experiments\\cyberpunk-opt\\dlss720-color-%lu.bin\r\n",
                      GetCurrentProcessId(),row_pitch,byte_count,GetCurrentProcessId());
            if(logger) logger(line);
        } else fail(logger,"file create/write");
    }
    return 0;
}
}

void one_frame_readback_record(ID3D12GraphicsCommandList* list,
                               ID3D12Resource* color,
                               const SyncObservation& sync,
                               ReadbackLog log) {
    // A wrapped diagnostic ring does not invalidate a recent observed color
    // transition; a single oversized unrecorded barrier call still does.
    if(!list || !color || !sync.reset_observed || sync.closed ||
       sync.oversized_barrier ||
       sync.submit_batch_before || !sync.color.seen ||
       sync.color.sequence>=sync.evaluate_sequence ||
       sync.evaluate_sequence-sync.color.sequence>16 ||
       sync.color.flags!=D3D12_RESOURCE_BARRIER_FLAG_NONE ||
       sync.color.subresource!=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES ||
       sync.color.after!=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE) return;
    const auto desc=color->GetDesc();
    if(desc.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
       desc.Width!=1280 || desc.Height!=720 ||
       desc.DepthOrArraySize!=1 || desc.MipLevels!=1 ||
       desc.SampleDesc.Count!=1 || desc.Format!=DXGI_FORMAT_R11G11B10_FLOAT) return;
    if(!armed_notice.exchange(true) && log)
        log("one_frame_color_readback armed: press F11 in a gameplay scene\r\n");
    if(!(GetAsyncKeyState(VK_F11)&0x8000)) return;
    std::lock_guard guard(capture_mutex);
    if(capture_state!=0) return;
    capture_state=-1; // Do not retry partially initialized GPU resources.
    ID3D12Device* device=nullptr;
    HRESULT hr=list->GetDevice(IID_PPV_ARGS(&device));
    if(FAILED(hr)) {fail(log,"device lookup",hr);return;}
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    device->GetCopyableFootprints(&desc,0,1,0,&footprint,nullptr,nullptr,&byte_count);
    row_pitch=footprint.Footprint.RowPitch;
    if(!byte_count || byte_count>16*1024*1024) {
        device->Release();fail(log,"unexpected footprint");return;
    }
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type=D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width=byte_count;
    buffer.Height=1;
    buffer.DepthOrArraySize=1;
    buffer.MipLevels=1;
    buffer.Format=DXGI_FORMAT_UNKNOWN;
    buffer.SampleDesc.Count=1;
    buffer.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    hr=device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&buffer,
        D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&readback));
    if(SUCCEEDED(hr)) hr=device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence));
    device->Release();
    if(FAILED(hr)) {fail(log,"readback resource/fence creation",hr);return;}
    fence_event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
    if(!fence_event) {fail(log,"fence event creation",HRESULT_FROM_WIN32(GetLastError()));return;}
    D3D12_RESOURCE_BARRIER to_copy{};
    to_copy.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    to_copy.Transition.pResource=color;
    to_copy.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    to_copy.Transition.StateBefore=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    to_copy.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1,&to_copy);
    D3D12_TEXTURE_COPY_LOCATION dest{};
    dest.pResource=readback;
    dest.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dest.PlacedFootprint=footprint;
    D3D12_TEXTURE_COPY_LOCATION source{};
    source.pResource=color;
    source.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    source.SubresourceIndex=0;
    list->CopyTextureRegion(&dest,0,0,0,&source,nullptr);
    auto restore=to_copy;
    restore.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_SOURCE;
    restore.Transition.StateAfter=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    list->ResourceBarrier(1,&restore);
    color->AddRef();captured_color=color;
    captured_list=list;captured_generation=sync.generation;
    logger=log;
    capture_state=1;
    if(log) log("one_frame_color_readback recorded on pre-SR command list; awaiting matching submit\r\n");
}

void one_frame_readback_submitted(ID3D12CommandQueue* queue,UINT count,
                                  ID3D12CommandList* const* lists,
                                  const SubmitObservation* observations,
                                  UINT observation_count,ReadbackLog log) {
    if(!queue || !lists || !observations) return;
    std::lock_guard guard(capture_mutex);
    if(capture_state!=1) return;
    for(UINT i=0;i<count && i<observation_count;i++) {
        if(lists[i]!=captured_list || !observations[i].correlated ||
           observations[i].generation!=captured_generation) continue;
        const HRESULT hr=queue->Signal(fence,1);
        if(FAILED(hr)) {fail(log,"queue signal",hr);return;}
        capture_state=2;
        HANDLE worker=CreateThread(nullptr,0,finish_capture,nullptr,0,nullptr);
        if(worker) CloseHandle(worker);
        else fail(log,"capture worker creation",HRESULT_FROM_WIN32(GetLastError()));
        return;
    }
}
}
