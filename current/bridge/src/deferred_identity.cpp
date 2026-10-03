#include "deferred_identity.h"
#include <wrl/client.h>
#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include <memory>
#include <chrono>
#include <cmath>
#include "periodic_flash_native_diag.h"
#include "nr_record_timing.h"
#include "nr_hdr_resource_pool_api.h"
#include "nr_hdr_resource_pool.h"
#include "nr_gpu_handoff.h"
#include "nr_gpu_handoff_reuse.h"
#ifdef NRB_LIVE_NR_TEST
#include "nr_hdr_proxy.h"
#endif

namespace nrb {
namespace {
using Microsoft::WRL::ComPtr;
struct Pending {
    ComPtr<ID3D12GraphicsCommandList> game_list;
    uint64_t game_list_generation=0;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandAllocator> prep_allocator;
    ComPtr<ID3D12GraphicsCommandList> prep_list;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> deferred_list;
    ComPtr<ID3D12Resource> color_copy;
    ComPtr<ID3D12Resource> original_color,output,motion,depth,exposure;
    ComPtr<ID3D12Fence> completion;
    NRB_Controls frame_controls{};
    NRB_Route route=NRB_ROUTE_UNKNOWN;
    NRB_RouteEvidence evidence=NRB_EVIDENCE_NONE;
    float motion_scale_x=1,motion_scale_y=1;
    bool game_motion=false,game_reset=false;
    flashdiag::Context diagnostic{};
    flashdiag::Reason diagnostic_reason=flashdiag::Reason::none;
#ifdef NRB_LIVE_NR_TEST
    std::unique_ptr<HdrProxy> hdr;
    ComPtr<ID3D12Fence> prepared;
    ComPtr<ID3D12CommandAllocator> composite_allocator;
    ComPtr<ID3D12GraphicsCommandList> composite_list;
    ComPtr<ID3D12Resource> nr_output;
    ComPtr<ID3D12Fence> nr_ready;
    ComPtr<ID3D12Resource> motion_probe_readback;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT motion_probe_layout{};
    unsigned motion_probe_index=0;
    NRB_Processor nr_processor{};
    bool nr_submitted=false;
    bool nr_processor_called=false, nr_helper_retired=false, nr_queue_exposed=false;
    bool nr_prepared_bypassed=false;
    ComPtr<ID3D12Fence> nr_reuse_completion;
    bool nr_timed=false;
    bool record_timed=false;
    uint64_t record_timing_epoch=0;
    double record_resources_ms=0, record_hdr_ms=0, record_xess_ms=0;
    std::chrono::steady_clock::time_point record_started{},record_finished{};
    uint64_t nr_timing_epoch=0;
    std::chrono::steady_clock::time_point nr_started{};
    std::chrono::steady_clock::time_point nr_prepared{};
    std::chrono::steady_clock::time_point nr_modeled{};
    std::chrono::steady_clock::time_point nr_pre_sr{};
#endif
    bool submitted=false;
#ifdef NRB_LIVE_NR_TEST
    ~Pending() {
        if(hdr && hdr->resource_proof()) {
            // Remove every private record before a pristine/terminal lease can
            // become free. No private list is exported outside this Pending.
            const auto& proof=*hdr->resource_proof();
            if(proof.pristine_unsubmitted() || proof.terminal_safe()) {
                prep_list.Reset();deferred_list.Reset();composite_list.Reset();
                hdr->finish_resource_lease();
            } else hdr->quarantine_resource_lease();
        }
    }
#endif
};
std::recursive_mutex pending_mutex;
std::vector<std::unique_ptr<Pending>> pending;
#ifdef NRB_LIVE_NR_TEST
struct PendingExitQuarantine {
    ~PendingExitQuarantine() {
        // Declared after pending, so this runs before its vector destructor.
        // Retain the complete Pending, not just the pool's textures, when DLL
        // teardown has no proof for GPU/XPU commands or borrowed originals.
        for(auto& item:pending) if(item) {
            if(item->hdr && item->hdr->resource_proof()) {
                const auto& proof=*item->hdr->resource_proof();
                if(!proof.pristine_unsubmitted() && !proof.terminal_safe()) (void)item.release();
            } else if(item->nr_queue_exposed && (!item->submitted ||
                      item->completion->GetCompletedValue()<1 ||
                      item->completion->GetCompletedValue()==UINT64_MAX ||
                      (item->nr_processor_called && !item->nr_helper_retired))) (void)item.release();
        }
    }
} pending_exit_quarantine;
#endif
std::atomic<unsigned> attempts{0}, recorded{0}, retired{0};
std::atomic<unsigned> tail_samples[4]{},tail_valid_samples[4]{};
std::atomic<unsigned> signal_failures{0},generation_mismatches{0},invalidated{0};
#ifdef NRB_LIVE_NR_TEST
HdrProxyCache& hdr_cache_session() {
    static auto cache=[] {
        auto result=std::make_unique<HdrProxyCache>(2);
        result->set_enabled(true);return result;
    }();
    return *cache;
}
HdrProxyResourcePool& hdr_resource_pool_session() {
    // Process lifetime, independent default-OFF session. No unsafe shutdown of
    // quarantined GPU ownership during DLL/static destruction.
    static auto* pool=new HdrProxyResourcePool;return *pool;
}
std::atomic<uint64_t> nr_completed{0};
std::atomic<double> nr_last_ms{0};
std::atomic<double> nr_average_ms{0};
std::atomic<unsigned> nr_sample_count{0};
std::atomic<double> prep_last_ms{0}, prep_average_ms{0};
std::atomic<double> host_last_ms{0}, host_average_ms{0};
std::atomic<double> composite_submit_last_ms{0}, composite_submit_average_ms{0};
std::atomic<double> tail_last_ms{0}, tail_average_ms{0};
// The five channels share one 32-frame window, so their averages add up to
// the average total from the same completed frames.
std::array<std::array<double,32>,5> nr_timing_samples{};
std::array<double,5> nr_timing_sums{};
unsigned nr_timing_count=0;
std::mutex nr_timing_mutex;
RecordTimingWindow record_timing_window;
std::atomic<bool> nr_timing_enabled{true};
std::atomic<uint64_t> nr_timing_epoch{0};
std::atomic<bool> nr_failed{false};
std::atomic<uint64_t> frame_serial{0};
std::atomic<unsigned> hdr_ready{0};
std::atomic<unsigned> motion_probe_claimed{0};
#endif
std::once_flag trigger_path_once;
std::wstring trigger_path;

bool user_armed_diagnostic() {
    try {std::call_once(trigger_path_once,[] {
        HMODULE module=nullptr;
        if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&record_deferred_identity),&module)) return;
        wchar_t path[MAX_PATH]{};
        if(!GetModuleFileNameW(module,path,MAX_PATH)) return;
        trigger_path=path;
        const auto slash=trigger_path.find_last_of(L"\\/");
        if(slash==std::wstring::npos) {trigger_path.clear();return;}
        trigger_path.resize(slash+1);
        trigger_path+=L"NRB-deferred-identity.enable";
    });} catch(...) {return false;}
    if(trigger_path.empty()) return false;
    const DWORD attr=GetFileAttributesW(trigger_path.c_str());
    return attr!=INVALID_FILE_ATTRIBUTES &&
           !(attr&FILE_ATTRIBUTE_DIRECTORY);
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource,
    D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource=resource;
    b.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore=before;
    b.Transition.StateAfter=after;
    return b;
}
#ifdef NRB_LIVE_NR_TEST
bool pool_terminal_proof(Pending& item,uint64_t done) {
    auto* proof=item.hdr?item.hdr->resource_proof():nullptr;
    if(!proof) return false;
    if(proof->poisoned() || FAILED(item.device->GetDeviceRemovedReason())) {
        proof->poison();item.hdr->quarantine_resource_lease();return false;
    }
    // Exact Pending completion, signaled after all inserted lists and the
    // remaining original batch. No prepare fence or unrelated fence suffices.
    proof->consumer_completed(done,true);
    if(proof->processor_called() && !proof->processor_retired() && !proof->poisoned()) {
        bool retired_ok=false;
        try {
            retired_ok=proof->processor_safe() && (item.nr_helper_retired ||
                (item.nr_processor.retire &&
                 item.nr_processor.retire(item.nr_processor.context,item.completion.Get(),1)==0));
        } catch(...) {}
        proof->processor_retirement(retired_ok);
        if(retired_ok) {item.nr_submitted=false;item.nr_helper_retired=true;}
    }
    if(!proof->terminal_safe()) {item.hdr->quarantine_resource_lease();return false;}
    return true;
}
// The helper owns one output. A second callback in the SAME intercepted batch
// must observe the preceding output's actual consumer completion and retire it.
// This extra fence follows that output's composite + deferred SR. It NEVER
// replaces Pending::completion (the final original-batch lifetime/pool proof).
bool retire_previous_for_reuse(Pending& previous,ID3D12CommandQueue* queue) {
    if(previous.nr_helper_retired) return true;
    bool ok=false;
    try {
        if(previous.nr_submitted && previous.nr_processor.retire &&
           SUCCEEDED(previous.device->GetDeviceRemovedReason()) &&
           (previous.nr_reuse_completion || SUCCEEDED(previous.device->CreateFence(0,
                D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(previous.nr_reuse_completion.GetAddressOf()))))) {
            const auto* exact=previous.nr_reuse_completion.Get();
            ok=gpuhandoff::retire_previous_consumer(exact,1,
                [&] {return SUCCEEDED(queue->Signal(previous.nr_reuse_completion.Get(),1));},
                [&] {
                    HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
                    const auto registered=event?previous.nr_reuse_completion->SetEventOnCompletion(1,event):E_FAIL;
                    const auto waited=event && SUCCEEDED(registered)?WaitForSingleObject(event,10000):WAIT_FAILED;
                    const auto done=previous.nr_reuse_completion->GetCompletedValue();
                    if(event) CloseHandle(event);
                    return gpuhandoff::ReuseObservation{exact,done,SUCCEEDED(registered) && waited==WAIT_OBJECT_0,
                        SUCCEEDED(previous.device->GetDeviceRemovedReason())};
                },
                [&] {return previous.nr_processor.retire(previous.nr_processor.context,
                    previous.nr_reuse_completion.Get(),1)==0;});
        }
    } catch(...) {ok=false;}
    gpuhandoff::previous_consumer_wait(ok);
    if(ok) previous.nr_helper_retired=true;
    else {nr_failed.store(true);if(previous.hdr) previous.hdr->quarantine_resource_lease();}
    return ok;
}
#endif
void reap_locked() {
    for(size_t i=0;i<pending.size();) {
        const uint64_t done=pending[i]->submitted?
            pending[i]->completion->GetCompletedValue():0;
        if(done>=1 && done!=UINT64_MAX) {
#ifdef NRB_LIVE_NR_TEST
            if(pending[i]->hdr && pending[i]->hdr->resource_proof()) {
                if(!pool_terminal_proof(*pending[i],done)) {++i;continue;}
            } else {
                auto& item=*pending[i];
                if(item.nr_processor_called && !item.nr_helper_retired) {
                    bool ok=false;
                    try {ok=item.nr_submitted && item.nr_processor.retire &&
                        SUCCEEDED(item.device->GetDeviceRemovedReason()) &&
                        item.nr_processor.retire(item.nr_processor.context,item.completion.Get(),1)==0;}
                    catch(...) {}
                    if(!ok) {nr_failed.store(true);gpuhandoff::process_failed();++i;continue;}
                    item.nr_helper_retired=true;item.nr_submitted=false;
                }
            }
#endif
            pending.erase(pending.begin()+i);
            retired.fetch_add(1,std::memory_order_relaxed);
        } else ++i;
    }
}
bool matching_device(ID3D12Device* device,ID3D12Resource* resource) {
    ComPtr<ID3D12Device> owner;
    return SUCCEEDED(resource->GetDevice(IID_PPV_ARGS(owner.GetAddressOf()))) &&
           owner.Get()==device;
}
}

bool record_deferred_identity(ID3D12GraphicsCommandList* game_list,
    const NVSDK_NGX_Handle* handle,NVSDK_NGX_Parameter* params,
    PFN_NVSDK_NGX_ProgressCallback callback,NgxEvaluate original,
    uint64_t game_list_generation,const NRB_Controls& frame_controls,
    NRB_Route route,NRB_RouteEvidence evidence,
    NVSDK_NGX_Result& result,flashdiag::Context* diagnostic) {
#ifdef NRB_LIVE_NR_TEST
    const bool record_timed=nr_timing_enabled.load(std::memory_order_acquire);
    const uint64_t record_epoch=nr_timing_epoch.load(std::memory_order_acquire);
    const auto record_started=record_timed?std::chrono::steady_clock::now():
        std::chrono::steady_clock::time_point{};
#endif
    flashdiag::Context diag=diagnostic?*diagnostic:flashdiag::Context{};
    const auto reject=[&](flashdiag::Reason reason,uint64_t detail=0,
                          uint64_t detail2=0) {
        if(diagnostic) diagnostic->record_reason=uint32_t(reason);
        flashdiag::recorder.event(flashdiag::EventKind::record_reject,
            diag,reason,detail,detail2);
        return false;
    };
    if(!deferred_identity_tail_verified(route))
        return reject(flashdiag::Reason::tail_unverified);
    if(!user_armed_diagnostic()) return reject(flashdiag::Reason::disarmed);
    if(!game_list || !handle || !params || !original || !game_list_generation)
        return reject(flashdiag::Reason::arguments);
    if(attempts.fetch_add(1,std::memory_order_relaxed)>=
#ifdef NRB_LIVE_NR_TEST
            100000
#else
            8
#endif
       ) return reject(flashdiag::Reason::attempt_budget);
    ID3D12Resource *color=nullptr,*output=nullptr;
    if(params->Get(NVSDK_NGX_Parameter_Color,&color)!=NVSDK_NGX_Result_Success ||
       params->Get(NVSDK_NGX_Parameter_Output,&output)!=NVSDK_NGX_Result_Success ||
        !color || !output) return reject(flashdiag::Reason::textures);
    const auto source_desc=color->GetDesc();
    const auto output_desc=output->GetDesc();
    if(source_desc.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
       source_desc.Format!=DXGI_FORMAT_R11G11B10_FLOAT ||
       source_desc.Width!=1280 || source_desc.Height!=720 ||
       source_desc.DepthOrArraySize!=1 || source_desc.MipLevels!=1 ||
       source_desc.SampleDesc.Count!=1 ||
       output_desc.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
       output_desc.Width!=2560 || output_desc.Height!=1440 ||
       output_desc.DepthOrArraySize!=1 || output_desc.MipLevels!=1 ||
        output_desc.SampleDesc.Count!=1)
        return reject(flashdiag::Reason::textures,
            (source_desc.Width<<32)|source_desc.Height,
            (output_desc.Width<<32)|output_desc.Height);

    {
        std::lock_guard guard(pending_mutex);
        reap_locked();
        if(pending.size()>=8)
            return reject(flashdiag::Reason::pending_full,pending.size());
        try {pending.reserve(8);}
        catch(...) {return reject(flashdiag::Reason::allocation);}
    }

    std::unique_ptr<Pending> owned;
    try {owned=std::make_unique<Pending>();}
    catch(...) {return reject(flashdiag::Reason::allocation);}
    Pending& entry=*owned;
    entry.diagnostic=diag;
#ifdef NRB_LIVE_NR_TEST
    entry.record_timed=record_timed;entry.record_timing_epoch=record_epoch;
    entry.record_started=record_started;
#endif
    entry.game_list=game_list;
    entry.original_color=color;
    entry.output=output;
    entry.frame_controls=frame_controls;
    entry.route=route;
    entry.evidence=evidence;
    ID3D12Resource* optional=nullptr;
    if(params->Get(NVSDK_NGX_Parameter_MotionVectors,&optional)==NVSDK_NGX_Result_Success && optional)
        entry.motion=optional;
    int reset=0;
    params->Get(NVSDK_NGX_Parameter_Reset,&reset);
    entry.game_reset=reset!=0;
    params->Get(NVSDK_NGX_Parameter_MV_Scale_X,&entry.motion_scale_x);
    params->Get(NVSDK_NGX_Parameter_MV_Scale_Y,&entry.motion_scale_y);
    entry.game_motion=frame_controls.history==NRB_HISTORY_REFERENCE ||
                      frame_controls.history==NRB_HISTORY_FUSED;
    unsigned int motion_base_x=0,motion_base_y=0;
    params->Get(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X,&motion_base_x);
    params->Get(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y,&motion_base_y);
    int feature_flags=0;
    params->Get(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags,&feature_flags);
    if(entry.game_motion && (!entry.motion ||
       motion_base_x || motion_base_y ||
       (feature_flags & NVSDK_NGX_DLSS_Feature_Flags_MVJittered) ||
       !std::isfinite(entry.motion_scale_x) || !std::isfinite(entry.motion_scale_y) ||
       entry.motion_scale_x==0 || entry.motion_scale_y==0 ||
       std::fabs(entry.motion_scale_x)>4096 || std::fabs(entry.motion_scale_y)>4096))
        return reject(flashdiag::Reason::motion_contract,
            (uint64_t(motion_base_x)<<32)|motion_base_y,
            unsigned(feature_flags));
    optional=nullptr;
    if(params->Get(NVSDK_NGX_Parameter_Depth,&optional)==NVSDK_NGX_Result_Success && optional)
        entry.depth=optional;
    optional=nullptr;
    if(params->Get(NVSDK_NGX_Parameter_ExposureTexture,&optional)==NVSDK_NGX_Result_Success && optional)
        entry.exposure=optional;
    entry.game_list_generation=game_list_generation;
    if(FAILED(game_list->GetDevice(IID_PPV_ARGS(entry.device.GetAddressOf()))) ||
       !matching_device(entry.device.Get(),color) ||
        !matching_device(entry.device.Get(),output)) return reject(flashdiag::Reason::device);
#ifdef NRB_LIVE_NR_TEST
    const auto resource_started=record_timed?std::chrono::steady_clock::now():
        std::chrono::steady_clock::time_point{};
#endif
    if(FAILED(entry.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(entry.prep_allocator.GetAddressOf()))) ||
       FAILED(entry.device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,
        entry.prep_allocator.Get(),nullptr,
        IID_PPV_ARGS(entry.prep_list.GetAddressOf()))) ||
       FAILED(entry.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(entry.allocator.GetAddressOf()))) ||
       FAILED(entry.device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,
        entry.allocator.Get(),nullptr,
        IID_PPV_ARGS(entry.deferred_list.GetAddressOf()))) ||
       FAILED(entry.device->CreateFence(0,D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(entry.completion.GetAddressOf())))) return reject(flashdiag::Reason::record_resources);
    auto copy_desc=source_desc;
    copy_desc.Flags=
#ifdef NRB_LIVE_NR_TEST
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
#else
        D3D12_RESOURCE_FLAG_NONE;
#endif
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type=D3D12_HEAP_TYPE_DEFAULT;
    if(FAILED(entry.device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,
        &copy_desc,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,
        IID_PPV_ARGS(entry.color_copy.GetAddressOf())))) return reject(flashdiag::Reason::record_resources);

#ifdef NRB_LIVE_NR_TEST
    if(record_timed) entry.record_resources_ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-resource_started).count();
#endif
    auto* prep=entry.prep_list.Get();
    // The game's post-Evaluate tail leaves the DLSS color in UAV, but the
    // native XeSS route leaves it in a combined read state. The private list
    // executes only after that entire game list finishes.
    const auto color_tail_state=route==NRB_ROUTE_XESS?
        static_cast<D3D12_RESOURCE_STATES>(D3D12_RESOURCE_STATE_COPY_SOURCE |
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE):
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    auto color_to_copy=transition(color,color_tail_state,
        D3D12_RESOURCE_STATE_COPY_SOURCE);
    prep->ResourceBarrier(1,&color_to_copy);
    prep->CopyResource(entry.color_copy.Get(),color);
    D3D12_RESOURCE_BARRIER ready[2]={
        transition(color,D3D12_RESOURCE_STATE_COPY_SOURCE,color_tail_state),
        transition(entry.color_copy.Get(),D3D12_RESOURCE_STATE_COPY_DEST,
            route==NRB_ROUTE_XESS?
                static_cast<D3D12_RESOURCE_STATES>(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE):
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)};
    prep->ResourceBarrier(2,ready);
#ifdef NRB_LIVE_NR_TEST
    const auto hdr_started=record_timed?std::chrono::steady_clock::now():
        std::chrono::steady_clock::time_point{};
    wchar_t module_path[MAX_PATH]{};
    HMODULE module=nullptr;
    if(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&record_deferred_identity),&module) &&
       GetModuleFileNameW(module,module_path,MAX_PATH)) {
        std::wstring shader=module_path;
        shader.resize(shader.find_last_of(L"\\/")+1);
        shader+=L"nr_hdr_proxy.hlsl";
        entry.hdr=std::make_unique<HdrProxy>();
        if(!entry.hdr->initialize(entry.device.Get(),color,entry.motion.Get(),
            entry.color_copy.Get(),shader,&hdr_cache_session(),&hdr_resource_pool_session())) entry.hdr.reset();
    }
    if(record_timed) entry.record_hdr_ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-hdr_started).count();
    if(entry.game_motion && !entry.hdr) return reject(flashdiag::Reason::hdr_missing);
    if(entry.hdr) {
        if(entry.game_motion && !entry.hdr->game_motion_available()) return reject(flashdiag::Reason::motion_proxy);
        hdr_ready.fetch_add(1,std::memory_order_relaxed);
        entry.hdr->record_prepare(prep,color,entry.game_motion,
            entry.motion_scale_x,entry.motion_scale_y,color_tail_state);
        // Sample a few later frames too: the first frames after switching
        // modes can have a stationary camera and therefore zero motion.
        const unsigned motion_index=entry.game_motion?
            motion_probe_claimed.fetch_add(1,std::memory_order_relaxed):0;
        if(entry.game_motion) entry.diagnostic.motion_probe_index=motion_index;
        if(entry.game_motion && (motion_index<4 ||
           (motion_index<=480 && motion_index%60==0))) {
            entry.motion_probe_index=motion_index;
            UINT64 bytes=0;
            auto motion_desc=entry.hdr->nr_motion()->GetDesc();
            entry.device->GetCopyableFootprints(&motion_desc,0,1,0,
                &entry.motion_probe_layout,nullptr,nullptr,&bytes);
            D3D12_HEAP_PROPERTIES readback_heap{};
            readback_heap.Type=D3D12_HEAP_TYPE_READBACK;
            D3D12_RESOURCE_DESC buffer{};
            buffer.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
            buffer.Width=bytes;buffer.Height=1;buffer.DepthOrArraySize=1;
            buffer.MipLevels=1;buffer.SampleDesc.Count=1;
            buffer.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            if(bytes && SUCCEEDED(entry.device->CreateCommittedResource(
                &readback_heap,D3D12_HEAP_FLAG_NONE,&buffer,
                D3D12_RESOURCE_STATE_COPY_DEST,nullptr,
                IID_PPV_ARGS(entry.motion_probe_readback.GetAddressOf())))) {
                entry.diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::motion_probe_recorded);
                auto to_copy=transition(entry.hdr->nr_motion(),
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COPY_SOURCE);
                prep->ResourceBarrier(1,&to_copy);
                D3D12_TEXTURE_COPY_LOCATION src{},dst{};
                src.pResource=entry.hdr->nr_motion();
                src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                src.SubresourceIndex=0;
                dst.pResource=entry.motion_probe_readback.Get();
                dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                dst.PlacedFootprint=entry.motion_probe_layout;
                prep->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
                auto restored=transition(entry.hdr->nr_motion(),
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                prep->ResourceBarrier(1,&restored);
            }
        }
        if(FAILED(entry.device->CreateFence(0,D3D12_FENCE_FLAG_NONE,
            IID_PPV_ARGS(entry.prepared.GetAddressOf()))) ||
           FAILED(entry.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(entry.composite_allocator.GetAddressOf()))) ||
           FAILED(entry.device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,
            entry.composite_allocator.Get(),nullptr,
            IID_PPV_ARGS(entry.composite_list.GetAddressOf())))) return reject(flashdiag::Reason::record_resources);
    }
#endif
    if(FAILED(prep->Close())) return reject(flashdiag::Reason::record_close);
    auto* list=entry.deferred_list.Get();
    auto output_write=transition(output,static_cast<D3D12_RESOURCE_STATES>(
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->ResourceBarrier(1,&output_write);
    params->Set(NVSDK_NGX_Parameter_Color,entry.color_copy.Get());
#ifdef NRB_LIVE_NR_TEST
    const auto xess_started=record_timed?std::chrono::steady_clock::now():
        std::chrono::steady_clock::time_point{};
#endif
    result=original(list,handle,params,callback);
#ifdef NRB_LIVE_NR_TEST
    if(record_timed) entry.record_xess_ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-xess_started).count();
#endif
    params->Set(NVSDK_NGX_Parameter_Color,color);
    if(result!=NVSDK_NGX_Result_Success)
        return reject(flashdiag::Reason::original_evaluate,unsigned(result));
    auto output_read=transition(output,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        static_cast<D3D12_RESOURCE_STATES>(
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
    list->ResourceBarrier(1,&output_read);
    if(FAILED(list->Close())) return reject(flashdiag::Reason::record_close);
#ifdef NRB_LIVE_NR_TEST
    if(entry.hdr && entry.hdr->resource_proof()) entry.hdr->resource_proof()->private_records_closed();
#endif
    {
        std::lock_guard guard(pending_mutex);
        reap_locked();
        entry.diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::recorded);
        flashdiag::recorder.progress(entry.diagnostic);
#ifdef NRB_LIVE_NR_TEST
        if(record_timed) entry.record_finished=std::chrono::steady_clock::now();
#endif
        pending.push_back(std::move(owned));
    }
    recorded.fetch_add(1,std::memory_order_relaxed);
    flashdiag::recorder.count(flashdiag::Counter::delegated);
    return true;
}

unsigned submit_deferred_identity(ID3D12CommandQueue* queue,UINT count,
    ID3D12CommandList* const* lists,const SubmitObservation* observations,
    void (STDMETHODCALLTYPE *original_execute)(ID3D12CommandQueue*,UINT,
        ID3D12CommandList* const*),const NRB_Processor* processor,
    const NRB_Controls* controls,void (*log)(const char*)) {
    if(!queue || !count || !lists || !original_execute) return 0;
    std::lock_guard guard(pending_mutex);
    reap_locked();
    if(pending.empty()) return 0;
    struct Match {UINT batch_index;Pending* item;};
    std::array<Match,8> matches{};
    unsigned matched=0;
    for(size_t p=0;p<pending.size();p++) {
        const auto& item=*pending[p];
        if(item.submitted) continue;
        for(UINT i=0;i<count;i++) if(item.game_list.Get()==lists[i]) {
            if(observations && (!observations[i].correlated ||
               item.game_list_generation!=observations[i].generation)) {
                generation_mismatches.fetch_add(1,std::memory_order_relaxed);
                flashdiag::recorder.count(flashdiag::Counter::generation_mismatch);
                flashdiag::recorder.event(flashdiag::EventKind::generation,
                    item.diagnostic,flashdiag::Reason::generation,
                    observations[i].generation,unsigned(observations[i].correlated));
            }
            if(matched==matches.size()) return 0;
            pending[p]->diagnostic.submit_batch=observations?observations[i].batch:0;
            matches[matched++]={i,pending[p].get()};break;
        }
    }
    if(!matched) return 0;
    std::sort(matches.begin(),matches.begin()+matched,
        [](const Match& a,const Match& b) {return a.batch_index<b.batch_index;});
    // A Direct list can only be executed on a Direct queue of its own device.
    // Allocate all fallible D3D12 objects while recording, before withholding
    // XeSS from the game list. Once that list appears here, always insert the
    // private list; a failed retirement Signal must not drop the frame.
    UINT begin=0;
#ifdef NRB_LIVE_NR_TEST
    Pending* previous_processor=nullptr;
#endif
    for(unsigned m=0;m<matched;m++) {
        const UINT index=matches[m].batch_index;
        if(index>=begin) {
            original_execute(queue,index+1-begin,lists+begin);
            begin=index+1;
        }
        flashdiag::recorder.count(flashdiag::Counter::submitted);
        matches[m].item->diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::submitted);
        ID3D12CommandList* prep=matches[m].item->prep_list.Get();
#ifdef NRB_LIVE_NR_TEST
        if(matches[m].item->hdr && matches[m].item->hdr->resource_proof())
            matches[m].item->hdr->resource_proof()->prepare_submitted();
#endif
        original_execute(queue,1,&prep);
#ifdef NRB_LIVE_NR_TEST
        auto& item=*matches[m].item;
        item.nr_queue_exposed=true;
        // Keep the single helper's previous output alive through real completion.
        // Failure latches NR; never ignore busy or reset the helper's live frame.
        if(previous_processor && !retire_previous_for_reuse(*previous_processor,queue))
            gpuhandoff::process_failed();
        previous_processor=nullptr;
        auto fallback_reason=flashdiag::Reason::prepared_wait;
        if(!item.hdr) fallback_reason=flashdiag::Reason::hdr_missing;
        else if(!processor || !processor->process || !processor->retire)
            fallback_reason=flashdiag::Reason::processor_missing;
        else if(!controls || !controls->enabled) fallback_reason=flashdiag::Reason::disabled;
        else if(nr_failed.load()) fallback_reason=flashdiag::Reason::latched_failure;
        if(item.hdr && processor && processor->process && processor->retire &&
           controls && controls->enabled && !nr_failed.load()) {
            item.nr_timing_epoch=nr_timing_epoch.load(std::memory_order_acquire);
            item.nr_timed=nr_timing_enabled.load(std::memory_order_acquire);
            const auto started=item.nr_timed?std::chrono::steady_clock::now():
                std::chrono::steady_clock::time_point{};
            bool valid=false;
            NRB_Result result{};
            const HRESULT prepare_signal=queue->Signal(item.prepared.Get(),1);
            if(SUCCEEDED(prepare_signal)) {
                // Signal the ACTUAL prepared point first. The first enable frame
                // has no confirmed capability and follows the CPU wait below.
                const bool bypass=gpuhandoff::prepared_bypass(*processor,item.device.Get(),queue,
                    item.motion_probe_readback.Get()!=nullptr);
                item.nr_prepared_bypassed=bypass;
                HANDLE event=bypass?nullptr:CreateEventW(nullptr,FALSE,FALSE,nullptr);
                if(bypass || event) {
                    const HRESULT registered=bypass?S_OK:item.prepared->SetEventOnCompletion(1,event);
                    const DWORD waited=!bypass && SUCCEEDED(registered)?WaitForSingleObject(event,10000):WAIT_FAILED;
                    const auto prepared_done=item.hdr->resource_proof()?item.prepared->GetCompletedValue():0;
                    const bool pooled_prepare_ok=!item.hdr->resource_proof() ||
                        (prepared_done>=1 && prepared_done!=UINT64_MAX &&
                         SUCCEEDED(item.device->GetDeviceRemovedReason()));
                    if(bypass || (SUCCEEDED(registered) && waited==WAIT_OBJECT_0 && pooled_prepare_ok)) {
                        if(auto* proof=item.hdr->resource_proof()) {
                            if(bypass) proof->prepare_gpu_ordered(true);
                            else proof->prepare_completed(true);
                        }
                        if(item.motion_probe_readback && log) {
                            void* mapped=nullptr;
                            if(SUCCEEDED(item.motion_probe_readback->Map(0,nullptr,
                                &mapped)) && mapped) {
                                unsigned tested=0,nonzero=0,max_half=0;
                                const auto* data=static_cast<const uint8_t*>(mapped);
                                for(unsigned y=0;y<720;y+=16) {
                                    const auto* row=reinterpret_cast<const uint16_t*>(
                                        data+y*item.motion_probe_layout.Footprint.RowPitch);
                                    for(unsigned x=0;x<1280;x+=16) {
                                        const unsigned u=row[2*x]&0x7fff;
                                        const unsigned v=row[2*x+1]&0x7fff;
                                        ++tested;
                                        nonzero+=(u|v)!=0;
                                        max_half=std::max({max_half,u,v});
                                    }
                                }
                                item.motion_probe_readback->Unmap(0,nullptr);
                                char line[240];
                                sprintf_s(line,"game_motion_probe sample=%u pixels=%u nonzero=%u max_half_bits=%u scale=[%.3f,%.3f] reset=%u\r\n",
                                    item.motion_probe_index,tested,nonzero,max_half,item.motion_scale_x,
                                    item.motion_scale_y,unsigned(item.game_reset));
                                log(line);
                            }
                        }
                        if(item.nr_timed) item.nr_prepared=std::chrono::steady_clock::now();
                        NRB_Frame frame{};
                        frame.abi_size=sizeof(frame);frame.route=item.route;
                        frame.route_evidence=item.evidence;
                        frame.device=item.device.Get();frame.queue=queue;
                        frame.command_list=item.prep_list.Get();
                        frame.color=item.hdr->proxy_color();
                        frame.motion=item.hdr->nr_motion();
                        frame.output=item.output.Get();
                        frame.render_width=1280;frame.render_height=720;
                        frame.output_width=2560;frame.output_height=1440;
                        frame.motion_scale_x=1;frame.motion_scale_y=1;
                        frame.motion_scale_origin=item.route==NRB_ROUTE_FSR?
                            NRB_MV_FSR_TO_NGX:item.route==NRB_ROUTE_XESS?
                            NRB_MV_XESS_TO_NGX:NRB_MV_DLSS_NGX;
                        frame.pre_exposure=1;frame.exposure_scale=1;
                        frame.low_resolution_motion=1;
                        frame.frame_id=frame_serial.fetch_add(1)+1;
                        frame.reset_history=frame.frame_id==1 || item.game_reset;
                        frame.controls=item.frame_controls;
                        result.abi_size=sizeof(result);
                        fallback_reason=flashdiag::Reason::processor_result;
                        item.diagnostic.nr_frame_id=frame.frame_id;
                        item.diagnostic.nr_reset=frame.reset_history!=0;
                        item.diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::processor_called);
                        flashdiag::recorder.count(flashdiag::Counter::nr_started);
                        flashdiag::recorder.progress(item.diagnostic);
                        int processor_result=-1;
                        item.nr_processor=*processor;item.nr_processor_called=true;
                        if(auto* proof=item.hdr->resource_proof()) proof->processor_begin();
                        try {
                            gpuhandoff::CallbackScope producer_scope(*processor,frame,item.prepared.Get(),1,!bypass);
                            flashdiag::ActiveContext context_scope(item.diagnostic);
                            processor_result=processor->process(processor->context,&frame,&result);
                            if(processor_result==0) {
                                item.diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::processor_return_ok);
                                flashdiag::recorder.count(flashdiag::Counter::nr_return_ok);
                            }
                            valid=processor_result==0 && result.color && result.ready_fence && result.ready_value &&
                                result.color->GetDesc().Format==DXGI_FORMAT_R32G32B32A32_FLOAT &&
                                result.color->GetDesc().Width==1280 && result.color->GetDesc().Height==720;
                        } catch(...) {valid=false;fallback_reason=flashdiag::Reason::cpp_exception;}
                        // Returned references, even malformed ones, stay with
                        // Pending when processor use is unproven (pool ON/OFF).
                        item.nr_output.Attach(result.color);item.nr_ready.Attach(result.ready_fence);
                        if(auto* proof=item.hdr->resource_proof()) if(!valid) proof->poison();
                        if(valid) {
                            item.diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::result_valid);
                            flashdiag::recorder.count(flashdiag::Counter::nr_processed);
                        } else flashdiag::recorder.event(flashdiag::EventKind::failure,item.diagnostic,
                            fallback_reason,uint64_t(int64_t(processor_result)));
                        if(item.nr_timed) item.nr_modeled=std::chrono::steady_clock::now();
                    }
                    else {
                        fallback_reason=FAILED(registered)?flashdiag::Reason::prepare_registration:
                            flashdiag::Reason::prepare_timeout;
                        flashdiag::recorder.event(flashdiag::EventKind::failure,item.diagnostic,
                            fallback_reason,uint64_t(uint32_t(registered)),waited);
                    }
                    if(event) CloseHandle(event);
                } else fallback_reason=flashdiag::Reason::prepare_event;
            } else fallback_reason=flashdiag::Reason::prepare_signal;
            if(valid) {
                fallback_reason=flashdiag::Reason::composite_or_wait;
                const HRESULT ready_wait=queue->Wait(item.nr_ready.Get(),result.ready_value);
                if(auto* proof=item.hdr->resource_proof()) proof->processor_ready(SUCCEEDED(ready_wait));
                bool composite_recorded=false;HRESULT composite_closed=S_FALSE;
                valid=SUCCEEDED(ready_wait) &&
                    (composite_recorded=
                    item.hdr->record_composite(item.composite_list.Get(),
                        item.original_color.Get(),item.nr_output.Get(),item.color_copy.Get(),
                        item.route==NRB_ROUTE_XESS?
                            static_cast<D3D12_RESOURCE_STATES>(
                                D3D12_RESOURCE_STATE_COPY_SOURCE |
                                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE):
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        item.route==NRB_ROUTE_XESS?
                            static_cast<D3D12_RESOURCE_STATES>(
                                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE):
                            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                        item.nr_prepared_bypassed)) &&
                    SUCCEEDED(composite_closed=item.composite_list->Close());
                if(!valid) fallback_reason=FAILED(ready_wait)?flashdiag::Reason::queue_wait:
                    !composite_recorded?flashdiag::Reason::composite_record:flashdiag::Reason::composite_close;
                if(valid) {
                    ID3D12CommandList* composite=item.composite_list.Get();
                    if(auto* proof=item.hdr->resource_proof()) {
                        proof->composite_closed();proof->composite_submitted();
                    }
                    original_execute(queue,1,&composite);
                    flashdiag::recorder.count(flashdiag::Counter::nr_composited);
                    item.diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::composite_submitted);
                    if(item.nr_timed) item.nr_pre_sr=std::chrono::steady_clock::now();
                    item.nr_processor=*processor;
                    item.nr_submitted=true;
                    item.nr_started=started;
                    previous_processor=&item;
                }
            }
            if(!valid) {
                if(item.hdr->resource_proof()) item.hdr->quarantine_resource_lease();
                nr_failed.store(true);
                gpuhandoff::process_failed();
                flashdiag::recorder.count(flashdiag::Counter::failure);
                flashdiag::recorder.event(flashdiag::EventKind::failure,
                    item.diagnostic,fallback_reason);
            }
        }
        if(!item.nr_submitted) {
            item.diagnostic_reason=fallback_reason;
            flashdiag::recorder.fallback(item.diagnostic,fallback_reason);
        }
#else
        (void)processor;(void)controls;
        flashdiag::recorder.fallback(matches[m].item->diagnostic,
            flashdiag::Reason::disabled);
#endif
        ID3D12CommandList* deferred=matches[m].item->deferred_list.Get();
#ifdef NRB_LIVE_NR_TEST
        if(matches[m].item->hdr && matches[m].item->hdr->resource_proof())
            matches[m].item->hdr->resource_proof()->deferred_submitted();
#endif
        original_execute(queue,1,&deferred);
        matches[m].item->diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::sr_submitted);
        flashdiag::recorder.progress(matches[m].item->diagnostic);
    }
    if(begin<count) original_execute(queue,count-begin,lists+begin);
    for(unsigned m=0;m<matched;m++) {
        auto& item=*matches[m].item;
        item.submitted=true;
        const HRESULT completion_signal=queue->Signal(item.completion.Get(),1);
#ifdef NRB_LIVE_NR_TEST
        if(item.hdr && item.hdr->resource_proof()) {
            item.hdr->resource_proof()->completion_signal(SUCCEEDED(completion_signal));
            if(FAILED(completion_signal)) item.hdr->quarantine_resource_lease();
        }
#endif
        if(FAILED(completion_signal)) {
            item.diagnostic_reason=flashdiag::Reason::completion_signal;
            signal_failures.fetch_add(1,std::memory_order_relaxed);
            flashdiag::recorder.count(flashdiag::Counter::failure);
            flashdiag::recorder.event(flashdiag::EventKind::failure,
                item.diagnostic,flashdiag::Reason::completion_signal);
        }
#ifdef NRB_LIVE_NR_TEST
        else if(item.nr_submitted) {
            HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
            const HRESULT registered=event?item.completion->SetEventOnCompletion(1,event):E_FAIL;
            const DWORD waited=event && SUCCEEDED(registered)?WaitForSingleObject(event,10000):WAIT_FAILED;
            const auto completed_value=item.completion->GetCompletedValue();
            const bool done=event && SUCCEEDED(registered) && waited==WAIT_OBJECT_0 &&
                completed_value>=1 && completed_value!=UINT64_MAX && SUCCEEDED(item.device->GetDeviceRemovedReason());
            if(event) CloseHandle(event);
            bool retired_ok=false;
            if(item.hdr && item.hdr->resource_proof()) {
                retired_ok=done && pool_terminal_proof(item,item.completion->GetCompletedValue());
                if(!retired_ok) item.hdr->quarantine_resource_lease();
            } else {
                try {retired_ok=done && (item.nr_helper_retired ||
                    item.nr_processor.retire(item.nr_processor.context,item.completion.Get(),1)==0);}
                catch(...) {retired_ok=false;}
            }
            if(retired_ok) {
                item.nr_submitted=false;item.nr_helper_retired=true;
                item.diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::retired);
                flashdiag::recorder.count(flashdiag::Counter::nr_retired);
                const auto ended=item.nr_timed?std::chrono::steady_clock::now():
                    std::chrono::steady_clock::time_point{};
                const uint64_t completed=nr_completed.fetch_add(1,std::memory_order_relaxed)+1;
                if(!item.nr_timed || !nr_timing_enabled.load(std::memory_order_acquire) ||
                   item.nr_timing_epoch!=nr_timing_epoch.load(std::memory_order_acquire))
                    continue;
                std::lock_guard timing_guard(nr_timing_mutex);
                if(!item.nr_timed || !nr_timing_enabled.load(std::memory_order_relaxed) ||
                   item.nr_timing_epoch!=nr_timing_epoch.load(std::memory_order_relaxed))
                    continue;
                const auto millis=[](auto begin,auto end) {
                    return std::chrono::duration<double,std::milli>(end-begin).count();
                };
                const double ms=millis(item.nr_started,ended);
                const double prep=millis(item.nr_started,item.nr_prepared);
                const double host=millis(item.nr_prepared,item.nr_modeled);
                const double composite=millis(item.nr_modeled,item.nr_pre_sr);
                const double tail=millis(item.nr_pre_sr,ended);
                if(item.record_timed && item.record_timing_epoch==item.nr_timing_epoch) {
                    record_timing_window.append(item.diagnostic.nr_frame_id,{
                        millis(item.record_started,item.record_finished),
                        item.record_resources_ms,item.record_hdr_ms,item.record_xess_ms,
                        millis(item.record_finished,item.nr_started),
                        millis(item.record_started,ended)});
                }
                const unsigned count=++nr_timing_count;
                const unsigned samples=std::min(32u,count);
                const size_t slot=(count-1)%nr_timing_samples[0].size();
                const auto update_stage=[&](size_t channel,std::atomic<double>& last,
                    std::atomic<double>& average,double value) {
                    if(count>nr_timing_samples[channel].size())
                        nr_timing_sums[channel]-=nr_timing_samples[channel][slot];
                    nr_timing_samples[channel][slot]=value;
                    nr_timing_sums[channel]+=value;
                    last.store(value,std::memory_order_relaxed);
                    average.store(nr_timing_sums[channel]/samples,
                        std::memory_order_relaxed);
                };
                update_stage(0,nr_last_ms,nr_average_ms,ms);
                update_stage(1,prep_last_ms,prep_average_ms,prep);
                update_stage(2,host_last_ms,host_average_ms,host);
                update_stage(3,composite_submit_last_ms,composite_submit_average_ms,composite);
                update_stage(4,tail_last_ms,tail_average_ms,tail);
                nr_sample_count.store(count,std::memory_order_release);
                if(log && (completed<=4 || completed%120==0)) {
                    char line[280];
                    sprintf_s(line,"nr_stage_timing frame=%llu input=%u history=%u prep_ms=%.3f host_ms=%.3f composite_submit_ms=%.3f tail_ms=%.3f total_ms=%.3f\r\n",
                        static_cast<unsigned long long>(completed),item.frame_controls.input_height,
                        unsigned(item.frame_controls.history),prep,host,composite,tail,ms);
                    log(line);
                }
            }
            else {
                nr_failed.store(true);
                gpuhandoff::process_failed();
                flashdiag::recorder.count(flashdiag::Counter::failure);
                item.diagnostic_reason=!event?flashdiag::Reason::completion_event:
                    FAILED(registered)?flashdiag::Reason::completion_registration:
                    !done?flashdiag::Reason::completion_timeout:flashdiag::Reason::retire_callback;
                flashdiag::recorder.event(flashdiag::EventKind::failure,
                    item.diagnostic,item.diagnostic_reason,uint64_t(uint32_t(registered)),waited);
            }
        }
#endif
    }
    for(unsigned m=0;m<matched;m++)
        flashdiag::recorder.finish(matches[m].item->diagnostic,matches[m].item->diagnostic_reason);
    // A failed Signal retains the submitted objects until process exit; never
    // recycle an allocator or texture that might still be used by the GPU.
    return matched;
}
unsigned deferred_identity_recorded() {return recorded.load(std::memory_order_relaxed);}
uint64_t deferred_nr_processed() {
#ifdef NRB_LIVE_NR_TEST
    return nr_completed.load(std::memory_order_relaxed);
#else
    return 0;
#endif
}
double deferred_nr_last_ms() {
#ifdef NRB_LIVE_NR_TEST
    return nr_last_ms.load(std::memory_order_relaxed);
#else
    return 0;
#endif
}
double deferred_nr_average_ms() {
#ifdef NRB_LIVE_NR_TEST
    return nr_average_ms.load(std::memory_order_relaxed);
#else
    return 0;
#endif
}
bool deferred_hdr_cache_stats(NRB_HdrCacheStats* stats) {
    if(!stats || stats->abi_size!=sizeof(NRB_HdrCacheStats)) return false;
    NRB_HdrCacheStats out{};out.abi_size=sizeof(out);
#ifdef NRB_LIVE_NR_TEST
    const auto value=hdr_cache_session().stats();out.enabled=value.enabled?1:0;
    out.shader_compile_calls=value.shader_compile_calls;
    out.shader_compile_failures=value.shader_compile_failures;
    out.source_reads=value.source_reads;
    out.source_hashes=value.source_hashes;
    out.pso_creates=value.pso_creates;
    out.root_signature_creates=value.root_signature_creates;
    out.pipeline_hits=value.pipeline_hits;
    out.pipeline_misses=value.pipeline_misses;
    out.uncached_initializations=value.uncached_initializations;
    out.shader_generation=value.shader_generation;
    out.resident_device_entries=value.resident_device_entries;
#endif
    *stats=out;return true;
}
bool deferred_hdr_set_resource_pool_enabled(bool enabled) {
#ifdef NRB_LIVE_NR_TEST
    hdr_resource_pool_session().set_enabled(enabled);return true;
#else
    (void)enabled;return false;
#endif
}
bool deferred_hdr_resource_pool_stats(HdrResourcePoolStats& stats) {
#ifdef NRB_LIVE_NR_TEST
    stats=hdr_resource_pool_session().stats();return true;
#else
    (void)stats;return false;
#endif
}
void deferred_hdr_set_cache_enabled(bool enabled) {
#ifdef NRB_LIVE_NR_TEST
    hdr_cache_session().set_enabled(enabled);
#else
    (void)enabled;
#endif
}
bool deferred_hdr_warmup_shader() {
#ifdef NRB_LIVE_NR_TEST
    HMODULE module=nullptr;wchar_t path[MAX_PATH]{};
    if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&record_deferred_identity),&module) ||
        !GetModuleFileNameW(module,path,MAX_PATH)) return false;
    std::wstring shader=path;const auto slash=shader.find_last_of(L"\\/");
    if(slash==std::wstring::npos) return false;
    shader.resize(slash+1);shader+=L"nr_hdr_proxy.hlsl";
    return hdr_cache_session().warmup_shader(shader);
#else
    return false;
#endif
}
bool deferred_nr_record_times(NRB_RecordTimes* times) {
    if(!times || times->abi_size!=sizeof(NRB_RecordTimes)) return false;
#ifdef NRB_LIVE_NR_TEST
    std::lock_guard guard(nr_timing_mutex);
    return record_timing_window.snapshot(times);
#else
    *times=NRB_RecordTimes{};times->abi_size=sizeof(NRB_RecordTimes);return false;
#endif
}
bool deferred_nr_stage_times(NRB_StageTimes* times) {
    if(!times || times->abi_size!=sizeof(NRB_StageTimes)) return false;
    NRB_StageTimes result{};
    result.abi_size=sizeof(result);
#ifdef NRB_LIVE_NR_TEST
    std::lock_guard guard(nr_timing_mutex);
    result.frames=nr_sample_count.load(std::memory_order_relaxed);
    result.prep_last_ms=prep_last_ms.load(std::memory_order_relaxed);
    result.prep_average_ms=prep_average_ms.load(std::memory_order_relaxed);
    result.host_last_ms=host_last_ms.load(std::memory_order_relaxed);
    result.host_average_ms=host_average_ms.load(std::memory_order_relaxed);
    result.composite_submit_last_ms=composite_submit_last_ms.load(std::memory_order_relaxed);
    result.composite_submit_average_ms=composite_submit_average_ms.load(std::memory_order_relaxed);
    result.tail_last_ms=tail_last_ms.load(std::memory_order_relaxed);
    result.tail_average_ms=tail_average_ms.load(std::memory_order_relaxed);
#endif
    *times=result;
    return result.frames>0;
}
namespace {
void reset_nr_timing_locked() {
#ifdef NRB_LIVE_NR_TEST
    nr_timing_count=0;
    record_timing_window.reset();
    nr_timing_sums.fill(0);
    for(auto& samples:nr_timing_samples) samples.fill(0);
    nr_last_ms.store(0,std::memory_order_relaxed);
    nr_average_ms.store(0,std::memory_order_relaxed);
    nr_sample_count.store(0,std::memory_order_relaxed);
    for(auto* value:{&prep_last_ms,&prep_average_ms,&host_last_ms,&host_average_ms,
                     &composite_submit_last_ms,&composite_submit_average_ms,
                     &tail_last_ms,&tail_average_ms})
        value->store(0,std::memory_order_relaxed);
#endif
}
}
bool deferred_nr_timing_enabled() {
#ifdef NRB_LIVE_NR_TEST
    return nr_timing_enabled.load(std::memory_order_acquire);
#else
    return false;
#endif
}
void deferred_nr_set_timing_enabled(bool enabled) {
#ifdef NRB_LIVE_NR_TEST
    std::lock_guard guard(nr_timing_mutex);
    if(nr_timing_enabled.load(std::memory_order_relaxed)==enabled) return;
    nr_timing_enabled.store(enabled,std::memory_order_release);
    nr_timing_epoch.fetch_add(1,std::memory_order_release);
    reset_nr_timing_locked();
#else
    (void)enabled;
#endif
}
void deferred_nr_reset_metrics() {
#ifdef NRB_LIVE_NR_TEST
    std::lock_guard guard(nr_timing_mutex);
    nr_timing_epoch.fetch_add(1,std::memory_order_release);
    reset_nr_timing_locked();
#endif
}
unsigned deferred_hdr_ready() {
#ifdef NRB_LIVE_NR_TEST
    return hdr_ready.load(std::memory_order_relaxed);
#else
    return 0;
#endif
}
bool deferred_nr_failed() {
#ifdef NRB_LIVE_NR_TEST
    return nr_failed.load(std::memory_order_relaxed);
#else
    return false;
#endif
}
unsigned deferred_identity_retired() {return retired.load(std::memory_order_relaxed);}
unsigned deferred_identity_signal_failures() {return signal_failures.load(std::memory_order_relaxed);}
unsigned deferred_identity_generation_mismatches() {return generation_mismatches.load(std::memory_order_relaxed);}
unsigned deferred_identity_invalidated() {return invalidated.load(std::memory_order_relaxed);}
void deferred_identity_list_reset(ID3D12GraphicsCommandList* list) {
    if(!list) return;
    std::lock_guard guard(pending_mutex);
    for(size_t i=0;i<pending.size();) {
        if(!pending[i]->submitted && pending[i]->game_list.Get()==list) {
#ifdef NRB_LIVE_NR_TEST
            // A recursive Reset during insertion must not erase GPU-exposed
            // ownership merely because the final submitted flag is not set.
            if(pending[i]->hdr && pending[i]->hdr->resource_proof() &&
               !pending[i]->hdr->resource_proof()->pristine_unsubmitted()) {
                pending[i]->hdr->quarantine_resource_lease();++i;continue;
            }
#endif
            pending[i]->diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::invalidated);
            flashdiag::recorder.count(flashdiag::Counter::invalidated);
            flashdiag::recorder.event(flashdiag::EventKind::invalidated,
                pending[i]->diagnostic,flashdiag::Reason::list_reset);
            flashdiag::recorder.finish(pending[i]->diagnostic,flashdiag::Reason::list_reset);
            pending.erase(pending.begin()+i);
            invalidated.fetch_add(1,std::memory_order_relaxed);
        } else ++i;
    }
}
void observe_deferred_identity_tail(const TailSummary& s) {
    if(!s.valid || s.route==NRB_ROUTE_UNKNOWN || s.route>NRB_ROUTE_XESS) return;
    const bool safe=tail_safe_for_deferred_identity(s);
    if(safe) tail_valid_samples[s.route].fetch_add(1,std::memory_order_relaxed);
    tail_samples[s.route].fetch_add(1,std::memory_order_release);
}
void reset_deferred_identity_tail(NRB_Route route) {
    if(route==NRB_ROUTE_UNKNOWN || route>NRB_ROUTE_XESS) return;
    tail_samples[route].store(0,std::memory_order_release);
    tail_valid_samples[route].store(0,std::memory_order_release);
}
bool deferred_identity_tail_verified(NRB_Route route) {
    return route!=NRB_ROUTE_UNKNOWN && route<=NRB_ROUTE_XESS &&
           tail_samples[route].load(std::memory_order_acquire)>=8 &&
           tail_valid_samples[route].load(std::memory_order_relaxed)==8;
}
}


NRB_API int NRB_SetHdrResourcePoolEnabled(int enabled) {
    return nrb::deferred_hdr_set_resource_pool_enabled(enabled!=0)?1:0;
}
NRB_API int NRB_GetHdrResourcePoolStats(NRB_HdrResourcePoolStats* stats) {
    if(!stats || stats->abi_size!=sizeof(*stats)) return 0;
#ifdef NRB_LIVE_NR_TEST
    nrb::HdrResourcePoolStats value;
    if(!nrb::deferred_hdr_resource_pool_stats(value)) return 0;
    NRB_HdrResourcePoolStats out{};out.abi_size=sizeof(out);
    out.enabled=value.enabled;out.capacity=nrb::HdrProxyResourcePool::capacity;
    out.shutdown=value.shutdown;
    out.acquires=value.acquires;
    out.hits=value.hits;
    out.creates=value.creates;
    out.disabled_fallbacks=value.disabled_fallbacks;
    out.busy_fallbacks=value.busy_fallbacks;
    out.creation_failures=value.creation_failures;
    out.retire_returns=value.retire_returns;
    out.unsubmitted_returns=value.unsubmitted_returns;
    out.quarantines=value.quarantines;
    out.evictions=value.evictions;
    out.generation=value.generation;
    out.resident_slots=value.resident_slots;
    out.free_slots=value.free_slots;
    out.leased_slots=value.leased_slots;
    out.quarantined_slots=value.quarantined_slots;
    out.texture_bytes=value.texture_bytes;
    out.peak_texture_bytes=value.peak_texture_bytes;
    out.descriptor_bytes_estimate=value.descriptor_bytes_estimate;
    out.max_texture_bytes=nrb::HdrProxyResourcePool::max_texture_bytes;
    *stats=out;return 1;
#else
    return 0;
#endif
}
