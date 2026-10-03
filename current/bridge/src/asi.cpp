#include "nr_bridge.h"
#include "nr_gpu_handoff.h"
#include "nr_core.h"
#include "nr_diag.h"
#include "nr_python.h"
#include "nr_sync_probe.h"
#include "periodic_flash_native_diag.h"
#include <algorithm>
#ifdef NRB_XESS_IDENTITY_TEST
#include "xess_identity.h"
#endif
#ifdef NRB_DEFERRED_IDENTITY_TEST
#include "deferred_identity.h"
#endif
#ifdef NRB_TAIL_ORDER_PROBE
#include "nr_tail_probe.h"
#endif
#ifdef NRB_ONE_FRAME_COLOR_READBACK
#include "nr_one_frame_readback.h"
#endif
#include <nvsdk_ngx.h>
#include <d3d12.h>
#include <windows.h>
#include <detours/detours.h>
#include <intrin.h>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <cstring>
#include <vector>
#include <array>
#include <chrono>
#include <cstdio>

namespace {
using Evaluate = NVSDK_NGX_Result (__cdecl *)(ID3D12GraphicsCommandList*,
    const NVSDK_NGX_Handle*, NVSDK_NGX_Parameter*, PFN_NVSDK_NGX_ProgressCallback);
Evaluate original = nullptr;
// Streamline's public slEvaluateFeature(feature=0) is the authoritative DLSS
// tag when the NGX call is nested synchronously from sl.common.dll.
using SLEvaluate = int32_t (__cdecl *)(uint32_t,const void*,const void**,
                                       uint32_t,void*);
SLEvaluate original_sl_evaluate=nullptr;
thread_local bool in_sl_dlss_evaluate=false;
// These tags are set only while the game's public XeSS/FSR entry is executing.
// OptiScaler's nested NGX call can then be attributed without relying on
// optional parameter names or on whichever upscaler was selected previously.
thread_local NRB_Route in_upscaler_api=NRB_ROUTE_UNKNOWN;
using XeSSExecute = int32_t (__cdecl *)(void*,ID3D12GraphicsCommandList*,const void*);
using FSR3Dispatch = int32_t (__cdecl *)(void*,const void*);
XeSSExecute original_xess_execute=nullptr;
FSR3Dispatch original_fsr3_dispatch=nullptr;
#ifdef NRB_TAIL_ORDER_PROBE
void schedule_list_hooks(ID3D12GraphicsCommandList* list);
#endif
// Prefix of the public FFX FSR3 dispatch ABI. This diagnostic reads the
// descriptor without changing the game's resources or dispatch arguments.
// The audited OptiScaler source used to build this ASI omits the FFX headers.
struct Fsr3ResourceProbe {
    void* resource;
    uint32_t type,format,width,height,depth,mip_count,flags,usage,state;
    wchar_t name[64];
};
static_assert(sizeof(Fsr3ResourceProbe)==176);
struct Fsr3DispatchProbe {
    void* command_list;
    Fsr3ResourceProbe resources[10];
    float jitter_x,jitter_y,motion_scale_x,motion_scale_y;
    uint32_t render_width,render_height;
};
static_assert(offsetof(Fsr3DispatchProbe,resources)==8);
static_assert(offsetof(Fsr3DispatchProbe,render_width)==1784);
HMODULE host = nullptr;
std::atomic<int> init_state{0};
std::mutex state_mutex;
NRB_Processor processor{};
ID3D12CommandQueue* queue = nullptr;
ID3D12Fence* completion = nullptr;
HANDLE completion_event = nullptr;
uint64_t completion_value = 0;
uint64_t frame_id = 0;
uint64_t last_processed_id = 0, control_epoch = 0, last_processed_epoch = 0;
uint32_t last_width = 0, last_height = 0;
NRB_Route last_route = NRB_ROUTE_UNKNOWN;
NRB_Controls controls{sizeof(NRB_Controls), 0, 360, 0, NRB_HISTORY_FUSED,
    1, 0, 0, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f};
NRB_Result borrowed{};
// A Boolean from a caller is not source-submission evidence. The signed host
// exposes no current-frame producer callback, so this remains fail closed.
std::atomic<bool> source_submission_proven{false};
std::atomic<NRB_Status> last_status{NRB_BYPASS_DISABLED};
std::atomic<uint64_t> counts[4]{};
std::atomic<uint64_t> processed_count{0};
std::atomic<double> last_process_ms{0.0}, average_process_ms{0.0};
std::array<double,32> process_samples{};
size_t process_sample_count=0, process_sample_cursor=0;
double process_sample_sum=0.0;
thread_local bool inside = false;
#ifdef NRB_XESS_IDENTITY_TEST
std::atomic<unsigned> identity_relay_logs{0};
#endif
std::mutex log_mutex, queue_mutex, hook_mutex;
nrb::DiagnosticGate diagnostic_gate;
void log_line(const char* message);
nrb::PythonProcessor python_processor{log_line};
bool runtime_assets_available=false;
bool built_in_processor=false;
bool processor_failed=false;
using CreateDevice = HRESULT (WINAPI *)(IUnknown*,D3D_FEATURE_LEVEL,REFIID,void**);
using CreateQueue = HRESULT (STDMETHODCALLTYPE *)(ID3D12Device*,const D3D12_COMMAND_QUEUE_DESC*,REFIID,void**);
using ExecuteLists = void (STDMETHODCALLTYPE *)(ID3D12CommandQueue*,UINT,ID3D12CommandList* const*);
CreateDevice original_create_device=nullptr;
CreateDevice original_create_sl=nullptr;
CreateDevice original_create_host=nullptr;
std::array<CreateDevice,3> hooked_device_exports{};
CreateQueue original_create_queue=nullptr;
ExecuteLists original_execute=nullptr;
#ifdef NRB_DEFERRED_IDENTITY_TEST
std::recursive_mutex execute_serial_mutex;
#endif
struct QueueRecord { ID3D12CommandQueue* object; ID3D12Device* device; uint32_t ordinal; uint64_t submissions; };
struct ListRecord { ID3D12CommandList* object=nullptr; uint32_t queue_ordinal=0; uint64_t serial=0; };
std::vector<QueueRecord> queues;
std::array<ListRecord,128> recent_lists{};
uint64_t submit_serial=0;
uint64_t submit_batch=0;
#ifdef NRB_DLSS_COLOR_STATE_PROOF_V1
// Recorded legacy/direct scope only; not GPU execution-state proof.
// Enable the exact reviewed Cyberpunk profile without a special launcher.
nrb::ColorStateScopeMode select_color_state_scope() noexcept {
    char value[64]{};
    SetLastError(ERROR_SUCCESS);
    const auto size=GetEnvironmentVariableA("NRB_DLSS_COLOR_STATE_PROOF_SCOPE",value,sizeof(value));
    const auto error=GetLastError();
    const bool override_present=size!=0 || error!=ERROR_ENVVAR_NOT_FOUND;
    if(size>=sizeof(value)) return nrb::ColorStateScopeMode::disabled;
    wchar_t executable[1024]{};
    const auto length=GetModuleFileNameW(nullptr,executable,1024);
    const std::wstring_view path=length>0 && length<1024?
        std::wstring_view(executable,length):std::wstring_view{};
    return nrb::dlss_color_select_scope(override_present,
        std::string_view(value,size),path);
}
const auto color_state_scope_mode=select_color_state_scope();
bool color_state_scope_enabled() noexcept {
    return color_state_scope_mode!=nrb::ColorStateScopeMode::disabled;
}
const bool color_state_shadow_mode=color_state_scope_mode==nrb::ColorStateScopeMode::shadow;
nrb::ProofAcquire acquire_color_identity(void*,void* object,bool resource,nrb::ProofIdentity& out) noexcept {
    if(!object) return nrb::ProofAcquire::unknown;
    IUnknown* identity=nullptr;
    if(resource) {
        auto* r=static_cast<ID3D12Resource*>(object);
        if(!nrb::dlss_color_candidate(r->GetDesc())) return nrb::ProofAcquire::not_candidate;
        if(FAILED(r->QueryInterface(IID_PPV_ARGS(&identity)))) return nrb::ProofAcquire::unknown;
    } else {
        auto* list=static_cast<ID3D12CommandList*>(object);
        if(list->GetType()!=D3D12_COMMAND_LIST_TYPE_DIRECT ||
           FAILED(list->QueryInterface(IID_PPV_ARGS(&identity)))) return nrb::ProofAcquire::unknown;
    }
    if(!identity) return nrb::ProofAcquire::unknown;
    out.cookie=reinterpret_cast<uintptr_t>(identity);out.lease=identity;
    return nrb::ProofAcquire::acquired;
}
void release_color_identity(void*,nrb::ProofIdentity& identity) noexcept {
    if(identity.lease) static_cast<IUnknown*>(identity.lease)->Release();
    identity={};
}
nrb::SyncProbe sync_probe({nullptr,acquire_color_identity,release_color_identity},color_state_scope_enabled());
#else
nrb::SyncProbe sync_probe;
#endif
std::atomic<uint32_t> sync_eval_logs{0},sync_submit_logs{0};
using ListReset=HRESULT (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList*,ID3D12CommandAllocator*,ID3D12PipelineState*);
using ListClose=HRESULT (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList*);
using ListBarriers=void (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList*,UINT,const D3D12_RESOURCE_BARRIER*);
#ifdef NRB_TAIL_ORDER_PROBE
using ListDraw=void (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList*,UINT,UINT,UINT,UINT);
using ListDrawIndexed=void (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList*,UINT,UINT,UINT,INT,UINT);
using ListDispatch=void (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList*,UINT,UINT,UINT);
using ListCopyTexture=void (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList*,
    const D3D12_TEXTURE_COPY_LOCATION*,UINT,UINT,UINT,
    const D3D12_TEXTURE_COPY_LOCATION*,const D3D12_BOX*);
using ListCopyResource=void (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList*,
    ID3D12Resource*,ID3D12Resource*);
ListDraw original_list_draw=nullptr;
ListDrawIndexed original_list_draw_indexed=nullptr;
ListDispatch original_list_dispatch=nullptr;
ListCopyTexture original_list_copy_texture=nullptr;
ListCopyResource original_list_copy_resource=nullptr;
#endif
ListReset original_list_reset=nullptr;
ListClose original_list_close=nullptr;
ListBarriers original_list_barriers=nullptr;
struct ListMethods {ListReset reset=nullptr;ListClose close=nullptr;ListBarriers barriers=nullptr;
#ifdef NRB_TAIL_ORDER_PROBE
    ListDraw draw=nullptr;ListDrawIndexed draw_indexed=nullptr;ListDispatch dispatch=nullptr;
    ListCopyTexture copy_texture=nullptr;ListCopyResource copy_resource=nullptr;
#endif
};
ListMethods first_list_methods{};
std::atomic<int> list_hook_state{0}; // 0 absent, 1 installing, 2 installed, 3 failed
ID3D12CommandList* consumer_list=nullptr;
uint64_t consumer_submit_before=0;
uint32_t consumer_queue_ordinal=0;

bool attach_detour(void** target,void* replacement) {
    LONG rc=DetourTransactionBegin();
    if(rc==NO_ERROR) rc=DetourUpdateThread(GetCurrentThread());
    if(rc==NO_ERROR) rc=DetourAttach(target,replacement);
    if(rc==NO_ERROR) rc=DetourTransactionCommit();
    else DetourTransactionAbort();
    return rc==NO_ERROR;
}
int32_t __cdecl observed_xess_execute(void* context,ID3D12GraphicsCommandList* list,
                                      const void* params) {
    static std::atomic<unsigned> calls{0};
    if(calls.fetch_add(1,std::memory_order_relaxed)<3)
        log_line("native XeSS execute entered\r\n");
    const auto previous=in_upscaler_api;
    in_upscaler_api=NRB_ROUTE_XESS;
    const auto result=original_xess_execute(context,list,params);
    in_upscaler_api=previous;
    return result;
}
int32_t __cdecl observed_fsr3_dispatch(void* context,const void* params) {
    static std::atomic<unsigned> calls{0};
    const unsigned call=calls.fetch_add(1,std::memory_order_relaxed);
    if(call<3)
        log_line("native FSR3 dispatch entered\r\n");
    // Read only the public FFX descriptor. OptiScaler may pass this dispatch
    // straight through when its D3D12 device discovery fails, so an NGX
    // evaluate hook alone cannot describe the FSR source in that case.
    if(params && (call<3 || (call<=600 && call%120==0))) {
        const auto* frame=static_cast<const Fsr3DispatchProbe*>(params);
        const auto& color=frame->resources[0];
        const auto& depth=frame->resources[1];
        const auto& motion=frame->resources[2];
        const auto& output=frame->resources[9];
        char line[640];
        sprintf_s(line,
            "fsr3_source call=%u context=%p list=%p render=[%u,%u] "
            "color=[%p,%u,%u,%u,%u] motion=[%p,%u,%u,%u,%u] "
            "depth=[%p,%u,%u,%u,%u] output=[%p,%u,%u,%u,%u] "
            "mv_scale=[%.3f,%.3f] jitter=[%.3f,%.3f]\r\n",
            call,context,frame->command_list,frame->render_width,
            frame->render_height,color.resource,color.width,color.height,
            color.format,color.state,motion.resource,motion.width,motion.height,
            motion.format,motion.state,depth.resource,depth.width,depth.height,
            depth.format,depth.state,output.resource,output.width,output.height,
            output.format,output.state,frame->motion_scale_x,frame->motion_scale_y,
            frame->jitter_x,frame->jitter_y);
        log_line(line);
    }
    const auto* frame=static_cast<const Fsr3DispatchProbe*>(params);
#ifdef NRB_TAIL_ORDER_PROBE
    ID3D12GraphicsCommandList* observed_list=nullptr;
    if(frame && frame->command_list && frame->render_width==1280 &&
       frame->render_height==720 && frame->resources[0].resource &&
       frame->resources[0].format==13 && frame->resources[0].width==1280 &&
       frame->resources[0].height==720 && frame->resources[9].resource &&
       frame->resources[9].width==2560 && frame->resources[9].height==1440) {
        observed_list=static_cast<ID3D12GraphicsCommandList*>(frame->command_list);
        schedule_list_hooks(observed_list);
        nrb::tail_probe_rearm(NRB_ROUTE_FSR);
        nrb::tail_before_evaluate(observed_list,
            static_cast<ID3D12Resource*>(frame->resources[0].resource),
            static_cast<ID3D12Resource*>(frame->resources[9].resource),
            NRB_ROUTE_FSR);
    }
#endif
    const auto previous=in_upscaler_api;
    in_upscaler_api=NRB_ROUTE_FSR;
    const auto result=original_fsr3_dispatch(context,params);
    in_upscaler_api=previous;
#ifdef NRB_TAIL_ORDER_PROBE
    if(observed_list) nrb::tail_after_evaluate(observed_list);
#endif
    return result;
}
DWORD WINAPI install_upscaler_api_probes(void*) {
    // The game loads these modules when its menu selects the corresponding
    // renderer. Hook them off the render thread and leave the signed Opti DLL
    // unchanged. A missing/failed hook leaves that route fail-closed.
    for(unsigned tick=0;tick<600 && (!original_xess_execute || !original_fsr3_dispatch);++tick) {
        if(!original_xess_execute) if(auto module=GetModuleHandleW(L"libxess.dll")) {
            auto target=reinterpret_cast<XeSSExecute>(GetProcAddress(module,"xessD3D12Execute"));
            if(target) {
                original_xess_execute=target;
                if(attach_detour(reinterpret_cast<void**>(&original_xess_execute),
                                 reinterpret_cast<void*>(observed_xess_execute)))
                    log_line("XeSS public execute route tag attached\r\n");
                else {original_xess_execute=nullptr;
                    log_line("XeSS route tag unavailable; XeSS NR remains bypassed\r\n");}
            }
        }
        if(!original_fsr3_dispatch) if(auto module=GetModuleHandleW(L"ffx_fsr3upscaler_x64.dll")) {
            auto target=reinterpret_cast<FSR3Dispatch>(GetProcAddress(module,
                "ffxFsr3UpscalerContextDispatch"));
            if(target) {
                original_fsr3_dispatch=target;
                if(attach_detour(reinterpret_cast<void**>(&original_fsr3_dispatch),
                                 reinterpret_cast<void*>(observed_fsr3_dispatch)))
                    log_line("FSR3 public dispatch route tag attached\r\n");
                else {original_fsr3_dispatch=nullptr;
                    log_line("FSR3 route tag unavailable; FSR NR remains bypassed\r\n");}
            }
        }
        Sleep(1000);
    }
    return 0;
}
int32_t __cdecl observed_sl_evaluate(uint32_t feature,const void* frame,
    const void** inputs,uint32_t count,void* command_buffer) {
    const bool previous=in_sl_dlss_evaluate;
    if(feature==0 && command_buffer) in_sl_dlss_evaluate=true;
    const int32_t result=original_sl_evaluate(feature,frame,inputs,count,command_buffer);
    in_sl_dlss_evaluate=previous;
    return result;
}
void install_sl_feature_probe() {
    HMODULE sl=GetModuleHandleW(L"sl.interposer.dll");
    if(!sl) {log_line("Streamline feature entry absent; external route stays unknown\r\n");return;}
    original_sl_evaluate=reinterpret_cast<SLEvaluate>(
        GetProcAddress(sl,"slEvaluateFeature"));
    if(!original_sl_evaluate || !attach_detour(
            reinterpret_cast<void**>(&original_sl_evaluate),
            reinterpret_cast<void*>(observed_sl_evaluate))) {
        original_sl_evaluate=nullptr;
        log_line("Streamline feature hook unavailable; external route stays unknown\r\n");
        return;
    }
    log_line("Streamline slEvaluateFeature feature tag attached\r\n");
}

void STDMETHODCALLTYPE observed_execute(ID3D12CommandQueue* q,UINT count,ID3D12CommandList* const* lists) {
#ifdef NRB_DEFERRED_IDENTITY_TEST
    // Keep other intercepted ExecuteCommandLists calls from interleaving with
    // the prefix/private/suffix submissions on this diagnostic queue path.
    std::lock_guard execute_guard(execute_serial_mutex);
#endif
    uint64_t batch=0;
    std::array<nrb::SubmitObservation,64> correlated{};
    {
        std::lock_guard guard(queue_mutex);
        for(auto& entry:queues) if(entry.object==q) {
            ++entry.submissions;
            batch=++submit_batch;
            for(UINT i=0;i<count && lists;i++) {
                const uint64_t serial=++submit_serial;
                recent_lists[serial & (recent_lists.size()-1)]={lists[i],entry.ordinal,serial};
            }
            break;
        }
    }
    // Diagnostic correlation only. No queue pointer or submit observation
    // certifies that this frame's resource writes completed before Evaluate.
    for(UINT i=0;i<count && lists && i<correlated.size() && batch;i++)
        correlated[i]=sync_probe.submit(lists[i],batch);
    // Bind the list generation at call entry: another thread may Reset the
    // list immediately after ExecuteCommandLists returns. This is still only
    // an API-call observation, never a GPU-completion fence.
#ifdef NRB_DEFERRED_IDENTITY_TEST
    NRB_Controls deferred_controls{};
    NRB_Processor deferred_processor{};
    {std::lock_guard guard(state_mutex);
        deferred_controls=controls;deferred_processor=processor;}
    const unsigned deferred=nrb::submit_deferred_identity(q,count,lists,
        count<=correlated.size()?correlated.data():nullptr,original_execute,
        &deferred_processor,&deferred_controls,log_line);
    if(!deferred) original_execute(q,count,lists);
    else {
        static std::atomic<unsigned> deferred_logs{0};
        if(deferred_logs.fetch_add(1,std::memory_order_relaxed)<8) {
            char line[320];
            sprintf_s(line,"deferred_identity_submitted batch_count=%u inserted=%u recorded=%u retired=%u signal_failures=%u generation_mismatches=%u invalidated=%u hdr_ready=%u nr_frames=%llu nr_failed=%u nr_last_ms=%.2f\r\n",
                count,deferred,nrb::deferred_identity_recorded(),
                nrb::deferred_identity_retired(),
                nrb::deferred_identity_signal_failures(),
                nrb::deferred_identity_generation_mismatches(),
                nrb::deferred_identity_invalidated(),nrb::deferred_hdr_ready(),
                static_cast<unsigned long long>(nrb::deferred_nr_processed()),
                unsigned(nrb::deferred_nr_failed()),nrb::deferred_nr_last_ms());
            log_line(line);
        }
    }
#else
    original_execute(q,count,lists);
#endif
#ifdef NRB_XESS_IDENTITY_TEST
    const unsigned relayed=nrb::xess_identity_submitted(q,count,lists);
    if(relayed) {
        static std::atomic<unsigned> submit_logs{0};
        if(submit_logs.fetch_add(1,std::memory_order_relaxed)<8) {
            char line[160];
            sprintf_s(line,"ngx_identity_submitted count=%u retired=%u\r\n",
                relayed,nrb::xess_identity_completed());
            log_line(line);
        }
    }
#endif
#ifdef NRB_ONE_FRAME_COLOR_READBACK
    nrb::one_frame_readback_submitted(q,count,lists,correlated.data(),
                                     static_cast<UINT>(correlated.size()),log_line);
#endif
    for(UINT i=0;i<count && lists && i<correlated.size();i++) {
        const auto& observation=correlated[i];
        if(!observation.correlated || sync_submit_logs.fetch_add(1)>=16) continue;
        char line[320];
        sprintf_s(line,"{\"event\":\"execute_lists_call\",\"list\":\"%p\",\"generation\":%llu,\"batch\":%llu,\"batch_count\":%u,\"batch_index\":%u,\"evaluations\":%u,\"latest_evaluate_sequence\":%llu,\"proof\":\"unknown\"}\r\n",
            static_cast<void*>(lists[i]),observation.generation,observation.batch,
            count,i,observation.evaluations,observation.latest_evaluate_sequence);
        log_line(line);
    }
}
HRESULT STDMETHODCALLTYPE observed_list_reset(ID3D12GraphicsCommandList* list,
    ID3D12CommandAllocator* allocator,ID3D12PipelineState* state) {
    const HRESULT hr=original_list_reset(list,allocator,state);
    sync_probe.reset(list,SUCCEEDED(hr));
#ifdef NRB_TAIL_ORDER_PROBE
    if(SUCCEEDED(hr)) nrb::tail_reset(list);
#endif
#ifdef NRB_DEFERRED_IDENTITY_TEST
    if(SUCCEEDED(hr)) nrb::deferred_identity_list_reset(list);
#endif
    return hr;
}
HRESULT STDMETHODCALLTYPE observed_list_close(ID3D12GraphicsCommandList* list) {
    const HRESULT hr=original_list_close(list);
    sync_probe.close(list,SUCCEEDED(hr));
#ifdef NRB_TAIL_ORDER_PROBE
    if(SUCCEEDED(hr)) {
        const auto s=nrb::tail_close(list);
        if(s.valid) {
            char line[800];
            sprintf_s(line,"{\"event\":\"same_list_tail\",\"route\":%u,\"pre_actions\":%u,\"in_xess_actions\":%u,\"after_actions\":%u,\"after_dispatches\":%u,\"after_draws\":%u,\"last_dispatch\":[%u,%u,%u],\"after_color_barriers\":%u,\"after_output_barriers\":%u,\"output_read_barriers\":%u,\"actions_after_output_read\":%u,\"after_output_copies\":%u,\"color_state\":[%u,%u],\"output_state\":[%u,%u],\"color_flags\":%u,\"output_flags\":%u,\"color_subresource\":%u,\"output_subresource\":%u}\r\n",
                unsigned(s.route),s.pre_actions,s.in_evaluate_actions,s.after_actions,
                s.after_dispatches,s.after_draws,
                s.last_dispatch_x,s.last_dispatch_y,s.last_dispatch_z,
                s.after_color_barriers,s.after_output_barriers,
                s.output_read_barriers,s.actions_after_output_read,
                s.after_output_copies,s.color_before,s.color_after,
                s.output_before,s.output_after,s.color_flags,s.output_flags,
                s.color_subresource,s.output_subresource);
            log_line(line);
            if(s.route==NRB_ROUTE_XESS) {
                char copy_line[220];
                sprintf_s(copy_line,
                    "xess_tail_copy safe=%u from_color=%u to_color=%u to_output=%u src=[%u,%u,%u] dst=[%u,%u,%u]\r\n",
                    unsigned(nrb::tail_safe_for_deferred_identity(s)),
                    s.after_copy_from_color,s.after_copy_to_color,
                    s.after_copy_to_output,s.copy_source_format,
                    s.copy_source_width,s.copy_source_height,
                    s.copy_dest_format,s.copy_dest_width,s.copy_dest_height);
                log_line(copy_line);
            }
#ifdef NRB_DEFERRED_IDENTITY_TEST
            nrb::observe_deferred_identity_tail(s);
#endif
        }
    }
#endif
    return hr;
}
void STDMETHODCALLTYPE observed_list_barriers(ID3D12GraphicsCommandList* list,UINT count,
                                               const D3D12_RESOURCE_BARRIER* barriers) {
    original_list_barriers(list,count,barriers);
    sync_probe.barriers(list,count,barriers);
#ifdef NRB_TAIL_ORDER_PROBE
    nrb::tail_barriers(list,count,barriers);
#endif
}
#ifdef NRB_TAIL_ORDER_PROBE
void STDMETHODCALLTYPE observed_list_draw(ID3D12GraphicsCommandList* list,
    UINT vertices,UINT instances,UINT first_vertex,UINT first_instance) {
    original_list_draw(list,vertices,instances,first_vertex,first_instance);
    nrb::tail_action(list);
}
void STDMETHODCALLTYPE observed_list_draw_indexed(ID3D12GraphicsCommandList* list,
    UINT indices,UINT instances,UINT first_index,INT base_vertex,UINT first_instance) {
    original_list_draw_indexed(list,indices,instances,first_index,base_vertex,first_instance);
    nrb::tail_action(list);
}
void STDMETHODCALLTYPE observed_list_dispatch(ID3D12GraphicsCommandList* list,
    UINT x,UINT y,UINT z) {
    original_list_dispatch(list,x,y,z);
    nrb::tail_action(list,true,x,y,z);
}
void STDMETHODCALLTYPE observed_list_copy_texture(ID3D12GraphicsCommandList* list,
    const D3D12_TEXTURE_COPY_LOCATION* dest,UINT x,UINT y,UINT z,
    const D3D12_TEXTURE_COPY_LOCATION* source,const D3D12_BOX* box) {
    original_list_copy_texture(list,dest,x,y,z,source,box);
    nrb::tail_copy(list,source?source->pResource:nullptr,
        dest?dest->pResource:nullptr,true);
}
void STDMETHODCALLTYPE observed_list_copy_resource(ID3D12GraphicsCommandList* list,
    ID3D12Resource* dest,ID3D12Resource* source) {
    original_list_copy_resource(list,dest,source);
    nrb::tail_copy(list,source,dest,true);
}
#endif
DWORD WINAPI install_list_hooks_worker(void*) {
    std::lock_guard guard(hook_mutex);
    original_list_reset=first_list_methods.reset;
    original_list_close=first_list_methods.close;
    original_list_barriers=first_list_methods.barriers;
#ifdef NRB_TAIL_ORDER_PROBE
    original_list_draw=first_list_methods.draw;
    original_list_draw_indexed=first_list_methods.draw_indexed;
    original_list_dispatch=first_list_methods.dispatch;
    original_list_copy_texture=first_list_methods.copy_texture;
    original_list_copy_resource=first_list_methods.copy_resource;
#endif
    LONG rc=DetourTransactionBegin();
    if(rc==NO_ERROR) rc=DetourUpdateThread(GetCurrentThread());
    if(rc==NO_ERROR) rc=DetourAttach(reinterpret_cast<void**>(&original_list_reset),
                                     reinterpret_cast<void*>(observed_list_reset));
    if(rc==NO_ERROR) rc=DetourAttach(reinterpret_cast<void**>(&original_list_close),
                                     reinterpret_cast<void*>(observed_list_close));
    if(rc==NO_ERROR) rc=DetourAttach(reinterpret_cast<void**>(&original_list_barriers),
                                     reinterpret_cast<void*>(observed_list_barriers));
#ifdef NRB_TAIL_ORDER_PROBE
    if(rc==NO_ERROR) rc=DetourAttach(reinterpret_cast<void**>(&original_list_draw),
                                     reinterpret_cast<void*>(observed_list_draw));
    if(rc==NO_ERROR) rc=DetourAttach(reinterpret_cast<void**>(&original_list_draw_indexed),
                                     reinterpret_cast<void*>(observed_list_draw_indexed));
    if(rc==NO_ERROR) rc=DetourAttach(reinterpret_cast<void**>(&original_list_dispatch),
                                     reinterpret_cast<void*>(observed_list_dispatch));
    if(rc==NO_ERROR) rc=DetourAttach(reinterpret_cast<void**>(&original_list_copy_texture),
                                     reinterpret_cast<void*>(observed_list_copy_texture));
    if(rc==NO_ERROR) rc=DetourAttach(reinterpret_cast<void**>(&original_list_copy_resource),
                                     reinterpret_cast<void*>(observed_list_copy_resource));
#endif
    if(rc==NO_ERROR) rc=DetourTransactionCommit();
    else DetourTransactionAbort();
    list_hook_state.store(rc==NO_ERROR?2:3,std::memory_order_release);
    log_line(rc==NO_ERROR?"D3D12 list Reset/Close/ResourceBarrier metadata hooks installed\r\n":
                          "D3D12 list metadata hook installation failed; sync unknown\r\n");
    return 0;
}
void schedule_list_hooks(ID3D12GraphicsCommandList* list) {
    int expected=0;
    if(!list || !list_hook_state.compare_exchange_strong(expected,1)) return;
    auto** methods=*reinterpret_cast<void***>(list);
    first_list_methods.reset=reinterpret_cast<ListReset>(methods[10]);
    first_list_methods.close=reinterpret_cast<ListClose>(methods[9]);
    first_list_methods.barriers=reinterpret_cast<ListBarriers>(methods[26]);
#ifdef NRB_TAIL_ORDER_PROBE
    first_list_methods.draw=reinterpret_cast<ListDraw>(methods[12]);
    first_list_methods.draw_indexed=reinterpret_cast<ListDrawIndexed>(methods[13]);
    first_list_methods.dispatch=reinterpret_cast<ListDispatch>(methods[14]);
    first_list_methods.copy_texture=reinterpret_cast<ListCopyTexture>(methods[16]);
    first_list_methods.copy_resource=reinterpret_cast<ListCopyResource>(methods[17]);
#endif
    HANDLE worker=CreateThread(nullptr,0,install_list_hooks_worker,nullptr,0,nullptr);
    if(worker) CloseHandle(worker);
    else {list_hook_state.store(3,std::memory_order_release);
          log_line("D3D12 list metadata hook worker failed; sync unknown\r\n");}
}
void observe_queue(ID3D12CommandQueue* q) {
    if(!q || q->GetDesc().Type!=D3D12_COMMAND_LIST_TYPE_DIRECT) return;
    ID3D12Device* d=nullptr;
    if(FAILED(q->GetDevice(IID_PPV_ARGS(&d)))) return;
    {
        std::lock_guard guard(queue_mutex);
        for(auto& e:queues) if(e.object==q) {d->Release();return;}
        q->AddRef(); queues.push_back({q,d,static_cast<uint32_t>(queues.size()+1),0});
    }
    std::lock_guard hook_guard(hook_mutex);
    if(!original_execute) {
        auto** methods=*reinterpret_cast<void***>(q);
        original_execute=reinterpret_cast<ExecuteLists>(methods[10]);
        if(!attach_detour(reinterpret_cast<void**>(&original_execute),
                          reinterpret_cast<void*>(observed_execute))) {
            original_execute=nullptr; log_line("queue ExecuteCommandLists detour failed\r\n");
        }
    }
}
HRESULT STDMETHODCALLTYPE observed_create_queue(ID3D12Device* d,const D3D12_COMMAND_QUEUE_DESC* desc,
                                                 REFIID iid,void** output) {
    const HRESULT hr=original_create_queue(d,desc,iid,output);
    if(SUCCEEDED(hr) && output && *output && iid==__uuidof(ID3D12CommandQueue))
        observe_queue(static_cast<ID3D12CommandQueue*>(*output));
    return hr;
}
void observe_device(ID3D12Device* d) {
    if(!d) return;
    std::lock_guard hook_guard(hook_mutex);
    if(original_create_queue) return;
    auto** methods=*reinterpret_cast<void***>(d);
    original_create_queue=reinterpret_cast<CreateQueue>(methods[8]);
    if(!attach_detour(reinterpret_cast<void**>(&original_create_queue),
                      reinterpret_cast<void*>(observed_create_queue))) {
        original_create_queue=nullptr; log_line("device CreateCommandQueue detour failed\r\n");
    }
}
HRESULT record_created_device(HRESULT hr,void** output) {
    if(SUCCEEDED(hr) && output && *output) {
        ID3D12Device* d=nullptr;
        if(SUCCEEDED(static_cast<IUnknown*>(*output)->QueryInterface(IID_PPV_ARGS(&d)))) {
            observe_device(d); d->Release();
        }
    }
    return hr;
}
HRESULT WINAPI observed_create_device(IUnknown* adapter,D3D_FEATURE_LEVEL level,REFIID iid,void** output) {
    return record_created_device(original_create_device(adapter,level,iid,output),output);
}
HRESULT WINAPI observed_create_sl(IUnknown* adapter,D3D_FEATURE_LEVEL level,REFIID iid,void** output) {
    return record_created_device(original_create_sl(adapter,level,iid,output),output);
}
HRESULT WINAPI observed_create_host(IUnknown* adapter,D3D_FEATURE_LEVEL level,REFIID iid,void** output) {
    return record_created_device(original_create_host(adapter,level,iid,output),output);
}
void hook_device_export(HMODULE module,CreateDevice& slot,CreateDevice replacement) {
    if(!module) return;
    auto address=reinterpret_cast<CreateDevice>(GetProcAddress(module,"D3D12CreateDevice"));
    if(!address) return;
    for(auto installed:hooked_device_exports) if(installed==address) return;
    slot=address;
    if(!attach_detour(reinterpret_cast<void**>(&slot),reinterpret_cast<void*>(replacement))) {
        slot=nullptr;log_line("D3D12CreateDevice export detour failed\r\n");
    } else for(auto& installed:hooked_device_exports) if(!installed) {installed=address;break;}
}
void install_queue_capture() {
    hook_device_export(GetModuleHandleW(L"d3d12.dll"),original_create_device,observed_create_device);
    hook_device_export(GetModuleHandleW(L"sl.interposer.dll"),original_create_sl,observed_create_sl);
    if(!original_create_device && !original_create_sl)
        log_line("D3D12 queue capture: no device factory export loaded at ASI init\r\n");
}
struct QueueSnapshot { nrb::QueueMeta meta; ID3D12CommandQueue* selected=nullptr; uint64_t serial=0; };
QueueSnapshot queue_snapshot(ID3D12Device* d,ID3D12CommandList* list) {
    QueueSnapshot result{};
    std::lock_guard guard(queue_mutex);
    uint32_t matching=0;
    for(auto& e:queues) if(e.device==d) {
        ++matching; result.selected=e.object;
        result.meta.ordinal=e.ordinal;
        result.meta.type=static_cast<uint32_t>(e.object->GetDesc().Type);
        result.meta.submissions=e.submissions;
    }
    result.meta.candidate_count=matching;
    if(matching!=1) {result.selected=nullptr;result.meta.ordinal=0;}
    result.meta.same_device=matching>0;
    for(auto& item:recent_lists) if(item.object==list) {
        result.meta.consumer_list_previously_submitted=true;
        if(item.serial>result.serial) {
            result.serial=item.serial;
            result.meta.last_seen_consumer_queue_ordinal=item.queue_ordinal;
            result.meta.last_seen_consumer_submit_serial=item.serial;
        }
    }
    result.meta.producer_status=matching==0?"no_direct_queue_observed":
        matching>1?"ambiguous_direct_queues":"unproven_current_frame_resource_write";
    if(result.selected) result.selected->AddRef();
    return result;
}
bool consumer_submitted_after(ID3D12CommandList* list,uint64_t prior,uint32_t ordinal) {
    std::lock_guard guard(queue_mutex);
    for(auto& item:recent_lists) if(item.object==list && item.serial>prior &&
        item.queue_ordinal==ordinal) return true;
    return false;
}

void log_line(const char* message) {
    std::lock_guard guard(log_mutex);
    HMODULE self = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&log_line), &self)) return;
    wchar_t path[MAX_PATH]{};
    DWORD size = GetModuleFileNameW(self, path, MAX_PATH);
    if (!size || size >= MAX_PATH) return;
    wchar_t* slash = wcsrchr(path, L'\\');
    if (!slash) return;
    wcscpy_s(slash + 1, MAX_PATH - (slash + 1 - path), L"CyberpunkNRBridge.log");
    HANDLE file = CreateFileW(path, FILE_APPEND_DATA,
        FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return;
    DWORD written = 0;
    WriteFile(file, message, static_cast<DWORD>(strlen(message)), &written, nullptr);
    CloseHandle(file);
}

bool sync_queue() {
    // A failed retire has already poisoned the processor. Keep the borrowed
    // resource pinned and avoid retrying/logging the same Python error every
    // frame; recovery requires a fresh process with a known GPU state.
    if (processor_failed) return false;
    if (!queue) return false;
    if (borrowed.color && (!consumer_list || !consumer_queue_ordinal ||
        !consumer_submitted_after(consumer_list,consumer_submit_before,consumer_queue_ordinal)))
        return false;
    if (!completion) {
        ID3D12Device* d = nullptr;
        if (FAILED(queue->GetDevice(IID_PPV_ARGS(&d)))) return false;
        const HRESULT hr = d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&completion));
        d->Release();
        if (FAILED(hr)) return false;
        completion_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!completion_event) return false;
    }
    if (FAILED(queue->Signal(completion, ++completion_value)) ||
        FAILED(completion->SetEventOnCompletion(completion_value, completion_event)) ||
        WaitForSingleObject(completion_event, 10000) != WAIT_OBJECT_0) return false;
    if (borrowed.color) {
        try {
            if (!processor.retire || processor.retire(processor.context,completion,completion_value)!=0) {
                processor_failed=true;
                return false;
            }
        } catch (...) { processor_failed=true; return false; }
        borrowed.color->Release();
        borrowed.ready_fence->Release();
        borrowed = {};
        consumer_list=nullptr;
        consumer_queue_ordinal=0;
    }
    return true;
}

struct RouteDecision { NRB_Route route; NRB_RouteEvidence evidence; };
RouteDecision classify(void* caller, NVSDK_NGX_Parameter* params) {
    if(in_upscaler_api==NRB_ROUTE_XESS)
        return {NRB_ROUTE_XESS,NRB_EVIDENCE_XESS_API};
    if(in_upscaler_api==NRB_ROUTE_FSR)
        return {NRB_ROUTE_FSR,NRB_EVIDENCE_FSR3_API};
    if(in_sl_dlss_evaluate) {
        static std::atomic<bool> logged{false};
        if(!logged.exchange(true,std::memory_order_relaxed))
            log_line("NGX Evaluate nested in Streamline kFeatureDLSS=0\r\n");
        return {NRB_ROUTE_DLSS,NRB_EVIDENCE_STREAMLINE_FEATURE};
    }
    HMODULE caller_module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(caller), &caller_module))
        return {NRB_ROUTE_UNKNOWN,NRB_EVIDENCE_NONE};
    if (caller_module == GetModuleHandleW(nullptr))
        return {NRB_ROUTE_DLSS,NRB_EVIDENCE_GAME_NGX};
    HMODULE streamline=GetModuleHandleW(L"sl.interposer.dll");
    if (streamline && caller_module == streamline)
        return {NRB_ROUTE_DLSS,NRB_EVIDENCE_STREAMLINE_NGX};
    if (caller_module != host)
        return {NRB_ROUTE_UNKNOWN,NRB_EVIDENCE_EXTERNAL_UNTAGGED};
    float fsr_time = 0;
    if (params && params->Get("FSR.frameTimeDelta", &fsr_time) == NVSDK_NGX_Result_Success)
        return {NRB_ROUTE_FSR,NRB_EVIDENCE_FSR_MARKER};
    int xess_marker=0;
    if (params && (params->Get("XeSS.ExposureScaleTexture",&xess_marker)==NVSDK_NGX_Result_Success ||
                   params->Get("XeSS.ResponsivePixelMask",&xess_marker)==NVSDK_NGX_Result_Success))
        return {NRB_ROUTE_XESS,NRB_EVIDENCE_XESS_MARKER};
    // Internal source without an explicit marker may be FSR, XeSS, or another
    // proxy. Do not infer XeSS by exclusion across different games.
    return {NRB_ROUTE_UNKNOWN,NRB_EVIDENCE_INTERNAL_UNTAGGED};
}

NVSDK_NGX_Result __cdecl intercept(ID3D12GraphicsCommandList* list,
    const NVSDK_NGX_Handle* handle, NVSDK_NGX_Parameter* params,
    PFN_NVSDK_NGX_ProgressCallback callback) {
    if (!original) return NVSDK_NGX_Result_Fail;
    if (inside || !list || !handle || !params) return original(list, handle, params, callback);
    void* caller = _ReturnAddress();
    inside = true;
    const RouteDecision decision = classify(caller, params);
    const NRB_Route route = decision.route;
    if(route==NRB_ROUTE_FSR || route==NRB_ROUTE_XESS) {
        static std::atomic<unsigned> route_logs[4]{};
        if(route_logs[route].fetch_add(1,std::memory_order_relaxed)<8) {
            char line[180];
            sprintf_s(line,"upscaler_entry route=%u evidence=%u handle=%llu\r\n",
                unsigned(route),unsigned(decision.evidence),
                static_cast<unsigned long long>(handle->Id));
            log_line(line);
        }
    }
#if defined(NRB_XESS_IDENTITY_TEST) || defined(NRB_TAIL_ORDER_PROBE)
    HMODULE scope_module=nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(caller),&scope_module);
    // Cyberpunk's Streamline invokes the signed OptiScaler evaluator from
    // sl.common.dll, outside slEvaluateFeature; keep production route UNKNOWN.
    // This diagnostic only gates the byte-identical relay on that caller.
    const bool diagnostic_dlss_scope=route==NRB_ROUTE_DLSS ||
        (decision.evidence==NRB_EVIDENCE_EXTERNAL_UNTAGGED &&
         scope_module && scope_module==GetModuleHandleW(L"sl.common.dll") &&
         GetModuleHandleW(L"Cyberpunk2077.exe") &&
         handle->Id>=1000000 && handle->Id<2000000);
    const bool trusted_other_sr_scope=
        (route==NRB_ROUTE_XESS && decision.evidence==NRB_EVIDENCE_XESS_API) ||
        (route==NRB_ROUTE_FSR && decision.evidence==NRB_EVIDENCE_FSR3_API);
#ifdef NRB_MULTI_ROUTE_LIVE
    const bool diagnostic_other_sr_scope=trusted_other_sr_scope;
#else
    const bool diagnostic_other_sr_scope=false;
#endif
    const bool diagnostic_sr_scope=diagnostic_dlss_scope || diagnostic_other_sr_scope;
#ifdef NRB_TAIL_ORDER_PROBE
    if(diagnostic_dlss_scope || trusted_other_sr_scope) {
        static std::atomic<NRB_Route> last_tail_route{NRB_ROUTE_UNKNOWN};
        const NRB_Route sample_route=diagnostic_dlss_scope?NRB_ROUTE_DLSS:route;
        const auto previous=last_tail_route.exchange(sample_route,std::memory_order_acq_rel);
        if(previous!=sample_route) {
            nrb::tail_probe_rearm(sample_route);
#ifdef NRB_DEFERRED_IDENTITY_TEST
            nrb::reset_deferred_identity_tail(sample_route);
#endif
        }
    }
#endif
#ifdef NRB_XESS_IDENTITY_TEST
    bool identity_source_state_proven=false;
    uint64_t identity_source_generation=0;
#endif
#endif
    nrb::flashdiag::Context flash{};
    flash.eval_id=nrb::flashdiag::recorder.begin();
    flash.list=reinterpret_cast<uintptr_t>(list);flash.feature_id=handle->Id;
    flash.route=unsigned(route);flash.evidence=unsigned(decision.evidence);
#if defined(NRB_XESS_IDENTITY_TEST) || defined(NRB_TAIL_ORDER_PROBE)
    const bool flash_tagged=decision.evidence==NRB_EVIDENCE_XESS_API ||
        decision.evidence==NRB_EVIDENCE_FSR3_API;
    flash.sr_slot=flash_tagged || (handle->Id>=1000000 && handle->Id<2000000);
    flash.eligible=diagnostic_sr_scope && flash.sr_slot;
    flash.effective_route=unsigned(diagnostic_dlss_scope?NRB_ROUTE_DLSS:route);
#endif
    if(flash.eligible) nrb::flashdiag::recorder.count(nrb::flashdiag::Counter::eligible);
    nrb::flashdiag::recorder.observe(flash);
    bool flash_inline_nr=false;
    auto flash_fallback_reason=nrb::flashdiag::Reason::source_state;
    const auto call_original=[&]() {
#ifdef NRB_DEFERRED_IDENTITY_TEST
        NVSDK_NGX_Result deferred_result=NVSDK_NGX_Result_Fail;
        if(route==NRB_ROUTE_XESS && nrb::deferred_identity_tail_verified(route)) {
            static std::atomic<unsigned> native_gate_logs{0};
            if(native_gate_logs.fetch_add(1,std::memory_order_relaxed)<8) {
                char gate_line[180];
                sprintf_s(gate_line,
                    "xess_native_gate scope=%u source=%u generation=%llu\r\n",
                    unsigned(diagnostic_sr_scope),unsigned(identity_source_state_proven),
                    static_cast<unsigned long long>(identity_source_generation));
                log_line(gate_line);
            }
        }
        // The outer Evaluate path already holds state_mutex here. A nested
        // NRB_GetControls call would deadlock the game's render thread.
        if(diagnostic_sr_scope && identity_source_state_proven &&
           nrb::record_deferred_identity(list,handle,params,callback,original,
                                         identity_source_generation,controls,
                                         diagnostic_dlss_scope?NRB_ROUTE_DLSS:route,
                                         diagnostic_dlss_scope?NRB_EVIDENCE_STREAMLINE_NGX:
                                             decision.evidence,
                                         deferred_result,&flash)) {
#ifdef NRB_TAIL_ORDER_PROBE
            nrb::tail_after_evaluate(list);
#endif
            static std::atomic<unsigned> deferred_record_logs{0};
            if(deferred_record_logs.fetch_add(1,std::memory_order_relaxed)<8) {
                char line[160];
                sprintf_s(line,"deferred_identity_recorded count=%u retired=%u\r\n",
                    nrb::deferred_identity_recorded(),
                    nrb::deferred_identity_retired());
                log_line(line);
            }
            return deferred_result;
        }
        if(diagnostic_sr_scope && identity_source_state_proven)
            flash_fallback_reason=flash.record_reason?
                nrb::flashdiag::Reason(flash.record_reason):nrb::flashdiag::Reason::record_rejected;
#endif
#ifdef NRB_XESS_IDENTITY_TEST
        ID3D12Resource* source=nullptr;
        ID3D12Resource* copy=nullptr;
        if(diagnostic_dlss_scope && identity_source_state_proven &&
           params->Get(NVSDK_NGX_Parameter_Color,&source)==NVSDK_NGX_Result_Success && source) {
            copy=nrb::record_ngx_identity(list,source);
            if(copy) params->Set(NVSDK_NGX_Parameter_Color,copy);
        }
        if(diagnostic_dlss_scope &&
           identity_relay_logs.fetch_add(1,std::memory_order_relaxed)<8) {
            char line[256];
            sprintf_s(line,"ngx_identity_relay copied=%d state_proven=%d recorded=%u retired=%u\r\n",
                int(copy!=nullptr),int(identity_source_state_proven),
                nrb::xess_identity_recorded(),nrb::xess_identity_completed());
            log_line(line);
        }
#endif
        flash.stage_bits|=nrb::flashdiag::bit(nrb::flashdiag::Stage::original_called);
        const auto rc=original(list,handle,params,callback);
        if(!flash_inline_nr) {
            const auto why=flash.eligible?flash_fallback_reason:
                nrb::flashdiag::Reason::scope_or_feature;
            nrb::flashdiag::recorder.fallback(flash,why,unsigned(rc));
            nrb::flashdiag::recorder.finish(flash,why);
        }
#ifdef NRB_TAIL_ORDER_PROBE
        nrb::tail_after_evaluate(list);
#endif
#ifdef NRB_XESS_IDENTITY_TEST
        if(copy) params->Set(NVSDK_NGX_Parameter_Color,source);
#endif
        return rc;
    };
    if (decision.evidence==NRB_EVIDENCE_EXTERNAL_UNTAGGED) {
        static std::atomic<bool> logged_external{false};
        if (!logged_external.exchange(true,std::memory_order_relaxed)) {
            HMODULE module=nullptr;
            char path[MAX_PATH]{};
            if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                    static_cast<LPCSTR>(caller), &module) &&
                GetModuleFileNameA(module,path,MAX_PATH)) {
                char line[MAX_PATH+48]{};
                sprintf_s(line,"external NGX caller module=%s\r\n",path);
                log_line(line);
            }
        }
    }
    // OptiScaler's native DLSS and DLSSG IDs are not our XeSS SR slot.
    // FSR/XeSS IDs are independent: only an actual outer API tag can promote
    // one to a candidate. Unknown high IDs stay out of the FG path.
    const bool tagged_other_sr=
        decision.evidence==NRB_EVIDENCE_XESS_API ||
        decision.evidence==NRB_EVIDENCE_FSR3_API;
    if ((!tagged_other_sr && handle->Id < 1000000) ||
        (!tagged_other_sr && handle->Id >= 2000000 &&
         (route==NRB_ROUTE_DLSS || route==NRB_ROUTE_UNKNOWN))) {
        last_status.store(NRB_BYPASS_UNSUPPORTED, std::memory_order_relaxed);
        auto rc = call_original();
        inside = false;
        return rc;
    }
    counts[route].fetch_add(1, std::memory_order_relaxed);
    std::lock_guard guard(state_mutex);
    NRB_Frame frame{};
    nrb::extract_ngx(params, route, list, frame);
    flash.enabled=controls.enabled;flash.history=unsigned(controls.history);flash.controls_known=1;
    flash.game_reset=frame.reset_history!=0;
    // Resource/list ordering is useful even when the external NGX caller has
    // not yet been classified. This probe cannot promote an unknown route or
    // authorize NR; the processor still sees the original fail-closed route.
    if ((route==NRB_ROUTE_DLSS || route==NRB_ROUTE_FSR ||
         route==NRB_ROUTE_XESS || route==NRB_ROUTE_UNKNOWN) &&
        frame.color && frame.motion && frame.output) {
        schedule_list_hooks(list);
#ifdef NRB_DLSS_COLOR_STATE_PROOF_V1
        auto** source_methods=*reinterpret_cast<void***>(list);
        const bool color_implementation_known=list_hook_state.load(std::memory_order_acquire)==2 &&
            source_methods[10]==reinterpret_cast<void*>(first_list_methods.reset) &&
            source_methods[9]==reinterpret_cast<void*>(first_list_methods.close) &&
            source_methods[26]==reinterpret_cast<void*>(first_list_methods.barriers);
        if(diagnostic_dlss_scope && sync_probe.color_state_enabled() && !color_implementation_known)
            sync_probe.unknown_recording(list);
#endif
        auto sync=sync_probe.evaluate(list,frame.color,frame.motion,diagnostic_dlss_scope);
#ifdef NRB_TAIL_ORDER_PROBE
        if(sync.reset_observed && !sync.closed &&
           (diagnostic_dlss_scope || trusted_other_sr_scope))
            nrb::tail_before_evaluate(list,frame.color,frame.output,
                diagnostic_dlss_scope?NRB_ROUTE_DLSS:route);
#endif
#ifdef NRB_XESS_IDENTITY_TEST
#ifdef NRB_DLSS_COLOR_STATE_PROOF_V1
        const bool dlss_source_proven=nrb::dlss_color_processing_source(sync,color_implementation_known,color_state_shadow_mode);
#else
        const bool dlss_source_proven=nrb::legacy_dlss_color_source(sync);
#endif
        // Native routes are deferred until the entire producer list has
        // executed. Their eight exact tail samples, rather than a preceding
        // color barrier on this list, establish the state of the color copy.
        bool native_deferred_source_proven=false;
#ifdef NRB_DEFERRED_IDENTITY_TEST
        native_deferred_source_proven=trusted_other_sr_scope &&
            sync.reset_observed && !sync.closed &&
            nrb::deferred_identity_tail_verified(route);
#endif
        identity_source_state_proven=dlss_source_proven || native_deferred_source_proven;
        identity_source_generation=sync.generation;
#endif
        const int hook_state=list_hook_state.load(std::memory_order_acquire);
        if(hook_state==2) {
            auto** methods=*reinterpret_cast<void***>(list);
            if(methods[10]!=reinterpret_cast<void*>(first_list_methods.reset) ||
               methods[26]!=reinterpret_cast<void*>(first_list_methods.barriers))
                sync.incomplete=true; // another command-list implementation
        } else sync.incomplete=true;
        flash.generation=sync.generation;
        flash.evaluate_sequence=sync.evaluate_sequence;flash.color_sequence=sync.color.sequence;
        flash.color_after=unsigned(sync.color.after);
        flash.color_age=sync.color.seen && sync.evaluate_sequence>sync.color.sequence?
            static_cast<uint32_t>(std::min<uint64_t>(UINT32_MAX,
                sync.evaluate_sequence-sync.color.sequence)):UINT32_MAX;
        flash.source_bits=unsigned(sync.reset_observed) |
            (unsigned(sync.closed)<<1) | (unsigned(sync.oversized_barrier)<<2) |
            (unsigned(sync.color.seen)<<3) |
            (unsigned(sync.color.flags==D3D12_RESOURCE_BARRIER_FLAG_NONE)<<4) |
            (unsigned(sync.color.subresource==D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES)<<5) |
            (unsigned(sync.color.after==D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)<<6) |
            (unsigned(flash.color_age<=64)<<7) | (unsigned(sync.incomplete)<<9);
#ifdef NRB_DLSS_COLOR_STATE_PROOF_V1
        // Additive source-mask fields; the frozen v2 buffer ABI is unchanged.
        if(sync.color_state.enabled) flash.source_bits|=(1u<<10) |
            (unsigned(sync.color_state.known)<<11) | (unsigned(color_state_shadow_mode)<<12) |
            (unsigned(sync.color_state.reason)<<16);
#endif
#ifdef NRB_XESS_IDENTITY_TEST
        flash.source_bits|=unsigned(identity_source_state_proven)<<8;
        if(flash.eligible && identity_source_state_proven)
            nrb::flashdiag::recorder.count(nrb::flashdiag::Counter::source_proven);
#endif
#ifdef NRB_ONE_FRAME_COLOR_READBACK
        nrb::one_frame_readback_record(list,frame.color,sync,log_line);
#endif
        // Keep a few early unknown samples, then reserve the bounded log for
        // generations whose successful Reset was actually observed.
        static std::atomic<uint32_t> unknown_logs{0};
        const bool report=sync.reset_observed?
            sync_eval_logs.fetch_add(1,std::memory_order_relaxed)<16:
            unknown_logs.fetch_add(1,std::memory_order_relaxed)<4;
        if(report) {
            char line[1000];
            sprintf_s(line,"{\"event\":\"sync_boundary\",\"list\":\"%p\",\"hook_state\":%d,\"generation\":%llu,\"evaluate_sequence\":%llu,\"reset_observed\":%d,\"closed\":%d,\"incomplete\":%d,\"submit_batch_before\":%llu,\"color\":{\"resource\":\"%p\",\"barrier_seen\":%d,\"sequence\":%llu,\"before\":%u,\"after\":%u,\"flags\":%u,\"subresource\":%u,\"full_non_pixel_read\":%d},\"motion\":{\"resource\":\"%p\",\"barrier_seen\":%d,\"sequence\":%llu,\"before\":%u,\"after\":%u,\"flags\":%u,\"subresource\":%u,\"full_non_pixel_read\":%d},\"source_submission_proven\":false,\"conclusion\":\"unknown\"}\r\n",
                static_cast<void*>(list),hook_state,sync.generation,sync.evaluate_sequence,
                int(sync.reset_observed),int(sync.closed),int(sync.incomplete),sync.submit_batch_before,
                static_cast<void*>(frame.color),int(sync.color.seen),sync.color.sequence,
                unsigned(sync.color.before),unsigned(sync.color.after),unsigned(sync.color.flags),
                sync.color.subresource,int(sync.color.full_non_pixel_read),
                static_cast<void*>(frame.motion),int(sync.motion.seen),sync.motion.sequence,
                unsigned(sync.motion.before),unsigned(sync.motion.after),unsigned(sync.motion.flags),
                sync.motion.subresource,int(sync.motion.full_non_pixel_read));
            log_line(line);
        }
    }
    if(flash.eligible && flash.game_reset) {
        nrb::flashdiag::recorder.count(nrb::flashdiag::Counter::game_reset);
        nrb::flashdiag::recorder.event(nrb::flashdiag::EventKind::reset,flash);
    }
    nrb::flashdiag::recorder.progress(flash);
    frame.route_evidence=decision.evidence;
    frame.frame_id = ++frame_id;
    frame.controls = controls;
    frame.control_epoch = control_epoch;
    frame.reset_history |= controls.history == NRB_HISTORY_RESET ||
        last_processed_id + 1 != frame.frame_id || last_processed_epoch != control_epoch ||
        last_width != frame.render_width || last_height != frame.render_height || last_route != route;
    auto status = nrb::validate_metadata(frame);
    // FSR.frameTimeDelta and XeSS optional markers are useful diagnostics,
    // not cross-game proof of the external API entry. Only the game's/SL's
    // direct NGX call currently establishes the DLSS route for processing.
    if (route == NRB_ROUTE_UNKNOWN ||
        decision.evidence == NRB_EVIDENCE_FSR_MARKER ||
        decision.evidence == NRB_EVIDENCE_XESS_MARKER)
        status=NRB_BYPASS_UNKNOWN_ROUTE;
    QueueSnapshot observed{};
    // Collect real resource descriptors even when HDR or another metadata
    // contract makes NR ineligible. The mock test has no resources and skips
    // COM inspection entirely.
    if (frame.color && frame.motion && frame.output) {
        ID3D12Device* d=nullptr;
        if (FAILED(list->GetDevice(IID_PPV_ARGS(&d)))) status=NRB_BYPASS_UNSUPPORTED;
        else {
            frame.device=d;
            observed=queue_snapshot(d,list);
            if (!queue && observed.selected) {
                queue=observed.selected; observed.selected=nullptr;
            }
            if (queue && observed.selected && queue!=observed.selected)
                observed.meta.same_device=false;
            if (observed.selected) observed.selected->Release();
        }
    }
    frame.queue=queue;
    if (frame.device && queue) {
        ID3D12Device* q_device=nullptr;
        if(FAILED(queue->GetDevice(IID_PPV_ARGS(&q_device))) || q_device!=frame.device)
            observed.meta.same_device=false;
        if(q_device) q_device->Release();
    }
    nrb::ResourceMeta color_meta{},motion_meta{},depth_meta{},exposure_meta{},output_meta{};
    if (frame.device) {
        const auto resource_meta=[](ID3D12Resource* r) {
            nrb::ResourceMeta meta{};
            if(r) {const auto desc=r->GetDesc();meta.format=static_cast<uint32_t>(desc.Format);
                meta.width=static_cast<uint32_t>(desc.Width);meta.height=desc.Height;}
            return meta;
        };
        color_meta=resource_meta(frame.color);
        motion_meta=resource_meta(frame.motion);
        depth_meta=resource_meta(frame.depth);
        exposure_meta=resource_meta(frame.exposure);
        output_meta=resource_meta(frame.output);
    }
    bool synced=!borrowed.color || sync_queue();
    if (status == NRB_OK && !controls.enabled) status = NRB_BYPASS_DISABLED;
    if (status == NRB_OK && (!processor.process || !processor.retire)) status = NRB_BYPASS_DISABLED;
    if (status == NRB_OK && built_in_processor && !runtime_assets_available)
        status = NRB_BYPASS_RUNTIME_MISSING;
    if (status == NRB_OK && processor_failed) status=NRB_PROCESS_ERROR;
    if (status == NRB_OK && !synced) status=NRB_BYPASS_UNSYNCED;
    if (status == NRB_OK) status = nrb::validate_resources(frame);
    if (status == NRB_OK) status=nrb::validate_python_contract(frame);
    if (status == NRB_OK && (!queue || !observed.meta.ordinal || !observed.meta.same_device ||
        !source_submission_proven.load(std::memory_order_acquire))) status = NRB_BYPASS_UNSYNCED;
    if (status == NRB_OK && !sync_queue()) status=NRB_PROCESS_ERROR;
    NRB_Result result{};
    result.abi_size = sizeof(result);
    double process_ms=0.0;
    if (status == NRB_OK) {
        const auto started=std::chrono::steady_clock::now();
        try {
            if (processor.process(processor.context,&frame,&result)!=0 ||
                !nrb::validate_result(frame,result) ||
                FAILED(queue->Wait(result.ready_fence,result.ready_value)))
                status=NRB_PROCESS_ERROR;
        } catch (...) {status=NRB_PROCESS_ERROR;}
        process_ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-started).count();
        if(status!=NRB_OK) processor_failed=true;
        if(status!=NRB_OK && result.color) {
            // Keep the borrowed output pinned: its GPU work or consumer order
            // is unknown, so it is unsafe to recycle it into a later frame.
            borrowed=result;
            consumer_list=nullptr;
        }
    }
    if(frame.device) frame.device->Release();
    const uint32_t sample=diagnostic_gate.claim(route);
    if (sample) {
        const auto line=nrb::format_diagnostic(sample,frame,color_meta,motion_meta,
            depth_meta,exposure_meta,output_meta,observed.meta,status);
        log_line(line.c_str());
    }
    last_status.store(status, std::memory_order_relaxed);
    NVSDK_NGX_Result rc;
    if (status == NRB_OK) {
        flash_inline_nr=true; // current result only; never count it as raw fallback
        params->Set(NVSDK_NGX_Parameter_Color, result.color);
        borrowed = result;
        consumer_list=list;
        {std::lock_guard queue_guard(queue_mutex);consumer_submit_before=submit_serial;}
        consumer_queue_ordinal=observed.meta.ordinal;
        rc = call_original();
        params->Set(NVSDK_NGX_Parameter_Color, frame.color);
        if (rc == NVSDK_NGX_Result_Success) {
            last_processed_id = frame.frame_id;
            last_processed_epoch = control_epoch;
            last_width = frame.render_width;
            last_height = frame.render_height;
            last_route = route;
            processed_count.fetch_add(1,std::memory_order_relaxed);
            if (process_sample_count==process_samples.size())
                process_sample_sum-=process_samples[process_sample_cursor];
            else ++process_sample_count;
            process_samples[process_sample_cursor]=process_ms;
            process_sample_sum+=process_ms;
            process_sample_cursor=(process_sample_cursor+1)%process_samples.size();
            last_process_ms.store(process_ms,std::memory_order_relaxed);
            average_process_ms.store(process_sample_sum/process_sample_count,
                                     std::memory_order_relaxed);
        } else {
            last_processed_id = 0;
            last_status.store(NRB_PROCESS_ERROR,std::memory_order_relaxed);
        }
    } else {
        rc = call_original();
    }
    inside = false;
    return rc;
}
}

NRB_API int NRB_RegisterProcessor(const NRB_Processor* p) {
    std::lock_guard guard(state_mutex);
    if (borrowed.color) return 0;
    if (!p) { processor = {}; built_in_processor=false; processor_failed=false;
        nrb::gpuhandoff::processor_changed(nullptr);return 1; }
    if (p->abi_size != sizeof(NRB_Processor) || !p->process || !p->retire) return 0;
    processor = *p;
    built_in_processor=p->context==&python_processor;
    processor_failed=false;
    nrb::gpuhandoff::processor_changed(p);
    return 1;
}
NRB_API int NRB_RegisterQueue(ID3D12CommandQueue* q) {
    if(q) observe_queue(q);
    std::lock_guard guard(state_mutex);
    if (borrowed.color || (q && q->GetDesc().Type != D3D12_COMMAND_LIST_TYPE_DIRECT)) return 0;
    if (q) q->AddRef();
    if (queue) queue->Release();
    queue = q;
    nrb::gpuhandoff::queue_changed();
    if (completion) { completion->Release(); completion = nullptr; }
    if (completion_event) { CloseHandle(completion_event); completion_event = nullptr; }
    completion_value = 0;
    return 1;
}
NRB_API void NRB_SetProducerReady(int ready) {
    // Kept for old callers. A Boolean contains no resource/queue/frame proof.
    static std::atomic<bool> warned=false;
    if(ready && !warned.exchange(true))
        log_line("NRB_SetProducerReady ignored: source submission proof required\r\n");
}
NRB_API int NRB_SetControls(const NRB_Controls* c) {
    if (!c || !nrb::validate_controls(*c)) return 0;
    std::lock_guard guard(state_mutex);
    controls = *c;
    ++control_epoch;
    process_sample_count=process_sample_cursor=0;
    process_sample_sum=0.0;
    last_process_ms.store(0.0,std::memory_order_relaxed);
    average_process_ms.store(0.0,std::memory_order_relaxed);
#ifdef NRB_LIVE_NR_TEST
    nrb::deferred_nr_reset_metrics();
#endif
    return 1;
}
NRB_API int NRB_GetControls(NRB_Controls* c) {
    if (!c || c->abi_size != sizeof(NRB_Controls)) return 0;
    std::lock_guard guard(state_mutex);
    *c = controls;
    return 1;
}
// Read-only, fixed-size CPU snapshot. No file write or GPU synchronization.
NRB_API int NRB_GetPeriodicFlashDiagV2(nrb::flashdiag::Snapshot* snapshot) {
    return nrb::flashdiag::recorder.snapshot(snapshot)?1:0;
}
NRB_API int NRB_GetPeriodicFlashContextV2(nrb::flashdiag::Context* context) {
    return nrb::flashdiag::current_context(context)?1:0;
}
NRB_API NRB_Status NRB_LastStatus() {
#ifdef NRB_LIVE_NR_TEST
    if(nrb::deferred_nr_failed()) return NRB_PROCESS_ERROR;
    if(nrb::deferred_nr_processed()) {
        std::lock_guard guard(state_mutex);
        return controls.enabled?NRB_OK:NRB_BYPASS_DISABLED;
    }
#endif
    return last_status.load(std::memory_order_relaxed);
}
NRB_API uint64_t NRB_InterceptCount(NRB_Route r) {
    return r <= NRB_ROUTE_XESS ? counts[r].load(std::memory_order_relaxed) : 0;
}
NRB_API uint64_t NRB_ProcessedCount() {
#ifdef NRB_LIVE_NR_TEST
    return processed_count.load(std::memory_order_relaxed)+nrb::deferred_nr_processed();
#else
    return processed_count.load(std::memory_order_relaxed);
#endif
}
NRB_API double NRB_LastProcessMs() {
#ifdef NRB_LIVE_NR_TEST
    if(!nrb::deferred_nr_timing_enabled()) return 0;
    const double t=nrb::deferred_nr_last_ms();
    if(t>0) return t;
#endif
    return last_process_ms.load(std::memory_order_relaxed);
}
NRB_API double NRB_AverageProcessMs() {
#ifdef NRB_LIVE_NR_TEST
    if(!nrb::deferred_nr_timing_enabled()) return 0;
    const double t=nrb::deferred_nr_average_ms();
    if(t>0) return t;
#endif
    return average_process_ms.load(std::memory_order_relaxed);
}
NRB_API int NRB_GetStageTimes(NRB_StageTimes* times) {
#ifdef NRB_DEFERRED_IDENTITY_TEST
    return nrb::deferred_nr_stage_times(times) ? 1 : 0;
#else
    if(!times || times->abi_size!=sizeof(NRB_StageTimes)) return 0;
    *times=NRB_StageTimes{};
    times->abi_size=sizeof(NRB_StageTimes);
    return 0;
#endif
}
NRB_API int NRB_GetRecordTimes(NRB_RecordTimes* times) {
#ifdef NRB_DEFERRED_IDENTITY_TEST
    return nrb::deferred_nr_record_times(times)?1:0;
#else
    if(!times || times->abi_size!=sizeof(NRB_RecordTimes)) return 0;
    *times=NRB_RecordTimes{};times->abi_size=sizeof(NRB_RecordTimes);return 0;
#endif
}
NRB_API int NRB_GetHdrCacheStats(NRB_HdrCacheStats* stats) {
#ifdef NRB_LIVE_NR_TEST
    return nrb::deferred_hdr_cache_stats(stats)?1:0;
#else
    if(!stats || stats->abi_size!=sizeof(NRB_HdrCacheStats)) return 0;
    *stats=NRB_HdrCacheStats{};stats->abi_size=sizeof(NRB_HdrCacheStats);return 0;
#endif
}
NRB_API int NRB_SetHdrCacheEnabled(int enabled) {
#ifdef NRB_LIVE_NR_TEST
    if(enabled!=0 && enabled!=1) return 0;
    nrb::deferred_hdr_set_cache_enabled(enabled!=0);return 1;
#else
    (void)enabled;return 0;
#endif
}
NRB_API int NRB_GetTimingEnabled() {
#ifdef NRB_LIVE_NR_TEST
    return nrb::deferred_nr_timing_enabled()?1:0;
#else
    return 0;
#endif
}
NRB_API int NRB_SetTimingEnabled(int enabled) {
#ifdef NRB_LIVE_NR_TEST
    if(enabled!=0 && enabled!=1) return 0;
    nrb::deferred_nr_set_timing_enabled(enabled!=0);
    return 1;
#else
    (void)enabled;
    return 0;
#endif
}

namespace {
DWORD WINAPI initialize_worker(void*) {
    // OptiScaler calls InitializeASI from its DLL_PROCESS_ATTACH. Keep all
    // Detours, file I/O and Python initialization outside the loader lock.
    try {
    NRB_Processor built_in{sizeof(NRB_Processor),&python_processor,
        nrb::PythonProcessor::process_callback,nrb::PythonProcessor::retire_callback};
    NRB_RegisterProcessor(&built_in);
#ifdef NRB_LIVE_NR_TEST
    {std::lock_guard guard(state_mutex);
        controls.enabled=1;controls.input_height=540;
        controls.history=NRB_HISTORY_FUSED;controls.graph_replay=1;}
#endif
#ifdef NRB_LIVE_NR_TEST
    if(!nrb::deferred_hdr_warmup_shader())
        log_line("HDR shader cache warmup unavailable; first frame will retry\r\n");
#endif
    runtime_assets_available=python_processor.assets_available();
    if(!runtime_assets_available) log_line("Portable RE8 runtime assets missing; processor registered but disabled\r\n");
    install_queue_capture();
    const wchar_t* names[] = {L"OptiScaler.dll", L"dxgi.dll", L"version.dll",
        L"winmm.dll", L"winhttp.dll", L"OptiScaler.asi"};
    for (auto name : names) {
        if (original) break;
        HMODULE candidate = GetModuleHandleW(name);
        if (!candidate) continue;
        auto address = GetProcAddress(candidate, "NVSDK_NGX_D3D12_EvaluateFeature");
        if (address) { host = candidate; original = reinterpret_cast<Evaluate>(address); break; }
    }
    hook_device_export(host,original_create_host,observed_create_host);
    if (!original) {
        log_line("ASI loaded; OptiScaler NGX evaluator export missing\r\n");
        init_state.store(3, std::memory_order_release);
        return 0;
    }
    if (!attach_detour(reinterpret_cast<void**>(&original),reinterpret_cast<void*>(intercept))) {
        original = nullptr;
        log_line("ASI loaded; NGX evaluator detour failed\r\n");
        init_state.store(3, std::memory_order_release);
    } else {
        install_sl_feature_probe();
        if(HANDLE worker=CreateThread(nullptr,0,install_upscaler_api_probes,
                                      nullptr,0,nullptr)) CloseHandle(worker);
        log_line("ASI loaded; pre-SR NGX evaluator detour attached\r\n");
        init_state.store(2, std::memory_order_release);
        // Controls must be reachable even while every frame is bypassed.
        if (runtime_assets_available && !python_processor.start_controls())
            log_line("Portable NR control page startup failed\r\n");
    }
    } catch (...) {
        log_line("ASI deferred initialization threw an exception\r\n");
        init_state.store(3, std::memory_order_release);
    }
    return 0;
}
}

extern "C" __declspec(dllexport) void InitializeASI() {
    int expected=0;
    if (!init_state.compare_exchange_strong(expected,1,std::memory_order_acq_rel)) return;
    HANDLE worker=CreateThread(nullptr,0,initialize_worker,nullptr,0,nullptr);
    if (worker) CloseHandle(worker);
    else init_state.store(3,std::memory_order_release);
}
NRB_API int NRB_InitState() { return init_state.load(std::memory_order_acquire); }

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }
