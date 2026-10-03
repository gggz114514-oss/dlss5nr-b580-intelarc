"""Build an UNAPPLIED diagnostic patch from v1; stdlib only, no native loading.

Only the named v2 patch is written. Target C++/host sources remain untouched.
The host diff targets the E worktree which already contains the v1 hooks.
"""
from __future__ import annotations

import difflib
import hashlib
import importlib.util
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
HOST = WORKSPACE / ".codex-worktrees/re8-fp8-unround-fast-20260928/game/nr_game_pre_xess_host.py"
PATCH = PROJECT / "tools/periodic_flash_native_diag_v2.patch"
HEADER = "src/periodic_flash_native_diag.h"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise AssertionError("expected one frozen callsite: " + old[:100])
    return text.replace(old, new, 1)


NATIVE_HEADER = r'''#pragma once
// Fixed CPU buffers only. No GPU calls, file I/O, heap growth, or frame changes.
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>

namespace nrb::flashdiag {
enum class Counter : uint32_t {
    seen, eligible, source_proven, delegated, submitted, nr_composited,
    original_fallback, excluded_original, game_reset, generation_mismatch,
    invalidated, failure, nr_started, nr_return_ok, nr_processed, nr_retired,
    nr_skipped, count
};
enum class Reason : uint32_t {
    none, scope_or_feature, source_state, record_rejected, tail_unverified,
    disarmed, arguments, attempt_budget, textures, pending_full, allocation,
    motion_contract, device, record_resources, hdr_missing, motion_proxy,
    record_close, original_evaluate, processor_missing, disabled, latched_failure,
    prepared_wait, processor_result, composite_or_wait, completion_signal,
    retire, generation, list_reset, cpp_exception, prepare_signal,
    prepare_event, prepare_registration, prepare_timeout, queue_wait,
    composite_record, composite_close, completion_event, completion_registration,
    completion_timeout, retire_callback, count
};
enum class EventKind : uint32_t {
    record_reject, fallback, reset, failure, generation, invalidated
};
enum class Stage : uint32_t {
    recorded, submitted, processor_called, processor_return_ok, result_valid,
    composite_submitted, sr_submitted, retired, raw_fallback, invalidated,
    motion_probe_recorded, original_called
};
constexpr uint32_t bit(Stage stage) noexcept {return 1u<<uint32_t(stage);}
struct Context {
    uint64_t eval_id=0, list=0, generation=0, feature_id=0, nr_frame_id=0, sr_sequence=0;
    uint64_t evaluate_sequence=0, color_sequence=0, submit_batch=0;
    uint32_t route=0, evidence=0, effective_route=0, eligible=0, sr_slot=0;
    uint32_t source_bits=0, color_after=0, color_age=0, game_reset=0, enabled=0, history=0;
    uint32_t controls_known=0, nr_reset=0, stage_bits=0, motion_probe_index=UINT32_MAX;
    uint32_t record_reason=0;
};
struct Event {
    uint64_t serial=0, steady_us=0, composited_before=0, detail=0, detail2=0;
    Context context{};
    uint32_t kind=0, reason=0;
};
struct Frame {
    uint64_t serial=0, steady_us=0;
    Context context{};
    uint32_t reason=0, reserved=0;
};
constexpr uint32_t capacity=64, frame_capacity=128;
constexpr uint32_t counter_count=uint32_t(Counter::count), reason_count=uint32_t(Reason::count);
struct Snapshot {
    uint32_t abi_size=sizeof(Snapshot), version=2;
    uint64_t retained=0, overwritten=0, dropped=0;
    uint64_t frames_retained=0, frames_overwritten=0, frames_dropped=0, sampled_steady_us=0;
    std::array<uint64_t,counter_count> counters{};
    std::array<uint64_t,reason_count> reasons{};
    Context current{};
    std::array<Event,capacity> events{};
    std::array<Frame,frame_capacity> frames{};
};
static_assert(sizeof(Context)==136 && sizeof(Event)==184 && sizeof(Frame)==160);
static_assert(sizeof(Snapshot)==32912);
inline uint64_t steady_us() noexcept {
    return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
// This is a callback scope, not a latest/global sample. The web thread cannot
// manufacture a join by observing another CPU thread's active NR call.
inline thread_local const Context* active_context=nullptr;
class ActiveContext {
    const Context* previous_;
public:
    explicit ActiveContext(const Context& context) noexcept:previous_(active_context) {
        active_context=&context;
    }
    ~ActiveContext() {active_context=previous_;}
};
inline bool current_context(Context* out) noexcept {
    if(!out || !active_context) return false;
    *out=*active_context;return true;
}
class Recorder {
public:
    void count(Counter c) noexcept {counters_[uint32_t(c)].fetch_add(1,std::memory_order_relaxed);}
    uint64_t begin() noexcept {return counters_[uint32_t(Counter::seen)].fetch_add(
        1,std::memory_order_relaxed)+1;}
    void observe(Context& c) noexcept {
        if(c.sr_slot) c.sr_sequence=sr_serial_.fetch_add(1,std::memory_order_relaxed)+1;
        progress(c);
    }
    void progress(const Context& c) noexcept {
        if(!c.sr_slot) return;
        try {std::unique_lock guard(mutex_,std::try_to_lock);
            if(!guard.owns_lock()) {frames_dropped_.fetch_add(1,std::memory_order_relaxed);return;}
            if(c.sr_sequence>=current_.sr_sequence) current_=c;
        } catch(...) {frames_dropped_.fetch_add(1,std::memory_order_relaxed);}
    }
    void event(EventKind kind,const Context& context,Reason reason=Reason::none,
               uint64_t detail=0,uint64_t detail2=0) noexcept {
        reasons_[uint32_t(reason)].fetch_add(1,std::memory_order_relaxed);
        try {std::unique_lock guard(mutex_,std::try_to_lock);
            if(!guard.owns_lock()) {dropped_.fetch_add(1,std::memory_order_relaxed);return;}
            Event e{};e.serial=++serial_;e.steady_us=steady_us();
            e.composited_before=counters_[uint32_t(Counter::nr_composited)].load(std::memory_order_relaxed);
            e.context=context;e.kind=uint32_t(kind);e.reason=uint32_t(reason);
            e.detail=detail;e.detail2=detail2;events_[(serial_-1)%capacity]=e;
        } catch(...) {dropped_.fetch_add(1,std::memory_order_relaxed);}
    }
    void fallback(Context& context,Reason reason,uint64_t original_result=0) noexcept {
        if(!context.eligible && !context.sr_slot) {count(Counter::excluded_original);return;}
        context.stage_bits|=bit(Stage::raw_fallback);count(Counter::original_fallback);
        if(!(context.stage_bits&bit(Stage::processor_called))) count(Counter::nr_skipped);
        event(EventKind::fallback,context,reason,original_result);
    }
    void finish(const Context& context,Reason reason=Reason::none) noexcept {
        if(!context.sr_slot) return;
        try {std::unique_lock guard(mutex_,std::try_to_lock);
            if(!guard.owns_lock()) {frames_dropped_.fetch_add(1,std::memory_order_relaxed);return;}
            Frame f{};f.serial=++frame_serial_;f.steady_us=steady_us();
            f.context=context;f.reason=uint32_t(reason);frames_[(frame_serial_-1)%frame_capacity]=f;
            if(context.sr_sequence>=current_.sr_sequence) current_=context;
        } catch(...) {frames_dropped_.fetch_add(1,std::memory_order_relaxed);}
    }
    bool snapshot(Snapshot* out) noexcept {
        if(!out || out->abi_size!=sizeof(Snapshot) || out->version!=2) return false;
        try {std::unique_lock guard(mutex_,std::try_to_lock);if(!guard.owns_lock()) return false;
            Snapshot result{};result.retained=serial_<capacity?serial_:capacity;
            result.overwritten=serial_-result.retained;result.dropped=dropped_.load(std::memory_order_relaxed);
            result.frames_retained=frame_serial_<frame_capacity?frame_serial_:frame_capacity;
            result.frames_overwritten=frame_serial_-result.frames_retained;
            result.frames_dropped=frames_dropped_.load(std::memory_order_relaxed);
            result.sampled_steady_us=steady_us();result.current=current_;
            for(uint32_t i=0;i<counter_count;++i) result.counters[i]=counters_[i].load(std::memory_order_relaxed);
            for(uint32_t i=0;i<reason_count;++i) result.reasons[i]=reasons_[i].load(std::memory_order_relaxed);
            const auto first=serial_-result.retained, frame_first=frame_serial_-result.frames_retained;
            for(uint64_t i=0;i<result.retained;++i) result.events[i]=events_[(first+i)%capacity];
            for(uint64_t i=0;i<result.frames_retained;++i) result.frames[i]=frames_[(frame_first+i)%frame_capacity];
            *out=result;return true;
        } catch(...) {return false;}
    }
private:
    std::array<std::atomic<uint64_t>,counter_count> counters_{};
    std::array<std::atomic<uint64_t>,reason_count> reasons_{};
    std::atomic<uint64_t> dropped_{0},frames_dropped_{0},sr_serial_{0};
    std::mutex mutex_;
    std::array<Event,capacity> events_{};
    std::array<Frame,frame_capacity> frames_{};
    Context current_{};
    uint64_t serial_=0,frame_serial_=0;
};
inline Recorder recorder;
} // namespace nrb::flashdiag
'''


def patched_sources():
    spec = importlib.util.spec_from_file_location("flashdiag_v1_patch_parser", PROJECT / "tools/test_periodic_flash_native_diag_cpu_v1.py")
    v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)
    for name, digest in v1.BASE_SHA.items():
        if hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() != digest:
            raise AssertionError("native base changed: " + name)
    native = v1.apply_in_memory()
    native[HEADER] = NATIVE_HEADER
    header = native["src/deferred_identity.h"].replace("const flashdiag::Context* diagnostic=nullptr", "flashdiag::Context* diagnostic=nullptr")
    native["src/deferred_identity.h"] = header
    asi = native["src/asi.cpp"]
    asi = replace_once(asi, "    bool flash_inline_nr=false;", "    nrb::flashdiag::recorder.observe(flash);\n    bool flash_inline_nr=false;")
    asi = replace_once(asi, "            flash_fallback_reason=nrb::flashdiag::Reason::record_rejected;",
        "            flash_fallback_reason=flash.record_reason?\n                nrb::flashdiag::Reason(flash.record_reason):nrb::flashdiag::Reason::record_rejected;")
    asi = replace_once(asi, '''        if(!flash_inline_nr)
            nrb::flashdiag::recorder.fallback(flash,flash.eligible?
                flash_fallback_reason:nrb::flashdiag::Reason::scope_or_feature);
        const auto rc=original(list,handle,params,callback);''', '''        flash.stage_bits|=nrb::flashdiag::bit(nrb::flashdiag::Stage::original_called);
        const auto rc=original(list,handle,params,callback);
        if(!flash_inline_nr) {
            const auto why=flash.eligible?flash_fallback_reason:
                nrb::flashdiag::Reason::scope_or_feature;
            nrb::flashdiag::recorder.fallback(flash,why,unsigned(rc));
            nrb::flashdiag::recorder.finish(flash,why);
        }''')
    asi = replace_once(asi, "    flash.enabled=controls.enabled;flash.history=unsigned(controls.history);",
        "    flash.enabled=controls.enabled;flash.history=unsigned(controls.history);flash.controls_known=1;")
    asi = replace_once(asi, "        flash.generation=sync.generation;", "        flash.generation=sync.generation;\n        flash.evaluate_sequence=sync.evaluate_sequence;flash.color_sequence=sync.color.sequence;")
    asi = replace_once(asi, "    frame.route_evidence=decision.evidence;", "    nrb::flashdiag::recorder.progress(flash);\n    frame.route_evidence=decision.evidence;")
    asi = replace_once(asi, '''NRB_API int NRB_GetPeriodicFlashDiag(nrb::flashdiag::Snapshot* snapshot) {
    return nrb::flashdiag::recorder.snapshot(snapshot)?1:0;
}''', '''NRB_API int NRB_GetPeriodicFlashDiagV2(nrb::flashdiag::Snapshot* snapshot) {
    return nrb::flashdiag::recorder.snapshot(snapshot)?1:0;
}
NRB_API int NRB_GetPeriodicFlashContextV2(nrb::flashdiag::Context* context) {
    return nrb::flashdiag::current_context(context)?1:0;
}''')
    native["src/asi.cpp"] = asi
    cpp = native["src/deferred_identity.cpp"]
    cpp = replace_once(cpp, "    flashdiag::Context diagnostic{};", "    flashdiag::Context diagnostic{};\n    flashdiag::Reason diagnostic_reason=flashdiag::Reason::none;")
    cpp = replace_once(cpp, "NVSDK_NGX_Result& result,const flashdiag::Context* diagnostic)", "NVSDK_NGX_Result& result,flashdiag::Context* diagnostic)")
    cpp = replace_once(cpp, "        flashdiag::recorder.event(flashdiag::EventKind::record_reject,", "        if(diagnostic) diagnostic->record_reason=uint32_t(reason);\n        flashdiag::recorder.event(flashdiag::EventKind::record_reject,")
    cpp = replace_once(cpp, "        if(entry.game_motion && (motion_index<4 ||", "        if(entry.game_motion) entry.diagnostic.motion_probe_index=motion_index;\n        if(entry.game_motion && (motion_index<4 ||")
    cpp = replace_once(cpp, "                auto to_copy=transition(entry.hdr->nr_motion(),", "                entry.diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::motion_probe_recorded);\n                auto to_copy=transition(entry.hdr->nr_motion(),")
    cpp = replace_once(cpp, "        pending.push_back(std::move(owned));", "        entry.diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::recorded);\n        flashdiag::recorder.progress(entry.diagnostic);\n        pending.push_back(std::move(owned));")
    cpp = replace_once(cpp, "            matches[matched++]={i,pending[p].get()};break;", "            pending[p]->diagnostic.submit_batch=observations?observations[i].batch:0;\n            matches[matched++]={i,pending[p].get()};break;")
    cpp = replace_once(cpp, "        flashdiag::recorder.count(flashdiag::Counter::submitted);", "        flashdiag::recorder.count(flashdiag::Counter::submitted);\n        matches[m].item->diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::submitted);")
    cpp = replace_once(cpp, "            if(SUCCEEDED(queue->Signal(item.prepared.Get(),1))) {", "            const HRESULT prepare_signal=queue->Signal(item.prepared.Get(),1);\n            if(SUCCEEDED(prepare_signal)) {")
    cpp = replace_once(cpp, '''                    if(SUCCEEDED(item.prepared->SetEventOnCompletion(1,event)) &&
                       WaitForSingleObject(event,10000)==WAIT_OBJECT_0) {''', '''                    const HRESULT registered=item.prepared->SetEventOnCompletion(1,event);
                    const DWORD waited=SUCCEEDED(registered)?WaitForSingleObject(event,10000):WAIT_FAILED;
                    if(SUCCEEDED(registered) && waited==WAIT_OBJECT_0) {''')
    cpp = replace_once(cpp, '''                        try {valid=processor->process(processor->context,&frame,&result)==0 &&
                            result.color && result.ready_fence && result.ready_value &&
                            result.color->GetDesc().Format==DXGI_FORMAT_R32G32B32A32_FLOAT &&
                            result.color->GetDesc().Width==1280 &&
                            result.color->GetDesc().Height==720;} catch(...) {valid=false;}''', '''                        item.diagnostic.nr_frame_id=frame.frame_id;
                        item.diagnostic.nr_reset=frame.reset_history!=0;
                        item.diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::processor_called);
                        flashdiag::recorder.count(flashdiag::Counter::nr_started);
                        flashdiag::recorder.progress(item.diagnostic);
                        int processor_result=-1;
                        try {
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
                        if(valid) {
                            item.diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::result_valid);
                            flashdiag::recorder.count(flashdiag::Counter::nr_processed);
                        } else flashdiag::recorder.event(flashdiag::EventKind::failure,item.diagnostic,
                            fallback_reason,uint64_t(int64_t(processor_result)));''')
    cpp = replace_once(cpp, '''                    CloseHandle(event);
                }
            }
            if(valid) {''', '''                    else {
                        fallback_reason=FAILED(registered)?flashdiag::Reason::prepare_registration:
                            flashdiag::Reason::prepare_timeout;
                        flashdiag::recorder.event(flashdiag::EventKind::failure,item.diagnostic,
                            fallback_reason,uint64_t(uint32_t(registered)),waited);
                    }
                    CloseHandle(event);
                } else fallback_reason=flashdiag::Reason::prepare_event;
            } else fallback_reason=flashdiag::Reason::prepare_signal;
            if(valid) {''')
    cpp = replace_once(cpp, "                valid=SUCCEEDED(queue->Wait(item.nr_ready.Get(),result.ready_value)) &&", "                const HRESULT ready_wait=queue->Wait(item.nr_ready.Get(),result.ready_value);\n                bool composite_recorded=false;HRESULT composite_closed=S_FALSE;\n                valid=SUCCEEDED(ready_wait) &&\n                    (composite_recorded=")
    cpp = replace_once(cpp, '''                            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE) &&
                    SUCCEEDED(item.composite_list->Close());''', '''                            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)) &&
                    SUCCEEDED(composite_closed=item.composite_list->Close());
                if(!valid) fallback_reason=FAILED(ready_wait)?flashdiag::Reason::queue_wait:
                    !composite_recorded?flashdiag::Reason::composite_record:flashdiag::Reason::composite_close;''')
    cpp = replace_once(cpp, "                    flashdiag::recorder.count(flashdiag::Counter::nr_composited);", "                    flashdiag::recorder.count(flashdiag::Counter::nr_composited);\n                    item.diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::composite_submitted);")
    cpp = replace_once(cpp, '''        if(!item.nr_submitted)
            flashdiag::recorder.fallback(item.diagnostic,fallback_reason);''', '''        if(!item.nr_submitted) {
            item.diagnostic_reason=fallback_reason;
            flashdiag::recorder.fallback(item.diagnostic,fallback_reason);
        }''')
    cpp = replace_once(cpp, '''        original_execute(queue,1,&deferred);
    }''', '''        original_execute(queue,1,&deferred);
        matches[m].item->diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::sr_submitted);
        flashdiag::recorder.progress(matches[m].item->diagnostic);
    }''')
    cpp = replace_once(cpp, "        if(FAILED(queue->Signal(item.completion.Get(),1))) {", "        const HRESULT completion_signal=queue->Signal(item.completion.Get(),1);\n        if(FAILED(completion_signal)) {\n            item.diagnostic_reason=flashdiag::Reason::completion_signal;")
    cpp = replace_once(cpp, '''            const bool done=event &&
                SUCCEEDED(item.completion->SetEventOnCompletion(1,event)) &&
                WaitForSingleObject(event,10000)==WAIT_OBJECT_0;''', '''            const HRESULT registered=event?item.completion->SetEventOnCompletion(1,event):E_FAIL;
            const DWORD waited=event && SUCCEEDED(registered)?WaitForSingleObject(event,10000):WAIT_FAILED;
            const bool done=event && SUCCEEDED(registered) && waited==WAIT_OBJECT_0;''')
    cpp = replace_once(cpp, '''                item.nr_submitted=false;
                const auto ended''', '''                item.nr_submitted=false;
                item.diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::retired);
                flashdiag::recorder.count(flashdiag::Counter::nr_retired);
                const auto ended''')
    cpp = replace_once(cpp, '''                flashdiag::recorder.event(flashdiag::EventKind::failure,
                    item.diagnostic,flashdiag::Reason::retire);''', '''                item.diagnostic_reason=!event?flashdiag::Reason::completion_event:
                    FAILED(registered)?flashdiag::Reason::completion_registration:
                    !done?flashdiag::Reason::completion_timeout:flashdiag::Reason::retire_callback;
                flashdiag::recorder.event(flashdiag::EventKind::failure,
                    item.diagnostic,item.diagnostic_reason,uint64_t(uint32_t(registered)),waited);''')
    cpp = replace_once(cpp, '''    // A failed Signal retains the submitted objects until process exit; never''', '''    for(unsigned m=0;m<matched;m++)
        flashdiag::recorder.finish(matches[m].item->diagnostic,matches[m].item->diagnostic_reason);
    // A failed Signal retains the submitted objects until process exit; never''')
    cpp = replace_once(cpp, '''            flashdiag::recorder.count(flashdiag::Counter::invalidated);''', '''            pending[i]->diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::invalidated);
            flashdiag::recorder.count(flashdiag::Counter::invalidated);''')
    cpp = replace_once(cpp, '''            pending.erase(pending.begin()+i);
            invalidated.fetch_add''', '''            flashdiag::recorder.finish(pending[i]->diagnostic,flashdiag::Reason::list_reset);
            pending.erase(pending.begin()+i);
            invalidated.fetch_add''')
    native["src/deferred_identity.cpp"] = cpp
    result = {"cyberpunk-b580-nr-opt/" + name: value for name, value in native.items()}
    host = HOST.read_text(encoding="utf-8")
    host = replace_once(host, "_temporal_diagnostics = None", "_temporal_diagnostics = None\n_periodic_flash_diagnostics = None\n_periodic_flash_sampler = None\n_periodic_flash_enabled = os.environ.get(\"NR_DIAG_PERIODIC_FLASH_V2\") == \"1\"")
    host = replace_once(host, '''    return (_temporal_diagnostics.snapshot()
            if _temporal_diagnostics is not None else {"enabled": False})''', '''    result = (_temporal_diagnostics.snapshot()
              if _temporal_diagnostics is not None else {"enabled": False})
    if _periodic_flash_enabled:
        result["periodic_flash_v2"] = periodic_flash_diagnostics()
    return result''')
    host = replace_once(host, "def set_timing_enabled(enabled):", '''def periodic_flash_diagnostics():
    global _periodic_flash_sampler
    if _periodic_flash_diagnostics is None:
        return {"status": "unavailable", "reason": "no_python_nr_call_yet"}
    from periodic_flash_snapshot_v2 import WindowSampler
    if _periodic_flash_sampler is None:
        _periodic_flash_sampler = WindowSampler()
    web = sys.modules.get("cyberpunk_nr_web")
    return _periodic_flash_sampler.sample(getattr(web, "_native", None),
                                         _periodic_flash_diagnostics.snapshot())


def set_timing_enabled(enabled):''')
    host = replace_once(host, "    global _temporal_diagnostics\n", "    global _temporal_diagnostics, _periodic_flash_diagnostics\n")
    host = replace_once(host, '''    try:
        measure = _timing_enabled''', '''    flash_frame = None
    try:
        if _periodic_flash_enabled:
            try:
                from periodic_flash_snapshot_v2 import PythonFrameRecorder
                if _periodic_flash_diagnostics is None:
                    _periodic_flash_diagnostics = PythonFrameRecorder()
                web = sys.modules.get("cyberpunk_nr_web")
                flash_frame = _periodic_flash_diagnostics.start(frame_id, game_reset, getattr(web, "_native", None))
                if flash_frame is not None:
                    flash_frame["stage"] = "initialize"
            except Exception:
                flash_frame = None  # An absent optional CPU helper cannot disable NR.
        measure = _timing_enabled''')
    host = replace_once(host, "        if _temporal_events_enabled:\n", "        if _temporal_events_enabled or _periodic_flash_enabled:\n")
    host = replace_once(host, "        color, motion = _bridge.prepare(frame)", "        if flash_frame is not None:\n            flash_frame[\"stage\"] = \"prepare\"\n        color, motion = _bridge.prepare(frame)")
    host = replace_once(host, "        result = _modes.process(color, motion, height=setting.input_size,", '''        if flash_frame is not None:
            _periodic_flash_diagnostics.prepare(flash_frame, _modes, causes, {
                "experiment": setting.experiment_720, "optimizations": list(optimizations),
                "history": history_mode, "graph_requested": graph_replay, "backend": setting.backend_variant,
                "input_height": setting.input_size, "style": setting.style, "intensity": setting.model_intensity,
                "local_tone": setting.local_tone, "local_structure": setting.local_structure,
                "auto_mask": setting.auto_mask, "skin_structure": setting.skin_structure})
        result = _modes.process(color, motion, height=setting.input_size,''')
    host = replace_once(host, "        _bridge.export(result.contiguous(), frame_id=frame_id)", "        if flash_frame is not None:\n            flash_frame[\"stage\"] = \"export\"\n        _bridge.export(result.contiguous(), frame_id=frame_id)")
    host = replace_once(host, "        return output.resource, output.fence, output.value", "        if flash_frame is not None:\n            _periodic_flash_diagnostics.complete(flash_frame, _modes)\n        return output.resource, output.fence, output.value")
    host = replace_once(host, '''    except BaseException as exc:
        _failed = True''', '''    except BaseException as exc:
        if flash_frame is not None:
            _periodic_flash_diagnostics.complete(flash_frame, _modes, stage=flash_frame["stage"], exception=exc)
        _failed = True''') if host.count("    except BaseException as exc:\n        _failed = True") == 1 else host.replace(
        "    except BaseException as exc:\n        _failed = True", "    except BaseException as exc:\n        if flash_frame is not None:\n            _periodic_flash_diagnostics.complete(flash_frame, _modes, stage=flash_frame[\"stage\"], exception=exc)\n        _failed = True", 1)
    result[HOST.relative_to(WORKSPACE).as_posix()] = host
    return result


def patch_text(sources):
    chunks = []
    for name, value in sources.items():
        path = WORKSPACE / name
        exists = path.exists()
        original = path.read_text(encoding="utf-8") if exists else ""
        chunks.append("diff --git a/" + name + " b/" + name + "\n")
        if not exists:
            chunks.append("new file mode 100644\n")
        chunks.extend(difflib.unified_diff(original.splitlines(keepends=True), value.splitlines(keepends=True),
                      fromfile="a/" + name if exists else "/dev/null", tofile="b/" + name))
    return "".join(chunks)


if __name__ == "__main__":
    PATCH.write_text(patch_text(patched_sources()), encoding="utf-8", newline="\n")
    print("wrote unapplied v2 patch:", PATCH)
