#pragma once
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
