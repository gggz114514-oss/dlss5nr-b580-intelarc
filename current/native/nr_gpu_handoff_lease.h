#pragma once
#include <cstdint>
#include <limits>

// This policy is used by the product helper and the CPU fake-completion tests.
// Fence registration and queued GPU waits are never completion observations.
namespace nr_handoff {
enum class Phase : uint32_t { idle, prepared, exported, borrowed };
struct Completion {
    uint64_t pack = 0, unpack = 0, consumer = 0;
    bool device_ok = false, events_complete = false, events_success = false;
};
inline bool reached(uint64_t observed, uint64_t wanted) noexcept {
    return wanted && observed != std::numeric_limits<uint64_t>::max() && observed >= wanted;
}
struct Lease {
    static constexpr uint32_t event_capacity = 6; // wait, 2 inputs, result, signal, explicit close barrier
    Phase phase = Phase::idle;
    bool poisoned = false, closing = false;
    bool pack_possible = false, pack_signaled = false;
    bool unpack_possible = false, unpack_signaled = false;
    bool consumer_registered = false;
    uint64_t id = 0, pack_value = 0, unpack_value = 0, consumer_value = 0;
    uint32_t events_issued = 0, events_recorded = 0;
    bool idle() const noexcept { return phase == Phase::idle && !poisoned; }
    bool begin(uint64_t frame_id) noexcept {
        if (!idle()) return false;
        *this = Lease{}; phase = Phase::prepared; id = frame_id; return true;
    }
    bool reset_pack_allowed() const noexcept {
        return !poisoned && phase == Phase::prepared && !pack_possible;
    }
    bool reset_unpack_allowed() const noexcept {
        return !poisoned && phase == Phase::prepared && pack_signaled && !unpack_possible;
    }
    bool issue_event() noexcept {
        if (poisoned || events_issued >= event_capacity) return false;
        ++events_issued; return true;
    }
    bool record_event() noexcept {
        if (events_recorded >= events_issued) return false;
        ++events_recorded; return true;
    }
    bool borrow() noexcept {
        if (poisoned || phase != Phase::exported) return false;
        phase = Phase::borrowed; return true;
    }
    bool register_consumer(uint64_t value) noexcept {
        if (poisoned || phase != Phase::borrowed || consumer_registered || !value ||
            value == std::numeric_limits<uint64_t>::max()) return false;
        consumer_registered = true; consumer_value = value; return true;
    }
    bool safe(const Completion& c) const noexcept {
        if (poisoned || phase == Phase::idle || !c.device_ok || !c.events_complete || !c.events_success ||
            events_issued != events_recorded || !pack_possible || !pack_signaled ||
            !reached(c.pack, pack_value)) return false;
        if (phase == Phase::prepared && !closing) return false;
        if (events_recorded < (phase == Phase::prepared ? 4u : 5u)) return false;
        if (unpack_possible && (!unpack_signaled || !reached(c.unpack, unpack_value))) return false;
        if (phase != Phase::prepared && !unpack_possible) return false;
        if (phase == Phase::borrowed && (!consumer_registered || !reached(c.consumer, consumer_value))) return false;
        return true;
    }
    bool retire(const Completion& c) noexcept {
        if (!safe(c)) return false;
        *this = Lease{}; return true;
    }
    void poison() noexcept { poisoned = true; }
};
} // namespace nr_handoff
