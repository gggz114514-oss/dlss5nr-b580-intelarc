#pragma once
#include <cstdint>
#include <limits>
namespace nrb::gpuhandoff {
struct ReuseObservation {
    const void* fence;
    uint64_t completed_value;
    bool wait_succeeded, device_healthy;
};
// Shared by actual deferred reuse and CPU fake fences. Registration or a ready
// fence never substitutes for actual success of the exact consumer fence.
template<class Signal,class Wait,class Retire>
bool retire_previous_consumer(const void* fence,uint64_t value,Signal signal,Wait wait,Retire retire) {
    try {
        if(!fence || !value || value==std::numeric_limits<uint64_t>::max() || !signal()) return false;
        const auto observed=wait();
        return observed.fence==fence && observed.wait_succeeded && observed.device_healthy &&
            observed.completed_value>=value && observed.completed_value!=std::numeric_limits<uint64_t>::max() &&
            retire();
    } catch(...) {return false;}
}
}
