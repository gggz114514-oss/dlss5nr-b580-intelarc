#pragma once
#include "nr_gpu_handoff_api.h"

namespace nrb::gpuhandoff {
void processor_changed(const NRB_Processor* processor);
void queue_changed();
bool prepared_bypass(const NRB_Processor& processor, ID3D12Device* device,
                     ID3D12CommandQueue* queue, bool cpu_motion_readback);
void previous_consumer_wait(bool success);
void process_failed();
class CallbackScope {
public:
    CallbackScope(const NRB_Processor&, const NRB_Frame&, ID3D12Fence*,
                  uint64_t value, bool prepared_cpu_waited);
    ~CallbackScope();
    CallbackScope(const CallbackScope&)=delete;
    CallbackScope& operator=(const CallbackScope&)=delete;
private:
    NRB_ProducerPoint point_{};
    const NRB_ProducerPoint* previous_=nullptr;
    bool installed_=false;
};
}
