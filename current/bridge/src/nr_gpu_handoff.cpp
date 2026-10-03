#include "nr_gpu_handoff.h"
#include <mutex>
#include <thread>
#include <cstring>
#include <limits>
#ifdef _WIN32
#include <windows.h>
#endif

namespace nrb::gpuhandoff {
namespace {
thread_local const NRB_ProducerPoint* current=nullptr;
uint64_t thread_key() {
#ifdef _WIN32
    return GetCurrentThreadId();
#else
    return std::hash<std::thread::id>{}(std::this_thread::get_id());
#endif
}
bool same_processor(const NRB_Processor& a,const NRB_Processor& b) {
    return a.abi_size==sizeof(a) && b.abi_size==sizeof(b) && a.context==b.context &&
        a.process==b.process && a.retire==b.retire && a.process && a.retire;
}
struct Gate {
    std::mutex mutex;
    NRB_Processor processor{}, armed_processor{};
    NRB_GpuHandoffStats stats{sizeof(NRB_GpuHandoffStats),1};
    std::thread::id owner{};
    uint64_t owner_transfer_function=0;
    uint64_t cookie=0;
    void clear(unsigned reason) {
        if(stats.armed) ++stats.disarms;
        stats.armed=stats.cap_healthy=0;stats.pending=stats.requested;
        stats.last_rejection=reason;
        stats.native_helper=stats.device=stats.queue=stats.native_context=0;
        stats.borrowed_sycl_queue=stats.owner_thread=0;
        armed_processor={};owner={};owner_transfer_function=0;
    }
} gate;
bool scoped_valid() {
    return current && current->registry_epoch==gate.stats.registry_epoch &&
        current->processor_context==gate.processor.context &&
        current->processor_process==gate.processor.process &&
        current->owner_thread==thread_key();
}
bool exact_point(const NRB_ProducerPoint& a,const NRB_ProducerPoint& b) {
    return a.abi_size==sizeof(a) && a.version==1 && a.device==b.device &&
        a.queue==b.queue && a.fence==b.fence && a.color==b.color && a.motion==b.motion &&
        a.processor_context==b.processor_context && a.processor_process==b.processor_process &&
        a.value==b.value && a.frame_id==b.frame_id && a.registry_epoch==b.registry_epoch &&
        a.request_epoch==b.request_epoch && a.callback_cookie==b.callback_cookie &&
        a.owner_thread==b.owner_thread && a.prepared_cpu_waited==b.prepared_cpu_waited && !a.reserved;
}
}
void processor_changed(const NRB_Processor* p) {
    std::lock_guard lock(gate.mutex);
    ++gate.stats.registry_epoch;gate.processor=p?*p:NRB_Processor{};gate.clear(1);
}
void queue_changed() {
    std::lock_guard lock(gate.mutex);
    ++gate.stats.registry_epoch;gate.clear(2);
}
bool prepared_bypass(const NRB_Processor& p,ID3D12Device* device,
                     ID3D12CommandQueue* queue,bool motion) {
    std::lock_guard lock(gate.mutex);
    const bool identity=gate.stats.armed && same_processor(p,gate.processor) &&
        same_processor(p,gate.armed_processor) &&
        (gate.owner_transfer_function || gate.owner==std::this_thread::get_id()) &&
        gate.stats.device==reinterpret_cast<uint64_t>(device) &&
        gate.stats.queue==reinterpret_cast<uint64_t>(queue);
    if(gate.stats.armed && !identity) {++gate.stats.identity_mismatch_waits;gate.clear(3);}
    const bool bypass=gate.stats.requested && identity && gate.stats.cap_healthy && !motion;
    if(bypass) ++gate.stats.prepared_bypasses;
    else {++gate.stats.prepared_cpu_waits;if(motion) ++gate.stats.motion_readback_waits;}
    return bypass;
}
void previous_consumer_wait(bool success) {
    std::lock_guard lock(gate.mutex);
    ++gate.stats.previous_consumer_waits;
    if(!success) {++gate.stats.previous_consumer_wait_failures;gate.clear(4);}
}
void process_failed() {
    std::lock_guard lock(gate.mutex);++gate.stats.process_failures;gate.clear(5);
}
CallbackScope::CallbackScope(const NRB_Processor& p,const NRB_Frame& f,
                            ID3D12Fence* fence,uint64_t value,bool waited) {
    std::lock_guard lock(gate.mutex);
    previous_=current;current=nullptr;installed_=true;
    if(previous_ || !same_processor(p,gate.processor) || f.abi_size!=sizeof(f) ||
       !f.device || !f.queue || !f.color || !f.motion || !f.frame_id || !fence ||
       !value || value==std::numeric_limits<uint64_t>::max()) return;
    point_.abi_size=sizeof(point_);point_.version=1;
    point_.device=f.device;point_.queue=f.queue;point_.fence=fence;
    point_.color=f.color;point_.motion=f.motion;point_.processor_context=p.context;
    point_.processor_process=p.process;point_.value=value;point_.frame_id=f.frame_id;
    point_.registry_epoch=gate.stats.registry_epoch;point_.request_epoch=gate.stats.request_epoch;
    point_.callback_cookie=++gate.cookie;point_.owner_thread=thread_key();
    point_.prepared_cpu_waited=waited;current=&point_;installed_=true;
}
CallbackScope::~CallbackScope() {if(installed_) current=previous_;}
}

NRB_API int NRB_SetGpuHandoffRequested(int requested) {
    if(requested!=0 && requested!=1) return 0;
    auto& g=nrb::gpuhandoff::gate;std::lock_guard lock(g.mutex);
    if(g.stats.requested!=uint32_t(requested)) {
        g.stats.requested=requested;++g.stats.request_epoch;g.clear(6);
    }
    return 1;
}
NRB_API int NRB_GetGpuHandoffStats(NRB_GpuHandoffStats* stats) {
    if(!stats || stats->abi_size!=sizeof(*stats)) return 0;
    auto& g=nrb::gpuhandoff::gate;std::lock_guard lock(g.mutex);*stats=g.stats;return 1;
}
NRB_API int NRB_GetCurrentProducerPoint(NRB_ProducerPoint* point) {
    auto& g=nrb::gpuhandoff::gate;std::lock_guard lock(g.mutex);++g.stats.source_queries;
    if(!point || point->abi_size!=sizeof(*point) || !nrb::gpuhandoff::scoped_valid()) {
        ++g.stats.source_rejections;return 0;
    }
    *point=*nrb::gpuhandoff::current;return 1;
}
NRB_API int NRB_ClearGpuHandoffCapability() {
    auto& g=nrb::gpuhandoff::gate;std::lock_guard lock(g.mutex);
    if(!nrb::gpuhandoff::scoped_valid()) return 0;
    g.clear(7);return 1;
}
static int arm_gpu_handoff_capability(const NRB_GpuHandoffCapability* c,
                                     bool serial,uint64_t owner_transfer_function) {
    using namespace nrb::gpuhandoff;
    auto& g=gate;std::lock_guard lock(g.mutex);
    const auto reject=[&]() {++g.stats.arm_rejections;g.clear(8);return 0;};
    if((serial && !owner_transfer_function) || !c || c->abi_size!=sizeof(*c) || c->version!=1 || !scoped_valid() ||
       !g.stats.requested || current->request_epoch!=g.stats.request_epoch ||
       !exact_point(c->point,*current) || !c->native_helper) return reject();
    // The optional serial contract may only be minted/requalified on a
    // CPU-prepared frame. A legacy arm or a replaced helper/context cannot
    // grant permission to upgrade an already committed bypass.
    if((serial || !g.stats.armed) && !current->prepared_cpu_waited) return reject();
    const auto& i=c->native_info;
    if(i.abi_size!=sizeof(i) || i.version!=1 || i.enabled!=1 || i.healthy!=1 ||
       i.reuse_safe_idle!=1 || i.active || i.poisoned || i.queue_context_equal!=1 ||
       i.in_order!=1 || i.native_luid_matched!=1 || i.max_active_frames!=1 ||
       i.private_records!=2 || i.forward_import_per_wait!=1 ||
       i.producer_cpu_wait_bypass_supported!=1 || i.producer_cpu_wait_bypass_configured!=1 ||
       i.producer_fence_required!=1 || !i.borrowed_sycl_queue || !i.native_context || !i.native_device ||
       i.d3d12_device!=reinterpret_cast<uint64_t>(current->device) ||
       i.d3d12_queue!=reinterpret_cast<uint64_t>(current->queue) ||
       i.retained_producer_fence || i.retained_producer_value) return reject();
    g.stats.armed=g.stats.cap_healthy=1;g.stats.pending=0;g.stats.last_rejection=0;++g.stats.arms;
    g.armed_processor=g.processor;g.owner=std::this_thread::get_id();
    g.owner_transfer_function=serial?owner_transfer_function:0;
    g.stats.native_helper=c->native_helper;g.stats.device=i.d3d12_device;g.stats.queue=i.d3d12_queue;
    g.stats.native_context=i.native_context;g.stats.borrowed_sycl_queue=i.borrowed_sycl_queue;
    g.stats.owner_thread=current->owner_thread;return 1;
}
NRB_API int NRB_ArmGpuHandoffCapability(const NRB_GpuHandoffCapability* c) {
    return arm_gpu_handoff_capability(c,false,0);
}
NRB_API int NRB_ArmSerialGpuHandoffCapability(const NRB_GpuHandoffCapability* c,
                                           uint64_t owner_transfer_function) {
    return arm_gpu_handoff_capability(c,true,owner_transfer_function);
}
NRB_API int NRB_GetSerialGpuHandoffArmed() {
    auto& g=nrb::gpuhandoff::gate;std::lock_guard lock(g.mutex);
    return g.stats.armed && g.stats.cap_healthy && g.owner_transfer_function;
}
