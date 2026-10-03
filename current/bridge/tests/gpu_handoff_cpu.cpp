#include "nr_gpu_handoff.h"
#include "nr_gpu_handoff_reuse.h"
#include "nr_hdr_resource_pool.h"
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <stdexcept>

using namespace nrb::gpuhandoff;
unsigned checks=0;
void require(bool ok,const char* label) {
    ++checks;if(!ok) {std::fprintf(stderr,"FAIL %u: %s\n",checks,label);std::exit(1);}
}
int process(void*,const NRB_Frame*,NRB_Result*) {return 0;}
int other_process(void*,const NRB_Frame*,NRB_Result*) {return 0;}
int retire(void*,ID3D12Fence*,uint64_t) {return 0;}
template<class T> T* ptr(uint64_t v) {return reinterpret_cast<T*>(v);}
NRB_Frame frame() {
    NRB_Frame f{};f.abi_size=sizeof(f);f.device=ptr<ID3D12Device>(0x1000);
    f.queue=ptr<ID3D12CommandQueue>(0x2000);f.color=ptr<ID3D12Resource>(0x3000);
    f.motion=ptr<ID3D12Resource>(0x4000);f.frame_id=123;return f;
}
NRB_ProducerPoint point() {
    NRB_ProducerPoint p{};p.abi_size=sizeof(p);require(NRB_GetCurrentProducerPoint(&p)==1,"actual TLS scoped getter");return p;
}
NRB_GpuHandoffCapability capability() {
    NRB_GpuHandoffCapability c{};c.abi_size=sizeof(c);c.version=1;c.point=point();c.native_helper=0x5000;
    auto& i=c.native_info;i.abi_size=sizeof(i);i.version=i.enabled=i.healthy=i.reuse_safe_idle=1;
    i.queue_context_equal=i.in_order=i.native_luid_matched=i.max_active_frames=1;i.private_records=2;
    i.forward_import_per_wait=i.producer_cpu_wait_bypass_supported=i.producer_cpu_wait_bypass_configured=i.producer_fence_required=1;
    i.borrowed_sycl_queue=0x6000;i.native_context=0x7000;i.native_device=0x8000;
    i.d3d12_device=reinterpret_cast<uint64_t>(c.point.device);i.d3d12_queue=reinterpret_cast<uint64_t>(c.point.queue);
    return c;
}
NRB_GpuHandoffStats stats() {
    NRB_GpuHandoffStats s{};s.abi_size=sizeof(s);require(NRB_GetGpuHandoffStats(&s)==1,"cheap stats snapshot");return s;
}
int main() {
    static_assert(sizeof(NRB_Controls)==52 && sizeof(NRB_Processor)==32 && sizeof(NRB_Frame)==256 && sizeof(NRB_Result)==32);
    static_assert(sizeof(NRB_ProducerPoint)==120 && sizeof(NRB_NativeGpuHandoffInfo)==128 && sizeof(NRB_GpuHandoffCapability)==264 && sizeof(NRB_GpuHandoffStats)==192);
    NRB_Processor p{sizeof(p),ptr<void>(0x9000),process,retire};auto f=frame();auto* fence=ptr<ID3D12Fence>(0xa000);
    processor_changed(&p);
    require(!stats().requested && !stats().armed,"default OFF and unarmed");
    NRB_ProducerPoint out{};out.abi_size=sizeof(out);
    require(!NRB_GetCurrentProducerPoint(&out),"no invented fence outside callback");
    require(!NRB_ArmGpuHandoffCapability(nullptr) && !NRB_ClearGpuHandoffCapability(),"arm/clear outside callback rejected");
    require(!NRB_SetGpuHandoffRequested(2),"invalid request ABI value rejected");
    require(!prepared_bypass(p,f.device,f.queue,false),"unrequested CPU wait");
    NRB_GpuHandoffCapability stale{};
    {CallbackScope scope(p,f,fence,7,true);auto c=capability();
        require(c.point.fence==fence && c.point.value==7 && c.point.frame_id==f.frame_id &&
            c.point.device==f.device && c.point.queue==f.queue && c.point.prepared_cpu_waited,"actual prepared point retained");
        require(!NRB_ArmGpuHandoffCapability(&c),"OFF cannot arm");}
    require(!NRB_GetCurrentProducerPoint(&out),"TLS cleared on normal return");
    require(NRB_SetGpuHandoffRequested(1)==1 && !prepared_bypass(p,f.device,f.queue,false),"first enable frame keeps CPU wait");
    {CallbackScope scope(p,f,fence,7,false);auto c=capability();
        require(!NRB_ArmGpuHandoffCapability(&c),"first arm requires the confirmed CPU prepared boundary");}
    {CallbackScope scope(p,f,fence,7,true);auto c=capability();stale=c;
        require(NRB_ArmGpuHandoffCapability(&c)==1,"healthy exact callback/native proof arms");
        {CallbackScope nested(p,f,fence,7,true);require(!NRB_GetCurrentProducerPoint(&out),"nested callback cannot borrow outer source");}
        require(NRB_GetCurrentProducerPoint(&out)==1,"outer source restored after nested callback");}
    require(prepared_bypass(p,f.device,f.queue,false),"confirmed exact processor/device/queue bypass");
    require(!prepared_bypass(p,f.device,f.queue,true) && stats().armed,"motion CPU readback preserves legacy wait");
    try {CallbackScope scope(p,f,fence,7,false);throw std::runtime_error("fake callback exception");} catch(...) {}
    require(!NRB_GetCurrentProducerPoint(&out),"TLS cleared on exception");
    require(!NRB_ArmGpuHandoffCapability(&stale),"stale capability outside callback rejected");
    {CallbackScope scope(p,f,fence,7,true);auto c=capability();
        ++c.point.callback_cookie;require(!NRB_ArmGpuHandoffCapability(&c),"previous callback token rejected");
        c=capability();c.native_info.active=1;require(!NRB_ArmGpuHandoffCapability(&c),"in-flight native helper cannot arm");
        c=capability();c.native_info.producer_fence_required=0;require(!NRB_ArmGpuHandoffCapability(&c),"unconfigured producer dependency rejected");
        c=capability();c.native_info.d3d12_queue+=8;require(!NRB_ArmGpuHandoffCapability(&c),"wrong exact native queue rejected");
        c=capability();c.native_info.native_luid_matched=0;require(!NRB_ArmGpuHandoffCapability(&c),"unverified LUID rejected");
        c=capability();c.native_info.in_order=0;require(!NRB_ArmGpuHandoffCapability(&c),"out of order queue rejected");
        c=capability();c.native_info.queue_context_equal=0;require(!NRB_ArmGpuHandoffCapability(&c),"context mismatch rejected");
        c=capability();c.native_info.healthy=0;require(!NRB_ArmGpuHandoffCapability(&c),"unhealthy capability rejected");
        c=capability();require(NRB_ArmGpuHandoffCapability(&c)==1,"re-arm exact healthy scope");
        bool thread_get=true,thread_arm=true;
        std::thread worker([&] {NRB_ProducerPoint t{};t.abi_size=sizeof(t);
            thread_get=NRB_GetCurrentProducerPoint(&t);thread_arm=NRB_ArmGpuHandoffCapability(&c);});worker.join();
        require(!thread_get && !thread_arm,"cross-thread getter and arm rejected");
        c=capability();require(NRB_ArmGpuHandoffCapability(&c)==1,"owning thread re-arm");}
    require(!prepared_bypass(p,f.device,ptr<ID3D12CommandQueue>(0x2008),false) && !stats().armed,"queue change forces wait and disarms");
    {CallbackScope scope(p,f,fence,7,true);auto c=capability();require(NRB_ArmGpuHandoffCapability(&c)==1,"arm before OFF");}
    require(NRB_SetGpuHandoffRequested(0)==1 && !prepared_bypass(p,f.device,f.queue,false) && !stats().armed,"OFF restores wait immediately without native mutation");
    require(NRB_SetGpuHandoffRequested(1)==1,"request re-enable");
    {CallbackScope scope(p,f,fence,7,true);auto c=capability();require(NRB_ArmGpuHandoffCapability(&c)==1,"arm before registry replacement");}
    NRB_Processor replacement{sizeof(p),ptr<void>(0x9010),other_process,retire};processor_changed(&replacement);
    require(!prepared_bypass(p,f.device,f.queue,false) && !stats().armed,"processor replacement clears capability");
    {CallbackScope scope(p,f,fence,7,true);require(!NRB_GetCurrentProducerPoint(&out),"stale copied processor context rejected");}
    processor_changed(&p);
    {CallbackScope scope(p,f,fence,7,true);auto c=capability();queue_changed();
        require(!NRB_GetCurrentProducerPoint(&out) && !NRB_ArmGpuHandoffCapability(&c),"reinitialization invalidates an active scope");}
    {CallbackScope scope(p,f,nullptr,7,true);require(!NRB_GetCurrentProducerPoint(&out),"missing actual producer fence rejected");}
    {auto invalid=f;invalid.color=nullptr;CallbackScope scope(p,invalid,fence,7,true);require(!NRB_GetCurrentProducerPoint(&out),"invalid source rejected");}
    {CallbackScope scope(p,f,fence,7,true);auto c=capability();NRB_SetGpuHandoffRequested(0);NRB_SetGpuHandoffRequested(1);
        require(!NRB_ArmGpuHandoffCapability(&c),"request epoch prevents OFF/ON stale arm");}
    bool signal=true,waited=true,healthy=true,retired=false;uint64_t done=0;const void* observed=fence;
    auto reuse=[&] {retired=false;return retire_previous_consumer(fence,7,[&] {return signal;},
        [&] {return ReuseObservation{observed,done,waited,healthy};},[&] {retired=true;return true;});};
    require(!reuse() && !retired,"busy previous consumer cannot retire/reset owners");
    done=7;observed=ptr<void>(0xa008);require(!reuse() && !retired,"unrelated fence is not a reuse proof");
    observed=fence;signal=false;require(!reuse() && !retired,"failed consumer Signal rejects reuse");
    signal=true;waited=false;require(!reuse() && !retired,"timeout rejects reuse");
    waited=true;done=UINT64_MAX;require(!reuse() && !retired,"device-lost value rejects reuse");
    done=7;healthy=false;require(!reuse() && !retired,"device loss rejects reuse");
    healthy=true;require(reuse() && retired,"actual previous consumer completion permits retirement");
    require(!retire_previous_consumer(fence,7,[] {return true;},[&] {return ReuseObservation{fence,7,true,true};},
        []()->bool {throw std::runtime_error("retire failure");}),"throwing retire retains ownership");
    previous_consumer_wait(true);previous_consumer_wait(false);require(stats().previous_consumer_waits==2 && stats().previous_consumer_wait_failures==1,"reuse counter distinct from prepared bypass");
    nrb::HdrLeaseProof proof;proof.prepare_recorded();proof.private_records_closed();proof.prepare_submitted();
    proof.prepare_gpu_ordered(true);proof.processor_begin();proof.processor_ready(true);
    require(!proof.poisoned() && !proof.can_rewrite_descriptors() && proof.can_record_private_composite(),"GPU order never permits original descriptor rewrite");
    proof.composite_recorded(true);proof.composite_closed();proof.composite_submitted();proof.deferred_submitted();
    require(!proof.terminal_safe(),"prepared GPU order is not final consumer completion");
    proof.completion_signal(true);proof.consumer_completed(1,true);proof.processor_retirement(true);
    require(proof.terminal_safe(),"exact final consume + retire proof still required");
    NRB_SetGpuHandoffRequested(0);
    std::printf("PASS: %u CPU protocol checks; frozen ABI; no GPU/ASI load\n",checks);
}
