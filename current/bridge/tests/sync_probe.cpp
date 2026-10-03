#define CINTERFACE
#include "nr_sync_probe.h"
#include <cstdlib>
#include <cstddef>
#include <iostream>
#define CHECK(x) do { if (!(x)) { std::cerr << "CHECK failed: " #x << "\n"; std::abort(); } } while(false)

// This audits the actual Windows SDK header used by this build, not an
// unverified diagram of the COM method order.
static_assert(offsetof(ID3D12GraphicsCommandListVtbl, Close)==9*sizeof(void*));
static_assert(offsetof(ID3D12GraphicsCommandListVtbl, Reset)==10*sizeof(void*));
static_assert(offsetof(ID3D12GraphicsCommandListVtbl, CopyTextureRegion)==16*sizeof(void*));
static_assert(offsetof(ID3D12GraphicsCommandListVtbl, CopyResource)==17*sizeof(void*));
static_assert(offsetof(ID3D12GraphicsCommandListVtbl, ResourceBarrier)==26*sizeof(void*));
static_assert(offsetof(ID3D12CommandQueueVtbl, ExecuteCommandLists)==10*sizeof(void*));

int main() {
    nrb::SyncProbe probe;
    auto* list=reinterpret_cast<ID3D12GraphicsCommandList*>(0x1000);
    auto* color=reinterpret_cast<ID3D12Resource*>(0x2000);
    auto* motion=reinterpret_cast<ID3D12Resource*>(0x3000);
    auto first=probe.evaluate(list,color,motion);
    CHECK(!first.reset_observed && first.incomplete);
    probe.reset(list,false);
    CHECK(!probe.evaluate(list,color,motion).reset_observed);
    probe.reset(list,true);
    D3D12_RESOURCE_BARRIER barriers[2]{};
    for(auto& b:barriers) {
        b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_DEST;
        b.Transition.StateAfter=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }
    barriers[0].Transition.pResource=color;
    barriers[1].Transition.pResource=motion;
    probe.barriers(list,2,barriers);
    auto observed=probe.evaluate(list,color,motion);
    CHECK(observed.reset_observed && !observed.incomplete && !observed.closed);
    CHECK(observed.color.seen && observed.color.full_non_pixel_read);
    CHECK(observed.motion.seen && observed.motion.full_non_pixel_read);
    CHECK(observed.color.sequence<observed.evaluate_sequence);
    CHECK(observed.submit_batch_before==0);
    probe.close(list,true);
    auto submitted=probe.submit(reinterpret_cast<ID3D12CommandList*>(list),7);
    CHECK(submitted.correlated && submitted.batch==7 && submitted.generation==observed.generation);
    CHECK(submitted.evaluations==1 && submitted.latest_evaluate_sequence==observed.evaluate_sequence);
    CHECK(probe.evaluate(list,color,motion).submit_batch_before==7);
    probe.reset(list,true);
    auto next=probe.evaluate(list,color,motion);
    CHECK(next.generation!=observed.generation && !next.color.seen && !next.motion.seen);
    CHECK(next.submit_batch_before==0 && !next.closed);
    barriers[0].Flags=D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
    probe.barriers(list,1,barriers);
    CHECK(!probe.evaluate(list,color,motion).color.full_non_pixel_read);
    barriers[0].Flags=D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barriers[0].Transition.Subresource=0;
    probe.barriers(list,1,barriers);
    CHECK(!probe.evaluate(list,color,motion).color.full_non_pixel_read);
    barriers[0].Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    probe.barriers(list,65,barriers); // no dereference after the bound check
    auto oversized=probe.evaluate(list,color,motion);
    CHECK(oversized.incomplete && oversized.oversized_barrier);
    probe.reset(list,true);
    for(int i=0;i<270;i++) probe.barriers(list,1,barriers);
    auto wrapped=probe.evaluate(list,color,motion);
    CHECK(wrapped.incomplete && !wrapped.oversized_barrier); // ring lost a transition
    std::cout << "SDK vtable indices and bounded sync metadata tracker passed\n";
}
