#include "nr_sync_probe.h"
#include <algorithm>

namespace nrb {
SyncProbe::SyncProbe(ProofIdentityOps ops,bool enabled) noexcept:
    identities_(ops),color_state_enabled_(enabled) {}
SyncProbe::~SyncProbe() {for(auto& list:lists_) clear_proof(list);}
ProofAcquire SyncProbe::acquire(void* object,bool resource,ProofIdentity& identity) noexcept {
    if(!identities_.acquire || !identities_.release || !object) return ProofAcquire::unknown;
    const auto result=identities_.acquire(identities_.context,object,resource,identity);
    if(result==ProofAcquire::acquired && identity.cookie && identity.lease) return result;
    release(identity);return result==ProofAcquire::not_candidate?result:ProofAcquire::unknown;
}
void SyncProbe::release(ProofIdentity& identity) noexcept {
    if(identity.lease && identities_.release) identities_.release(identities_.context,identity);
    identity={};
}
void SyncProbe::clear_proof(List& list) noexcept {
    for(auto& color:list.colors) {release(color.identity);color={};}
    release(list.identity);
}
void SyncProbe::invalidate_all(ColorProofReason reason) noexcept {
    for(auto& list:lists_) if(list.generation && !list.closed) list.rejected=reason;
}
void SyncProbe::unknown_recording(ID3D12GraphicsCommandList* list) {
    if(!color_state_enabled_) return;
    std::lock_guard lock(mutex_);
    if(auto* slot=find(list)) slot->rejected=ColorProofReason::implementation;
    else invalidate_all(ColorProofReason::implementation);
}
void SyncProbe::observe_color_barrier(List* own,const D3D12_RESOURCE_BARRIER& b,uint64_t event_sequence) {
    if(!color_state_enabled_) return;
    if(own) ++own->local_ordinal;
    // Aliasing synchronizes overlapping memory usages; it is not an access-state
    // transition and cannot rewrite every list's recorded StateAfter. The validated
    // direct DLSS scope supplies the current live resource; actual copies and
    // prefix/private/suffix queue ordering remain checked separately.
    if(b.Type==D3D12_RESOURCE_BARRIER_TYPE_ALIASING) return;
    const bool transition=b.Type==D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    if(!transition && b.Type!=D3D12_RESOURCE_BARRIER_TYPE_UAV) {
        invalidate_all(ColorProofReason::unknown_event);return;
    }
    auto* resource=transition?b.Transition.pResource:b.UAV.pResource;
    if(!resource) {
        if(transition) invalidate_all(ColorProofReason::unknown_event);
        return; // a null UAV barrier asserts no new resource state
    }
    ProofIdentity identity{};const auto acquired=acquire(resource,true,identity);
    if(acquired==ProofAcquire::unknown) {invalidate_all(ColorProofReason::identity);return;}
    // This observer describes each list's recorded local declaration. Other
    // lists may be recorded in parallel and cannot overwrite that declaration
    // by CPU arrival order. GPU execution ordering remains the reviewed direct
    // route's existing prefix/private/suffix queue and fence contract.
    for(auto& list:lists_) if(list.generation && !list.closed) for(auto& c:list.colors) {
        if(c.identity.cookie && c.object==resource &&
                (acquired!=ProofAcquire::acquired || c.identity.cookie!=identity.cookie))
            list.rejected=ColorProofReason::identity;
    }
    if(acquired!=ProofAcquire::acquired) return;
    if(!transition || !own || own->closed || own->rejected!=ColorProofReason::ready || !own->identity.cookie) {
        release(identity);return;
    }
    ColorState* color=nullptr;
    for(auto& c:own->colors) if(c.identity.cookie==identity.cookie) {color=&c;break;}
    if(!color) {
        for(auto& c:own->colors) if(!c.identity.cookie) {color=&c;break;}
        if(!color) {own->rejected=ColorProofReason::capacity;release(identity);return;}
        color->object=resource;color->identity=identity;identity={};
    }
    release(identity);
    const bool plain=b.Flags==D3D12_RESOURCE_BARRIER_FLAG_NONE &&
        b.Transition.Subresource==D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    if(!plain) color->rejected=ColorProofReason::partial_split; // sticky until successful Reset
    else if(color->transition.seen && color->rejected==ColorProofReason::ready &&
            color->transition.after!=b.Transition.StateBefore)
        color->rejected=ColorProofReason::state_chain;
    auto& t=color->transition;t.seen=true;t.sequence=event_sequence;
    t.before=b.Transition.StateBefore;t.after=b.Transition.StateAfter;t.flags=b.Flags;
    t.subresource=b.Transition.Subresource;t.full_non_pixel_read=plain &&
        (t.after&D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)!=0;
    color->local_ordinal=own->local_ordinal;
}
ColorStateObservation SyncProbe::color_state(List* own,ID3D12Resource* resource) {
    ColorStateObservation out{};out.enabled=true;
    if(!own || !own->generation) {out.reason=ColorProofReason::no_reset;return out;}
    out.generation=own->generation;
    if(own->closed) {out.reason=ColorProofReason::closed;return out;}
    if(own->rejected!=ColorProofReason::ready) {out.reason=own->rejected;return out;}
    ProofIdentity list_identity{},resource_identity{};
    const auto list_acquired=acquire(own->object,false,list_identity);
    const bool list_matches=list_acquired==ProofAcquire::acquired &&
        list_identity.cookie==own->identity.cookie;
    release(list_identity);
    if(!list_matches) {own->rejected=out.reason=ColorProofReason::identity;return out;}
    const auto resource_acquired=acquire(resource,true,resource_identity);
    if(resource_acquired!=ProofAcquire::acquired) {
        own->rejected=out.reason=ColorProofReason::identity;return out;
    }
    ColorState* color=nullptr;
    for(auto& c:own->colors) {
        if(c.object==resource && c.identity.cookie && c.identity.cookie!=resource_identity.cookie)
            own->rejected=ColorProofReason::identity;
        if(c.identity.cookie==resource_identity.cookie) color=&c;
    }
    release(resource_identity);
    if(own->rejected!=ColorProofReason::ready) {out.reason=own->rejected;return out;}
    if(!color) {out.reason=ColorProofReason::identity;return out;}
    out.resource_cookie=color->identity.cookie;out.local_ordinal=color->local_ordinal;
    out.transition=color->transition;out.reason=color->rejected;
    if(out.reason!=ColorProofReason::ready) return out;
    out.known=out.transition.seen;
    if(out.transition.after!=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)
        out.reason=ColorProofReason::not_psr;
    return out;
}
SyncProbe::List* SyncProbe::find(ID3D12CommandList* list) {
    for(auto& item:lists_) if(item.object==list && list) return &item;
    return nullptr;
}
SyncProbe::List& SyncProbe::allocate(ID3D12CommandList* list) {
    if(auto* item=find(list)) return *item;
    auto* slot=&*std::min_element(lists_.begin(),lists_.end(),
        [](const List& a,const List& b){return a.touched<b.touched;});
    clear_proof(*slot);*slot={}; slot->object=list; slot->touched=++sequence_;
    return *slot;
}
void SyncProbe::reset(ID3D12GraphicsCommandList* list,bool succeeded) {
    if(!list) return;
    if(!succeeded) {
        if(color_state_enabled_) {std::lock_guard lock(mutex_);if(auto* slot=find(list)) slot->rejected=ColorProofReason::lifecycle;}
        return;
    }
    std::lock_guard lock(mutex_);
    auto& slot=allocate(list);
    clear_proof(slot);slot.colors={};slot.local_ordinal=0;slot.rejected=ColorProofReason::ready;
    if(color_state_enabled_ && acquire(list,false,slot.identity)!=ProofAcquire::acquired)
        slot.rejected=ColorProofReason::identity;
    slot.generation=++generation_;
    slot.reset_sequence=slot.touched=++sequence_;
    slot.submit_batch=0; slot.closed=false; slot.incomplete=false;
    slot.oversized_barrier=false;
}
void SyncProbe::close(ID3D12GraphicsCommandList* list,bool succeeded) {
    if(!list) return;
    std::lock_guard lock(mutex_);
    if(auto* slot=find(list); slot && slot->generation) {
        if(succeeded) {slot->closed=true;slot->touched=++sequence_;clear_proof(*slot);}
        else if(color_state_enabled_) slot->rejected=ColorProofReason::lifecycle;
    }
}
void SyncProbe::barriers(ID3D12GraphicsCommandList* list,UINT count,
                         const D3D12_RESOURCE_BARRIER* barriers) {
    if(!list || !count) return;
    std::lock_guard lock(mutex_);
    auto* slot=find(list);
    // Advance the original diagnostic sequence exactly as before; proof state
    // uses its own ordinal and cannot expire due to another list's sequence.
    if(color_state_enabled_ && (!barriers || count>64)) invalidate_all(ColorProofReason::unknown_event);
    if(!barriers) return;
    if(!slot || !slot->generation) {
        if(color_state_enabled_ && count<=64) for(UINT i=0;i<count;++i)
            observe_color_barrier(nullptr,barriers[i],sequence_);
        return; // list was created before our hooks; no legacy diagnostic ordinal
    }
    slot->touched=++sequence_;
    if(count>64) {slot->incomplete=true;slot->oversized_barrier=true;return;}
    for(UINT i=0;i<count;i++) {
        const auto& b=barriers[i];
        if(b.Type==D3D12_RESOURCE_BARRIER_TYPE_TRANSITION) {
            const auto index=(++sequence_)%transitions_.size();
            transitions_[index]={list,b.Transition.pResource,slot->generation,sequence_,
                b.Transition.StateBefore,b.Transition.StateAfter,b.Flags,
                b.Transition.Subresource};
        }
        observe_color_barrier(slot,b,sequence_);
    }
}
TransitionObservation SyncProbe::latest(const List& slot,ID3D12Resource* resource) const {
    TransitionObservation result{};
    if(!resource) return result;
    for(const auto& t:transitions_) if(t.list==slot.object &&
        t.generation==slot.generation && t.resource==resource &&
        t.sequence>result.sequence) {
        result.seen=true; result.sequence=t.sequence;
        result.before=t.before; result.after=t.after;
        result.flags=t.flags; result.subresource=t.subresource;
        result.full_non_pixel_read=t.flags==D3D12_RESOURCE_BARRIER_FLAG_NONE &&
            t.subresource==D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES &&
            (t.after&D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)!=0;
    }
    return result;
}
SyncObservation SyncProbe::evaluate(ID3D12GraphicsCommandList* list,
                                   ID3D12Resource* color,ID3D12Resource* motion,bool dlss_scope) {
    SyncObservation result{};
    if(!list) return result;
    std::lock_guard lock(mutex_);
    result.evaluate_sequence=++sequence_;
    auto* slot=find(list);
    if(color_state_enabled_ && dlss_scope) result.color_state=color_state(slot,color);
    if(!slot || !slot->generation) {result.incomplete=true;return result;}
    slot->touched=sequence_;
    result.generation=slot->generation;
    result.reset_observed=true; result.closed=slot->closed;
    // A fixed ring may have overwritten an earlier transition in this same
    // recording generation; absence then must not be interpreted as absence
    // of a barrier. Conservatively mark the entire observation incomplete.
    result.oversized_barrier=slot->oversized_barrier;
    result.incomplete=slot->incomplete ||
        sequence_-slot->reset_sequence>=transitions_.size();
    result.submit_batch_before=slot->submit_batch;
    result.color=latest(*slot,color);
    if(result.color_state.enabled && result.color_state.transition.seen)
        result.color=result.color_state.transition; // global age remains diagnostic
    result.motion=latest(*slot,motion);
    pending_[sequence_%pending_.size()]={list,slot->generation,sequence_};
    return result;
}
SubmitObservation SyncProbe::submit(ID3D12CommandList* list,uint64_t batch) {
    SubmitObservation result{};
    if(!list || !batch) return result;
    std::lock_guard lock(mutex_);
    auto* slot=find(list);
    if(!slot || !slot->generation) return result;
    slot->submit_batch=batch;slot->touched=++sequence_;
    result.generation=slot->generation;result.batch=batch;
    for(auto& p:pending_) if(p.list==list && p.generation==slot->generation) {
        ++result.evaluations;
        result.latest_evaluate_sequence=std::max(result.latest_evaluate_sequence,p.evaluate_sequence);
        p={};
    }
    result.correlated=result.evaluations!=0;
    return result;
}
}
