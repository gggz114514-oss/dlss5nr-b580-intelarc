#include "nr_tail_probe.h"
#include <atomic>
#include <mutex>
#include <array>

namespace nrb {
namespace {
struct Slot {
    ID3D12GraphicsCommandList* list=nullptr;
    NRB_Route route=NRB_ROUTE_UNKNOWN;
    ID3D12Resource* color=nullptr;
    ID3D12Resource* output=nullptr;
    unsigned phase=0; // 0 before Evaluate, 1 in Evaluate, 2 after it
    unsigned actions=0,in_evaluate_actions=0,after_actions=0;
    unsigned pre_actions=0,color_barriers=0,output_barriers=0;
    unsigned output_read_barriers=0,actions_after_output_read=0;
    unsigned after_output_copies=0;
    unsigned color_before=0,color_after=0;
    unsigned output_before=0,output_after=0;
    unsigned color_flags=0,output_flags=0;
    unsigned color_subresource=0,output_subresource=0;
    unsigned after_dispatches=0,after_draws=0;
    unsigned last_dispatch_x=0,last_dispatch_y=0,last_dispatch_z=0;
    unsigned after_copy_from_color=0,after_copy_to_color=0,after_copy_to_output=0;
    unsigned copy_source_format=0,copy_dest_format=0;
    unsigned copy_source_width=0,copy_source_height=0;
    unsigned copy_dest_width=0,copy_dest_height=0;
    bool output_read_seen=false;
};
std::array<Slot,64> slots{};
std::mutex lock;
std::atomic<bool> active{true};
unsigned sampled=0,closed=0;
NRB_Route active_route=NRB_ROUTE_UNKNOWN;
Slot* find(ID3D12GraphicsCommandList* list) {
    for(auto& slot:slots) if(slot.list==list && list) return &slot;
    return nullptr;
}
}
bool tail_probe_active() {return active.load(std::memory_order_relaxed);}
void tail_probe_rearm(NRB_Route route) {
    if(route==NRB_ROUTE_UNKNOWN || route>NRB_ROUTE_XESS) return;
    std::lock_guard guard(lock);
    if(active_route==route) return;
    active_route=route;
    sampled=closed=0;
    active.store(true,std::memory_order_release);
}
void tail_reset(ID3D12GraphicsCommandList* list) {
    if(!tail_probe_active() || !list) return;
    std::lock_guard guard(lock);
    Slot* slot=find(list);
    if(!slot) for(auto& candidate:slots) if(!candidate.list) {slot=&candidate;break;}
    if(slot) {*slot={};slot->list=list;}
}
void tail_action(ID3D12GraphicsCommandList* list,bool dispatch,UINT x,UINT y,UINT z) {
    if(!tail_probe_active() || !list) return;
    std::lock_guard guard(lock);
    if(auto* slot=find(list)) {
        ++slot->actions;
        if(slot->phase==1) ++slot->in_evaluate_actions;
        if(slot->phase==2) {
            ++slot->after_actions;
            if(dispatch) {
                ++slot->after_dispatches;
                slot->last_dispatch_x=x;
                slot->last_dispatch_y=y;
                slot->last_dispatch_z=z;
            } else ++slot->after_draws;
            if(slot->output_read_seen) ++slot->actions_after_output_read;
        }
    }
}
void tail_copy(ID3D12GraphicsCommandList* list,ID3D12Resource* source,
               ID3D12Resource* destination,bool inspect_descriptors) {
    if(!tail_probe_active() || !list) return;
    std::lock_guard guard(lock);
    if(auto* slot=find(list)) {
        ++slot->actions;
        if(slot->phase==1) ++slot->in_evaluate_actions;
        if(slot->phase==2) {
            ++slot->after_actions;
            if(slot->output_read_seen) ++slot->actions_after_output_read;
            if(source && source==slot->output) ++slot->after_output_copies;
            if(source && source==slot->color) ++slot->after_copy_from_color;
            if(destination && destination==slot->color) ++slot->after_copy_to_color;
            if(destination && destination==slot->output) ++slot->after_copy_to_output;
            if(inspect_descriptors && source) {
                const auto desc=source->GetDesc();
                slot->copy_source_format=static_cast<unsigned>(desc.Format);
                slot->copy_source_width=static_cast<unsigned>(desc.Width);
                slot->copy_source_height=desc.Height;
            }
            if(inspect_descriptors && destination) {
                const auto desc=destination->GetDesc();
                slot->copy_dest_format=static_cast<unsigned>(desc.Format);
                slot->copy_dest_width=static_cast<unsigned>(desc.Width);
                slot->copy_dest_height=desc.Height;
            }
        }
    }
}
void tail_barriers(ID3D12GraphicsCommandList* list,UINT count,
                   const D3D12_RESOURCE_BARRIER* barriers) {
    if(!tail_probe_active() || !list || !barriers) return;
    std::lock_guard guard(lock);
    auto* slot=find(list);
    if(!slot || slot->phase!=2) return;
    for(UINT i=0;i<count;i++) {
        const auto& b=barriers[i];
        if(b.Type!=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION) continue;
        if(b.Transition.pResource==slot->color) {
            ++slot->color_barriers;
            slot->color_before=static_cast<unsigned>(b.Transition.StateBefore);
            slot->color_after=static_cast<unsigned>(b.Transition.StateAfter);
            slot->color_flags=static_cast<unsigned>(b.Flags);
            slot->color_subresource=b.Transition.Subresource;
        }
        if(b.Transition.pResource==slot->output) {
            ++slot->output_barriers;
            slot->output_before=static_cast<unsigned>(b.Transition.StateBefore);
            slot->output_after=static_cast<unsigned>(b.Transition.StateAfter);
            slot->output_flags=static_cast<unsigned>(b.Flags);
            slot->output_subresource=b.Transition.Subresource;
            // BEGIN_ONLY starts a split transition; the resource is readable
            // only after END_ONLY (or a non-split transition) is recorded.
            if(b.Flags!=D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY &&
               (b.Transition.StateAfter & (D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
                                           D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE))) {
                ++slot->output_read_barriers;
                slot->output_read_seen=true;
            }
        }
    }
}
void tail_before_evaluate(ID3D12GraphicsCommandList* list,
                          ID3D12Resource* color,ID3D12Resource* output,
                          NRB_Route route) {
    if(!tail_probe_active() || !list || !color || !output) return;
    std::lock_guard guard(lock);
    if(sampled>=8 || active_route!=route) return;
    auto* slot=find(list);
    if(!slot || slot->phase) return;
    ++sampled;
    slot->phase=1;slot->route=route;slot->color=color;slot->output=output;
    slot->pre_actions=slot->actions;
}
void tail_after_evaluate(ID3D12GraphicsCommandList* list) {
    if(!tail_probe_active() || !list) return;
    std::lock_guard guard(lock);
    if(auto* slot=find(list);slot && slot->phase==1) slot->phase=2;
}
TailSummary tail_close(ID3D12GraphicsCommandList* list) {
    TailSummary result{};
    if(!tail_probe_active() || !list) return result;
    std::lock_guard guard(lock);
    if(auto* slot=find(list);slot && slot->phase==2 &&
       slot->route==active_route) {
        result={true,slot->route,slot->pre_actions,slot->in_evaluate_actions,
                slot->after_actions,slot->color_barriers,
                slot->output_barriers,slot->output_read_barriers,
                slot->actions_after_output_read,slot->after_output_copies,
                slot->color_before,slot->color_after,
                slot->output_before,slot->output_after,
                slot->color_flags,slot->output_flags,
                slot->color_subresource,slot->output_subresource,
                slot->after_dispatches,slot->after_draws,
                slot->last_dispatch_x,slot->last_dispatch_y,
                slot->last_dispatch_z,
                slot->after_copy_from_color,slot->after_copy_to_color,
                slot->after_copy_to_output,
                slot->copy_source_format,slot->copy_dest_format,
                slot->copy_source_width,slot->copy_source_height,
                slot->copy_dest_width,slot->copy_dest_height};
        ++closed;
        slot->phase=0;
        if(closed>=8) active.store(false,std::memory_order_relaxed);
    }
    return result;
}
bool tail_safe_for_deferred_identity(const TailSummary& s) {
    if(!s.valid || s.after_color_barriers!=1 ||
       s.after_output_barriers!=1 || s.output_after!=192 ||
       s.color_flags!=D3D12_RESOURCE_BARRIER_FLAG_NONE ||
       s.output_flags!=D3D12_RESOURCE_BARRIER_FLAG_NONE ||
       s.color_subresource!=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES ||
       s.output_subresource!=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES ||
       s.after_output_copies || s.after_copy_to_color ||
       s.after_copy_to_output || s.actions_after_output_read) return false;
    if(s.route==NRB_ROUTE_DLSS)
        return s.after_actions==0 && s.color_before==128 &&
               s.color_after==8 && s.output_before==4;
    if(s.route==NRB_ROUTE_XESS)
        return s.after_actions==1 && s.after_copy_from_color==1 &&
               s.after_dispatches==0 && s.after_draws==0 &&
               s.copy_source_format==DXGI_FORMAT_R11G11B10_FLOAT &&
               s.copy_dest_format==DXGI_FORMAT_R11G11B10_FLOAT &&
               s.copy_source_width==1280 && s.copy_source_height==720 &&
               s.copy_dest_width==1280 && s.copy_dest_height==720 &&
               s.color_before==2240 && s.color_after==2112 &&
               s.output_before==8;
    return false;
}
}
