#pragma once
#include <d3d12.h>
#include <cstdint>
#include "nr_bridge.h"

namespace nrb {
struct TailSummary {
    bool valid=false;
    NRB_Route route=NRB_ROUTE_UNKNOWN;
    unsigned pre_actions=0;
    unsigned in_evaluate_actions=0;
    unsigned after_actions=0;
    unsigned after_color_barriers=0;
    unsigned after_output_barriers=0;
    unsigned output_read_barriers=0;
    unsigned actions_after_output_read=0;
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
};
bool tail_probe_active();
void tail_probe_rearm(NRB_Route route);
void tail_reset(ID3D12GraphicsCommandList* list);
void tail_action(ID3D12GraphicsCommandList* list,
                 bool dispatch=false,UINT x=0,UINT y=0,UINT z=0);
void tail_copy(ID3D12GraphicsCommandList* list,ID3D12Resource* source,
               ID3D12Resource* destination=nullptr,
               bool inspect_descriptors=false);
void tail_barriers(ID3D12GraphicsCommandList* list,UINT count,
                   const D3D12_RESOURCE_BARRIER* barriers);
void tail_before_evaluate(ID3D12GraphicsCommandList* list,
                          ID3D12Resource* color,ID3D12Resource* output,
                          NRB_Route route=NRB_ROUTE_DLSS);
void tail_after_evaluate(ID3D12GraphicsCommandList* list);
TailSummary tail_close(ID3D12GraphicsCommandList* list);
bool tail_safe_for_deferred_identity(const TailSummary& sample);
}
