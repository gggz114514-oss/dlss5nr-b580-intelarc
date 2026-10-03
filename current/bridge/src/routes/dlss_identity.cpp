#include "nr_identity.h"

namespace nrb {
uint32_t plan_dlss_identity_copy(const D3D12_RESOURCE_DESC& source,
    const D3D12_RESOURCE_DESC& destination, const IdentityBoundary& boundary) {
    uint32_t issues=IDENTITY_READY;
    if (!boundary.current_source_state_proven) issues|=IDENTITY_STATE_UNPROVEN;
    if (!boundary.destination_lifetime_proven ||
        !boundary.consumer_retirement_tracking_id)
        issues|=IDENTITY_LIFETIME_UNPROVEN;
    if (!boundary.destination_copy_dest_state_proven) issues|=IDENTITY_DEST_STATE_UNPROVEN;
    if (!boundary.list_reset_generation ||
        boundary.last_color_write_generation!=boundary.list_reset_generation ||
        !boundary.last_color_write_sequence ||
        boundary.last_color_write_sequence>=boundary.evaluate_entry_sequence)
        issues|=IDENTITY_CURRENT_WRITE_UNPROVEN;
    if (source.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        source.Width==0 || source.Height==0 || source.MipLevels!=1 ||
        source.Format==DXGI_FORMAT_UNKNOWN ||
        (source.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0 ||
        destination.Dimension!=source.Dimension ||
        source.Width!=destination.Width || source.Height!=destination.Height ||
        source.DepthOrArraySize!=destination.DepthOrArraySize ||
        source.MipLevels!=destination.MipLevels ||
        source.SampleDesc.Count!=destination.SampleDesc.Count ||
        source.SampleDesc.Quality!=destination.SampleDesc.Quality ||
        source.Format!=destination.Format)
        issues|=IDENTITY_LAYOUT_MISMATCH;
    return issues;
}

bool record_dlss_identity_copy(ID3D12GraphicsCommandList* list,
    ID3D12Resource* source, ID3D12Resource* destination,
    const IdentityBoundary& boundary) {
    if (!list || !source || !destination || source==destination ||
        plan_dlss_identity_copy(source->GetDesc(),destination->GetDesc(),boundary)!=IDENTITY_READY)
        return false;
    ID3D12Device *list_device=nullptr,*source_device=nullptr,*dest_device=nullptr;
    const bool same=SUCCEEDED(list->GetDevice(IID_PPV_ARGS(&list_device))) &&
        SUCCEEDED(source->GetDevice(IID_PPV_ARGS(&source_device))) &&
        SUCCEEDED(destination->GetDevice(IID_PPV_ARGS(&dest_device))) &&
        list_device==source_device && list_device==dest_device;
    if(list_device)list_device->Release();
    if(source_device)source_device->Release();
    if(dest_device)dest_device->Release();
    if(!same)return false;
    const auto barrier=[](ID3D12Resource* resource,D3D12_RESOURCE_STATES before,
                          D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER result{};
        result.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        result.Transition.pResource=resource;
        result.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        result.Transition.StateBefore=before;
        result.Transition.StateAfter=after;
        return result;
    };
    if(boundary.current_source_state!=D3D12_RESOURCE_STATE_COPY_SOURCE) {
        auto b=barrier(source,boundary.current_source_state,D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1,&b);
    }
    list->CopyResource(destination,source);
    D3D12_RESOURCE_BARRIER after[2]{};
    UINT count=0;
    if(boundary.current_source_state!=D3D12_RESOURCE_STATE_COPY_SOURCE)
        after[count++]=barrier(source,D3D12_RESOURCE_STATE_COPY_SOURCE,
                               boundary.current_source_state);
    after[count++]=barrier(destination,D3D12_RESOURCE_STATE_COPY_DEST,
                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    list->ResourceBarrier(count,after);
    return true;
}
}
