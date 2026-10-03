#pragma once
#include <d3d12.h>
#include <cstdint>

namespace nrb {
enum IdentityIssue : uint32_t {
    IDENTITY_READY=0,
    IDENTITY_STATE_UNPROVEN=1u << 0,
    IDENTITY_LIFETIME_UNPROVEN=1u << 1,
    IDENTITY_LAYOUT_MISMATCH=1u << 2,
    IDENTITY_DEST_STATE_UNPROVEN=1u << 3,
    IDENTITY_CURRENT_WRITE_UNPROVEN=1u << 4,
};

// The ASI currently has neither proof. This contract is for a future
// route-specific source-state tracker / consumer completion callback, not a
// user switch. It must refer to this exact source and command-list generation.
struct IdentityBoundary {
    bool current_source_state_proven=false;
    bool destination_lifetime_proven=false;
    bool destination_copy_dest_state_proven=false;
    D3D12_RESOURCE_STATES current_source_state=D3D12_RESOURCE_STATE_COMMON;
    // Reset generation and write/entry sequence must come from a read-only
    // command-list tracker, not from a prior submission of the same pointer.
    uint64_t list_reset_generation=0;
    uint64_t last_color_write_generation=0;
    uint64_t last_color_write_sequence=0;
    uint64_t evaluate_entry_sequence=0;
    // Pinning/retirement owner for the future XeSS consumer submission.
    uint64_t consumer_retirement_tracking_id=0;
};

uint32_t plan_dlss_identity_copy(const D3D12_RESOURCE_DESC& source,
    const D3D12_RESOURCE_DESC& destination, const IdentityBoundary& boundary);

// Records a same-list, same-format copy and restores the source state. The
// destination must be created in COPY_DEST and retained until the actual XeSS
// consumer submission completes. Nothing in the current ASI calls this yet.
bool record_dlss_identity_copy(ID3D12GraphicsCommandList* list,
    ID3D12Resource* source, ID3D12Resource* destination,
    const IdentityBoundary& boundary);
}
