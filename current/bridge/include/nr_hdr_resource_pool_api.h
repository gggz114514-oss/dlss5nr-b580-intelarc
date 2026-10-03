#pragma once
#include "nr_bridge.h"

// Independent experimental control. Defaults OFF; affects new frames only.
// Texture bytes use D3D12 allocation info. Descriptor bytes are only an
// increment-size estimate, excluding opaque driver/heap bookkeeping.
struct NRB_HdrResourcePoolStats {
    uint32_t abi_size,enabled,capacity,shutdown;
    uint64_t acquires,hits,creates,disabled_fallbacks,busy_fallbacks,creation_failures;
    uint64_t retire_returns,unsubmitted_returns,quarantines,evictions,generation;
    uint64_t resident_slots,free_slots,leased_slots,quarantined_slots;
    uint64_t texture_bytes,peak_texture_bytes,descriptor_bytes_estimate,max_texture_bytes;
};
NRB_API int NRB_SetHdrResourcePoolEnabled(int enabled);
NRB_API int NRB_GetHdrResourcePoolStats(NRB_HdrResourcePoolStats* stats);
