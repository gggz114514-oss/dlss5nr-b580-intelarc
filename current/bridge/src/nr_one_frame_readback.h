#pragma once

#include "nr_sync_probe.h"

namespace nrb {
using ReadbackLog = void (*)(const char*);

// Diagnostic only: records one color copy on the producer/consumer list.
// This never enables NR or treats a queue observation as producer proof.
void one_frame_readback_record(ID3D12GraphicsCommandList* list,
                               ID3D12Resource* color,
                               const SyncObservation& sync,
                               ReadbackLog log);
void one_frame_readback_submitted(ID3D12CommandQueue* queue, UINT count,
                                  ID3D12CommandList* const* lists,
                                  const SubmitObservation* observations,
                                  UINT observation_count, ReadbackLog log);
}
