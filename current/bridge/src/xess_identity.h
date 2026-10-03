#pragma once
#include <d3d12.h>

namespace nrb {
// Diagnostic only: copy the current game's color to an identical private
// texture before OptiScaler records XeSS on the same command list.
ID3D12Resource* record_ngx_identity(ID3D12GraphicsCommandList* list,
                                    ID3D12Resource* source);
unsigned xess_identity_submitted(ID3D12CommandQueue* queue, UINT count,
                                 ID3D12CommandList* const* lists);
unsigned xess_identity_recorded();
unsigned xess_identity_completed();
}
