#pragma once
#include "nr_bridge.h"

struct NVSDK_NGX_Parameter;

namespace nrb {
bool validate_controls(const NRB_Controls& controls);
bool extract_ngx(NVSDK_NGX_Parameter* params, NRB_Route route,
                 ID3D12GraphicsCommandList* list, NRB_Frame& frame);
NRB_Status validate_metadata(const NRB_Frame& frame);
NRB_Status validate_resources(const NRB_Frame& frame);
bool validate_result(const NRB_Frame& frame, const NRB_Result& result);
}
