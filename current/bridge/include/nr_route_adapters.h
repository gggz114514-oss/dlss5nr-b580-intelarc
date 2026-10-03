#pragma once
#include "nr_bridge.h"

struct NVSDK_NGX_Parameter;

namespace nrb {
bool extract_common_ngx(NVSDK_NGX_Parameter*, NRB_Route,
                        ID3D12GraphicsCommandList*, NRB_Frame&);
bool extract_dlss_input(NVSDK_NGX_Parameter*, ID3D12GraphicsCommandList*, NRB_Frame&);
bool extract_fsr_input(NVSDK_NGX_Parameter*, ID3D12GraphicsCommandList*, NRB_Frame&);
bool extract_xess_input(NVSDK_NGX_Parameter*, ID3D12GraphicsCommandList*, NRB_Frame&);
}
