#include "nr_route_adapters.h"

namespace nrb {
bool extract_xess_input(NVSDK_NGX_Parameter* params,
                        ID3D12GraphicsCommandList* list, NRB_Frame& frame) {
    // OptiScaler's XeSS proxy has already converted NDC/pixel velocity units
    // into NGX MV_Scale. Preserve that scale for the downstream NR adapter;
    // never assume the RE8 unit-scale motion contract here.
    if (!extract_common_ngx(params,NRB_ROUTE_XESS,list,frame)) return false;
    frame.motion_scale_origin=NRB_MV_XESS_TO_NGX;
    return true;
}
}
