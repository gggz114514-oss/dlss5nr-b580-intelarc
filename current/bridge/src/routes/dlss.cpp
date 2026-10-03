#include "nr_route_adapters.h"

namespace nrb {
bool extract_dlss_input(NVSDK_NGX_Parameter* params,
                        ID3D12GraphicsCommandList* list, NRB_Frame& frame) {
    // OptiScaler's DLSS entry already presents the standard NGX resource and
    // render-subrect keys. No Cyberpunk setting is interpreted here.
    if (!extract_common_ngx(params,NRB_ROUTE_DLSS,list,frame)) return false;
    frame.motion_scale_origin=NRB_MV_DLSS_NGX;
    return true;
}
}
