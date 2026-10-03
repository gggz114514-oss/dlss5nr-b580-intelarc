#include "nr_route_adapters.h"
#include <nvsdk_ngx.h>

namespace nrb {
bool extract_fsr_input(NVSDK_NGX_Parameter* params,
                       ID3D12GraphicsCommandList* list, NRB_Frame& frame) {
    if (!extract_common_ngx(params,NRB_ROUTE_FSR,list,frame)) return false;
    frame.motion_scale_origin=NRB_MV_FSR_TO_NGX;
    // FFX dispatch may change upscaleSize while the context's OutWidth/Height
    // remain at maxUpscaleSize. FSR3 lacks these keys; retain the context size.
    unsigned int width=0,height=0;
    if (params->Get("FSR.upscaleSize.width",&width)==NVSDK_NGX_Result_Success && width)
        frame.output_width=width;
    if (params->Get("FSR.upscaleSize.height",&height)==NVSDK_NGX_Result_Success && height)
        frame.output_height=height;
    return true;
}
}
