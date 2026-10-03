#include "nr_core.h"
#include "nr_route_adapters.h"
#include <nvsdk_ngx.h>
#include <d3d12.h>

namespace {
template<class T> void scalar(NVSDK_NGX_Parameter* p, const char* key, T& value) {
    T incoming{};
    if (p->Get(key, &incoming) == NVSDK_NGX_Result_Success) value = incoming;
}
ID3D12Resource* texture(NVSDK_NGX_Parameter* p, const char* key) {
    ID3D12Resource* resource = nullptr;
    if (p->Get(key, &resource) != NVSDK_NGX_Result_Success)
        p->Get(key, reinterpret_cast<void**>(&resource));
    return resource;
}
}

namespace nrb {
bool extract_common_ngx(NVSDK_NGX_Parameter* p, NRB_Route route,
                        ID3D12GraphicsCommandList* list, NRB_Frame& f) {
    if (!p || !list) return false;
    f = {};
    f.abi_size = sizeof(f);
    f.route = route;
    f.command_list = list;
    f.color = texture(p, NVSDK_NGX_Parameter_Color);
    f.motion = texture(p, NVSDK_NGX_Parameter_MotionVectors);
    f.depth = texture(p, NVSDK_NGX_Parameter_Depth);
    f.exposure = texture(p, NVSDK_NGX_Parameter_ExposureTexture);
    f.output = texture(p, NVSDK_NGX_Parameter_Output);
    scalar(p, NVSDK_NGX_Parameter_Width, f.render_width);
    scalar(p, NVSDK_NGX_Parameter_Height, f.render_height);
    scalar(p, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, f.render_width);
    scalar(p, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, f.render_height);
    scalar(p, NVSDK_NGX_Parameter_OutWidth, f.output_width);
    scalar(p, NVSDK_NGX_Parameter_OutHeight, f.output_height);
    scalar(p, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X, f.color_base_x);
    scalar(p, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y, f.color_base_y);
    scalar(p, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X, f.motion_base_x);
    scalar(p, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y, f.motion_base_y);
    scalar(p, NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X, f.depth_base_x);
    scalar(p, NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y, f.depth_base_y);
    scalar(p, NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X, f.output_base_x);
    scalar(p, NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y, f.output_base_y);
    f.motion_scale_x = f.motion_scale_y = 1.0f;
    f.pre_exposure = f.exposure_scale = 1.0f;
    scalar(p, NVSDK_NGX_Parameter_MV_Scale_X, f.motion_scale_x);
    scalar(p, NVSDK_NGX_Parameter_MV_Scale_Y, f.motion_scale_y);
    scalar(p, NVSDK_NGX_Parameter_Jitter_Offset_X, f.jitter_x);
    scalar(p, NVSDK_NGX_Parameter_Jitter_Offset_Y, f.jitter_y);
    scalar(p, NVSDK_NGX_Parameter_DLSS_Pre_Exposure, f.pre_exposure);
    scalar(p, NVSDK_NGX_Parameter_DLSS_Exposure_Scale, f.exposure_scale);
    int reset = 0, flags = 0;
    scalar(p, NVSDK_NGX_Parameter_Reset, reset);
    scalar(p, NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, flags);
    f.reset_history = reset != 0;
    f.hdr_input = (flags & NVSDK_NGX_DLSS_Feature_Flags_IsHDR) != 0;
    f.motion_jittered = (flags & NVSDK_NGX_DLSS_Feature_Flags_MVJittered) != 0;
    f.low_resolution_motion = (flags & NVSDK_NGX_DLSS_Feature_Flags_MVLowRes) != 0;
    f.depth_inverted = (flags & NVSDK_NGX_DLSS_Feature_Flags_DepthInverted) != 0;
    f.auto_exposure = (flags & NVSDK_NGX_DLSS_Feature_Flags_AutoExposure) != 0;
    return true;
}

bool extract_ngx(NVSDK_NGX_Parameter* p, NRB_Route route,
                 ID3D12GraphicsCommandList* list, NRB_Frame& f) {
    switch (route) {
    case NRB_ROUTE_DLSS: return extract_dlss_input(p,list,f);
    case NRB_ROUTE_FSR: return extract_fsr_input(p,list,f);
    case NRB_ROUTE_XESS: return extract_xess_input(p,list,f);
    default: return extract_common_ngx(p,NRB_ROUTE_UNKNOWN,list,f);
    }
}
}
