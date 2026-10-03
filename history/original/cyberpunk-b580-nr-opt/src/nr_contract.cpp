#include "nr_contract.h"
#include <cmath>

namespace nrb {
namespace {
bool close_to(float a, float b) {
    return std::isfinite(a) && std::fabs(a - b) < 0.001f;
}
}

uint32_t plan_re8_adaptations(const NRB_Frame& f, const SourceTextures& t) {
    uint32_t work = RE8_DIRECT;
    if (!f.color || !f.motion || !f.render_width || !f.render_height)
        work |= RE8_MISSING_INPUT;
    if (!((f.render_width == 960 && f.render_height == 540) ||
          (f.render_width == 1280 && f.render_height == 720)))
        work |= RE8_SOURCE_CACHE;
    if (f.color_base_x || f.color_base_y || f.motion_base_x || f.motion_base_y ||
        t.color_width != f.render_width || t.color_height != f.render_height ||
        t.motion_width != f.render_width || t.motion_height != f.render_height)
        work |= RE8_SUBRECT_OR_EXTENT;
    if (t.color_format != DXGI_FORMAT_R8G8B8A8_UNORM &&
        t.color_format != DXGI_FORMAT_B8G8R8A8_UNORM &&
        t.color_format != DXGI_FORMAT_R32G32B32A32_FLOAT)
        work |= RE8_COLOR_FORMAT;
    if (f.hdr_input) work |= RE8_HDR_POLICY;
    if (f.motion_jittered || !f.low_resolution_motion)
        work |= RE8_MOTION_LAYOUT;
    if (!close_to(f.pre_exposure, 1) || !close_to(f.exposure_scale, 1))
        work |= RE8_EXPOSURE_POLICY;
    if (t.motion_format == DXGI_FORMAT_R16G16B16A16_SNORM) {
        // The existing RE8 shader applies exactly these factors itself.
        if (!close_to(f.motion_scale_x, float(f.render_width) * 0.5f) ||
            !close_to(f.motion_scale_y, -float(f.render_height) * 0.5f))
            work |= RE8_MOTION_FORMAT_OR_SCALE;
    } else if (t.motion_format == DXGI_FORMAT_R16G16_FLOAT ||
               t.motion_format == DXGI_FORMAT_R32G32_FLOAT) {
        if (!close_to(f.motion_scale_x, 1) || !close_to(f.motion_scale_y, 1))
            work |= RE8_MOTION_FORMAT_OR_SCALE;
    } else {
        work |= RE8_MOTION_FORMAT_OR_SCALE;
    }
    return work;
}
}
