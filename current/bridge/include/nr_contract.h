#pragma once
#include "nr_bridge.h"
#include <dxgiformat.h>
#include <cstdint>

namespace nrb {
// This is a CPU-only admission plan for the existing RE8 model/texture host.
// A valid game upscaler frame may still need these adaptations. No flag here
// authorizes processing: current-frame producer/consumer fences are separate.
enum RE8Adaptation : uint32_t {
    RE8_DIRECT = 0,
    RE8_MISSING_INPUT = 1u << 0,
    RE8_SOURCE_CACHE = 1u << 1,
    RE8_SUBRECT_OR_EXTENT = 1u << 2,
    RE8_COLOR_FORMAT = 1u << 3,
    RE8_HDR_POLICY = 1u << 4,
    RE8_MOTION_FORMAT_OR_SCALE = 1u << 5,
    RE8_MOTION_LAYOUT = 1u << 6,
    RE8_EXPOSURE_POLICY = 1u << 7,
};

struct SourceTextures {
    DXGI_FORMAT color_format = DXGI_FORMAT_UNKNOWN;
    DXGI_FORMAT motion_format = DXGI_FORMAT_UNKNOWN;
    uint32_t color_width = 0, color_height = 0;
    uint32_t motion_width = 0, motion_height = 0;
};

uint32_t plan_re8_adaptations(const NRB_Frame& frame, const SourceTextures& textures);
}
