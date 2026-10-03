#include "nr_core.h"
#include <d3d12.h>
#include <cmath>

namespace nrb {
bool validate_controls(const NRB_Controls& c) {
    auto unit = [](float x) { return std::isfinite(x) && x >= 0 && x <= 1; };
    auto model = [](float x) { return std::isfinite(x) && x >= 0 && x <= 2; };
    return c.abi_size == sizeof(NRB_Controls) && c.enabled <= 1 &&
        (c.input_height == 360 || c.input_height == 480 ||
         c.input_height == 540 || c.input_height == 720) &&
        c.style <= 2 && c.history <= NRB_HISTORY_FUSED &&
        c.graph_replay <= 1 && c.auto_mask <= 1 && c.skin_structure_enabled <= 1 &&
        unit(c.display_strength) && model(c.model_intensity) &&
        model(c.local_tone) && model(c.local_structure) &&
        (!c.skin_structure_enabled || model(c.skin_structure));
}
NRB_Status validate_metadata(const NRB_Frame& f) {
    if (f.abi_size != sizeof(NRB_Frame) || !f.command_list || !f.color || !f.motion || !f.output ||
        f.render_width == 0 || f.render_height == 0 || f.output_width == 0 || f.output_height == 0 ||
        f.render_width > 16384 || f.render_height > 16384 ||
        f.output_width > 16384 || f.output_height > 16384 ||
        f.render_width > f.output_width || f.render_height > f.output_height ||
        !std::isfinite(f.jitter_x) || !std::isfinite(f.jitter_y) ||
        !std::isfinite(f.motion_scale_x) || !std::isfinite(f.motion_scale_y) ||
        !std::isfinite(f.pre_exposure) || !std::isfinite(f.exposure_scale) ||
        f.pre_exposure <= 0 || f.exposure_scale <= 0)
        return NRB_BYPASS_UNSUPPORTED;
    if (f.route == NRB_ROUTE_UNKNOWN) return NRB_BYPASS_UNKNOWN_ROUTE;
    if ((f.route == NRB_ROUTE_DLSS && f.motion_scale_origin != NRB_MV_DLSS_NGX) ||
        (f.route == NRB_ROUTE_FSR && f.motion_scale_origin != NRB_MV_FSR_TO_NGX) ||
        (f.route == NRB_ROUTE_XESS && f.motion_scale_origin != NRB_MV_XESS_TO_NGX))
        return NRB_BYPASS_UNSUPPORTED;
    return NRB_OK;
}

static bool same_device(ID3D12Device* d, ID3D12Resource* r) {
    if (!r) return true;
    ID3D12Device* owner = nullptr;
    if (FAILED(r->GetDevice(IID_PPV_ARGS(&owner)))) return false;
    const bool same = owner == d;
    owner->Release();
    return same;
}

NRB_Status validate_resources(const NRB_Frame& f) {
    auto status = validate_metadata(f);
    if (status != NRB_OK) return status;
    ID3D12Device* d = nullptr;
    if (FAILED(f.command_list->GetDevice(IID_PPV_ARGS(&d)))) return NRB_BYPASS_UNSUPPORTED;
    const auto c = f.color->GetDesc(), m = f.motion->GetDesc(), o = f.output->GetDesc();
    const bool valid = same_device(d, f.color) && same_device(d, f.motion) &&
        same_device(d, f.depth) && same_device(d, f.exposure) && same_device(d, f.output) &&
        c.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && c.SampleDesc.Count == 1 &&
        c.DepthOrArraySize == 1 && c.MipLevels == 1 &&
        (c.Format == DXGI_FORMAT_R8G8B8A8_UNORM || c.Format == DXGI_FORMAT_B8G8R8A8_UNORM ||
         c.Format == DXGI_FORMAT_R11G11B10_FLOAT ||
         c.Format == DXGI_FORMAT_R16G16B16A16_FLOAT || c.Format == DXGI_FORMAT_R32G32B32A32_FLOAT) &&
        m.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && m.SampleDesc.Count == 1 &&
        (m.Format == DXGI_FORMAT_R16G16_FLOAT || m.Format == DXGI_FORMAT_R32G32_FLOAT ||
         m.Format == DXGI_FORMAT_R16G16B16A16_SNORM) &&
        o.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
        f.color_base_x + uint64_t(f.render_width) <= c.Width &&
        f.color_base_y + uint64_t(f.render_height) <= c.Height &&
        f.motion_base_x + uint64_t(f.low_resolution_motion ? f.render_width : f.output_width) <= m.Width &&
        f.motion_base_y + uint64_t(f.low_resolution_motion ? f.render_height : f.output_height) <= m.Height &&
        f.output_base_x + uint64_t(f.output_width) <= o.Width &&
        f.output_base_y + uint64_t(f.output_height) <= o.Height;
    d->Release();
    return valid ? NRB_OK : NRB_BYPASS_UNSUPPORTED;
}

bool validate_result(const NRB_Frame& f, const NRB_Result& r) {
    if (r.abi_size != sizeof(NRB_Result) || !r.color || !r.ready_fence || !r.ready_value)
        return false;
    const auto source = f.color->GetDesc(), replacement = r.color->GetDesc();
    if (replacement.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        replacement.Width != source.Width || replacement.Height != source.Height ||
        replacement.DepthOrArraySize != 1 || replacement.MipLevels != 1 ||
        replacement.SampleDesc.Count != 1 ||
        replacement.Format != DXGI_FORMAT_R32G32B32A32_FLOAT)
        return false;
    ID3D12Device* device = nullptr;
    if (FAILED(f.command_list->GetDevice(IID_PPV_ARGS(&device)))) return false;
    ID3D12Device* ready_owner = nullptr;
    const bool valid = same_device(device, r.color) &&
        SUCCEEDED(r.ready_fence->GetDevice(IID_PPV_ARGS(&ready_owner))) && ready_owner == device;
    if (ready_owner) ready_owner->Release();
    device->Release();
    return valid;
}
}
