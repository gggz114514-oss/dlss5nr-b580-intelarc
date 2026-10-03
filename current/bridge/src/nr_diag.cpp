#include "nr_diag.h"
#include "nr_contract.h"
#include <sstream>
#include <iomanip>

namespace nrb {
const char* route_name(NRB_Route r) {
    switch(r) { case NRB_ROUTE_DLSS:return "DLSS"; case NRB_ROUTE_FSR:return "FSR";
        case NRB_ROUTE_XESS:return "XeSS"; default:return "unknown"; }
}
const char* route_evidence_name(NRB_RouteEvidence evidence) {
    switch(evidence) {
    case NRB_EVIDENCE_GAME_NGX:return "game_ngx_call";
    case NRB_EVIDENCE_STREAMLINE_NGX:return "streamline_ngx_call";
    case NRB_EVIDENCE_FSR_MARKER:return "fsr_optional_marker_only";
    case NRB_EVIDENCE_XESS_MARKER:return "xess_optional_marker_only";
    case NRB_EVIDENCE_INTERNAL_UNTAGGED:return "internal_untagged";
    case NRB_EVIDENCE_EXTERNAL_UNTAGGED:return "external_untagged";
    case NRB_EVIDENCE_STREAMLINE_FEATURE:return "streamline_feature_dlss";
    case NRB_EVIDENCE_XESS_API:return "xess_d3d12_execute";
    case NRB_EVIDENCE_FSR3_API:return "fsr3_upscaler_dispatch";
    default:return "none";
    }
}
const char* status_name(NRB_Status s) {
    switch(s) { case NRB_OK:return "nr_substituted"; case NRB_BYPASS_DISABLED:return "disabled";
        case NRB_BYPASS_UNSUPPORTED:return "unsupported_input";
        case NRB_BYPASS_UNSYNCED:return "producer_or_consumer_order_unproven";
        case NRB_BYPASS_UNKNOWN_ROUTE:return "unknown_route";
        case NRB_BYPASS_RUNTIME_MISSING:return "portable_runtime_assets_missing";
        default:return "processor_error"; }
}
std::string format_diagnostic(uint32_t index, const NRB_Frame& f, ResourceMeta c,
    ResourceMeta m, ResourceMeta d, ResourceMeta e, ResourceMeta o,
    QueueMeta q, NRB_Status status) {
    std::ostringstream out;
    out << std::setprecision(7)
        << "{\"event\":\"pre_sr_input\",\"route\":\"" << route_name(f.route)
        << "\",\"route_evidence\":\"" << route_evidence_name(f.route_evidence) << '"'
        << ",\"sample\":" << index << ",\"source\":[" << f.render_width << ',' << f.render_height
        << "],\"output\":[" << f.output_width << ',' << f.output_height << ']';
    const auto resource=[&](const char* name, ResourceMeta r, uint32_t x, uint32_t y) {
        out << ",\"" << name << "\":{\"format\":" << r.format
            << ",\"extent\":[" << r.width << ',' << r.height << "],\"base\":["
            << x << ',' << y << "]}";
    };
    resource("color",c,f.color_base_x,f.color_base_y);
    resource("motion",m,f.motion_base_x,f.motion_base_y);
    resource("depth",d,f.depth_base_x,f.depth_base_y);
    resource("exposure",e,0,0);
    resource("output_resource",o,f.output_base_x,f.output_base_y);
    const SourceTextures textures{static_cast<DXGI_FORMAT>(c.format),
        static_cast<DXGI_FORMAT>(m.format),c.width,c.height,m.width,m.height};
    const uint32_t work=plan_re8_adaptations(f,textures);
    out << ",\"re8_adapter_work\":[";
    bool first=true;
    const auto issue=[&](uint32_t flag,const char* label) {
        if (work & flag) {if (!first) out << ',';out << '"' << label << '"';first=false;}
    };
    issue(RE8_MISSING_INPUT,"missing_input");
    issue(RE8_SOURCE_CACHE,"source_cache");
    issue(RE8_SUBRECT_OR_EXTENT,"subrect_or_extent");
    issue(RE8_COLOR_FORMAT,"color_format");
    issue(RE8_HDR_POLICY,"hdr_policy");
    issue(RE8_MOTION_FORMAT_OR_SCALE,"motion_format_or_scale");
    issue(RE8_MOTION_LAYOUT,"motion_layout");
    issue(RE8_EXPOSURE_POLICY,"exposure_policy");
    out << ']';
    out << ",\"mv_scale\":[" << f.motion_scale_x << ',' << f.motion_scale_y
        << "],\"mv_scale_origin\":" << f.motion_scale_origin
        << ",\"jitter\":[" << f.jitter_x << ',' << f.jitter_y
        << "],\"pre_exposure\":" << f.pre_exposure
        << ",\"exposure_scale\":" << f.exposure_scale
        << ",\"flags\":{\"hdr\":" << f.hdr_input
        << ",\"motion_jittered\":" << f.motion_jittered
        << ",\"low_resolution_motion\":" << f.low_resolution_motion
        << ",\"depth_inverted\":" << f.depth_inverted
        << ",\"auto_exposure\":" << f.auto_exposure << '}'
        << ",\"reset\":" << f.reset_history
        << ",\"controls\":{\"enabled\":" << f.controls.enabled
        << ",\"input_height\":" << f.controls.input_height
        << ",\"style\":" << f.controls.style
        << ",\"history\":" << f.controls.history
        << ",\"graph_replay\":" << f.controls.graph_replay << '}'
        << ",\"queue\":{\"ordinal\":" << q.ordinal << ",\"type\":" << q.type
        << ",\"candidate_count\":" << q.candidate_count
        << ",\"same_device\":" << (q.same_device?"true":"false")
        << ",\"submissions\":" << q.submissions
        << ",\"consumer_list_previously_submitted\":" << (q.consumer_list_previously_submitted?"true":"false")
        << ",\"last_seen_consumer_queue_ordinal\":" << q.last_seen_consumer_queue_ordinal
        << ",\"last_seen_consumer_submit_serial\":" << q.last_seen_consumer_submit_serial
        << ",\"producer_status\":\"" << q.producer_status << "\"}"
        << ",\"bypass\":\"" << status_name(status) << "\"}\r\n";
    return out.str();
}
}
