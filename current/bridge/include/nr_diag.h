#pragma once
#include "nr_bridge.h"
#include <array>
#include <cstdint>
#include <string>

namespace nrb {
constexpr uint32_t kDiagnosticFramesPerRoute = 8;
struct ResourceMeta { uint32_t format=0, width=0, height=0; };
struct QueueMeta {
    uint32_t ordinal=0, type=0;
    uint32_t candidate_count=0;
    bool same_device=false;
    uint64_t submissions=0;
    bool consumer_list_previously_submitted=false;
    // Diagnostic only: command-list pointers may be reused after Reset.
    uint32_t last_seen_consumer_queue_ordinal=0;
    uint64_t last_seen_consumer_submit_serial=0;
    const char* producer_status="unproven";
};
struct DiagnosticGate {
    std::array<uint32_t, 4> seen{};
    uint32_t claim(NRB_Route route) {
        if (route > NRB_ROUTE_XESS) return 0;
        auto& value=seen[route];
        return value < kDiagnosticFramesPerRoute ? ++value : 0;
    }
};
const char* route_name(NRB_Route);
const char* status_name(NRB_Status);
std::string format_diagnostic(uint32_t index, const NRB_Frame&, ResourceMeta color,
    ResourceMeta motion, ResourceMeta depth, ResourceMeta exposure,
    ResourceMeta output, QueueMeta queue, NRB_Status status);
}
