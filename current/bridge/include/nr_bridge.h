#pragma once
#include <stdint.h>

#ifdef _WIN32
#define NRB_API extern "C" __declspec(dllexport)
#else
#define NRB_API extern "C"
#endif

struct ID3D12Device;
struct ID3D12CommandQueue;
struct ID3D12GraphicsCommandList;
struct ID3D12Resource;
struct ID3D12Fence;

enum NRB_Route : uint32_t { NRB_ROUTE_UNKNOWN=0, NRB_ROUTE_DLSS=1, NRB_ROUTE_FSR=2, NRB_ROUTE_XESS=3 };
enum NRB_RouteEvidence : uint32_t {
    NRB_EVIDENCE_NONE=0, NRB_EVIDENCE_GAME_NGX=1,
    NRB_EVIDENCE_STREAMLINE_NGX=2, NRB_EVIDENCE_FSR_MARKER=3,
    NRB_EVIDENCE_XESS_MARKER=4, NRB_EVIDENCE_INTERNAL_UNTAGGED=5,
    NRB_EVIDENCE_EXTERNAL_UNTAGGED=6,
    NRB_EVIDENCE_STREAMLINE_FEATURE=7,
    NRB_EVIDENCE_XESS_API=8, NRB_EVIDENCE_FSR3_API=9
};
enum NRB_Status : uint32_t {
    NRB_OK=0, NRB_BYPASS_DISABLED=1, NRB_BYPASS_UNSUPPORTED=2,
    NRB_BYPASS_UNSYNCED=3, NRB_BYPASS_UNKNOWN_ROUTE=4, NRB_PROCESS_ERROR=5,
    NRB_BYPASS_RUNTIME_MISSING=6
};
enum NRB_History : uint32_t {
    NRB_HISTORY_REFERENCE=0, NRB_HISTORY_ZERO_MOTION=1,
    NRB_HISTORY_RESET=2, NRB_HISTORY_FUSED=3
};
enum NRB_MotionScaleOrigin : uint32_t {
    NRB_MV_UNKNOWN=0,
    NRB_MV_DLSS_NGX=1,
    NRB_MV_FSR_TO_NGX=2,
    NRB_MV_XESS_TO_NGX=3
};
struct NRB_Controls {
    uint32_t abi_size;
    uint32_t enabled;
    uint32_t input_height; // 360, 480, or 540; source size remains unchanged
    uint32_t style;        // standard, natural, film
    NRB_History history;
    uint32_t graph_replay;
    uint32_t auto_mask;
    uint32_t skin_structure_enabled;
    float display_strength;
    float model_intensity;
    float local_tone;
    float local_structure;
    float skin_structure;
};

// A frame is a game upscaler input, never a Present/backbuffer image. The
// resources and command list are borrowed for this synchronous callback.
// Motion scale converts sampled velocity to current-to-previous source pixels.
struct NRB_Frame {
    uint32_t abi_size;
    NRB_Route route;
    NRB_RouteEvidence route_evidence;
    ID3D12Device* device;
    ID3D12CommandQueue* queue;
    ID3D12GraphicsCommandList* command_list;
    ID3D12Resource* color;
    ID3D12Resource* motion;
    ID3D12Resource* depth;
    ID3D12Resource* exposure;
    ID3D12Resource* output;
    uint32_t render_width, render_height;
    uint32_t output_width, output_height;
    uint32_t color_base_x, color_base_y;
    uint32_t motion_base_x, motion_base_y;
    uint32_t depth_base_x, depth_base_y;
    uint32_t output_base_x, output_base_y;
    float motion_scale_x, motion_scale_y;
    NRB_MotionScaleOrigin motion_scale_origin;
    float jitter_x, jitter_y;
    float pre_exposure, exposure_scale;
    uint64_t frame_id;
    uint32_t reset_history;
    uint32_t hdr_input;
    uint32_t motion_jittered;
    uint32_t low_resolution_motion;
    uint32_t depth_inverted;
    uint32_t auto_exposure;
    NRB_Controls controls;
    uint64_t control_epoch;
};

// The processor may return a source-sized NR texture only after consuming the
// same frame's color and motion. It must retain the texture until retire() is
// called with a completed consumer fence. A nullptr result means bypass.
struct NRB_Result {
    uint32_t abi_size;
    ID3D12Resource* color;
    ID3D12Fence* ready_fence;
    uint64_t ready_value;
};
using NRB_Process = int (*)(void* context, const NRB_Frame*, NRB_Result*);
using NRB_Retire = int (*)(void* context, ID3D12Fence*, uint64_t);
struct NRB_Processor {
    uint32_t abi_size;
    void* context;
    NRB_Process process;
    NRB_Retire retire;
};

// CPU wall-clock stages of completed split-batch frames. The tail includes
// queued composite work, XeSS and completion; it is not pure bridge overhead.
struct NRB_StageTimes {
    uint32_t abi_size;
    uint32_t reserved;
    uint64_t frames;
    double prep_last_ms, prep_average_ms;
    double host_last_ms, host_average_ms;
    double composite_submit_last_ms, composite_submit_average_ms;
    double tail_last_ms, tail_average_ms;
};

// Completed-frame CPU recording intervals; gap/span include game work.
struct NRB_RecordTimes {
    uint32_t abi_size, reserved;
    uint64_t frames, last_nr_frame_id;
    double record_last_ms, record_average_ms;
    double resources_last_ms, resources_average_ms;
    double hdr_initialize_last_ms, hdr_initialize_average_ms;
    double xess_record_last_ms, xess_record_average_ms;
    double record_to_submit_gap_last_ms, record_to_submit_gap_average_ms;
    double record_to_retire_span_last_ms, record_to_retire_span_average_ms;
};
NRB_API int NRB_GetRecordTimes(NRB_RecordTimes* times);

// Immutable pipeline cache; toggles affect new frames, not in-flight resources.
struct NRB_HdrCacheStats {
    uint32_t abi_size, enabled;
    uint64_t shader_compile_calls;
    uint64_t shader_compile_failures;
    uint64_t source_reads;
    uint64_t source_hashes;
    uint64_t pso_creates;
    uint64_t root_signature_creates;
    uint64_t pipeline_hits;
    uint64_t pipeline_misses;
    uint64_t uncached_initializations;
    uint64_t shader_generation;
    uint64_t resident_device_entries;
};
NRB_API int NRB_GetHdrCacheStats(NRB_HdrCacheStats* stats);
NRB_API int NRB_SetHdrCacheEnabled(int enabled);

// The built-in processor delegates to the existing portable RE8 Python host.
// No model, shader, or GPU Block/DIS implementation is bundled in the ASI.
NRB_API int NRB_RegisterProcessor(const NRB_Processor* processor);
NRB_API int NRB_RegisterQueue(ID3D12CommandQueue* queue);
// Legacy export retained for callers; it cannot assert producer proof.
NRB_API void NRB_SetProducerReady(int ready);
NRB_API int NRB_SetControls(const NRB_Controls* controls);
NRB_API int NRB_GetControls(NRB_Controls* controls);
NRB_API NRB_Status NRB_LastStatus();
NRB_API uint64_t NRB_InterceptCount(NRB_Route route);
// 0 = not requested, 1 = deferred initialization running, 2 = ready, 3 = failed.
NRB_API int NRB_InitState();
NRB_API uint64_t NRB_ProcessedCount();
NRB_API double NRB_LastProcessMs();
NRB_API double NRB_AverageProcessMs();
NRB_API int NRB_GetStageTimes(NRB_StageTimes* times);
// Timing can be disabled without changing NR controls, history or graph replay.
NRB_API int NRB_GetTimingEnabled();
NRB_API int NRB_SetTimingEnabled(int enabled);
