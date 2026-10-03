#pragma once
#include <stdint.h>
#ifdef _WIN32
#define NR_TEXTURE_API extern "C" __declspec(dllexport)
#endif
// All pointers are in-process native objects. Pass both D3D12 device and queue
// to attach to an existing worker; two nulls create an isolated test device.
NR_TEXTURE_API void* nr_texture_create(void* sycl_queue, void* d3d12_device,
    void* d3d12_queue, uint32_t width, uint32_t height, char* error, uint32_t error_size);
// Source textures: RGB RGBA8_UNORM, BGRA8_UNORM or RGBA32F, motion RG16F or RG32F; same
// dimensions/device, SRV state, SDR. Motion is source-pixel current->previous.
// Source textures remain unchanged. This interface does not estimate motion.
NR_TEXTURE_API int nr_texture_prepare(void* bridge, void* color, void* motion,
    void* producer_fence, uint64_t producer_value, uint64_t frame_id,
    uint64_t previous_id, uint32_t reset, void* rgb_xpu, void* motion_xpu,
    char* error, uint32_t error_size);
NR_TEXTURE_API int nr_texture_export(void* bridge, uint64_t frame_id,
    const void* rgb_xpu, char* error, uint32_t error_size);
// Borrowed RGBA32F output resource and ready fence. Lifetime ends at the next
// successful export/close. Register the downstream completion before reuse.
NR_TEXTURE_API int nr_texture_output(void* bridge, void** resource, void** fence,
    uint64_t* value, char* error, uint32_t error_size);
NR_TEXTURE_API int nr_texture_retire(void* bridge, void* fence, uint64_t value,
    char* error, uint32_t error_size);
NR_TEXTURE_API int nr_texture_close(void* bridge, char* error, uint32_t error_size);
// Probe-only entrypoints. No model or motion estimation is implemented here.
NR_TEXTURE_API int nr_texture_fixture(void* bridge, uint32_t seed,
    uint32_t motion_bits, void** color, void** motion, char* error, uint32_t error_size);
// RE8-only synthetic BGRA8 + SNORM source for a GPU format-conversion check.
NR_TEXTURE_API int nr_texture_fixture_re8(void* bridge, void** color,
    void** motion, char* error, uint32_t error_size);
NR_TEXTURE_API int nr_texture_audit_output(void* bridge, void* rgb_xpu,
    char* error, uint32_t error_size);
NR_TEXTURE_API int nr_texture_audit_inputs(void* bridge, void* rgb_xpu,
    void* motion_xpu, char* error, uint32_t error_size);
NR_TEXTURE_API int nr_texture_compile_check(char* error, uint32_t error_size);
// Probe only: borrowed context pointers used to verify same-device attachment.
NR_TEXTURE_API int nr_texture_test_context(void* bridge, void** device, void** queue,
    char* error, uint32_t error_size);

// Optional ABI extension. Default OFF. Calls must remain on the borrowed Torch
// stream and owning host thread. The caller retains all input/result allocations
// until poll_idle reports true or close succeeds. prepare/export only enqueue in
// this mode; output's ready fence may be incomplete when returned.
// Setter succeeds only before first prepare or after exact safe retirement.
NR_TEXTURE_API int nr_texture_set_gpu_handoff(void* bridge, uint32_t enabled,
    char* error, uint32_t error_size);
// Nonblocking completion observation. An active prepared frame remains busy.
// Registering retire is insufficient; the actual consumer fence and every
// relevant SYCL event must have completed successfully before idle is returned.
NR_TEXTURE_API int nr_texture_poll_idle(void* bridge, uint32_t* idle,
    char* error, uint32_t error_size);
struct NR_TextureHandoffStats {
    uint32_t abi_size, version, enabled, active, poisoned, phase, records, forward_live;
    uint64_t frames_started, frames_retired, forward_imports, forward_releases;
    uint64_t pack_submits, unpack_submits, xpu_waits, xpu_signals;
    uint64_t consumer_registrations, busy_rejections, pack_value, output_value;
    uint64_t backward_value, retained_bytes, borrowed_queue, native_context, native_device;
};
NR_TEXTURE_API int nr_texture_gpu_handoff_stats(void* bridge, NR_TextureHandoffStats* stats,
    char* error, uint32_t error_size);

// Isolated ASI capability-gating contract; no change to the existing Frame ABI.
// producer_fence_required=0 preserves the existing caller-prepared CPU boundary.
// Setting it to 1 authorizes native prepare to replace that boundary with the
// exact nonzero producer fence/value carried by every nr_texture_prepare call.
// The source/ASI adapter MUST propagate that pair before bypassing its CPU wait.
// Borrowed queue addresses must match those used at create. The native helper
// copies (does not create) the same SYCL queue and checks context/in-order/LUID.
struct NR_TextureGpuHandoffConfig {
    uint32_t abi_size, version, enabled, producer_fence_required;
    uint64_t borrowed_sycl_queue, borrowed_d3d12_queue;
};
struct NR_TextureGpuHandoffInfo {
    uint32_t abi_size, version, enabled, healthy;
    uint32_t reuse_safe_idle, active, poisoned, queue_context_equal;
    uint32_t in_order, native_luid_matched, max_active_frames, private_records;
    uint32_t forward_import_per_wait, producer_cpu_wait_bypass_supported;
    uint32_t producer_cpu_wait_bypass_configured, producer_fence_required;
    uint64_t borrowed_sycl_queue, native_context, native_device, d3d12_device;
    uint64_t d3d12_queue, active_frame, retained_producer_fence, retained_producer_value;
};
NR_TEXTURE_API int nr_texture_configure_gpu_handoff(void* bridge,
    const NR_TextureGpuHandoffConfig* config, char* error, uint32_t error_size);
NR_TEXTURE_API int nr_texture_gpu_handoff_info(void* bridge,
    NR_TextureGpuHandoffInfo* info, char* error, uint32_t error_size);

// Caller holds its frame/retire serial lock. A new CPU worker may adopt the
// SAME queue and context only after the exact previous consumer/events retire.
// adopted=0 is a busy observation, not success or a change of ownership.
NR_TEXTURE_API int nr_texture_transfer_owner(void* bridge, uint64_t previous_native_thread,
    uint64_t borrowed_sycl_queue, uint64_t borrowed_d3d12_queue, uint32_t* adopted,
    char* error, uint32_t error_size);
