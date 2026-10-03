#pragma once
#include "nr_bridge.h"

// Optional exports only. The frozen Frame/Controls/Processor layouts stay intact.
// Pointers are borrowed for the current owning-thread process callback ONLY.
struct NRB_ProducerPoint {
    uint32_t abi_size, version;
    ID3D12Device* device;
    ID3D12CommandQueue* queue;
    ID3D12Fence* fence;
    ID3D12Resource* color;
    ID3D12Resource* motion;
    void* processor_context;
    NRB_Process processor_process;
    uint64_t value, frame_id, registry_epoch, request_epoch, callback_cookie, owner_thread;
    uint32_t prepared_cpu_waited, reserved;
};

// Exact expected NR_TextureGpuHandoffInfo v1 seam (128 bytes on Win64).
// This declaration does not pin, link, or bundle Hilbert's moving native helper.
struct NRB_NativeGpuHandoffInfo {
    uint32_t abi_size, version, enabled, healthy;
    uint32_t reuse_safe_idle, active, poisoned, queue_context_equal;
    uint32_t in_order, native_luid_matched, max_active_frames, private_records;
    uint32_t forward_import_per_wait, producer_cpu_wait_bypass_supported;
    uint32_t producer_cpu_wait_bypass_configured, producer_fence_required;
    uint64_t borrowed_sycl_queue, native_context, native_device, d3d12_device;
    uint64_t d3d12_queue, active_frame, retained_producer_fence, retained_producer_value;
};
struct NRB_GpuHandoffCapability {
    uint32_t abi_size, version;
    NRB_ProducerPoint point;
    NRB_NativeGpuHandoffInfo native_info;
    uint64_t native_helper;
};
struct NRB_GpuHandoffStats {
    uint32_t abi_size, version, requested, armed, cap_healthy, pending, last_rejection, reserved;
    uint64_t request_epoch, registry_epoch, arms, disarms, arm_rejections;
    uint64_t source_queries, source_rejections, prepared_cpu_waits, prepared_bypasses;
    uint64_t motion_readback_waits, identity_mismatch_waits;
    uint64_t previous_consumer_waits, previous_consumer_wait_failures, process_failures;
    uint64_t native_helper, device, queue, native_context, borrowed_sycl_queue, owner_thread;
};
NRB_API int NRB_GetCurrentProducerPoint(NRB_ProducerPoint* point);
// Trusted host attestation: configure producer_fence_required=1 and inspect the
// actual helper's info on this callback/thread at safe idle BEFORE calling arm.
NRB_API int NRB_ArmGpuHandoffCapability(const NRB_GpuHandoffCapability* capability);
// Opt-in contract for the serial adapter with the proved native owner-transfer
// ABI. Each changed CPU worker must adopt actual retired native/XPU ownership
// before consuming this callback's producer point. The old arm stays thread-bound.
// Minting/requalification requires this callback's prepared CPU wait; an old
// armed capability never authorizes a warm contract/helper/context upgrade.
NRB_API int NRB_ArmSerialGpuHandoffCapability(
    const NRB_GpuHandoffCapability* capability, uint64_t owner_transfer_function);
NRB_API int NRB_GetSerialGpuHandoffArmed(); // CPU snapshot, no helper/GPU access
NRB_API int NRB_ClearGpuHandoffCapability(); // owning callback only; CPU state only
NRB_API int NRB_SetGpuHandoffRequested(int requested); // any thread; CPU state only
NRB_API int NRB_GetGpuHandoffStats(NRB_GpuHandoffStats* stats); // cheap CPU snapshot
