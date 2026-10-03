"""Owning-NR-thread GPU handoff transition; HTTP only edits the ASI request.

Expected optional wrapper seam: configure_gpu_handoff(enabled,
producer_fence_required=bool), gpu_handoff_info(), poll_idle(), stream.sycl_queue,
handle, thread and gpu_handoff. No mutable wrapper is bundled or pinned here.
"""
import threading
import time
import sys


class HandoffController:
    def __init__(self):
        self._lock = threading.Lock()
        self._qualified = None
        self._owner_key = None
        self._snapshot = {"applied": False, "cap_healthy": False,
                          "serial_owner_transfer": False,
                          "reason": "OFF", "native_idle_waits": 0,
                          "native_idle_timeouts": 0, "transitions": 0}

    def snapshot(self):
        with self._lock:
            return dict(self._snapshot)

    def _remember(self, **values):
        with self._lock:
            self._snapshot.update(values)

    def ensure_idle(self, bridge, *, timeout=10.0):
        # Only actual completion observation may release retained native/XPU
        # owners. Registering retire, an old ready fence, or request OFF cannot.
        if not getattr(bridge, "gpu_handoff", False):
            return
        bridge._ready()
        if bridge.poll_idle():
            return
        with self._lock:
            self._snapshot["native_idle_waits"] += 1
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if bridge.poll_idle():
                return
            time.sleep(0.001)
        with self._lock:
            self._snapshot["native_idle_timeouts"] += 1
        raise RuntimeError("Previous GPU handoff consumer/events did not actually complete")

    @staticmethod
    def _verified(info, bridge, point):
        ones = ("version", "enabled", "healthy", "reuse_safe_idle",
                "queue_context_equal", "in_order", "native_luid_matched",
                "max_active_frames", "forward_import_per_wait",
                "producer_cpu_wait_bypass_supported", "producer_cpu_wait_bypass_configured",
                "producer_fence_required")
        return (all(info.get(name) == 1 for name in ones) and
                info.get("abi_size") == 128 and info.get("private_records") == 2 and
                all(info.get(name) == 0 for name in
                    ("active", "poisoned", "retained_producer_fence", "retained_producer_value")) and
                bool(info.get("native_context")) and bool(info.get("native_device")) and
                info.get("borrowed_sycl_queue") == bridge.stream.sycl_queue and
                info.get("d3d12_device") == point.device and info.get("d3d12_queue") == point.queue)

    def source(self, bridge, api, device, queue, color, motion, frame_id):
        point = api.current_producer_point(device, queue, color, motion, frame_id)
        state = api.get_gpu_handoff_state()
        requested = bool(state and state["requested"])
        if point is None:
            if getattr(bridge, "gpu_handoff", False) or (state and state["armed"]):
                raise RuntimeError("Qualified GPU handoff has no current scoped producer point")
            self._remember(applied=False, cap_healthy=False,
                           serial_owner_transfer=False, reason="SCOPED_ABI_UNAVAILABLE")
            return {}
        bridge._ready()
        self.ensure_idle(bridge)
        if not requested:
            if not point.prepared_cpu_waited:
                # OFF restores the ASI CPU wait for the NEXT prepared decision.
                # This frame has already committed a bypass, so keep its native
                # producer wait until it finishes instead of disabling it early.
                if not getattr(bridge, "gpu_handoff", False):
                    raise RuntimeError("Committed producer bypass has no active native handoff")
                self._qualified = None
                self._remember(applied=True, cap_healthy=False,
                               serial_owner_transfer=False,
                               reason="FINISH_COMMITTED_GPU_FRAME_BEFORE_OFF")
                return {"producer_fence": point.fence, "producer_value": point.value}
            if getattr(bridge, "gpu_handoff", False):
                # Request OFF already restored the ASI CPU wait. Mutate native
                # configuration here, on the owner, only AFTER actual idle.
                bridge.configure_gpu_handoff(False, producer_fence_required=False)
            self._qualified = None
            self._remember(applied=False, cap_healthy=False,
                           serial_owner_transfer=False, reason="OFF")
        else:
            methods = ("configure_gpu_handoff", "gpu_handoff_info", "poll_idle")
            if not all(callable(getattr(bridge, name, None)) for name in methods):
                api.clear_gpu_handoff_capability()
                self._qualified = None
                self._remember(applied=False, cap_healthy=False,
                               serial_owner_transfer=False, reason="OPTIONAL_NATIVE_ABI_UNAVAILABLE")
                if not point.prepared_cpu_waited:
                    raise RuntimeError("Native GPU handoff seam disappeared after prepared bypass")
                return {"producer_fence": point.fence, "producer_value": point.value}
            serial = (getattr(bridge, '_owner_transfer_supported', False) is True and
                      callable(getattr(bridge, 'transfer_serial_thread', None)) and
                      callable(getattr(api, 'serial_gpu_handoff_supported', None)) and
                      api.serial_gpu_handoff_supported() and
                      callable(getattr(api, 'arm_serial_gpu_handoff_capability', None)))
            # This is a GPU-queue capability, not an assumption that the game
            # always submits on the same CPU worker. The actual serial adapter
            # already proved retired native ownership before entering source().
            # Unpaired older helpers/APIs keep their strict CPU-thread key.
            owner_key = (id(bridge), bridge.handle, None if serial else bridge.thread,
                         bridge.stream.sycl_queue, point.device, point.queue, bool(serial))
            key = owner_key + (point.registry_epoch, point.request_epoch)
            if key != self._qualified or not state["armed"]:
                api.clear_gpu_handoff_capability()
                # A substituted/reinitialized helper cannot inherit the old
                # capability after the ASI has already chosen a bypass.
                if not point.prepared_cpu_waited:
                    if owner_key != self._owner_key:
                        raise RuntimeError("Helper/context changed after prepared bypass; capability cleared")
                    # OFF/ON can arrive after this frame's ASI wait decision.
                    # Finish the already authorized frame with its actual point;
                    # re-arm only in the next frame, which now takes the CPU wait.
                    self._qualified = None
                    self._remember(applied=bridge.gpu_handoff, cap_healthy=False,
                                   serial_owner_transfer=False,
                                   reason="WAIT_FOR_CPU_PREPARED_FRAME")
                    return {"producer_fence": point.fence, "producer_value": point.value}
                if not bridge.poll_idle():
                    raise RuntimeError("GPU handoff transition requires actual safe idle")
                bridge.configure_gpu_handoff(True, producer_fence_required=True)
                info = bridge.gpu_handoff_info()  # cold transition only; no per-frame constants/hash scan
                if not self._verified(info, bridge, point):
                    raise RuntimeError("Native GPU handoff capability proof rejected")
                armed = (api.arm_serial_gpu_handoff_capability(point, bridge, info) if serial else
                         api.arm_gpu_handoff_capability(point, bridge.handle, info))
                self._owner_key = owner_key
                self._qualified = key if armed else None
                with self._lock:
                    self._snapshot["transitions"] += 1
                self._remember(applied=True, cap_healthy=armed,
                               serial_owner_transfer=bool(serial and armed),
                               reason="ARMED" if armed else "REQUEST_CHANGED_OR_ARM_REJECTED")
        return {"producer_fence": point.fence, "producer_value": point.value}

    def invalidate(self, api):
        self._qualified = None
        self._owner_key = None
        self._remember(cap_healthy=False, serial_owner_transfer=False,
                       reason="HOST_REINITIALIZATION_OR_FAILURE")
        if api is not None:
            api.clear_gpu_handoff_capability()


controller = HandoffController()


def source_handoff(bridge, device, queue, color, motion, frame_id):
    api = sys.modules.get("cyberpunk_nr_web")
    if api is None or getattr(api, "_native", None) is None:
        # The shared RE8 host and web-disabled legacy callers keep their old
        # path; this optional seam never loads an ASI from the host module.
        if getattr(bridge, "gpu_handoff", False):
            raise RuntimeError("Active GPU handoff has no owning ASI binding")
        return {}
    return controller.source(bridge, api, device, queue, color, motion, frame_id)


def ensure_idle_before_reinitialize(bridge):
    api = sys.modules.get("cyberpunk_nr_web")
    controller.ensure_idle(bridge)
    controller.invalidate(api if api and getattr(api, "_native", None) is not None else None)


def invalidate_after_failure():
    api = sys.modules.get("cyberpunk_nr_web")
    try:
        controller.invalidate(api if api and getattr(api, "_native", None) is not None else None)
    except Exception:
        # The ASI also disarms on a failed callback. Preserve the original host
        # exception/failure latch when an optional CPU binding itself failed.
        pass


def host_snapshot():
    return controller.snapshot()
