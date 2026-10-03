"""Narrow controls shim. The model and texture bridge stay in the installed RE8 runtime."""
from contextlib import nullcontext
import json
import inspect
import os
import sys
import threading
import time
import nr_game_pre_xess_host as host
from nr_game_controls import Settings
from numeric_game_profiles_720_v1 import FUSED_REPLAY_PROFILES, numeric_health
from numeric_frame_validation_720_v1 import serial_validation_frame


def _variant_default():
    # Older standalone control-panel fixtures predate the fast-backend selector.
    return ({"backend_variant": "unrounded"}
            if "backend_variant" in getattr(Settings, "__dataclass_fields__", {}) else {})


class _Panel:
    def __init__(self):
        self.setting = Settings(input_size=540, history_mode="fused",
                                graph_replay=True, **_variant_default())
        self.active = None

    def snapshot(self):
        return self.setting

    def mark_active(self, *args, **kwargs):
        self.active = (args, kwargs)


_panel = _Panel()
_web_started = False
_update_values = threading.local()
_extra_experiment_720 = ("c512_k8" if "c512_k8" in host.available_experiments_720() else "baseline")
_extra_optimizations_720 = ()
_settings_lock = threading.RLock()
_native_controls_read = None
_process_serial_lock = threading.RLock()
_validation_batch_enabled = True
_validation_batch_last_snapshot = None
_lifecycle_audit_requested = False
_lifecycle_audit_applied = False
_lifecycle_audit_epoch = 0


def validation_batch_state():
    with _settings_lock:
        return {"enabled": _validation_batch_enabled,
                "last_snapshot": (dict(_validation_batch_last_snapshot)
                                  if _validation_batch_last_snapshot is not None else None)}


def set_validation_batch(enabled: bool):
    if type(enabled) is not bool:
        raise ValueError("Expected a boolean validation setting")
    global _validation_batch_enabled
    # This CPU-only switch must never wait for the frame/native state lock.
    with _settings_lock:
        _validation_batch_enabled = enabled
        return validation_batch_state()


def lifecycle_audit_state():
    # No graph/source/weight traversal, native lock or GPU access from HTTP.
    with _settings_lock:
        return {"enabled": _lifecycle_audit_requested,
                "applied": _lifecycle_audit_applied,
                "pending": _lifecycle_audit_requested != _lifecycle_audit_applied,
                "epoch": _lifecycle_audit_epoch,
                "last_snapshot": (dict(_validation_batch_last_snapshot)
                                  if _validation_batch_last_snapshot is not None else None)}


def set_lifecycle_audit(enabled):
    if type(enabled) is not bool:
        raise ValueError("Expected a boolean lifecycle audit setting")
    global _lifecycle_audit_requested
    with _settings_lock:
        _lifecycle_audit_requested = enabled
        return lifecycle_audit_state()


def _apply_lifecycle_audit():
    """Worker-thread-only cold transition, after the old owner's handoff.

    The caller aborts the old frame first. Keep the old policy while retiring
    its hooks/graphs; never adopt a graph captured under a different policy.
    """
    global _lifecycle_audit_applied, _lifecycle_audit_epoch
    if not _process_serial_lock._is_owned():
        raise RuntimeError("Lifecycle transition requires the adapter process lock")
    if host._bridge is not None and host._thread != threading.get_ident():
        raise RuntimeError("Lifecycle transition requires the actual bound host thread")
    with _settings_lock:
        requested = _lifecycle_audit_requested
    if requested == _lifecycle_audit_applied:
        return False
    import replay_lifecycle_audit_base_720_v1 as lifecycle
    old_modes = getattr(host, "_modes", None)
    try:
        if getattr(host, "_failed", False):
            raise RuntimeError("Cannot reselect a failed GPU session")
        if old_modes is not None:
            old_modes.close()
            host._modes = None
        lifecycle.TRIAL_ENABLED = requested
        if old_modes is not None:
            host._modes = host._new_modes(host._active_experiment,
                                         host._active_optimizations)
        host._last_frame = None
        host.reset_stage_times()
        old_modes = None
        with _settings_lock:
            _lifecycle_audit_applied = requested
            _lifecycle_audit_epoch += 1
            return True
    except BaseException:
        host._failed = True
        host._failure_reason = "NR_LIFECYCLE_TRANSITION_FAILED"
        host._error("lifecycle-audit-transition")
        raise


def _remember_validation_frame(cycle):
    snapshot = dict(cycle.snapshot())
    global _validation_batch_last_snapshot
    with _settings_lock:
        _validation_batch_last_snapshot = snapshot


def _from_native(values):
    global _extra_experiment_720, _extra_optimizations_720
    if (values["input_height"] != 720 or
            (_extra_experiment_720 == "history_compact" and values["history"] != 3) or
            (_extra_experiment_720 in FUSED_REPLAY_PROFILES and
             (values["history"] != 3 or not values["graph_replay"]))):
        _extra_experiment_720 = "baseline"
    if (values["input_height"] != 720 or values["history"] != 3 or
            not values["graph_replay"] or _extra_experiment_720 != "c512_k8"):
        _extra_optimizations_720 = ()
    return Settings(
        enabled=bool(values["enabled"]),
        input_size=values["input_height"],
        style=values["style"],
        history_mode=("reference", "zero_motion", "reset", "fused")[values["history"]],
        graph_replay=bool(values["graph_replay"]),
        auto_mask=bool(values["auto_mask"]),
        display_strength=values["display_strength"],
        model_intensity=values["model_intensity"],
        local_tone=values["local_tone"],
        local_structure=values["local_structure"],
        skin_structure=(values["skin_structure"] if values["skin_structure_enabled"] else None),
        experiment_720=_extra_experiment_720,
        optimizations_720=_extra_optimizations_720,
        **_variant_default(),
    )


def configure(payload):
    values = json.loads(payload)
    with _settings_lock:
        # Deferred queue processing releases the native state mutex before
        # entering Python. Read current controls inside this commit boundary:
        # the payload may have queued before a successful web update.
        if _native_controls_read is not None:
            values = _native_controls_read()
        setting = _from_native(values)
        if isinstance(_panel, _Panel):
            _panel.setting = setting
        else:
            with _panel._lock:
                _panel._settings = setting
        host._panel = _panel


def process(device, queue, color, motion, frame_id, reset, width, height):
    """Serial frame with optional CPU validation reuse and uncached exit checks.

    Successful exit validates every owned child after host.process returns.
    This cost belongs to meter.process_wall_ms, outside host's stage timings.
    """
    # Keep stream binding, numerical ownership transfer and the frame in one
    # serial transaction even when the game's queue worker changes CPU thread.
    with _process_serial_lock:
        start_controls()
        host._panel = _panel
        meter = None
        if os.environ.get("CYBERPUNK_NR_COST_METER", "0") == "1":
            from nr_numeric_cost_meter_v1 import optional_meter
            meter = optional_meter()
        handoff_started = time.perf_counter_ns() if meter is not None else None
        modes = getattr(host, "_modes", None)
        owner = getattr(modes, "numeric_cleanup_calls", None)
        with _settings_lock:
            validation_enabled = _validation_batch_enabled
        cycle = serial_validation_frame(
            owner, game_adapter=sys.modules[__name__], serial_guard=_process_serial_lock,
            frame_id=frame_id, enabled=validation_enabled)
        closed = False
        try:
            current_thread = threading.get_ident()
            previous_thread = getattr(host, "_thread", None)
            if getattr(host, "_bridge", None) is not None and previous_thread != current_thread:
                host._bridge.torch.xpu.set_stream(host._bridge.stream)
                host._bridge.transfer_serial_thread(
                    game_adapter=sys.modules[__name__], serial_guard=_process_serial_lock,
                    previous_thread=previous_thread)
                host._thread = current_thread
                if owner is not None:
                    try:
                        owner.transfer_serial_thread(
                            game_adapter=sys.modules[__name__], serial_guard=_process_serial_lock,
                            previous_thread=previous_thread)
                    except BaseException:
                        host._error("numeric-serial-thread-handoff")
                        raise
            with _settings_lock:
                lifecycle_pending = _lifecycle_audit_requested != _lifecycle_audit_applied
            if lifecycle_pending:
                # Keep the old live frame through normal scope/graph retirement.
                # abort before close would falsely mark the live session failed.
                _apply_lifecycle_audit()
                cycle.abort()
                cycle = None
                owner = None
                modes = None
                # No old frame/owner/graph references survive reclamation.
                if host._bridge is not None:
                    import gc
                    gc.collect()
                    host._bridge.torch.xpu.empty_cache()
                modes = getattr(host, "_modes", None)
                owner = getattr(modes, "numeric_cleanup_calls", None)
                # The helper verifies the actual adapter.process caller.
                cycle = serial_validation_frame(
                    owner, game_adapter=sys.modules[__name__], serial_guard=_process_serial_lock,
                    frame_id=frame_id, enabled=validation_enabled)
            if meter is not None:
                meter.note_handoff((time.perf_counter_ns() - handoff_started) / 1e6)
            with meter.frame(host, frame_id) if meter is not None else nullcontext():
                success = False
                host_error = None
                try:
                    result = host.process(device, queue, color, motion, frame_id, reset, width, height)
                    success = True
                    return result
                except BaseException as error:
                    host_error = error
                    raise
                finally:
                    try:
                        # Include uncached exit validation in process_wall_ms.
                        cycle.close(success=success)
                        closed = True
                        _remember_validation_frame(cycle)
                    except BaseException as closing_error:
                        if host_error is not None:
                            raise host_error from closing_error
                        raise
        finally:
            if not closed and cycle is not None:
                cycle.abort()
                _remember_validation_frame(cycle)


def start_controls():
    global _web_started, _panel, _native_controls_read
    if not _web_started and os.environ.get("CYBERPUNK_NR_WEB", "1") != "0":
        try:
            from nr_game_controls import ControlPanel
            import cyberpunk_nr_web as native

            _native_controls_read = lambda: native._values(native.get_controls())

            def apply_native(_enabled, _strength):
                values = _update_values.values
                current = native._values(native.get_controls())
                current.update({
                    "enabled": int(values["enabled"]),
                    "input_height": values["input_size"],
                    "style": values["style"],
                    "history": ("reference", "zero_motion", "reset", "fused").index(values["history_mode"]),
                    "graph_replay": int(values["graph_replay"]),
                    "auto_mask": int(values["auto_mask"]),
                    "skin_structure_enabled": int(values["skin_structure"] is not None),
                    "display_strength": values["display_strength"],
                    "model_intensity": values["model_intensity"],
                    "local_tone": values["local_tone"],
                    "local_structure": values["local_structure"],
                    "skin_structure": values["skin_structure"] if values["skin_structure"] is not None else 0.0,
                })
                native.set_controls(current)
                if callable(getattr(host, "reset_stage_times", None)):
                    host.reset_stage_times()
                return True

            class CyberpunkPanel(ControlPanel):
                def update(self, values):
                    global _extra_experiment_720, _extra_optimizations_720
                    with _settings_lock:
                        _update_values.values = values
                        try:
                            result = super().update(values)
                            _extra_experiment_720 = result["settings"]["experiment_720"]
                            _extra_optimizations_720 = tuple(result["settings"]["optimizations_720"])
                            return result
                        finally:
                            del _update_values.values

                def status(self):
                    state = super().status()
                    state["available_history_modes"] = ("reference", "zero_motion", "reset", "fused")
                    state["resolution_note"] = "游戏 1280×720 输入；NR 可选 360p、480p、540p、720p，再交给 XeSS。真实运动档读取 DLSS 的运动纹理；零运动档供对照。"
                    return state

            def metrics():
                info = native.health()
                stages = info["stages"]
                runtime = (host.stage_times() if callable(getattr(host, "stage_times", None))
                           else None)
                if stages is not None:
                    stages["runtime"] = runtime
                    if os.environ.get("CYBERPUNK_NR_COST_METER", "0") == "1":
                        from nr_numeric_cost_meter_v1 import optional_meter
                        stages["cost_decomposition"] = optional_meter().snapshot()
                    stages["bridge_known_average_ms"] = (
                        stages["bridge_pre_sr_average_ms"] +
                        runtime["prepare_average_ms"] + runtime["export_average_ms"]
                        if runtime is not None else None)
                return {"last_ms": info["last_ms"], "average_ms": info["average_ms"],
                        "frames": info["processed_frames"],
                        "estimated_fps": info["estimated_fps"],
                        "stages": stages, "validation_batch": validation_batch_state()}

            def health():
                info = native.health()
                failed = bool(getattr(host, "_failed", False) or info["status"] == 5)
                modes = getattr(host, "_modes", None)
                return {"failed": failed, "reason": getattr(host, "_failure_reason", None),
                        "temporal_diagnostics": (host.temporal_diagnostics()
                            if callable(getattr(host, "temporal_diagnostics", None))
                            else {"enabled": False}),
                        "c128_pairwise_720_active": bool(
                            getattr(modes, "c128_pairwise_calls", None) is not None),
                        "c128_dual_qkv_720_active": bool(
                            getattr(modes, "c128_dual_calls", None) is not None),
                        "c64_attention_project_720_active": bool(
                            getattr(modes, "c64_attention_project_calls", None) is not None),
                        "c128_attention_project_720_active": bool(
                            getattr(modes, "c128_attention_project_calls", None) is not None),
                        "c512_library_720_active": bool(
                            len(getattr(modes, "c512_library_calls", None) or {}) == 16 and
                            all(value > 0 for value in modes.c512_library_calls.values())),
                        "native_k8_720_active": bool(
                            getattr(modes, "native_k8_calls", None) and
                            all(modes.native_k8_calls.get(name, 0) > 0
                                for name in ("pre", "post"))),
                        "history_compact_720_active": bool(
                            getattr(modes, "history_compact_calls", None) and
                            modes.history_compact_calls.get("compact", 0)),
                        "post_native_fma_720_active": bool(
                            getattr(modes, "post_native_fma_720", None) and
                            getattr(modes, "height", None) == 720 and
                            (getattr(modes, "post_calls", None) or {}).get("entry", 0) > 0),
                        "vit_head_720_active": bool(
                            len(getattr(modes, "vit_head_calls", None) or {}) == 8 and
                            all(value > 0 for value in modes.vit_head_calls.values())),
                        "experiment_720": getattr(host, "_active_experiment", "baseline"),
                        "optimizations_720": getattr(host, "_active_optimizations", ()),
                        "structure_combo_frame_route": (
                            getattr(modes, "last_combo_frame_route", None)),
                        "structure_combo_capture_gate": (
                            getattr(modes, "last_combo_capture_gate", None)),
                        **numeric_health(modes)}

            def geometry():
                bridge = getattr(host, "_bridge", None)
                return {"width": bridge.width, "height": bridge.height} if bridge else None

            def apply_timing(enabled):
                native.set_timing_enabled(enabled)
                if callable(getattr(host, "set_timing_enabled", None)):
                    host.set_timing_enabled(enabled)

            timing_options = ({"timing_read": native.get_timing_enabled,
                               "timing_apply": apply_timing}
                              if "timing_read" in inspect.signature(ControlPanel).parameters
                              else {})
            validation_options = ({"validation_read": validation_batch_state,
                                   "validation_apply": set_validation_batch}
                                  if "validation_read" in inspect.signature(ControlPanel).parameters
                                  else {})

            bridge_cache_options = ({"bridge_cache_read": native.get_hdr_cache_state,
                                     "bridge_cache_apply": native.set_hdr_cache_enabled}
                                    if "bridge_cache_read" in inspect.signature(ControlPanel).parameters
                                    else {})
            bridge_pool_options = ({"bridge_pool_read": native.get_hdr_resource_pool_state,
                                    "bridge_pool_apply": native.set_hdr_resource_pool_enabled}
                                   if "bridge_pool_read" in inspect.signature(ControlPanel).parameters
                                   else {})
            gpu_handoff_options = ({"gpu_handoff_read": native.get_gpu_handoff_state,
                                    "gpu_handoff_apply": native.set_gpu_handoff_requested}
                                   if "gpu_handoff_read" in inspect.signature(ControlPanel).parameters
                                   else {})
            lifecycle_options = ({"lifecycle_read": lifecycle_audit_state,
                                  "lifecycle_apply": set_lifecycle_audit}
                                 if "lifecycle_read" in inspect.signature(ControlPanel).parameters
                                 else {})

            panel = CyberpunkPanel(apply_native,
                modes=tuple((size, style) for size in (360, 480, 540, 720)
                            for style in (0, 1, 2)),
                route="pre-xess-fullsize",
                history_modes=("reference", "zero_motion", "reset", "fused"),
                graph_modes=(360, 480, 540, 720),
                available_experiments_720=host.available_experiments_720(),
                metrics_read=metrics, health_read=health, geometry_read=geometry,
                url_file=host.LOG / "cyberpunk-nr-control-url.txt",
                **timing_options, **validation_options, **lifecycle_options, **bridge_cache_options,
                **bridge_pool_options, **gpu_handoff_options)
            current = native._values(native.get_controls())
            with panel._lock:
                panel._settings = _from_native(current)
            _panel = panel
            host._panel = panel
            _web_started = True
        except Exception:
            # The NR processor is controlled by the C ABI even if the local
            # browser endpoint cannot bind or its URL file cannot be written.
            host._error("web-control")
            _web_started = True


def retire(fence, value):
    with _process_serial_lock:
        return host.retire(fence, value)
