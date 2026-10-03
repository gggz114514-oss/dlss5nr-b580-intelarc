"""Opt-in RE8 source-frame NR: game color and motion -> matching XeSS input.

This module runs in the game process, on one thread. Missing Triton artifacts
are fatal here; all exposed geometries must have been prepared offline.
"""
from __future__ import annotations

from collections import deque
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

from numeric_game_profiles_720_v1 import (
    EXPERIMENTS_720, FUSED_REPLAY_PROFILES, PROFILES, combined_mode_options, numeric_health)


BASE = Path(__file__).resolve().parents[1]
FAST = BASE
DATA = BASE / "data"
LOG = BASE / "logs"
PYTHON = BASE / "python"
BRIDGE_DLL = BASE / "native/nr_texture_bridge_re8_v1.dll"

_dll_dirs = []
_disk_only = None
_bridge = None
_modes = None
_mode_options = None
_active_experiment = "baseline"
_active_optimizations = ()
_panel = None
_thread = None
_device = _queue = None
_last_frame = None
_last_history_mode = None
_last_graph_replay = None
_last_backend_variant = None
_failed = False
_failure_reason = None
_stage_lock = threading.Lock()
_stage_samples = deque(maxlen=32)
_stage_frames = 0
_timing_enabled = True
_timing_epoch = 0
_temporal_diagnostics = None
_periodic_flash_diagnostics = None
_periodic_flash_sampler = None
_periodic_flash_enabled = os.environ.get("NR_DIAG_PERIODIC_FLASH_V2") == "1"
_temporal_events_enabled = os.environ.get("NR_DIAG_TEMPORAL_EVENTS") == "1"


def temporal_diagnostics():
    result = (_temporal_diagnostics.snapshot()
              if _temporal_diagnostics is not None else {"enabled": False})
    if _periodic_flash_enabled:
        result["periodic_flash_v2"] = periodic_flash_diagnostics()
    return result


def periodic_flash_diagnostics():
    global _periodic_flash_sampler
    from periodic_flash_snapshot_v2 import PythonFrameRecorder, WindowSampler
    # Native rejection evidence must also be visible before the first NR callback.
    python_observed = _periodic_flash_diagnostics is not None
    python_snapshot = (_periodic_flash_diagnostics.snapshot() if python_observed
                       else PythonFrameRecorder().snapshot())
    if _periodic_flash_sampler is None:
        _periodic_flash_sampler = WindowSampler()
    web = sys.modules.get("cyberpunk_nr_web")
    result = _periodic_flash_sampler.sample(getattr(web, "_native", None), python_snapshot)
    result["python_nr_call_observed"] = python_observed
    return result


def set_timing_enabled(enabled):
    global _timing_enabled, _timing_epoch
    if type(enabled) is not bool:
        raise ValueError("timing enabled must be boolean")
    _timing_enabled = enabled
    _timing_epoch += 1
    reset_stage_times()


def timing_enabled():
    return _timing_enabled


def reset_stage_times():
    global _stage_frames
    with _stage_lock:
        _stage_samples.clear()
        _stage_frames = 0


def stage_times():
    with _stage_lock:
        if not _stage_samples:
            return None
        last = _stage_samples[-1]
        averages = tuple(sum(sample[i] for sample in _stage_samples) /
                         len(_stage_samples) for i in range(3))
        frames = _stage_frames
    return {"frames": frames,
            "prepare_last_ms": last[0], "prepare_average_ms": averages[0],
            "model_last_ms": last[1], "model_average_ms": averages[1],
            "export_last_ms": last[2], "export_average_ms": averages[2]}


def _record_stage_times(prepared, modeled, exported):
    global _stage_frames
    with _stage_lock:
        _stage_samples.append((prepared, modeled, exported))
        _stage_frames += 1


def _error(stage):
    try:
        LOG.mkdir(parents=True, exist_ok=True)
        (LOG / "nr-pre-xess-python-error.txt").write_text(
            stage + "\n" + traceback.format_exc(), encoding="utf-8")
    except OSError:
        pass


def available_experiments_720():
    cfg = json.loads((DATA / "product-v1/local-runtime-v1.json").read_text(encoding="utf-8-sig"))
    values = tuple(cfg.get("available_720_experiments", ("baseline",)))
    allowed = set(EXPERIMENTS_720)
    if not values or values[0] != "baseline" or len(set(values)) != len(values) or set(values) - allowed:
        raise ValueError("Invalid installed 720p experiment list")
    return values


def _new_modes(experiment, optimizations=()):
    from nr_game_fullsize import FullsizeGameModes

    if _mode_options is None:
        raise RuntimeError("Game mode configuration is not initialized")
    switches = {
        "baseline": {},
        "c512": {"c512_qkv_library_720": True},
        "k8": {"native_k8_720": True},
        "c512_k8": {"c512_qkv_library_720": True, "native_k8_720": True},
        "c512_k8_decoder": {"c512_qkv_library_720": True,
                              "native_k8_720": True,
                              "decoder_gather_unround_720": True},
        "c512_k8_c32": {"c512_qkv_library_720": True,
                          "native_k8_720": True, "c32_window_chain_720": True},
        "c512_k8_c32_post": {"c512_qkv_library_720": True,
                               "native_k8_720": True, "c32_window_chain_720": True,
                               "post_rgb_tail_720": True},
        "history_compact": {"history_compact_720": True},
        "post_fma_fp16": {"post_native_fma_720": "fp16_fma"},
        "post_fma_fp32": {"post_native_fma_720": "fp32_fma"},
        "vit_head": {"vit_head_720": True},
    }
    if experiment not in available_experiments_720():
        raise ValueError("720p experiment has no installed validation")
    if experiment in PROFILES:
        switches[experiment] = PROFILES[experiment].mode_options()
    if optimizations:
        if experiment != "c512_k8" or set(optimizations) - set(available_experiments_720()):
            raise ValueError("Checked optimizations require the installed C512+K8 baseline")
        switches[experiment] = combined_mode_options(optimizations)
    runtime_config = json.loads((DATA / "product-v1/local-runtime-v1.json").read_text(
        encoding="utf-8-sig"))
    selected = dict(switches[experiment])
    if selected.get("numeric_cleanup_720", {}).get("post_store") == "native_rtz":
        receipt = runtime_config.get("native_rtz_receipt_720")
        if not isinstance(receipt, str) or not receipt:
            raise RuntimeError("Native RTZ has no installed production evidence receipt")
        selected["native_rtz_receipt"] = receipt
    return FullsizeGameModes(BASE / "exact", DATA / "product-v1/profile-v1.json",
                             runtime_config["profile_sha256"],
                             **_mode_options, **selected)


def _initialize(device, queue, width, height):
    global _disk_only, _bridge, _modes, _mode_options, _active_experiment, _active_optimizations
    global _panel, _thread, _device, _queue
    if sys.flags.utf8_mode != 1:
        raise RuntimeError("Embedded Python must use UTF-8 mode")
    for path in (PYTHON / "Library/bin",
                 BASE / "openvino-libs"):
        if path.is_dir():
            _dll_dirs.append(os.add_dll_directory(str(path)))
    sys.path[:0] = [str(BASE), str(BASE / "game"), str(BASE / "modules"),
                    str(BASE / "fast/backend")]
    runtime_bin = PYTHON / "Library/bin"
    os.environ["PATH"] = os.pathsep.join([str(runtime_bin)] + [
        entry for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry and "oneapi" not in entry.lower() and
        not (Path(entry.strip(chr(34))) / "icpx.exe").is_file()])
    for key in ("ONEAPI_DEVICE_SELECTOR", "ONEAPI_ROOT",
                "TRITON_INTEL_SYCL_COMPILER", "TRITON_INTEL_DEVICE_EXTENSIONS"):
        os.environ.pop(key, None)
    _dll_dirs.append(ctypes.WinDLL(str(runtime_bin / "sycl9.dll")))
    from fast_cached_runtime_v1 import DiskOnly, bootstrap
    bootstrap()
    # Select the audited overlay before any nr_backend or model module import.
    overlay_bootstrap = BASE / "experimental/fp8_unround_overlay/bootstrap.py"
    spec = importlib.util.spec_from_file_location("nr_fast_game_overlay_bootstrap", overlay_bootstrap)
    if spec is None or spec.loader is None:
        raise RuntimeError("Fast backend overlay is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.activate(BASE, "")
    import rows_fscache_guard_v1 as guard
    guard.install()
    _disk_only = DiskOnly()
    _disk_only.__enter__()
    from nr_texture_bridge_v1 import TextureBridge
    from nr_game_fullsize import FullsizeGameModes

    cfg = json.loads((DATA / "product-v1/local-runtime-v1.json").read_text(encoding="utf-8-sig"))
    pairwise_720 = cfg.get("c128_pairwise_720", False)
    dual_qkv_720 = cfg.get("c128_dual_qkv_720", False)
    c64_attention_project_720 = cfg.get("c64_attention_project_720", False)
    c128_attention_project_720 = cfg.get("c128_attention_project_720", False)
    if any(type(value) is not bool for value in (
            pairwise_720, dual_qkv_720, c64_attention_project_720,
            c128_attention_project_720)):
        raise ValueError("720p structure switches must be JSON booleans")
    # Fixed before the first graph capture.  An already captured graph cannot
    # be switched by the web panel; restart with an empty list to roll back.
    combo_setting = os.environ.get("NR_STRUCTURE_COMBO_MODES")
    if combo_setting is None:
        combo_modes = tuple(cfg.get("structure_combo_modes", ()))
    else:
        combo_modes = tuple(int(value.strip()) for value in combo_setting.split(",")
                            if value.strip())
    _bridge = TextureBridge(BRIDGE_DLL, width, height,
                            native_device=device, native_queue=queue)
    _mode_options = dict(controlled=True, graph_capture_policy="all",
                         combo_modes=combo_modes,
                         c128_pairwise_720=pairwise_720,
                         c128_dual_qkv_720=dual_qkv_720,
                         c64_attention_project_720=c64_attention_project_720,
                         c128_attention_project_720=c128_attention_project_720)
    _modes = _new_modes("baseline")
    _active_experiment = "baseline"
    _active_optimizations = ()
    _thread, _device, _queue = threading.get_ident(), device, queue

    if Path(sys.executable).name.casefold() != "re8.exe":
        return
    try:
        from nr_game_controls import ControlPanel
        native = ctypes.CDLL(str(Path(sys.executable).parent / "dxgi.dll"))
        setter = native.nr_live_set_display_controls
        setter.argtypes, setter.restype = (ctypes.c_int, ctypes.c_float), ctypes.c_int
        stats = native.nr_live_get_processing_stats
        stats.argtypes = (ctypes.POINTER(ctypes.c_double),
                          ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint64))
        stats.restype = ctypes.c_int

        def metrics():
            last, average, count = ctypes.c_double(), ctypes.c_double(), ctypes.c_uint64()
            if not stats(ctypes.byref(last), ctypes.byref(average), ctypes.byref(count)):
                return None
            return {"last_ms": last.value, "average_ms": average.value,
                    "frames": count.value,
                    "estimated_fps": 1000.0 / average.value if average.value > 0 else 0.0}

        _panel = ControlPanel(lambda enabled, strength: bool(setter(int(enabled), strength)),
                              modes=tuple((size, style) for size in (360, 480, 540, 720)
                                          for style in (0, 1, 2)),
                              route="pre-xess-fullsize", metrics_read=metrics,
                              health_read=lambda: {"failed": _failed,
                                                   "reason": _failure_reason,
                                                   "structure_combo_modes": sorted(_modes.combo_modes),
                                                   "structure_combo_active": _modes.height in _modes.combo_modes,
                                                   "c128_pairwise_720_active": bool(
                                                       _modes.c128_pairwise_calls is not None),
                                                   "c128_dual_qkv_720_active": bool(
                                                       _modes.c128_dual_calls is not None),
                                                   "c64_attention_project_720_active": bool(
                                                       _modes.c64_attention_project_calls is not None),
                                                   "c128_attention_project_720_active": bool(
                                                       _modes.c128_attention_project_calls is not None),
                                                   "c512_library_720_active": bool(
                                                       _modes.c512_library_calls is not None),
                                                   "native_k8_720_active": bool(
                                                       _modes.native_k8_calls is not None),
                                                   "c32_window_chain_720_active": bool(
                                                       _modes.c32_window_calls is not None and
                                                       all(_modes.c32_window_calls.get(name, 0) > 0
                                                           for name in ("encoder32_1", "encoder32_2",
                                                                        "decoder32_2", "decoder32_3"))),
                                                   "post_rgb_tail_720_active": bool(
                                                       _modes.post_rgb_tail_calls is not None and
                                                       _modes.post_rgb_tail_calls.get("tail", 0) > 0),
                                                   "decoder_gather_unround_720_active": bool(
                                                       _modes.height == 720 and _modes.session is not None and
                                                       _modes.session._stack.decoder_gather.unround_activations),
                                                   "history_compact_720_active": bool(
                                                       _modes.history_compact_calls and
                                                       _modes.history_compact_calls.get("compact", 0)),
                                                   "post_native_fma_720_active": bool(
                                                       _modes.post_native_fma_720 and _modes.height == 720),
                                                   "vit_head_720_active": bool(
                                                       _modes.vit_head_calls is not None),
                                                   "structure_combo_frame_route":
                                                   _modes.last_combo_frame_route,
                                                   "structure_combo_capture_gate":
                                                   _modes.last_combo_capture_gate,
                                                   **numeric_health(_modes)},
                              geometry_read=lambda: {"width": _bridge.width,
                                                     "height": _bridge.height},
                              history_modes=("reference", "zero_motion", "reset", "fused"),
                              graph_modes=(360, 480, 540, 720),
                              available_experiments_720=tuple(
                                  cfg.get("available_720_experiments", ("baseline",))),
                              url_file=LOG / "nr-pre-xess-control-url.txt")
    except Exception:
        _error("controls")
        raise


def process(device_ptr: int, queue_ptr: int, color_ptr: int, motion_ptr: int,
            frame_id: int, game_reset: int, width: int = 960,
            height: int = 540) -> tuple[int, int, int]:
    """Return borrowed (RGBA32F texture, ready fence, value) native pointers."""
    global _bridge, _modes, _active_experiment, _active_optimizations, _failed, _failure_reason
    global _last_frame, _last_history_mode, _last_graph_replay, _last_backend_variant
    global _temporal_diagnostics, _periodic_flash_diagnostics
    if _failed:
        raise RuntimeError("Pre-XeSS NR disabled after failure")
    flash_frame = None
    try:
        if _periodic_flash_enabled:
            try:
                from periodic_flash_snapshot_v2 import PythonFrameRecorder
                if _periodic_flash_diagnostics is None:
                    _periodic_flash_diagnostics = PythonFrameRecorder()
                web = sys.modules.get("cyberpunk_nr_web")
                flash_frame = _periodic_flash_diagnostics.start(frame_id, game_reset, getattr(web, "_native", None))
                if flash_frame is not None:
                    flash_frame["stage"] = "initialize"
            except Exception:
                flash_frame = None  # An absent optional CPU helper cannot disable NR.
        measure = _timing_enabled
        timing_epoch = _timing_epoch
        started = time.perf_counter() if measure else 0.0
        if any(type(v) is not int or v <= 0 for v in
               (device_ptr, queue_ptr, color_ptr, motion_ptr, frame_id)):
            raise ValueError("Missing native NR input")
        if game_reset not in (0, 1):
            raise ValueError("Invalid game history reset")
        if (type(width) is not int or type(height) is not int or
                (width, height) not in ((960, 540), (1280, 720))):
            raise ValueError("XeSS source geometry has no verified offline cache")
        if _bridge is None:
            _initialize(device_ptr, queue_ptr, width, height)
        elif (device_ptr, queue_ptr, threading.get_ident()) != (_device, _queue, _thread):
            raise RuntimeError("D3D12 queue/device or NR thread changed")
        geometry_changed = (_bridge.width, _bridge.height) != (width, height)
        if geometry_changed:
            # Native host has already retired the previously borrowed output.
            # Recreate all source-sized buffers; model geometry is selected below.
            from nr_texture_bridge_v1 import TextureBridge
            from nr_gpu_handoff_host_v1 import ensure_idle_before_reinitialize
            ensure_idle_before_reinitialize(_bridge)
            _bridge.close()
            _bridge = TextureBridge(BRIDGE_DLL, width, height,
                                    native_device=device_ptr, native_queue=queue_ptr)
        from nr_texture_bridge_v1 import SourceFrame
        from nr_gpu_handoff_host_v1 import source_handoff
        producer_point = source_handoff(_bridge, device_ptr, queue_ptr, color_ptr,
                                        motion_ptr, frame_id)
        initialized_at = time.perf_counter() if measure else 0.0

        if _panel is None:
            from nr_game_controls import Settings
            setting = Settings(input_size=480)
        else:
            setting = _panel.snapshot()
        if setting.experiment_720 != "baseline" and (
                setting.input_size != 720 or setting.backend_variant != "unrounded" or
                (setting.experiment_720 == "history_compact" and setting.history_mode != "fused")):
            raise ValueError("720p candidate settings left the validated route")
        optimizations = tuple(getattr(setting, "optimizations_720", ()))
        if (setting.experiment_720 in FUSED_REPLAY_PROFILES or optimizations) and (
                (width, height) != (1280, 720) or setting.history_mode != "fused" or
                not setting.graph_replay):
            raise ValueError("Numerical game comparisons require 720p source, fused history and graph replay")
        candidate_changed = (setting.experiment_720 != _active_experiment or
                             optimizations != _active_optimizations)
        if candidate_changed:
            import gc
            import torch

            previous_experiment = _active_experiment
            before_reserved = torch.xpu.memory_reserved()
            before_allocated = torch.xpu.memory_allocated()
            replacement = _new_modes(setting.experiment_720, optimizations)
            old_modes = _modes
            _modes = replacement
            _active_experiment = setting.experiment_720
            _active_optimizations = optimizations
            old_modes.close()
            del old_modes
            # A captured graph keeps Python cycles and private XPU allocations.
            # Release them before the next candidate builds reset/history graphs.
            gc.collect()
            torch.xpu.empty_cache()
            try:
                LOG.mkdir(parents=True, exist_ok=True)
                with (LOG / "nr-mode-switch-memory.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps({
                        "from": previous_experiment,
                        "to": _active_experiment,
                        "optimizations": optimizations,
                        "reserved_before": before_reserved,
                        "allocated_before": before_allocated,
                        "reserved_after_release": torch.xpu.memory_reserved(),
                        "allocated_after_release": torch.xpu.memory_allocated(),
                    }) + "\n")
            except OSError:
                pass
        history_mode = setting.history_mode
        graph_replay = setting.graph_replay
        reset = (candidate_changed or geometry_changed or bool(game_reset) or _last_frame != frame_id - 1 or
                 _last_history_mode != history_mode or
                 _last_graph_replay != graph_replay or
                 _last_backend_variant != setting.backend_variant or history_mode == "reset")
        diagnostic_before = None
        if _temporal_events_enabled or _periodic_flash_enabled:
            from nr_temporal_diagnostics_v1 import TemporalDiagnostics, model_state, reset_causes
            if _temporal_diagnostics is None:
                _temporal_diagnostics = TemporalDiagnostics(enabled=True)
            causes = reset_causes(candidate_changed=candidate_changed,
                                  geometry_changed=geometry_changed, game_reset=game_reset,
                                  last_frame=_last_frame, frame_id=frame_id,
                                  last_history_mode=_last_history_mode, history_mode=history_mode,
                                  last_graph_replay=_last_graph_replay, graph_replay=graph_replay,
                                  last_backend_variant=_last_backend_variant,
                                  backend_variant=setting.backend_variant)
            diagnostic_before = model_state(_modes)
        frame = SourceFrame(color_ptr, motion_ptr, width, height, frame_id,
                            frame_id - 1, reset, **producer_point)
        if flash_frame is not None:
            flash_frame["stage"] = "prepare"
        color, motion = _bridge.prepare(frame)
        # The bridge allocates motion for this frame, so this diagnostic never
        # modifies the game's source texture or XeSS's own motion input.
        if history_mode == "zero_motion":
            motion.zero_()
        prepared_at = time.perf_counter() if measure else 0.0
        from nr_backend.controlled_temporal import NRControls
        model_controls = NRControls(style=setting.style,
                                    intensity=setting.model_intensity,
                                    local_tone=setting.local_tone,
                                    local_structure=setting.local_structure,
                                    auto_mask=setting.auto_mask,
                                    skin_structure=setting.skin_structure)
        if flash_frame is not None:
            _periodic_flash_diagnostics.prepare(flash_frame, _modes, causes, {
                "experiment": setting.experiment_720, "optimizations": list(optimizations),
                "history": history_mode, "graph_requested": graph_replay, "backend": setting.backend_variant,
                "input_height": setting.input_size, "style": setting.style, "intensity": setting.model_intensity,
                "local_tone": setting.local_tone, "local_structure": setting.local_structure,
                "auto_mask": setting.auto_mask, "skin_structure": setting.skin_structure})
        result = _modes.process(color, motion, height=setting.input_size,
                                reset=reset,
                                history_warp="fused" if history_mode == "fused" else "reference",
                                graph_replay=graph_replay, controls=model_controls,
                                variant=setting.backend_variant)
        if diagnostic_before is not None:
            _temporal_diagnostics.record(frame_id, causes, diagnostic_before,
                                         model_state(_modes), input_height=setting.input_size,
                                         selection={"experiment": setting.experiment_720,
                                                    "optimizations": list(optimizations),
                                                    "history": history_mode,
                                                    "graph_requested": graph_replay,
                                                    "graph_used": _modes.last_graph_used,
                                                    "backend": setting.backend_variant,
                                                    "style": setting.style,
                                                    "intensity": setting.model_intensity,
                                                    "tone": setting.local_tone,
                                                    "structure": setting.local_structure,
                                                    "auto_mask": setting.auto_mask,
                                                    "skin_structure": setting.skin_structure})
        modeled_at = time.perf_counter() if measure else 0.0
        strength = setting.display_strength
        if strength < 1:
            result = (color + (result - color) * strength).clamp(0, 1)
        if flash_frame is not None:
            flash_frame["stage"] = "export"
        _bridge.export(result.contiguous(), frame_id=frame_id)
        output = _bridge.output()
        exported_at = time.perf_counter() if measure else 0.0
        if measure and _timing_enabled and timing_epoch == _timing_epoch:
            _record_stage_times((prepared_at - initialized_at) * 1000,
                                (modeled_at - prepared_at) * 1000,
                                (exported_at - modeled_at) * 1000)
        _last_frame = frame_id
        _last_history_mode = history_mode
        _last_graph_replay = graph_replay
        _last_backend_variant = setting.backend_variant
        if _panel is not None:
            _panel.mark_active(setting.input_size, setting.style, history_mode,
                               _modes.last_graph_used,
                               {"intensity": setting.model_intensity,
                                "local_tone": setting.local_tone,
                                "local_structure": setting.local_structure,
                                "auto_mask": setting.auto_mask,
                                "skin_structure": setting.skin_structure},
                               graph_requested=graph_replay,
                               backend_variant=setting.backend_variant,
                               experiment_720=setting.experiment_720,
                               optimizations_720=optimizations)
        if (measure and _timing_enabled and timing_epoch == _timing_epoch and
                exported_at - started > 0.15):
            spike = {"frame": frame_id, "input_height": setting.input_size,
                     "game_reset": bool(game_reset), "history_mode": history_mode,
                     "graph_replay": graph_replay,
                     "backend_variant": setting.backend_variant,
                     "experiment_720": setting.experiment_720,
                     "effective_reset": reset,
                     "init_ms": round((initialized_at - started) * 1000, 3),
                     "prepare_ms": round((prepared_at - initialized_at) * 1000, 3),
                     "model_ms": round((modeled_at - prepared_at) * 1000, 3),
                     "export_ms": round((exported_at - modeled_at) * 1000, 3)}
            try:
                with (LOG / "nr-pre-xess-python-spikes.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(spike) + "\n")
            except OSError:
                pass
        if flash_frame is not None:
            _periodic_flash_diagnostics.complete(flash_frame, _modes)
        return output.resource, output.fence, output.value
    except BaseException as exc:
        from nr_gpu_handoff_host_v1 import invalidate_after_failure
        invalidate_after_failure()
        if flash_frame is not None:
            _periodic_flash_diagnostics.complete(flash_frame, _modes, stage=flash_frame["stage"], exception=exc)
        _failed = True
        _failure_reason = ("GPU_DEVICE_LOST" if "DEVICE_LOST" in str(exc) else
                           "NR_PROCESS_FAILED")
        _error("process")
        raise


def retire(fence_ptr: int, value: int) -> None:
    global _failed, _failure_reason
    if _failed or _bridge is None:
        raise RuntimeError("Pre-XeSS NR unavailable")
    try:
        _bridge.retire(fence_ptr, value)
    except BaseException as exc:
        _failed = True
        _failure_reason = ("GPU_DEVICE_LOST" if "DEVICE_LOST" in str(exc) else
                           "NR_RETIRE_FAILED")
        _error("retire")
        raise
