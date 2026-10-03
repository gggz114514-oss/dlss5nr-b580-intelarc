"""Default-off, independently selected history numerics for actual 720p.

CPU source handoff only: no model import, GPU execution or compilation was
performed. Use after selecting a fresh controlled unrounded C512-library /
native-K8 session, with fused history and the existing 768x1280 body graph:

    with installed(modes, history_value="fp32_all_paths") as counter:
        counter.preflight()  # Luna: compile/load, never launches a kernel
        counter.validate()   # CPU identities/versions/sources, before replay
        # modes.process(..., history_warp="fused", graph_replay=True, ...)
        receipt = counter.snapshot()  # JSON metadata, no device tensors

All-reference is a true no-op and still yields this three-method interface.
Only this module and its same-prefix kernel are new. Seven reviewed sources
are authenticated at their ACTUAL loaded paths; the candidate's neighbours
are not used as the reference. D snapshots of these candidate files work.

The child layers session._installed, calling the previous scope first and
then installing local hooks. It does not own modes._select/_process or graph
signatures. The main numeric suite owns configuration signatures and final
graph/session closure. On exit this child restores hooks, retires the session
even without captures, and retains its constants until the owner is closed.

Fractional keeps axes/axis-invalid and NUMERATOR/RECIPROCAL/NORMALIZED/INVALID/
debug-TAPS allocations/stores and raw two-tensor return to post. Near-integer
keeps the actual torch _sample_five_axes code, five taps, original FMAs and
whole-frame half-motion bool synchronization; only its local callee aliases
change. fp32_fractional deliberately leaves near-integer VALUES at reference;
the other selected dimensions still apply there. fp32_all_paths also replaces
_half_texture_counts' shared-exponent INT64 value sum / _integer_half_away.

direct_pixel uses (pixel + .5) + motion.half().float(), original cubic axes,
clamp and pixel-center 1/256 rounding, including cross=(ax*ay+128)>>8. It
removes the normalized coordinate round trips / 21-bit conversion, not the
integer texture weights or value algorithm. It accepts reference values.
direct_pixel + native dimension reciprocals is explicitly invalid: direct
coordinates do not consume dimension reciprocals. All other enum combinations
are implemented and require independent numerical/temporal/quality screening.

Native scalar inverse means FP32 tl.div_rn(1,x), independently for both groups
and total. Native dimensions are immutable FP32 1/1280 and 1/720. Table assets
stay owned/live; no memory-release claim. Near retains the reference's three
scalar-domain bool checks. Fractional INVALID stores stay intact and their
last-frame host audit is explicit report() work, never extra timed-frame sync.
For validated finite motion, clamped t in [0,1] gives group in [1,1.125] and
five-tap total in [.984375,1], within the reviewed interval with FP32 margin.
INVALID audit across the numerical test frames remains a required Luna gate.

The fractional arithmetic remains two main launches (axes + five-tap), zero
main launches saved. Near all-path values use five counts kernels on the same
torch route; native scalar inverse uses three division kernels. Net ATen /
domain-check launch count and timings are unmeasured. Python history counters
are outside-body calls, never presented as body-capture or replay counts.

Pending Luna: off bytes; both motion dtypes, both DEBUG specializations, zero
spill and actual binary/source hashes; reset/history/capture/replay, threshold
and single-outlier motion, signed texels/zero/midpoints/clamp; raw private
history and full-frame finite/error gates; independent 50-frame B-C-C-B and
complete continuous-history video/user review. No installation or adoption.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import update_wrapper
import hashlib
from importlib import import_module, util
import json
from pathlib import Path
import struct
import sys
from threading import get_ident
from types import FunctionType, MethodType


H, W = 720, 1280
INTERNAL = (768, 1280)
AXIS_BLOCK, SAMPLE_BLOCK, DIVIDE_BLOCK = 128, 64, 128
_IN_FLIGHT = None  # Frame-only serial guard, not a scope-wide exclusive hook.
REFERENCE_SHA256 = {
    "joint": "4621cda4df3426cd6898ace624432b148ea17f77e5374b2cd5be9b22812c2ca5",
    "fused": "14df7f80a39c99f877993afdb878c701de832b427ca763b9fc3934745361e647",
    "game_sampler": "6e4f73ce30a29ffe3366abe558dbb1ef204726a10a1facab2487f1ff7d48e1f0",
    "sampling": "e95228c1743bdb5013d3c3b68cbb85fae315c6deeb913b6c4a057c76d2ed89f8",
    "reciprocal": "b150d33db7b3fcde300fef118bf2a1d874dd8fe510e40d2941d30a43ae6aee28",
    "temporal": "35c1c1d5c1475ac0e3385dfa76f0e1020109a376a1d233a64e6c2a1036ca1828",
    "policy": "87bf4ba9b9f60e8c10173dc8498d54e23e3387dee0a4485fae08db21dedda827",
    "execution": "8fbfe06ff6bcd94011da4ec4038a9735ada672dd90df2ff793417344fd045462",
}


@dataclass(frozen=True)
class _Options:
    history_value: str = "reference"
    history_reciprocal: str = "table"
    history_dimension_rcp: str = "table"
    history_coord: str = "reference"

    def __post_init__(self):
        enums = {
            "history_value": ("reference", "fp32_fractional", "fp32_all_paths"),
            "history_reciprocal": ("table", "native"),
            "history_dimension_rcp": ("table", "native"),
            "history_coord": ("reference", "direct_pixel"),
        }
        for name, values in enums.items():
            if type(getattr(self, name)) is not str or getattr(self, name) not in values:
                raise ValueError(f"Unknown {name}: {getattr(self, name)!r}")
        if self.history_coord == "direct_pixel" and self.history_dimension_rcp == "native":
            raise ValueError("direct_pixel does not consume dimension reciprocals; select table")

    @property
    def identity(self):
        return ("history-numeric-suite-720-v1", self.history_value,
                self.history_reciprocal, self.history_dimension_rcp, self.history_coord)


def _loaded(name):
    module = sys.modules.get(name)
    if module is None:
        raise RuntimeError(f"Expected the owner's already-loaded module: {name}")
    return module


def _actual_module(name, path, *, require_path=False):
    """Prefer the actual loaded callee; resolve absent roles from the real caller.

    Reviewed E/G/D reference-role mixtures are allowed. Each loaded role still
    needs its known source hash / code filename checks, not adjacent E bytes.
    The candidate kernel itself must come from this candidate's own snapshot.
    """
    module = sys.modules.get(name)
    if module is None:
        path = Path(path).resolve(strict=True)
        spec = util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load the actual history role: {name}")
        module = util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    if require_path and Path(module.__file__).resolve(strict=True) != Path(path).resolve(strict=True):
        raise RuntimeError(f"Loaded history role differs from the actual caller path: {name}")
    return module


def _closure(function, name):
    cells = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
    if name not in cells:
        raise RuntimeError(f"Missing actual owner closure: {name}")
    return cells[name].cell_contents


def _plain(function):
    function = getattr(function, "fn", function)
    seen = set()
    while hasattr(function, "__wrapped__"):
        if id(function) in seen:
            raise RuntimeError("History callable wrapper cycle")
        seen.add(id(function))
        function = function.__wrapped__
    return function


def _reviewed(role, module, *functions):
    path = Path(module.__file__).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA256[role]:
        raise RuntimeError(f"Reviewed history source changed: {role}: {path}")
    for function in functions:
        code = getattr(_plain(function), "__code__", None)
        if code is None or Path(code.co_filename).resolve(strict=True) != path:
            raise RuntimeError(f"History callable outside authenticated source: {role}")


def _manifest(modules):
    return {role: {"path": str(Path(module.__file__).resolve(strict=True)),
                   "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
            for role, module in modules.items()}


def _clone(function, aliases):
    result = FunctionType(function.__code__, dict(function.__globals__, **aliases),
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return update_wrapper(result, function)


def _installed_contains(current, target):
    """Follow explicit child links without assuming any child is the chain top."""
    seen = set()
    for _ in range(64):
        if current is target:
            return True
        if not isinstance(current, FunctionType) or id(current) in seen:
            raise RuntimeError("History installed chain has a cycle or unknown callable")
        seen.add(id(current))
        previous = getattr(current, "__nr_numeric_original_installed__", None)
        if previous is not None:
            current = previous
            continue
        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is not None:
            current = wrapped
            continue
        cells = dict(zip(current.__code__.co_freevars, current.__closure__ or ()))
        if current.__module__ == "c512_window_projection_game_v1" and "original_installed" in cells:
            current = cells["original_installed"].cell_contents
            continue
        return False
    raise RuntimeError("History installed chain exceeded its bounded depth")


def _tensor(torch, name, tensor, shape, dtype, device):
    if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or
            tensor.dtype != dtype or tensor.device != device or not tensor.is_contiguous()):
        raise ValueError(f"Unexpected 720p history tensor: {name}")


def _constant(tensor):
    # Process-local guard, NOT a stable cross-process model identifier.
    return (id(tensor), tensor._version, tensor.data_ptr(), tuple(tensor.shape),
            tuple(tensor.stride()), str(tensor.dtype), str(tensor.device))


def _constant_receipt(state):
    return dict(identity=state[0], version=state[1], data_ptr=state[2],
                shape=list(state[3]), stride=list(state[4]), dtype=state[5],
                device=state[6], identity_scope="current_process_only")


class ReferenceCounter:
    """No imports, hooks, tensor operations or retirement for the default arm."""
    def validate(self):
        return True

    def preflight(self):
        return self.snapshot()

    def snapshot(self):
        return dict(options=asdict(_Options()), active=False, noop=True,
                    sources={}, calls={}, capture_calls={}, frame_route_calls={},
                    frontend_route_calls={}, resources={}, actualkernelhash={},
                    capture_gates=[], preflight_complete=True, launches_saved=0)


class HistoryNumericCounter720:
    def __init__(self, modes, options, debug):
        self.modes, self.options, self._debug = modes, options, debug
        self.flags_identity = options.identity + (debug,)
        self.thread = get_ident()
        self.session = modes.session
        if self.session is None:
            raise RuntimeError("Select the fresh 720p session before installing history numerics")
        self.stack = self.session._stack
        self.model, self.provider, self.graph = self.stack.model, self.stack.provider, self.stack.graph
        self.model_type = type(self.model)
        self.model_forward_fn = self.model.forward.__func__
        self.graph_forward_fn = self.graph.forward.__func__
        self.graph_installed_fn = self.graph.installed.__func__
        self.graph_original = self.graph.original
        self.idle_model_slots = {name: (name in self.model.__dict__, self.model.__dict__.get(name))
                                 for name in ("forward", "_forward_front")}
        self.graph_controls_fn = self.model.graph_controls.__func__
        self.model_module = _loaded(self.model_type.__module__)
        if (self.model_forward_fn.__qualname__ != "GameLiveControlledNR.forward" or
                _plain(self.model_forward_fn).__module__ != self.model_module.__name__ or
                Path(_plain(self.model_forward_fn).__code__.co_filename).resolve(strict=True) !=
                Path(self.model_module.__file__).resolve(strict=True)):
            raise RuntimeError("History requires the actual controlled model forward")
        self.geometry = modes.geometry
        self.torch, self.triton = _loaded("torch"), _loaded("triton")
        self.fullsize = _loaded(type(modes).__module__)
        game_dir = Path(self.fullsize.__file__).resolve(strict=True).parent
        self.live = _actual_module("nr_game_history_fused", game_dir / "nr_game_history_fused.py")
        self.fused = _actual_module("fused", Path(self.live.FUSED) / "fused.py")
        self.sampling, self.temporal = _loaded("nr_backend.sampling"), _loaded("nr_backend.temporal")
        self.reciprocal_module = _loaded("nr_backend.reciprocal")
        self.policy, self.execution = _loaded("nr_backend.unround_policy"), _loaded("nr_backend.execution")
        self.reference_game = self.live.warp_history_fused
        self.reference_warp = self.sampling.warp_history_normalized
        self.reference_callees = {name: getattr(self.sampling, name) for name in (
            "_sample_five_axes", "half_texture_linear", "half_texture_normalized",
            "_half_texture_counts", "_integer_half_away", "_axis_weights",
            "_axis_weights_fraction", "fma32")}
        _reviewed("game_sampler", self.live, self.reference_game)
        _reviewed("fused", self.fused, self.fused._axis, self.fused._prepare_axes,
                  self.fused._half_texture, self.fused._five_tap)
        _reviewed("sampling", self.sampling, self.reference_warp, *self.reference_callees.values())
        _reviewed("temporal", self.temporal, self.temporal.MotionNR.forward)
        _reviewed("reciprocal", self.reciprocal_module,
                  self.reciprocal_module.NativeReciprocalTable.forward,
                  self.reciprocal_module.NativeDimensionReciprocalTable.forward)
        _reviewed("policy", self.policy, self.policy.selected_game_variant)
        _reviewed("execution", self.execution, self.execution.current_arithmetic_backend)
        backend = Path(self.sampling.__file__).resolve().parent
        if (backend.parts[-3:] != ("experimental", "fp8_unround_overlay", "nr_backend") or
                any(Path(module.__file__).resolve().parent != backend for module in (
                    self.temporal, self.reciprocal_module, self.policy, self.execution)) or
                self.temporal.warp_history_normalized is not self.reference_warp or
                self.temporal.NativeReciprocalTable is not self.reciprocal_module.NativeReciprocalTable or
                self.temporal.NativeDimensionReciprocalTable is not
                self.reciprocal_module.NativeDimensionReciprocalTable):
            raise RuntimeError("History requires one actual fast unround backend and its real callees")
        self.temporal_forward = self.temporal.MotionNR.forward
        kernel_name = "history_numeric_suite_720_v1_kernel"
        if __package__:
            kernel_name = __package__ + "." + kernel_name
        self.kernels = _actual_module(kernel_name, Path(__file__).with_name(
            "history_numeric_suite_720_v1_kernel.py"), require_path=True)
        self.jits = {name: getattr(self.kernels, name) for name in (
            "_prepare_axes", "_five_tap", "_counts_fp32", "_divide")}
        self.kernel_helpers = {name: getattr(self.kernels, name) for name in (
            "_axis", "_inverse", "_exp32", "_half_texture", "_texture_counts")}
        kernel_path = Path(self.kernels.__file__).resolve(strict=True)
        for function in (*self.jits.values(), *self.kernel_helpers.values()):
            actual = _plain(function)
            if (actual.__module__ != self.kernels.__name__ or
                    Path(actual.__code__.co_filename).resolve(strict=True) != kernel_path):
                raise RuntimeError("History candidate JIT/helper is outside its actual snapshot source")
        if (self.kernels.triton is not self.triton or self.kernels.tl is not _loaded("triton.language") or
                self.fused.triton is not self.triton or self.fused.torch is not self.torch or
                self.sampling.torch is not self.torch):
            raise RuntimeError("History actual kernel/tensor runtime aliases differ from the owner")
        self.fused_callees = {name: getattr(self.fused, name) for name in (
            "prepare_axes", "sample_five", "_axis", "_prepare_axes", "_half_texture", "_five_tap")}
        self.post = _loaded("post_attention_k8_combined_v1")
        self.dense = self.provider.__dict__.get("dense")
        if (not isinstance(self.dense, MethodType) or self.dense.__self__ is not self.provider or
                self.dense.__func__.__module__ != "c512_k8_joint_scope_720_v1"):
            raise RuntimeError("History requires the actual joint C512/pre-K8 provider")
        self.delegated = _closure(self.dense.__func__, "delegated_dense")
        if (not isinstance(self.delegated, MethodType) or self.delegated.__self__ is not self.provider or
                self.delegated.__func__.__module__ != "c512_qkv_library_16_v1" or
                _closure(self.dense.__func__, "model") is not self.model):
            raise RuntimeError("Joint K8 must delegate to the actual sixteen-weight C512 library")
        self.post_head = self.post._contiguous_cropped_head
        self.c512_calls, self.k8_calls = modes.c512_library_calls, modes.native_k8_calls
        if (self.post_head.__module__ != "c512_k8_joint_scope_720_v1" or
                _closure(self.post_head, "k8_calls") is not self.k8_calls or
                _closure(self.dense.__func__, "k8_calls") is not self.k8_calls or
                _closure(self.delegated.__func__, "calls") is not self.c512_calls):
            raise RuntimeError("History lost its actual C512/K8 dispatch receipts")
        self.targets = _closure(self.delegated.__func__, "targets")
        self.weights = []
        self.c512_blocks = {}
        for side in ("encoder512", "decoder512"):
            blocks = getattr(self.model, side)
            if len(blocks) != 8:
                raise RuntimeError(f"History requires eight owned {side} blocks")
            self.c512_blocks[side] = tuple(blocks)
            for i, block in enumerate(blocks):
                weight, label = block.attention.qkv, f"{side}_{i}"
                target = self.targets.get(id(weight))
                if (tuple(weight.shape) != (512, 1536) or weight.dtype != self.torch.float16 or
                        target is None or target[0] is not weight or target[1] != label):
                    raise RuntimeError(f"History C512 ownership changed: {label}")
                self.weights.append((block.attention, weight, label))
        if (len(self.targets) != 16 or len({id(w) for _, w, _ in self.weights}) != 16 or
                set(self.c512_calls) != {label for _, _, label in self.weights} or
                set(self.k8_calls) != {"pre", "post"}):
            raise RuntimeError("History requires sixteen distinct C512 weights and both K8 sites")
        self.pre_module, self.post_module = self.model.pre, self.model.post
        self.pre_weight = self.pre_module.front_weight
        self.post_weight = self.post_module.head_weight
        self.reciprocal, self.dimensions = self.model.reciprocal, self.model.dimension_reciprocal
        if (type(self.reciprocal) is not self.reciprocal_module.NativeReciprocalTable or
                type(self.dimensions) is not self.reciprocal_module.NativeDimensionReciprocalTable or
                "forward" in self.reciprocal.__dict__ or "forward" in self.dimensions.__dict__):
            raise RuntimeError("History tables must be the actual owned reference modules")
        self.values, self.dimension_values = self.reciprocal.values, self.dimensions.values
        self.scalar_forward = self.reciprocal_module.NativeReciprocalTable.forward
        self.dimension_forward = self.reciprocal_module.NativeDimensionReciprocalTable.forward
        self.device = self.values.device
        if self.device.type != "xpu":
            raise RuntimeError("History numerics require the selected XPU owner")
        _tensor(self.torch, "pre K8 weight", self.pre_weight, (16, 32), self.torch.float16, self.device)
        _tensor(self.torch, "post K8 weight", self.post_weight, (32, 8), self.torch.float16, self.device)
        _tensor(self.torch, "scalar table", self.values,
                (0x3F940000 - 0x3F780000,), self.torch.float32, self.device)
        _tensor(self.torch, "dimension table", self.dimension_values, (4096,), self.torch.float32, self.device)
        self.constants = [("reciprocal_table", self.values, _constant(self.values)),
                          ("dimension_table", self.dimension_values, _constant(self.dimension_values)),
                          ("pre_K8_weight", self.pre_weight, _constant(self.pre_weight)),
                          ("post_K8_weight", self.post_weight, _constant(self.post_weight))]
        self.constants.extend((label, weight, _constant(weight)) for _, weight, label in self.weights)
        self.joint_scope = modes._c512_library_scope
        self.sources = dict(candidate=_loaded(__name__), candidate_kernel=self.kernels,
                            fused=self.fused, game_sampler=self.live, sampling=self.sampling,
                            reciprocal=self.reciprocal_module, temporal=self.temporal,
                            policy=self.policy, execution=self.execution, fullsize=self.fullsize,
                            model=self.model_module,
                            graph_forward=_loaded(self.graph_forward_fn.__module__),
                            graph_installed=_loaded(self.graph_installed_fn.__module__),
                            joint=_loaded(self.dense.__func__.__module__),
                            c512_library=_loaded(self.delegated.__func__.__module__), post=self.post)
        _reviewed("joint", self.sources["joint"], self.dense, self.post_head)
        self.source_manifest = _manifest(self.sources)
        self.source_files = {role: module.__file__ for role, module in self.sources.items()}
        self._build_owned_near_callees()
        self.owned_near = (self.near_normalized, self.near_linear, self.near_sample, self.near_warp)
        self.candidate_methods = {name: getattr(type(self), name) for name in (
            "_near_counts", "_near_five", "_near_reciprocal", "_dimensions", "_run_near",
            "_run_fractional", "_launch", "_frame")}
        self.calls = dict.fromkeys((
            "fractional.axes", "fractional.sample_five", "fractional.texture",
            "fractional.group_x", "fractional.group_y", "fractional.total",
            "near_integer.sample_five", "near_integer.texture_reference",
            "near_integer.texture_fp32", "near_integer.group_x",
            "near_integer.group_y", "near_integer.total", "dimension.width", "dimension.height"), 0)
        self.capture_calls = dict.fromkeys(self.calls, 0)
        self.frame_route_calls = dict(fractional=0, near_integer=0, reset_no_warp=0)
        self.frontend_route_calls = dict(capture=0, replay=0, eager_controlled=0)
        self.frame_routes, self.capture_gates = [], []
        self.resources, self._compiled, self._binary_objects, self.actualkernelhash = {}, {}, {}, {}
        self.kernels_by_site, self.table_hashes = {}, {}
        self.domain_checks, self.domain_failures, self.failed_calls = 0, 0, 0
        self.validate_calls, self.dispatch_cache_queries = 0, 0
        self.last_route = self.last_diagnostics = None
        self.last_domain_buffers = None
        self.last_domain_audit = None
        self.native_dimensions = None
        self.native_dimension_owner = None
        self.active, self.retired, self.in_frame = False, False, False
        self._require_fresh()

    def _build_owned_near_callees(self):
        counts = {"_half_texture_counts": self._near_counts}
        self.near_normalized = _clone(self.reference_callees["half_texture_normalized"], counts)
        self.near_linear = _clone(self.reference_callees["half_texture_linear"], counts)
        self.near_sample = _clone(self.reference_callees["_sample_five_axes"], dict(
            half_texture_normalized=self.near_normalized, half_texture_linear=self.near_linear))
        self.near_warp = _clone(self.reference_warp, {"_sample_five_axes": self._near_five})

    def _require_fresh(self):
        self._validate_owner()
        if (self.graph.entries or self.graph.replays or self.modes._session_frames or
                self.model._previous is not None or self.model._next_seed != 0 or
                any(self.c512_calls.values()) or any(self.k8_calls.values())):
            raise RuntimeError("History numeric options must be fixed before any frame/warmup/capture")

    def _validate_owner(self):
        self._validate_fixed_owner()
        current_thread = get_ident()
        if current_thread != self.thread:
            raise RuntimeError(
                f"History numeric CPU thread owner changed: expected={self.thread}, current={current_thread}; "
                "explicit owned Cyberpunk serial handoff is required")

    def _validate_fixed_owner(self):
        from replay_lifecycle_audit_rest_720_v1 import live_guard
        if live_guard(self, 'history', check_thread=False):
            return
        """All original fixed-owner gates, independent of a CPU thread move."""
        self.session._ready()
        modes = self.modes
        if (modes.session is not self.session or
                self.session._stack is not self.stack or self.stack.model is not self.model or
                type(self.model) is not self.model_type or
                self.stack.provider is not self.provider or self.stack.graph is not self.graph or
                modes.geometry is not self.geometry or not modes.controlled or modes.height != H or
                tuple(modes.source) != (H, W) or modes.variant != "unrounded" or
                not modes.c512_qkv_library_720 or not modes.native_k8_720 or
                modes.history_compact_720 or getattr(modes, "post_rgb_tail_720", False) or
                getattr(modes, "vit_head_720", False) or H not in modes.combo_modes or
                self.geometry.mode.model != (H, W) or self.geometry.mode.active != (H, W) or
                self.geometry.mode.internal != INTERNAL or self.geometry.mode.inset != (0, 0) or
                tuple(self.geometry.source) != (H, W) or
                tuple(self.model.PADDED_SIZES.get((H, W), ())) != INTERNAL or
                self.graph.closed or self.graph.model is not self.model or
                self.graph.arithmetic is not self.provider or self.provider.mode != "fp16_xmx" or
                self.model_type.forward is not self.model_forward_fn or
                self.model.graph_controls.__func__ is not self.graph_controls_fn or
                self.graph.forward.__func__ is not self.graph_forward_fn or
                self.graph.installed.__func__ is not self.graph_installed_fn or
                self.graph.original is not self.graph_original or
                self.options.identity + (self._debug,) != self.flags_identity):
            raise RuntimeError("History numeric fixed actual 720p config/graph owner changed (not the CPU thread gate)")
        if (self.provider.__dict__.get("dense") is not self.dense or
                self.post._contiguous_cropped_head is not self.post_head or
                modes.c512_library_calls is not self.c512_calls or modes.native_k8_calls is not self.k8_calls or
                modes._c512_library_scope is not self.joint_scope or
                _closure(self.dense.__func__, "delegated_dense") is not self.delegated or
                _closure(self.dense.__func__, "model") is not self.model or
                _closure(self.dense.__func__, "k8_calls") is not self.k8_calls or
                _closure(self.delegated.__func__, "calls") is not self.c512_calls or
                _closure(self.post_head, "k8_calls") is not self.k8_calls or
                _closure(self.delegated.__func__, "targets") is not self.targets or
                len(self.targets) != 16 or any(
                    self.targets.get(id(weight), (None, None))[0] is not weight or
                    self.targets[id(weight)][1] != label or module.qkv is not weight
                    for module, weight, label in self.weights) or
                any(len(getattr(self.model, side)) != 8 or any(
                    actual is not expected for actual, expected in zip(getattr(self.model, side), blocks))
                    for side, blocks in self.c512_blocks.items()) or
                any(block.attention is not self.weights[i][0]
                    for i, block in enumerate(self.c512_blocks["encoder512"] + self.c512_blocks["decoder512"])) or
                self.model.pre is not self.pre_module or self.model.post is not self.post_module or
                self.model.pre.front_weight is not self.pre_weight or
                self.model.post.head_weight is not self.post_weight or
                self.model.reciprocal is not self.reciprocal or self.model.dimension_reciprocal is not self.dimensions or
                self.reciprocal.values is not self.values or self.dimensions.values is not self.dimension_values or
                self.reciprocal_module.NativeReciprocalTable.forward is not self.scalar_forward or
                self.reciprocal_module.NativeDimensionReciprocalTable.forward is not self.dimension_forward or
                "forward" in self.reciprocal.__dict__ or "forward" in self.dimensions.__dict__ or
                any(_constant(tensor) != state for _, tensor, state in self.constants) or
                self.live.warp_history_fused is not self.reference_game or
                self.sampling.warp_history_normalized is not self.reference_warp or
                self.temporal.MotionNR.forward is not self.temporal_forward or
                any(getattr(self.sampling, name) is not function for name, function in self.reference_callees.items()) or
                any(getattr(self.fused, name) is not function for name, function in self.fused_callees.items()) or
                any(getattr(self.kernels, name) is not function for name, function in
                    dict(self.jits, **self.kernel_helpers).items())):
            raise RuntimeError("History actual callee/weight/table/constant ownership changed")
        if ((self.near_normalized, self.near_linear, self.near_sample, self.near_warp) != self.owned_near or
                self.near_normalized.__code__ is not self.reference_callees["half_texture_normalized"].__code__ or
                self.near_linear.__code__ is not self.reference_callees["half_texture_linear"].__code__ or
                self.near_sample.__code__ is not self.reference_callees["_sample_five_axes"].__code__ or
                self.near_warp.__code__ is not self.reference_warp.__code__ or
                self.near_sample.__globals__["fma32"] is not self.reference_callees["fma32"] or
                self.near_warp.__globals__["_axis_weights_fraction"] is not
                self.reference_callees["_axis_weights_fraction"] or
                getattr(self.near_normalized.__globals__["_half_texture_counts"], "__func__", None) is not
                type(self)._near_counts or
                self.near_linear.__globals__["_half_texture_counts"] != self._near_counts or
                self.near_sample.__globals__["half_texture_normalized"] is not self.near_normalized or
                self.near_sample.__globals__["half_texture_linear"] is not self.near_linear or
                self.near_warp.__globals__["_sample_five_axes"] != self._near_five or
                (self.native_dimension_owner is not None and
                 self.native_dimensions is not self.native_dimension_owner)):
            raise RuntimeError("History local callee aliases / dimension constant owner changed")
        if any(getattr(getattr(self, name), "__func__", None) is not function or
               getattr(getattr(self, name), "__self__", None) is not self
               for name, function in self.candidate_methods.items()):
            raise RuntimeError("History owned candidate dispatcher/callee changed")
        for role, module in self.sources.items():
            # Frozen manifest was authenticated at initialization/preflight.
            # No path resolution, disk reads or hashing in the frame path.
            if (sys.modules.get(module.__name__) is not module or
                    module.__file__ != self.source_files[role]):
                raise RuntimeError(f"History source identity changed after binding: {role}")

    def _require_serial_idle(self):
        if self.in_frame or _IN_FLIGHT is not None:
            raise RuntimeError("History serial thread handoff refused: in_frame or global history frame is active")
        if self.torch.xpu.is_current_stream_capturing():
            raise RuntimeError("History serial thread handoff refused during XPU stream capture")
        if (self.temporal.warp_history_normalized is not self.reference_warp or
                self.session.__dict__.get("_numeric_model_forward_registry_720") or
                any((name in self.model.__dict__) != present or
                    self.model.__dict__.get(name) is not saved
                    for name, (present, saved) in self.idle_model_slots.items())):
            raise RuntimeError("History serial thread handoff refused: temporary sampler/model forward hooks are active or changed")
        dataflow = _loaded("quantization_dataflow_v1")
        if any(f"{self.kernels.__name__}.{name}" in dataflow.CONTRACTS for name in
               ("_prepare_axes", "_five_tap", "_counts_fp32", "_divide")):
            raise RuntimeError("History serial thread handoff refused: active/colliding history JIT contracts")

    def _transfer_serial_thread(self, owner, *, serial_guard, previous_thread):
        """Private child transaction; only the actual suite handoff may call it."""
        old_thread = self.thread
        try:
            module = _loaded("numeric_cleanup_suite_720_v1")
            caller = sys._getframe(1)
            if (type(owner) is not module.NumericCleanupCounter or
                    caller.f_code is not module.NumericCleanupCounter.transfer_serial_thread.__code__ or
                    caller.f_locals.get("self") is not owner or
                    getattr(self.modes, "_numeric_cleanup_owner", None) is not owner or
                    owner.children.get("history") is not self or owner.session is not self.session or
                    not serial_guard._is_owned()):
                raise RuntimeError("History serial thread handoff requires its owned suite transaction and held process guard")
            if type(previous_thread) is not int or previous_thread != old_thread:
                raise RuntimeError(
                    f"History serial thread handoff CPU owner mismatch: expected={old_thread}, previous={previous_thread!r}")
            self._require_serial_idle()
            # Never use a thread update to skip graph/config/callee/constant gates.
            self._validate_fixed_owner()
            self.thread = get_ident()
            self._validate_replay()
        except BaseException:
            self.thread = old_thread
            self.session._failed = True
            raise

    def _validate_replay(self):
        return self.validate()

    def validate(self):
        """No launch or device synchronization; usable outside temporary hooks."""
        try:
            if not self.active or self.retired or self.session.__dict__.get(
                    "_history_numeric_suite_720") is not self:
                raise RuntimeError("History numeric counter is not this session's active child")
            self._validate_owner()
            # Do not require this child to be the outermost installed wrapper.
            if (self.wrapper.__nr_numeric_original_installed__ is not self.previous_installed or
                    not _installed_contains(self.session._installed, self.wrapper)):
                raise RuntimeError("History installed-chain provenance changed")
            self.validate_calls += 1
            return True
        except BaseException:
            self.session._failed = True
            raise

    def _sample_buffers(self, image, debug):
        torch = self.torch
        numerator = torch.empty((H, W, 3), device=self.device, dtype=torch.float32)
        normalized = torch.empty_like(numerator)
        reciprocal = torch.empty((H, W), device=self.device, dtype=torch.float32)
        invalid = torch.empty((H, W), device=self.device, dtype=torch.int32)
        taps = torch.empty((5, H, W, 3), device=self.device, dtype=torch.float32) if debug else numerator
        return numerator, reciprocal, normalized, invalid, taps

    def _dimensions(self, dimension):
        if dimension not in (W, H):
            raise ValueError("History dimension constants are only defined for 1280x720")
        self._hit("dimension.width" if dimension == W else "dimension.height")
        if self.options.history_dimension_rcp == "native":
            if self.native_dimensions is None:
                raise RuntimeError("History native dimension constants require preflight")
            return self.native_dimensions[0 if dimension == W else 1]
        return self.dimensions(dimension)

    def _dimension_args(self, *, dispatch=False):
        if self.options.history_coord == "direct_pixel":
            # Unread pointer arguments preserve the fractional buffer signature.
            return self.dimension_values[W - 1], self.dimension_values[H - 1]
        if dispatch:
            return self._dimensions(W), self._dimensions(H)
        if self.options.history_dimension_rcp == "native":
            return self.native_dimensions
        return self.dimensions(W), self.dimensions(H)

    def _specializations(self):
        result = {"axes_fp16", "axes_fp32", "five_tap_debug_false", "five_tap_debug_true"}
        if self.options.history_value == "fp32_all_paths":
            result.add("near_counts_fp32")
        if self.options.history_reciprocal == "native":
            result.add("near_div_rn")
        return result

    def _binary_receipt(self, kernel):
        binary = getattr(kernel, "kernel", None)
        if type(binary) is not bytes or not binary:
            raise RuntimeError("Unavailable actual history compiled binary bytes")
        format_name = getattr(kernel.metadata, "binary_ext", None)
        if format_name is None:
            formats = [name for name in ("zebin", "spv") if kernel.asm.get(name) == binary]
            if len(formats) != 1:
                raise RuntimeError("Unknown actual Intel history binary format")
            format_name = formats[0]
        if format_name not in ("zebin", "spv"):
            raise RuntimeError(f"Unreviewed actual history binary format: {format_name}")
        if type(getattr(kernel, "n_spills", None)) is not int or kernel.n_spills != 0:
            raise RuntimeError("History zero-spill metadata unknown or nonzero")
        return dict(actualbinary_sha256=hashlib.sha256(binary).hexdigest(),
                    actualbinary_bytes=len(binary), actualbinary_format=format_name,
                    kernel_hash=kernel.hash, spills=kernel.n_spills,
                    registers=kernel.n_regs, shared_bytes=kernel.metadata.shared)

    def preflight(self):
        """Luna-only compile/load of exact specializations, no kernel launches."""
        try:
            self.validate()
            self._require_fresh()
            if self.torch.xpu.is_current_stream_capturing():
                raise RuntimeError("History preflight must precede capture")
            if _manifest(self.sources) != self.source_manifest:
                raise RuntimeError("Actual history source hashes changed before preflight")
            if set(self._compiled) == self._specializations():
                return self.snapshot()
            screen = _loaded("spill_preflight_v1") if "spill_preflight_v1" in sys.modules else import_module("spill_preflight_v1")
            self.sources["spill_preflight"] = screen
            self.source_manifest["spill_preflight"] = _manifest({"screen": screen})["screen"]
            self.source_files["spill_preflight"] = screen.__file__
            torch = self.torch
            with torch.inference_mode(False):
                scalar_bytes = self.values.detach().cpu().numpy().tobytes()
                dimension_bytes = self.dimension_values.detach().cpu().numpy().tobytes()
                self.table_hashes = dict(reciprocal=hashlib.sha256(scalar_bytes).hexdigest(),
                                        dimension_reciprocal=hashlib.sha256(dimension_bytes).hexdigest())
                if (self.table_hashes["reciprocal"] != self.reciprocal_module.RECIPROCAL_SHA256 or
                        self.table_hashes["dimension_reciprocal"] != self.reciprocal_module.NativeDimensionReciprocalTable.SHA256):
                    raise RuntimeError("Owned history reference table bytes failed authentication")
                if self.options.history_dimension_rcp == "native" and self.native_dimensions is None:
                    self.native_dimensions = (torch.tensor(1.0 / W, device=self.device, dtype=torch.float32),
                                              torch.tensor(1.0 / H, device=self.device, dtype=torch.float32))
                    self.native_dimension_owner = self.native_dimensions
                    for name, tensor, extent in zip(("native_width_rcp", "native_height_rcp"), self.native_dimensions, (W, H)):
                        _tensor(torch, name, tensor, (), torch.float32, self.device)
                        self.constants.append((name, tensor, _constant(tensor)))
                        raw = tensor.detach().cpu().numpy().tobytes()
                        if raw != struct.pack("<f", 1.0 / extent):
                            raise RuntimeError("History native immutable FP32 dimension bits differ from 1/extent")
                        self.table_hashes[name] = dict(sha256=hashlib.sha256(raw).hexdigest(), fp32_le_hex=raw.hex())
                iw, ih = self._dimension_args()
                image = torch.empty((H, W, 3), device=self.device, dtype=torch.float16)
                axes = torch.empty((10, H, W), device=self.device, dtype=torch.float32)
                axis_invalid = torch.empty((H, W), device=self.device, dtype=torch.int32)
                native = self.options.history_reciprocal == "native"
                direct = self.options.history_coord == "direct_pixel"
                fp32 = self.options.history_value != "reference"
                choices = []
                for dtype, label in ((torch.float16, "axes_fp16"), (torch.float32, "axes_fp32")):
                    motion = torch.empty((H, W, 2), device=self.device, dtype=dtype)
                    params = (motion, iw, ih, self.values, axes, axis_invalid, H, W, AXIS_BLOCK, native, direct)
                    choices.append((label, self.jits["_prepare_axes"], params, H * W, AXIS_BLOCK))
                for debug in (False, True):
                    buffers = self._sample_buffers(image, debug)
                    params = (image, axes, iw, ih, self.values, *buffers, H, W, debug, SAMPLE_BLOCK, fp32, native, direct)
                    choices.append((f"five_tap_debug_{str(debug).lower()}", self.jits["_five_tap"], params, H * W * 3, SAMPLE_BLOCK))
                if self.options.history_value == "fp32_all_paths":
                    cx = torch.empty((H, W), device=self.device, dtype=torch.int64)
                    cy = torch.empty_like(cx)
                    out = torch.empty((H, W, 3), device=self.device, dtype=torch.float32)
                    choices.append(("near_counts_fp32", self.jits["_counts_fp32"],
                                    (image, cx, cy, out, H, W, SAMPLE_BLOCK), H * W * 3, SAMPLE_BLOCK))
                if native:
                    value = torch.empty((H, W), device=self.device, dtype=torch.float32)
                    out = torch.empty_like(value)
                    invalid = torch.empty((H, W), device=self.device, dtype=torch.int32)
                    choices.append(("near_div_rn", self.jits["_divide"],
                                    (value, out, invalid, self.values, H * W, DIVIDE_BLOCK), H * W, DIVIDE_BLOCK))
                for label, jit, params, count, block in choices:
                    grid = (self.triton.cdiv(count, block),)
                    _, kernel, gate = screen.select(jit, [(block,)], lambda _: params,
                                                    lambda _: grid, num_warps=4, enable_fp_fusion=False)
                    resource = self._binary_receipt(kernel)
                    resource.update(resource_gate=gate, source_kernel=f"{jit.fn.__module__}.{jit.fn.__name__}",
                                    grid=list(grid), block=block, num_warps=4, enable_fp_fusion=False,
                                    options=asdict(self.options), specialization=label,
                                    tensor_arguments=[dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype))
                                                      for t in params if isinstance(t, torch.Tensor)],
                                    constexpr_arguments=[p for p in params if type(p) in (int, bool)])
                    self.resources[label], self._compiled[label] = resource, kernel
                    self._binary_objects[label] = kernel.kernel
            return self.snapshot()
        except BaseException:
            self.session._failed = True
            raise

    def _launch(self, label, jit, params, count, block):
        if label not in self._compiled:
            raise RuntimeError(f"History specialization needs preflight: {label}")
        grid = (self.triton.cdiv(count, block),)
        expected = self._compiled[label]
        # Exact actual arguments must select the screened object BEFORE launch.
        self.dispatch_cache_queries += 1
        ready = jit.warmup(*params, grid=grid, num_warps=4, enable_fp_fusion=False)
        from replay_lifecycle_audit_rest_720_v1 import history_binary
        history_binary(self, label, ready)
        launched = jit[grid](*params, num_warps=4, enable_fp_fusion=False)
        history_binary(self, label, launched, launched=True)
        self.actualkernelhash[label] = dict(kernel_hash=launched.hash,
                                           actualbinary_sha256=self.resources[label]["actualbinary_sha256"])
        return launched

    def _hit(self, site, count=1):
        self.calls[site] += count
        if self.torch.xpu.is_current_stream_capturing():
            self.capture_calls[site] += count

    def _domain_gate(self, invalid, site):
        self.domain_checks += 1
        if bool(invalid.any()):
            self.domain_failures += 1
            raise ValueError(f"History reciprocal input outside the reviewed FP32 interval: {site}")

    def _near_reciprocal(self, value):
        sites = ("group_x", "group_y", "total")
        if self.near_inverse_index >= len(sites):
            raise RuntimeError("Near-integer five-tap route made an unexpected reciprocal call")
        site = "near_integer." + sites[self.near_inverse_index]
        self.near_inverse_index += 1
        _tensor(self.torch, site, value, (H, W), self.torch.float32, self.device)
        if self.options.history_reciprocal == "native":
            output = self.torch.empty_like(value)
            invalid = self.torch.empty((H, W), device=self.device, dtype=self.torch.int32)
            self._launch("near_div_rn", self.jits["_divide"],
                         (value, output, invalid, self.values, H * W, DIVIDE_BLOCK), H * W, DIVIDE_BLOCK)
            self._domain_gate(invalid, site)
            self.kernels_by_site[site] = self.resources["near_div_rn"]["kernel_hash"]
        else:
            # Actual reference forward performs its original input-domain bool.
            output = self.reciprocal(value)
            self.domain_checks += 1
        self._hit(site)
        return output

    def _near_counts(self, image, cx, cy):
        if image is not self.model._previous:
            raise RuntimeError("Near texture callee lost the owned raw private history")
        _tensor(self.torch, "near cx", cx, (H, W), self.torch.int64, self.device)
        _tensor(self.torch, "near cy", cy, (H, W), self.torch.int64, self.device)
        self.near_texture_count += 1
        if self.options.history_value == "fp32_all_paths":
            value = self.torch.empty((H, W, 3), device=self.device, dtype=self.torch.float32)
            self._launch("near_counts_fp32", self.jits["_counts_fp32"],
                         (image, cx, cy, value, H, W, SAMPLE_BLOCK), H * W * 3, SAMPLE_BLOCK)
            self._hit("near_integer.texture_fp32")
            self.kernels_by_site["near_integer.texture_fp32"] = self.resources["near_counts_fp32"]["kernel_hash"]
        else:
            value = self.reference_callees["_half_texture_counts"](image, cx, cy)
            self._hit("near_integer.texture_reference")
        if self._debug:
            self.near_taps.append(value)
        return value

    def _near_five(self, image, axis_x, axis_y, *, return_components=False,
                   reciprocal_source=None, normalized_scale=None):
        if not return_components or getattr(reciprocal_source, "__self__", None) is not self:
            raise RuntimeError("Near-integer requires the owned raw five-tap callee")
        result = self.near_sample(image, axis_x, axis_y, return_components=True,
                                  reciprocal_source=reciprocal_source, normalized_scale=normalized_scale)
        self._hit("near_integer.sample_five")
        if self._debug:
            if len(self.near_taps) != 5:
                raise RuntimeError("Near DEBUG must retain all five actual taps")
            numerator, reciprocal = result
            self.last_diagnostics = dict(axis_x=axis_x, axis_y=axis_y, numerator=numerator,
                                         reciprocal=reciprocal, normalized=numerator * reciprocal[..., None],
                                         taps=self.torch.stack([self.near_taps[i] for i in (1, 0, 2, 3, 4)]))
        return result

    def _run_near(self, image, motion):
        self.near_inverse_index, self.near_texture_count, self.near_taps = 0, 0, []
        if self.options.history_coord == "direct_pixel":
            torch = self.torch
            y, x = torch.meshgrid(torch.arange(H, device=self.device, dtype=torch.float32),
                                  torch.arange(W, device=self.device, dtype=torch.float32), indexing="ij")
            mv = motion.half().float()
            axis = self.reference_callees["_axis_weights"]
            result = self._near_five(image, axis((x + .5) + mv[..., 0], self._near_reciprocal),
                                     axis((y + .5) + mv[..., 1], self._near_reciprocal),
                                     return_components=True, reciprocal_source=self._near_reciprocal,
                                     normalized_scale=None)
        else:
            result = self.near_warp(image, motion, return_components=True,
                                    reciprocal_source=self._near_reciprocal, dimension_reciprocal=self._dimensions)
        if self.near_inverse_index != 3 or self.near_texture_count != 5:
            raise RuntimeError("Near-integer missed the actual two groups / total / five textures")
        return result

    def _run_fractional(self, image, motion):
        torch = self.torch
        iw, ih = self._dimension_args(dispatch=True)
        if not motion.is_contiguous():
            motion = motion.contiguous()
        axes = torch.empty((10, H, W), device=self.device, dtype=torch.float32)
        axis_invalid = torch.empty((H, W), device=self.device, dtype=torch.int32)
        native = self.options.history_reciprocal == "native"
        direct = self.options.history_coord == "direct_pixel"
        fp32 = self.options.history_value != "reference"
        label = "axes_fp16" if motion.dtype == torch.float16 else "axes_fp32"
        self._launch(label, self.jits["_prepare_axes"],
                     (motion, iw, ih, self.values, axes, axis_invalid, H, W, AXIS_BLOCK, native, direct), H * W, AXIS_BLOCK)
        self._hit("fractional.axes")
        self._hit("fractional.group_x")
        self._hit("fractional.group_y")
        numerator, reciprocal, normalized, invalid, taps = self._sample_buffers(image, self._debug)
        label = f"five_tap_debug_{str(self._debug).lower()}"
        self._launch(label, self.jits["_five_tap"],
                     (image, axes, iw, ih, self.values, numerator, reciprocal, normalized, invalid, taps,
                      H, W, self._debug, SAMPLE_BLOCK, fp32, native, direct), H * W * 3, SAMPLE_BLOCK)
        # Same INVALID stores as reference, with no extra per-frame host read.
        self.last_domain_buffers = (axis_invalid, invalid)
        self.last_domain_audit = None
        self._hit("fractional.sample_five")
        self._hit("fractional.texture", 5)
        self._hit("fractional.total")
        for site in ("fractional.sample_five", "fractional.texture", "fractional.total"):
            self.kernels_by_site[site] = self.resources[label]["kernel_hash"]
        for site in ("fractional.axes", "fractional.group_x", "fractional.group_y"):
            self.kernels_by_site[site] = self.resources["axes_fp16" if motion.dtype == torch.float16 else "axes_fp32"]["kernel_hash"]
        if self._debug:
            self.last_diagnostics = dict(axes=axes, axis_invalid=axis_invalid, numerator=numerator,
                                         reciprocal=reciprocal, normalized=normalized, invalid=invalid, taps=taps)
        return numerator, reciprocal

    def __call__(self, image, motion, *, return_components=False,
                 reciprocal_source, dimension_reciprocal):
        try:
            if (not self.in_frame or _IN_FLIGHT is not self or
                    self.temporal.warp_history_normalized is not self or not return_components or
                    image is not self.model._previous or reciprocal_source is not self.reciprocal or
                    dimension_reciprocal is not self.dimensions):
                raise RuntimeError("History dispatcher requires its actual owned fused raw-history call")
            _tensor(self.torch, "private image", image, (H, W, 3), self.torch.float16, self.device)
            if (tuple(motion.shape) != (H, W, 2) or motion.device != self.device or
                    motion.dtype not in (self.torch.float16, self.torch.float32)):
                raise ValueError("Expected same-device 720p FP16/FP32 pixel motion")
            if self.torch.xpu.is_current_stream_capturing():
                raise RuntimeError("History sampling must remain outside the frozen body graph")
            # Unchanged real game predicate, including the whole-frame bool sync.
            rounded_motion = motion.half().float()
            near = bool(((rounded_motion - rounded_motion.round()).abs() <= 1 / 256).all())
            self.last_route, self.last_diagnostics = ("near_integer" if near else "fractional"), None
            result = self._run_near(image, motion) if near else self._run_fractional(image, motion)
            self.frame_route_calls[self.last_route] += 1
            self.frame_history_calls += 1
            if self.frame_history_calls != 1:
                raise RuntimeError("Expected exactly one history warp in an actual temporal frame")
            return result
        except BaseException:
            self.failed_calls += 1
            self.session._failed = True
            raise

    def _require_front(self):
        front = self.model.__dict__.get("_forward_front")
        if (getattr(front, "__self__", None) is self.graph and
                getattr(front, "__func__", None) is self.graph_forward_fn):
            return
        if (getattr(front, "__qualname__", None) != "GameLiveControlledNR.graph_controls.<locals>.controlled_front" or
                getattr(_closure(front, "original"), "__self__", None) is not self.graph or
                getattr(_closure(front, "original"), "__func__", None) is not self.graph_forward_fn):
            raise RuntimeError("History requires the actual installed owned body graph/control front")

    def _frame(self, module, original, *args, **kwargs):
        self._validate_replay()
        from replay_lifecycle_audit_rest_720_v1 import history_ready
        if (module is not self.model or not self.in_frame or
                self.temporal.warp_history_normalized is not self or
                self.policy.ENABLED != self.policy.FAMILIES or
                self.execution.current_arithmetic_backend() != "triton" or kwargs.get("progress") is not None or
                not history_ready(self)):
            raise RuntimeError("History mode requires preflight, fused history and the owned unrounded graph route")
        self._require_front()
        before_entries, before_replays = set(self.graph.entries), self.graph.replays
        before_c512, before_k8 = dict(self.c512_calls), dict(self.k8_calls)
        before_sites = dict(self.calls)
        self.frame_history_calls, self.last_route = 0, None
        seed_before = self.model._next_seed
        result = original(*args, **kwargs)
        if self.frame_history_calls == 0:
            self.last_route = "reset_no_warp"
            self.frame_route_calls[self.last_route] += 1
        delta_sites = {site: count - before_sites[site] for site, count in self.calls.items()}
        expected = dict.fromkeys(self.calls, 0)
        if self.last_route == "fractional":
            expected.update({"fractional.axes": 1, "fractional.sample_five": 1,
                             "fractional.texture": 5, "fractional.group_x": 1,
                             "fractional.group_y": 1, "fractional.total": 1})
        elif self.last_route == "near_integer":
            expected.update({"near_integer.sample_five": 1, "near_integer.group_x": 1,
                             "near_integer.group_y": 1, "near_integer.total": 1})
            expected["near_integer.texture_fp32" if self.options.history_value == "fp32_all_paths"
                     else "near_integer.texture_reference"] = 5
        if self.last_route != "reset_no_warp" and self.options.history_coord == "reference":
            expected["dimension.width"] = expected["dimension.height"] = 1
        if delta_sites != expected:
            raise RuntimeError("History frame missed its frozen per-route mode/site selection")
        added = set(self.graph.entries) - before_entries
        replays = self.graph.replays - before_replays
        frontend = "capture" if added else "replay" if replays else "eager_controlled"
        self.frontend_route_calls[frontend] += 1
        if added:
            dc512 = {k: v - before_c512[k] for k, v in self.c512_calls.items()}
            dk8 = {k: v - before_k8[k] for k, v in self.k8_calls.items()}
            # These are actual Python BUILD totals, NOT capture-only +3 claims.
            builds = set(dc512.values()) | set(dk8.values())
            passed = len(builds) == 1 and next(iter(builds)) >= len(added) and replays >= len(added)
            self.capture_gates.append(dict(new_entries=len(added), body_build_calls_c512=dc512,
                                           body_build_calls_k8=dk8, actual_replays=replays,
                                           history_capture_calls=0, passed=passed,
                                           count_scope="warmup_and_capture_build_not_capture_only"))
            if not passed:
                raise RuntimeError("New history-mode body graph missed actual C512/K8 owned sites")
        self.frame_routes.append(dict(frame=len(self.frame_routes), history_route=self.last_route,
                                      frontend_route=frontend, new_entries=len(added), replay_delta=replays,
                                      seed_before=seed_before, seed_after=self.model._next_seed,
                                      options=asdict(self.options), history_calls=self.frame_history_calls,
                                      site_calls=delta_sites, selection_gate=True))
        return result

    def snapshot(self):
        """CPU JSON receipt; diagnostics tensors remain in last_diagnostics only."""
        result = dict(options=asdict(self.options), flags_identity=list(self.flags_identity),
                    active=self.active, retired=self.retired, model=[H, W], internal=list(INTERNAL),
                    debug=self._debug, sources={k: dict(v) for k, v in self.source_manifest.items()},
                    expected_sites=list(self.calls), calls=dict(self.calls), capture_calls=dict(self.capture_calls),
                    frame_route_calls=dict(self.frame_route_calls), frontend_route_calls=dict(self.frontend_route_calls),
                    frame_routes=list(self.frame_routes), resources=dict(self.resources),
                    actualkernelhash=dict(self.actualkernelhash), kernels_by_site=dict(self.kernels_by_site),
                    capture_gates=list(self.capture_gates), body_graph_replays=self.graph.replays,
                    c512_calls=dict(self.c512_calls), k8_calls=dict(self.k8_calls),
                    preflight_complete=set(self._compiled) == self._specializations(),
                    required_specializations=sorted(self._specializations()), table_hashes=dict(self.table_hashes),
                    constant_guards={name: _constant_receipt(state) for name, _, state in self.constants},
                    domain_checks=self.domain_checks, domain_failures=self.domain_failures,
                    fractional_invalid_audit=self.last_domain_audit,
                    failed_calls=self.failed_calls, outside_body_graph=True,
                    value_paths=dict(fractional="reference" if self.options.history_value == "reference" else "fp32",
                                     near_integer="fp32" if self.options.history_value == "fp32_all_paths" else "reference"),
                    scalar_inverse="tl.div_rn_fp32_1_over_x" if self.options.history_reciprocal == "native" else "reference_table",
                    dimension_effective="unused_direct_pixel" if self.options.history_coord == "direct_pixel" else self.options.history_dimension_rcp,
                    near_integer_callee="nr_backend.sampling.warp_history_normalized/_sample_five_axes/local_texture_aliases",
                    owned_entry=dict(frame_slot="nr_backend.temporal.warp_history_normalized",
                                     parent_binding="nr_game_history_fused.warp_history_fused",
                                     dispatcher=f"{__name__}.HistoryNumericCounter720.__call__",
                                     near_sample_code_source=self.source_manifest["sampling"]["path"]),
                    fractional_main_launches=2, fractional_main_launches_saved=0,
                    fractional_added_domain_syncs=0,
                    fractional_domain_gate="finite_motion_clamped_cubic_bounds_and_stored_INVALID; report audits last fractional only",
                    near_counts_launches_per_warp=5 if self.options.history_value == "fp32_all_paths" else 0,
                    near_div_launches_per_warp=3 if self.options.history_reciprocal == "native" else 0,
                    net_launch_delta="unmeasured_including_ATen_and_domain_checks",
                    validation_overhead=dict(validate_calls=self.validate_calls,
                                             dispatch_warmup_cache_queries=self.dispatch_cache_queries,
                                             timed_validation_included=True, frame_disk_reads=0,
                                             immutable_constant_host_readbacks=0),
                    constants_lifetime="counter_and_retired_session_until_main_closes_graph",
                    gpu_numeric_quality_performance="unverified")
        return json.loads(json.dumps(result, allow_nan=False))

    def report(self):
        """Explicit, non-timed audit; never called by validate/dispatch/snapshot.

        Audits only the retained LAST fractional flags, not every stream frame.
        Luna should call it between numerical test frames, outside timing.
        Source/constant byte audits belong here or preflight, not frame guards.
        """
        actual_sources = _manifest(self.sources)
        if actual_sources != self.source_manifest:
            self.session._failed = True
            raise RuntimeError("Actual history source hashes changed at report audit")
        if self.last_domain_buffers is not None:
            groups, total = (tensor.detach().cpu().numpy() for tensor in self.last_domain_buffers)
            self.last_domain_audit = dict(scope="last_fractional_frame_only",
                                          invalid_groups=int((groups != 0).sum()),
                                          invalid_totals=int((total != 0).sum()))
            if self.last_domain_audit["invalid_groups"] or self.last_domain_audit["invalid_totals"]:
                self.domain_failures += 1
                self.session._failed = True
                raise ValueError("Reported fractional reciprocal INVALID flags are nonzero")
        return self.snapshot()


@contextmanager
def installed(modes, *, history_value="reference", history_reciprocal="table",
              history_dimension_rcp="table", history_coord="reference", debug=False):
    """Schema-compatible child; the main suite owns signatures/final closure."""
    options = _Options(history_value, history_reciprocal, history_dimension_rcp, history_coord)
    if type(debug) is not bool:
        raise ValueError("History DEBUG must be fixed before preflight/capture")
    if options == _Options():
        if debug:
            raise ValueError("DEBUG requires an active history arm; all-reference installed is a no-op")
        yield ReferenceCounter()
        return
    session = modes.session
    if session is None or "_history_numeric_suite_720" in session.__dict__:
        raise RuntimeError("History requires a selected session with no duplicate history child")
    try:
        counter = HistoryNumericCounter720(modes, options, debug)
    except BaseException:
        session._failed = True
        raise
    previous = session._installed
    counter.previous_installed = previous

    @contextmanager
    def wrapper():
        global _IN_FLIGHT
        try:
            counter._validate_replay()
            with previous():
                if (_IN_FLIGHT is not None or counter.in_frame or
                        counter.temporal.warp_history_normalized is not counter.reference_game):
                    raise RuntimeError("History child requires this actual fused per-frame sampler/serial owner")
                original_forward = counter.model.forward
                from numeric_model_forward_720_v1 import (
                    register_forward, unregister_forward, require_forward)
                require_forward(original_forward, MethodType(counter.model_forward_fn, counter.model),
                                counter.model, session)
                had_forward = "forward" in counter.model.__dict__
                saved_forward = counter.model.__dict__.get("forward")
                saved_sampler = counter.temporal.warp_history_normalized
                dataflow = _loaded("quantization_dataflow_v1")
                contracts = {
                    f"{counter.kernels.__name__}._prepare_axes": (("AXES", "INVALID"), ()),
                    f"{counter.kernels.__name__}._five_tap": (
                        ("NUMERATOR", "RECIPROCAL", "NORMALIZED", "INVALID", "TAPS") if debug else
                        ("NUMERATOR", "RECIPROCAL", "NORMALIZED", "INVALID"), ()),
                    f"{counter.kernels.__name__}._counts_fp32": (("OUTPUT",), ()),
                    f"{counter.kernels.__name__}._divide": (("OUTPUT", "INVALID"), ()),
                }
                if any(key in dataflow.CONTRACTS for key in contracts):
                    raise RuntimeError("History top-level JIT contract ownership collision")

                def forward(module, *args, **kwargs):
                    return counter._frame(module, original_forward, *args, **kwargs)

                replacement = MethodType(forward, counter.model)
                register_forward(replacement, original_forward, counter)
                sampler_set = forward_set = contracts_set = flight_set = False
                try:
                    counter.temporal.warp_history_normalized = counter
                    sampler_set = True
                    counter.model.forward = replacement
                    forward_set = True
                    dataflow.CONTRACTS.update(contracts)
                    contracts_set = True
                    _IN_FLIGHT, counter.in_frame = counter, True
                    flight_set = True
                    yield
                finally:
                    valid = ((not forward_set or counter.model.__dict__.get("forward") is replacement) and
                             (not sampler_set or counter.temporal.warp_history_normalized is counter) and
                             (not flight_set or _IN_FLIGHT is counter) and
                             (not contracts_set or all(dataflow.CONTRACTS.get(k) == v for k, v in contracts.items())))
                    counter.in_frame = False
                    if _IN_FLIGHT is counter:
                        _IN_FLIGHT = None
                    if sampler_set:
                        counter.temporal.warp_history_normalized = saved_sampler
                    if forward_set:
                        if had_forward:
                            counter.model.forward = saved_forward
                        else:
                            counter.model.__dict__.pop("forward", None)
                    if contracts_set:
                        for key in contracts:
                            dataflow.CONTRACTS.pop(key, None)
                    unregister_forward(replacement, counter)
                    if not valid:
                        raise RuntimeError("History per-frame local hook/contract ownership changed")
        except BaseException:
            counter.failed_calls += 1
            session._failed = True
            raise

    # contextmanager.__wrapped__ still points to its generator. This explicit
    # link points to the previous *real scope*, for static core-vforward tracing.
    wrapper.__nr_numeric_original_installed__ = previous
    counter.wrapper = wrapper
    session._installed = wrapper
    session._history_numeric_suite_720 = counter
    counter.active = True
    failure = None
    try:
        yield counter
    except BaseException as exc:
        failure = exc
        session._failed = True
        raise
    finally:
        valid = (session._installed is wrapper and not counter.in_frame and
                 session.__dict__.get("_history_numeric_suite_720") is counter and
                 modes.session is session)
        counter.active, counter.retired = False, True
        session._installed = previous
        session.__dict__.pop("_history_numeric_suite_720", None)
        # Main closes the graph after children unwind; keep device constants
        # strongly owned even if the caller discarded its yielded counter.
        session._history_numeric_retired_720 = counter
        session._failed = True
        if not valid:
            error = "History numeric child exited with changed session/hook ownership"
            if failure is None:
                raise RuntimeError(error)
            failure.add_note(error)
