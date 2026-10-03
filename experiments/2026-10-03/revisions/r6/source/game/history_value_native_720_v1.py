"""Default-off, fractional-only history value ablation for actual 720p C512+K8.

Only _half_texture's shared-exponent/INT64 value sum and artificial half
reconstruction change. Coordinates still use 21-bit normalization, 1/256
fractions and the rounded integer cross weight. The four FP32 products add in
the original fixed order, divide by 256, then convert half -> float. Integer
coordinates return f0. Explicit five-tap FMAs, negative weights, reference
reciprocals, all buffers/stores and raw (numerator, reciprocal) stay intact.
This is lossy and GPU/temporal quality and performance remain UNVERIFIED.

Static E/G inspection on 2026-09-30 found the real near-integer callee to be
nr_backend.sampling.warp_history_normalized -> _sample_five_axes ->
half_texture_normalized -> _half_texture_counts -> _integer_half_away.
It still samples five taps; the old game wrapper's single-tap comment is
misleading. This arm keeps that entire path and the whole-frame bool sync.
It does NOT clean all history paths and reduces zero launches (axes + sample).

Luna integration, after selecting a fresh controlled 720p unrounded owner with
c512_qkv_library_720=True and native_k8_720=True, before any frame/capture:

    with installed(owner, history_value_720="fp32_fractional") as candidate:
        candidate.preflight()  # explicit GPU compile/load gate; NOT run here
        # owner.process(..., height=720, variant="unrounded",
        #               history_warp="fused", graph_replay=True, ...)
        # Inspect candidate.receipt(), and close the owner before reuse.

The default installed(owner) is a no-op. The opt-in hook owns the game module's
sampler because FullsizeGameModes reimports/rebinds temporal's callee per frame.
Keep the scope for the entire stream, including reset and body graph replay.
Exit retires the session; captured graphs/private history must not be reused.
No controller/fullsize/front/layout/seed/reset changes or automatic deployment.

Pending Luna gates: actual 720x1280 / internal 768x1280 source/module/config/
model/input hashes; three exact launch specializations with zero spills; 13
reset/history/capture/replay frames with all 16 C512 and pre/post K8 hits;
flag-off byte identity; separate fractional/near-integer routes and timing.
Compare axes/invalid/taps/numerator/reciprocal/normalized/private history on
zero/integer/threshold motions, a single outlier pixel, signed fractional
motion, clamp edges, opposite half signs, negative zero and halfway values.
Then paired 50-frame B-C-C-B P50/P95 and complete continuous-history video,
preserving motion/depth/control/seed/frame/resolution/audio semantics; user
visual review is required. Preflight compiles/loads but never launches kernels.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from importlib import import_module
from pathlib import Path
import sys
from threading import get_ident

import torch
import triton
import triton.language as tl

if __package__:
    from . import fused
else:
    import fused


H, W = 720, 1280
INTERNAL = (768, 1280)
AXIS_BLOCK, BLOCK = 128, 64
_ACTIVE_SCOPE = None
REFERENCE_SHA256 = {
    "fused": "14df7f80a39c99f877993afdb878c701de832b427ca763b9fc3934745361e647",
    "game_sampler": "6e4f73ce30a29ffe3366abe558dbb1ef204726a10a1facab2487f1ff7d48e1f0",
    "sampling": "e95228c1743bdb5013d3c3b68cbb85fae315c6deeb913b6c4a057c76d2ed89f8",
    "reciprocal": "b150d33db7b3fcde300fef118bf2a1d874dd8fe510e40d2941d30a43ae6aee28",
    "temporal": "35c1c1d5c1475ac0e3385dfa76f0e1020109a376a1d233a64e6c2a1036ca1828",
    "policy": "87bf4ba9b9f60e8c10173dc8498d54e23e3387dee0a4485fae08db21dedda827",
    "execution": "8fbfe06ff6bcd94011da4ec4038a9735ada672dd90df2ff793417344fd045462",
}


@triton.jit
def _half_texture_native(IMAGE, px, py, ch, IW, IH,
                         H: tl.constexpr, W: tl.constexpr):
    # Identical to fused._half_texture through f3, including INT64 coordinates.
    iw = tl.load(IW)
    ih = tl.load(IH)
    xc = tl.minimum(W - 0.5, tl.maximum(0.5, px))
    yc = tl.minimum(H - 0.5, tl.maximum(0.5, py))
    u = tl.fma(xc * iw, W, 0.0) * (1.0 / W)
    v = tl.fma(yc * ih, H, 0.0) * (1.0 / H)
    nx = tl.floor(u * 2097152.0).to(tl.int64)
    ny = tl.floor(v * 2097152.0).to(tl.int64)
    cx = ((nx * W + 4096) >> 13) - 128
    cy = ((ny * H + 4096) >> 13) - 128
    cx = tl.minimum((W - 1) * 256, tl.maximum(0, cx))
    cy = tl.minimum((H - 1) * 256, tl.maximum(0, cy))
    ix = (cx >> 8).to(tl.int32)
    iy = (cy >> 8).to(tl.int32)
    jx = tl.minimum(ix + 1, W - 1)
    jy = tl.minimum(iy + 1, H - 1)
    ax = (cx & 255).to(tl.int32)
    ay = (cy & 255).to(tl.int32)
    cross = (ax * ay + 128) >> 8
    k0 = 256 - ax - ay + cross
    k1 = ax - cross
    k2 = ay - cross
    k3 = cross
    a0 = tl.load(IMAGE + (iy * W + ix) * 3 + ch).to(tl.float16)
    a1 = tl.load(IMAGE + (iy * W + jx) * 3 + ch).to(tl.float16)
    a2 = tl.load(IMAGE + (jy * W + ix) * 3 + ch).to(tl.float16)
    a3 = tl.load(IMAGE + (jy * W + jx) * 3 + ch).to(tl.float16)
    f0 = a0.to(tl.float32)
    f1 = a1.to(tl.float32)
    f2 = a2.to(tl.float32)
    f3 = a3.to(tl.float32)
    weighted = ((f0 * k0 + f1 * k1) + f2 * k2) + f3 * k3
    fractional = (weighted * (1.0 / 256.0)).to(tl.float16).to(tl.float32)
    return tl.where(((ax == 0) & (ay == 0)), f0, fractional)


@triton.jit
def _five_tap_native(IMAGE, AXES, IW, IH, TABLE, NUMERATOR, RECIPROCAL, NORMALIZED,
                     INVALID, TAPS, H: tl.constexpr, W: tl.constexpr,
                     DEBUG: tl.constexpr, BLOCK: tl.constexpr):
    # Identical signature/body/stores to fused._five_tap except texture callee.
    start_bits: tl.constexpr = 0x3F780000
    table_size: tl.constexpr = 0x3F940000 - 0x3F780000
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = q < H * W * 3
    p = q // 3
    ch = q % 3
    n = H * W
    bx = tl.load(AXES + p, valid, 0)
    x0 = tl.load(AXES + n + p, valid, 0)
    gx = tl.load(AXES + 2 * n + p, valid, 0)
    x3 = tl.load(AXES + 3 * n + p, valid, 0)
    mx = tl.load(AXES + 4 * n + p, valid, 0)
    by = tl.load(AXES + 5 * n + p, valid, 0)
    y0 = tl.load(AXES + 6 * n + p, valid, 0)
    gy = tl.load(AXES + 7 * n + p, valid, 0)
    y3 = tl.load(AXES + 8 * n + p, valid, 0)
    my = tl.load(AXES + 9 * n + p, valid, 0)
    k0, k1, k2, k3, k4 = x0 * gy, y0 * gx, gx * gy, y3 * gx, x3 * gy
    total = (((k0 + k1) + k2) + k3) + k4
    index = total.to(tl.int32, bitcast=True) - start_bits
    good = (index >= 0) & (index < table_size)
    inverse = tl.load(TABLE + tl.minimum(table_size - 1, tl.maximum(0, index)))
    s1 = _half_texture_native(IMAGE, mx, by - 1.0, ch, IW, IH, H, W)
    s0 = _half_texture_native(IMAGE, bx - 1.0, my, ch, IW, IH, H, W)
    s2 = _half_texture_native(IMAGE, mx, my, ch, IW, IH, H, W)
    s3 = _half_texture_native(IMAGE, mx, by + 2.0, ch, IW, IH, H, W)
    s4 = _half_texture_native(IMAGE, bx + 2.0, my, ch, IW, IH, H, W)
    if DEBUG:
        tl.store(TAPS + q, s0, valid)
        tl.store(TAPS + n * 3 + q, s1, valid)
        tl.store(TAPS + n * 6 + q, s2, valid)
        tl.store(TAPS + n * 9 + q, s3, valid)
        tl.store(TAPS + n * 12 + q, s4, valid)
    value = s1 * k1
    value = tl.fma(s0, k0, value)
    value = tl.fma(s2, k2, value)
    value = tl.fma(s3, k3, value)
    value = tl.fma(s4, k4, value)
    tl.store(NUMERATOR + q, value, valid)
    tl.store(NORMALIZED + q, value * inverse, valid)
    tl.store(RECIPROCAL + p, inverse, valid & (ch == 0))
    tl.store(INVALID + p, (~good).to(tl.int32), valid & (ch == 0))


def _loaded(name):
    module = sys.modules.get(name)
    if module is None:
        raise RuntimeError(f"Expected the owner's already-loaded module: {name}")
    return module


def _closure(function, name):
    cells = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
    if name not in cells:
        raise RuntimeError(f"C512/K8 dispatch is missing its owned {name}")
    return cells[name].cell_contents


def _source_manifest(modules):
    return {role: {"path": str(Path(module.__file__).resolve()),
                   "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
            for role, module in modules.items()}


def _reviewed_source(role, module, *functions):
    """Authenticate the loaded role, independent of the candidate's directory."""
    path = Path(module.__file__).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA256[role]:
        raise RuntimeError(f"Reviewed history source changed: {role}: {path}")
    for function in functions:
        code = getattr(getattr(function, "fn", function), "__code__", None)
        if code is None or Path(code.co_filename).resolve(strict=True) != path:
            raise RuntimeError(f"History callable is outside its authenticated source: {role}")
    return path


def _tensor(name, tensor, shape, dtype, device):
    if (tuple(tensor.shape) != shape or tensor.dtype != dtype or
            tensor.device != device or not tensor.is_contiguous()):
        raise ValueError(f"Unexpected 720p history {name} tensor contract")


def _sample_buffers(image, debug):
    # Keep every allocation in fused.sample_five, even when callers discard it.
    numerator = torch.empty((H, W, 3), device=image.device, dtype=torch.float32)
    normalized = torch.empty_like(numerator)
    reciprocal = torch.empty((H, W), device=image.device, dtype=torch.float32)
    invalid = torch.empty((H, W), device=image.device, dtype=torch.int32)
    taps = torch.empty((5, H, W, 3), device=image.device, dtype=torch.float32) if debug else numerator
    return numerator, reciprocal, normalized, invalid, taps


class HistoryValueNative720:
    """Single-owner fractional sampler; reference near-integer path is intact."""

    def __init__(self, owner, live, *, debug=False):
        if type(debug) is not bool:
            raise ValueError("debug must be a bool fixed before preflight/capture")
        self.owner, self.live, self._debug = owner, live, debug
        self.thread = get_ident()
        self.session = owner.session
        if self.session is None:
            raise RuntimeError("Select the fresh 720p C512+K8 owner before installing")
        self.stack = self.session._stack
        self.model, self.provider, self.graph = self.stack.model, self.stack.provider, self.stack.graph
        self.geometry = owner.geometry
        self.original = live.warp_history_fused
        if (getattr(self.original, "__name__", None) != "warp_history_fused" or
                self.original.__module__ != live.__name__):
            raise RuntimeError("Fused sampler already has an override")
        _reviewed_source("game_sampler", live, self.original)
        _reviewed_source("fused", fused, fused._prepare_axes)
        self.sampling = _loaded("nr_backend.sampling")
        self.temporal = _loaded("nr_backend.temporal")
        self.policy = _loaded("nr_backend.unround_policy")
        self.execution = _loaded("nr_backend.execution")
        self.reference = self.sampling.warp_history_normalized
        self.reference_callees = {name: getattr(self.sampling, name) for name in (
            "_sample_five_axes", "half_texture_normalized", "_half_texture_counts", "_integer_half_away")}
        _reviewed_source("sampling", self.sampling, self.reference, *self.reference_callees.values())
        _reviewed_source("temporal", self.temporal)
        _reviewed_source("policy", self.policy)
        _reviewed_source("execution", self.execution)
        _reviewed_source("reciprocal", _loaded("nr_backend.reciprocal"))
        self.temporal_original = self.temporal.warp_history_normalized
        if self.temporal_original is not self.reference:
            raise RuntimeError("Temporal normalized sampler already has an override")
        backend = Path(self.sampling.__file__).resolve().parent
        if (backend.parts[-3:] != ("experimental", "fp8_unround_overlay", "nr_backend") or
                any(Path(module.__file__).resolve().parent != backend
                    for module in (self.temporal, self.policy, self.execution))):
            raise RuntimeError("History scope requires one isolated fast unrounded backend")
        self.post = _loaded("post_attention_k8_combined_v1")
        self.dense = self.provider.__dict__.get("dense")
        if (getattr(self.dense, "__self__", None) is not self.provider or
                self.dense.__func__.__module__ != "c512_k8_joint_scope_720_v1"):
            raise RuntimeError("Expected the installed joint C512/pre-K8 provider")
        delegated = _closure(self.dense.__func__, "delegated_dense")
        if (getattr(delegated, "__self__", None) is not self.provider or
                delegated.__func__.__module__ != "c512_qkv_library_16_v1"):
            raise RuntimeError("Joint K8 must delegate to the sixteen-target C512 library")
        self.post_head = self.post._contiguous_cropped_head
        if self.post_head.__module__ != "c512_k8_joint_scope_720_v1":
            raise RuntimeError("Expected the joint post-K8 owner")
        self.c512_calls, self.k8_calls = owner.c512_library_calls, owner.native_k8_calls
        if (_closure(delegated.__func__, "calls") is not self.c512_calls or
                _closure(self.dense.__func__, "k8_calls") is not self.k8_calls or
                _closure(self.post_head, "k8_calls") is not self.k8_calls):
            raise RuntimeError("C512/K8 receipts must belong to these live dispatches")
        self.weights = []
        targets = _closure(delegated.__func__, "targets")
        self.c512_delegate, self.c512_targets = delegated, targets
        for family in ("encoder512", "decoder512"):
            blocks = getattr(self.model, family)
            if len(blocks) != 8:
                raise RuntimeError(f"Expected eight owned {family} blocks")
            for index, block in enumerate(blocks):
                weight, label = block.attention.qkv, f"{family}_{index}"
                target = targets.get(id(weight))
                if (tuple(weight.shape) != (512, 1536) or weight.dtype != torch.float16 or
                        target is None or target[0] is not weight or target[1] != label):
                    raise RuntimeError(f"C512 library ownership changed: {label}")
                self.weights.append((block.attention, weight, label))
        if (len(targets) != 16 or len({id(w) for _, w, _ in self.weights}) != 16 or
                set(self.c512_calls) != {label for _, _, label in self.weights} or
                set(self.k8_calls) != {"pre", "post"}):
            raise RuntimeError("Expected sixteen distinct C512 targets and both K8 sites")
        self.pre_weight = self.model.pre.front_weight
        self.reciprocal = self.model.reciprocal
        self.dimensions = self.model.dimension_reciprocal
        self.values, self.dimension_values = self.reciprocal.values, self.dimensions.values
        self.device = self.values.device
        if self.device.type != "xpu":
            raise RuntimeError("720p history owner must use the XPU device")
        _tensor("reciprocal", self.values, (fused.END_BITS - fused.START_BITS,), torch.float32, self.device)
        _tensor("dimensions", self.dimension_values, (4096,), torch.float32, self.device)
        self.versions = (self.values._version, self.dimension_values._version)
        self.table_pointers = (self.values.data_ptr(), self.dimension_values.data_ptr())
        self.axis_jit, self.sample_jit, self.texture_jit = fused._prepare_axes, _five_tap_native, _half_texture_native
        self.joint_scope = owner._c512_library_scope
        self.sources = {
            "candidate": _loaded(__name__), "fused": fused, "game_sampler": live,
            "sampling": self.sampling, "temporal": self.temporal,
            "policy": self.policy, "execution": self.execution,
            "reciprocal": _loaded("nr_backend.reciprocal"),
            "fullsize": _loaded(type(owner).__module__),
            "joint": _loaded(self.dense.__func__.__module__),
            "c512_library": _loaded(delegated.__func__.__module__), "post": self.post,
        }
        self.source_manifest = _source_manifest(self.sources)
        self.calls = {"fp32_fractional": 0, "near_integer_reference": 0}
        self.last_route, self.last_diagnostics = None, None
        self.launch_hashes, self.resources, self._compiled = {}, {}, {}
        self.failed_calls = 0
        self.active = False
        self._require_owner(fresh=True)

    @property
    def debug(self):
        return self._debug

    def _require_owner(self, *, fresh=False, dispatch=False):
        self.session._ready()
        owner = self.owner
        if (get_ident() != self.thread or owner.session is not self.session or
                self.session._stack is not self.stack or self.stack.model is not self.model or
                self.stack.provider is not self.provider or self.stack.graph is not self.graph or
                owner.geometry is not self.geometry or not owner.controlled or
                owner.height != H or tuple(owner.source) != (H, W) or owner.variant != "unrounded" or
                not owner.c512_qkv_library_720 or not owner.native_k8_720 or
                owner.history_compact_720 or H not in owner.combo_modes or
                self.geometry.mode.model != (H, W) or self.geometry.mode.active != (H, W) or
                self.geometry.mode.internal != INTERNAL or self.geometry.mode.inset != (0, 0) or
                tuple(self.geometry.source) != (H, W) or
                tuple(self.model.PADDED_SIZES.get((H, W), ())) != INTERNAL or
                self.graph.closed or self.graph.model is not self.model or
                self.graph.arithmetic is not self.provider or self.provider.mode != "fp16_xmx"):
            raise RuntimeError("History candidate left its actual 720p unrounded graph owner")
        if (self.provider.__dict__.get("dense") is not self.dense or
                self.post._contiguous_cropped_head is not self.post_head or
                owner.c512_library_calls is not self.c512_calls or owner.native_k8_calls is not self.k8_calls or
                owner._c512_library_scope is not self.joint_scope or
                _closure(self.dense.__func__, "delegated_dense") is not self.c512_delegate or
                _closure(self.c512_delegate.__func__, "targets") is not self.c512_targets or
                len(self.c512_targets) != 16 or any(
                    self.c512_targets.get(id(weight), (None, None))[0] is not weight or
                    self.c512_targets[id(weight)][1] != label for _, weight, label in self.weights) or
                any(module.qkv is not weight for module, weight, _ in self.weights) or
                self.model.pre.front_weight is not self.pre_weight or
                self.model.reciprocal is not self.reciprocal or self.model.dimension_reciprocal is not self.dimensions or
                self.reciprocal.values is not self.values or self.dimensions.values is not self.dimension_values or
                (self.values._version, self.dimension_values._version) != self.versions or
                (self.values.data_ptr(), self.dimension_values.data_ptr()) != self.table_pointers or
                self.sampling.warp_history_normalized is not self.reference or
                any(getattr(self.sampling, name) is not function
                    for name, function in self.reference_callees.items()) or
                fused._prepare_axes is not self.axis_jit or _five_tap_native is not self.sample_jit or
                _half_texture_native is not self.texture_jit):
            raise RuntimeError("History/C512/K8 callable, weight or reciprocal ownership changed")
        if fresh and (self.graph.entries or self.graph.replays or owner._session_frames or
                      self.model._previous is not None or self.model._next_seed != 0):
            raise RuntimeError("History arm requires a fresh session before any frame or capture")
        if dispatch and (not self.active or _ACTIVE_SCOPE is not self or
                         self.live.warp_history_fused is not self or
                         self.temporal.warp_history_normalized is not self or
                         self.policy.ENABLED != self.policy.FAMILIES or
                         self.execution.current_arithmetic_backend() != "triton"):
            raise RuntimeError("Use this owned sampler through FullsizeGameModes' fused graph route")
        if dispatch:
            front = self.model.__dict__.get("_forward_front")
            if getattr(front, "__self__", None) is not self.graph:
                if (getattr(front, "__module__", None) != type(self.model).__module__ or
                        getattr(front, "__qualname__", None) !=
                        "GameLiveControlledNR.graph_controls.<locals>.controlled_front" or
                        getattr(_closure(front, "original"), "__self__", None) is not self.graph):
                    raise RuntimeError("History arm requires the owned installed body graph")

    def _dimension_reciprocals(self):
        iw, ih = self.dimensions(W), self.dimensions(H)
        _tensor("width reciprocal", iw, (), torch.float32, self.device)
        _tensor("height reciprocal", ih, (), torch.float32, self.device)
        return iw, ih

    def preflight(self, *, expected_source_sha256=None):
        """Explicit future Luna gate: real geometry/hash/no-spill, no dispatch.

        This method compiles and initializes GPU handles. It has NOT been run
        during CPU source development. Unknown/nonzero spill metadata fails.
        Both supported motion dtypes and this scope's fixed DEBUG are screened.
        """
        try:
            self._require_owner(fresh=True)
            if not self.active or self.live.warp_history_fused is not self:
                raise RuntimeError("Preflight must run inside the candidate's owned scope")
            if torch.xpu.is_current_stream_capturing():
                raise RuntimeError("Preflight must precede graph capture")
            # Add the exact screen implementation to the frozen source receipt.
            screen = import_module("spill_preflight_v1")
            modules = dict(self.sources, spill_preflight=screen)
            manifest = _source_manifest(modules)
            if any(manifest[role] != row for role, row in self.source_manifest.items()):
                raise RuntimeError("History source bytes changed after scope construction")
            if expected_source_sha256 is not None:
                if set(expected_source_sha256) != set(manifest) or any(
                        row["sha256"] != expected_source_sha256[role] for role, row in manifest.items()):
                    raise RuntimeError("History source hashes differ from the requested frozen manifest")
            if self._compiled:
                return self.receipt()
            # No new constants/buffers are registered on the model or its graph.
            with torch.inference_mode(False):
                scalar_hash = hashlib.sha256(self.values.detach().cpu().numpy().tobytes()).hexdigest()
                dimension_hash = hashlib.sha256(self.dimension_values.detach().cpu().numpy().tobytes()).hexdigest()
                reference_tables = self.sources["reciprocal"]
                if (scalar_hash != reference_tables.RECIPROCAL_SHA256 or
                        dimension_hash != reference_tables.NativeDimensionReciprocalTable.SHA256):
                    raise RuntimeError("Owned reference reciprocal table hash mismatch")
                iw, ih = self._dimension_reciprocals()
                image = torch.empty((H, W, 3), device=self.device, dtype=torch.float16)
                axes = torch.empty((10, H, W), device=self.device, dtype=torch.float32)
                axis_invalid = torch.empty((H, W), device=self.device, dtype=torch.int32)
                numerator, reciprocal, normalized, invalid, taps = _sample_buffers(image, self.debug)
                for dtype, label in ((torch.float16, "axes_fp16"), (torch.float32, "axes_fp32")):
                    motion = torch.empty((H, W, 2), device=self.device, dtype=dtype)
                    params = (motion, iw, ih, self.values, axes, axis_invalid, H, W, AXIS_BLOCK)
                    _, kernel, gate = screen.select(
                        self.axis_jit, [(AXIS_BLOCK,)], lambda _: params,
                        lambda _: (triton.cdiv(H * W, AXIS_BLOCK),),
                        num_warps=4, enable_fp_fusion=False)
                    self._compiled[label] = kernel
                    self.resources[label] = {"kernel_hash": kernel.hash, "resource_gate": gate,
                                             "motion": [H, W, 2], "dtype": str(dtype)}
                params = (image, axes, iw, ih, self.values, numerator, reciprocal,
                          normalized, invalid, taps, H, W, self.debug, BLOCK)
                _, kernel, gate = screen.select(
                    self.sample_jit, [(BLOCK,)], lambda _: params,
                    lambda _: (triton.cdiv(H * W * 3, BLOCK),),
                    num_warps=4, enable_fp_fusion=False)
                self._compiled["five_tap"] = kernel
                self.resources["five_tap"] = {"kernel_hash": kernel.hash, "resource_gate": gate,
                                             "image": [H, W, 3], "debug": self.debug}
            self.source_manifest = manifest
            self.table_hashes = {"reciprocal": scalar_hash, "dimension_reciprocal": dimension_hash}
            return self.receipt()
        except BaseException:
            self.session._failed = True
            raise

    def _require_kernel(self, label, kernel):
        expected = self._compiled[label]
        if kernel is not expected or kernel.hash != self.resources[label]["kernel_hash"]:
            raise RuntimeError(f"History launch missed its screened binary: {label}")
        self.launch_hashes[label] = kernel.hash

    def __call__(self, image, motion, *, return_components=False,
                 reciprocal_source, dimension_reciprocal):
        self.last_route, self.last_diagnostics = None, None
        try:
            self._require_owner(dispatch=True)
            if (not return_components or image is not self.model._previous or
                    reciprocal_source is not self.reciprocal or dimension_reciprocal is not self.dimensions):
                raise RuntimeError("History sampler requires this model's raw private history and tables")
            _tensor("image", image, (H, W, 3), torch.float16, self.device)
            if (tuple(motion.shape) != (H, W, 2) or motion.device != self.device or
                    motion.dtype not in (torch.float16, torch.float32)):
                raise ValueError("Expected matching 720p FP16/FP32 pixel motion")
            if set(self._compiled) != {"axes_fp16", "axes_fp32", "five_tap"}:
                raise RuntimeError("History candidate requires explicit preflight before frames")
            if torch.xpu.is_current_stream_capturing():
                raise RuntimeError("History must remain outside the frozen body graph")
            # Exactly the original whole-frame half-motion predicate/sync.
            rounded_motion = motion.half().float()
            if bool(((rounded_motion - rounded_motion.round()).abs() <= 1 / 256).all()):
                self.last_route = "near_integer_reference"
                result = self.sampling.warp_history_normalized(
                    image, motion, return_components=True,
                    reciprocal_source=reciprocal_source,
                    dimension_reciprocal=dimension_reciprocal)
            else:
                self.last_route = "fp32_fractional"
                iw, ih = self._dimension_reciprocals()
                # Same optional contiguity copy and allocations as prepare_axes.
                if not motion.is_contiguous():
                    motion = motion.contiguous()
                axes = torch.empty((10, H, W), device=motion.device, dtype=torch.float32)
                axis_invalid = torch.empty((H, W), device=motion.device, dtype=torch.int32)
                kernel = self.axis_jit[(triton.cdiv(H * W, AXIS_BLOCK),)](
                    motion, iw, ih, self.values, axes, axis_invalid, H, W, AXIS_BLOCK,
                    num_warps=4, enable_fp_fusion=False)
                label = "axes_fp16" if motion.dtype == torch.float16 else "axes_fp32"
                self._require_kernel(label, kernel)
                numerator, reciprocal, normalized, invalid, taps = _sample_buffers(image, self.debug)
                kernel = self.sample_jit[(triton.cdiv(H * W * 3, BLOCK),)](
                    image, axes, iw, ih, self.values, numerator, reciprocal, normalized,
                    invalid, taps, H, W, self.debug, BLOCK, num_warps=4, enable_fp_fusion=False)
                self._require_kernel("five_tap", kernel)
                if self.debug:
                    self.last_diagnostics = dict(axes=axes, axis_invalid=axis_invalid,
                                                 numerator=numerator, reciprocal=reciprocal,
                                                 normalized=normalized, invalid=invalid, taps=taps)
                result = numerator, reciprocal
            self.calls[self.last_route] += 1
            return result
        except BaseException:
            self.failed_calls += 1
            self.session._failed = True
            raise

    def receipt(self):
        """CPU metadata only; these outside-body calls are not GPU timings."""
        return dict(mode="fp32_fractional", all_history_cleaned=False,
                    active=self.active, model=[H, W], internal=list(INTERNAL),
                    debug=self.debug, outside_body_graph=True, launches_saved=0,
                    calls=dict(self.calls), last_route=self.last_route, failed_calls=self.failed_calls,
                    preflight_complete=set(self._compiled) == {"axes_fp16", "axes_fp32", "five_tap"},
                    resources=dict(self.resources), actual_launch_hashes=dict(self.launch_hashes),
                    sources=dict(self.source_manifest), table_hashes=dict(getattr(self, "table_hashes", {})),
                    c512_calls=dict(self.c512_calls), k8_calls=dict(self.k8_calls),
                    body_graph_replays=self.graph.replays,
                    near_integer_callee="nr_backend.sampling.warp_history_normalized/_sample_five_axes",
                    gpu_quality_performance="unverified")


@contextmanager
def installed(owner, *, history_value_720="reference", debug=False):
    """Default-off scope on the sampler imported/rebound for each game frame.

    Only reference and fp32_fractional are implemented. fp32_all_paths is a
    separate future numeric arm. Do not use a captured/previously run session,
    nest scopes, replace another hook, or reuse this session after scope exit.
    """
    global _ACTIVE_SCOPE
    if _ACTIVE_SCOPE is not None:
        raise RuntimeError("Another history value scope already owns the sampler")
    if history_value_720 == "reference":
        yield None
        return
    if history_value_720 != "fp32_fractional":
        raise ValueError("history_value_720 must be reference or fp32_fractional")
    live = import_module("nr_game_history_fused")
    try:
        candidate = HistoryValueNative720(owner, live, debug=debug)
    except BaseException:
        if getattr(owner, "session", None) is not None:
            owner.session._failed = True
        raise
    _ACTIVE_SCOPE = candidate
    live.warp_history_fused = candidate
    candidate.active = True
    failure = None
    try:
        yield candidate
    except BaseException as exc:
        failure = exc
        candidate.session._failed = True
        raise
    finally:
        problems = []
        candidate.active = False
        if live.warp_history_fused is candidate:
            live.warp_history_fused = candidate.original
        else:
            problems.append("game sampler ownership changed")
        if candidate.temporal.warp_history_normalized is candidate:
            candidate.temporal.warp_history_normalized = candidate.temporal_original
            problems.append("frame did not restore the temporal sampler")
        elif candidate.temporal.warp_history_normalized is not candidate.temporal_original:
            problems.append("temporal sampler ownership changed")
        if owner.session is not candidate.session and owner.session is not None:
            owner.session._failed = True
            problems.append("fullsize session changed")
        # Retire even eager-only temporal history; do not reset seeds or graphs.
        if not getattr(candidate.session, "_closed", False):
            candidate.session._failed = True
        if _ACTIVE_SCOPE is candidate:
            _ACTIVE_SCOPE = None
        else:
            problems.append("scope registry ownership changed")
        if problems:
            message = "History value scope exit: " + "; ".join(problems)
            if failure is None:
                raise RuntimeError(message)
            failure.add_note(message)
