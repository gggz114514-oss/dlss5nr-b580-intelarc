"""Default-off ViT norm/exp FMA ablations for the fresh 720p C512+K8 session.

CPU source handoff, 2026-09-30; no GPU/model import, compilation or runtime test was run
to establish this candidate. G/E fullsize_session, vit_block, attention and both
reference kernels were read and their SHA256s agreed (REFERENCE_SHA256 below).

Use after modes.select(720, (720, 1280), variant="unrounded"), before any body
capture, with the controlled C512 library + native K8 scopes already installed::

    with installed(modes, vit_norm_fma_720=True, vit_exp_fma_720=False) as receipt:
        receipt.preflight()  # Luna only: compile/load real shapes, no launch
        # modes.process(..., height=720, history_warp="fused", graph_replay=True,
        #               variant="unrounded", reset=...) for reset/history/replay

Each body still has 16 norms + 8 score exps + 8 exp(0)s. Only the two compensated
norm FMAs and the one ViT affine FMA become tl.fma(half, half, half).half(). Half
squares, XOR tree, NaN selection, rsqrt/clamp, query scale, special exp constants,
clamp/bit map, 16 padded keys and padding correction are retained. No launch is
removed. No layout, matrix schedule, ordinary softmax or exp(0) folding change.

The existing 240-token vforward closure and vit_attention bytecode are cloned
with session-owned aliases; no global helper or dormant forward_boundaries is
patched. Original FFN/c512/vit and arithmetic receipts remain logical work,
not evidence of physical FP8. New kernels write unrestricted half Y (CONTRACTS).
Warmup/eager and capture receipts are separate; Python counters do not advance
on graph replay. The immutable pair of bools enters the graph signature. Exit
restores hooks/contracts and closes captured graphs, requiring a new session.

Luna acceptance remains pending: four fresh arms (off/off, norm, exp, both),
real-shape zero-spill/hash gates, per-entry 16/8/8 capture coverage, flag-off byte
gate, finite/half/shape and reset/history private-state checks, then paired full
frame replay timing and complete user video review. Include zero/tiny/large Q/K,
half midpoint/cancellation/subnormal/NaN cases. Capture/preflight are not timing;
keep fused history, controls, seed, motion, depth, audio and source semantics
fixed. The user's approximately 70 ms baseline is not a candidate speed result.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import FunctionType, MethodType

import torch
import triton
import triton.language as tl

from nr_backend.execution import current_arithmetic_backend, record_arithmetic_dispatch
from nr_backend.triton_attention_normalize import _nan_left, rsqrt_half_clamped
from nr_backend.unround_policy import ENABLED, FAMILIES
from quantization_dataflow_v1 import CONTRACTS


TOKENS, HEADS, CHANNELS, PADDED_KEYS = 240, 32, 32, 256
NORM_SHAPE = (TOKENS, HEADS, CHANNELS)
SCORE_SHAPE = (HEADS, TOKENS, PADDED_KEYS)
SITES = ("query_norm", "key_norm", "score_exp", "zero_exp")
NORM_KERNEL = f"{__name__}._normalize"
EXP_KERNEL = f"{__name__}._exp_vit"
EXTRA = {NORM_KERNEL: (("Y",), ()), EXP_KERNEL: (("Y",), ())}
REFERENCE_SHA256 = {
    "fullsize_session_v1.py": "ff475dba5fe6f55073600906d81eddbe3df31e1d75764997fc6ad19d6ffb5cfb",
    "vit_block.py": "b7f8a992d04f3c5eceb4daf007b3153178f818ccb376fbb219aea0364469d4d5",
    "attention.py": "db67828fecec5c48e120a564163b72503b73a4fcd7efb674cadbaf671b1d8841",
    "triton_attention_normalize.py": "9a7a91f13d485392b8b3b4045600cd15633903532266c058cda7e50c1ab439cc",
    "triton_attention_exp.py": "5ec5eb86a3d26ffb48899307c3dac2b1f8ee1603ad99946f3945f4c3db18d7dd",
}


@triton.jit
def _normalize(X, Y, ROWS: tl.constexpr, BR: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    lanes = tl.arange(0, 8)
    offsets = rows[:, None] * 32 + lanes[None, :]
    valid = rows[:, None] < ROWS
    x0 = tl.load(X + offsets, valid, other=0)
    x8 = tl.load(X + offsets + 8, valid, other=0)
    x16 = tl.load(X + offsets + 16, valid, other=0)
    x24 = tl.load(X + offsets + 24, valid, other=0)
    a = (x16.to(tl.float32) * x16.to(tl.float32)).to(tl.float16)
    a = _nan_left(tl.fma(x0.to(tl.float16), x0.to(tl.float16), a.to(tl.float16)).to(tl.float16), x0, a)
    b = (x24.to(tl.float32) * x24.to(tl.float32)).to(tl.float16)
    b = _nan_left(tl.fma(x8.to(tl.float16), x8.to(tl.float16), b.to(tl.float16)).to(tl.float16), x8, b)
    total = _nan_left((a.to(tl.float32) + b.to(tl.float32)).to(tl.float16), a, b)
    for mask in tl.static_range(3):
        xor_mask = 4 >> mask
        other = tl.gather(total, tl.broadcast_to((lanes ^ xor_mask)[None, :], (BR, 8)), 1)
        total = _nan_left((total.to(tl.float32) + other.to(tl.float32)).to(tl.float16), total, other)
    denominator = tl.gather(total, tl.full((BR, 1), 0, tl.int32), 1)
    scale = rsqrt_half_clamped(denominator)
    tl.store(Y + offsets, _nan_left((x0.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x0, scale), valid)
    tl.store(Y + offsets + 8, _nan_left((x8.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x8, scale), valid)
    tl.store(Y + offsets + 16, _nan_left((x16.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x16, scale), valid)
    tl.store(Y + offsets + 24, _nan_left((x24.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x24, scale), valid)


@triton.jit
def _exp_vit(X, Y, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, other=0)
    affine = tl.fma(x.to(tl.float16), tl.full((), 0.08953857421875, tl.float16),
                    tl.full((), 1.708984375, tl.float16)).to(tl.float16)
    affine = tl.minimum(tl.maximum(affine.to(tl.float32), 1.439453125), 1.9775390625).to(tl.float16)
    bits = affine.to(tl.uint16, bitcast=True).to(tl.int32)
    input_bits = x.to(tl.uint16, bitcast=True).to(tl.int32)
    bits = tl.where((input_bits & 0x7fff) > 0x7c00, input_bits | 0x200, bits)
    result = ((bits << 4) + 0x4000) & 0xffff
    tl.store(Y + i, result.to(tl.uint16).to(tl.float16, bitcast=True), i < N)


def _closure(function):
    return dict(zip(function.__code__.co_freevars,
                    (cell.cell_contents for cell in function.__closure__ or ())))


def _cell(value):
    return (lambda: value).__closure__[0]


def _half_xpu(value, shape, device, *, contiguous=False, dtype=torch.float16):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or
            value.dtype != dtype or value.device.type != "xpu" or value.device != device or
            (contiguous and not value.is_contiguous())):
        raise RuntimeError(f"ViT FMA boundary differs from {shape}/{dtype}")


@dataclass(frozen=True)
class Mode:
    vit_norm_fma_720: bool = False
    vit_exp_fma_720: bool = False

    @property
    def identity(self):
        return (__name__, self.vit_norm_fma_720, self.vit_exp_fma_720)


class NativeFma720Counter:
    """Per-owner successful dispatch receipts, with capture-only coverage gates."""

    def __init__(self, session, modes, blocks, mode, sources, reference_vforward):
        self.session, self.modes, self.blocks = session, modes, blocks
        self.stack = session._stack
        self._mode = mode
        self.sources = sources
        self.reference_vforward = reference_vforward
        path = Path(__file__).resolve()
        self.candidate_source = {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
        self.calls = {f"vit.{i}.{site}": 0 for i in range(8) for site in SITES}
        self.capture_calls = dict.fromkeys(self.calls, 0)
        self.eager_calls = dict.fromkeys(self.calls, 0)
        self.owner_calls = dict.fromkeys(range(8), 0)
        self.capture_owner_calls = dict.fromkeys(range(8), 0)
        self.resources, self.kernels_by_site, self.capture_gates = {}, {}, []
        self.capture_kernels_by_site = {}
        self._compiled = {}
        self._active = None
        self._live = True
        self.weights = tuple(tuple(getattr(block, name) for name in
                                  ("qkv_weight", "query_scale", "projection", "attn_skip"))
                             for block in blocks)
        self.weight_versions = tuple(tuple(t._version for t in weights) for weights in self.weights)

    @property
    def mode(self):
        return self._mode

    def _guard(self):
        self.session._ready()
        if (not self._live or self.session._stack is not self.stack or
                self.modes.session is not self.session or self.stack.provider.mode != "fp16_xmx" or
                len(self.stack.model.vit) != 8 or
                any(a is not b for a, b in zip(self.stack.model.vit, self.blocks)) or
                self.stack.int8_vit.modules != {id(block): i for i, block in enumerate(self.blocks)}):
            raise RuntimeError("ViT FMA scope lost its live unrounded session/owners")
        for i, block in enumerate(self.blocks):
            now = tuple(getattr(block, name) for name in
                        ("qkv_weight", "query_scale", "projection", "attn_skip"))
            if (any(a is not b for a, b in zip(now, self.weights[i])) or
                    tuple(t._version for t in now) != self.weight_versions[i]):
                raise RuntimeError(f"ViT FMA constants changed at owner {i}")

    def preflight(self):
        """Luna entry: compile/load 7680 C32 rows, 32x240x256 scores and scalar 0.

        Uses the reference launch defaults (norm BR16/warps4; exp B512).
        select() requires integer zero-spill metadata. This never launches a
        kernel or imports/forwards a model; do not call in the CPU source task.
        """
        try:
            from spill_preflight_v1 import select

            self._guard()
            if self.stack.graph.entries or self._active is not None or torch.xpu.is_current_stream_capturing():
                raise RuntimeError("ViT FMA preflight must precede warmup/capture")
            specs = []
            if self.mode.vit_norm_fma_720:
                specs.append(("norm", _normalize, NORM_SHAPE, 7680, 16, {"num_warps": 4}))
            if self.mode.vit_exp_fma_720:
                specs.extend((("score_exp", _exp_vit, SCORE_SHAPE, 1966080, 512, {}),
                              ("zero_exp", _exp_vit, (), 1, 512, {})))
            for name, jit, shape, count, tile, options in specs:
                if name in self._compiled:
                    continue
                source = torch.empty(shape, dtype=torch.float16, device=self.weights[0][0].device)
                output = torch.empty_like(source)
                params = (source, output, count, tile)
                _, kernel, gate = select(jit, [(tile,)], lambda _: params,
                                         lambda _: (triton.cdiv(count, tile),),
                                         enable_fp_fusion=False, **options)
                if type(kernel.n_spills) is not int or kernel.n_spills != 0:
                    raise RuntimeError(f"ViT FMA zero-spill preflight failed: {name}")
                self._compiled[name] = kernel
                self.resources[name] = {"kernel_hash": kernel.hash, "shape": list(shape),
                                        "source_kernel": NORM_KERNEL if name == "norm" else EXP_KERNEL,
                                        "count": count, "tile": tile, "launch_options": options,
                                        "enable_fp_fusion": False, "resource_gate": gate,
                                        "registers": kernel.n_regs,
                                        "shared_bytes": kernel.metadata.shared}
            for label in self.calls:
                site = label.rsplit(".", 1)[1]
                name = "norm" if site.endswith("norm") else site
                if name in self._compiled:
                    self.kernels_by_site[label] = self._compiled[name].hash
            return self.resources
        except BaseException:
            self.session._failed = True
            raise

    def _dispatch(self, value, site):
        if self._active is None or current_arithmetic_backend() != "triton" or ENABLED != FAMILIES:
            raise RuntimeError("ViT FMA dispatch outside its owned Triton body")
        index, progress, capturing = self._active
        if site != SITES[len(progress)]:
            raise RuntimeError("ViT FMA norm/score/exp(0) order or count changed")
        device = self.weights[index][0].device
        shape = NORM_SHAPE if site.endswith("norm") else SCORE_SHAPE if site == "score_exp" else ()
        _half_xpu(value, shape, device, dtype=torch.float32 if site == "zero_exp" else torch.float16)
        if site.endswith("norm") and tuple(value.stride()) != (3072, 96, 1):
            raise RuntimeError("ViT FMA must retain the original strided Q/K boundary")
        if bool(torch.xpu.is_current_stream_capturing()) != capturing:
            raise RuntimeError("ViT FMA capture state changed within a block")
        native = self.mode.vit_norm_fma_720 if site.endswith("norm") else self.mode.vit_exp_fma_720
        if native:
            name = "norm" if site.endswith("norm") else site
            if name not in self._compiled:
                raise RuntimeError(f"ViT FMA needs real-shape preflight before dispatch: {name}")
            # Exactly the conversions/allocation made by the reference wrappers.
            source = value.half().contiguous()
            output = torch.empty_like(source)
            if site.endswith("norm"):
                kernel = _normalize[(480,)](source, output, 7680, 16,
                                           num_warps=4, enable_fp_fusion=False)
                kind = "attention_normalize_c32"
            else:
                count = source.numel()
                kernel = _exp_vit[(triton.cdiv(count, 512),)](source, output, count, 512,
                                                           enable_fp_fusion=False)
                kind = "attention_exp_vit"
            if kernel is not self._compiled[name]:
                raise RuntimeError(f"ViT FMA launch missed the screened binary: {name}")
            record_arithmetic_dispatch(kind)
        elif site.endswith("norm"):
            output = self._reference_norm(value)
        else:
            output = self._reference_exp(value)
        _half_xpu(output, shape, device, contiguous=True)
        label = f"vit.{index}.{site}"
        self.calls[label] += 1
        bucket = self.capture_calls if capturing else self.eager_calls
        bucket[label] += 1
        if native and capturing:
            self.capture_kernels_by_site[label] = kernel.hash
        progress.append(site)
        return output

    def _norm(self, value):
        if self._active is None or len(self._active[1]) not in (0, 1):
            raise RuntimeError("Unexpected owned ViT normalization")
        return self._dispatch(value, SITES[len(self._active[1])])

    def _exp(self, value):
        if self._active is None or len(self._active[1]) not in (2, 3):
            raise RuntimeError("Unexpected owned ViT special exponential")
        return self._dispatch(value, SITES[len(self._active[1])])

    def run(self, index, module, value, forward):
        self._guard()
        if module is not self.blocks[index] or self._active is not None:
            raise RuntimeError("ViT FMA owner mismatch or recursive body")
        _half_xpu(value, (TOKENS, 1024), self.weights[index][0].device, contiguous=True)
        capturing = bool(torch.xpu.is_current_stream_capturing())
        progress = []
        self._active = (index, progress, capturing)
        try:
            output = forward(module, value)
            if tuple(progress) != SITES:
                raise RuntimeError(f"ViT FMA missed real Q/K/score/zero entries at owner {index}")
            _half_xpu(output, (TOKENS, 1024), value.device, contiguous=True)
            self.owner_calls[index] += 1
            if capturing:
                self.capture_owner_calls[index] += 1
            return output
        finally:
            self._active = None

    def report(self):
        return {"mode": self.mode.identity, "sources": self.sources,
                "candidate_source": self.candidate_source,
                "calls": dict(self.calls), "capture_calls": dict(self.capture_calls),
                "owner_calls": dict(self.owner_calls),
                "eager_or_warmup_calls": dict(self.eager_calls),
                "capture_owner_calls": dict(self.capture_owner_calls),
                "expected_per_body": {"norm": 16, "score_exp": 8, "zero_exp": 8},
                "resources": dict(self.resources), "kernels_by_site": dict(self.kernels_by_site),
                "capture_kernels_by_site": dict(self.capture_kernels_by_site),
                "capture_gates": list(self.capture_gates),
                "graph_replays": self.stack.graph.replays,
                "replay_evidence": "graph.replays/entry.replays; Python counters skip replay",
                "launches_removed": 0, "validation": "pending Luna GPU and user visual review"}


def _reference_vforward(session):
    """Find the authenticated active closure without entering a model scope."""
    current, seen = session._installed, set()
    for _ in range(12):
        if not isinstance(current, FunctionType) or id(current) in seen:
            raise RuntimeError("ViT FMA install chain has a cycle or unknown callable")
        seen.add(id(current))
        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is not None:
            current = wrapped
            continue
        cells = _closure(current)
        forward = cells.get("vforward")
        if forward is not None:
            if not isinstance(forward, FunctionType):
                raise RuntimeError("ViT FMA install chain has an invalid vforward")
            owner = _closure(forward)
            if (forward.__code__.co_name != "vforward" or
                    owner.get("stack") is not session._stack or
                    owner.get("counts") is not session._counts or
                    owner.get("vit") is not session._vit or
                    current.__code__.co_name != "installed" or
                    Path(current.__code__.co_filename).resolve() !=
                    Path(forward.__code__.co_filename).resolve()):
                raise RuntimeError("ViT FMA vforward does not belong to the selected session")
            return forward
        if (current.__module__ not in ("c512_window_projection_game_v1",) or
                current.__qualname__ != "installed.<locals>.wrapper" or
                "original_installed" not in cells):
            raise RuntimeError("ViT FMA encountered an unreviewed install wrapper")
        current = cells["original_installed"]
    raise RuntimeError("ViT FMA install chain exceeded its reviewed depth")


def _sources(session):
    import nr_backend.attention as attention
    import nr_backend.triton_attention_exp as exp
    import nr_backend.triton_attention_normalize as norm

    vit = session._vit
    reference_vforward = _reference_vforward(session)
    paths = [Path(reference_vforward.__code__.co_filename).resolve(strict=True)]
    for module in (vit, attention, norm, exp):
        path = Path(module.__file__).resolve()
        if path.parent.parts[-3:] != ("experimental", "fp8_unround_overlay", "nr_backend"):
            raise RuntimeError("ViT FMA requires the isolated unround overlay, never the exact backend")
        paths.append(path)
    result = {}
    for path in paths:
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != REFERENCE_SHA256[path.name]:
            raise RuntimeError(f"ViT FMA reviewed source changed: {path}")
        result[path.name] = {"path": str(path), "sha256": digest}
    if (vit.normalize_c32 is not attention.normalize_c32 or
            vit.vit_attention.__globals__["vit_exponential"] is not vit.vit_exponential):
        raise RuntimeError("ViT FMA reference aliases were already replaced")
    return result, reference_vforward


def _owned_baseline(modes, session, stack):
    """Authenticate the installed joint scope's actual closures, not just flags."""
    import c512_k8_joint_scope_720_v1 as joint

    dense = stack.provider.__dict__.get("dense")
    if (not isinstance(dense, MethodType) or dense.__self__ is not stack.provider or
            dense.__func__.__module__ != joint.__name__ or
            dense.__func__.__qualname__ != "installed.<locals>.dense"):
        raise RuntimeError("ViT FMA requires the installed joint C512+K8 provider")
    owned = _closure(dense.__func__)
    delegated = owned.get("delegated_dense")
    if (not isinstance(delegated, MethodType) or delegated.__self__ is not stack.provider or
            delegated.__func__.__module__ != joint.c512.__name__ or
            delegated.__func__.__qualname__ != "installed.<locals>.dense"):
        raise RuntimeError("ViT FMA lost the delegated C512 library owner")
    targets = _closure(delegated.__func__).get("targets", {})
    expected = {id(block.attention.qkv): (block.attention.qkv, f"{side}_{i}")
                for side in ("encoder512", "decoder512")
                for i, block in enumerate(getattr(stack.model, side))}
    if (len(expected) != 16 or set(targets) != set(expected) or
            any(targets[key][0] is not weight or targets[key][1] != name
                for key, (weight, name) in expected.items()) or
            owned.get("model") is not stack.model or owned.get("k8_calls") is not modes.native_k8_calls or
            _closure(delegated.__func__).get("calls") is not modes.c512_library_calls):
        raise RuntimeError("ViT FMA C512+K8 ownership/receipts changed")
    post = joint.active_post._contiguous_cropped_head
    if (not isinstance(post, FunctionType) or post.__module__ != joint.__name__ or
            post.__qualname__ != "installed.<locals>.post_head" or
            _closure(post).get("k8_calls") is not modes.native_k8_calls):
        raise RuntimeError("ViT FMA lost the actual post K8 entry")
    return dense, post


class _VitAliases:
    def __init__(self, original, norm, attention):
        self._original = original
        self.normalize_c32 = norm
        self.vit_attention = attention

    def __getattr__(self, name):
        return getattr(self._original, name)


def _owned_forward(session, counter):
    vit = session._vit
    existing = vit.VitBlock.forward  # active fullsize vforward, not boundaries
    if existing is not counter.reference_vforward:
        raise RuntimeError("Active ViT forward differs from its authenticated owner")
    cells = _closure(existing)
    expected_path = Path(counter.sources["fullsize_session_v1.py"]["path"])
    if (existing.__code__.co_name != "vforward" or
            Path(existing.__code__.co_filename).resolve() != expected_path or
            cells.get("stack") is not counter.stack or cells.get("counts") is not session._counts or
            cells.get("vit") is not vit):
        raise RuntimeError("ViT FMA requires the reviewed active fullsize vforward")
    counter._reference_norm, counter._reference_exp = vit.normalize_c32, vit.vit_exponential
    attention = vit.vit_attention
    globals_owned = dict(attention.__globals__, vit_exponential=counter._exp)
    owned_attention = FunctionType(attention.__code__, globals_owned, attention.__name__,
                                   attention.__defaults__, attention.__closure__)
    owned_attention.__kwdefaults__ = attention.__kwdefaults__
    aliases = _VitAliases(vit, counter._norm, owned_attention)
    closure = tuple(_cell(aliases) if name == "vit" else cell
                    for name, cell in zip(existing.__code__.co_freevars, existing.__closure__))
    owned = FunctionType(existing.__code__, existing.__globals__, existing.__name__,
                         existing.__defaults__, closure)
    owned.__kwdefaults__ = existing.__kwdefaults__
    return owned


@contextmanager
def installed(modes, *, vit_norm_fma_720=False, vit_exp_fma_720=False):
    """Own the selected modes.session until reset/history capture and replay end.

    Both flags are independent exact bools, default False. No controller or
    fullsize source integration is implied; invoke this scope explicitly.
    """
    session = getattr(modes, "session", None)
    if session is None:
        raise RuntimeError("Select a fresh controlled 720p C512+K8 session first")
    try:
        if type(vit_norm_fma_720) is not bool or type(vit_exp_fma_720) is not bool:
            raise ValueError("ViT norm/exp FMA flags must be independent fixed bools")
        session._ready()
        if (modes.height != 720 or tuple(modes.source) != (720, 1280) or
                modes.variant != "unrounded" or not modes.controlled or
                not modes.c512_qkv_library_720 or not modes.native_k8_720 or
                getattr(modes, "vit_head_720", False) or
                tuple(session.fullsize_geometry) != (720, 1280) or
                tuple(session.fullsize_padding["padding"]) != (768, 1280)):
            raise RuntimeError("ViT FMA requires actual 720p unrounded C512+K8 without ViT layout")
        stack, graph, vit = session._stack, session._stack.graph, session._vit
        if (stack.provider.mode != "fp16_xmx" or graph.entries or graph.closed or
                "_signature" in graph.__dict__ or "_select" in modes.__dict__ or
                "_process" in modes.__dict__ or "_vit_native_fma_720" in session.__dict__):
            raise RuntimeError("ViT FMA requires a fresh graph and unmodified scoped entry hooks")
        blocks = tuple(stack.model.vit)
        if (len(blocks) != 8 or len({id(b) for b in blocks}) != 8 or
                stack.int8_vit.modules != {id(block): i for i, block in enumerate(blocks)} or
                any(type(block) is not vit.VitBlock or "forward" in block.__dict__ for block in blocks)):
            raise RuntimeError("ViT FMA requires eight distinct owned active ViT/FFN blocks")
        for block in blocks:
            if block.qkv_weight.device != blocks[0].qkv_weight.device:
                raise RuntimeError("ViT FMA owners must share the one session device")
            for name, shape in (("qkv_weight", (1024, 3072)), ("query_scale", (32,)),
                                ("projection", (1024, 1024)), ("attn_skip", (1024,))):
                _half_xpu(getattr(block, name), shape, block.qkv_weight.device, contiguous=True)
        sources, reference_vforward = _sources(session)
        baseline_dense, baseline_post = _owned_baseline(modes, session, stack)
        mode = Mode(vit_norm_fma_720, vit_exp_fma_720)
        counter = NativeFma720Counter(session, modes, blocks, mode, sources, reference_vforward)
        # Select contracts independently; exp-only must not add a norm contract.
        extra = {}
        if mode.vit_norm_fma_720:
            extra[NORM_KERNEL] = EXTRA[NORM_KERNEL]
        if mode.vit_exp_fma_720:
            extra[EXP_KERNEL] = EXTRA[EXP_KERNEL]
        if any(name in CONTRACTS for name in extra):
            raise RuntimeError("ViT FMA candidate contracts already installed")
    except BaseException:
        session._failed = True
        raise

    original_scope, original_select, original_process = session._installed, modes._select, modes._process
    original_signature, saved_signature = graph._signature, graph.signature

    def guard():
        counter._guard()
        if (counter.mode is not mode or session.__dict__.get("_vit_native_fma_720") is not counter or
                modes.height != 720 or modes.variant != "unrounded" or
                tuple(modes.source) != (720, 1280) or
                not modes.controlled or not modes.c512_qkv_library_720 or not modes.native_k8_720 or
                getattr(modes, "vit_head_720", False) or
                stack.provider.__dict__.get("dense") is not baseline_dense or
                session._installed is not wrapped_scope or
                modes.__dict__.get("_select") is not select_hook or
                modes.__dict__.get("_process") is not process_hook or
                graph.__dict__.get("_signature") is not scope_signature or
                not all(CONTRACTS.get(name) == contract for name, contract in extra.items())):
            raise RuntimeError("ViT FMA session configuration changed")
        import c512_k8_joint_scope_720_v1 as joint
        if joint.active_post._contiguous_cropped_head is not baseline_post:
            raise RuntimeError("ViT FMA post K8 owner changed")

    def signature(self):
        return original_signature() + (mode.identity,)

    def guarded_select(self, height, source, *, variant):
        try:
            if (self is not modes or self.session is not session or height != 720 or
                    tuple(source) != (720, 1280) or variant != "unrounded"):
                raise RuntimeError("ViT FMA scope cannot switch session/geometry/variant")
            return original_select(height, source, variant=variant)
        except BaseException:
            session._failed = True
            raise

    def guarded_process(self, color, motion, *, height, reset, history_warp,
                        graph_replay, controls, variant):
        try:
            guard()
            if (self is not modes or height != 720 or tuple(color.shape[:2]) != (720, 1280) or
                    variant != "unrounded" or history_warp != "fused" or graph_replay is not True):
                raise RuntimeError("ViT FMA measurements require actual 720p fused-history graph replay")
            before_replays = graph.replays
            output = original_process(color, motion, height=height, reset=reset,
                                      history_warp=history_warp, graph_replay=graph_replay,
                                      controls=controls, variant=variant)
            if graph.replays != before_replays + 1:
                raise RuntimeError("ViT FMA request did not execute exactly one graph replay")
            return output
        except BaseException:
            session._failed = True
            raise

    @contextmanager
    def wrapped_scope():
        replacements = {}
        try:
            guard()
            before = dict(counter.capture_calls)
            owners_before = dict(counter.capture_owner_calls)
            entries_before = set(graph.entries)
            with original_scope():
                owned = _owned_forward(session, counter)
                for index, block in enumerate(blocks):
                    if "forward" in block.__dict__:
                        raise RuntimeError("ViT FMA instance forward already overridden")

                    def forward(module, value, *, _index=index):
                        try:
                            return counter.run(_index, module, value, owned)
                        except BaseException:
                            session._failed = True
                            raise

                    replacement = MethodType(forward, block)
                    replacements[index] = replacement
                    block.forward = replacement
                try:
                    yield
                    added = set(graph.entries) - entries_before
                    if added:
                        delta = {name: value - before[name] for name, value in counter.capture_calls.items()}
                        owner_delta = {i: value - owners_before[i] for i, value in counter.capture_owner_calls.items()}
                        passed = (all(value == len(added) for value in delta.values()) and
                                  all(value == len(added) for value in owner_delta.values()))
                        counter.capture_gates.append({"new_entries": len(added), "by_site": delta,
                                                      "by_owner": owner_delta, "passed": passed})
                        if not passed:
                            raise RuntimeError("New ViT FMA graph missed its eight owners/16 norm/8+8 exp")
                finally:
                    valid = all(blocks[i].__dict__.get("forward") is value
                                for i, value in replacements.items())
                    for i in replacements:
                        blocks[i].__dict__.pop("forward", None)
                    if not valid:
                        raise RuntimeError("ViT FMA owned forwards changed during execution")
        except BaseException:
            session._failed = True
            raise
        finally:
            # Also restore partial installations if setup failed before yield.
            for i in replacements:
                blocks[i].__dict__.pop("forward", None)

    scope_signature = MethodType(signature, graph)
    select_hook, process_hook = MethodType(guarded_select, modes), MethodType(guarded_process, modes)
    try:
        session._vit_native_fma_720 = counter
        graph._signature = scope_signature
        graph.signature = saved_signature + (mode.identity,)
        modes._select, modes._process, session._installed = select_hook, process_hook, wrapped_scope
        CONTRACTS.update(extra)
        yield counter
    except BaseException:
        session._failed = True
        raise
    finally:
        valid = (session._installed is wrapped_scope and modes.__dict__.get("_select") is select_hook and
                 modes.__dict__.get("_process") is process_hook and
                 graph.__dict__.get("_signature") is scope_signature and
                 session.__dict__.get("_vit_native_fma_720") is counter and
                 all(CONTRACTS.get(name) == contract for name, contract in extra.items()))
        counter._live = False
        session._installed = original_scope
        modes.__dict__.pop("_select", None)
        modes.__dict__.pop("_process", None)
        session.__dict__.pop("_vit_native_fma_720", None)
        try:
            # Captured commands outlive Python aliases: actually reset/close them.
            if graph.entries:
                session._failed = True
                graph.close()
        except BaseException:
            session._failed = True
            raise
        finally:
            graph.__dict__.pop("_signature", None)
            graph.signature = saved_signature
            for name in extra:
                CONTRACTS.pop(name, None)
        if not valid:
            session._failed = True
            raise RuntimeError("ViT FMA session/graph hooks or contracts changed during its scope")


def installed_720(session, *, modes, vit_norm_fma_720=False, vit_exp_fma_720=False):
    """Explicit session form; modes supplies real C512+K8 and source-route guards."""
    if modes.session is not session:
        session._failed = True
        raise RuntimeError("ViT FMA modes does not own this session")
    return installed(modes, vit_norm_fma_720=vit_norm_fma_720,
                     vit_exp_fma_720=vit_exp_fma_720)
