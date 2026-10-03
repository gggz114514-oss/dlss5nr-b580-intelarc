"""Opt-in 720p C128/C256 branched-MLP expansion experiment.

Only the expansion schedule changes: each X tile feeds two independent part
accumulators. Each accumulator sees the original increasing K-block sequence;
activation and reduction still consume parts 0, 1, 2, 3 in that order. The
input rounding, LUT/cubic policy, half boundaries and projection use the active
batched implementation. The new pairwise kernel must compile without spill;
the unchanged baseline projection's existing spill is recorded separately.

This is a candidate, not a byte-equivalence or speed claim. Compare captured
frame outputs with the same standard/unrounded FusedBatched baseline.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch
import triton
import triton.language as tl

import batched_branched_mlp_v1 as batched
from fused_c32_mlp_lut_v1 import lookup
from native_half_cubic_v1 import cubic
from nr_backend.execution import current_arithmetic_backend, record_arithmetic_dispatch
from nr_backend.triton_fp8 import _round_fp8_half
from nr_backend.unround_policy import ENABLED, round_multihead


_BASE = {128: (96, 160), 256: (48, 80)}
_COMPILED = {}


@triton.jit
def _pairs_pairwise(X, EXPAND, REDUCE, LUT, LATENT,
                    M: tl.constexpr, C: tl.constexpr, BM: tl.constexpr,
                    ROUND_REDUCED: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    branch = tl.program_id(1)
    lane = tl.arange(0, 32)
    reduced = tl.full((BM, 32), 0., tl.float32)
    for pair in range(2):
        expanded0 = tl.full((BM, 32), 0., tl.float32)
        expanded1 = tl.full((BM, 32), 0., tl.float32)
        for block in range(C // 32):
            k = block * 32 + lane
            x = tl.load(X + row[:, None] * C + k[None, :],
                        row[:, None] < M, other=0)
            w0 = tl.load(EXPAND + (branch * C + k[:, None]) * 128 +
                         (pair * 2) * 32 + lane[None, :])
            w1 = tl.load(EXPAND + (branch * C + k[:, None]) * 128 +
                         (pair * 2 + 1) * 32 + lane[None, :])
            expanded0 = tl.dot(x, w0, expanded0, out_dtype=tl.float32)
            expanded1 = tl.dot(x, w1, expanded1, out_dtype=tl.float32)

        if ROUND_REDUCED:
            hidden0 = lookup(expanded0.to(tl.float16), LUT)
        else:
            hidden0 = cubic(expanded0.to(tl.float16), ROUND_OUTPUT=False)
        r0 = tl.load(REDUCE + (branch * 128 + (pair * 2) * 32 +
                               lane[:, None]) * 32 + lane[None, :])
        reduced = tl.dot(hidden0, r0, reduced, out_dtype=tl.float32)

        if ROUND_REDUCED:
            hidden1 = lookup(expanded1.to(tl.float16), LUT)
        else:
            hidden1 = cubic(expanded1.to(tl.float16), ROUND_OUTPUT=False)
        r1 = tl.load(REDUCE + (branch * 128 + (pair * 2 + 1) * 32 +
                               lane[:, None]) * 32 + lane[None, :])
        reduced = tl.dot(hidden1, r1, reduced, out_dtype=tl.float32)

    value = reduced.to(tl.float16)
    if ROUND_REDUCED:
        value = _round_fp8_half(value)
    tl.store(LATENT + (branch * M + row[:, None]) * 32 + lane[None, :],
             value, row[:, None] < M)


def _validate(features, module, fused):
    c = module.channels
    if c not in _BASE or not isinstance(features, torch.Tensor):
        raise ValueError("Expected a C128/C256 720p branched MLP input")
    if (features.ndim != 3 or features.dtype != torch.float16 or
            features.device.type != "xpu" or not features.is_contiguous()):
        raise ValueError("Expected contiguous HWC FP16 XPU features")
    h, w = _BASE[c]
    if tuple(features.shape) not in {
            (h + dh, w + dw, c) for dh in (0, 8) for dw in (0, 8)}:
        raise ValueError("Unexpected 720p branched MLP shape")
    if id(module) not in fused.modules or fused.provider.mode != "fp16_xmx":
        raise ValueError("Selected MLP is not owned by the active FP16 fused component")
    config = fused.configurations[c]
    if (set(config) != {"pair_bm", "pair_stages", "project_bm", "project_bn"} or
            config["pair_bm"] not in (16, 32) or config["pair_stages"] not in (1, 2) or
            config["project_bm"] not in (16, 32) or config["project_bn"] not in (32, 64)):
        raise ValueError("Unsupported active FusedBatched launch configuration")
    branches = c // 32
    for value, shape in ((module.expand, (branches, c, 128)),
                         (module.reduce, (branches, 128, 32)),
                         (module.project, (branches, 32, c)),
                         (module.skip_scale, (c,))):
        if (tuple(value.shape) != shape or value.dtype != torch.float16 or
                value.device != features.device or not value.is_contiguous()):
            raise ValueError("Invalid branched MLP weight buffer")
    lut = fused.constant.require()
    if (tuple(lut.shape) != (65536,) or lut.dtype != torch.int16 or
            lut.device != features.device or not lut.is_contiguous()):
        raise ValueError("Invalid active cubic LUT")
    return config, lut


def _key(features, config, round_reduced):
    return (features.device, tuple(features.shape), config["pair_bm"],
            config["pair_stages"], config["project_bm"],
            config["project_bn"], round_reduced)


def _resources(kernel):
    kernel._init_handles()
    spills = getattr(kernel, "n_spills", None)
    if not isinstance(spills, int) or spills < 0:
        raise RuntimeError("Compiled spill metadata is unavailable")
    return {"hash": kernel.hash, "spills": spills,
            "registers": kernel.n_regs, "shared_bytes": kernel.metadata.shared,
            "build_flags": str(getattr(kernel.metadata, "build_flags", None))}


def preflight(features, module, fused):
    """Compile both launches without executing them; reject pairwise spill.

    Call once with the selected block's actual 720p input before graph capture,
    for each standard/unrounded policy and shape that will be captured.
    """
    config, lut = _validate(features, module, fused)
    c = module.channels
    m = features.numel() // c
    branches = c // 32
    round_reduced = f"c{c}" not in ENABLED
    latent = torch.empty((branches, m, 32), dtype=features.dtype, device=features.device)
    out = torch.empty_like(features)
    pair = _pairs_pairwise.warmup(
        features, module.expand, module.reduce, lut, latent,
        m, c, config["pair_bm"], round_reduced,
        grid=(triton.cdiv(m, config["pair_bm"]), branches),
        num_warps=4, num_stages=config["pair_stages"], enable_fp_fusion=False)
    project = batched._project.warmup(
        features, latent, module.project, module.skip_scale, out,
        m, c, config["project_bm"], config["project_bn"],
        grid=(triton.cdiv(m, config["project_bm"]),
              triton.cdiv(c, config["project_bn"])),
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    resources = {"pairwise": _resources(pair), "projection": _resources(project)}
    # Projection is the same baseline _project kernel with the same launch
    # configuration. Its existing resource use is not a new pairwise regression.
    if resources["pairwise"]["spills"] != 0:
        raise RuntimeError(f"Pairwise MLP compile spilled: {resources}")
    _COMPILED[_key(features, config, round_reduced)] = pair, project
    return resources


def forward(features, module, fused):
    """Return the same unquantized block output and two handles as FusedBatched."""
    config, lut = _validate(features, module, fused)
    c = module.channels
    m = features.numel() // c
    branches = c // 32
    round_reduced = f"c{c}" not in ENABLED
    compiled = _COMPILED.get(_key(features, config, round_reduced))
    if compiled is None:
        raise RuntimeError("Preflight this shape and rounding policy before capture")
    x = round_multihead(c, features)
    latent = torch.empty((branches, m, 32), dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    pair, project = compiled
    pair[(triton.cdiv(m, config["pair_bm"]), branches, 1)](
        x, module.expand, module.reduce, lut, latent,
        m, c, config["pair_bm"], round_reduced)
    project[(triton.cdiv(m, config["project_bm"]),
             triton.cdiv(c, config["project_bn"]), 1)](
        x, latent, module.project, module.skip_scale, out,
        m, c, config["project_bm"], config["project_bn"])
    for _ in range(branches):
        for kind in ("dense", "dense", "cubic_fp8", "fp8", "dense"):
            record_arithmetic_dispatch(kind)
    return out, (pair, project)


class CaptureCounter:
    """Python dispatch count; graph replays do not re-enter Python."""

    def __init__(self, module, fused, shape):
        self.calls = 0
        self.capture_calls = 0
        self._capturing = False
        self._module = module
        self._fused = fused
        self.shape = shape

    def preflight(self, features):
        if tuple(features.shape) != self.shape:
            raise ValueError("Preflight input does not match selected block shape")
        return preflight(features, self._module, self._fused)

    @contextmanager
    def capture(self, *, expected=1):
        """Mark a caller-known capture region and require its dispatch count."""
        if self._capturing or not isinstance(expected, int) or expected < 1:
            raise ValueError("Expected one non-nested capture region")
        before = self.capture_calls
        self._capturing = True
        try:
            yield self
        finally:
            self._capturing = False
        if self.capture_calls - before != expected:
            raise RuntimeError("Selected MLP missed or repeated the marked capture")


def _select(stack, family, side, index):
    from native_cubic_adapters_v1 import FusedBatched
    from nr_backend.multihead_block import BranchedMLP, MultiHeadSwinBlock

    if family not in _BASE or side not in ("encoder", "decoder"):
        raise ValueError("Select one C128/C256 encoder or decoder block")
    group_index = {"encoder": {128: 2, 256: 3},
                   "decoder": {128: 1, 256: 0}}[side][family]
    group = getattr(stack.model, side)[group_index]
    if not isinstance(index, int) or not 0 <= index < len(group):
        raise ValueError("Invalid selected block index")
    block = group[index]
    if side == "decoder" and index == 0:
        block = block.body
    if (not isinstance(block, MultiHeadSwinBlock) or block.channels != family or
            not isinstance(block.mlp, BranchedMLP)):
        raise RuntimeError("Unexpected selected 720p block")
    matches = [item for item in stack.components if isinstance(item, FusedBatched)]
    if len(matches) != 1 or id(block.mlp) not in matches[0].modules:
        raise RuntimeError("Expected one owning active FusedBatched component")
    return block, matches[0]


@contextmanager
def installed(stack, *, family: int, side: str, index: int):
    """Replace only the selected block's fused MLP call, before graph capture.

    Caller explicitly uses ``counter.preflight(sample)`` on an FP16 XPU sample
    with ``counter.shape``, then marks capture with ``counter.capture()``.
    Compare standard/unrounded frame tensors against
    the same run with this scope absent; replay counts need graph route evidence.
    """
    block, fused = _select(stack, family, side, index)
    if "apply" in fused.__dict__:
        raise RuntimeError("Active FusedBatched already has an instance override")
    target = block.mlp
    h, w = _BASE[family]
    sy, sx = block.window_shift
    shape = (h + (8 if sy else 0), w + (8 if sx else 0), family)
    original = fused.apply
    counter = CaptureCounter(target, fused, shape)

    def selected_apply(self, module, features):
        if module is not target:
            return original(module, features)
        if (tuple(features.shape) != shape or current_arithmetic_backend() != "triton"
                or self.provider.mode != "fp16_xmx"):
            raise RuntimeError("Selected 720p MLP execution boundary changed")
        value, _ = forward(features, module, self)
        self.calls[str(family)] = self.calls.get(str(family), 0) + 1
        counter.calls += 1
        if counter._capturing and torch.xpu.is_current_stream_capturing():
            counter.capture_calls += 1
        return value

    bound = MethodType(selected_apply, fused)
    fused.apply = bound
    try:
        yield counter
    finally:
        if fused.__dict__.get("apply") is not bound:
            raise RuntimeError("Selected FusedBatched override changed during scope")
        del fused.apply
