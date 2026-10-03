"""Native half unary cubic in the existing batched_branched_mlp_v1.py dataflow.

Keep all matrix and half/FP8 boundaries, launch options and ownership guards.
The LUT contract remains in the LUT-based APIs even though this kernel uses arithmetic.
"""
import torch
import triton
import triton.language as tl
from native_half_cubic_v1 import cubic
import nr_backend.multihead_block as multihead
from nr_backend.execution import current_arithmetic_backend, record_arithmetic_dispatch
from nr_backend.triton_fp8 import _round_fp8_half
from cubic_lut_constant_v1 import Constant
from fused_c32_mlp_lut_v1 import lookup
from fused_branched_pairs_v1 import FusedPairs as Base
from nr_backend.unround_policy import ENABLED, round_multihead


@triton.jit
def _pairs(X, EXPAND, REDUCE, LUT, LATENT, M: tl.constexpr, C: tl.constexpr, BM: tl.constexpr,
           ROUND_REDUCED: tl.constexpr = True):
    row = tl.program_id(0)*BM + tl.arange(0, BM)
    branch = tl.program_id(1)
    lane = tl.arange(0, 32)
    reduced = tl.full((BM, 32), 0., tl.float32)
    for part in range(4):
        expanded = tl.full((BM, 32), 0., tl.float32)
        for block in range(C//32):
            k = block*32 + lane
            x = tl.load(X + row[:, None]*C + k[None, :], row[:, None] < M, other=0)
            w = tl.load(EXPAND + (branch*C + k[:, None])*128 + part*32 + lane[None, :])
            expanded = tl.dot(x, w, expanded, out_dtype=tl.float32)
        hidden = cubic(expanded.to(tl.float16), ROUND_OUTPUT=ROUND_REDUCED)
        w = tl.load(REDUCE + (branch*128 + part*32 + lane[:, None])*32 + lane[None, :])
        reduced = tl.dot(hidden, w, reduced, out_dtype=tl.float32)
    value = reduced.to(tl.float16)
    if ROUND_REDUCED:
        value = _round_fp8_half(value)
    tl.store(LATENT + (branch*M + row[:, None])*32 + lane[None, :], value, row[:, None] < M)


@triton.jit
def _project(X, LATENT, PROJECT, SCALE, OUT, M: tl.constexpr, C: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0)*BM + tl.arange(0, BM)
    column = tl.program_id(1)*BN + tl.arange(0, BN)
    lane = tl.arange(0, 32)
    valid = (row[:, None] < M) & (column[None, :] < C)
    x = tl.load(X + row[:, None]*C + column[None, :], valid, other=0)
    scale = tl.load(SCALE + column, column < C, other=0)
    result = (x.to(tl.float32)*scale[None, :].to(tl.float32)).to(tl.float16)
    for branch in range(C//32):
        hidden = tl.load(LATENT + (branch*M + row[:, None])*32 + lane[None, :], row[:, None] < M, other=0)
        weight = tl.load(PROJECT + (branch*32 + lane[:, None])*C + column[None, :], column[None, :] < C, other=0)
        projected = tl.dot(hidden, weight, out_dtype=tl.float32)
        # This boundary is necessary: add to the previous HALF result in FP32,
        # then round to HALF before the next branch's projection is added.
        result = (projected + result.to(tl.float32)).to(tl.float16)
    tl.store(OUT + row[:, None]*C + column[None, :], result, valid)


def forward(features, expand, reduce, project, skip_scale, lut, *,
            pair_bm=16, pair_stages=1, project_bm=16, project_bn=32):
    c = features.shape[-1]
    if features.device.type != 'xpu' or c not in (64, 128, 256) or features.numel() == 0:
        raise ValueError('Expected nonempty XPU branch features')
    if pair_bm not in (16, 32) or pair_stages not in (1, 2) or project_bm not in (16, 32) or project_bn not in (32, 64):
        raise ValueError('Unsupported batched branch launch configuration')
    branches = c//32
    for value, shape in ((expand, (branches, c, 128)), (reduce, (branches, 128, 32)),
                         (project, (branches, 32, c)), (skip_scale, (c,))):
        if tuple(value.shape) != shape or value.dtype != torch.float16 or value.device != features.device or not value.is_contiguous():
            raise ValueError('Invalid branch weight buffer')
    if lut.device != features.device or lut.dtype != torch.int16 or lut.shape != (65536,) or not lut.is_contiguous():
        raise ValueError('Invalid owned half-bit cubic LUT')
    x = round_multihead(c,features)
    m = x.numel()//c
    latent = torch.empty((branches, m, 32), dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    first = _pairs[(triton.cdiv(m, pair_bm), branches)](
        x, expand, reduce, lut, latent, m, c, pair_bm,
        ROUND_REDUCED=f"c{c}" not in ENABLED,
        num_warps=4, num_stages=pair_stages, enable_fp_fusion=False)
    second = _project[(triton.cdiv(m, project_bm), triton.cdiv(c, project_bn))](
        x, latent, project, skip_scale, out, m, c, project_bm, project_bn,
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    # Keep logical accounting while replacing individual launches by two kernels.
    for _ in range(branches):
        record_arithmetic_dispatch('dense')
        record_arithmetic_dispatch('dense')
        record_arithmetic_dispatch('cubic_fp8')
        record_arithmetic_dispatch('fp8')
        record_arithmetic_dispatch('dense')
    return out, (first, second)


class FusedBatched(Base):
    def __init__(self, model, provider, *, configurations=None):
        super().__init__(model, provider)
        self.constant = Constant(model)
        self.configurations = {c: dict(pair_bm=16, pair_stages=2 if c == 256 else 1,
                                       project_bm=16, project_bn=32) for c in (64, 128, 256)}
        if configurations is not None:
            if set(configurations) != {64, 128, 256}:
                raise ValueError('Expected configurations for all three channel families')
            self.configurations = {c: dict(v) for c, v in configurations.items()}

    def apply(self, module, features):
        if (id(module) not in self.modules or current_arithmetic_backend() != 'triton'
                or features.device.type != 'xpu' or self.provider.mode != 'fp16_xmx'):
            return self.original(module, features)
        c = module.channels
        value, _ = forward(features, module.expand, module.reduce, module.project,
                           module.skip_scale, self.constant.require(), **self.configurations[c])
        self.calls[str(c)] = self.calls.get(str(c), 0) + 1
        return value
