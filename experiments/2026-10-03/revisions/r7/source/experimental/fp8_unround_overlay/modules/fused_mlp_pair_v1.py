"""Stream four 32-channel expansion tiles through cubic into the reduction.

One branch's expand/cubic/FP8/reduce/FP8 is fused. The 128-channel activation is
never materialized or kept fully live. Consecutive K32 FP32 sums and every old
half/FP8 boundary are preserved. Projection and sequential half skip stay separate.
"""
import torch
import triton
import triton.language as tl
from nr_backend.triton_fp8 import _round_fp8_half
from fused_branched_mlp_v1 import _cubic


@triton.jit
def _kernel(X, EXPAND, REDUCE, OUT, M:tl.constexpr, C:tl.constexpr, BM:tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    lane = tl.arange(0, 32)
    reduced = tl.full((BM, 32), 0., tl.float32)
    for part in range(4):
        expanded = tl.full((BM, 32), 0., tl.float32)
        for block in range(C // 32):
            k = block * 32 + lane
            x = tl.load(X + row[:, None] * C + k[None, :], row[:, None] < M, other=0)
            w = tl.load(EXPAND + k[:, None] * 128 + part * 32 + lane[None, :])
            expanded = tl.dot(x, w, expanded, out_dtype=tl.float32)
        hidden = _cubic(expanded.to(tl.float16))
        w = tl.load(REDUCE + (part * 32 + lane[:, None]) * 32 + lane[None, :])
        reduced = tl.dot(hidden, w, reduced, out_dtype=tl.float32)
    result = _round_fp8_half(reduced.to(tl.float16))
    tl.store(OUT + row[:, None] * 32 + lane[None, :], result, row[:, None] < M)


def forward(x, expand, reduce, *, bm=16, warps=4, stages=1):
    c = x.shape[-1]
    if c not in (64,128,256) or bm not in (16,32) or warps not in (4,8) or stages not in (1,2):
        raise ValueError('Unsupported pair configuration')
    if x.device.type != 'xpu' or x.dtype != torch.float16 or x.numel() == 0:
        raise ValueError('Expected prequantized finite half XPU features')
    for value, shape in ((expand,(c,128)), (reduce,(128,32))):
        if value.shape != shape or value.dtype != torch.float16 or value.device != x.device or not value.is_contiguous():
            raise ValueError('Invalid pair weight')
    x = x.contiguous()
    out = torch.empty((*x.shape[:-1],32), dtype=x.dtype, device=x.device)
    m = x.numel() // c
    kernel = _kernel[(triton.cdiv(m,bm),)](x,expand,reduce,out,m,c,bm,
        num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    return out,kernel
