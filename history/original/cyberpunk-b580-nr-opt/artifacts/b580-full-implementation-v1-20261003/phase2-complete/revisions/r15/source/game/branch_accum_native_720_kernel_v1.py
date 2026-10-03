"""Existing HWC branched projection with FP32 accumulation across branches.

Keep half latent, half-rounded skip initial, increasing branch order, tiles
and addresses. Only the inter-branch half stores become one final half store.
"""
import triton
import triton.language as tl


@triton.jit
def _project_fp32(X, LATENT, PROJECT, SCALE, OUT,
                  M: tl.constexpr, C: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    column = tl.program_id(1) * BN + tl.arange(0, BN)
    lane = tl.arange(0, 32)
    valid = (row[:, None] < M) & (column[None, :] < C)
    x = tl.load(X + row[:, None] * C + column[None, :], valid, other=0)
    scale = tl.load(SCALE + column, column < C, other=0)
    # This HALF initial is part of the original residual contract.
    initial = (x.to(tl.float32) * scale[None, :].to(tl.float32)).to(tl.float16)
    result = initial.to(tl.float32)
    for branch in range(C // 32):
        hidden = tl.load(LATENT + (branch * M + row[:, None]) * 32 + lane[None, :],
                         row[:, None] < M, other=0)
        weight = tl.load(PROJECT + (branch * 32 + lane[:, None]) * C + column[None, :],
                         column[None, :] < C, other=0)
        projected = tl.dot(hidden, weight, out_dtype=tl.float32)
        result = projected + result
    tl.store(OUT + row[:, None] * C + column[None, :], result.to(tl.float16), valid)
