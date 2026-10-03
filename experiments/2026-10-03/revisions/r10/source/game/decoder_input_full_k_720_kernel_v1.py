"""DecoderInput only: one full K1024 half GEMM, FP32 sum, one final half.

M240/N512 and HWC consumers are unchanged. This file is imported lazily by
the enabled scope; importing the public default-off API needs no GPU runtime.
"""
import triton
import triton.language as tl


@triton.jit
def _project_full_k(X, WEIGHT, INITIAL, OUT,
                    BM: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    column = tl.program_id(1) * BN + tl.arange(0, BN)
    lane = tl.arange(0, 32)
    valid = (row[:, None] < 240) & (column[None, :] < 512)
    total = tl.full((BM, BN), 0., tl.float32)
    for block in range(1024 // 32):
        k = block * 32 + lane
        x = tl.load(X + row[:, None] * 1024 + k[None, :],
                    row[:, None] < 240, other=0)
        weight = tl.load(WEIGHT + k[:, None] * 512 + column[None, :],
                         column[None, :] < 512, other=0)
        total = tl.dot(x, weight, total, out_dtype=tl.float32)
    initial = tl.load(INITIAL + row[:, None] * 512 + column[None, :],
                      valid, other=0)
    value = (total + initial.to(tl.float32)).to(tl.float16)
    tl.store(OUT + row[:, None] * 512 + column[None, :], value, valid)
