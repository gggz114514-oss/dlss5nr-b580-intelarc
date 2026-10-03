"""C512 complete INT8 FFN screen with a runtime row count instead of M144.

Same five-kernel split as the reviewed module, same BM/BN/BK, same num_warps,
num_stages and enable_fp_fusion, same packed weights and scales, same K
accumulation order, same rounding, same cubic activation and same residual. The
only change is that the total row count is a constexpr instead of the literal
144, so one launch covers every row and the caller no longer pads a tail batch
and concatenates the pieces.

Every kernel here is row-independent: loads are masked per row, reductions run
along K only, and no op mixes two rows. Rows 0..M-1 must therefore come out
byte-identical to the reviewed padded-batch path. Zero spills stays a resource
gate, not a speed claim.
"""
import torch
import triton
import triton.language as tl
from short_fp8_v2 import round_half
from int8_ffn_segment_gpu_v1 import _q
from spill_preflight_v1 import select


@triton.jit
def _cubic(x):
    t = tl.minimum(tl.maximum(x.to(tl.float32), -4.), 4.).to(tl.float16)
    p = tl.fma(-tl.abs(t), tl.full((), .055908203125, tl.float16), tl.full((), .447265625, tl.float16)).to(tl.float16)
    v = tl.fma(t, p, tl.full((), .89453125, tl.float16)).to(tl.float16)
    return (x.to(tl.float32) * v.to(tl.float32)).to(tl.float16)


@triton.jit
def _entry(X, QX, SX, M: tl.constexpr, ROUND_FP8: tl.constexpr=True):
    row = tl.program_id(0)
    col = tl.arange(0, 512)
    x = tl.load(X + row*512 + col)
    if ROUND_FP8:
        x = round_half(x)
    x = x.to(tl.float32)
    maximum = tl.max(tl.abs(x), 0)
    scale = tl.where(maximum > 0, tl.div_rn(maximum, 127.), 1.)
    tl.store(QX + row*512 + col, _q(x, scale))
    tl.store(SX + row, scale)


@triton.jit
def _linear(QX, SX, W, SW, SZ, QZ, RAW, M: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, DEBUG: tl.constexpr):
    row = tl.program_id(0)*BM + tl.arange(0, BM)
    col = tl.program_id(1)*BN + tl.arange(0, BN)
    lane = tl.arange(0, 32)
    total = tl.full((BM, BN), 0, tl.int32)
    for start in range(16):
        k = start*32 + lane
        a = tl.load(QX + row[:, None]*512 + k[None, :], row[:, None] < M, other=0)
        w = tl.load(W + col[None, :]*512 + k[:, None])
        total = tl.dot(a, w, total, out_dtype=tl.int32)
    sx = tl.load(SX + row, row < M, other=1.)
    sw, sz = tl.load(SW + col), tl.load(SZ + col)
    z = ((total.to(tl.float32)*sx[:, None])*sw[None, :]).to(tl.float16)
    offset = row[:, None]*512 + col[None, :]
    tl.store(QZ + offset, _q(z.to(tl.float32), sz[None, :]), row[:, None] < M)
    if DEBUG:
        tl.store(RAW + offset, z, row[:, None] < M)


@triton.jit
def _expand(QZ, W, SW, SH, QH, RAW, M: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, DEBUG: tl.constexpr):
    row = tl.program_id(0)*BM + tl.arange(0, BM)
    group = tl.program_id(1)
    col = tl.program_id(2)*BN + tl.arange(0, BN)
    lane = tl.arange(0, 32)
    total = tl.full((BM, BN), 0, tl.int32)
    for start in range(2):
        k = start*32 + lane
        a = tl.load(QZ + row[:, None]*512 + group*64 + k[None, :], row[:, None] < M, other=0)
        w = tl.load(W + group*256*64 + col[None, :]*64 + k[:, None])
        total = tl.dot(a, w, total, out_dtype=tl.int32)
    sw, sh = tl.load(SW + group*256 + col), tl.load(SH + group*256 + col)
    h = _cubic((total.to(tl.float32)*sw[None, :]).to(tl.float16))
    offset = row[:, None]*2048 + group*256 + col[None, :]
    tl.store(QH + offset, _q(h.to(tl.float32), sh[None, :]), row[:, None] < M)
    if DEBUG:
        tl.store(RAW + offset, h, row[:, None] < M)


@triton.jit
def _reduce(QH, W, SW, SG, QG, RAW, M: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, DEBUG: tl.constexpr):
    row = tl.program_id(0)*BM + tl.arange(0, BM)
    group = tl.program_id(1)
    col = tl.program_id(2)*BN + tl.arange(0, BN)
    lane = tl.arange(0, 32)
    total = tl.full((BM, BN), 0, tl.int32)
    for start in range(8):
        k = start*32 + lane
        a = tl.load(QH + row[:, None]*2048 + group*256 + k[None, :], row[:, None] < M, other=0)
        w = tl.load(W + group*64*256 + col[None, :]*256 + k[:, None])
        total = tl.dot(a, w, total, out_dtype=tl.int32)
    sw, sg = tl.load(SW + group*64 + col), tl.load(SG + group*64 + col)
    g = (total.to(tl.float32)*sw[None, :]).to(tl.float16)
    offset = row[:, None]*512 + group*64 + col[None, :]
    tl.store(QG + offset, _q(g.to(tl.float32), sg[None, :]), row[:, None] < M)
    if DEBUG:
        tl.store(RAW + offset, g, row[:, None] < M)


@triton.jit
def _groups(QZ, WE, SE, SH, WR, SR, SG, QG, RAW, M: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, DEBUG: tl.constexpr):
    row = tl.program_id(0)*BM + tl.arange(0, BM)
    group = tl.program_id(1)
    col = tl.program_id(2)*BN + tl.arange(0, BN)
    lane = tl.arange(0, 32)
    total = tl.full((BM, BN), 0, tl.int32)
    for part in range(8):
        hcol = part*32 + lane
        expanded = tl.full((BM, 32), 0, tl.int32)
        for start in range(2):
            k = start*32 + lane
            a = tl.load(QZ + row[:, None]*512 + group*64 + k[None, :], row[:, None] < M, other=0)
            we = tl.load(WE + group*256*64 + hcol[None, :]*64 + k[:, None])
            expanded = tl.dot(a, we, expanded, out_dtype=tl.int32)
        se, sh = tl.load(SE + group*256 + hcol), tl.load(SH + group*256 + hcol)
        hidden = _cubic((expanded.to(tl.float32)*se[None, :]).to(tl.float16))
        qh = _q(hidden.to(tl.float32), sh[None, :])
        wr = tl.load(WR + group*64*256 + col[None, :]*256 + hcol[:, None])
        total = tl.dot(qh, wr, total, out_dtype=tl.int32)
    sr, sg = tl.load(SR + group*64 + col), tl.load(SG + group*64 + col)
    g = (total.to(tl.float32)*sr[None, :]).to(tl.float16)
    offset = row[:, None]*512 + group*64 + col[None, :]
    tl.store(QG + offset, _q(g.to(tl.float32), sg[None, :]), row[:, None] < M)
    if DEBUG:
        tl.store(RAW + offset, g, row[:, None] < M)


@triton.jit
def _project(QG, W, SW, X, SKIP, OUT, RAW, M: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, DEBUG: tl.constexpr,
             ROUND_FP8: tl.constexpr=True):
    row = tl.program_id(0)*BM + tl.arange(0, BM)
    col = tl.program_id(1)*BN + tl.arange(0, BN)
    lane = tl.arange(0, 32)
    total = tl.full((BM, BN), 0, tl.int32)
    for start in range(16):
        k = start*32 + lane
        a = tl.load(QG + row[:, None]*512 + k[None, :], row[:, None] < M, other=0)
        w = tl.load(W + col[None, :]*512 + k[:, None])
        total = tl.dot(a, w, total, out_dtype=tl.int32)
    offset = row[:, None]*512 + col[None, :]
    x = tl.load(X + offset, row[:, None] < M, other=0)
    if ROUND_FP8:
        x = round_half(x)
    sw, skip = tl.load(SW + col), tl.load(SKIP + col)
    initial = (x.to(tl.float32)*skip[None, :].to(tl.float32)).to(tl.float16)
    raw = (total.to(tl.float32)*sw[None, :] + initial.to(tl.float32)).to(tl.float16)
    output = raw
    if ROUND_FP8:
        output = round_half(raw)
    tl.store(OUT + offset, output, row[:, None] < M)
    if DEBUG:
        tl.store(RAW + offset, raw, row[:, None] < M)


# Written-tensor contract, keyed by the real module name. Dataflow.launch fails
# closed on unknown kernels, so these must be registered before any dispatch.
EXTRA = {f'{__name__}.{name}': value for name, value in {
    '_entry': (('QX', 'SX'), ()),
    '_linear': (('QZ',), ()),
    '_expand': (('QH',), ()),
    '_reduce': (('QG',), ()),
    '_project': (('OUT',), ('OUT',)),
}.items()}

# Frozen launch choices from the reviewed module. select() walks them in order.
DENSE_CONFIGS = ((32, 64), (16, 64), (16, 32))
GROUP_CONFIGS = ((16, 64), (16, 32))
OPTIONS = dict(num_warps=4, num_stages=1, enable_fp_fusion=False)
