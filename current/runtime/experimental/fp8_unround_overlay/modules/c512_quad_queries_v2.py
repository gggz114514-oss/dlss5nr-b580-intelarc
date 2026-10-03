"""C512 valid queries in native contiguous 4x4 tiles, consumed without a copy.

Fixed pixel_order: four row-major 4x4 quadrants in each 8x8 window. Callers
must authenticate the model permutation before integration. All 64 K/V rows,
original head-specific bias rows and the fast route's half/FP8 math remain.
Output is [head,9 valid tiles,16 native query rows,32 lanes], not HWC order.
"""
import triton
import triton.language as tl
from fused_swin_core_native_half_v1 import _exp, _weights_pair
from nr_backend.triton_fp8 import _round_fp8_half


def tile_geometry(shift):
    sy, sx = shift
    if sy not in (0, 4) or sx not in (0, 4):
        raise ValueError('Expected C512 shifts 0 or 4')
    rows = []
    for tile in range(9):
        qy, qx = tile//3+sy//4, tile%3+sx//4
        rows.append(dict(tile=tile, window=(qy//2)*2+qx//2,
                         query_start=((qy%2)*2+qx%2)*16))
    return rows


@triton.jit
def _attend(Q, K, V, BIAS, OUT, SY:tl.constexpr, SX:tl.constexpr):
    tile, head = tl.program_id(0), tl.program_id(1)
    qy, qx = tile//3+SY//4, tile%3+SX//4
    window = (qy//2)*2+qx//2
    row = ((qy%2)*2+qx%2)*16+tl.arange(0, 16)
    lane = tl.arange(0, 32)
    base = (head*4+window)*2048
    q = tl.load(Q+base+row[:, None]*32+lane[None, :])
    k0 = tl.load(K+base+lane[None, :]*32+lane[:, None])
    k1 = tl.load(K+base+(lane[None, :]+32)*32+lane[:, None])
    score0 = tl.dot(q, k0, out_dtype=tl.float32)
    score1 = tl.dot(q, k1, out_dtype=tl.float32)
    b0 = tl.load(BIAS+head*4096+row[:, None]*64+lane[None, :])
    b1 = tl.load(BIAS+head*4096+row[:, None]*64+lane[None, :]+32)
    e0 = _exp((score0+b0.to(tl.float32)).to(tl.float16))
    e1 = _exp((score1+b1.to(tl.float32)).to(tl.float16))
    w0, w1 = _weights_pair(e0, e1, 16)
    v0 = tl.load(V+base+lane[:, None]*32+lane[None, :])
    v1 = tl.load(V+base+(lane[:, None]+32)*32+lane[None, :])
    result = tl.dot(w0, v0, out_dtype=tl.float32)
    result = tl.dot(w1, v1, result, out_dtype=tl.float32)
    offset = (head*9+tile)*512+tl.arange(0, 16)[:, None]*32+lane[None, :]
    tl.store(OUT+offset, _round_fp8_half(result.to(tl.float16)))


@triton.jit
def _project(X, W, RESIDUAL, SCALE, OUT):
    rows = tl.program_id(0)*16+tl.arange(0, 16)
    cols = tl.program_id(1)*32+tl.arange(0, 32)
    kk = tl.arange(0, 32)
    y, x = rows//12, rows%12
    tile, local = (y//4)*3+x//4, (y%4)*4+x%4
    total = tl.full((16, 32), 0, tl.float32)
    for head in range(16):
        av = tl.load(X+((head*9+tile[:, None])*16+local[:, None])*32+kk[None, :])
        wv = tl.load(W+(head*32+kk[:, None])*512+cols[None, :])
        total = tl.dot(av, wv, total, out_dtype=tl.float32)
    offset = rows[:, None]*512+cols[None, :]
    r = tl.load(RESIDUAL+offset)
    s = tl.load(SCALE+cols)
    initial = (r.to(tl.float32)*s[None, :].to(tl.float32)).to(tl.float16)
    tl.store(OUT+offset, (total+initial.to(tl.float32)).to(tl.float16))
