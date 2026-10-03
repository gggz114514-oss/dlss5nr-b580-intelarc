"""Compute only used C512 queries, preserving every original key/value position.

The complete Q/K/V inputs stay [16,2,2,64,32]. Output is [16,144,32] in
head/HWC-pixel/lane order; the projection consumes it without a layout copy.
Only query rows are rescheduled. Bias rows use the original pixel permutation,
and K32 dot, native half arithmetic, FP8 weights and FP8 output are unchanged.
"""
import torch
import triton
import triton.language as tl
from fused_swin_core_native_half_v1 import _exp, _weights_pair
from nr_backend.triton_fp8 import _round_fp8_half
from spill_preflight_v1 import select


def geometry(shift, bm):
    sy, sx = shift
    if sy not in (0, 4) or sx not in (0, 4) or bm not in (16, 32):
        raise ValueError('Unsupported C512 query geometry')
    result = []
    for window in range(4):
        wy, wx = divmod(window, 2)
        y0, x0 = max(sy-wy*8, 0), max(sx-wx*8, 0)
        height = min(sy+12-wy*8, 8)-y0
        width = min(sx+12-wx*8, 8)-x0
        result.append(dict(window=window, y0=y0, x0=x0, height=height,
                           width=width, tiles=(height*width+bm-1)//bm))
    assert sum(v['height']*v['width'] for v in result) == 144
    return result


@triton.jit
def _attend(Q, K, V, BIAS, INVERSE, OUT, SY:tl.constexpr, SX:tl.constexpr,
            BM:tl.constexpr, C0:tl.constexpr, C1:tl.constexpr, C2:tl.constexpr):
    tile, head = tl.program_id(0), tl.program_id(1)
    window = tl.where(tile < C0, 0, tl.where(tile < C0+C1, 1,
                        tl.where(tile < C0+C1+C2, 2, 3)))
    first = tl.where(window == 0, 0, tl.where(window == 1, C0,
                        tl.where(window == 2, C0+C1, C0+C1+C2)))
    row = (tile-first)*BM+tl.arange(0, BM)
    wy, wx = window//2, window%2
    y0, x0 = tl.maximum(SY-wy*8, 0), tl.maximum(SX-wx*8, 0)
    height = tl.minimum(SY+12-wy*8, 8)-y0
    width = tl.minimum(SX+12-wx*8, 8)-x0
    valid = row < height*width
    y, x = y0+row//width, x0+row%width
    local = tl.load(INVERSE+y*8+x, valid, other=0).to(tl.int32)
    lane = tl.arange(0, 32)
    base = (head*4+window)*2048
    q = tl.load(Q+base+local[:, None]*32+lane[None, :], valid[:, None], other=0)
    k0 = tl.load(K+base+lane[None, :]*32+lane[:, None])
    k1 = tl.load(K+base+(lane[None, :]+32)*32+lane[:, None])
    score0 = tl.dot(q, k0, out_dtype=tl.float32)
    score1 = tl.dot(q, k1, out_dtype=tl.float32)
    b0 = tl.load(BIAS+head*4096+local[:, None]*64+lane[None, :])
    b1 = tl.load(BIAS+head*4096+local[:, None]*64+lane[None, :]+32)
    e0 = _exp((score0+b0.to(tl.float32)).to(tl.float16))
    e1 = _exp((score1+b1.to(tl.float32)).to(tl.float16))
    w0, w1 = _weights_pair(e0, e1, BM)
    v0 = tl.load(V+base+lane[:, None]*32+lane[None, :])
    v1 = tl.load(V+base+(lane[:, None]+32)*32+lane[None, :])
    result = tl.dot(w0, v0, out_dtype=tl.float32)
    result = tl.dot(w1, v1, result, out_dtype=tl.float32)
    pixel = (wy*8+y-SY)*12+wx*8+x-SX
    tl.store(OUT+(head*144+pixel[:, None])*32+lane[None, :],
             _round_fp8_half(result.to(tl.float16)), valid[:, None])


@triton.jit
def _project(X, W, RESIDUAL, SCALE, OUT, BM:tl.constexpr, BN:tl.constexpr):
    rows = tl.program_id(0)*BM+tl.arange(0, BM)
    cols = tl.program_id(1)*BN+tl.arange(0, BN)
    kk = tl.arange(0, 32)
    total = tl.full((BM, BN), 0, tl.float32)
    for head in range(16):
        av = tl.load(X+(head*144+rows[:, None])*32+kk[None, :],
                     rows[:, None] < 144, other=0)
        wv = tl.load(W+(head*32+kk[:, None])*512+cols[None, :],
                     cols[None, :] < 512, other=0)
        total = tl.dot(av, wv, total, out_dtype=tl.float32)
    offset = rows[:, None]*512+cols[None, :]
    valid = (rows[:, None] < 144)&(cols[None, :] < 512)
    r = tl.load(RESIDUAL+offset, valid, other=0)
    s = tl.load(SCALE+cols, cols < 512, other=0)
    initial = (r.to(tl.float32)*s[None, :].to(tl.float32)).to(tl.float16)
    result = (total+initial.to(tl.float32)).to(tl.float16)
    tl.store(OUT+offset, result, valid)


def operands(values):
    first = values[0]
    if any(t.device.type != 'xpu' or t.device != first.device or
           t.dtype != torch.float16 or not t.is_contiguous() for t in values):
        raise ValueError('Expected contiguous half tensors on one XPU')


def attend(query, key, value, bias, inverse, *, shift):
    if any(tuple(t.shape) != (16, 2, 2, 64, 32) for t in (query, key, value)):
        raise ValueError('Expected complete C512 Q/K/V windows')
    if tuple(bias.shape) != (16, 64, 64):
        raise ValueError('Expected full head-specific bias')
    operands((query, key, value, bias))
    if inverse.shape != (64,) or inverse.dtype != torch.int64 or inverse.device != query.device or not inverse.is_contiguous():
        raise ValueError('Expected owned inverse pixel order')
    geometry(shift, 32)
    out = torch.empty((16, 144, 32), device=query.device, dtype=query.dtype)
    def args(config):
        g = geometry(shift, config[0])
        return (query, key, value, bias, inverse, out, *shift, config[0],
                g[0]['tiles'], g[1]['tiles'], g[2]['tiles'])
    def grid(config):
        return (sum(v['tiles'] for v in geometry(shift, config[0])), 16)
    options = dict(num_warps=4, num_stages=1, enable_fp_fusion=False)
    config, compiled, decision = select(_attend, [(32,), (16,)], args, grid, **options)
    assert _attend[grid(config)](*args(config), **options) is compiled
    return out, compiled, decision


def project(features, weight, residual, scale):
    if features.shape != (16, 144, 32) or weight.shape != (512, 512) or residual.shape != (12, 12, 512) or scale.shape != (512,):
        raise ValueError('Expected compact C512 projection geometry')
    operands((features, weight, residual, scale))
    out = torch.empty_like(residual)
    args = (features, weight, residual, scale, out, 16, 32)
    grid = (9, 16)
    options = dict(num_warps=4, enable_fp_fusion=False)
    _, compiled, decision = select(_project, [(16, 32)], lambda _: args, lambda _: grid, **options)
    assert _project[grid](*args, **options) is compiled
    return out, compiled, decision
