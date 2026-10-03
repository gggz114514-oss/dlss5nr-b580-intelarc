"""Owned 720p FP16 XMX kernels; fast numerical variants, never exact-backend hooks.

The packed ABI stays [heads, Hp/8, Wp/8, 64, 32]. The Swin exponential
bit transform is retained. None of these kernels reads the cubic LUT.
Compilation/resource/numerical/performance gates require a separate GPU run.
"""
import triton
import triton.language as tl

from native_half_cubic_v1 import cubic
from native_half_attention_fma_v1 import half_fma_attention
from fused_swin_core_native_half_v1 import _exp, _halves, _weights_pair
from nr_backend.triton_attention_normalize import _nan_left, rsqrt_half_clamped


@triton.jit
def _load_input(X, row, col, M: tl.constexpr, C: tl.constexpr,
                H: tl.constexpr, W: tl.constexpr, WP: tl.constexpr,
                SY: tl.constexpr, SX: tl.constexpr):
    """Zero extension from HWC input, with no materialized shifted input."""
    y = row // WP - SY
    x = row % WP - SX
    valid = (row < M) & (y >= 0) & (y < H) & (x >= 0) & (x < W)
    return tl.load(X + (y[:, None] * W + x[:, None]) * C + col[None, :],
                   valid[:, None] & (col[None, :] < C), other=0)


@triton.jit
def _join_parts(a, b, BM: tl.constexpr, N: tl.constexpr):
    return tl.reshape(tl.permute(tl.join(a, b), (0, 2, 1)), (BM, 2 * N))


@triton.jit
def _join64(a, b, BM: tl.constexpr):
    return _join_parts(a, b, BM, 32)


@triton.jit
def _mlp_pairs(X, EXPAND, REDUCE, LUT, LATENT,
               M: tl.constexpr, C: tl.constexpr, BM: tl.constexpr,
               ROUND_REDUCED: tl.constexpr,
               H: tl.constexpr, W: tl.constexpr, WP: tl.constexpr,
               SY: tl.constexpr, SX: tl.constexpr):
    """Two branches share X; N64 expansion, ordered two K32 contractions.

    Latent remains branch-major. Both branches are real for C64/128/256;
    no branch-padding convention is introduced. Each dot sees increasing K32.
    """
    tl.static_assert(not ROUND_REDUCED)
    tl.static_assert(C == 64 or C == 128 or C == 256)
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    branch = 2 * tl.program_id(1)
    lane = tl.arange(0, 32)
    wide = tl.arange(0, 64)
    reduced0 = tl.full((BM, 32), 0., tl.float32)
    reduced1 = tl.full((BM, 32), 0., tl.float32)
    for pair in range(2):
        expand0 = tl.full((BM, 64), 0., tl.float32)
        expand1 = tl.full((BM, 64), 0., tl.float32)
        for block in range(C // 32):
            k = block * 32 + lane
            x = _load_input(X, row, k, M, C, H, W, WP, SY, SX)
            w0 = tl.load(EXPAND + (branch * C + k[:, None]) * 128 +
                         pair * 64 + wide[None, :])
            w1 = tl.load(EXPAND + ((branch + 1) * C + k[:, None]) * 128 +
                         pair * 64 + wide[None, :])
            expand0 = tl.dot(x, w0, expand0, out_dtype=tl.float32)
            expand1 = tl.dot(x, w1, expand1, out_dtype=tl.float32)
        hidden0 = cubic(expand0.to(tl.float16), ROUND_OUTPUT=False)
        hidden1 = cubic(expand1.to(tl.float16), ROUND_OUTPUT=False)
        low0, high0 = _halves(hidden0, BM, 64)
        low1, high1 = _halves(hidden1, BM, 64)
        r00 = tl.load(REDUCE + (branch * 128 + pair * 64 + lane[:, None]) * 32 + lane[None, :])
        r01 = tl.load(REDUCE + (branch * 128 + pair * 64 + 32 + lane[:, None]) * 32 + lane[None, :])
        r10 = tl.load(REDUCE + ((branch + 1) * 128 + pair * 64 + lane[:, None]) * 32 + lane[None, :])
        r11 = tl.load(REDUCE + ((branch + 1) * 128 + pair * 64 + 32 + lane[:, None]) * 32 + lane[None, :])
        reduced0 = tl.dot(low0, r00, reduced0, out_dtype=tl.float32)
        reduced0 = tl.dot(high0, r01, reduced0, out_dtype=tl.float32)
        reduced1 = tl.dot(low1, r10, reduced1, out_dtype=tl.float32)
        reduced1 = tl.dot(high1, r11, reduced1, out_dtype=tl.float32)
    tl.store(LATENT + (branch * M + row[:, None]) * 32 + lane[None, :],
             reduced0.to(tl.float16), row[:, None] < M)
    tl.store(LATENT + ((branch + 1) * M + row[:, None]) * 32 + lane[None, :],
             reduced1.to(tl.float16), row[:, None] < M)


@triton.jit
def _mlp_project(X, LATENT, PROJECT, SCALE, OUT,
                 M: tl.constexpr, C: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                 H: tl.constexpr, W: tl.constexpr, WP: tl.constexpr,
                 SY: tl.constexpr, SX: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    lane = tl.arange(0, 32)
    raw = _load_input(X, row, col, M, C, H, W, WP, SY, SX)
    scale = tl.load(SCALE + col, col < C, other=0)
    result = (raw.to(tl.float32) * scale[None, :].to(tl.float32)).to(tl.float16).to(tl.float32)
    for branch in range(C // 32):
        hidden = tl.load(LATENT + (branch * M + row[:, None]) * 32 + lane[None, :],
                         row[:, None] < M, other=0)
        weight = tl.load(PROJECT + (branch * 32 + lane[:, None]) * C + col[None, :],
                         col[None, :] < C, other=0)
        projected = tl.dot(hidden, weight, out_dtype=tl.float32)
        result = projected + result
    tl.store(OUT + row[:, None] * C + col[None, :], result.to(tl.float16),
             (row[:, None] < M) & (col[None, :] < C))


@triton.jit
def _c32_mlp(X, EXPAND, CONTRACT, SCALE, LUT, OUT,
             M: tl.constexpr, BM: tl.constexpr, ROUND_ACTIVATION: tl.constexpr,
             H: tl.constexpr, W: tl.constexpr, WP: tl.constexpr,
             SY: tl.constexpr, SX: tl.constexpr):
    tl.static_assert(not ROUND_ACTIVATION)
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    lane = tl.arange(0, 32)
    wide = tl.arange(0, 64)
    raw = _load_input(X, row, lane, M, 32, H, W, WP, SY, SX)
    contracted = tl.full((BM, 32), 0., tl.float32)
    for pair in range(2):
        weight = tl.load(EXPAND + lane[:, None] * 128 + pair * 64 + wide[None, :])
        expanded = tl.dot(raw, weight, out_dtype=tl.float32)
        hidden = cubic(expanded.to(tl.float16), ROUND_OUTPUT=False)
        low, high = _halves(hidden, BM, 64)
        w0 = tl.load(CONTRACT + (pair * 64 + lane[:, None]) * 32 + lane[None, :])
        w1 = tl.load(CONTRACT + (pair * 64 + 32 + lane[:, None]) * 32 + lane[None, :])
        contracted = tl.dot(low, w0, contracted, out_dtype=tl.float32)
        contracted = tl.dot(high, w1, contracted, out_dtype=tl.float32)
    scale = tl.load(SCALE + lane)
    initial = (raw.to(tl.float32) * scale[None, :].to(tl.float32)).to(tl.float16)
    result = (contracted + initial.to(tl.float32)).to(tl.float16)
    tl.store(OUT + row[:, None] * 32 + lane[None, :], result, row[:, None] < M)


@triton.jit
def _normalize_ordered(z, BM: tl.constexpr):
    """Local copy of the current native-half algorithm, without shared edits."""
    lane = tl.arange(0, 8)
    lo, hi = _halves(z, BM, 32)
    x0, x8 = _halves(lo, BM, 16)
    x16, x24 = _halves(hi, BM, 16)
    a = (x16.to(tl.float32) * x16.to(tl.float32)).to(tl.float16)
    a = _nan_left(half_fma_attention(x0, x0, a), x0, a)
    b = (x24.to(tl.float32) * x24.to(tl.float32)).to(tl.float16)
    b = _nan_left(half_fma_attention(x8, x8, b), x8, b)
    total = _nan_left((a.to(tl.float32) + b.to(tl.float32)).to(tl.float16), a, b)
    for mask in tl.static_range(3):
        other = tl.gather(total, tl.broadcast_to((lane ^ (4 >> mask))[None, :], (BM, 8)), 1)
        total = _nan_left((total.to(tl.float32) + other.to(tl.float32)).to(tl.float16), total, other)
    denom = tl.gather(total, tl.full((BM, 1), 0, tl.int32), 1)
    scale = rsqrt_half_clamped(denom)
    x0 = _nan_left((x0.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x0, scale)
    x8 = _nan_left((x8.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x8, scale)
    x16 = _nan_left((x16.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x16, scale)
    x24 = _nan_left((x24.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x24, scale)
    return _join_parts(_join_parts(x0, x8, BM, 8), _join_parts(x16, x24, BM, 8), BM, 16)


@triton.jit
def _normalize(z, BM: tl.constexpr, FP32: tl.constexpr):
    if FP32:
        # Exceptional tiles retain the ordered half/left-NaN policy. Finite
        # native accumulation has no half-square/tree overflow boundaries.
        bits = z.to(tl.uint16, bitcast=True).to(tl.int32)
        exceptional = tl.sum(tl.sum(((bits & 0x7fff) >= 0x7c00).to(tl.int32), 1), 0)
        if exceptional > 0:
            result = _normalize_ordered(z, BM)
        else:
            squares = z.to(tl.float32) * z.to(tl.float32)
            denom = tl.sum(squares, 1)[:, None]
            scale = tl.rsqrt(tl.maximum(denom, 6.198883056640625e-05)).to(tl.float16)
            result = _nan_left((z.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), z, scale)
    else:
        result = _normalize_ordered(z, BM)
    return result


@triton.jit
def _qkv(X, WEIGHT, SCALE, ORDER, Q, K, V,
          M: tl.constexpr, WP: tl.constexpr, C: tl.constexpr, BM: tl.constexpr,
          NORM_FP32: tl.constexpr, WIDE_N: tl.constexpr):
    """One shared input feeds adjacent N32 segments in a single N64 dot.

    Row order is already the consumer's token order; physical input is gathered
    directly. Segment 3*heads (C32 final pair) is masked, never normalized/stored.
    """
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    pair = tl.program_id(1)
    lane = tl.arange(0, 32)
    N: tl.constexpr = 64 if WIDE_N else 32
    wide = tl.arange(0, N)
    window = row // 64
    token = row % 64
    physical = tl.load(ORDER + token).to(tl.int32)
    y = (window // (WP // 8)) * 8 + physical // 8
    x = (window % (WP // 8)) * 8 + physical % 8
    source_row = y * WP + x
    columns = pair * N + wide
    acc = tl.full((BM, N), 0., tl.float32)
    for block in range(C // 32):
        k = block * 32 + lane
        input = tl.load(X + source_row[:, None] * C + k[None, :], row[:, None] < M, other=0)
        weight = tl.load(WEIGHT + k[:, None] * (3 * C) + columns[None, :],
                         columns[None, :] < 3 * C, other=0)
        acc = tl.dot(input, weight, acc, out_dtype=tl.float32)
    if WIDE_N:
        z0, z1 = _halves(acc.to(tl.float16), BM, 64)
    else:
        z0 = acc.to(tl.float16)
        z1 = tl.full((BM, 32), 0., tl.float16)
    for half in tl.static_range(N // 32):
        segment = pair * (N // 32) + half
        family = segment % 3
        head = segment // 3
        z = z0 if half == 0 else z1
        if segment < 3 * (C // 32):
            if family < 2:
                z = _normalize(z, BM, NORM_FP32)
                if family == 0:
                    scale = tl.load(SCALE + head)
                    z = _nan_left((z.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), z, scale)
            output = tl.where(family == 0, Q, tl.where(family == 1, K, V))
            tl.store(output + (head * M + row[:, None]) * 32 + lane[None, :], z, row[:, None] < M)


@triton.jit
def _weights_native(e0, e1, BM: tl.constexpr, FP32: tl.constexpr):
    if FP32:
        values = _join64(e0, e1, BM)
        bits = values.to(tl.uint16, bitcast=True).to(tl.int32)
        exceptional = tl.sum(tl.sum(((bits & 0x7fff) >= 0x7c00).to(tl.int32), 1), 0)
        if exceptional > 0:
            p0, p1 = _weights_pair(e0, e1, BM, ROUND_WEIGHTS=False)
        else:
            denominator = tl.sum(values.to(tl.float32), 1)[:, None]
            reciprocal = tl.div_rn(1., tl.maximum(denominator, 6.198883056640625e-05))
            p0 = (e0.to(tl.float32) * reciprocal).to(tl.float16)
            p1 = (e1.to(tl.float32) * reciprocal).to(tl.float16)
    else:
        p0, p1 = _weights_pair(e0, e1, BM, ROUND_WEIGHTS=False)
    return p0, p1


@triton.jit
def _attention_project(MLP, Q, K, V, BIAS, WEIGHT, SCALE, ORDER, OUT,
                       HP: tl.constexpr, WP: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                       SY: tl.constexpr, SX: tl.constexpr, C: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr, DENOM_FP32: tl.constexpr,
                       WIDE_SCORE: tl.constexpr, WIDE_VALUE: tl.constexpr,
                       CROP_QUERIES: tl.constexpr):
    """All local families; stream heads, direct crop, no global attended tensor."""
    window = tl.program_id(1)
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    column = tl.program_id(2) * BN + tl.arange(0, BN)
    lane = tl.arange(0, 32)
    keys = tl.arange(0, 64)
    physical = tl.load(ORDER + row).to(tl.int32)
    y = (window // (WP // 8)) * 8 + physical // 8
    x = (window % (WP // 8)) * 8 + physical % 8
    oy, ox = y - SY, x - SX
    valid = (oy >= 0) & (oy < H) & (ox >= 0) & (ox < W)
    # Never omit padded keys. Only a uniform wholly-cropped query tile can skip.
    if not CROP_QUERIES or tl.sum(valid.to(tl.int32), 0) > 0:
        total = tl.full((BM, BN), 0., tl.float32)
        for head in range(C // 32):
            offset = (head * (HP * WP // 64) + window) * 2048
            q = tl.load(Q + offset + row[:, None] * 32 + lane[None, :])
            if WIDE_SCORE:
                k = tl.load(K + offset + keys[None, :] * 32 + lane[:, None])
                score = tl.dot(q, k, out_dtype=tl.float32)
                bias = tl.load(BIAS + head * 4096 + row[:, None] * 64 + keys[None, :])
                e = _exp((score + bias.to(tl.float32)).to(tl.float16))
                e0, e1 = _halves(e, BM, 64)
            else:
                k0 = tl.load(K + offset + lane[None, :] * 32 + lane[:, None])
                k1 = tl.load(K + offset + (lane[None, :] + 32) * 32 + lane[:, None])
                s0 = tl.dot(q, k0, out_dtype=tl.float32)
                s1 = tl.dot(q, k1, out_dtype=tl.float32)
                b0 = tl.load(BIAS + head * 4096 + row[:, None] * 64 + lane[None, :])
                b1 = tl.load(BIAS + head * 4096 + row[:, None] * 64 + 32 + lane[None, :])
                e0 = _exp((s0 + b0.to(tl.float32)).to(tl.float16))
                e1 = _exp((s1 + b1.to(tl.float32)).to(tl.float16))
            p0, p1 = _weights_native(e0, e1, BM, DENOM_FP32)
            if WIDE_VALUE:
                probabilities = _join64(p0, p1, BM)
                value = tl.load(V + offset + keys[:, None] * 32 + lane[None, :])
                attended = tl.dot(probabilities, value, out_dtype=tl.float32).to(tl.float16)
            else:
                v0 = tl.load(V + offset + lane[:, None] * 32 + lane[None, :])
                v1 = tl.load(V + offset + (lane[:, None] + 32) * 32 + lane[None, :])
                attended = tl.dot(p0, v0, out_dtype=tl.float32)
                attended = tl.dot(p1, v1, attended, out_dtype=tl.float32).to(tl.float16)
            projection = tl.load(WEIGHT + (head * 32 + lane[:, None]) * C + column[None, :],
                                 column[None, :] < C, other=0)
            total = tl.dot(attended, projection, total, out_dtype=tl.float32)
        residual = tl.load(MLP + (y[:, None] * WP + x[:, None]) * C + column[None, :],
                           column[None, :] < C, other=0)
        scale = tl.load(SCALE + column, column < C, other=0)
        initial = (residual.to(tl.float32) * scale[None, :].to(tl.float32)).to(tl.float16)
        result = (total + initial.to(tl.float32)).to(tl.float16)
        tl.store(OUT + (oy[:, None] * W + ox[:, None]) * C + column[None, :], result,
                 valid[:, None] & (column[None, :] < C))
