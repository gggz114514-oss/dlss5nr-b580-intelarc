"""Owned fast720 kernels. Imported only by the enabled runtime hook.

No E4M3 boundary, midpoint correction, or software RTZ in this profile.
Each half store uses native RNE. GPU compilation/quality remains a probe gate.
"""
import triton
import triton.language as tl
from native_half_cubic_v1 import cubic
from fused_swin_core_native_half_v1 import _exp, _weights_pair


@triton.jit
def merge(P, S, SCALE, OUT, H: tl.constexpr, W: tl.constexpr,
          C: tl.constexpr, PS0: tl.constexpr, PS1: tl.constexpr,
          PS2: tl.constexpr, SS0: tl.constexpr, SS1: tl.constexpr,
          SS2: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    c = i % C
    x = i // C % W
    y = i // (W * C)
    mask = i < H * W * C
    p = tl.load(P + (y // 2) * PS0 + (x // 2) * PS1 + c * PS2, mask, other=0)
    s = tl.load(S + y * SS0 + x * SS1 + c * SS2, mask, other=0)
    scale = tl.load(SCALE + c, mask, other=0)
    value = tl.fma(s.to(tl.float32), scale.to(tl.float32), p.to(tl.float32))
    tl.store(OUT + i, value.to(tl.float16), mask)


@triton.jit
def matrix(X, WEIGHT, OUT, M: tl.constexpr, K: tl.constexpr,
           N: tl.constexpr, XS0: tl.constexpr, XS1: tl.constexpr,
           XS2: tl.constexpr, XW: tl.constexpr, POOL: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Full-K native FP16/FP32 dot; optional 2x2 average in its A loader.

    POOL stores no half top/bottom tensors. Average is F32 then one RNE half.
    XW is *output* spatial width; padded contexts stay in the source canvas.
    """
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    lane = tl.arange(0, BK)
    total = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + lane
        y = row // XW
        x = row % XW
        if POOL:
            address = (y * 2)[:, None] * XS0 + (x * 2)[:, None] * XS1 + k[None, :] * XS2
            valid = (row[:, None] < M) & (k[None, :] < K)
            a = tl.load(X + address, valid, other=0).to(tl.float32)
            b = tl.load(X + address + XS1, valid, other=0).to(tl.float32)
            c = tl.load(X + address + XS0, valid, other=0).to(tl.float32)
            d = tl.load(X + address + XS0 + XS1, valid, other=0).to(tl.float32)
            value = (((a + b) + (c + d)) * 0.25).to(tl.float16)
        else:
            value = tl.load(X + y[:, None] * XS0 + x[:, None] * XS1 + k[None, :] * XS2,
                            (row[:, None] < M) & (k[None, :] < K), other=0)
        weight = tl.load(WEIGHT + k[:, None] * N + col[None, :],
                         (k[:, None] < K) & (col[None, :] < N), other=0)
        total = tl.dot(value, weight, total, out_dtype=tl.float32)
    tl.store(OUT + row[:, None] * N + col[None, :], total.to(tl.float16),
             (row[:, None] < M) & (col[None, :] < N))


@triton.jit
def average(X, OUT, H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
            S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    c = i % C
    x = i // C % W
    y = i // (W * C)
    valid = i < H * W * C
    address = y * 2 * S0 + x * 2 * S1 + c * S2
    a = tl.load(X + address, valid, other=0).to(tl.float32)
    b = tl.load(X + address + S1, valid, other=0).to(tl.float32)
    c0 = tl.load(X + address + S0, valid, other=0).to(tl.float32)
    d = tl.load(X + address + S0 + S1, valid, other=0).to(tl.float32)
    tl.store(OUT + i, (((a + b) + (c0 + d)) * 0.25).to(tl.float16), valid)


@triton.jit
def _entry_value(FEATURES, SKIP, WEIGHT, SCALE, BM: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    lane = tl.arange(0, 32)
    y = row // 1288 - 4
    x = row % 1288 - 4
    valid = (row < 776 * 1288) & (y >= 0) & (y < 768) & (x >= 0) & (x < 1280)
    f = tl.load(FEATURES + ((y // 2) * 640 + x // 2)[:, None] * 32 + lane[None, :],
                valid[:, None], other=0)
    s = tl.load(SKIP + (y * 1280 + x)[:, None] * 32 + lane[None, :], valid[:, None], other=0)
    weight = tl.load(WEIGHT + lane).to(tl.float32)
    scale = tl.load(SCALE + lane).to(tl.float32)
    expanded = (f.to(tl.float32) * weight[None, :]).to(tl.float16)
    value = tl.fma(s.to(tl.float32), scale[None, :], expanded.to(tl.float32)).to(tl.float16)
    return tl.where(valid[:, None], value, 0).to(tl.float16)


@triton.jit
def entry(FEATURES, SKIP, WEIGHT, SCALE, OUT, BM: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    lane = tl.arange(0, 32)
    value = _entry_value(FEATURES, SKIP, WEIGHT, SCALE, BM)
    tl.store(OUT + row[:, None] * 32 + lane[None, :], value, row[:, None] < 776 * 1288)


@triton.jit
def entry_mlp(FEATURES, SKIP, WEIGHT, SCALE, EXPAND, CONTRACT, MLP_SCALE,
              OUT, BM: tl.constexpr):
    """Produce the complete native C32 MLP without materializing entry HWC32."""
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    lane = tl.arange(0, 32)
    raw = _entry_value(FEATURES, SKIP, WEIGHT, SCALE, BM)
    contracted = tl.full((BM, 32), 0, tl.float32)
    for part in tl.static_range(4):
        e = tl.load(EXPAND + lane[:, None] * 128 + part * 32 + lane[None, :])
        r = tl.load(CONTRACT + (part * 32 + lane[:, None]) * 32 + lane[None, :])
        expanded = tl.dot(raw, e, out_dtype=tl.float32).to(tl.float16)
        hidden = cubic(expanded, ROUND_OUTPUT=False)
        contracted = tl.dot(hidden, r, contracted, out_dtype=tl.float32)
    scale = tl.load(MLP_SCALE + lane).to(tl.float32)
    initial = (raw.to(tl.float32) * scale[None, :]).to(tl.float16)
    value = (contracted + initial.to(tl.float32)).to(tl.float16)
    tl.store(OUT + row[:, None] * 32 + lane[None, :], value, row[:, None] < 776 * 1288)


@triton.jit
def attention_native_head(MLP, Q, K, V, BIAS, PROJECT, SKIPSCALE, ORDER,
                          HEAD, OUT):
    """Current C32 attention/projection feeds native head in registers.

    Retain all 776x1288 windows and half projection/residual semantics. Only
    the final public 720x1280 head4 is stored; no 768x1280x32 intermediate.
    """
    window = tl.program_id(1)
    row = tl.program_id(0) * 32 + tl.arange(0, 32)
    lane = tl.arange(0, 32)
    query = tl.load(Q + window * 2048 + row[:, None] * 32 + lane[None, :])
    key0 = tl.load(K + window * 2048 + lane[None, :] * 32 + lane[:, None])
    key1 = tl.load(K + window * 2048 + (lane[None, :] + 32) * 32 + lane[:, None])
    score0 = tl.dot(query, key0, out_dtype=tl.float32)
    score1 = tl.dot(query, key1, out_dtype=tl.float32)
    bias0 = tl.load(BIAS + row[:, None] * 64 + lane[None, :])
    bias1 = tl.load(BIAS + row[:, None] * 64 + 32 + lane[None, :])
    exp0 = _exp((score0 + bias0.to(tl.float32)).to(tl.float16))
    exp1 = _exp((score1 + bias1.to(tl.float32)).to(tl.float16))
    probability0, probability1 = _weights_pair(exp0, exp1, 32, ROUND_WEIGHTS=False)
    value0 = tl.load(V + window * 2048 + lane[:, None] * 32 + lane[None, :])
    value1 = tl.load(V + window * 2048 + (lane[:, None] + 32) * 32 + lane[None, :])
    attended = tl.dot(probability0, value0, out_dtype=tl.float32)
    attended = tl.dot(probability1, value1, attended, out_dtype=tl.float32).to(tl.float16)
    projection = tl.load(PROJECT + lane[:, None] * 32 + lane[None, :])
    projected = tl.dot(attended, projection, out_dtype=tl.float32)
    pixel = tl.load(ORDER + row).to(tl.int32)
    y = window // 161 * 8 + pixel // 8 - 4
    x = window % 161 * 8 + pixel % 8 - 4
    original = tl.load(MLP + ((y + 4) * 1288 + (x + 4))[:, None] * 32 + lane[None, :])
    skip_scale = tl.load(SKIPSCALE + lane)
    initial = (original.to(tl.float32) * skip_scale[None, :].to(tl.float32)).to(tl.float16)
    projected = (projected + initial.to(tl.float32)).to(tl.float16)
    column = tl.arange(0, 16)
    head_weight = tl.load(HEAD + lane[:, None] * 8 + column[None, :], column[None, :] < 8, other=0)
    head = tl.dot(projected, head_weight, out_dtype=tl.float32).to(tl.float16)
    valid = (y >= 0) & (y < 720) & (x >= 0) & (x < 1280)
    tl.store(OUT + (y * 1280 + x)[:, None] * 4 + column[None, :], head,
             valid[:, None] & (column[None, :] < 4))


@triton.jit
def _native_sigmoid_value(x):
    # Exact formula from current post_numeric_suite_720_kernel_v1.py
    # sigmoid_half_to_float; CPU AST check verifies all three expressions.
    z = tl.exp(-tl.abs(x))
    denominator = 1.0 + z
    value = tl.where(x >= 0.0, 1.0 / denominator, z / denominator)
    return value


@triton.jit
def tail(HEAD, RGB, PREVIOUS, RECIPROCAL, BLEND, SIGMOID, OUT,
         TEMPORAL: tl.constexpr, NORMALIZE: tl.constexpr,
         POST_SIGMOID: tl.constexpr, B: tl.constexpr):
    tl.static_assert(POST_SIGMOID == "table" or POST_SIGMOID == "native",
                     "Owned fused tail supports current table or current native sigmoid only")
    i = tl.program_id(0) * B + tl.arange(0, B)
    valid = i < 720 * 1280 * 3
    pixel, channel = i // 3, i % 3
    head = tl.load(HEAD + pixel * 4 + channel, valid, other=0).to(tl.float32)
    rgb = tl.load(RGB + i, valid, other=0).to(tl.float16).to(tl.float32)
    value = (head * .03125 + (rgb * .125 - .0625)) * 8.0 + .5
    value = tl.minimum(1.0, tl.maximum(0.0, value))
    if TEMPORAL:
        logit = tl.load(HEAD + pixel * 4 + 3, valid, other=0).to(tl.float16)
        if POST_SIGMOID == "native":
            probability = _native_sigmoid_value(logit.to(tl.float32))
        else:
            bits = logit.to(tl.uint16, bitcast=True).to(tl.int32)
            probability = tl.load(SIGMOID + bits, valid, other=0)
        alpha = tl.minimum(1.0, tl.maximum(0.0, probability * tl.load(BLEND).to(tl.float32)))
        previous = tl.load(PREVIOUS + i, valid, other=0).to(tl.float32)
        if NORMALIZE:
            reciprocal = tl.load(RECIPROCAL + pixel, valid, other=0).to(tl.float32)
            delta = tl.fma(previous, reciprocal, -value)
        else:
            delta = previous - value
        value = tl.fma(delta, alpha, value)
    # Private temporal overshoot is deliberately retained. Public geometry
    # publication is owned by the graph/history worker, which clamps its copy.
    tl.store(OUT + i, value, valid)


@triton.jit
def convert(X, OUT, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(X + i, i < 720 * 1280 * 3, other=0)
    tl.store(OUT + i, value.to(tl.float16), i < 720 * 1280 * 3)


@triton.jit
def _hsl_channel(hue, sat, light, shift: tl.constexpr):
    q = tl.where(light >= .5, tl.fma(-light, sat, light + sat), light * (sat + 1))
    p = (light + light) - q
    pos = hue + shift
    pos = tl.where(pos >= 0, pos, pos + 1)
    pos = tl.where(pos > 1, pos - 1, pos)
    difference = q - p
    rising = tl.fma(difference * 6, pos, p)
    falling = tl.fma(difference * (.6666666865348815918 - pos), 6., p)
    value = tl.where(pos < .16666667163372039795, rising,
                    tl.where(pos < .5, q, tl.where(pos < .6666666865348815918, falling, p)))
    return tl.where(sat > 0, value, light)


@triton.jit
def _style_curve(network, CURVE, channel: tl.constexpr,
                  STYLE: tl.constexpr, TONE, USE_CURVE: tl.constexpr):
    if USE_CURVE:
        bits = network.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.int32)
        # Invalid inputs are reported by STATUS, never used as an OOB index.
        # Explicit <=0 handling also keeps signed zero on the black curve row.
        bits = tl.where(network > 0, tl.minimum(bits, 0x3c00), 0)
        return tl.load(CURVE + bits * 3 + channel)
    if STYLE == 1:
        exposure = tl.exp2(-.10000000149011612 * TONE)
        contrast = -.25 * TONE
    else:
        exposure, contrast = 1., 0.
    value = tl.minimum(1., tl.maximum(0., network * exposure))
    delta = tl.fma(value * value, 3. - (value + value), -value)
    return tl.minimum(1., tl.maximum(0., tl.fma(delta, contrast, value)))


@triton.jit(do_not_specialize=["AMOUNT", "TONE"])
def style(NEURAL, ORIGINAL, CURVE, OUT, STATUS, AMOUNT, TONE,
          STYLE: tl.constexpr, USE_CURVE: tl.constexpr, B: tl.constexpr):
    pixel = tl.program_id(0) * B + tl.arange(0, B)
    valid = pixel < 720 * 1280
    n0 = tl.load(NEURAL + pixel * 3, valid, other=0).to(tl.float32)
    n1 = tl.load(NEURAL + pixel * 3 + 1, valid, other=0).to(tl.float32)
    n2 = tl.load(NEURAL + pixel * 3 + 2, valid, other=0).to(tl.float32)
    o0 = tl.load(ORIGINAL + pixel * 3, valid, other=0).to(tl.float32)
    o1 = tl.load(ORIGINAL + pixel * 3 + 1, valid, other=0).to(tl.float32)
    o2 = tl.load(ORIGINAL + pixel * 3 + 2, valid, other=0).to(tl.float32)
    bad = (tl.abs(n0) == float("inf")) | (n0 != n0) | (tl.abs(n1) == float("inf")) | (n1 != n1) | (tl.abs(n2) == float("inf")) | (n2 != n2)
    bad = bad | (o0 != o0) | (o1 != o1) | (o2 != o2) | (o0 < 0) | (o0 > 1) | (o1 < 0) | (o1 > 1) | (o2 < 0) | (o2 > 1)
    tl.store(STATUS + tl.program_id(0), tl.max((bad & valid).to(tl.int32), 0))
    r = _style_curve(tl.minimum(1., tl.maximum(0., n0.to(tl.float16).to(tl.float32) + 0.)), CURVE, 0, STYLE, TONE, USE_CURVE)
    g = _style_curve(tl.minimum(1., tl.maximum(0., n1.to(tl.float16).to(tl.float32) + 0.)), CURVE, 1, STYLE, TONE, USE_CURVE)
    b = _style_curve(tl.minimum(1., tl.maximum(0., n2.to(tl.float16).to(tl.float32) + 0.)), CURVE, 2, STYLE, TONE, USE_CURVE)
    maximum, minimum = tl.maximum(b, tl.maximum(r, g)), tl.minimum(b, tl.minimum(r, g))
    light, diff = (maximum + minimum) * .5, maximum - minimum
    active = maximum > minimum
    denominator = tl.where(light > .5, (2. - maximum) - minimum, maximum + minimum)
    inv_sum = 1. / tl.where(active, denominator, 1.)
    inv_diff = 1. / tl.where(active, diff, 1.)
    hue_r = tl.fma(g - b, inv_diff, tl.where(g >= b, 0., 6.)) * .16666667163372039795
    hue_g = tl.fma(b - r, inv_diff, 2.) * .16666667163372039795
    hue_b = tl.fma(r - g, inv_diff, 4.) * .16666667163372039795
    hue = tl.where(active, tl.where(maximum == r, hue_r, tl.where(maximum == g, hue_g, hue_b)), 0.)
    sat = tl.where(active, diff * inv_sum, 0.)
    factor = 1. + (-.10000000149011612 if STYLE == 1 else -.15000000596046448) * TONE
    sat = tl.minimum(1., tl.maximum(0., sat * factor))
    r = tl.minimum(1., tl.maximum(0., _hsl_channel(hue, sat, light, .3333333432674407959) + 0.))
    g = tl.minimum(1., tl.maximum(0., _hsl_channel(hue, sat, light, 0.) + 0.))
    b = tl.minimum(1., tl.maximum(0., _hsl_channel(hue, sat, light, -.3333333432674407959) + 0.))
    o0, o1, o2 = o0.to(tl.float16).to(tl.float32), o1.to(tl.float16).to(tl.float32), o2.to(tl.float16).to(tl.float32)
    tl.store(OUT + pixel * 3, tl.minimum(1., tl.maximum(0., tl.fma(r - o0, AMOUNT, o0) + 0.)), valid)
    tl.store(OUT + pixel * 3 + 1, tl.minimum(1., tl.maximum(0., tl.fma(g - o1, AMOUNT, o1) + 0.)), valid)
    tl.store(OUT + pixel * 3 + 2, tl.minimum(1., tl.maximum(0., tl.fma(b - o2, AMOUNT, o2) + 0.)), valid)
