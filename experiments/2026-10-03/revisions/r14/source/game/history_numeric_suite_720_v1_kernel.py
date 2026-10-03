"""Owned fixed720 history native-ALU kernels, authored but not GPU-compiled.

Axes and complete five bilinear taps preserve the HWC3/HW raw contract.
Pixel lanes share RGB addressing/coefficients; NEAR retains all five taps.
WRITE_NORMALIZED=false removes only the unused non-debug store. DEBUG,
reference/table fallbacks and INVALID outputs remain independently screened.
Input summary combines actual finite/range and half-motion near predicates.
"""
import triton
import triton.language as tl


@triton.jit
def _inverse(value, TABLE, NATIVE: tl.constexpr):
    start: tl.constexpr = 0x3F780000
    size: tl.constexpr = 0x3F940000 - 0x3F780000
    index = value.to(tl.int32, bitcast=True) - start
    good = (index >= 0) & (index < size)
    if NATIVE:
        # Correctly rounded FP32 1/x division, not an approximate rcp intrinsic.
        inverse = tl.div_rn(1.0, value)
    else:
        inverse = tl.load(TABLE + tl.minimum(size - 1, tl.maximum(0, index)))
    return inverse, good


@triton.jit
def _axis(position, extent: tl.constexpr, TABLE, p, base, A,
          N: tl.constexpr, NATIVE: tl.constexpr, DIRECT: tl.constexpr):
    if DIRECT:
        center = tl.floor(position - 0.5) + 0.5
        t = tl.minimum(1.0, tl.maximum(0.0, position - center))
    else:
        center = tl.floor(tl.fma(position, extent, -0.5)) + 0.5
        t = tl.minimum(1.0, tl.maximum(0.0, tl.fma(position, extent, -center)))
    t2 = t * t
    t3 = t * t2
    w0 = tl.fma(t + t3, -0.5, t2)
    w1 = tl.fma(t3, 1.5, -(t2 * 2.5)) + 1.0
    w3 = (t3 - t2) * 0.5
    w2 = ((1.0 - w0) - w1) - w3
    group = w1 + w2
    inverse, good = _inverse(group, TABLE, NATIVE)
    middle = tl.fma(w2, inverse, center)
    tl.store(A + (base + 0) * N + p, center, p < N)
    tl.store(A + (base + 1) * N + p, w0, p < N)
    tl.store(A + (base + 2) * N + p, group, p < N)
    tl.store(A + (base + 3) * N + p, w3, p < N)
    tl.store(A + (base + 4) * N + p, middle, p < N)
    return ~good


@triton.jit
def _prepare_axes(MOTION, IW, IH, TABLE, AXES, INVALID,
                  H: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr,
                  NATIVE: tl.constexpr, DIRECT: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = p < H * W
    x = p % W
    y = p // W
    mx = tl.load(MOTION + p * 2, valid, 0).to(tl.float16).to(tl.float32)
    my = tl.load(MOTION + p * 2 + 1, valid, 0).to(tl.float16).to(tl.float32)
    if DIRECT:
        # Same current-to-previous displacement in pixels, after half motion.
        u = (x.to(tl.float32) + 0.5) + mx
        v = (y.to(tl.float32) + 0.5) + my
    else:
        iw = tl.load(IW)
        ih = tl.load(IH)
        u = tl.fma(mx, 1.0 / W, (x.to(tl.float32) + 0.5) * iw)
        v = tl.fma(my, 1.0 / H, (y.to(tl.float32) + 0.5) * ih)
    badx = _axis(u, W, TABLE, p, 0, AXES, H * W, NATIVE, DIRECT)
    bady = _axis(v, H, TABLE, p, 5, AXES, H * W, NATIVE, DIRECT)
    tl.store(INVALID + p, (badx | bady).to(tl.int32), valid)


@triton.jit
def _exp32(value):
    bits = value.to(tl.int32, bitcast=True)
    return tl.where(value == 0.0, -1000, ((bits >> 23) & 255) - 127)


@triton.jit
def _texture_counts(IMAGE, cx, cy, ch, H: tl.constexpr, W: tl.constexpr,
                    VALUE_FP32: tl.constexpr, INTEGER_TEXEL: tl.constexpr):
    cx = tl.minimum((W - 1) * 256, tl.maximum(0, cx))
    cy = tl.minimum((H - 1) * 256, tl.maximum(0, cy))
    ix = (cx >> 8).to(tl.int32)
    iy = (cy >> 8).to(tl.int32)
    jx = tl.minimum(ix + 1, W - 1)
    jy = tl.minimum(iy + 1, H - 1)
    ax = (cx & 255).to(tl.int32)
    ay = (cy & 255).to(tl.int32)
    cross = (ax * ay + 128) >> 8
    k0 = 256 - ax - ay + cross
    k1 = ax - cross
    k2 = ay - cross
    k3 = cross
    a0 = tl.load(IMAGE + (iy * W + ix) * 3 + ch).to(tl.float16)
    a1 = tl.load(IMAGE + (iy * W + jx) * 3 + ch).to(tl.float16)
    a2 = tl.load(IMAGE + (jy * W + ix) * 3 + ch).to(tl.float16)
    a3 = tl.load(IMAGE + (jy * W + jx) * 3 + ch).to(tl.float16)
    f0 = a0.to(tl.float32)
    f1 = a1.to(tl.float32)
    f2 = a2.to(tl.float32)
    f3 = a3.to(tl.float32)
    if VALUE_FP32:
        # Fixed reference integer coefficients and product/add order, no lerp.
        weighted = ((f0 * k0 + f1 * k1) + f2 * k2) + f3 * k3
        fractional = (weighted * (1.0 / 256.0)).to(tl.float16).to(tl.float32)
    else:
        # Reviewed fused._half_texture value algorithm, including signed zero.
        e0 = tl.where((f0 != 0.0) & (k0 != 0), _exp32(tl.abs(f0)), -1000)
        e1 = tl.where((f1 != 0.0) & (k1 != 0), _exp32(tl.abs(f1)), -1000)
        e2 = tl.where((f2 != 0.0) & (k2 != 0), _exp32(tl.abs(f2)), -1000)
        e3 = tl.where((f3 != 0.0) & (k3 != 0), _exp32(tl.abs(f3)), -1000)
        exponent = tl.maximum(-24, tl.maximum(tl.maximum(e0, e1), tl.maximum(e2, e3)))
        scale_bits = ((14 - exponent + 127) << 23).to(tl.int32)
        scale = scale_bits.to(tl.float32, bitcast=True)
        summed = ((f0 * scale).to(tl.int64) * k0.to(tl.int64) +
                  (f1 * scale).to(tl.int64) * k1.to(tl.int64) +
                  (f2 * scale).to(tl.int64) * k2.to(tl.int64) +
                  (f3 * scale).to(tl.int64) * k3.to(tl.int64))
        magnitude = tl.abs(summed)
        lead = tl.minimum(62, tl.maximum(0, _exp32(magnitude.to(tl.float32))))
        lead = lead - (magnitude < (tl.full(magnitude.shape, 1, tl.int64) << lead)).to(tl.int32)
        lead = tl.where(magnitude == 0, 0, lead)
        scale_exponent = exponent - 22
        result_exponent = lead + scale_exponent
        shift = tl.maximum(result_exponent - 10, -24) - scale_exponent
        rs = tl.minimum(62, tl.maximum(0, shift))
        quotient = magnitude >> rs
        remainder = magnitude - (quotient << rs)
        midpoint = tl.full(magnitude.shape, 1, tl.int64) << tl.minimum(61, tl.maximum(0, rs - 1))
        rounded = quotient + ((rs > 0) & (remainder >= midpoint)).to(tl.int64)
        rounded = tl.where(shift < 0, magnitude << tl.minimum(62, tl.maximum(0, -shift)), rounded)
        bits = tl.minimum(0x7c00, tl.maximum(0, result_exponent + 14).to(tl.int64) * 1024 + rounded)
        bits = tl.where(magnitude == 0, 0, bits) | ((summed < 0).to(tl.int64) << 15)
        negzero = (summed == 0) & ((k0 == 0) | ((a0.to(tl.uint16, bitcast=True) & 0x8000) != 0)) & ((k1 == 0) | ((a1.to(tl.uint16, bitcast=True) & 0x8000) != 0)) & ((k2 == 0) | ((a2.to(tl.uint16, bitcast=True) & 0x8000) != 0)) & ((k3 == 0) | ((a3.to(tl.uint16, bitcast=True) & 0x8000) != 0))
        bits = bits.to(tl.uint16) | (negzero.to(tl.uint16) << 15)
        fractional = bits.to(tl.float16, bitcast=True).to(tl.float32)
    if INTEGER_TEXEL:
        # Fractional fused reference shortcut; backend counts has no shortcut.
        return tl.where(((ax == 0) & (ay == 0)), f0, fractional)
    return fractional


@triton.jit
def _half_texture(IMAGE, px, py, ch, IW, IH, H: tl.constexpr, W: tl.constexpr,
                  VALUE_FP32: tl.constexpr, DIRECT: tl.constexpr):
    xc = tl.minimum(W - 0.5, tl.maximum(0.5, px))
    yc = tl.minimum(H - 0.5, tl.maximum(0.5, py))
    if DIRECT:
        # Reference pixel-center 1/256 rounding; remove only normalized mapping.
        x = xc - 0.5
        y = yc - 0.5
        ix = tl.floor(x)
        iy = tl.floor(y)
        ax = tl.floor((x - ix) * 256.0 + 0.5)
        ay = tl.floor((y - iy) * 256.0 + 0.5)
        cx = ix.to(tl.int64) * 256 + ax.to(tl.int64)
        cy = iy.to(tl.int64) * 256 + ay.to(tl.int64)
    else:
        iw = tl.load(IW)
        ih = tl.load(IH)
        u = tl.fma(xc * iw, W, 0.0) * (1.0 / W)
        v = tl.fma(yc * ih, H, 0.0) * (1.0 / H)
        nx = tl.floor(u * 2097152.0).to(tl.int64)
        ny = tl.floor(v * 2097152.0).to(tl.int64)
        cx = ((nx * W + 4096) >> 13) - 128
        cy = ((ny * H + 4096) >> 13) - 128
    return _texture_counts(IMAGE, cx, cy, ch, H, W, VALUE_FP32, True)


@triton.jit
def _five_tap(IMAGE, AXES, IW, IH, TABLE, NUMERATOR, RECIPROCAL, NORMALIZED,
              INVALID, TAPS, H: tl.constexpr, W: tl.constexpr,
              DEBUG: tl.constexpr, BLOCK: tl.constexpr,
              VALUE_FP32: tl.constexpr, NATIVE: tl.constexpr, DIRECT: tl.constexpr,
              WRITE_NORMALIZED: tl.constexpr = True):
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = q < H * W * 3
    p = q // 3
    ch = q % 3
    n = H * W
    bx = tl.load(AXES + p, valid, 0)
    x0 = tl.load(AXES + n + p, valid, 0)
    gx = tl.load(AXES + 2 * n + p, valid, 0)
    x3 = tl.load(AXES + 3 * n + p, valid, 0)
    mx = tl.load(AXES + 4 * n + p, valid, 0)
    by = tl.load(AXES + 5 * n + p, valid, 0)
    y0 = tl.load(AXES + 6 * n + p, valid, 0)
    gy = tl.load(AXES + 7 * n + p, valid, 0)
    y3 = tl.load(AXES + 8 * n + p, valid, 0)
    my = tl.load(AXES + 9 * n + p, valid, 0)
    k0, k1, k2, k3, k4 = x0 * gy, y0 * gx, gx * gy, y3 * gx, x3 * gy
    total = (((k0 + k1) + k2) + k3) + k4
    inverse, good = _inverse(total, TABLE, NATIVE)
    s1 = _half_texture(IMAGE, mx, by - 1.0, ch, IW, IH, H, W, VALUE_FP32, DIRECT)
    s0 = _half_texture(IMAGE, bx - 1.0, my, ch, IW, IH, H, W, VALUE_FP32, DIRECT)
    s2 = _half_texture(IMAGE, mx, my, ch, IW, IH, H, W, VALUE_FP32, DIRECT)
    s3 = _half_texture(IMAGE, mx, by + 2.0, ch, IW, IH, H, W, VALUE_FP32, DIRECT)
    s4 = _half_texture(IMAGE, bx + 2.0, my, ch, IW, IH, H, W, VALUE_FP32, DIRECT)
    if DEBUG:
        tl.store(TAPS + q, s0, valid)
        tl.store(TAPS + n * 3 + q, s1, valid)
        tl.store(TAPS + n * 6 + q, s2, valid)
        tl.store(TAPS + n * 9 + q, s3, valid)
        tl.store(TAPS + n * 12 + q, s4, valid)
    value = s1 * k1
    value = tl.fma(s0, k0, value)
    value = tl.fma(s2, k2, value)
    value = tl.fma(s3, k3, value)
    value = tl.fma(s4, k4, value)
    tl.store(NUMERATOR + q, value, valid)
    if WRITE_NORMALIZED:
        tl.store(NORMALIZED + q, value * inverse, valid)
    tl.store(RECIPROCAL + p, inverse, valid & (ch == 0))
    tl.store(INVALID + p, (~good).to(tl.int32), valid & (ch == 0))


@triton.jit
def _pixel_coefficients(px, py, IW, IH, H: tl.constexpr, W: tl.constexpr,
                        DIRECT: tl.constexpr):
    """Coordinates and four coefficients once per pixel/tap, shared by RGB."""
    xc = tl.minimum(W - 0.5, tl.maximum(0.5, px))
    yc = tl.minimum(H - 0.5, tl.maximum(0.5, py))
    if DIRECT:
        x, y = xc - 0.5, yc - 0.5
        ix, iy = tl.floor(x), tl.floor(y)
        cx = ix.to(tl.int32) * 256 + tl.floor((x - ix) * 256.0 + 0.5).to(tl.int32)
        cy = iy.to(tl.int32) * 256 + tl.floor((y - iy) * 256.0 + 0.5).to(tl.int32)
    else:
        u = tl.fma(xc * tl.load(IW), W, 0.0) * (1.0 / W)
        v = tl.fma(yc * tl.load(IH), H, 0.0) * (1.0 / H)
        # Clamped Q21 * extent fits uint32 (1280 * 2^21 < 2^32).
        # Convert to signed ONLY after shifting and before subtracting 128.
        nx = tl.floor(u * 2097152.0).to(tl.uint32)
        ny = tl.floor(v * 2097152.0).to(tl.uint32)
        cx = ((nx * W + 4096) >> 13).to(tl.int32) - 128
        cy = ((ny * H + 4096) >> 13).to(tl.int32) - 128
    cx = tl.minimum((W - 1) * 256, tl.maximum(0, cx))
    cy = tl.minimum((H - 1) * 256, tl.maximum(0, cy))
    ix, iy = cx >> 8, cy >> 8
    jx, jy = tl.minimum(ix + 1, W - 1), tl.minimum(iy + 1, H - 1)
    ax, ay = cx & 255, cy & 255
    cross = (ax * ay + 128) >> 8
    return (iy * W + ix) * 3, (iy * W + jx) * 3, (jy * W + ix) * 3, (jy * W + jx) * 3, 256 - ax - ay + cross, ax - cross, ay - cross, cross, (ax == 0) & (ay == 0)


@triton.jit
def _pixel_value(IMAGE, coefficients, ch: tl.constexpr, mask,
                 INTEGER_TEXEL: tl.constexpr):
    a, b, c, d, k0, k1, k2, k3, integer = coefficients
    f0 = tl.load(IMAGE + a + ch, mask, 0).to(tl.float16).to(tl.float32)
    f1 = tl.load(IMAGE + b + ch, mask, 0).to(tl.float16).to(tl.float32)
    f2 = tl.load(IMAGE + c + ch, mask, 0).to(tl.float16).to(tl.float32)
    f3 = tl.load(IMAGE + d + ch, mask, 0).to(tl.float16).to(tl.float32)
    weighted = ((f0 * k0 + f1 * k1) + f2 * k2) + f3 * k3
    value = (weighted * (1.0 / 256.0)).to(tl.float16).to(tl.float32)
    if INTEGER_TEXEL:
        value = tl.where(integer, f0, value)
    return value


@triton.jit
def _five_tap_rgb(IMAGE, AXES, IW, IH, TABLE, NUMERATOR, RECIPROCAL, NORMALIZED,
                  INVALID, TAPS, H: tl.constexpr, W: tl.constexpr,
                  DEBUG: tl.constexpr, BLOCK: tl.constexpr, NATIVE: tl.constexpr,
                  DIRECT: tl.constexpr, NEAR: tl.constexpr,
                  WRITE_NORMALIZED: tl.constexpr):
    """Native ALU sampler: complete five bilinear taps, HWC3 consumers unchanged.

    NEAR disables only the fractional integer-texel shortcut. The half-motion
    route predicate and all cubic/group/total weights remain intact.
    """
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n: tl.constexpr = H * W
    mask = p < n
    bx = tl.load(AXES + p, mask, 0)
    x0 = tl.load(AXES + n + p, mask, 0)
    gx = tl.load(AXES + 2 * n + p, mask, 0)
    x3 = tl.load(AXES + 3 * n + p, mask, 0)
    mx = tl.load(AXES + 4 * n + p, mask, 0)
    by = tl.load(AXES + 5 * n + p, mask, 0)
    y0 = tl.load(AXES + 6 * n + p, mask, 0)
    gy = tl.load(AXES + 7 * n + p, mask, 0)
    y3 = tl.load(AXES + 8 * n + p, mask, 0)
    my = tl.load(AXES + 9 * n + p, mask, 0)
    k0, k1, k2, k3, k4 = x0 * gy, y0 * gx, gx * gy, y3 * gx, x3 * gy
    total = (((k0 + k1) + k2) + k3) + k4
    inverse, good = _inverse(total, TABLE, NATIVE)
    c0 = _pixel_coefficients(bx - 1.0, my, IW, IH, H, W, DIRECT)
    c1 = _pixel_coefficients(mx, by - 1.0, IW, IH, H, W, DIRECT)
    c2 = _pixel_coefficients(mx, my, IW, IH, H, W, DIRECT)
    c3 = _pixel_coefficients(mx, by + 2.0, IW, IH, H, W, DIRECT)
    c4 = _pixel_coefficients(bx + 2.0, my, IW, IH, H, W, DIRECT)
    for ch in tl.static_range(3):
        s1 = _pixel_value(IMAGE, c1, ch, mask, not NEAR)
        s0 = _pixel_value(IMAGE, c0, ch, mask, not NEAR)
        s2 = _pixel_value(IMAGE, c2, ch, mask, not NEAR)
        s3 = _pixel_value(IMAGE, c3, ch, mask, not NEAR)
        s4 = _pixel_value(IMAGE, c4, ch, mask, not NEAR)
        value = s1 * k1
        value = tl.fma(s0, k0, value)
        value = tl.fma(s2, k2, value)
        value = tl.fma(s3, k3, value)
        value = tl.fma(s4, k4, value)
        q = p * 3 + ch
        tl.store(NUMERATOR + q, value, mask)
        if WRITE_NORMALIZED:
            tl.store(NORMALIZED + q, value * inverse, mask)
        if DEBUG:
            tl.store(TAPS + q, s0, mask)
            tl.store(TAPS + n * 3 + q, s1, mask)
            tl.store(TAPS + n * 6 + q, s2, mask)
            tl.store(TAPS + n * 9 + q, s3, mask)
            tl.store(TAPS + n * 12 + q, s4, mask)
    tl.store(RECIPROCAL + p, inverse, mask)
    tl.store(INVALID + p, (~good).to(tl.int32), mask)


@triton.jit
def _counts_fp32(IMAGE, CX, CY, OUTPUT, H: tl.constexpr, W: tl.constexpr,
                 BLOCK: tl.constexpr):
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = q < H * W * 3
    p = q // 3
    cx = tl.load(CX + p, valid, 0)
    cy = tl.load(CY + p, valid, 0)
    value = _texture_counts(IMAGE, cx, cy, q % 3, H, W, True, False)
    tl.store(OUTPUT + q, value, valid)


@triton.jit
def _divide(VALUES, OUTPUT, INVALID, TABLE, COUNT: tl.constexpr,
            BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = p < COUNT
    value = tl.load(VALUES + p, valid, 1.0)
    inverse, good = _inverse(value, TABLE, True)
    tl.store(OUTPUT + p, inverse, valid)
    tl.store(INVALID + p, (~good).to(tl.int32), valid)


@triton.jit
def _input_summary(MOTION, RGB, PARTIAL, MOTION_COUNT: tl.constexpr,
                   RGB_COUNT: tl.constexpr, CHECK_COLOR: tl.constexpr, BLOCK: tl.constexpr):
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mvalid = q < MOTION_COUNT
    value = tl.load(MOTION + q, mvalid, 0).to(tl.float32)
    finite = (value == value) & (tl.abs(value) <= 3.4028234663852886e38)
    range_bad = tl.abs(value) > 65504.0
    mv = value.to(tl.float16).to(tl.float32)
    # Round ties do not affect near admission: half-integer distance .5 is
    # rejected on either side. This implements the same <=1/256 predicate.
    outside_near = (~finite) | range_bad | (tl.abs(mv - tl.floor(mv + 0.5)) > (1.0 / 256.0))
    flags = tl.max((mvalid & ~finite).to(tl.int32), 0)
    flags = flags | (tl.max((mvalid & range_bad).to(tl.int32), 0) << 1)
    flags = flags | (tl.max((mvalid & outside_near).to(tl.int32), 0) << 2)
    if CHECK_COLOR:
        rvalid = q < RGB_COUNT
        rgb = tl.load(RGB + q, rvalid, 0).to(tl.float32)
        color_finite = (rgb == rgb) & (tl.abs(rgb) <= 3.4028234663852886e38)
        flags = flags | (tl.max((rvalid & ~color_finite).to(tl.int32), 0) << 3)
        flags = flags | (tl.max((rvalid & ((rgb < 0.0) | (rgb > 1.0))).to(tl.int32), 0) << 4)
    tl.store(PARTIAL + tl.program_id(0), flags)


@triton.jit
def _summary_flags(PARTIAL, FLAGS, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    q = tl.arange(0, BLOCK)
    parts = tl.load(PARTIAL + q, q < COUNT, 0)
    flags = tl.full((), 0, tl.int32)
    for bit in tl.static_range(5):
        flags = flags | (tl.max(((parts & (1 << bit)) != 0).to(tl.int32), 0) << bit)
    tl.store(FLAGS, flags)
