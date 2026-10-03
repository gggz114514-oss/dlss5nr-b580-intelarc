"""Stdlib numerical oracle and predicates for the fixed 1280x720 contract.

CPU evidence only; this does not predict GPU error, bandwidth or timing.
"""
import math
import struct

H, W = 720, 1280


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def half(value):
    try:
        return struct.unpack('<e', struct.pack('<e', value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def half_bits(value):
    return struct.unpack('<H', struct.pack('<e', half(value)))[0]


def motion_summary(values):
    """bit0 invalid/nonfinite, bit1 out of half range, bit2 not whole-frame near."""
    flags = 0
    for value in values:
        if not math.isfinite(value):
            flags |= 1 | 4
        elif abs(value) > 65504:
            flags |= 2 | 4
        else:
            quantized = half(value)
            if abs(quantized - round(quantized)) > 1 / 256:
                flags |= 4
    return flags


def axis(position):
    center = math.floor(f32(position - 0.5)) + 0.5
    t = max(0.0, min(1.0, f32(position - center)))
    t2, t3 = f32(t * t), f32(t * f32(t * t))
    w0 = f32(f32(t + t3) * -0.5 + t2)
    w1 = f32(f32(t3 * 1.5 - f32(t2 * 2.5)) + 1.0)
    w3 = f32(f32(t3 - t2) * 0.5)
    w2 = f32(f32(f32(1.0 - w0) - w1) - w3)
    group = f32(w1 + w2)
    return center, w0, group, w3, f32(w2 * f32(1.0 / group) + center)


def coefficients(px, py, *, direct=True, iw=None, ih=None):
    xc, yc = max(0.5, min(W - 0.5, px)), max(0.5, min(H - 0.5, py))
    if direct:
        x, y = f32(xc - 0.5), f32(yc - 0.5)
        cx = math.floor(x) * 256 + math.floor(f32(f32(x - math.floor(x)) * 256.0) + 0.5)
        cy = math.floor(y) * 256 + math.floor(f32(f32(y - math.floor(y)) * 256.0) + 0.5)
    else:
        iw = f32(1.0 / W) if iw is None else iw
        ih = f32(1.0 / H) if ih is None else ih
        u = f32(f32(f32(xc * iw) * W) * f32(1.0 / W))
        v = f32(f32(f32(yc * ih) * H) * f32(1.0 / H))
        nx, ny = math.floor(f32(u * 2097152.0)), math.floor(f32(v * 2097152.0))
        if max(nx * W + 4096, ny * H + 4096) >= 2**32:
            raise OverflowError('Clamped Q21 coordinate exceeds uint32')
        cx, cy = ((nx * W + 4096) >> 13) - 128, ((ny * H + 4096) >> 13) - 128
    cx, cy = max(0, min((W - 1) * 256, cx)), max(0, min((H - 1) * 256, cy))
    ix, iy, ax, ay = cx >> 8, cy >> 8, cx & 255, cy & 255
    cross = (ax * ay + 128) >> 8
    return ((iy, ix), (iy, min(ix + 1, W - 1)),
            (min(iy + 1, H - 1), ix), (min(iy + 1, H - 1), min(ix + 1, W - 1))), (
                256 - ax - ay + cross, ax - cross, ay - cross, cross)


def texture_native(pixels, weights, *, integer_shortcut=False):
    values = [half(v) for v in pixels]
    if integer_shortcut and weights == (256, 0, 0, 0):
        return values[0]
    weighted = f32(f32(f32(f32(values[0] * weights[0]) + f32(values[1] * weights[1]))
                           + f32(values[2] * weights[2])) + f32(values[3] * weights[3]))
    return half(f32(weighted * (1.0 / 256)))


def texture_reference(pixels, weights):
    """Measured 14-bit alignment and half ties-away value, CPU exact integer sum."""
    values = [half(v) for v in pixels]
    exponents = [math.frexp(abs(v))[1] - 1 for v, k in zip(values, weights) if v and k]
    exponent = max([-24] + exponents)
    summed = sum(math.trunc(math.ldexp(v, 14 - exponent)) * k for v, k in zip(values, weights))
    magnitude = abs(summed)
    if not magnitude:
        return -0.0 if all(k == 0 or math.copysign(1, v) < 0 for v, k in zip(values, weights)) else 0.0
    lead = magnitude.bit_length() - 1
    scale_exponent = exponent - 22
    result_exponent = lead + scale_exponent
    shift = max(result_exponent - 10, -24) - scale_exponent
    if shift > 0:
        quotient, remainder = divmod(magnitude, 1 << shift)
        rounded = quotient + (remainder >= 1 << (shift - 1))
    else:
        rounded = magnitude << -shift
    bits = min(0x7c00, max(0, result_exponent + 14) * 1024 + rounded)
    bits |= (summed < 0) << 15
    return struct.unpack('<e', struct.pack('<H', bits))[0]


def sample_pixel(image_at, x, y, mx, my, *, near=False, reference_values=False):
    bx, x0, gx, x3, middle_x = axis(f32(f32(x + 0.5) + half(mx)))
    by, y0, gy, y3, middle_y = axis(f32(f32(y + 0.5) + half(my)))
    weights = [f32(x0 * gy), f32(y0 * gx), f32(gx * gy), f32(y3 * gx), f32(x3 * gy)]
    coords = [(bx - 1, middle_y), (middle_x, by - 1), (middle_x, middle_y),
              (middle_x, by + 2), (bx + 2, middle_y)]
    taps = []
    for px, py in coords:
        indices, tex_weights = coefficients(px, py)
        values = []
        for ch in range(3):
            texels = [image_at(iy, ix, ch) for iy, ix in indices]
            values.append(texture_reference(texels, tex_weights) if reference_values else
                          texture_native(texels, tex_weights, integer_shortcut=not near))
        taps.append(values)
    total = weights[0]
    for value in weights[1:]:
        total = f32(total + value)
    numerator = []
    for ch in range(3):
        value = f32(taps[1][ch] * weights[1])
        for tap in (0, 2, 3, 4):
            value = f32(taps[tap][ch] * weights[tap] + value)
        numerator.append(value)
    return tuple(numerator), f32(1 / total), taps
