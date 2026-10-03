"""Stdlib-only independent CPU arithmetic/index checks for the owned profile.

These are mathematical oracles and interface planners, not GPU emulation or
4060 truth acceptance gates. They never import the runtime/kernel modules.
"""
import math
import struct


def f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def half(x):
    try:
        return struct.unpack("<e", struct.pack("<e", float(x)))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def merge_reference(projected, skip, scale):
    # Binary16 products and addition are represented exactly in binary64.
    return half(f32(skip * scale + projected))


def full_k_reference(x, w):
    if len(x) != 1024 or len(w) != 1024:
        raise ValueError("DecoderInput requires complete K1024")
    total = 0.0
    for a, b in zip(x, w):
        total = f32(total + a * b)
    return half(total)


def entry_source(row, lane):
    if not 0 <= row < 776 * 1288 or not 0 <= lane < 32:
        raise ValueError("Outside current720 padded post canvas")
    y, x = row // 1288 - 4, row % 1288 - 4
    if not (0 <= y < 768 and 0 <= x < 1280):
        return None
    return (((y // 2) * 640 + x // 2) * 32 + lane,
            (y * 1280 + x) * 32 + lane)


def average_reference(a, b, c, d):
    return half(f32(f32(f32(a + b) + f32(c + d)) * .25))


def sigmoid_native_reference(logit):
    """Independent scalar F32 rounding oracle for current native sigmoid.

    CPU exp is mathematical, not an emulation of the device tl.exp lowering.
    """
    x = half(logit)
    z = f32(math.exp(-abs(x)))
    denominator = f32(1.0 + z)
    return f32(1.0 / denominator if x >= 0.0 else z / denominator)


def tail_reference(head, rgb, *, previous=None, reciprocal=None, blend_scale=1,
                   return_float32=False, post_sigmoid="native", sigmoid_values=None):
    if len(head) < 4 or len(rgb) != 3:
        raise ValueError("Post requires head4/RGB3")
    result = []
    if post_sigmoid not in ("native", "table"):
        raise ValueError("Post sigmoid must be current native or table")
    if previous is None:
        alpha = 0.0
    elif post_sigmoid == "native":
        alpha = sigmoid_native_reference(head[3])
    else:
        if sigmoid_values is None or len(sigmoid_values) != 65536:
            raise ValueError("Table mode requires the complete half domain")
        bits = struct.unpack('<H', struct.pack('<e',head[3]))[0]
        alpha = sigmoid_values[bits]
    alpha = min(1., max(0., f32(alpha * blend_scale)))
    for i in range(3):
        texture = f32(f32(half(rgb[i]) * .125) - .0625)
        base = f32(f32(f32(head[i] * .03125) + texture) * 8)
        base = f32(base + .5)
        base = min(1., max(0., base))
        if previous is not None:
            delta = f32(previous[i] - base) if reciprocal is None else f32(previous[i] * reciprocal - base)
            base = f32(delta * alpha + base)
        result.append(base if return_float32 else half(base))
    return tuple(result)


def rgb_hsl(rgb):
    r, g, b = rgb
    hi, lo = max(b, max(r, g)), min(b, min(r, g))
    light = (hi + lo) * .5
    if hi == lo:
        return 0., 0., light
    diff = hi - lo
    sat = diff / (((2 - hi) - lo) if light > .5 else hi + lo)
    hue = ((g - b) / diff + (0 if g >= b else 6)) if hi == r else (((b - r) / diff + 2) if hi == g else ((r - g) / diff + 4))
    return hue / 6, sat, light


def hsl_rgb(hsl):
    hue, sat, light = hsl
    if sat == 0:
        return (light,) * 3
    q = light + sat - light * sat if light >= .5 else light * (sat + 1)
    p = light * 2 - q
    def channel(pos):
        if pos < 0: pos += 1
        if pos > 1: pos -= 1
        if pos < 1 / 6: return p + (q - p) * 6 * pos
        if pos < .5: return q
        if pos < 2 / 3: return p + (q - p) * (2 / 3 - pos) * 6
        return p
    return tuple(channel(hue + v) for v in (1 / 3, 0, -1 / 3))


def control_lanes(controls):
    skin = controls.local_structure if controls.skin_structure is None else controls.skin_structure
    return (controls.style / 128, controls.local_tone,
            1 if controls.auto_mask else controls.local_structure,
            skin if controls.auto_mask else -1,
            controls.local_structure if controls.auto_mask else -1)
