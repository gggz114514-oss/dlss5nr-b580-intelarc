"""Plan 10: one dynamic front kernel, only noise table arithmetic changes.

The three complete table pointers remain owned/allocated, including unused
pointers. Native trig is a lossy direct radians substitution, not reference
cycle emulation. CPU AST only; no compiled/resource/quality claim.
"""
import triton
import triton.language as tl


@triton.jit
def _permute(v):
    return ((v >> ((v >> 28) + 4)) ^ v) * 0x108ef2d9


@triton.jit
def _uniform(index):
    # The maximum 24-bit index maps exactly to 1; zero maps to 2**-24.
    return (index + 1).to(tl.float32) * 0.000000059604644775390625


@triton.jit
def _radius(index):
    uniform = _uniform(index)
    return tl.sqrt((tl.log2(uniform) * 0.69314718246459960938) * -2.0)


@triton.jit(do_not_specialize=["SEED"])
def front(RGB, PREVIOUS, RADIUS, SINE, COSINE, OUT, SEED,
          H: tl.constexpr, W: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
          TEMPORAL: tl.constexpr, NATIVE_RADIUS: tl.constexpr,
          NATIVE_TRIG: tl.constexpr, B: tl.constexpr):
    pixel = tl.program_id(0) * B + tl.arange(0, B)
    valid = pixel < PH * PW
    x, y = (pixel % PW).to(tl.uint32), (pixel // PW).to(tl.uint32)
    z = (x * 0x8da6b343) ^ (y * 0xd8163841) ^ (SEED.to(tl.uint32) * 0x9e3779b9) ^ 0x243f6a88
    z = _permute(z)
    z = z ^ (z >> 22)
    a = _permute(z * 0xcaa5b80d + 0x21dd796b)
    b = _permute(z * 0x83232c31 + 0x3463e0ac)
    c = _permute(z * 0x2c9277b5 + 0xac564b05)
    d = _permute(z * 0xfa6dc5f9 + 0x4712a88e)
    a, b, c, d = (a >> 30) ^ (a >> 8), (b >> 30) ^ (b >> 8), (c >> 30) ^ (c >> 8), (d >> 30) ^ (d >> 8)
    if NATIVE_RADIUS:
        ra, rc = _radius(a), _radius(c)
    else:
        ra, rc = tl.load(RADIUS + a, valid, other=0), tl.load(RADIUS + c, valid, other=0)
    if NATIVE_TRIG:
        theta_b = _uniform(b) * 6.283185307179586
        theta_d = _uniform(d) * 6.283185307179586
        cosd, sind, cosb = tl.cos(theta_d), tl.sin(theta_d), tl.cos(theta_b)
    else:
        cosd, sind, cosb = tl.load(COSINE + d, valid, other=0), tl.load(SINE + d, valid, other=0), tl.load(COSINE + b, valid, other=0)
    n0, n1, n2 = (rc * cosd).to(tl.float16), (rc * sind).to(tl.float16), (ra * cosb).to(tl.float16)
    sx, sy = tl.where(x < W, x, 2 * W - 2 - x), tl.where(y < H, y, 2 * H - 2 - y)
    offset = (sy * W + sx) * 3
    r = tl.load(RGB + offset, valid, other=0).to(tl.float16)
    g = tl.load(RGB + offset + 1, valid, other=0).to(tl.float16)
    b = tl.load(RGB + offset + 2, valid, other=0).to(tl.float16)
    r = ((r.to(tl.float32) - 0.5).to(tl.float16).to(tl.float32) * 0.125).to(tl.float16)
    g = ((g.to(tl.float32) - 0.5).to(tl.float16).to(tl.float32) * 0.125).to(tl.float16)
    b = ((b.to(tl.float32) - 0.5).to(tl.float16).to(tl.float32) * 0.125).to(tl.float16)
    hr, hg, hb = r, g, b
    if TEMPORAL:
        hr = tl.load(PREVIOUS + offset, valid, other=0).to(tl.float16)
        hg = tl.load(PREVIOUS + offset + 1, valid, other=0).to(tl.float16)
        hb = tl.load(PREVIOUS + offset + 2, valid, other=0).to(tl.float16)
        hr = ((hr.to(tl.float32) - 0.5).to(tl.float16).to(tl.float32) * 0.125).to(tl.float16)
        hg = ((hg.to(tl.float32) - 0.5).to(tl.float16).to(tl.float32) * 0.125).to(tl.float16)
        hb = ((hb.to(tl.float32) - 0.5).to(tl.float16).to(tl.float32) * 0.125).to(tl.float16)
    channel = tl.arange(0, 16)
    value = tl.full((B, 16), 0., tl.float16)
    value = tl.where(channel[None, :] == 0, n0[:, None], value)
    value = tl.where(channel[None, :] == 1, n1[:, None], value)
    value = tl.where(channel[None, :] == 2, n2[:, None], value)
    value = tl.where((channel[None, :] == 3) | (channel[None, :] == 11) | (channel[None, :] == 12), 1., value)
    value = tl.where((channel[None, :] == 13) | (channel[None, :] == 14), -1., value)
    value = tl.where(channel[None, :] == 4, r[:, None], value)
    value = tl.where(channel[None, :] == 5, g[:, None], value)
    value = tl.where(channel[None, :] == 6, b[:, None], value)
    value = tl.where(channel[None, :] == 7, hr[:, None], value)
    value = tl.where(channel[None, :] == 8, hg[:, None], value)
    value = tl.where(channel[None, :] == 9, hb[:, None], value)
    tl.store(OUT + pixel[:, None] * 16 + channel[None, :], value, valid[:, None])
