"""Noise table lookup, reflection and HWC16 front construction in one kernel.

Full authenticated noise tables remain; seed is a runtime uint32, not a cache of
frame-specific noise. Temporal history warp is left to the original MotionNR.
"""
from contextlib import contextmanager
import torch
import triton
import triton.language as tl
import nr_backend.temporal as temporal
from nr_backend.execution import current_arithmetic_backend


@triton.jit
def _permute(v):
    return ((v >> ((v >> 28) + 4)) ^ v) * 0x108ef2d9


@triton.jit(do_not_specialize=['SEED'])
def _kernel(RGB, PREVIOUS, RADIUS, SINE, COSINE, OUT, SEED,
            H: tl.constexpr, W: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
            TEMPORAL: tl.constexpr, B: tl.constexpr):
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
    ra, rc = tl.load(RADIUS + a, valid, other=0), tl.load(RADIUS + c, valid, other=0)
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


def forward(rgb, *, padded_size, seed, noise_source, previous=None):
    h, w = rgb.shape[:2]
    ph, pw = padded_size
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not rgb.is_floating_point() or rgb.device.type != 'xpu' or not rgb.is_contiguous():
        raise ValueError('Expected contiguous floating XPU RGB')
    if not (h <= ph <= 2*h-1 and w <= pw <= 2*w-1 and type(seed) is int and 0 <= seed <= 0xffffffff):
        raise ValueError('Unsupported reflection or seed')
    if previous is not None and (previous.shape != rgb.shape or previous.device != rgb.device or not previous.is_contiguous()):
        raise ValueError('Invalid history')
    for name in ('radius', 'sine', 'cosine'):
        t = getattr(noise_source, name)
        if t.device != rgb.device or t.dtype != torch.float32 or t.shape != (1 << 24,) or not t.is_contiguous():
            raise ValueError('Invalid complete noise table')
    out = torch.empty((ph, pw, 16), dtype=torch.float16, device=rgb.device)
    _kernel[(triton.cdiv(ph*pw, 128),)](rgb, rgb if previous is None else previous,
        noise_source.radius, noise_source.sine, noise_source.cosine, out, seed, h, w, ph, pw,
        previous is not None, 128, num_warps=4, enable_fp_fusion=False)
    return out


class FusedFront:
    def __init__(self, model):
        self.noise = model.noise
        self.original_reset = temporal.reset_front_features
        self.original_temporal = temporal.zero_motion_front_features
        self.calls = 0

    def apply(self, rgb, previous=None, **options):
        if (options.get('noise_source') is not self.noise or current_arithmetic_backend() != 'triton'
                or rgb.device.type != 'xpu' or rgb.shape[-1] != 3 or not rgb.is_contiguous()
                or (previous is not None and not previous.is_contiguous())):
            return self.original_reset(rgb, **options) if previous is None else self.original_temporal(rgb, previous, **options)
        result = forward(rgb, previous=previous, **options)
        self.calls += 1
        return result

    @contextmanager
    def installed(self):
        assert temporal.reset_front_features is self.original_reset and temporal.zero_motion_front_features is self.original_temporal
        reset = lambda rgb, **kw: self.apply(rgb, **kw)
        history = lambda rgb, previous, **kw: self.apply(rgb, previous, **kw)
        temporal.reset_front_features, temporal.zero_motion_front_features = reset, history
        try:
            yield self
        finally:
            assert temporal.reset_front_features is reset and temporal.zero_motion_front_features is history
            temporal.reset_front_features, temporal.zero_motion_front_features = self.original_reset, self.original_temporal
