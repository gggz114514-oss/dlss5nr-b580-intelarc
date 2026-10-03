"""Isolated 720p history-to-front candidate; not installed in any game.

Fuse the existing axis preparation and five-tap sampling into one Triton
dispatch. Preserve the current SM89 sampling arithmetic and return the two
values consumed by temporal.forward: unnormalized numerator and reciprocal.
The caller still builds normalized history for the existing fused front kernel.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import triton
import triton.language as tl

from fused import _half_texture, END_BITS, START_BITS


H, W = 720, 1280
BLOCK = 64


@triton.jit
def _axis_values(norm, extent: tl.constexpr, TABLE):
    center = tl.floor(tl.fma(norm, extent, -0.5)) + 0.5
    t = tl.minimum(1.0, tl.maximum(0.0, tl.fma(norm, extent, -center)))
    t2 = t * t
    t3 = t * t2
    w0 = tl.fma(t + t3, -0.5, t2)
    w1 = tl.fma(t3, 1.5, -(t2 * 2.5)) + 1.0
    w3 = (t3 - t2) * 0.5
    w2 = ((1.0 - w0) - w1) - w3
    group = w1 + w2
    index = group.to(tl.int32, bitcast=True) - 0x3F780000
    inverse = tl.load(TABLE + tl.minimum(0x1BFFFF, tl.maximum(0, index)))
    middle = tl.fma(w2, inverse, center)
    return center, w0, group, w3, middle


@triton.jit
def _compact_history(IMAGE, MOTION, IW, IH, TABLE, NUMERATOR, RECIPROCAL,
                     H: tl.constexpr, W: tl.constexpr, B: tl.constexpr):
    q = tl.program_id(0) * B + tl.arange(0, B)
    valid = q < H * W * 3
    p = q // 3
    ch = q % 3
    x = p % W
    y = p // W

    mx = tl.load(MOTION + p * 2, valid, 0).to(tl.float16).to(tl.float32)
    my = tl.load(MOTION + p * 2 + 1, valid, 0).to(tl.float16).to(tl.float32)
    iw = tl.load(IW)
    ih = tl.load(IH)
    u = tl.fma(mx, 1.0 / W, (x.to(tl.float32) + 0.5) * iw)
    v = tl.fma(my, 1.0 / H, (y.to(tl.float32) + 0.5) * ih)
    bx, x0, gx, x3, xmid = _axis_values(u, W, TABLE)
    by, y0, gy, y3, ymid = _axis_values(v, H, TABLE)
    k0, k1, k2, k3, k4 = x0 * gy, y0 * gx, gx * gy, y3 * gx, x3 * gy
    total = (((k0 + k1) + k2) + k3) + k4
    index = total.to(tl.int32, bitcast=True) - 0x3F780000
    inverse = tl.load(TABLE + tl.minimum(0x1BFFFF, tl.maximum(0, index)))

    # Preserve both normalized-coordinate round trips and the five-tap order.
    s1 = _half_texture(IMAGE, xmid, by - 1.0, ch, IW, IH, H, W)
    s0 = _half_texture(IMAGE, bx - 1.0, ymid, ch, IW, IH, H, W)
    s2 = _half_texture(IMAGE, xmid, ymid, ch, IW, IH, H, W)
    s3 = _half_texture(IMAGE, xmid, by + 2.0, ch, IW, IH, H, W)
    s4 = _half_texture(IMAGE, bx + 2.0, ymid, ch, IW, IH, H, W)
    value = s1 * k1
    value = tl.fma(s0, k0, value)
    value = tl.fma(s2, k2, value)
    value = tl.fma(s3, k3, value)
    value = tl.fma(s4, k4, value)
    tl.store(NUMERATOR + q, value, valid)
    tl.store(RECIPROCAL + p, inverse, valid & (ch == 0))


def _validate(image, motion, reciprocal_source, dimension_reciprocal):
    if (tuple(image.shape) != (H, W, 3) or image.dtype != torch.float16 or
            image.device.type != "xpu" or not image.is_contiguous()):
        raise ValueError("Compact candidate requires contiguous 720x1280 HWC3 FP16 XPU history")
    if (tuple(motion.shape) != (H, W, 2) or motion.dtype not in (torch.float16, torch.float32) or
            motion.device != image.device or not motion.is_contiguous()):
        raise ValueError("Compact candidate requires matching contiguous floating HWC2 motion")
    values = reciprocal_source.values
    if (values.dtype != torch.float32 or values.numel() != END_BITS - START_BITS or
            values.device != image.device or not values.is_contiguous()):
        raise ValueError("Invalid reciprocal table")
    iw, ih = dimension_reciprocal(W), dimension_reciprocal(H)
    if (iw.dtype != torch.float32 or ih.dtype != torch.float32 or
            iw.device != image.device or ih.device != image.device):
        raise ValueError("Invalid dimension reciprocals")
    return values, iw, ih


class CompactHistoryFront:
    """Drop-in history sampler for FullsizeGameModes' fused-history import.

    Only the fractional 720p HWC history route is replaced. The original
    near-integer full-frame shortcut remains intact; all other routes fall
    back unchanged. Graph replay increments Python counters at capture, not
    each replay, so a caller must also inspect replay status separately.
    """

    def __init__(self, original):
        self.original = original
        self.calls = {"compact": 0, "near_integer_reference": 0, "other_reference": 0}
        self.last_route = None

    def __call__(self, image, motion, *, return_components=False,
                 reciprocal_source, dimension_reciprocal):
        kw = dict(return_components=return_components,
                  reciprocal_source=reciprocal_source,
                  dimension_reciprocal=dimension_reciprocal)
        if not return_components or tuple(image.shape) != (H, W, 3):
            self.last_route = "other_reference"
            self.calls[self.last_route] += 1
            return self.original(image, motion, **kw)
        rounded = motion.half().float()
        if bool(((rounded - rounded.round()).abs() <= 1 / 256).all()):
            self.last_route = "near_integer_reference"
            self.calls[self.last_route] += 1
            from nr_backend import sampling
            return sampling.warp_history_normalized(image, motion, **kw)
        values, iw, ih = _validate(image, motion, reciprocal_source, dimension_reciprocal)
        numerator = torch.empty((H, W, 3), dtype=torch.float32, device=image.device)
        reciprocal = torch.empty((H, W), dtype=torch.float32, device=image.device)
        _compact_history[(triton.cdiv(H * W * 3, BLOCK),)](
            image, motion, iw, ih, values, numerator, reciprocal, H, W, BLOCK,
            num_warps=4, enable_fp_fusion=False)
        self.last_route = "compact"
        self.calls[self.last_route] += 1
        return numerator, reciprocal


@contextmanager
def installed():
    """Patch the module imported per frame by FullsizeGameModes; restore it."""
    import nr_game_history_fused as live

    original = live.warp_history_fused
    candidate = CompactHistoryFront(original)
    live.warp_history_fused = candidate
    try:
        yield candidate
    finally:
        changed = live.warp_history_fused is not candidate
        live.warp_history_fused = original
        if changed:
            raise RuntimeError("History candidate hook was changed during scope")
