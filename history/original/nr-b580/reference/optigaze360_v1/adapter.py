"""Isolated 480p -> 360p NR experiment with source-guided reconstruction.

The design question comes from OptiGazeScaler's reduced-resolution NR path;
this is an independent implementation, not a port of its GPL shader. It uses
the project's existing fast model and never changes its weights or arithmetic.
Geometry (360, 648) -> (384, 768) is derived, not 4060-observed, and must not
be described as an exact native NR contract.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


HIGH = (480, 864)
LOW = (360, 648)
PADDED = (384, 768)


def _resize360(x):
    return F.interpolate(x.permute(2, 0, 1).unsqueeze(0), size=LOW, mode="area").squeeze(0).permute(1, 2, 0).contiguous()


def prepare(rgb, motion):
    """Downsample color and current-to-previous pixel motion on the GPU."""
    if tuple(rgb.shape) != (*HIGH, 3) or tuple(motion.shape) != (*HIGH, 2):
        raise ValueError("Expected complete 480x864 RGB and motion frames")
    if rgb.device != motion.device or rgb.device.type != "xpu":
        raise ValueError("RGB and motion must be on the same XPU")
    if rgb.dtype != torch.float32 or motion.dtype != torch.float32:
        raise ValueError("Expected float32 source and pixel motion")
    return _resize360(rgb), (_resize360(motion) * 0.75).contiguous()


@triton.jit
def _guided(HIGH_RGB, BASE, EDIT, OUT, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = p < 480 * 864
    x = p % 864
    y = p // 864
    hx = tl.load(HIGH_RGB + p * 3, valid, 0).to(tl.float32)
    hy = tl.load(HIGH_RGB + p * 3 + 1, valid, 0).to(tl.float32)
    hz = tl.load(HIGH_RGB + p * 3 + 2, valid, 0).to(tl.float32)
    qx = x.to(tl.float32) * 0.75 - 0.125
    qy = y.to(tl.float32) * 0.75 - 0.125
    cx = (3 * x + 1) // 4
    cy = (3 * y + 1) // 4
    sw = tl.full((BLOCK,), 0.0, tl.float32)
    sx = tl.full((BLOCK,), 0.0, tl.float32)
    sy = tl.full((BLOCK,), 0.0, tl.float32)
    sz = tl.full((BLOCK,), 0.0, tl.float32)
    sd_x = tl.full((BLOCK,), 0.0, tl.float32)
    sd_y = tl.full((BLOCK,), 0.0, tl.float32)
    sd_z = tl.full((BLOCK,), 0.0, tl.float32)
    sxx = tl.full((BLOCK,), 0.0, tl.float32)
    syy = tl.full((BLOCK,), 0.0, tl.float32)
    szz = tl.full((BLOCK,), 0.0, tl.float32)
    sxd_x = tl.full((BLOCK,), 0.0, tl.float32)
    syd_y = tl.full((BLOCK,), 0.0, tl.float32)
    szd_z = tl.full((BLOCK,), 0.0, tl.float32)
    for oy in tl.static_range(-1, 2):
        for ox in tl.static_range(-1, 2):
            ux = tl.minimum(647, tl.maximum(0, cx + ox))
            uy = tl.minimum(359, tl.maximum(0, cy + oy))
            j = (uy * 648 + ux) * 3
            bx = tl.load(BASE + j, valid, 0).to(tl.float32)
            by = tl.load(BASE + j + 1, valid, 0).to(tl.float32)
            bz = tl.load(BASE + j + 2, valid, 0).to(tl.float32)
            dx = tl.load(EDIT + j, valid, 0).to(tl.float32)
            dy = tl.load(EDIT + j + 1, valid, 0).to(tl.float32)
            dz = tl.load(EDIT + j + 2, valid, 0).to(tl.float32)
            rx = (cx + ox).to(tl.float32) - qx
            ry = (cy + oy).to(tl.float32) - qy
            spatial = 1.0 / (1.0 + rx * rx + ry * ry)
            diff = (bx - hx) * (bx - hx) + (by - hy) * (by - hy) + (bz - hz) * (bz - hz)
            weight = spatial / (1.0 + 16.0 * diff)
            sw += weight
            sx += weight * bx
            sy += weight * by
            sz += weight * bz
            sd_x += weight * dx
            sd_y += weight * dy
            sd_z += weight * dz
            sxx += weight * bx * bx
            syy += weight * by * by
            szz += weight * bz * bz
            sxd_x += weight * bx * dx
            syd_y += weight * by * dy
            szd_z += weight * bz * dz
    inv = 1.0 / tl.maximum(sw, 1.0e-12)
    mx, my, mz = sx * inv, sy * inv, sz * inv
    mdx, mdy, mdz = sd_x * inv, sd_y * inv, sd_z * inv
    ax = tl.minimum(2.0, tl.maximum(-1.0, (sxd_x * inv - mx * mdx) / (sxx * inv - mx * mx + 0.002)))
    ay = tl.minimum(2.0, tl.maximum(-1.0, (syd_y * inv - my * mdy) / (syy * inv - my * my + 0.002)))
    az = tl.minimum(2.0, tl.maximum(-1.0, (szd_z * inv - mz * mdz) / (szz * inv - mz * mz + 0.002)))
    ox = tl.minimum(1.0, tl.maximum(0.0, hx + mdx + ax * (hx - mx)))
    oy = tl.minimum(1.0, tl.maximum(0.0, hy + mdy + ay * (hy - my)))
    oz = tl.minimum(1.0, tl.maximum(0.0, hz + mdz + az * (hz - mz)))
    tl.store(OUT + p * 3, ox, valid)
    tl.store(OUT + p * 3 + 1, oy, valid)
    tl.store(OUT + p * 3 + 2, oz, valid)


@triton.jit
def _residual(BASE, MODEL, OUT, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < 360 * 648 * 3
    delta = tl.load(MODEL + i, mask, 0).to(tl.float32) - tl.load(BASE + i, mask, 0)
    tl.store(OUT + i, delta, mask)


@triton.jit
def _temporal(BASE, CURRENT, FLOW, OLD_BASE, OLD_EDIT, OUT, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = p < 360 * 648
    x, y = p % 648, p // 648
    fx = tl.load(FLOW + p * 2, valid, 0)
    fy = tl.load(FLOW + p * 2 + 1, valid, 0)
    px = x.to(tl.float32) + fx
    py = y.to(tl.float32) + fy
    inside = (px >= 0.0) & (px <= 647.0) & (py >= 0.0) & (py <= 359.0) & (fx == fx) & (fy == fy)
    px = tl.minimum(647.0, tl.maximum(0.0, tl.where(inside, px, x.to(tl.float32))))
    py = tl.minimum(359.0, tl.maximum(0.0, tl.where(inside, py, y.to(tl.float32))))
    x0 = tl.floor(px).to(tl.int32)
    y0 = tl.floor(py).to(tl.int32)
    x1 = tl.minimum(x0 + 1, 647)
    y1 = tl.minimum(y0 + 1, 359)
    tx, ty = px - x0, py - y0
    a = (y0 * 648 + x0) * 3
    b = (y0 * 648 + x1) * 3
    c = (y1 * 648 + x0) * 3
    d = (y1 * 648 + x1) * 3
    for channel in tl.static_range(3):
        now = tl.load(CURRENT + p * 3 + channel, valid, 0)
        current_color = tl.load(BASE + p * 3 + channel, valid, 0)
        old = (1.0 - ty) * ((1.0 - tx) * tl.load(OLD_EDIT + a + channel, valid, 0)
                            + tx * tl.load(OLD_EDIT + b + channel, valid, 0)) + ty * (
                            (1.0 - tx) * tl.load(OLD_EDIT + c + channel, valid, 0)
                            + tx * tl.load(OLD_EDIT + d + channel, valid, 0))
        old_color = (1.0 - ty) * ((1.0 - tx) * tl.load(OLD_BASE + a + channel, valid, 0)
                                    + tx * tl.load(OLD_BASE + b + channel, valid, 0)) + ty * (
                                    (1.0 - tx) * tl.load(OLD_BASE + c + channel, valid, 0)
                                    + tx * tl.load(OLD_BASE + d + channel, valid, 0))
        lo = now
        hi = now
        for oy in tl.static_range(-1, 2):
            for ox in tl.static_range(-1, 2):
                xx = tl.minimum(647, tl.maximum(0, x + ox))
                yy = tl.minimum(359, tl.maximum(0, y + oy))
                sample = tl.load(CURRENT + (yy * 648 + xx) * 3 + channel, valid, 0)
                lo = tl.minimum(lo, sample)
                hi = tl.maximum(hi, sample)
        old = tl.minimum(hi, tl.maximum(lo, old))
        color_error = tl.abs(current_color - old_color)
        weight = tl.where(inside, 0.55 / (1.0 + 20.0 * color_error + 0.08 * (tl.abs(fx) + tl.abs(fy))), 0.0)
        value = now + weight * (old - now)
        tl.store(OUT + p * 3 + channel, value, valid)


def residual(base, model):
    if tuple(base.shape) != (*LOW, 3) or model.shape != base.shape:
        raise ValueError("Expected 360x648 model and baseline")
    out = torch.empty_like(base)
    _residual[(triton.cdiv(out.numel(), 256),)](base, model, out, 256, enable_fp_fusion=False)
    return out


def guided(rgb, baseline, edit):
    if tuple(rgb.shape) != (*HIGH, 3) or tuple(baseline.shape) != (*LOW, 3) or edit.shape != baseline.shape:
        raise ValueError("Mismatched source-guided reconstruction sizes")
    out = torch.empty_like(rgb)
    _guided[(triton.cdiv(HIGH[0] * HIGH[1], 128),)](rgb, baseline, edit, out, 128, enable_fp_fusion=False)
    return out


def temporal(baseline, edit, motion, previous_baseline, previous_edit):
    if any(tuple(x.shape) != (*LOW, c) for x, c in ((baseline, 3), (edit, 3), (motion, 2),
                                                    (previous_baseline, 3), (previous_edit, 3))):
        raise ValueError("Mismatched 360p temporal inputs")
    out = torch.empty_like(edit)
    _temporal[(triton.cdiv(LOW[0] * LOW[1], 128),)](baseline, edit, motion, previous_baseline,
                                                   previous_edit, out, 128, enable_fp_fusion=False)
    return out
