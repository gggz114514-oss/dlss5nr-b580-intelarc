"""Independent reduced-NR residual prototype, using standard separable filters.

1080p source -> antialiased Lanczos2 -> aspect-preserving active rectangle inside
an already supported square NR input. Replicate-edge RGB padding; zero motion
outside the active rectangle. Motion uses area averaging and pixel-unit scaling.
Crop the signed half (NR-lowRGB) residual, Catmull-Rom upsample, add to the original
full-resolution RGB, then clamp SDR. The complete NR graph remains unchanged.

Inspired by the reduced-input residual design reviewed in Magpie; not a port or
bit-equivalence claim for its HLSL, controls, resource API or filter arithmetic.
"""
import math
import numpy as np
import torch
import triton
import triton.language as tl


def filter_table(source, destination, kind):
    if source <= 0 or destination <= 0 or kind not in ('lanczos2', 'area', 'catmull'):
        raise ValueError('Unsupported resampling table')
    ratio = source / destination
    rows = []
    for out in range(destination):
        center = (out + .5) * ratio - .5
        if kind == 'area':
            left, right = out * ratio, (out + 1) * ratio
            indices = list(range(math.floor(left), math.ceil(right)))
            weights = [max(0., min(right, i + 1.) - max(left, float(i))) for i in indices]
        elif kind == 'lanczos2':
            radius = 2 * max(1., ratio)
            indices = list(range(math.ceil(center - radius), math.floor(center + radius) + 1))
            t = (np.asarray(indices, dtype='f8') - center) / max(1., ratio)
            weights = np.where(np.abs(t) < 2, np.sinc(t) * np.sinc(t / 2), 0.).tolist()
        else:
            base = math.floor(center)
            indices = list(range(base - 1, base + 3))
            weights = []
            for i in indices:
                t = abs(center - i)
                weights.append((1.5*t - 2.5)*t*t + 1 if t <= 1 else ((-.5*t + 2.5)*t - 4)*t + 2 if t < 2 else 0.)
        total = sum(weights)
        if not abs(total) > 1e-12:
            raise ValueError('Empty reconstruction filter')
        rows.append((np.clip(indices, 0, source - 1), np.asarray(weights) / total))
    taps = max(len(indices) for indices, _ in rows)
    indices = np.zeros((destination, taps), dtype='i4')
    weights = np.zeros((destination, taps), dtype='f4')
    for row, (ii, ww) in enumerate(rows):
        indices[row, :len(ii)] = ii
        weights[row, :len(ww)] = ww
    return indices, weights


@triton.jit
def _axis(X, INDICES, WEIGHTS, ORIGINAL, OUT, HS:tl.constexpr, WS:tl.constexpr,
          HO:tl.constexpr, WO:tl.constexpr, C:tl.constexpr, VERTICAL:tl.constexpr,
          TAPS:tl.constexpr, COMPOSITE:tl.constexpr, BLOCK:tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < HO * WO * C
    c = offset % C
    x, y = (offset // C) % WO, offset // (C * WO)
    coordinate = y if VERTICAL else x
    value = tl.full((BLOCK,), 0., tl.float32)
    for tap in range(TAPS):
        index = tl.load(INDICES + coordinate * TAPS + tap, valid, other=0)
        weight = tl.load(WEIGHTS + coordinate * TAPS + tap, valid, other=0)
        source = (index * WS + x) * C + c if VERTICAL else (y * WS + index) * C + c
        value += tl.load(X + source, valid, other=0).to(tl.float32) * weight
    if COMPOSITE:
        value += tl.load(ORIGINAL + offset, valid, other=0).to(tl.float32)
        value = tl.minimum(1., tl.maximum(0., value))
    tl.store(OUT + offset, value, valid)


@triton.jit
def _pad(X, OUT, SIZE:tl.constexpr, ACTIVE_H:tl.constexpr, TOP:tl.constexpr,
         C:tl.constexpr, MOTION:tl.constexpr, SX:tl.constexpr, SY:tl.constexpr, BLOCK:tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < SIZE * SIZE * C
    c, x, y = i % C, (i // C) % SIZE, i // (C * SIZE)
    yy = tl.minimum(ACTIVE_H - 1, tl.maximum(0, y - TOP))
    value = tl.load(X + (yy * SIZE + x) * C + c, valid, other=0).to(tl.float32)
    if MOTION:
        value *= tl.where(c == 0, SX, SY)
        value = tl.where((y >= TOP) & (y < TOP + ACTIVE_H), value, 0.)
    tl.store(OUT + i, value, valid)


@triton.jit
def _residual(NR, INPUT, OUT, SIZE:tl.constexpr, ACTIVE_H:tl.constexpr,
              TOP:tl.constexpr, BLOCK:tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < ACTIVE_H * SIZE * 3
    source = i + TOP * SIZE * 3
    nr = tl.load(NR + source, valid, other=0).to(tl.float32)
    rgb = tl.load(INPUT + source, valid, other=0).to(tl.float32)
    tl.store(OUT + i, (nr - rgb).to(tl.float16), valid)


class ResidualScale:
    def __init__(self, size, *, device='xpu'):
        if size not in (256, 512):
            raise ValueError('Prototype uses only existing native-validated square size contracts')
        self.size, self.active_h = size, size * 9 // 16
        self.top = (size - self.active_h) // 2
        self.device = device
        self.tables = {}
        for kind, axis, src, dst in [('lanczos2', 'y', 1080, self.active_h), ('lanczos2', 'x', 1920, size),
                                    ('area', 'y', 1080, self.active_h), ('area', 'x', 1920, size),
                                    ('catmull', 'x', size, 1920), ('catmull', 'y', self.active_h, 1080)]:
            indices, weights = filter_table(src, dst, kind)
            self.tables[(kind, axis)] = torch.from_numpy(indices).to(device), torch.from_numpy(weights).to(device)

    def axis(self, x, kind, axis, *, dtype=torch.float16, original=None):
        indices, weights = self.tables[(kind, axis)]
        h, w, c = x.shape
        ho, wo = (indices.shape[0], w) if axis == 'y' else (h, indices.shape[0])
        out = torch.empty((ho, wo, c), dtype=dtype, device=x.device)
        _axis[(triton.cdiv(out.numel(), 256),)](x, indices, weights, x if original is None else original,
            out, h, w, ho, wo, c, axis == 'y', indices.shape[1], original is not None, 256,
            enable_fp_fusion=False)
        return out

    def prepare(self, rgb, motion):
        if rgb.shape != (1080, 1920, 3) or motion.shape != (1080, 1920, 2) or rgb.device != motion.device:
            raise ValueError('Expected same-device full1080 RGB and pixel motion')
        rgb, motion = rgb.contiguous(), motion.contiguous()
        color = self.axis(self.axis(rgb, 'lanczos2', 'y'), 'lanczos2', 'x')
        flow = self.axis(self.axis(motion, 'area', 'y', dtype=torch.float32), 'area', 'x', dtype=torch.float32)
        canvas = torch.empty((self.size, self.size, 3), dtype=torch.float32, device=rgb.device)
        low_motion = torch.empty((self.size, self.size, 2), dtype=torch.float16, device=rgb.device)
        for source, out, channels, is_motion in [(color, canvas, 3, False), (flow, low_motion, 2, True)]:
            _pad[(triton.cdiv(out.numel(), 256),)](source, out, self.size, self.active_h, self.top,
                channels, is_motion, self.size / 1920., self.active_h / 1080., 256, enable_fp_fusion=False)
        return canvas, low_motion

    def composite(self, original, canvas, nr):
        if nr.shape != canvas.shape or nr.shape != (self.size, self.size, 3):
            raise ValueError('Residual shape mismatch')
        residual = torch.empty((self.active_h, self.size, 3), dtype=torch.float16, device=original.device)
        _residual[(triton.cdiv(residual.numel(), 256),)](nr, canvas, residual, self.size, self.active_h, self.top, 256, enable_fp_fusion=False)
        horizontal = self.axis(residual, 'catmull', 'x')
        return self.axis(horizontal, 'catmull', 'y', dtype=torch.float32, original=original)

    def metadata(self):
        return dict(source=[1080, 1920], model_input=[self.size, self.size], active_rgb=[self.active_h, self.size],
                    padding_top=self.top, aspect_preserved=True, color_downsample='vertical then horizontal Lanczos2 AA, half between passes',
                    motion='separable area average, XY pixel-unit scaling, half output, zero padded motion',
                    residual='signed half (low NR - low RGB), crop active rectangle',
                    composite='Catmull-Rom horizontal half then vertical FP32, add original RGB, clamp SDR',
                    complete_original_nr_graph=True, original_native_output_equivalence=False)
