"""Candidate complete NR for three captured small input sizes.

Native evidence still has a 320-square body and 64 ViT tokens. The original
pre kernel reflects once; negative reflected coordinates reach clamp samplers.
History must be sampled at the reflected coordinate, before clamp addressing,
rather than clamping a precomputed in-image history warp. Exact parity is gated
by the independent native captures, not by these proposed formulas alone.
"""
import torch

from nr_backend.temporal import MotionNR
from nr_backend.front import MASK32
from nr_backend.sampling import fma32, _axis_weights_fraction, _sample_five_axes


class SmallMotionNR(MotionNR):
    PADDED_SIZES = {(128, 128): (320, 320), (192, 192): (320, 320), (144, 256): (320, 320)}

    @torch.inference_mode()
    def forward(self, rgb, motion, *, reset=False, progress=None):
        self._validate_rgb(rgb)
        h, w = rgb.shape[:2]
        if motion.shape != (h, w, 2) or motion.device != rgb.device or not motion.is_floating_point():
            raise ValueError('Expected matching floating RGB and pixel motion')
        if not bool(torch.isfinite(rgb).all()) or not bool(torch.isfinite(motion).all()) or not bool((motion.abs() <= 65504).all()):
            raise ValueError('Finite RGB and bounded finite motion required')
        previous = None if reset else self._previous
        if previous is not None and previous.shape != rgb.shape:
            raise ValueError('Resolution change requires reset=True')
        seed = 0 if previous is None else self._next_seed
        ph, pw = self.PADDED_SIZES[(h, w)]
        y, x = torch.meshgrid(torch.arange(ph, device=rgb.device), torch.arange(pw, device=rgb.device), indexing='ij')
        sx, sy = torch.where(x < w, x, 2 * w - 2 - x), torch.where(y < h, y, 2 * h - 2 - y)
        cx, cy = sx.clamp(0, w - 1), sy.clamp(0, h - 1)
        current = rgb.half()[cy, cx]
        scaled = ((current - .5).half() * .125).half()
        front = torch.zeros((ph, pw, 16), device=rgb.device, dtype=torch.float16)
        front[..., :3] = self.noise(x, y, seed)
        front[..., 3] = 1
        front[..., 4:7] = scaled
        front[..., 7:10] = scaled
        front[..., 11:13] = 1
        front[..., 13:15] = -1
        numerator = reciprocal = None
        if previous is not None:
            iw, ih = self.dimension_reciprocal(w), self.dimension_reciprocal(h)
            mv = motion.half().float()[cy, cx]
            u = fma32(mv[..., 0], 1 / w, (sx.float() + .5) * iw)
            v = fma32(mv[..., 1], 1 / h, (sy.float() + .5) * ih)

            def axis(norm, extent):
                center = fma32(norm, extent, -.5).floor() + .5
                return _axis_weights_fraction(center, fma32(norm, extent, -center).clamp(0, 1), self.reciprocal)

            padded_numerator, padded_reciprocal = _sample_five_axes(
                previous, axis(u, w), axis(v, h), return_components=True,
                reciprocal_source=self.reciprocal, normalized_scale=(iw, ih))
            sample = (padded_numerator * padded_reciprocal[..., None]).half()
            front[..., 7:10] = ((sample - .5).half() * .125).half()
            numerator = padded_numerator[:h, :w].contiguous()
            reciprocal = padded_reciprocal[:h, :w].contiguous()
        result = self._forward_front(rgb, front, progress=progress, previous=numerator,
                                     history_reciprocal=reciprocal, sigmoid=self.sigmoid, blend_scale=self.blend_scale)
        self._previous = result.detach().clone()
        self._next_seed = (seed + 1) & MASK32
        return result
