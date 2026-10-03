"""Game-only fusion of the validated five-tap history sampler.

The underlying Triton kernels come from the 648x360 video experiment. This
wrapper limits them to the game's model canvases and retains the native
single-tap shortcut for near-integer motion. It never changes backend source.
"""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
FUSED = ROOT / "game"
GAME_MODEL_SHAPES = {(360, 640), (480, 864), (544, 960), (720, 1280)}


def warp_history_fused(image, motion, *, return_components=False,
                       reciprocal_source, dimension_reciprocal):
    if not return_components or tuple(image.shape[:2]) not in GAME_MODEL_SHAPES:
        raise ValueError("Fused game history requires a validated model canvas")
    from nr_backend import sampling

    # The reference has a whole-frame near-integer shortcut. The fused kernel
    # always executes five taps and is not byte-identical on those frames.
    rounded_motion = motion.half().float()
    if bool(((rounded_motion - rounded_motion.round()).abs() <= 1 / 256).all()):
        return sampling.warp_history_normalized(
            image, motion, return_components=True,
            reciprocal_source=reciprocal_source,
            dimension_reciprocal=dimension_reciprocal)
    if str(FUSED) not in sys.path:
        sys.path.insert(0, str(FUSED))
    import fused

    h, w = image.shape[:2]
    iw, ih = dimension_reciprocal(w), dimension_reciprocal(h)
    axes, _ = fused.prepare_axes(motion, reciprocal_source.values, iw, ih)
    numerator, reciprocal, _, _, _ = fused.sample_five(
        image, axes, reciprocal_source.values, iw, ih)
    return numerator, reciprocal
