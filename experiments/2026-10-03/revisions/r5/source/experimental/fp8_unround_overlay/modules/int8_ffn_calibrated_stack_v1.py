"""Candidate static hidden-range repair; identical continuous INT8 GPU kernels.

Fit once from independently selected calibration activations. Repack the folded
contraction weights at session construction; no per-frame calibration, new GPU
passes, float hidden storage, or output color correction. Never a default.
"""
import numpy as np
from int8_ffn_nr_stack_v1 import Stack as OriginalStack


def fit_scales(original_scales, channel_maxima, margin=1.25):
    if margin != 1.25 or len(original_scales) != 8 or len(channel_maxima) != 8:
        raise ValueError('v1 requires eight channel maxima and fixed margin 1.25')
    result = []
    for old, maximum in zip(original_scales, channel_maxima):
        old = np.asarray(old, dtype=np.float32).reshape(1, 4096)
        maximum = np.asarray(maximum, dtype=np.float32).reshape(1, 4096)
        if not (np.isfinite(old).all() and (old > 0).all()
                and np.isfinite(maximum).all() and (maximum >= 0).all()):
            raise ValueError('Invalid calibration range')
        # Keep the original range/floor; widen only where observed data needs it.
        scale = np.maximum(old, maximum * np.float32(margin) / np.float32(127))
        result.append(np.ascontiguousarray(scale))
    return result


class Stack(OriginalStack):
    def metadata(self):
        return dict(super().metadata(),
                    quantization='continuous_ffn_int8_p4_multiframe_range_v1',
                    range_policy='max(original scale, observed channel absmax * 1.25 / 127)',
                    runtime_dynamic_hidden_calibration=False,
                    original_int8_kernel_code_unchanged=True,
                    default_promoted=False)
