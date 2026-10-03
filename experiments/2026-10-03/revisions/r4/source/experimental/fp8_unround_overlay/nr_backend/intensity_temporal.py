"""Low-intensity display mixing before the half store, with unblended history."""
from __future__ import annotations
import torch
from .temporal import MotionNR
from .sampling import fma32
from .post import store_half_rz,store_unit_half_rz


def _checked_intensity(value):
    if type(value)not in(int,float)or value not in(0,.5,1,2):raise ValueError('Only measured fixed intensities 0, 0.5, 1 and 2 are supported')
    return float(value)


class IntensityMotionNR(MotionNR):
    """256x256 fixed intensity, zero depth, default style/tone/structure.

    For intensity <1 the native display branch mixes the unrounded FP32 private
    value. Private history is always the unblended RGB16F result. The measured
    intensity 1/2 paths return the ordinary private output without a new clamp.
    """
    PADDED_SIZES={(256,256):(320,320)}

    def __init__(self,*args,intensity=1,**kwargs):
        super().__init__(*args,**kwargs);self._intensity=_checked_intensity(intensity);self._raw_private=None

    @property
    def intensity(self) -> float:return self._intensity

    @classmethod
    def from_assets(cls,*args,intensity=1,**kwargs):
        value=_checked_intensity(intensity);model=super().from_assets(*args,**kwargs);model._intensity=value;return model

    def _forward_front(self,rgb,front,**kwargs):
        if self._intensity>=1:return super()._forward_front(rgb,front,**kwargs)
        kwargs['return_float32']=True
        self._raw_private=super()._forward_front(rgb,front,**kwargs)
        return store_half_rz(self._raw_private)

    @torch.inference_mode()
    def forward(self,rgb,motion,*,reset: bool=False,progress=None):
        if not bool((torch.isfinite(rgb)&(rgb>=0)&(rgb<=1)).all()):raise ValueError('Expected finite SDR RGB in [0,1]')
        previous,seed=self._previous,self._next_seed
        try:
            private=super().forward(rgb,motion,reset=reset,progress=progress)
            if self._intensity>=1:return private
            original=rgb.half().float()
            return store_unit_half_rz(fma32(self._raw_private-original,self._intensity,original))
        except Exception:
            self._previous,self._next_seed=previous,seed
            raise
        finally:self._raw_private=None
