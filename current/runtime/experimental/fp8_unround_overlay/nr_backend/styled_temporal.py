"""Fixed-style NR session with separate private history and visible output."""
from __future__ import annotations
from pathlib import Path
import torch
from .temporal import MotionNR
from .noise import NativeNoiseTable
from .sigmoid import NativeSigmoidTable
from .reciprocal import NativeReciprocalTable,NativeDimensionReciprocalTable
from .style import NativeStylePost
from .weights import load_pinned_records


class StyledMotionNR(MotionNR):
    """256x256, fixed style1/2 at creation, default strength/tone/structure.

    The model's ungraded half output is retained privately. Only the visible
    result passes through the style peripheral. Controls cannot change in-stream.
    Larger styled dimensions and combined controls require further validation.
    """
    PADDED_SIZES={(256,256):(320,320)}

    def __init__(self,records,noise,sigmoid,reciprocal,dimension_reciprocal,style_post,*,style: int):
        if type(style)is not int or style not in(1,2):raise ValueError('Only fixed styles 1 and 2 are supported')
        super().__init__(records,noise,sigmoid,reciprocal,dimension_reciprocal)
        self.style_post=style_post;self._style=style

    @property
    def style(self) -> int:return self._style

    @classmethod
    def from_assets(cls,weights,noise_directory,sigmoid_directory,reciprocal_directory=None,dimension_reciprocal_directory=None,*,style: int,style_directory=None):
        parent=Path(sigmoid_directory).parent
        reciprocal=parent/'reciprocal-sm89-v1'if reciprocal_directory is None else Path(reciprocal_directory)
        dimensions=parent/'reciprocal-dimensions-sm89-v1'if dimension_reciprocal_directory is None else Path(dimension_reciprocal_directory)
        grading=parent/'style-sm89-v1'if style_directory is None else Path(style_directory)
        return cls(load_pinned_records(weights),NativeNoiseTable.from_directory(noise_directory),NativeSigmoidTable.from_directory(sigmoid_directory),NativeReciprocalTable.from_directory(reciprocal),NativeDimensionReciprocalTable.from_directory(dimensions),NativeStylePost.from_directory(grading),style=style)

    def _forward_front(self,rgb,front,**kwargs):
        front[...,10]=self._style/128
        return super()._forward_front(rgb,front,**kwargs)

    @torch.inference_mode()
    def forward(self,rgb,motion,*,reset: bool=False,progress=None):
        if not bool((torch.isfinite(rgb)&(rgb>=0)&(rgb<=1)).all()):raise ValueError('Expected finite SDR RGB in [0,1]')
        previous,seed=self._previous,self._next_seed
        try:
            private=super().forward(rgb,motion,reset=reset,progress=progress)
            return self.style_post(private,rgb,style=self._style)
        except Exception:
            self._previous,self._next_seed=previous,seed
            raise
