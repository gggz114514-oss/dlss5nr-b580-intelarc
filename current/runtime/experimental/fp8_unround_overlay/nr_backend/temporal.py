"""Stateful zero-motion NR correctness session, using its own prior output."""
from __future__ import annotations
from collections.abc import Mapping,Callable
from pathlib import Path
import torch
from torch import Tensor
from .executor import ResetNR
from .front import reset_front_features,zero_motion_front_features,MASK32
from .noise import NativeNoiseTable
from .sigmoid import NativeSigmoidTable
from .sampling import warp_history_square,warp_history_normalized
from .reciprocal import NativeReciprocalTable,NativeDimensionReciprocalTable
from .weights import load_pinned_records


class ZeroMotionNR(ResetNR):
    """One sequential SDR stream, zero motion/depth and default controls.

    Reset starts at seed zero. Each successful frame increments the uint32 seed
    and retains a private copy of its final RGB16F output. Failed calls do not
    advance history. Sessions are not thread-safe; use one per independent
    stream. Nonzero motion, arbitrary controls and real-time speed remain open.
    """
    PADDED_SIZES={(256,256):(320,320)}

    def __init__(self,records: Mapping[str,bytes],noise: NativeNoiseTable,sigmoid: NativeSigmoidTable):
        super().__init__(records,noise);self.sigmoid=sigmoid
        scale=records['block70.layer0.blend_scale']
        if len(scale)!=2:raise ValueError('Expected scalar half blend scale')
        self.register_buffer('blend_scale',torch.frombuffer(bytearray(scale),dtype=torch.float16).clone().reshape(()))
        self.register_buffer('_previous',None,persistent=False);self._next_seed=0

    @classmethod
    def from_assets(cls,weights: str|Path,noise_directory: str|Path,sigmoid_directory: str|Path):
        return cls(load_pinned_records(weights),NativeNoiseTable.from_directory(noise_directory),NativeSigmoidTable.from_directory(sigmoid_directory))

    def reset(self) -> None:
        self._previous=None;self._next_seed=0

    @property
    def next_seed(self) -> int:
        return self._next_seed

    @torch.inference_mode()
    def forward(self,rgb: Tensor,*,reset: bool=False,progress: Callable[[str],None]|None=None) -> Tensor:
        self._validate_rgb(rgb)
        previous=None if reset else self._previous
        seed=0 if previous is None else self._next_seed
        options=dict(padded_size=self.PADDED_SIZES[tuple(rgb.shape[:2])],seed=seed,noise_source=self.noise)
        front=reset_front_features(rgb,**options)if previous is None else zero_motion_front_features(rgb,previous,**options)
        result=self._forward_front(rgb,front,progress=progress,previous=previous,sigmoid=self.sigmoid,blend_scale=self.blend_scale)
        self._previous=result.detach().clone();self._next_seed=(seed+1)&MASK32
        return result


class IntegerMotionNR(ZeroMotionNR):
    """Bounded integer-pixel motion path; fractional motion is not accepted.

    Motion is current-to-previous displacement in pixels, HWC2, at the same
    resolution as RGB. The current validation covers constant (-1,0) motion
    and zero depth. It does not establish arbitrary spatially varying motion.
    """
    @torch.inference_mode()
    def forward(self,rgb: Tensor,motion: Tensor,*,reset: bool=False,progress: Callable[[str],None]|None=None) -> Tensor:
        self._validate_rgb(rgb)
        if motion.shape!=(*rgb.shape[:2],2)or motion.device!=rgb.device or not motion.is_floating_point():
            raise ValueError('Expected same-device HWC2 floating point pixel motion')
        if not bool(torch.isfinite(motion).all())or not bool((motion==motion.round()).all())or not bool((motion.abs()<=256).all()):
            raise ValueError('This path requires finite integer motion within +/-256 pixels')
        previous=None if reset else self._previous
        seed=0 if previous is None else self._next_seed
        options=dict(padded_size=self.PADDED_SIZES[tuple(rgb.shape[:2])],seed=seed,noise_source=self.noise)
        history=None
        if previous is not None:
            h,w=rgb.shape[:2];y,x=torch.meshgrid(torch.arange(h,device=rgb.device),torch.arange(w,device=rgb.device),indexing='ij')
            sx=(x+motion[...,0].long()).clamp(0,w-1);sy=(y+motion[...,1].long()).clamp(0,h-1)
            history=previous[sy,sx]
        front=reset_front_features(rgb,**options)if history is None else zero_motion_front_features(rgb,history,**options)
        result=self._forward_front(rgb,front,progress=progress,previous=history,sigmoid=self.sigmoid,blend_scale=self.blend_scale)
        self._previous=result.detach().clone();self._next_seed=(seed+1)&MASK32
        return result


class MotionNR(ZeroMotionNR):
    """Observed SDR dimensions, pixel motion, zero depth and default controls.

    Original half texture sampling and temporal overshoot are retained. The
    Constant and two spatially varying motion test sequences define the
    currently validated scope. Depth is zero. Other sizes need validation.
    """
    PADDED_SIZES={(256,256):(320,320),(512,512):(512,576),(480,864):(512,896),(1080,1920):(1152,1920),(1439,2559):(1472,2560)}
    def __init__(self,records: Mapping[str,bytes],noise: NativeNoiseTable,sigmoid: NativeSigmoidTable,reciprocal: NativeReciprocalTable,dimension_reciprocal: NativeDimensionReciprocalTable|None=None):
        super().__init__(records,noise,sigmoid);self.reciprocal=reciprocal;self.dimension_reciprocal=dimension_reciprocal

    @classmethod
    def from_assets(cls,weights: str|Path,noise_directory: str|Path,sigmoid_directory: str|Path,reciprocal_directory: str|Path|None=None,dimension_reciprocal_directory: str|Path|None=None):
        directory=Path(sigmoid_directory).parent/'reciprocal-sm89-v1'if reciprocal_directory is None else Path(reciprocal_directory)
        dimensions=Path(sigmoid_directory).parent/'reciprocal-dimensions-sm89-v1'if dimension_reciprocal_directory is None else Path(dimension_reciprocal_directory)
        return cls(load_pinned_records(weights),NativeNoiseTable.from_directory(noise_directory),NativeSigmoidTable.from_directory(sigmoid_directory),NativeReciprocalTable.from_directory(directory),NativeDimensionReciprocalTable.from_directory(dimensions))

    @torch.inference_mode()
    def forward(self,rgb: Tensor,motion: Tensor,*,reset: bool=False,progress: Callable[[str],None]|None=None) -> Tensor:
        self._validate_rgb(rgb)
        if motion.shape!=(*rgb.shape[:2],2)or motion.device!=rgb.device or not motion.is_floating_point():
            raise ValueError('Expected same-device RG floating-point pixel motion matching RGB dimensions')
        if not bool(torch.isfinite(motion).all())or not bool((motion.abs()<=65504).all()):
            raise ValueError('Motion must be finite and within the FP16 texture range +/-65504 pixels')
        previous=None if reset else self._previous;seed=0 if previous is None else self._next_seed
        if previous is not None and previous.shape!=rgb.shape:raise ValueError('Resolution change requires reset=True')
        options=dict(padded_size=self.PADDED_SIZES[tuple(rgb.shape[:2])],seed=seed,noise_source=self.noise)
        numerator=reciprocal=None
        if previous is None:front=reset_front_features(rgb,**options)
        else:
            if tuple(rgb.shape[:2])in((256,256),(512,512)):
                numerator,reciprocal=warp_history_square(previous,motion,return_components=True,reciprocal_source=self.reciprocal)
            else:
                numerator,reciprocal=warp_history_normalized(previous,motion,return_components=True,reciprocal_source=self.reciprocal,dimension_reciprocal=self.dimension_reciprocal)
            front=zero_motion_front_features(rgb,numerator*reciprocal[...,None],**options)
        result=self._forward_front(rgb,front,progress=progress,previous=numerator,history_reciprocal=reciprocal,sigmoid=self.sigmoid,blend_scale=self.blend_scale)
        self._previous=result.detach().clone();self._next_seed=(seed+1)&MASK32
        return result


class MotionNR256(MotionNR):
    """Compatibility API retaining the strict 256x256 input contract."""
    PADDED_SIZES={(256,256):(320,320)}
