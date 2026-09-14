"""Unified fixed-control candidate with native style/intensity branch ordering."""
from __future__ import annotations
from dataclasses import dataclass
import math,struct
from pathlib import Path
import torch
from .temporal import MotionNR
from .style import NativeStylePost
from .sampling import fma32
from .post import store_half_rz,store_unit_half_rz
from .ui import NRUIInputs


@dataclass(frozen=True)
class NRControls:
    style: int=0
    intensity: float=1.0
    local_tone: float=1.0
    local_structure: float=1.0
    auto_mask: bool=False
    skin_structure: float|None=None

    def __post_init__(self):
        if type(self.style)is not int or self.style not in(0,1,2):raise ValueError('Expected style 0, 1 or 2')
        if type(self.auto_mask)is not bool:raise ValueError('auto_mask must be a bool')
        for name in('intensity','local_tone','local_structure','skin_structure'):
            value=getattr(self,name)
            if name=='skin_structure'and value is None:continue
            if type(value)not in(int,float)or not math.isfinite(value)or not 0<=value<=2:raise ValueError(f'{name} must be finite and in [0,2]')
            # The native feature parameters are FP32 before conversion to half.
            object.__setattr__(self,name,struct.unpack('<f',struct.pack('<f',value))[0])


class ControlledMotionNR(MotionNR):
    """Fixed controls over the core's observed SDR dimensions.

    Style0 low intensity mixes unrounded NR. Style1/2 grade private half NR and
    mix only at the end of grading. Every path keeps unblended private history.
    Controls are immutable for the lifetime of a session. Automatic masking is
    distinct from an explicit control-mask texture. Explicit masks are per-frame
    same-resolution HWC4 inputs: R scales display blending, G local tone, B local
    structure. Styled output applies R after grading. Adding or removing the
    binding preserves history. UI with explicit Alpha or UI.A is a final
    peripheral; Backbuffer-only selects the primary input. Grading never replaces
    private history. Validation coverage varies by dimension and control set;
    see the controlled-dimension reports. Wider UI/depth contracts remain open.
    """
    PADDED_SIZES=MotionNR.PADDED_SIZES.copy()

    def __init__(self,*args,controls=None,style_post=None,**kwargs):
        controls=NRControls()if controls is None else controls
        if not isinstance(controls,NRControls):raise TypeError('Expected NRControls')
        if controls.style and style_post is None:raise ValueError('Styled controls require the native style scalar assets')
        super().__init__(*args,**kwargs);self._controls=controls;self.style_post=style_post;self._raw_private=None
        self._control_mask=None;self._control_mask_bound=False

    @property
    def controls(self) -> NRControls:return self._controls

    @classmethod
    def from_assets(cls,weights,noise_directory,sigmoid_directory,reciprocal_directory=None,dimension_reciprocal_directory=None,*,controls=None,style_directory=None,enable_ui:bool=False):
        if type(enable_ui)is not bool:raise ValueError('enable_ui must be a bool')
        controls=NRControls()if controls is None else controls
        if not isinstance(controls,NRControls):raise TypeError('Expected NRControls')
        model=super().from_assets(weights,noise_directory,sigmoid_directory,reciprocal_directory,dimension_reciprocal_directory)
        if controls.style or enable_ui:
            directory=Path(sigmoid_directory).parent/'style-sm89-v1'if style_directory is None else Path(style_directory)
            model.style_post=NativeStylePost.from_directory(directory)
        model._controls=controls;return model

    def _forward_front(self,rgb,front,**kwargs):
        controls=self._controls
        front[...,10]=controls.style/128;front[...,11]=controls.local_tone
        # The native auto-mask branch moves structure strength to lanes13/14.
        front[...,12]=1 if controls.auto_mask else controls.local_structure
        skin=controls.local_structure if controls.skin_structure is None else controls.skin_structure
        front[...,13]=skin if controls.auto_mask else -1
        front[...,14]=controls.local_structure if controls.auto_mask else -1
        if self._control_mask is not None:
            h,w=self._control_mask.shape[:2];ph,pw=front.shape[:2]
            y=torch.arange(ph,device=front.device);x=torch.arange(pw,device=front.device)
            y=torch.where(y<h,y,2*h-2-y);x=torch.where(x<w,x,2*w-2-x)
            sampled=self._control_mask[y[:,None],x[None,:]].float()
            # Native multiplies sampled half G/B by FP32 strengths, then packs half.
            front[...,11]=sampled[...,1]*controls.local_tone
            front[...,12]=sampled[...,2]*controls.local_structure
            front[...,13:15]=-1
        if controls.style==0 and (controls.intensity<1 or self._control_mask is not None):
            kwargs['return_float32']=True;self._raw_private=super()._forward_front(rgb,front,**kwargs)
            return store_half_rz(self._raw_private)
        return super()._forward_front(rgb,front,**kwargs)

    @torch.inference_mode()
    def forward(self,rgb,motion,*,control_mask=None,ui:NRUIInputs|None=None,reset: bool=False,progress=None):
        if not bool((torch.isfinite(rgb)&(rgb>=0)&(rgb<=1)).all()):raise ValueError('Expected finite SDR RGB in [0,1]')
        if ui is not None:
            if not isinstance(ui,NRUIInputs):raise TypeError('Expected NRUIInputs')
            ui.validate(rgb)
            if not ui.has_alpha_source:
                # With Backbuffer bound but both UI/Alpha absent, native NR uses
                # Backbuffer as its primary color. It preserves the same private
                # history and skips the UI peripheral. Original native post
                # input mapping: live-ui-single-private-chain-v2 validation.
                rgb=ui.backbuffer
                ui=None
            else:
                if self.style_post is None:raise ValueError('UI requires preloaded peripheral assets: from_assets(enable_ui=True)')
                ui.prepare(rgb)
        bound=control_mask is not None
        if bound:
            if not isinstance(control_mask,torch.Tensor)or control_mask.shape!=(*rgb.shape[:2],4)or control_mask.device!=rgb.device or not control_mask.is_floating_point():
                raise ValueError('Expected same-device HWC4 floating point control mask matching RGB dimensions')
            if not bool((torch.isfinite(control_mask)&(control_mask>=0)&(control_mask<=1)).all()):raise ValueError('Control mask must be finite and in [0,1]')
        previous,seed,previous_bound=self._previous,self._next_seed,self._control_mask_bound
        self._control_mask=control_mask.half()if bound else None
        try:
            private=super().forward(rgb,motion,reset=reset,progress=progress);controls=self._controls
            if controls.style or ui is not None:
                result=self.style_post(private,rgb,style=controls.style,intensity=controls.intensity,local_tone=controls.local_tone,control_mask=self._control_mask,ui=ui)
                self._control_mask_bound=bound
                return result
            if bound:
                original=rgb.half().float()
                # Saturate the PRODUCT, not intensity alone: native post d160.
                amount=fma32(self._control_mask[...,:1].float(),controls.intensity,0.0).clamp(0,1)
                amount=torch.where(amount<torch.finfo(torch.float32).tiny,0.0,amount)
                result=store_unit_half_rz(fma32(self._raw_private-original,amount,original)+0.0)
                self._control_mask_bound=True
                return result
            self._control_mask_bound=False
            if controls.intensity>=1:return private
            original=rgb.half().float();return store_unit_half_rz(fma32(self._raw_private-original,controls.intensity,original))
        except Exception:
            self._previous,self._next_seed=previous,seed
            self._control_mask_bound=previous_bound
            raise
        finally:self._raw_private=None;self._control_mask=None
