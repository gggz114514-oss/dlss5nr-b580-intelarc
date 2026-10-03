"""Fixed SF-v2 style1/style2 peripheral on portable torch operations.

The scalar assets cover complete bounded numeric domains. They contain no image
or model activations. This module processes the ungraded private NR output; its
result must not replace the model's private temporal history.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from torch import Tensor,nn
from .sampling import fma32
from .post import store_half_rz
from .exposure import NativeExposureTable,float32
from .ui import NRUIInputs


class NativeStylePost(nn.Module):
    ASSETS={
        'curve-style1.rgb32f.bin':(184332,'8078b564d0838fd68aa6680f990f8e60a872f1e881c57130600d5666118c1a5c'),
        'curve-style2.rgb32f.bin':(184332,'7c373336c6d34c7d4343c365cd4b364b1c5a9be6c23d1e8a244996330d056b7d'),
        'reciprocal-mantissa.f32.bin':(33554432,'c408381446132e07fa5581e930dc33fb790bea7ec4276ef3621e322df2492a55'),
        'saturation-roundtrip.delta.i8.bin':(1056964609,'5e7195f445e0694d746a882c80b3ea28432f17b535fda0ec041eac3453913b7d'),
    }

    def __init__(self,curves: Tensor,reciprocal_bits: Tensor,roundtrip_delta: Tensor,exposure=None):
        super().__init__()
        if curves.dtype!=torch.float32 or curves.shape!=(2,0x3c01,3):raise ValueError('Expected complete RGB half-domain style curves')
        if reciprocal_bits.dtype!=torch.int32 or reciprocal_bits.shape!=(1<<23,):raise ValueError('Expected complete normalized reciprocal mantissa domain')
        if roundtrip_delta.dtype!=torch.int8 or roundtrip_delta.shape!=(0x3f000001,):raise ValueError('Expected complete saturation roundtrip domain')
        self.register_buffer('curves',curves);self.register_buffer('reciprocal_bits',reciprocal_bits);self.register_buffer('roundtrip_delta',roundtrip_delta)
        self.exposure=exposure

    @classmethod
    def from_directory(cls,path: str|Path):
        directory=Path(path)
        for name,(size,digest)in cls.ASSETS.items():
            source=directory/name
            with source.open('rb')as file:actual=hashlib.file_digest(file,'sha256').hexdigest()
            if source.stat().st_size!=size or actual!=digest:raise ValueError(f'Native style scalar asset mismatch: {name}')
        def mapped(name,dtype,count):return torch.from_file(str(directory/name),shared=False,size=count,dtype=dtype)
        curves=torch.stack([mapped(f'curve-style{style}.rgb32f.bin',torch.float32,0x3c01*3).reshape(0x3c01,3)for style in(1,2)])
        return cls(curves,mapped('reciprocal-mantissa.f32.bin',torch.int32,1<<23),mapped('saturation-roundtrip.delta.i8.bin',torch.int8,0x3f000001),NativeExposureTable(directory/NativeExposureTable.FILE))

    def _reciprocal(self,value: Tensor) -> Tensor:
        if not bool(((value>=2**-126)&(value<=2)).all()):raise ValueError('Style reciprocal outside the validated positive normal domain')
        bits=value.contiguous().view(torch.int32);exponent=((bits>>23)&255)-127
        result=self.reciprocal_bits[(bits&0x7fffff).long()]-(exponent<<23)
        return result.contiguous().view(torch.float32)

    def _saturation_roundtrip(self,value: Tensor) -> Tensor:
        if not bool(((value>=0)&(value<=1)).all()):raise ValueError('Style saturation outside the enumerated [0,1] domain')
        bits=value.contiguous().view(torch.int32);normal=bits>=0x800000;index=(bits-0x800000).clamp(min=0).long()
        result=bits+self.roundtrip_delta[index].int()
        return torch.where(normal,result,0).contiguous().view(torch.float32)

    def _saturated_roundtrip(self,value:Tensor) -> Tensor:
        if not bool(((value>=0)&(value<=2)).all()):raise ValueError('Saturated roundtrip outside the verified [0,2] domain')
        # Exhaustive original SM89 LG2/EX2 over all 8,388,609 FP32 [1,2] values
        # returns >=1. The following native SAT therefore yields exactly one.
        # Proof: reference/roundtrip-above-one-v1-validation.json, raw output
        # SHA256 ad4b114ed671acb9c2287bb32700834600a0253883e166a20a38984b12679eec.
        # This shortcut belongs to the saturated operation, not raw roundtrip.
        return self._saturation_roundtrip(value.clamp(max=1)).clamp(0,1)

    def _rgb_to_hsl(self,rgb: Tensor) -> tuple[Tensor,Tensor,Tensor]:
        r,g,b=rgb.unbind(-1)
        maximum=torch.maximum(b,torch.maximum(r,g));minimum=torch.minimum(b,torch.minimum(r,g))
        total=maximum+minimum;lightness=total*.5;active=maximum>minimum;difference=maximum-minimum
        denominator=torch.where(lightness>.5,(2-maximum)-minimum,total)
        inv_sum=self._reciprocal(torch.where(active,denominator,1.0));inv_difference=self._reciprocal(torch.where(active,difference,1.0))
        saturation=torch.where(active,difference*inv_sum,0.0)
        red_hue=fma32(g-b,inv_difference,torch.where(g>=b,0.0,6.0))*.16666667163372039795
        green_hue=fma32(b-r,inv_difference,2.0)*.16666667163372039795
        blue_hue=fma32(r-g,inv_difference,4.0)*.16666667163372039795
        hue=torch.where(active,torch.where(maximum==r,red_hue,torch.where(maximum==g,green_hue,blue_hue)),0.0)
        return hue,saturation,lightness

    @staticmethod
    def _hsl_to_rgb(hue: Tensor,saturation: Tensor,lightness: Tensor) -> Tensor:
        q=torch.where(lightness>=.5,fma32(-lightness,saturation,lightness+saturation),lightness*(saturation+1))
        p=(lightness+lightness)-q
        positions=torch.stack((hue+.3333333432674407959,hue,hue-.3333333432674407959),-1)
        positions=torch.where(positions>=0,positions,positions+1)
        positions=torch.where(positions>1,positions-1,positions)
        p=p[...,None];q=q[...,None];difference=q-p
        rising=fma32(difference*6,positions,p)
        falling=fma32(difference*(.6666666865348815918-positions),6.0,p)
        value=torch.where(positions<.16666667163372039795,rising,torch.where(positions<.5,q,torch.where(positions<.6666666865348815918,falling,p)))
        return torch.where((saturation>0)[...,None],value,lightness[...,None])

    @torch.inference_mode()
    def _tone_curve(self,network:Tensor,style:int,tone:float) -> Tensor:
        if style==1:
            if self.exposure is None:raise ValueError('Dynamic styled tone requires the complete native exposure asset')
            exposure=self.exposure(float32(float32(-.1)*tone));contrast=float32(-.25*tone)
        else:exposure=1.0;contrast=0.0
        value=fma32(network,exposure,0.0).clamp(0,1)
        delta=fma32(value*value,3-(value+value),-value)
        value=(fma32(delta,contrast,value)+0.0).clamp(0,1)
        # Captured gamma and five band weights are zero. The remaining native
        # transcendental path is two complete LG2/EX2 roundtrips, in this order.
        return self._saturation_roundtrip(self._saturation_roundtrip(value))

    @torch.inference_mode()
    def forward(self,neural: Tensor,original: Tensor,*,style: int,intensity: float=1.0,local_tone:float=1.0,control_mask:Tensor|None=None,ui:NRUIInputs|None=None,return_float32: bool=False) -> Tensor:
        if type(style)is not int or style not in(0,1,2):raise ValueError('Expected style 0, 1 or 2')
        if style==0 and ui is None:raise ValueError('Style0 uses this peripheral only when UI inputs are supplied')
        if type(intensity)not in(int,float)or not 0<=intensity<=2:raise ValueError('Style display intensity must be finite and in [0,2]')
        if type(local_tone)not in(int,float)or not 0<=local_tone<=2:raise ValueError('Style local tone must be finite and in [0,2]')
        tone=min(float32(local_tone),1.0)
        if neural.shape!=original.shape or neural.ndim!=3 or neural.shape[-1]!=3:raise ValueError('Expected matching HWC3 private neural and original RGB')
        if neural.device!=original.device or neural.device!=self.curves.device:raise ValueError('Style inputs and scalar assets must share a device')
        if not neural.is_floating_point()or not original.is_floating_point():raise ValueError('Expected floating point style inputs')
        if not bool(torch.isfinite(neural).all())or not bool((torch.isfinite(original)&(original>=0)&(original<=1)).all()):raise ValueError('Expected finite private neural output and SDR original RGB in [0,1]')
        amount=min(float32(intensity),1.0)
        if control_mask is not None:
            if not isinstance(control_mask,Tensor)or control_mask.shape!=(*neural.shape[:2],4)or control_mask.device!=neural.device or not control_mask.is_floating_point():
                raise ValueError('Expected same-device HWC4 control mask matching style inputs')
            if not bool((torch.isfinite(control_mask)&(control_mask>=0)&(control_mask<=1)).all()):raise ValueError('Control mask must be finite and in [0,1]')
            # cg2r 4e20 multiplies sampled half R by the unclipped FP32 intensity;
            # 6e90 saturates this product before the final graded-image blend.
            amount=(control_mask.half()[...,:1].float()*float32(intensity)).clamp(0,1)
            amount=torch.where(amount<torch.finfo(torch.float32).tiny,0.0,amount)
        additive_base=original.half().float()
        if ui is not None:
            if not isinstance(ui,NRUIInputs):raise TypeError('Expected NRUIInputs')
            additive_base,alpha=ui.prepare(original)
            # UI uses the unsaturated intensity/mask product until the final
            # native SAT, after scaling by (1-alpha), SASS4e20..4e40/6e90.
            unbounded=float32(intensity)if control_mask is None else control_mask.half()[...,:1].float()*float32(intensity)
            amount=(unbounded*(1-alpha[...,None])).clamp(0,1)
            amount=torch.where(amount<torch.finfo(torch.float32).tiny,0.0,amount)
        # FADD with +0 before SAT also canonicalizes the native negative zero.
        network=(neural.half().float()+0.0).clamp(0,1);original=original.half().float()
        indices=network.half().contiguous().view(torch.int16).long()
        channel=torch.arange(3,device=network.device)
        curve=self.curves[1][indices,channel]if style==0 else self.curves[style-1][indices,channel]if tone==1 else self._tone_curve(network,style,tone)
        hue,saturation,lightness=self._rgb_to_hsl(curve)
        factor=torch.tensor(float32(float32(0 if style==0 else -.1 if style==1 else -.15)*tone),device=network.device,dtype=torch.float32)+1
        saturation=fma32(saturation,factor,0.0).clamp(0,1)
        adjusted=self._hsl_to_rgb(hue,saturation,lightness)
        hue,saturation,lightness=self._rgb_to_hsl(adjusted)
        saturation=self._saturated_roundtrip(saturation)
        grade=(self._hsl_to_rgb(hue,saturation,lightness)+0.0).clamp(0,1)
        result=(fma32(grade-original,amount,additive_base)+0.0).clamp(0,1)
        return result if return_float32 else store_half_rz(result)
