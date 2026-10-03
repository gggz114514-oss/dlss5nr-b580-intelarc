"""Reusable reset-frame NR correctness executor with observed size contracts."""
from __future__ import annotations
from collections.abc import Mapping,Callable
from pathlib import Path
import torch
from torch import Tensor,nn
from .front import reset_front_features
from .noise import NativeNoiseTable
from .pre_block import PreBlock
from .c32_block import C32SwinBlock
from .multihead_block import MultiHeadSwinBlock
from .split_block import SplitSwinBlock
from .vit_block import VitBlock
from .decoder import DecoderInputUpsample,DecoderUpsampleSwin
from .post import ResetPostBlock
from .weights import load_pinned_records

SHIFTS=((0,0),(4,4),(0,4),(4,0))


class ResetNR(nn.Module):
    """Original weights, portable scalar tables and native-reference arithmetic.

    Inputs and output stay on the selected torch device. This API intentionally
    exposes only observed sizes, SDR, reset and default-control contracts.
    No CUDA/NVIDIA process, reference feature cache or vendor Python is used.
    The integer correctness arithmetic is not a real-time implementation.
    """
    # Native captures show asymmetric padding at 512. Do not extrapolate a
    # padding formula or silently resize arbitrary inputs to these contracts.
    PADDED_SIZES={(256,256):(320,320),(512,512):(512,576),(480,864):(512,896),(1080,1920):(1152,1920),(1439,2559):(1472,2560)}
    def __init__(self,records: Mapping[str,bytes],noise: NativeNoiseTable):
        super().__init__();self.noise=noise
        def record(block,layer=0):return records[f'block{block}.layer{layer}.layer']
        self.pre=PreBlock(record(0));groups=[]
        for first,count,c in [(1,4,32),(5,4,64),(9,6,128),(15,8,256)]:
            group=[]
            for i in range(count):
                options=dict(window_shift=SHIFTS[i%4],downsample=i==count-1)
                group.append(C32SwinBlock(record(first+i),**options)if c==32 else MultiHeadSwinBlock(record(first+i),c,**options))
            groups.append(nn.ModuleList(group))
        self.encoder=nn.ModuleList(groups)
        self.encoder512=nn.ModuleList([SplitSwinBlock([record(23+i,j)for j in range(5 if i==7 else 4)],window_shift=SHIFTS[i%4])for i in range(8)])
        self.vit=nn.ModuleList([VitBlock([record(31+i,j)for j in range(5)])for i in range(8)])
        self.decoder_input=DecoderInputUpsample(record(39))
        self.decoder512=nn.ModuleList([SplitSwinBlock([record(40+i,j)for j in range(4)],window_shift=SHIFTS[i%4])for i in range(8)])
        groups=[]
        for first,count,c,shift_start in [(48,8,256,0),(56,6,128,2),(62,4,64,0),(66,4,32,0)]:
            group=[]
            for i in range(count):
                shift=SHIFTS[(i+shift_start)%4]
                group.append(DecoderUpsampleSwin(record(first+i),c,window_shift=shift)if i==0 else (C32SwinBlock(record(first+i),window_shift=shift)if c==32 else MultiHeadSwinBlock(record(first+i),c,window_shift=shift)))
            groups.append(nn.ModuleList(group))
        self.decoder=nn.ModuleList(groups);self.post=ResetPostBlock(record(70))

    @classmethod
    def from_assets(cls,weights: str|Path,noise_directory: str|Path):
        return cls(load_pinned_records(weights),NativeNoiseTable.from_directory(noise_directory))

    @torch.inference_mode()
    def forward(self,rgb: Tensor,*,seed: int=0,progress: Callable[[str],None]|None=None) -> Tensor:
        self._validate_rgb(rgb)
        front=reset_front_features(rgb,padded_size=self.PADDED_SIZES[tuple(rgb.shape[:2])],seed=seed,noise_source=self.noise)
        return self._forward_front(rgb,front,progress=progress)

    def _validate_rgb(self,rgb: Tensor) -> None:
        if rgb.ndim!=3 or rgb.shape[-1]!=3 or tuple(rgb.shape[:2])not in self.PADDED_SIZES or not rgb.is_floating_point():raise ValueError(f'Expected floating-point HWC RGB with size in {tuple(self.PADDED_SIZES)}')
        if rgb.device!=self.pre.front_weight.device:raise ValueError('Input and model must share a device')

    def _forward_front(self,rgb: Tensor,front: Tensor,*,progress=None,previous=None,sigmoid=None,blend_scale=None,history_reciprocal=None,return_float32: bool=False) -> Tensor:
        def mark(name):
            if rgb.device.type=='xpu':torch.xpu.synchronize(rgb.device)
            if progress is not None:progress(name)
        pre_skip,x=self.pre.forward_features_outputs(front);del front;skips=[];mark('pre')
        for c,group in zip((32,64,128,256),self.encoder):
            for block in group:
                skip,down=block.forward_outputs(x);x=down if down is not None else skip
            skips.append(skip);mark(f'encoder C{c}')
        for i,block in enumerate(self.encoder512):
            values=block.forward_boundaries(x);x=values[-1]
            if i==7:skip512=values[3]
        mark('encoder C512');vit_shape=x.shape;x=x.reshape(-1,1024)
        for block in self.vit:x=block(x)
        mark('ViT');x=self.decoder_input(x.reshape(vit_shape),skip512)
        for block in self.decoder512:x=block(x)
        mark('decoder C512')
        for c,group,skip in zip((256,128,64,32),self.decoder,reversed(skips)):
            x=group[0](x,skip)
            for block in group[1:]:x=block(x)
            mark(f'decoder C{c}')
        skips.clear();del skip,skip512,values
        result=self.post(x,pre_skip,rgb,previous=previous,sigmoid=sigmoid,blend_scale=blend_scale,history_reciprocal=history_reciprocal,return_float32=return_float32);mark('RGB');return result


class ResetNR256(ResetNR):
    """Compatibility API retaining the original strict 256x256 contract."""
    PADDED_SIZES={(256,256):(320,320)}
