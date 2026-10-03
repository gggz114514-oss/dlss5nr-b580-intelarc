"""Untimed causal FFN interventions; never installed in captured production graphs.

CPU oracle substitutions deliberately trade all performance for isolation.
Unclipped hidden values are widened integers, not a deployable INT8 path.
Unrounded folded weights use an FP64 diagnostic sum, then FP32/half/outerFP8.
"""
import numpy as np
import torch
import nr_backend.vit_block as vit
from int8_ffn_body_scope_v1 import Int8VitLayout
import int8_ffn_segment_oracle_v1 as oracle


class ColorProbe(Int8VitLayout):
    def __init__(self,model,provider,hidden_scales):
        super().__init__(model,provider,hidden_scales)
        self.mode='int8';self.restore_block=None
        self.cpu_weights=[(b.expand.cpu().numpy().copy(),b.contract.cpu().numpy().copy(),
                          b.ffn_skip.cpu().numpy().copy(),np.asarray(sh).reshape(1,-1).copy())
                         for b,sh in zip(model.vit,hidden_scales)]

    def ffn(self,index,x):
        block=self.model.vit[index]
        if self.mode=='fp16_all' or (self.mode=='restore_block' and index==self.restore_block):
            hidden=vit.cubic_quantize(vit.dot(x,block.expand,chunk_k=16))
            mlp=vit.q(vit.split_k_projection(hidden,block.contract,(x*block.ffn_skip).half()))
            return mlp,None
        if self.mode in ('int8','restore_block'):return super().ffn(index,x)
        assert self.mode in ('cpu_oracle','unclipped_hidden','fp16_expand','restore_internal_fp8','unrounded_contract')
        a=x.cpu().numpy().copy();we,wc,skip,sh=self.cpu_weights[index]
        if self.mode=='fp16_expand':
            hidden=oracle.cubic_half(vit.dot(x,block.expand,chunk_k=16).cpu().numpy())
        else:hidden,_=oracle.expansion(a,we)
        if self.mode=='restore_internal_fp8':hidden=oracle.fp8_boundary(hidden)
        initial=(a.astype(np.float32)*skip.astype(np.float32)).astype(np.float16)
        if self.mode in ('unclipped_hidden','unrounded_contract'):
            folded=wc.astype(np.float32)*sh.reshape(-1,1)
            if self.mode=='unclipped_hidden':
                unit=hidden.astype(np.float32)/sh
                qwide=np.floor(np.abs(unit)+np.float32(.5))*np.where(unit<0,np.float32(-1),np.float32(1))
                qw,sw=oracle.quantize_axis(folded,0)
                total=qwide.astype(np.float64)@qw.astype(np.float64)
                assert np.isfinite(total).all() and np.abs(total).max()<2**53 and (total==np.rint(total)).all()
                value=total.astype(np.float32)*sw
            else:
                qh=oracle.quantize(hidden,sh)
                value=(qh.astype(np.float64)@folded.astype(np.float64)).astype(np.float32)
            mlp=oracle.fp8_boundary((value+initial.astype(np.float32)).astype(np.float16))
        else:_,mlp,_=oracle.contraction(hidden,sh,wc,initial)
        assert np.isfinite(mlp).all()
        return torch.from_numpy(mlp.copy()).to(x.device),None
