"""Use measured exact K8 layouts for this model's owned pre/head weights.

The existing integer K8 kernel preserves every original rounding boundary.
Only uninitialized pre16x32 and post32x8 in FP16 fast mode are selected here.
Other modes and operands retain the original provider. This is experimental.

NR_K8_FP16_XMX (default on)
---------------------------
These two owned K8 points were the last places in the fast line still replicating
rounding boundaries this project has already given up: the docstring above is the
evidence, and `fast_matrices_v3.py` states it is "deliberately NOT bit-equivalent
to the NVIDIA accumulator rules". When on, they run on the already-authorised FP16
XMX path (`fast_matrices_v3._matmul`) instead of the exact integer K8 simulation.
Numerics move by one half ULP (max_abs = 2**-14); measured in-situ saving is
1.6075 ms directly attributed / 1.9078 ms on device busy time (`l3_1_insitu_v1`).
Set `NR_K8_FP16_XMX=off` to restore the exact K8 path verbatim.
"""
import os
import torch
import triton
from nr_backend.triton_math import _tiled_dot
from strided_batched_v2 import StridedMatrices

_ON={'1','on','true','yes'}
K8_FP16_XMX=os.environ.get('NR_K8_FP16_XMX','on').strip().lower() in _ON


def _fp16_xmx_dense(a,w):
    # Lazy import: only the cut arm needs it, and it keeps module load order unchanged.
    from fast_matrices_v3 import _matmul
    k,n=w.shape
    bk=16 if k%32 else 32
    bm,bn=16,32
    aa=a.half().contiguous().reshape(-1,k)
    ww=w.half().contiguous()
    m=aa.shape[0]
    out=torch.empty((*a.shape[:-1],n),dtype=torch.float16,device=a.device)
    # Same call shape as fast_matrices_v3.dot(): (A,W,SA,SW,INITIAL,OUT,M,N,K,has_init,batched,int8,bm,bn,bk).
    # BK=min(16,K) needs no zero padding -- _matmul carries its own `k<K` mask with other=0.
    _matmul[(triton.cdiv(m,bm),triton.cdiv(n,bn),1)](
        aa,ww,aa,ww,aa,out,m,n,k,False,False,False,bm,bn,bk,
        num_warps=4,enable_fp_fusion=False)
    return out


class K8TiledMatrices(StridedMatrices):
    def __init__(self,model):
        super().__init__()
        self.owned={id(model.pre.front_weight):(model.pre.front_weight,'pre',(2,32,1)),
                    id(model.post.head_weight):(model.post.head_weight,'post',(4,8,1))}
        if tuple(model.pre.front_weight.shape)!=(16,32) or tuple(model.post.head_weight.shape)!=(32,8):
            raise ValueError('Unexpected pinned K8 weights')
        self.k8_calls={}

    def dense(self,a,w,*,chunk_k,initial=None,**kwargs):
        entry=self.owned.get(id(w))
        if (entry is None or entry[0] is not w or chunk_k!=8 or initial is not None
                or self.mode!='fp16_xmx' or a.device.type!='xpu' or a.dtype!=torch.float16):
            return super().dense(a,w,chunk_k=chunk_k,initial=initial,**kwargs)
        _,name,(bm,bn,warps)=entry
        if K8_FP16_XMX:
            out=_fp16_xmx_dense(a,w)
            self.record('fp16_xmx_k8')
            self.k8_calls[name]=self.k8_calls.get(name,0)+1
            return out
        out=_tiled_dot(a,w,chunk_k=8,bm=bm,bn=bn,warps=warps)
        self.record('exact_dense_k8')
        self.k8_calls[name]=self.k8_calls.get(name,0)+1
        return out
