"""Use measured exact K8 layouts for this model's owned pre/head weights.

The existing integer K8 kernel preserves every original rounding boundary.
Only uninitialized pre16x32 and post32x8 in FP16 fast mode are selected here.
Other modes and operands retain the original provider. This is experimental.
"""
import torch
from nr_backend.triton_math import _tiled_dot
from strided_batched_v2 import StridedMatrices


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
        out=_tiled_dot(a,w,chunk_k=8,bm=bm,bn=bn,warps=warps)
        self.record('exact_dense_k8')
        self.k8_calls[name]=self.k8_calls.get(name,0)+1
        return out
