"""Measured FP16 dense tiling for owned model buffers in the current NR256 stack.

BK32 and the original arithmetic kernel stay fixed. Only observed geometries,
layouts and registered full-weight buffers are selected; all other calls retain
K8TiledMatrices behavior. No runtime tuning, quantization or weight packing.
"""
from types import MappingProxyType
import torch
from k8_tiled_provider_v1 import K8TiledMatrices
from dense_tiles_v1 import dot

# Fixed from current-dense-tiles-v1; complete output comparisons for all8 tiles
# at34 observed geometries. The full model remains the final promotion gate.
POLICY=MappingProxyType({
    (256, 512, 1536, False, (True, True, True)): (64, 64, 32, 4),
    (64, 1024, 4096, False, (True, True, True)): (64, 32, 32, 4),
    (576, 256, 768, False, (True, True, True)): (64, 32, 32, 4),
    (1920, 128, 384, False, (True, True, True)): (32, 64, 32, 4),
    (7040, 64, 192, False, (True, True, True)): (64, 32, 32, 4),
    (2304, 128, 384, False, (True, True, True)): (64, 128, 32, 4),
    (1600, 128, 384, False, (True, True, True)): (32, 128, 32, 4),
    (7744, 64, 192, False, (True, True, True)): (64, 128, 32, 4),
    (1920, 128, 128, True, (True, True, True)): (32, 64, 32, 4),
    (6400, 64, 192, False, (True, True, True)): (32, 64, 32, 4),
    (7040, 64, 64, True, (True, True, True)): (64, 64, 32, 4),
    (2304, 128, 128, True, (True, True, True)): (32, 64, 32, 4),
    (26880, 32, 32, True, (True, True, True)): (32, 32, 32, 4),
    (7744, 64, 64, True, (True, True, True)): (64, 64, 32, 4),
    (6400, 64, 64, True, (True, True, True)): (64, 64, 32, 4),
    (6400, 32, 64, False, (True, True, True)): (64, 32, 32, 4),
    (1600, 64, 128, False, (True, True, True)): (64, 32, 32, 4),
})

def weight_key(t):
    return (t.device,t.dtype,t.data_ptr(),tuple(t.shape),tuple(t.stride()))

class CurrentDenseTiledMatrices(K8TiledMatrices):
    def __init__(self,model):
        super().__init__(model)
        self.dense_weights={weight_key(w):w for _,w in model.named_buffers() if w.ndim==2 and w.dtype==torch.float16}
        self.tiled_calls={}

    def dense(self,a,w,*,chunk_k,initial=None,**kwargs):
        if (self.mode!='fp16_xmx' or chunk_k!=16 or a.device.type!='xpu' or a.dtype!=torch.float16
                or w.ndim!=2 or w.dtype!=torch.float16 or a.ndim<2
                or (initial is not None and initial.dtype!=torch.float16)):
            return super().dense(a,w,chunk_k=chunk_k,initial=initial,**kwargs)
        k,n=w.shape;m=a.numel()//k
        layout=tuple(t is None or t.is_contiguous() for t in (a,w,initial))
        tile=POLICY.get((m,k,n,initial is not None,layout))
        if tile is None or weight_key(w) not in self.dense_weights:
            return super().dense(a,w,chunk_k=chunk_k,initial=initial,**kwargs)
        result,kernel=dot(a,w,initial=initial,tile=tile)
        self.record('fp16_dense');self.record('fp16_tiled')
        label=f'{m}x{k}x{n}:initial={initial is not None}:contiguous={layout}'
        self.tiled_calls[label]=self.tiled_calls.get(label,0)+1
        self.compiled.setdefault('fp16_tiled',kernel)
        return result
