"""Scoped W8A8 fusion experiment; original graph, cache invalidation and K8 remain."""
from static_weight_cache_v2 import CachedFastMatrices
from fast_matrices_v3 import dot as ordinary
from fused_activation_int8_v1 import dot as fused

def geometry(k,n):
    # The 256x256 primitive regressed; start with the measured smaller region.
    if not (16<=k<=128 and k%16==0 and 1<=n<=128):return None
    if n<=32:return 32
    if k==32 and n==96:return 128
    return 64

class FusedCachedMatrices(CachedFastMatrices):
    def __init__(self):
        super().__init__();self.use_fused=False;self.shapes={}

    def select(self,mode):
        self.use_fused=mode=='int8_fused'
        super().select('int8_cached' if self.use_fused else mode)
        self.shapes={}

    def dense(self,a,w,*,chunk_k,initial=None,**kwargs):
        if not self.use_view_cache or chunk_k==8:
            return super().dense(a,w,chunk_k=chunk_k,initial=initial,**kwargs)
        if chunk_k!=16 or a.shape[-1]%16:
            raise ValueError('Unsupported original reduction size')
        packed=self.static_cache.lookup(w)
        self.record('packed_weight_hit' if packed is not None else 'packed_weight_miss')
        k,n=w.shape;bn=geometry(k,n)
        eligible=self.use_fused and packed is not None and bn is not None
        if eligible:out,compiled=fused(a,w,initial=initial,packed=packed,bn=bn)
        else:out,compiled=ordinary(a,w,initial=initial,int8=True,packed=packed)
        kind='int8_fused' if eligible else 'int8_separate'
        self.record('int8_dense');self.record(kind)
        shape=f'{a.numel()//k}x{k}x{n}:initial={initial is not None}:fused={eligible}:bn={bn if eligible else 32}'
        self.shapes[shape]=self.shapes.get(shape,0)+1
        if kind not in self.compiled:self.compiled[kind]=compiled
        return out
