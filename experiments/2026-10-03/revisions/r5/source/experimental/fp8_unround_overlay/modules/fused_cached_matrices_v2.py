"""Extend fusion only to additional small-M geometries measured by shapes-v2."""
from fused_cached_matrices_v1 import FusedCachedMatrices as Base,geometry as previous_geometry
from fast_matrices_v3 import dot as ordinary
from fused_activation_int8_v1 import dot as fused

def geometry(m,k,n):
    previous=previous_geometry(k,n)
    if previous is not None:return previous
    if (k,n)==(64,256) and m<=512:return 64
    if (k,n)==(256,64) and m<=512:return 32
    if (k,n)==(32,256) and m<=2560:return 64
    if (k,n)==(256,128) and m<=2560:return 32
    if (k,n)==(256,512) and m<=128:return 64
    if (k,n)==(64,192) and m<=32768:return 64
    return None

class FusedCachedMatrices(Base):
    def __init__(self):super().__init__();self.expanded=False

    def select(self,mode):
        self.expanded=mode=='int8_fused_v2'
        super().select('int8_fused' if self.expanded else mode)

    def dense(self,a,w,*,chunk_k,initial=None,**kwargs):
        if not self.expanded or chunk_k==8:
            return super().dense(a,w,chunk_k=chunk_k,initial=initial,**kwargs)
        if chunk_k!=16 or a.shape[-1]%16:raise ValueError('Unsupported original reduction size')
        packed=self.static_cache.lookup(w)
        self.record('packed_weight_hit' if packed is not None else 'packed_weight_miss')
        k,n=w.shape;m=a.numel()//k;bn=geometry(m,k,n)
        eligible=packed is not None and bn is not None
        if eligible:out,compiled=fused(a,w,initial=initial,packed=packed,bn=bn)
        else:out,compiled=ordinary(a,w,initial=initial,int8=True,packed=packed)
        kind='int8_fused' if eligible else 'int8_separate'
        self.record('int8_dense');self.record(kind)
        shape=f'{m}x{k}x{n}:initial={initial is not None}:fused={eligible}:bn={bn if eligible else 32}'
        self.shapes[shape]=self.shapes.get(shape,0)+1
        if kind not in self.compiled:self.compiled[kind]=compiled
        return out
