"""Cache immutable registered matrix buffers and 2D slices of branched weights.

Parent tensors are retained so storage cannot be reused by dynamic activations.
Keys include byte address, dtype, device, shape and strides. Normal tensor version
counters invalidate modified buffers. Inference tensors have no version counter;
their owner must explicitly promise immutability or prepacking is rejected.
"""
from dataclasses import dataclass
import torch
from fast_matrices_v3 import FastMatrices,quantize,dot

def key(w):
    return (w.device.type,w.device.index,w.dtype,w.data_ptr(),tuple(w.shape),tuple(w.stride()))

def version(w):
    try:return w._version
    except RuntimeError:return None

@dataclass
class PackedView:
    parent:object
    matrix:object
    packed:object
    scale:object
    version:int|None
    name:str

class StaticWeightCache:
    def __init__(self):self.entries={};self.counts={}

    def prepack(self,model,*,assume_immutable=False):
        new={}
        for name,parent in model.named_buffers():
            if parent.dtype!=torch.float16 or parent.ndim not in (2,3):continue
            k,n=parent.shape[-2:]
            if not (16<=k<=4096 and k%16==0 and n%16==0 and k*n<=2**21):continue
            before=version(parent)
            if before is None and not assume_immutable:
                raise ValueError('Inference weights require explicit immutability contract')
            matrices=[parent] if parent.ndim==2 else list(parent.unbind(0))
            for i,w in enumerate(matrices):
                q,s,_=quantize(w,columns=True)
                new[key(w)]=PackedView(parent,w,q,s,before,name if parent.ndim==2 else f'{name}[{i}]')
            if before!=version(parent):raise RuntimeError('Weight changed while packing')
        torch.xpu.synchronize()
        self.entries=new
        self.counts={}

    def lookup(self,w):
        entry=self.entries.get(key(w))
        kind='miss'
        if entry is not None:
            if entry.version is None or version(entry.parent)==entry.version:
                kind='hit'
            else:kind='stale'
        self.counts[kind]=self.counts.get(kind,0)+1
        return (entry.packed,entry.scale) if kind=='hit' else None

    @property
    def packed_bytes(self):
        return sum(e.packed.numel()*e.packed.element_size()+e.scale.numel()*e.scale.element_size() for e in self.entries.values())

class CachedFastMatrices(FastMatrices):
    def __init__(self):
        super().__init__()
        self.static_cache=StaticWeightCache()
        self.use_view_cache=False

    def select(self,mode):
        self.use_view_cache=mode=='int8_cached'
        super().select('int8_dense' if self.use_view_cache else mode)
        self.static_cache.counts={}

    def dense(self,a,w,*,chunk_k,initial=None,**kwargs):
        if not self.use_view_cache or chunk_k==8:
            return super().dense(a,w,chunk_k=chunk_k,initial=initial,**kwargs)
        if chunk_k!=16 or a.shape[-1]%16:
            raise ValueError('Unsupported original reduction size')
        packed=self.static_cache.lookup(w)
        self.record('packed_weight_hit' if packed is not None else 'packed_weight_miss')
        out,compiled=dot(a,w,initial=initial,int8=True,packed=packed)
        self.record('int8_dense')
        if 'int8_dense' not in self.compiled:self.compiled['int8_dense']=compiled
        return out
