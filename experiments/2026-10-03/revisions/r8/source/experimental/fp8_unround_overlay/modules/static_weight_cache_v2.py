"""Include registered ViT/decoder split-K views, quantized per actual partition."""
import torch
from nr_backend.vit_block import VitBlock
from nr_backend.decoder import DecoderInputUpsample
from fast_matrices_v3 import quantize
from static_weight_cache_v1 import StaticWeightCache as BaseCache,PackedView,key,version,CachedFastMatrices as BaseExperiment

class StaticWeightCache(BaseCache):
    def prepack(self,model,*,assume_immutable=False):
        partitions={}
        for module in model.modules():
            if isinstance(module,VitBlock):
                partitions[id(module.contract)]=4
                partitions[id(module.projection)]=4
                partitions[id(module.qkv_weight)]=2
            elif isinstance(module,DecoderInputUpsample):
                partitions[id(module.weight)]=4
        new={}
        for name,parent in model.named_buffers():
            if parent.dtype!=torch.float16 or parent.ndim not in (2,3):continue
            k,n=parent.shape[-2:]
            if not (16<=k<=4096 and k%16==0 and n%16==0 and k*n<=2**24):continue
            before=version(parent)
            if before is None and not assume_immutable:
                raise ValueError('Inference weights require explicit immutability contract')
            parts=partitions.get(id(parent),1)
            if parts>1:
                assert parent.ndim==2 and k%parts==0
                matrices=list(parent.split(k//parts,dim=0))
            else:matrices=[parent] if parent.ndim==2 else list(parent.unbind(0))
            for i,w in enumerate(matrices):
                # A split-K block has its OWN maximum and scale. Slicing a
                # quantized full parent would change the existing W8A8 math.
                q,s,_=quantize(w,columns=True)
                new[key(w)]=PackedView(parent,w,q,s,before,name if len(matrices)==1 else f'{name}[{i}]')
            if before!=version(parent):raise RuntimeError('Weight changed while packing')
        torch.xpu.synchronize()
        self.entries=new;self.counts={}

class CachedFastMatrices(BaseExperiment):
    def __init__(self):
        super().__init__()
        self.static_cache=StaticWeightCache()
