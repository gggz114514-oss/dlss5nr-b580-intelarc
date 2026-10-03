"""Same NR body operations/chunking with explicit capture-safe scheduling.

Only this experiment's class methods are scoped during warmup/capture. No global
torch synchronization stub. Input validation, noise/seed and history construction
remain in MotionNR.forward, outside the graph.
"""
from contextlib import contextmanager
import torch
import nr_backend.pre_mlp as mlp
import nr_backend.attention as attention

def c32_mlp(self,projected):
    if projected.shape[-1]!=32 or projected.device!=self.expansion.device:
        raise ValueError('Expected projected features ending in C32 on the module device')
    if projected.numel()>32768*32:
        flat=projected.reshape(-1,32);result=torch.empty_like(flat,dtype=torch.float16)
        for start in range(0,len(flat),32768):
            result[start:start+32768]=self.forward_unquantized(flat[start:start+32768])
        return result.reshape(projected.shape)
    projected=projected.to(torch.float16)
    expanded=mlp.sm89_f16_dot(mlp.quantize_fp8(projected),self.expansion,chunk_k=16)
    hidden=mlp.cubic_quantize(expanded)
    initial=(projected*self.skip_scale).to(torch.float16)
    return mlp.sm89_f16_dot(hidden,self.contraction,chunk_k=16,initial=initial)

def c32_front(self,features):
    if features.numel()>32768*32:
        flat=features.reshape(-1,32);outputs=[torch.empty_like(flat,dtype=torch.float16) for _ in range(3)]
        for start in range(0,len(flat),32768):
            values=self.forward(flat[start:start+32768])
            for output,value in zip(outputs,values):output[start:start+32768]=value
        return tuple(t.reshape(features.shape) for t in outputs)
    projected=attention.sm89_f16_dot(features,self.qkv,chunk_k=16)
    q=attention.quantize_fp8((attention.normalize_c32(projected[...,:32])*self.scale).half())
    k=attention.quantize_fp8(attention.normalize_c32(projected[...,32:64]))
    return q,k,projected[...,64:96]

@contextmanager
def installed():
    original_mlp=mlp.C32MLP.forward_unquantized
    original_front=attention.C32AttentionFront.forward
    mlp.C32MLP.forward_unquantized=c32_mlp
    attention.C32AttentionFront.forward=c32_front
    try:yield
    finally:
        assert mlp.C32MLP.forward_unquantized is c32_mlp and attention.C32AttentionFront.forward is c32_front
        mlp.C32MLP.forward_unquantized=original_mlp
        attention.C32AttentionFront.forward=original_front

def forward_front(self,rgb,front,*,previous=None,sigmoid=None,blend_scale=None,history_reciprocal=None,return_float32=False):
    # Identical executor._forward_front dataflow. Progress/fences belong outside
    # capture; the graph adapter falls back to the original for progress callbacks.
    pre_skip,x=self.pre.forward_features_outputs(front);del front;skips=[]
    for c,group in zip((32,64,128,256),self.encoder):
        for block in group:
            skip,down=block.forward_outputs(x);x=down if down is not None else skip
        skips.append(skip)
    for i,block in enumerate(self.encoder512):
        values=block.forward_boundaries(x);x=values[-1]
        if i==7:skip512=values[3]
    vit_shape=x.shape;x=x.reshape(-1,1024)
    for block in self.vit:x=block(x)
    x=self.decoder_input(x.reshape(vit_shape),skip512)
    for block in self.decoder512:x=block(x)
    for c,group,skip in zip((256,128,64,32),self.decoder,reversed(skips)):
        x=group[0](x,skip)
        for block in group[1:]:x=block(x)
    skips.clear();del skip,skip512,values
    return self.post(x,pre_skip,rgb,previous=previous,sigmoid=sigmoid,blend_scale=blend_scale,history_reciprocal=history_reciprocal,return_float32=return_float32)
