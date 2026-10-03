# Explicit native-half attention version; see native_half_attention_fma_v1.py.
"""C32 QKV dot, ordered half normalization and window pack in one kernel.

The input already crossed its caller's FP8 boundary. Each Q/K/V projection keeps
the original FP32 K32 dot and half store. Gather input in the owned native pixel
order; write normalized/quantized windows directly without a global96-channel
projected intermediate. Logical32768-row chunk accounting is unchanged.
"""
import torch
import triton
import triton.language as tl
import nr_backend.attention as attention
from nr_backend.execution import current_arithmetic_backend,record_arithmetic_dispatch
from nr_backend.triton_attention_normalize import _nan_left,rsqrt_half_clamped
from native_half_attention_fma_v1 import half_fma_attention as _half_fma_value
from nr_backend.triton_fp8 import _round_fp8_half
from nr_backend.unround_policy import ENABLED, attention_owner
from fused_swin_core_v2 import _halves
from c32_chunk_layout_v2 import ChunkedHeadLayout as Base
from window_layout_v1 import unpack
import capture_body_v1 as body


@triton.jit
def _kernel(X,WEIGHT,SCALE,ORDER,Q,K,V,W:tl.constexpr,COLS:tl.constexpr,BM:tl.constexpr,
            ROUND_QKV:tl.constexpr=True):
    r=tl.program_id(0)*BM+tl.arange(0,BM)
    family=tl.program_id(1)
    lane=tl.arange(0,32)
    physical=tl.load(ORDER+r%64).to(tl.int32)
    window=r//64
    y=(window//COLS)*8+physical//8
    x=(window%COLS)*8+physical%8
    features=tl.load(X+(y*W+x)[:,None]*32+lane[None,:])
    weight=tl.load(WEIGHT+lane[:,None]*96+family*32+lane[None,:])
    z=tl.dot(features,weight,out_dtype=tl.float32).to(tl.float16)
    first,last=_halves(z,BM,32)
    x0,x8=_halves(first,BM,16)
    x16,x24=_halves(last,BM,16)
    lane8=tl.arange(0,8)
    if family<2:
        a=(x16.to(tl.float32)*x16.to(tl.float32)).to(tl.float16)
        a=_nan_left(_half_fma_value(x0,x0,a),x0,a)
        b=(x24.to(tl.float32)*x24.to(tl.float32)).to(tl.float16)
        b=_nan_left(_half_fma_value(x8,x8,b),x8,b)
        total=_nan_left((a.to(tl.float32)+b.to(tl.float32)).to(tl.float16),a,b)
        for mask in tl.static_range(3):
            other=tl.gather(total,tl.broadcast_to((lane8^(4>>mask))[None,:],(BM,8)),1)
            total=_nan_left((total.to(tl.float32)+other.to(tl.float32)).to(tl.float16),total,other)
        denominator=tl.gather(total,tl.full((BM,1),0,tl.int32),1)
        scale=rsqrt_half_clamped(denominator)
        x0=_nan_left((x0.to(tl.float32)*scale.to(tl.float32)).to(tl.float16),x0,scale)
        x8=_nan_left((x8.to(tl.float32)*scale.to(tl.float32)).to(tl.float16),x8,scale)
        x16=_nan_left((x16.to(tl.float32)*scale.to(tl.float32)).to(tl.float16),x16,scale)
        x24=_nan_left((x24.to(tl.float32)*scale.to(tl.float32)).to(tl.float16),x24,scale)
        if family==0:
            factor=tl.load(SCALE)
            x0=_nan_left((x0.to(tl.float32)*factor.to(tl.float32)).to(tl.float16),x0,factor)
            x8=_nan_left((x8.to(tl.float32)*factor.to(tl.float32)).to(tl.float16),x8,factor)
            x16=_nan_left((x16.to(tl.float32)*factor.to(tl.float32)).to(tl.float16),x16,factor)
            x24=_nan_left((x24.to(tl.float32)*factor.to(tl.float32)).to(tl.float16),x24,factor)
    output=tl.where(family==0,Q,tl.where(family==1,K,V))
    dest=r[:,None]*32+lane8[None,:]
    if ROUND_QKV:
        x0=_round_fp8_half(x0);x8=_round_fp8_half(x8)
        x16=_round_fp8_half(x16);x24=_round_fp8_half(x24)
    tl.store(output+dest,x0)
    tl.store(output+dest+8,x8)
    tl.store(output+dest+16,x16)
    tl.store(output+dest+24,x24)


def forward(features,weight,scale,order,*,bm=16,warps=4,stages=1,
            round_qkv=True):
    if features.ndim!=3 or features.shape[-1]!=32 or min(features.shape)<=0 or features.shape[0]%8 or features.shape[1]%8:
        raise ValueError('Expected whole nonempty HWC32 windows')
    if features.device.type!='xpu' or features.dtype!=torch.float16 or not features.is_contiguous():
        raise ValueError('Expected contiguous half XPU features at the caller FP8 boundary')
    for value,shape,dtype in ((weight,(32,96),torch.float16),(scale,(1,),torch.float16),(order,(64,),torch.int64)):
        if value.shape!=shape or value.dtype!=dtype or value.device!=features.device or not value.is_contiguous():
            raise ValueError('Invalid owned projection/scale/order')
    if bm not in (16,32) or warps not in (4,8) or stages not in (1,2):raise ValueError('Unsupported launch configuration')
    h,w=features.shape[:2]
    outputs=tuple(torch.empty((h//8,w//8,64,32),dtype=features.dtype,device=features.device) for _ in range(3))
    kernel=_kernel[(h*w//bm,3)](features,weight,scale,order,*outputs,w,w//8,bm,
                               ROUND_QKV=round_qkv,
                               num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    return outputs,kernel


class ProjectedHeadLayout(Base):
    def __init__(self,model,provider,*,bm=16,warps=4,stages=1):
        super().__init__(model,provider);self.projection_options=dict(bm=bm,warps=warps,stages=stages)

    def c32(self,module,features):
        if id(module) not in self.c32_modules or self.provider.mode!='fp16_xmx' or current_arithmetic_backend()!='triton' or features.device.type!='xpu':
            return super().c32(module,features)
        family=getattr(module,"rounding_family","c32")
        (q,k,v),_=forward(features,module.front.qkv,module.front.scale,module.pixel_order,
                          round_qkv=family not in ENABLED,**self.projection_options)
        chunks=triton.cdiv(features.numel()//32,32768)
        for _ in range(chunks):
            record_arithmetic_dispatch('dense')
            for _ in range(2):record_arithmetic_dispatch('attention_normalize_c32')
            for _ in range(2):record_arithmetic_dispatch('fp8')
        record_arithmetic_dispatch('fp8')
        if attention.C32AttentionFront.forward is not body.c32_front:
            for _ in range(chunks//8):torch.xpu.synchronize(features.device)
        with attention_owner(family):
            result=attention.swin_attention_windows(q,k,v,module.bias)
        self.calls['c32']=self.calls.get('c32',0)+1
        self.calls['c32_chunk_pack']=self.calls.get('c32_chunk_pack',0)+chunks
        self.calls['c32_projection_pack']=self.calls.get('c32_projection_pack',0)+1
        return unpack(result.unsqueeze(0),module.pixel_inverse).reshape(features.shape)
