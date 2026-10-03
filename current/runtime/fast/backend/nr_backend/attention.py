"""C32 cosine attention with observed SM89 rounding and fragment order."""
import torch
from torch import Tensor,nn
from .pre_mlp import _decode_weights,quantize_fp8
from .tensor_math import sm89_f16_dot,sm89_f16_batched_dot,half_fma
from .execution import current_arithmetic_backend,record_arithmetic_dispatch

def normalize_c32(vector: Tensor) -> Tensor:
    """Observed half dot/reduction order for a 32-channel cosine-attention vector.

    The reciprocal square root uses portable FP32 arithmetic and is only claimed
    where captured half outputs match; general MUFU equivalence is not established.
    """
    if vector.shape[-1]!=32:raise ValueError('Expected a C32 vector')
    if current_arithmetic_backend() == 'triton' and vector.device.type == 'xpu':
        from .triton_attention_normalize import normalize_c32 as fused_normalize_c32
        result = fused_normalize_c32(vector)
        record_arithmetic_dispatch('attention_normalize_c32')
        return result
    x=vector.to(torch.float16)
    a=(x[...,16:24]*x[...,16:24]).half()
    a=half_fma(x[...,:8],x[...,:8],a)
    b=(x[...,24:32]*x[...,24:32]).half()
    b=half_fma(x[...,8:16],x[...,8:16],b)
    total=(a+b).half()
    indices=torch.arange(8,device=x.device)
    for mask in (4,2,1):total=(total+total[...,indices^mask]).half()
    total=total[...,:1].clamp(min=0.00006198883056640625)
    scale=torch.rsqrt(total.float()).half()
    return (x*scale).half()


def score_exponential(scores: Tensor) -> Tensor:
    """Observed HFMA2/clamp/LEA exponential approximation before row normalization.

    This is deliberately not torch.exp or a complete softmax. The clamp bounds
    make the packed LEA carry equivalent to this independent half-bit transform.
    """
    if current_arithmetic_backend() == 'triton' and scores.device.type == 'xpu':
        from .triton_attention_exp import exponential
        result = exponential(scores, vit=False)
        record_arithmetic_dispatch('attention_exp_swin')
        return result
    affine=half_fma(scores,0.044921875,1.30078125)
    affine=affine.clamp(1.03125,1.5693359375)
    bits=affine.contiguous().view(torch.int16).to(torch.int32)&0xffff
    return (((bits<<5)+0x8000)&0xffff).to(torch.int16).contiguous().view(torch.float16)


def attention_row_sum64(exponential: Tensor) -> Tensor:
    """Captured reduction order for 64 keys in native fragment pixel order."""
    if exponential.shape[-1]!=64:raise ValueError('Expected 64 key weights')
    if current_arithmetic_backend() == 'triton' and exponential.device.type == 'xpu':
        from .triton_attention_weights import row_sum64
        result = row_sum64(exponential)
        record_arithmetic_dispatch('attention_row_sum64')
        return result
    x=exponential.half().reshape(*exponential.shape[:-1],8,8)
    pairs=[(x[...,i,:]+x[...,i+1,:]).half() for i in (0,2,4,6)]
    partial=(((pairs[0]+pairs[1]).half()+pairs[2]).half()+pairs[3]).half()
    total=(partial[...,:2]+partial[...,2:4]).half()
    total=(total+partial[...,4:6]).half()
    total=(total+partial[...,6:8]).half()
    return (total[...,:1]+total[...,1:2]).half()


def attention_row_reciprocal(exponential: Tensor) -> Tensor:
    denominator=attention_row_sum64(exponential).clamp(min=0.00006198883056640625)
    return denominator.float().reciprocal().half()


def normalize_attention_weights(exponential: Tensor) -> Tensor:
    """Native half reciprocal/multiply followed by the E4M3 boundary."""
    if current_arithmetic_backend() == 'triton' and exponential.device.type == 'xpu':
        from .triton_attention_weights import normalize_weights
        result = normalize_weights(exponential)
        record_arithmetic_dispatch('attention_weights')
        return result
    return quantize_fp8((exponential.half()*attention_row_reciprocal(exponential)).half())


def swin_attention_windows(query: Tensor,key: Tensor,value: Tensor,bias: Tensor) -> Tensor:
    """Compute independent windows in bounded groups, preserving each K order."""
    if query.shape!=key.shape or query.shape!=value.shape or query.shape[-2:]!=(64,32) or bias.shape!=(64,64):
        raise ValueError('Expected matching 64x32 windows and one 64x64 head bias')
    shape=query.shape;q=query.reshape(-1,64,32);k=key.reshape(-1,64,32);v=value.reshape(-1,64,32)
    result=torch.empty_like(q)
    windows_per_group=1024 if current_arithmetic_backend()=='triton' else 64
    groups_per_sync=max(1,1024//windows_per_group)
    for start in range(0,len(q),windows_per_group):
        sl=slice(start,start+windows_per_group);count=len(q[sl])
        scores=sm89_f16_batched_dot(q[sl],k[sl].transpose(-1,-2),initial=bias.expand(count,64,64))
        weights=normalize_attention_weights(score_exponential(scores))
        result[sl]=sm89_f16_batched_dot(weights,v[sl])
        if query.device.type=='xpu' and (start//windows_per_group)%groups_per_sync==groups_per_sync-1:torch.xpu.synchronize(query.device)
    return result.reshape(shape)


class C32AttentionFront(nn.Module):
    """QKV projection and Q/K normalization; does not apply attention or softmax."""
    def __init__(self,record: bytes, *, qkv_offset: int, scale_offset: int):
        super().__init__()
        if min(qkv_offset,scale_offset)<0 or len(record)<max(qkv_offset+3072,scale_offset+4):
            raise ValueError('Truncated C32 attention record')
        self.register_buffer('qkv',_decode_weights(record[qkv_offset:qkv_offset+3072],32,96))
        self.register_buffer('scale',torch.frombuffer(bytearray(record[scale_offset:scale_offset+4]),dtype=torch.float32).half().clone())

    def forward(self,features: Tensor) -> tuple[Tensor,Tensor,Tensor]:
        # QKV projection and normalization are independent at every pixel.
        # Keep the large image's QKV/normalization scratch bounded as well.
        if features.numel()>32768*32:
            flat=features.reshape(-1,32);outputs=[torch.empty_like(flat,dtype=torch.float16)for _ in range(3)]
            for start in range(0,len(flat),32768):
                values=self.forward(flat[start:start+32768])
                for output,value in zip(outputs,values):output[start:start+32768]=value
                if features.device.type=='xpu' and (start//32768)%8==7:torch.xpu.synchronize(features.device)
            return tuple(t.reshape(features.shape)for t in outputs)
        projected=sm89_f16_dot(features,self.qkv,chunk_k=16)
        q=quantize_fp8((normalize_c32(projected[...,:32])*self.scale).half())
        k=quantize_fp8(normalize_c32(projected[...,32:64]))
        # The core quantizes V at the value-product boundary.
        return q,k,projected[...,64:96]


class C32AttentionCore(nn.Module):
    """Unshifted 8x8 cosine attention through weighted V, before output projection.

    All 64 queries are computed. This remains a correctness implementation.
    """
    def __init__(self,record: bytes, *, qkv_offset: int, bias_offset: int, scale_offset: int):
        super().__init__()
        self.front=C32AttentionFront(record,qkv_offset=qkv_offset,scale_offset=scale_offset)
        pixels=[base+g%4+8*(g//4)+16*word for base in (0,4,32,36) for word in range(2) for g in range(8)]
        self.register_buffer('pixel_order',torch.tensor(pixels,dtype=torch.int64))
        self.register_buffer('pixel_inverse',torch.argsort(self.pixel_order))
        if bias_offset<0 or len(record)<bias_offset+8192:raise ValueError('Truncated C32 bias record')
        physical=torch.frombuffer(bytearray(record[bias_offset:bias_offset+8192]),dtype=torch.float16).reshape(4,4,32,8)
        bias=torch.empty((64,64),dtype=torch.float16)
        lane=torch.arange(32);g,t=lane//4,lane%4
        for a in range(4):
            for b in range(4):
                for frag in range(2):
                    for word in range(2):
                        for half in range(2):bias[a*16+g+8*word,b*16+frag*8+2*t+half]=physical[a,b,lane,frag*4+word*2+half]
        self.register_buffer('bias',bias)

    def forward(self,features: Tensor) -> Tensor:
        if features.ndim!=3 or features.shape[-1]!=32 or features.shape[0]%8 or features.shape[1]%8:
            raise ValueError('Expected padded HWC32 with dimensions divisible by eight')
        height,width=features.shape[:2];rows,cols=height//8,width//8
        q,k,v=self.front(features)
        def window(x):return x.reshape(rows,8,cols,8,32).permute(0,2,1,3,4).reshape(rows,cols,64,32)[...,self.pixel_order,:]
        q,k,v=window(q),window(k),window(quantize_fp8(v))
        attended=swin_attention_windows(q,k,v,self.bias)
        return attended[...,self.pixel_inverse,:].reshape(rows,cols,8,8,32).permute(0,2,1,3,4).reshape(height,width,32)


class PreAttentionFront(C32AttentionFront):
    def __init__(self,record: bytes):
        if len(record)!=21696:raise ValueError('Expected the pinned pre record')
        super().__init__(record,qkv_offset=0x2460,scale_offset=0x5060)


class PreAttentionCore(C32AttentionCore):
    def __init__(self,record: bytes):
        if len(record)!=21696:raise ValueError('Expected the pinned pre record')
        super().__init__(record,qkv_offset=0x2460,bias_offset=0x3060,scale_offset=0x5060)
