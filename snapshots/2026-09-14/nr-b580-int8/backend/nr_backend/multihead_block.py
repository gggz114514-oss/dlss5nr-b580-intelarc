"""Native-reference multi-head Swin research path for C64, C128 and C256."""
from __future__ import annotations
import torch
from torch import Tensor,nn
from .pre_mlp import cubic_quantize,_decode_weights,quantize_fp8
from .tensor_math import sm89_f16_dot,sm89_f16_batched_dot,cubic_activation
from .attention import normalize_c32,swin_attention_windows


class BranchedMLP(nn.Module):
    """Sum sequential C->128(cubic)->32(FP8)->C branches into the half skip.

    C64/C128/C256 use two/four/eight branches. Equivalence is established
    separately by the captured encoder-prefix checks, not by record size.
    """
    def __init__(self,record: bytes,channels: int):
        super().__init__()
        if channels not in (64,128,256):raise ValueError('Unsupported channel family')
        branches=channels//32;expand_bytes=branches*channels*128;reduce_bytes=branches*128*32
        self.end=expand_bytes+reduce_bytes+branches*32*channels
        if len(record)<self.end+16+channels*2:raise ValueError('Truncated branched MLP')
        self.channels=channels
        self.register_buffer('expand',torch.stack([_decode_weights(record[i*channels*128:(i+1)*channels*128],channels,128)for i in range(branches)]))
        self.register_buffer('reduce',torch.stack([_decode_weights(record[expand_bytes+i*4096:expand_bytes+(i+1)*4096],128,32)for i in range(branches)]))
        off=expand_bytes+reduce_bytes
        self.register_buffer('project',torch.stack([_decode_weights(record[off+i*32*channels:off+(i+1)*32*channels],32,channels)for i in range(branches)]))
        self.register_buffer('skip_scale',torch.frombuffer(bytearray(record[self.end+16:self.end+16+channels*2]),dtype=torch.float16).clone())

    def forward_unquantized(self,features: Tensor) -> Tensor:
        if features.shape[-1]!=self.channels:raise ValueError('Wrong MLP channel count')
        x=quantize_fp8(features);result=(x*self.skip_scale).half()
        for branch in range(self.channels//32):
            expanded=sm89_f16_dot(x,self.expand[branch],chunk_k=16)
            hidden=cubic_quantize(expanded)
            hidden=quantize_fp8(sm89_f16_dot(hidden,self.reduce[branch],chunk_k=16))
            result=sm89_f16_dot(hidden,self.project[branch],chunk_k=16,initial=result)
        return result

    def forward(self,features: Tensor) -> Tensor:return quantize_fp8(self.forward_unquantized(features))


class MultiHeadAttention(nn.Module):
    """Head-major Q/K/V groups with 32 channels per head and 8x8 windows."""
    def __init__(self,record: bytes,channels: int, *, qkv_offset: int,bias_offset: int,scale_offset: int):
        super().__init__();self.channels=channels;self.heads=channels//32
        self.register_buffer('qkv',_decode_weights(record[qkv_offset:qkv_offset+3*channels*channels],channels,3*channels))
        self.register_buffer('scale',torch.frombuffer(bytearray(record[scale_offset:scale_offset+self.heads*4]),dtype=torch.float32).half().clone())
        physical=torch.frombuffer(bytearray(record[bias_offset:bias_offset+self.heads*8192]),dtype=torch.float16).reshape(self.heads,4,4,32,8)
        bias=torch.empty((self.heads,64,64),dtype=torch.float16);lane=torch.arange(32);g,t=lane//4,lane%4
        for a in range(4):
            for b in range(4):
                for frag in range(2):
                    for word in range(2):
                        for h in range(2):bias[:,a*16+g+8*word,b*16+frag*8+2*t+h]=physical[:,a,b,lane,frag*4+word*2+h]
        self.register_buffer('bias',bias)
        self.register_buffer('pixel_order',torch.tensor([base+g%4+8*(g//4)+16*word for base in (0,4,32,36)for word in range(2)for g in range(8)]))
        self.register_buffer('pixel_inverse',torch.argsort(self.pixel_order))

    def forward(self,features: Tensor) -> Tensor:
        height,width,channels=features.shape
        if channels!=self.channels or height%8 or width%8:raise ValueError('Expected whole 8x8 attention windows')
        rows,cols=height//8,width//8
        z=sm89_f16_dot(quantize_fp8(features),self.qkv,chunk_k=16).reshape(height,width,self.heads,3,32)
        q=quantize_fp8((normalize_c32(z[:,:,:,0])*self.scale[None,None,:,None]).half())
        k=quantize_fp8(normalize_c32(z[:,:,:,1]));v=quantize_fp8(z[:,:,:,2])
        def window(t):return t.reshape(rows,8,cols,8,self.heads,32).permute(4,0,2,1,3,5).reshape(self.heads,rows,cols,64,32)[...,self.pixel_order,:]
        q,k,v=window(q),window(k),window(v)
        result=torch.stack([swin_attention_windows(q[h],k[h],v[h],self.bias[h])for h in range(self.heads)])
        return result[...,self.pixel_inverse,:].reshape(self.heads,rows,cols,8,8,32).permute(1,3,2,4,0,5).reshape(height,width,channels)


class MultiHeadSwinBlock(nn.Module):
    """Reference-math C64/C128/C256 encoder block with native padded down output."""
    def __init__(self,record: bytes,channels: int=64, *, window_shift: tuple[int,int]=(0,0),downsample: bool=False):
        super().__init__()
        if len(window_shift)!=2 or any(s not in (0,4)for s in window_shift):raise ValueError('Expected shifts zero or four')
        self.channels=channels;self.window_shift=tuple(window_shift)
        self.mlp=BranchedMLP(record,channels)
        qkv=self.mlp.end+32+channels*2;bias=qkv+3*channels*channels;scale=bias+(channels//32)*8192
        projection=scale+max(16,channels//32*4);skip=projection+channels*channels;end=skip+channels*2
        if len(record)!=end+(2*channels*channels if downsample else 16):raise ValueError('Unexpected multi-head Swin record size')
        self.attention=MultiHeadAttention(record,channels,qkv_offset=qkv,bias_offset=bias,scale_offset=scale)
        self.register_buffer('output_weight',_decode_weights(record[projection:skip],channels,channels))
        self.register_buffer('skip_scale',torch.frombuffer(bytearray(record[skip:end]),dtype=torch.float16).clone())
        self.register_buffer('down_weight',_decode_weights(record[end:end+2*channels*channels],channels,2*channels)if downsample else None)

    def forward_unquantized(self,features: Tensor) -> Tensor:
        if features.ndim!=3 or features.shape[-1]!=self.channels:raise ValueError('Wrong HWC features')
        height,width=features.shape[:2]
        if min(height,width)==0 or height%2 or width%2:raise ValueError('Expected positive even dimensions')
        sy,sx=self.window_shift
        features=torch.nn.functional.pad(quantize_fp8(features),(0,0,sx,(-width-sx)%8,sy,(-height-sy)%8))
        # This family reloads its residual from the shared FP8 boundary.
        mlp=self.mlp(features);attended=self.attention(mlp)
        result=sm89_f16_dot(quantize_fp8(attended),self.output_weight,chunk_k=16,initial=(mlp*self.skip_scale).half())
        return result[sy:sy+height,sx:sx+width]

    def forward(self,features: Tensor) -> Tensor:return quantize_fp8(self.forward_unquantized(features))

    def forward_outputs(self,features: Tensor) -> tuple[Tensor,Tensor|None]:
        full=self.forward_unquantized(features);skip=quantize_fp8(full)
        if self.down_weight is None:return skip,None
        top=(full[0::2,0::2]+full[0::2,1::2]).half();bottom=(full[1::2,0::2]+full[1::2,1::2]).half()
        pooled=quantize_fp8(((top+bottom).half()*.25).half())
        down=quantize_fp8(sm89_f16_dot(pooled,self.down_weight,chunk_k=16))
        # The next input view aligns each spatial dimension to four. In the
        # captured C256 transition this is 10x10 data inside a 12x12 zero view.
        down=torch.nn.functional.pad(down,(0,0,0,(-down.shape[1])%4,0,(-down.shape[0])%4))
        return skip,down
