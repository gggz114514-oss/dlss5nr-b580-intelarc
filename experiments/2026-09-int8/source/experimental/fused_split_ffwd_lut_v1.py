"""Batch eight C512 groups with explicit cubic LUT and ordered K32 sums.

The leading C512 projection stays unchanged. Expansion uses two consecutive K32
FP32 dot accumulations; each half/cubic/FP8 boundary is retained. Reduction uses
eight ordered K32 accumulations, then half and FP8. No full C2048 intermediate.
"""
from contextlib import contextmanager
import torch
import triton
import triton.language as tl
import nr_backend.split_block as split
from nr_backend.triton_fp8 import _round_fp8_half
from nr_backend.execution import current_arithmetic_backend, record_arithmetic_dispatch
from fused_c32_mlp_lut_v1 import lookup
from cubic_lut_constant_v1 import Constant
from fused_split_ffwd_v1 import FusedSplit as Base


@triton.jit
def _kernel(X, EXPAND, REDUCE, LUT, OUT, M:tl.constexpr, BM:tl.constexpr, BN:tl.constexpr):
    rows = tl.program_id(0)*BM + tl.arange(0,BM)
    group = tl.program_id(1)
    output_columns = tl.program_id(2)*BN + tl.arange(0,BN)
    lanes = tl.arange(0,32)
    reduced = tl.full((BM,BN),0.,tl.float32)
    for part in range(8):
        expanded = tl.full((BM,32),0.,tl.float32)
        for block in range(2):
            k = block*32 + lanes
            x = tl.load(X + rows[:,None]*512 + group*64 + k[None,:], rows[:,None]<M, other=0)
            w = tl.load(EXPAND + group*64*256 + k[:,None]*256 + part*32 + lanes[None,:])
            expanded = tl.dot(x,w,expanded,out_dtype=tl.float32)
        hidden = lookup(expanded.to(tl.float16), LUT)
        w = tl.load(REDUCE + group*256*64 + (part*32+lanes[:,None])*64 + output_columns[None,:])
        reduced = tl.dot(hidden,w,reduced,out_dtype=tl.float32)
    result = _round_fp8_half(reduced.to(tl.float16))
    tl.store(OUT + rows[:,None]*512 + group*64 + output_columns[None,:], result, rows[:,None]<M)


def forward(z, expand, reduce, lut, *, bm=16, bn=64, stages=1):
    if z.device.type!='xpu' or z.dtype!=torch.float16 or z.shape[-1]!=512 or z.numel()==0:
        raise ValueError('Expected nonempty prequantized half XPU C512 features')
    if bm not in (16,32) or bn not in (32,64) or stages not in (1,2):
        raise ValueError('Unsupported split FFWD configuration')
    for value, shape in ((expand,(8,64,256)),(reduce,(8,256,64))):
        if tuple(value.shape)!=shape or value.device!=z.device or value.dtype!=z.dtype or not value.is_contiguous():
            raise ValueError('Invalid C512 group weights')
    if lut.device!=z.device or lut.dtype!=torch.int16 or lut.shape!=(65536,) or not lut.is_contiguous():
        raise ValueError('Invalid owned cubic half-bit LUT')
    z=z.contiguous()
    out=torch.empty_like(z)
    m=z.numel()//512
    kernel=_kernel[(triton.cdiv(m,bm),8,64//bn)](z,expand,reduce,lut,out,m,bm,bn,
        num_warps=4,num_stages=stages,enable_fp_fusion=False)
    return out,kernel


class FusedSplit(Base):
    def __init__(self, model, provider, *, configuration=None):
        super().__init__(model, provider)
        self.constant = Constant(model)
        self.configuration = None if configuration is None else dict(configuration)

    def apply(self, module, features):
        if (id(module) not in self.modules or current_arithmetic_backend() != 'triton'
                or features.device.type != 'xpu' or self.provider.mode != 'fp16_xmx'):
            return self.original(module, features)
        lut = self.constant.require()
        z = split.q(split.dot(split.q(features), module.linear, chunk_k=16))
        config = self.configuration or dict(bm=16, bn=32 if z.numel()//512<=512 else 64, stages=1)
        value, _ = forward(z, module.expand, module.reduce, lut, **config)
        for _ in range(8):
            record_arithmetic_dispatch('dense')
            record_arithmetic_dispatch('cubic_fp8')
            record_arithmetic_dispatch('dense')
            record_arithmetic_dispatch('fp8')
        self.calls += 1
        return value
