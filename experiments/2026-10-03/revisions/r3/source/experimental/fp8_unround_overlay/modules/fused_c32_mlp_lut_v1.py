"""C32 streaming MLP with an explicit owned cubic+FP8 lookup pointer.

Matrix K32 reduction order, raw-input skip, half boundaries and output stay
identical to fused_c32_mlp_v1. Only the internal pointwise activation changes.
"""
import torch
import triton
import triton.language as tl
from nr_backend.triton_fp8 import _round_fp8_half
from nr_backend.execution import current_arithmetic_backend, record_arithmetic_dispatch
from fused_c32_mlp_v1 import FusedC32 as Base
from cubic_lut_constant_v1 import Constant
from cubic_half_lut_constant_v1 import BUFFER as HALF_BUFFER, Constant as HalfConstant
from nr_backend.unround_policy import ENABLED


@triton.jit
def lookup(x, lut):
    index = x.to(tl.uint16, bitcast=True).to(tl.int32)
    return tl.load(lut + index).to(tl.float16, bitcast=True)


@triton.jit
def _kernel(X, EXPAND, CONTRACT, SCALE, LUT, OUT, M: tl.constexpr, BM: tl.constexpr,
            ROUND_INPUT: tl.constexpr=True):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    lane = tl.arange(0, 32)
    raw = tl.load(X + row[:, None] * 32 + lane[None, :], row[:, None] < M, other=0)
    if ROUND_INPUT:
        x = _round_fp8_half(raw)
    else:
        x = raw
    contracted = tl.full((BM, 32), 0., tl.float32)
    for part in range(4):
        w = tl.load(EXPAND + lane[:, None] * 128 + part * 32 + lane[None, :])
        expanded = tl.dot(x, w, out_dtype=tl.float32)
        hidden = lookup(expanded.to(tl.float16), LUT)
        weight = tl.load(CONTRACT + (part * 32 + lane[:, None]) * 32 + lane[None, :])
        contracted = tl.dot(hidden, weight, contracted, out_dtype=tl.float32)
    scale = tl.load(SCALE + lane)
    initial = (raw.to(tl.float32) * scale[None, :].to(tl.float32)).to(tl.float16)
    value = (contracted + initial.to(tl.float32)).to(tl.float16)
    tl.store(OUT + row[:, None] * 32 + lane[None, :], value, row[:, None] < M)


def forward(features, expansion, contraction, skip_scale, lut, *, bm=32, warps=4, stages=1,
            round_input=True):
    if features.device.type != 'xpu' or features.shape[-1] != 32 or features.numel() == 0:
        raise ValueError('Expected projected XPU features with C32')
    if bm not in (16, 32) or warps not in (4, 8) or stages not in (1, 2):
        raise ValueError('Unsupported C32 launch configuration')
    for value, shape in ((expansion, (32, 128)), (contraction, (128, 32)), (skip_scale, (32,))):
        if value.shape != shape or value.device != features.device or value.dtype != torch.float16 or not value.is_contiguous():
            raise ValueError('Invalid C32 MLP weights')
    if lut.device != features.device or lut.dtype != torch.int16 or lut.shape != (65536,) or not lut.is_contiguous():
        raise ValueError('Invalid half-bit cubic LUT')
    x = features.half().contiguous()
    out = torch.empty_like(x)
    m = x.numel() // 32
    kernel = _kernel[(triton.cdiv(m, bm),)](x, expansion, contraction, skip_scale, lut, out, m, bm,
                                         ROUND_INPUT=round_input,
                                         num_warps=warps, num_stages=stages, enable_fp_fusion=False)
    return out, kernel


class FusedC32(Base):
    def __init__(self, model, provider, *, bm=32, warps=4, stages=1):
        super().__init__(model, provider, bm=bm, warps=warps, stages=stages)
        self.half_lut = hasattr(model, HALF_BUFFER)
        self.constant = HalfConstant(model) if self.half_lut else Constant(model)

    def apply(self, module, original, features):
        if current_arithmetic_backend() != 'triton' or features.device.type != 'xpu' or self.provider.mode != 'fp16_xmx':
            return original(features)
        family = getattr(module, "rounding_family", "c32")
        result, _ = forward(features, module.expansion, module.contraction, module.skip_scale,
                            self.constant.require(), round_input=family not in ENABLED,
                            **self.options)
        for _ in range(triton.cdiv(features.numel() // 32, 32768)):
            record_arithmetic_dispatch('fp8')
            record_arithmetic_dispatch('cubic_fp8')
            record_arithmetic_dispatch('dense')
            record_arithmetic_dispatch('dense')
        key = str(features.numel() // 32)
        self.calls[key] = self.calls.get(key, 0) + 1
        return result
