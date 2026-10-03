"""Current720 global ViT: native numerics and complete head-major consumers.

FP16 XMX dots use FP32 accumulators. Default full K changes the old two/P4
half boundaries and is a fast candidate. Ordered denominator and its padded16
correction remain; exponent is the model's special bit map, not softmax.
No Torch/XPU import or kernel compilation was performed by the author.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl
from nr_backend.triton_attention_normalize import _nan_left, rsqrt_half_clamped
from nr_backend.triton_cubic_fp8 import _half_fma_value
from nr_backend.triton_attention_weights import _add_half, reciprocal_half_clamped
from vit_numeric_suite_720_v1_kernels import _ordered_sum64
from vit_native_fma_720_v1 import _exp_vit
from nr_backend.triton_attention_exp import _kernel as _exp_reference

TOKENS, WIDTH, HEADS, PADDED = 240, 1024, 32, 256
ZERO_BITS, ZERO_VALUE, PADDING_CORRECTION = 0x2d60, 0.083984375, 1.34375


@triton.jit
def _norm_registers(z, BR: tl.constexpr, NATIVE: tl.constexpr):
    lane = tl.arange(0, 8)
    idx = tl.broadcast_to(lane[None, :], (BR, 8))
    x0 = tl.gather(z, idx, 1);x8 = tl.gather(z, idx + 8, 1)
    x16 = tl.gather(z, idx + 16, 1);x24 = tl.gather(z, idx + 24, 1)
    a = (x16.to(tl.float32) * x16.to(tl.float32)).to(tl.float16)
    b = (x24.to(tl.float32) * x24.to(tl.float32)).to(tl.float16)
    if NATIVE:
        a = _nan_left(tl.fma(x0, x0, a).to(tl.float16), x0, a)
        b = _nan_left(tl.fma(x8, x8, b).to(tl.float16), x8, b)
    else:
        a = _nan_left(_half_fma_value(x0, x0, a), x0, a)
        b = _nan_left(_half_fma_value(x8, x8, b), x8, b)
    total = _add_half(a, b)
    for shift in tl.static_range(3):
        other = tl.gather(total, tl.broadcast_to(
            (lane ^ (4 >> shift))[None, :], (BR, 8)), 1)
        total = _add_half(total, other)
    total = tl.gather(total, tl.full((BR, 1), 0, tl.int32), 1)
    scale = rsqrt_half_clamped(total)
    return _nan_left((z.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), z, scale)


@triton.jit
def _qkv(X, W, SCALE, Q, K, V, BM: tl.constexpr,
         FULL_K: tl.constexpr, NATIVE_NORM: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    head = tl.program_id(1);family = tl.program_id(2)
    lane = tl.arange(0, 32)
    col = head * 96 + family * 32 + lane
    if FULL_K:
        total = tl.full((BM, 32), 0, tl.float32)
        for block in range(32):
            kk = block * 32 + lane
            a = tl.load(X + row[:, None]*1024 + kk[None, :],
                        row[:, None] < 240, other=0)
            w = tl.load(W + kk[:, None]*3072 + col[None, :])
            total = tl.dot(a, w, total, out_dtype=tl.float32)
        z = total.to(tl.float16)
    else:
        first = tl.full((BM, 32), 0, tl.float32)
        second = tl.full((BM, 32), 0, tl.float32)
        for block in range(16):
            kk = block * 32 + lane
            a = tl.load(X + row[:, None]*1024 + kk[None, :],
                        row[:, None] < 240, other=0)
            b = tl.load(X + row[:, None]*1024 + 512 + kk[None, :],
                        row[:, None] < 240, other=0)
            wa = tl.load(W + kk[:, None]*3072 + col[None, :])
            wb = tl.load(W + (kk[:, None]+512)*3072 + col[None, :])
            first = tl.dot(a, wa, first, out_dtype=tl.float32)
            second = tl.dot(b, wb, second, out_dtype=tl.float32)
        z = _add_half(first.to(tl.float16), second.to(tl.float16))
    if family < 2:
        z = _norm_registers(z, BM, NATIVE_NORM)
        if family == 0:
            a = (z.to(tl.float32) * 5.65625).to(tl.float16)
            scale = tl.load(SCALE + head)
            z = _nan_left((a.to(tl.float32)*scale.to(tl.float32)).to(tl.float16),
                          a, scale)
    if family == 0:
        tl.store(Q + (head*240 + row[:, None])*32 + lane[None, :],
                 z, row[:, None] < 240)
    else:
        dst = tl.where(family == 1, K, V)
        # The last block additionally writes padded key/value zero rows.
        tl.store(dst + (head*256 + row[:, None])*32 + lane[None, :],
                 tl.where(row[:, None] < 240, z, 0), row[:, None] < 256)


@triton.jit
def _denominator_apply(EXPONENT, NUMERATOR, OUT, BR: tl.constexpr, ORDERED: tl.constexpr):
    rows = tl.program_id(0)*BR + tl.arange(0, BR)
    if ORDERED:
        eight = tl.arange(0, 8)
        total = _ordered_sum64(EXPONENT, rows, eight, 0, 7680, BR)
        for part in tl.static_range(1, 4):
            total = _add_half(total, _ordered_sum64(EXPONENT, rows, eight, part, 7680, BR))
    else:
        keys = tl.arange(0, 256)
        values = tl.load(EXPONENT + rows[:, None]*256 + keys[None, :],
                         rows[:, None] < 7680, other=0)
        total = tl.sum(values.to(tl.float32), 1)[:, None].to(tl.float16)
    correction = tl.full((BR, 1), 1.34375, tl.float16)
    corrected = _nan_left((total.to(tl.float32)-correction.to(tl.float32)).to(tl.float16),
                           total, correction)
    reciprocal = reciprocal_half_clamped(corrected)
    lane = tl.arange(0, 32)
    offsets = rows[:, None]*32 + lane[None, :]
    value = tl.load(NUMERATOR + offsets, rows[:, None] < 7680, other=0)
    result = _nan_left((value.to(tl.float32)*reciprocal.to(tl.float32)).to(tl.float16),
                       value, reciprocal)
    tl.store(OUT + offsets, result, rows[:, None] < 7680)


@triton.jit
def _projection(HEADS_IN, W, RESIDUAL, SKIP, OUT, BM: tl.constexpr,
                BN: tl.constexpr, FULL_K: tl.constexpr):
    row = tl.program_id(0)*BM + tl.arange(0, BM)
    col = tl.program_id(1)*BN + tl.arange(0, BN)
    lane = tl.arange(0, 32)
    valid = row[:, None] < 240
    offsets = row[:, None]*1024 + col[None, :]
    r = tl.load(RESIDUAL + offsets, valid, other=0)
    skip = tl.load(SKIP + col)
    initial = (r.to(tl.float32)*skip[None, :].to(tl.float32)).to(tl.float16)
    if FULL_K:
        total = tl.full((BM, BN), 0, tl.float32)
        for block in range(32):
            kk = block*32 + lane
            src = (kk[None, :]//32)*240*32 + row[:, None]*32 + kk[None, :]%32
            a = tl.load(HEADS_IN + src, valid, other=0)
            w = tl.load(W + kk[:, None]*1024 + col[None, :])
            total = tl.dot(a, w, total, out_dtype=tl.float32)
        result = (total + initial.to(tl.float32)).to(tl.float16)
    else:
        result = tl.full((BM, BN), 0, tl.float16)
        for part in range(4):
            total = tl.full((BM, BN), 0, tl.float32)
            for block in range(8):
                kk = part*256 + block*32 + lane
                src = (kk[None, :]//32)*240*32 + row[:, None]*32 + kk[None, :]%32
                a = tl.load(HEADS_IN + src, valid, other=0)
                w = tl.load(W + kk[:, None]*1024 + col[None, :])
                total = tl.dot(a, w, total, out_dtype=tl.float32)
            if part == 0:result = (total + initial.to(tl.float32)).to(tl.float16)
            else:result = _add_half(result, total.to(tl.float16))
    tl.store(OUT + offsets, result, valid)


def require_half(tensor, shape, device):
    if (tuple(tensor.shape) != tuple(shape) or tensor.device != device or
            tensor.device.type != "xpu" or tensor.dtype != torch.float16 or
            not tensor.is_contiguous()):
        raise ValueError("Current720 requires owned contiguous same-device FP16 operands")


def forward(module, mlp, counter, *, full_qkv=True, full_projection=True,
            native_norm=True, native_exp=True, bm=32, bn=64, denominator="ordered_fused"):
    import nr_backend.vit_block as vit
    from nr_backend.unround_policy import ENABLED
    from nr_backend.execution import current_arithmetic_backend
    if denominator not in ("ordered_fused", "fp32_reduction"):
        raise ValueError("Unknown owned global denominator")
    if "vit" not in ENABLED or current_arithmetic_backend() != "triton":
        raise RuntimeError("Owned global candidate is restricted to current720 unrounded fast")
    device = module.qkv_weight.device
    require_half(mlp, (240, 1024), device)
    counter.observe_layout("global ViT FFN -> QKV",mlp)
    for name, shape in (("qkv_weight",(1024,3072)),("query_scale",(32,)),
                        ("projection",(1024,1024)),("attn_skip",(1024,))):
        require_half(getattr(module,name),shape,device)
    q = torch.empty((32,240,32),device=device,dtype=torch.float16)
    k = torch.empty((32,256,32),device=device,dtype=torch.float16)
    v = torch.empty_like(k)
    # Grid covers256 so K/V padded16 are explicitly written; Q writes only240.
    counter.launch("vit_qkv", _qkv, (triton.cdiv(256,bm),32,3),
                   (mlp,module.qkv_weight,module.query_scale,q,k,v,bm,full_qkv,native_norm))
    key_transposed=k.transpose(-1,-2)
    counter.observe_layout("global ViT Q -> retained score provider",q)
    counter.observe_layout("global ViT K transpose -> retained score provider",key_transposed)
    scores = vit.sm89_f16_batched_dot(q,key_transposed)
    exponent = torch.empty_like(scores)
    if native_exp:
        counter.launch("vit_exp_native", _exp_vit, (triton.cdiv(1966080,512),),
                       (scores,exponent,1966080,512))
    else:
        counter.launch("vit_exp_reference", _exp_reference, (triton.cdiv(1966080,512),),
                       (scores,exponent,1966080,True,512))
    # Raw special exponent is the numerator operand, never normalized softmax.
    counter.observe_layout("global ViT exponent -> retained value provider",exponent)
    counter.observe_layout("global ViT V -> retained value provider",v)
    numerator = vit.sm89_f16_batched_dot(exponent,v)
    heads = torch.empty_like(numerator)
    counter.launch("vit_denominator_apply", _denominator_apply, (480,),
                   (exponent,numerator,heads,16,denominator=="ordered_fused"))
    counter.observe_layout("global ViT heads -> projection",heads)
    out = torch.empty_like(mlp)
    counter.launch("vit_projection", _projection,
                   (triton.cdiv(240,bm),triton.cdiv(1024,bn)),
                   (heads,module.projection,mlp,module.attn_skip,out,bm,bn,full_projection))
    return out


EXTRA = {
    __name__+"._qkv": (("Q","K","V"),()),
    __name__+"._denominator_apply": (("OUT",),()),
    __name__+"._projection": (("OUT",),()),
}
