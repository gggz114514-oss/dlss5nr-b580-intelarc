"""Isolated W8A8->E4M3 producer and E4M3->C32 MLP consumer kernels."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _round_fp8_half(x):
    bits = x.to(tl.uint16, bitcast=True).to(tl.int32)
    sign = bits & 0x8000
    raw = bits & 0x7fff
    mag = tl.minimum(raw, 0x5f00)
    exponent = mag >> 10
    sig = (mag & 1023) + tl.where(exponent != 0, 1024, 0)
    shift = tl.maximum(16 - tl.maximum(exponent, 1), 1)
    quotient = sig >> shift
    remainder = sig - (quotient << shift)
    midpoint = 1 << (shift - 1)
    q = quotient + ((remainder > midpoint) | ((remainder == midpoint) & ((quotient & 1) != 0))).to(tl.int32)
    sub = (q.to(tl.float32) * 0.001953125).to(tl.float16).to(tl.uint16, bitcast=True).to(tl.int32)
    normal = (mag + 63 + ((mag >> 7) & 1)) & 0x7f80
    result = tl.where(mag < 0x2400, sub, normal) | sign
    result = tl.where(raw > 0x7c00, 0x7f80 | sign, result)
    return result.to(tl.uint16).to(tl.float16, bitcast=True)


@triton.jit
def _half_to_e4m3_byte(x):
    rounded = _round_fp8_half(x)
    bits = rounded.to(tl.uint16, bitcast=True).to(tl.int32)
    sign = (bits >> 8) & 0x80
    magnitude = bits & 0x7fff
    exponent = (magnitude >> 10) & 31
    subnormal = (tl.abs(rounded.to(tl.float32)) * 512.0).to(tl.int32)
    normal = ((exponent - 8) << 3) | ((magnitude & 1023) >> 7)
    code = tl.where(exponent == 0, sign,
                    tl.where(exponent < 9, sign | subnormal,
                             tl.where(exponent >= 31, sign | 0x7f, sign | normal)))
    return code.to(tl.uint8)


@triton.jit
def _decode_e4m3_byte(code):
    value = code.to(tl.int32)
    sign = (value & 0x80) != 0
    exponent = (value >> 3) & 15
    mantissa = value & 7
    sub = mantissa.to(tl.float32) * 0.001953125
    normal = (1.0 + mantissa.to(tl.float32) * 0.125) * tl.exp2(exponent.to(tl.float32) - 7.0)
    decoded = tl.where(exponent == 0, sub, normal)
    return tl.where(sign, -decoded, decoded).to(tl.float16)


@triton.jit
def _w8a8_fp8(A, W, SW, INITIAL, OUT, RAW_OUT, ROW_SCALE,
              M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
              HAS_INITIAL: tl.constexpr, DEBUG: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    columns = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    a = tl.load(A + rows[:, None] * K + kk[None, :],
                (rows[:, None] < M) & (kk[None, :] < K), other=0).to(tl.float32)
    maximum = tl.max(tl.abs(a), 1)
    scale = tl.where(maximum > 0, tl.div_rn(maximum, 127.0), 1.0)
    normalized = tl.div_rn(a, scale[:, None])
    magnitude = tl.minimum(tl.floor(tl.abs(normalized) + 0.5), 127.0)
    qa = (magnitude * tl.where(normalized < 0, -1.0, 1.0)).to(tl.int8)
    qw = tl.load(W + columns[None, :] * K + kk[:, None],
                 (columns[None, :] < N) & (kk[:, None] < K), other=0)
    total = tl.dot(qa, qw, out_dtype=tl.int32)
    sw = tl.load(SW + columns, columns < N, other=1.0)
    value = (total.to(tl.float32) * scale[:, None]) * sw[None, :]
    offsets = rows[:, None] * N + columns[None, :]
    valid = (rows[:, None] < M) & (columns[None, :] < N)
    if HAS_INITIAL:
        value += tl.load(INITIAL + offsets, valid, other=0).to(tl.float32)
    half_value = value.to(tl.float16)
    tl.store(OUT + offsets, _half_to_e4m3_byte(half_value), valid)
    if DEBUG:
        tl.store(RAW_OUT + offsets, half_value, valid)
        tl.store(ROW_SCALE + rows, scale, rows < M)


@triton.jit
def _c32_from_e4m3(X, EXPAND, CONTRACT, SCALE, LUT, OUT,
                   H: tl.constexpr, W: tl.constexpr,
                   PAD_H: tl.constexpr, PAD_W: tl.constexpr,
                   SY: tl.constexpr, SX: tl.constexpr,
                   M: tl.constexpr, BM: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    lane = tl.arange(0, 32)
    y = row // PAD_W
    x = row % PAD_W
    source_y = y - SY
    source_x = x - SX
    inside = (source_y >= 0) & (source_y < H) & (source_x >= 0) & (source_x < W)
    source_row = source_y * W + source_x
    code = tl.load(X + source_row[:, None] * 32 + lane[None, :],
                   inside[:, None] & (row[:, None] < M), other=0)
    raw = _decode_e4m3_byte(code)
    xq = _round_fp8_half(raw)
    contracted = tl.full((BM, 32), 0.0, tl.float32)
    for part in range(4):
        expand_weight = tl.load(EXPAND + lane[:, None] * 128 + part * 32 + lane[None, :])
        expanded = tl.dot(xq, expand_weight, out_dtype=tl.float32)
        lut_index = expanded.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.int32)
        hidden = tl.load(LUT + lut_index).to(tl.float16, bitcast=True)
        contract_weight = tl.load(CONTRACT + (part * 32 + lane[:, None]) * 32 + lane[None, :])
        contracted = tl.dot(hidden, contract_weight, contracted, out_dtype=tl.float32)
    skip_scale = tl.load(SCALE + lane)
    initial = (raw.to(tl.float32) * skip_scale[None, :].to(tl.float32)).to(tl.float16)
    result = (contracted + initial.to(tl.float32)).to(tl.float16)
    tl.store(OUT + row[:, None] * 32 + lane[None, :], result, row[:, None] < M)


@triton.jit
def _decode_grid_kernel(X, Y, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(X + i, i < N, other=0)
    tl.store(Y + i, _decode_e4m3_byte(value), i < N)


def w8a8_fp8(a, packed_weight, weight_scale, *, initial=None, debug=False,
              bm=16, bn=64, warps=4):
    if a.device.type != "xpu" or a.dtype not in (torch.float16, torch.float32):
        raise ValueError("Expected finite FP16/FP32 XPU activations")
    if a.shape[-1] != 32 or packed_weight.dtype != torch.int8 or packed_weight.ndim != 2:
        raise ValueError("Expected K=32 prepacked INT8 weights")
    n, k = packed_weight.shape
    if (k, n) != (32, 32) or weight_scale.shape != (n,) or weight_scale.dtype != torch.float32:
        raise ValueError("Expected the captured 32x32 packed weight and column scales")
    shape = (*a.shape[:-1], n)
    if initial is not None and (tuple(initial.shape) != shape or initial.device != a.device):
        raise ValueError("Initial residual shape mismatch")
    aa = a.half().contiguous().reshape(-1, k)
    ini = aa if initial is None else initial.half().contiguous().reshape(-1, n)
    m = aa.shape[0]
    output = torch.empty(shape, device=a.device, dtype=torch.uint8)
    raw = torch.empty(shape, device=a.device, dtype=torch.float16) if debug else output
    row_scale = torch.empty((m,), device=a.device, dtype=torch.float32) if debug else weight_scale
    kernel = _w8a8_fp8[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        aa, packed_weight, weight_scale, ini, output, raw, row_scale,
        m, n, k, initial is not None, debug, bm, bn, max(32, triton.next_power_of_2(k)),
        num_warps=warps, enable_fp_fusion=False)
    return output, kernel, (raw if debug else None), (row_scale if debug else None)


def c32_from_e4m3(codes, expansion, contraction, skip_scale, lut,
                   *, source_height=256, source_width=448, sy=4, sx=4,
                   bm=32, warps=4, stages=1):
    if codes.device.type != "xpu" or codes.dtype != torch.uint8 or tuple(codes.shape) != (source_height, source_width, 32):
        raise ValueError("Expected captured HWC32 E4M3 byte input")
    pad_h, pad_w = source_height + 2 * sy, source_width + 2 * sx
    rows = pad_h * pad_w
    out = torch.empty((pad_h, pad_w, 32), device=codes.device, dtype=torch.float16)
    kernel = _c32_from_e4m3[(triton.cdiv(rows, bm),)](
        codes, expansion, contraction, skip_scale, lut, out,
        source_height, source_width, pad_h, pad_w, sy, sx, rows, bm,
        num_warps=warps, num_stages=stages, enable_fp_fusion=False)
    return out, kernel


def decode_e4m3(codes):
    if codes.device.type != "xpu" or codes.dtype != torch.uint8:
        raise ValueError("Expected XPU E4M3 byte codes")
    source = codes.contiguous()
    out = torch.empty(source.shape, device=source.device, dtype=torch.float16)
    count = source.numel()
    kernel = _decode_grid_kernel[(triton.cdiv(count, 512),)](
        source, out, count, 512, enable_fp_fusion=False)
    return out, kernel
