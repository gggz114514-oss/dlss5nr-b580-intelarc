"""Explicit dense launch geometry; reuse the frozen arithmetic kernel.

FP16 always retains BK32. INT8 K is bounded to 4096: every partial integer sum
fits INT32, followed by the same FP32 conversion and ordered scale multiplies.
No automatic tuning during inference or graph capture.
"""
import torch
import triton
from fast_matrices_v3 import _matmul, quantize


def dot(a, w, *, initial=None, int8=False, packed=None, tile=(16, 32, 32, 4)):
    if a.device.type != 'xpu' or a.device != w.device or a.ndim < 2 or w.ndim != 2 or a.shape[-1] != w.shape[0]:
        raise ValueError('Expected compatible nonbatched XPU matrices')
    bm, bn, bk, warps = tile
    if bm not in (16, 32, 64) or bn not in (32, 64, 128) or bk not in (32, 64, 128) or warps != 4:
        raise ValueError('Unvalidated launch geometry')
    if not int8 and bk != 32:
        raise ValueError('FP16 reduction remains BK32')
    shape = (*a.shape[:-1], w.shape[-1])
    if initial is not None and (initial.shape != shape or initial.device != a.device):
        raise ValueError('Invalid initial accumulator')
    aa = a.half().contiguous().reshape(-1, a.shape[-1])
    ww = w.half().contiguous()
    m, k = aa.shape
    n = w.shape[-1]
    if min(m, k, n) <= 0 or (int8 and k > 4096):
        raise ValueError('Unsupported matrix extent')
    if int8:
        aa, sa, _ = quantize(aa)
        if packed is None:
            ww, sw, _ = quantize(ww, columns=True)
        else:
            ww, sw = packed
            if ww.shape != (n, k) or sw.shape != (n,) or ww.dtype != torch.int8 or sw.dtype != torch.float32 or ww.device != a.device or sw.device != a.device or not ww.is_contiguous() or not sw.is_contiguous():
                raise ValueError('Invalid packed weights')
    else:
        sa = sw = aa
    ini = aa if initial is None else initial.half().contiguous()
    out = torch.empty(shape, device=a.device, dtype=torch.float16)
    kernel = _matmul[(triton.cdiv(m, bm), triton.cdiv(n, bn), 1)](
        aa, ww, sa, sw, ini, out, m, n, k, initial is not None, False, int8,
        bm, bn, bk, num_warps=warps, enable_fp_fusion=False)
    return out, kernel
