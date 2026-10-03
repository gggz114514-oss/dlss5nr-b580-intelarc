"""Read 3D FP16 attention operands by stride, retaining the original DPAS reduction."""
import torch
import triton
import triton.language as tl
from fused_cached_matrices_v2 import FusedCachedMatrices


@triton.jit
def _matmul(A, W, INITIAL, OUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            AB: tl.constexpr, AM: tl.constexpr, AK: tl.constexpr,
            WB: tl.constexpr, WK: tl.constexpr, WN: tl.constexpr,
            IB: tl.constexpr, IM: tl.constexpr, IN: tl.constexpr,
            INITIALIZED: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    batch = tl.program_id(2)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    total = tl.full((BM, BN), 0, tl.float32)
    for start in range(0, tl.cdiv(K, BK)):
        k = start * BK + kk
        av = tl.load(A + batch * AB + rows[:, None] * AM + k[None, :] * AK,
                     (rows[:, None] < M) & (k[None, :] < K), other=0)
        wv = tl.load(W + batch * WB + k[:, None] * WK + cols[None, :] * WN,
                     (cols[None, :] < N) & (k[:, None] < K), other=0)
        total = tl.dot(av, wv, total, out_dtype=tl.float32)
    value = total.to(tl.float32)
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    if INITIALIZED:
        value += tl.load(INITIAL + batch * IB + rows[:, None] * IM + cols[None, :] * IN,
                         valid, other=0).to(tl.float32)
    tl.store(OUT + (batch * M + rows[:, None]) * N + cols[None, :], value.to(tl.float16), valid)


def dot(a, w, *, initial=None):
    if a.ndim != 3 or w.ndim != 3 or a.shape[0] != w.shape[0] or a.shape[-1] != w.shape[-2]:
        raise ValueError('Expected compatible rank-three batched matrices')
    if a.device.type != 'xpu' or a.device != w.device or min(*a.shape, *w.shape) <= 0:
        raise ValueError('Expected nonempty same-device XPU matrices')
    shape = (a.shape[0], a.shape[1], w.shape[2])
    if initial is not None and (initial.device != a.device or initial.shape != shape):
        raise ValueError('Invalid initial accumulator')
    aa, ww = a.half(), w.half()
    ini = aa if initial is None else initial.half()
    b, m, k = aa.shape
    out = torch.empty(shape, dtype=torch.float16, device=a.device)
    compiled = _matmul[(triton.cdiv(m, 16), triton.cdiv(shape[2], 32), b)](
        aa, ww, ini, out, m, shape[2], k, *aa.stride(), *ww.stride(), *ini.stride(),
        initial is not None, 16, 32, 32, num_warps=4, enable_fp_fusion=False)
    return out, compiled


class StridedMatrices(FusedCachedMatrices):
    """Same dense arithmetic and graph contracts; higher-rank calls keep the old path."""
    def batched(self, a, w, *, initial=None, **kwargs):
        if self.mode == 'baseline' or a.ndim != 3 or w.ndim != 3:
            return super().batched(a, w, initial=initial, **kwargs)
        out, compiled = dot(a, w, initial=initial)
        self.record('fp16_batched')
        self.record('strided_batched')
        for value in (a, w, initial):
            if value is not None and value.dtype == torch.float16 and not value.is_contiguous():
                self.calls['avoided_contiguous_bytes'] = self.calls.get('avoided_contiguous_bytes', 0) + value.numel() * value.element_size()
        if 'strided_batched' not in self.compiled:
            self.compiled['strided_batched'] = compiled
        return out
