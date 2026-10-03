"""Actual240 INT8 FFN: one row quantization, blocked full-K producer/consumer.

Both contract schedules cover K4096 exactly. Split4 stores INT32 partitions and
uses the accepted integer merge before the unchanged half residual epilogue.
There is no half hidden tensor, weight repack, reduced K or network bypass.
Only cold preflight may compile; the existing CompleteCounter owns zero-spill,
source/layout/binary admission, capture and serial lifecycle.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl
import int8_ffn_segment_rows_v1 as original
from audit_int8_ffn_complete_720_v1 import _require, _vit_merge_p4
from nr_backend.unround_policy import ENABLED


@triton.jit
def _expand_blocked(QX, SX, W, SW, SH, QH,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tl.static_assert((BK == 32) or (BK == 64) or (BK == 128))
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    lane = tl.arange(0, BK)
    total = tl.full((BM, BN), 0, tl.int32)
    for start in range(1024 // BK):
        kk = start * BK + lane
        a = tl.load(QX + row[:, None] * 1024 + kk[None, :],
                    row[:, None] < 240, other=0)
        w = tl.load(W + col[None, :] * 1024 + kk[:, None])
        total = tl.dot(a, w, total, out_dtype=tl.int32)
    sa = tl.load(SX + row, row < 240, other=1.)
    sw, sh = tl.load(SW + col), tl.load(SH + col)
    # Exact accepted expand epilogue: two scale multiplies, each existing
    # activation half boundary, then INT8 hidden quantization with frozen SH.
    x = ((total.to(tl.float32) * sa[:, None]) * sw[None, :]).to(tl.float16)
    t = tl.minimum(tl.maximum(x.to(tl.float32), -4.), 4.).to(tl.float16)
    p = tl.fma(-tl.abs(t), tl.full((), .055908203125, tl.float16),
               tl.full((), .447265625, tl.float16)).to(tl.float16)
    v = tl.fma(t, p, tl.full((), .89453125, tl.float16)).to(tl.float16)
    hidden = (x.to(tl.float32) * v.to(tl.float32)).to(tl.float16)
    tl.store(QH + row[:, None] * 4096 + col[None, :],
             original._q(hidden.to(tl.float32), sh[None, :]), row[:, None] < 240)


@triton.jit
def _contract_blocked(QH, W, SW, X, SKIP, PARTIAL, OUT,
                      PARTS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                      BK: tl.constexpr):
    tl.static_assert((PARTS == 1) or (PARTS == 4))
    tl.static_assert((BK == 32) or (BK == 64) or (BK == 128))
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    lane = tl.arange(0, BK)
    part = tl.program_id(2)
    total = tl.full((BM, BN), 0, tl.int32)
    for start in range(4096 // (PARTS * BK)):
        kk = part * (4096 // PARTS) + start * BK + lane
        a = tl.load(QH + row[:, None] * 4096 + kk[None, :],
                    row[:, None] < 240, other=0)
        w = tl.load(W + col[None, :] * 4096 + kk[:, None])
        total = tl.dot(a, w, total, out_dtype=tl.int32)
    offset = row[:, None] * 1024 + col[None, :]
    if PARTS == 1:
        x = tl.load(X + offset, row[:, None] < 240, other=0)
        sw, skip = tl.load(SW + col), tl.load(SKIP + col)
        initial = (x.to(tl.float32) * skip[None, :].to(tl.float32)).to(tl.float16)
        result = (total.to(tl.float32) * sw[None, :] + initial.to(tl.float32)).to(tl.float16)
        tl.store(OUT + offset, result, row[:, None] < 240)
    else:
        tl.store(PARTIAL + part * 240 * 1024 + offset, total, row[:, None] < 240)


@triton.jit
def _contract_full_blocked(QH, W, SW, X, SKIP, OUT,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    _contract_blocked(QH, W, SW, X, SKIP, OUT, OUT, 1, BM, BN, BK)


@triton.jit
def _contract_p4_blocked(QH, W, PARTIAL,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # PARTS=4 removes the entire half epilogue at compile time. The placeholders
    # are never runtime reads/stores; this launched ABI owns only PARTIAL.
    _contract_blocked(QH, W, PARTIAL, PARTIAL, PARTIAL, PARTIAL, PARTIAL, 4, BM, BN, BK)


def specs(x, qx, sx, qh, out, partial, packed, *, bm, bn, bk, parts):
    """Shared actual-Tensor call plan used by cold admission and live capture."""
    if type(bm) is not int or bm not in (16, 32) or type(bn) is not int or bn not in (32, 64):
        raise ValueError('Reviewed actual240 BM/BN required')
    if type(bk) is not int or bk not in (32, 64, 128) or type(parts) is not int or parts not in (1, 4):
        raise ValueError('Full-K blocked INT32 contract required')
    we, se, sh, wc, sc, skip = packed
    result = [
        ('vit_ffn_entry', original._entry, (240,), (x, qx, sx, 240, 1024)),
        ('vit_ffn_expand_blocked', _expand_blocked,
         (triton.cdiv(240, bm), 4096 // bn), (qx, sx, we, se, sh, qh, bm, bn, bk)),
    ]
    if parts == 4:
        result.append(('vit_ffn_contract_blocked', _contract_p4_blocked,
                       (triton.cdiv(240, bm), 1024 // bn, 4),
                       (qh, wc, partial, bm, bn, bk)))
        result.append(('vit_ffn_merge', _vit_merge_p4, (480,), (partial, sc, x, skip, out)))
    else:
        result.append(('vit_ffn_contract_blocked', _contract_full_blocked,
                       (triton.cdiv(240, bm), 1024 // bn, 1),
                       (qh, wc, sc, x, skip, out, bm, bn, bk)))
    return tuple(result)


def preflight(counter, empty, prepare, *, bm, bn, bk, parts):
    x = empty((240, 1024))
    qx = empty((240, 1024), torch.int8)
    sx = empty((240,), torch.float32)
    qh = empty((240, 4096), torch.int8)
    out = empty((240, 1024))
    partial = empty((4, 240, 1024), torch.int32) if parts == 4 else None
    for packed in counter.stack.int8_vit.packed:
        for label, jit, grid, args in specs(x, qx, sx, qh, out, partial, packed,
                                            bm=bm, bn=bn, bk=bk, parts=parts):
            prepare(label, jit, grid, args)


def forward(stack, index, x, counter, *, bm=32, bn=64, bk=64, parts=1):
    _require(x, (240, 1024))
    if 'vit' not in ENABLED:
        raise RuntimeError('Exact ViT path is not a throughput candidate')
    if stack is not counter.stack or type(index) is not int or not 0 <= index < 8:
        raise RuntimeError('Use the actual owned eight-block ViT consumer')
    qx = torch.empty((240, 1024), device=x.device, dtype=torch.int8)
    sx = torch.empty((240,), device=x.device, dtype=torch.float32)
    qh = torch.empty((240, 4096), device=x.device, dtype=torch.int8)
    out = torch.empty_like(x)
    partial = torch.empty((4, 240, 1024), device=x.device, dtype=torch.int32) if parts == 4 else None
    for label, jit, grid, args in specs(x, qx, sx, qh, out, partial, stack.int8_vit.packed[index],
                                       bm=bm, bn=bn, bk=bk, parts=parts):
        counter.launch(label, jit, grid, args)
    # This tuple is consumed by actual fullsize_session.vforward's FFN -> QKV.
    return out, qh


EXTRA = {
    __name__ + '._expand_blocked': (('QH',), ()),
    __name__ + '._contract_full_blocked': (('OUT',), ()),
    __name__ + '._contract_p4_blocked': (('PARTIAL',), ()),
}
