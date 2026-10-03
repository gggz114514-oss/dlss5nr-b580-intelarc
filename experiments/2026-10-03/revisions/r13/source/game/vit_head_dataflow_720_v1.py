"""Isolated 240-token ViT QKV/attention/projection dataflow candidate.

The active fullsize_session_v1.vforward remains the reference.  This file is
not installed by the game.  It keeps two separately rounded K512 QKV products,
the installed ViT attention (including its exponent and padded-key correction),
and four independently rounded K256 projection parts merged in order.  QKV is
written directly in head-major form and projection reads head-major attention
without materializing a token-major copy or four full-size projection results.

Kernel FP16/NaN edge behaviour and performance are UNVERIFIED until the probe
runs on the B580.  Do not install merely because finite-output errors are small.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from nr_backend.triton_attention_normalize import _nan_left, rsqrt_half_clamped
from nr_backend.triton_cubic_fp8 import _half_fma_value
from nr_backend.triton_fp8 import _round_fp8_half


TOKENS = 240
HEADS = 32
CHANNELS = 32


@triton.jit
def _qkv_head_major(X, W, SCALE, Q, K, V,
                    M: tl.constexpr, BM: tl.constexpr, ROUND_VIT: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    head = tl.program_id(1)
    family = tl.program_id(2)
    lane = tl.arange(0, 32)
    out_col = head * 96 + family * 32 + lane

    # Each half has its own FP32 dot accumulator and FP16 store boundary.
    first = tl.full((BM, 32), 0, tl.float32)
    second = tl.full((BM, 32), 0, tl.float32)
    for block in range(16):
        kk = block * 32 + lane
        xa = tl.load(X + row[:, None] * 1024 + kk[None, :],
                     row[:, None] < M, other=0)
        xb = tl.load(X + row[:, None] * 1024 + 512 + kk[None, :],
                     row[:, None] < M, other=0)
        wa = tl.load(W + kk[:, None] * 3072 + out_col[None, :])
        wb = tl.load(W + (512 + kk[:, None]) * 3072 + out_col[None, :])
        first = tl.dot(xa, wa, first, out_dtype=tl.float32)
        second = tl.dot(xb, wb, second, out_dtype=tl.float32)
    a = first.to(tl.float16)
    b = second.to(tl.float16)
    z = _nan_left((a.to(tl.float32) + b.to(tl.float32)).to(tl.float16), a, b)

    if family < 2:
        # The installed C32 normalization squares/FMA/XOR tree, operating on
        # the register tile instead of materializing [T,32,3,32] and Q/K.
        eight = tl.arange(0, 8)
        indexes = tl.broadcast_to(eight[None, :], (BM, 8))
        x0 = tl.gather(z, indexes, 1)
        x8 = tl.gather(z, indexes + 8, 1)
        x16 = tl.gather(z, indexes + 16, 1)
        x24 = tl.gather(z, indexes + 24, 1)
        aa = (x16.to(tl.float32) * x16.to(tl.float32)).to(tl.float16)
        aa = _nan_left(_half_fma_value(x0, x0, aa), x0, aa)
        bb = (x24.to(tl.float32) * x24.to(tl.float32)).to(tl.float16)
        bb = _nan_left(_half_fma_value(x8, x8, bb), x8, bb)
        total = _nan_left((aa.to(tl.float32) + bb.to(tl.float32)).to(tl.float16), aa, bb)
        for mask in tl.static_range(3):
            other = tl.gather(total, tl.broadcast_to(
                (eight ^ (4 >> mask))[None, :], (BM, 8)), 1)
            total = _nan_left((total.to(tl.float32) + other.to(tl.float32)).to(tl.float16), total, other)
        denominator = tl.gather(total, tl.full((BM, 1), 0, tl.int32), 1)
        norm = rsqrt_half_clamped(denominator)
        normalized = _nan_left((z.to(tl.float32) * norm.to(tl.float32)).to(tl.float16),
                               z, norm)
        if family == 0:
            scaled = (normalized.to(tl.float32) * 5.65625).to(tl.float16)
            scale = tl.load(SCALE + head)
            z = _nan_left((scaled.to(tl.float32) * scale.to(tl.float32)).to(tl.float16),
                          scaled, scale)
        else:
            z = normalized

    if ROUND_VIT:
        z = _round_fp8_half(z)
    destination = tl.where(family == 0, Q, tl.where(family == 1, K, V))
    offset = (head * M + row[:, None]) * 32 + lane[None, :]
    tl.store(destination + offset, z, row[:, None] < M)


@triton.jit
def _project_head_major(HEADS_IN, WEIGHT, INITIAL, OUT,
                        M: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    lane = tl.arange(0, 32)
    valid = (row[:, None] < M) & (col[None, :] < 1024)
    skip = tl.load(INITIAL + row[:, None] * 1024 + col[None, :], valid, other=0)
    result = tl.full((BM, BN), 0, tl.float16)
    for part in range(4):
        accumulator = tl.full((BM, BN), 0, tl.float32)
        for block in range(8):
            kk = part * 256 + block * 32 + lane
            # The attention output is physically [head, token, channel].
            source = (kk[None, :] // 32) * M * 32 + row[:, None] * 32 + kk[None, :] % 32
            x = tl.load(HEADS_IN + source, row[:, None] < M, other=0)
            w = tl.load(WEIGHT + kk[:, None] * 1024 + col[None, :],
                        col[None, :] < 1024, other=0)
            accumulator = tl.dot(x, w, accumulator, out_dtype=tl.float32)
        if part == 0:
            result = (accumulator + skip.to(tl.float32)).to(tl.float16)
        else:
            partial = accumulator.to(tl.float16)
            result = _nan_left((result.to(tl.float32) + partial.to(tl.float32)).to(tl.float16),
                               result, partial)
    tl.store(OUT + row[:, None] * 1024 + col[None, :], result, valid)


def _require_half_xpu(name, tensor, shape):
    if (tuple(tensor.shape) != shape or tensor.dtype != torch.float16 or
            tensor.device.type != "xpu" or not tensor.is_contiguous()):
        raise ValueError(f"{name}: expected contiguous XPU FP16 {shape}")


def _require_block(module, mlp):
    _require_half_xpu("mlp", mlp, (TOKENS, 1024))
    for name, shape in (("qkv_weight", (1024, 3072)),
                        ("query_scale", (32,)), ("projection", (1024, 1024)),
                        ("attn_skip", (1024,))):
        tensor = getattr(module, name)
        _require_half_xpu(name, tensor, shape)
        if tensor.device != mlp.device:
            raise ValueError(f"{name}: device differs from mlp")


@dataclass
class Boundaries:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended_heads: torch.Tensor
    output: torch.Tensor


def forward_from_mlp(module, mlp, *, return_boundaries=False):
    """Run the active 720p ViT tail without changing global model methods.

    Caller must already be inside the fast session's installed and arithmetic
    scopes.  A guard rejects other token counts and non-XPU/half contracts.
    """
    import nr_backend.vit_block as vit
    from nr_backend.execution import current_arithmetic_backend
    from nr_backend.unround_policy import ENABLED

    _require_block(module, mlp)
    if current_arithmetic_backend() != "triton":
        raise RuntimeError("ViT dataflow candidate requires active Triton backend")
    q, k, v = (torch.empty((HEADS, TOKENS, CHANNELS), device=mlp.device,
                           dtype=torch.float16) for _ in range(3))
    _qkv_head_major[(triton.cdiv(TOKENS, 16), HEADS, 3)](
        mlp, module.qkv_weight, module.query_scale, q, k, v,
        TOKENS, 16, "vit" not in ENABLED, num_warps=4, num_stages=1,
        enable_fp_fusion=False)
    # Keep the installed attention; in particular, do not substitute softmax
    # or remove its padded-key correction for 240 -> 256 keys.
    heads = vit.vit_attention(q, k, v)
    skip = (mlp * module.attn_skip).half()
    out = torch.empty((TOKENS, 1024), device=mlp.device, dtype=torch.float16)
    _project_head_major[(triton.cdiv(TOKENS, 16), triton.cdiv(1024, 32))](
        heads.contiguous(), module.projection, skip.contiguous(), out, TOKENS, 16, 32,
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    out = vit.q(out)
    if return_boundaries:
        return Boundaries(q, k, v, heads, out)
    return out


@contextmanager
def installed(session):
    """Opt-in, session-local experiment hook for full-frame validation later.

    It wraps the active session's own installed scope because that scope sets
    VitBlock.forward on entry.  Nothing is patched at import or in the game.
    Keep this hook out of production until D: probe and 13-frame replay pass.
    """
    import nr_backend.vit_block as vit
    from nr_backend.execution import current_arithmetic_backend

    stack = session._stack
    if stack.provider.mode != "fp16_xmx" or len(stack.model.vit) != 8:
        raise RuntimeError("Expected active FP16 provider and eight ViT blocks")
    original_scope = session._installed
    counts = {index: 0 for index in range(8)}
    owned = {id(module): index for index, module in enumerate(stack.model.vit)}

    @contextmanager
    def wrapped_scope():
        with original_scope():
            existing_forward = vit.VitBlock.forward

            def replacement(module, x):
                index = owned.get(id(module))
                if index is None or tuple(x.shape) != (TOKENS, 1024):
                    raise RuntimeError("ViT candidate left the reviewed 720p/240-token route")
                if current_arithmetic_backend() != "triton":
                    raise RuntimeError("ViT candidate lost the Triton arithmetic scope")
                x = vit.q(x)
                mlp = vit.q(stack.int8_vit.ffn(stack.int8_vit.modules[id(module)], x)[0])
                output = forward_from_mlp(module, mlp)
                counts[index] += 1
                session._counts["vit"] += 1
                return output

            vit.VitBlock.forward = replacement
            try:
                yield
            finally:
                unchanged = vit.VitBlock.forward is replacement
                vit.VitBlock.forward = existing_forward
                if not unchanged:
                    raise RuntimeError("ViT experiment scope was replaced unexpectedly")

    session._installed = wrapped_scope
    try:
        yield counts
    finally:
        unchanged = session._installed is wrapped_scope
        session._installed = original_scope
        if not unchanged:
            raise RuntimeError("ViT session scope was replaced unexpectedly")
