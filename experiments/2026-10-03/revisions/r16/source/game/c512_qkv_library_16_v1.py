"""Isolated fast-line QKV GEMM candidate for all sixteen C512 blocks.

Replaces only the active provider's owned 512x1536 QKV matrices. The model's
attention/packing/projection, FP8 boundaries and exact branch are untouched.
The library implementation is tested as a scheduler alternative, not assumed
to use a particular hardware instruction without ISA evidence.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch


@contextmanager
def installed(session):
    stack = session._stack
    provider = stack.provider
    if provider.mode != "fp16_xmx" or "dense" in provider.__dict__:
        raise RuntimeError("Expected unmodified active FP16 provider")
    targets = {}
    for family in ("encoder512", "decoder512"):
        blocks = getattr(stack.model, family)
        if len(blocks) != 8:
            raise RuntimeError(f"Unexpected {family} block count")
        for index, block in enumerate(blocks):
            weight = block.attention.qkv
            if tuple(weight.shape) != (512, 1536) or weight.dtype != torch.float16:
                raise RuntimeError(f"Unexpected {family}[{index}] QKV weight")
            targets[id(weight)] = (weight, f"{family}_{index}")
    if len(targets) != 16:
        raise RuntimeError("C512 QKV weights are not distinct")
    calls = {name: 0 for _, name in targets.values()}
    original = provider.dense

    def dense(self, a, w, *, chunk_k, initial=None, **kwargs):
        target = targets.get(id(w))
        if target is None:
            return original(a, w, chunk_k=chunk_k, initial=initial, **kwargs)
        if (target[0] is not w or self.mode != "fp16_xmx" or chunk_k != 16 or
                initial is not None or kwargs or a.device.type != "xpu" or
                a.dtype != torch.float16 or a.shape[-1] != 512 or
                w.device != a.device):
            raise RuntimeError("C512 QKV left the reviewed FP16 GEMM contract")
        output = torch.mm(a.contiguous().reshape(-1, 512), w)
        self.record("fp16_dense")
        calls[target[1]] += 1
        return output.reshape(*a.shape[:-1], 1536)

    replacement = MethodType(dense, provider)
    provider.dense = replacement
    try:
        yield calls
    finally:
        if provider.dense is not replacement:
            raise RuntimeError("C512 provider override changed during experiment")
        del provider.dense
