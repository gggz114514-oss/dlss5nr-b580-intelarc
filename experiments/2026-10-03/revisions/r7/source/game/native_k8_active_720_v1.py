"""720p fast-line experiment: remove the two *active* K8 reference dots.

Pre uses the owned provider projection. Post bypasses that provider in the
installed post-attention combo, so its head injection must be changed there.
This changes numerical behavior and must never be installed on the exact path.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager

import torch
import triton
import triton.language as tl

import native_k8_fp16_v1 as pre_kernel
import post_attention_k8_combined_v1 as active_post


@triton.jit
def _post_half_dot(A, W, OUT, M: tl.constexpr, WIDTH: tl.constexpr,
                   A0: tl.constexpr, A1: tl.constexpr, A2: tl.constexpr,
                   W0: tl.constexpr, W1: tl.constexpr,
                   BM: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    kk = tl.arange(0, 32)
    cols = tl.arange(0, 16)
    a = tl.load(A + (rows // WIDTH)[:, None] * A0 +
                (rows % WIDTH)[:, None] * A1 + kk[None, :] * A2,
                rows[:, None] < M, other=0)
    w = tl.load(W + kk[:, None] * W0 + cols[None, :] * W1,
                cols[None, :] < 8, other=0)
    value = tl.dot(a, w, out_dtype=tl.float32)
    tl.store(OUT + rows[:, None] * 4 + cols[None, :], value.to(tl.float16),
             (rows[:, None] < M) & (cols[None, :] < 4))


def post_head(a: torch.Tensor, weight: torch.Tensor,
              output_size: tuple[int, int]) -> torch.Tensor:
    if (tuple(a.shape) != (768, 1280, 32) or not a.is_contiguous()
            or tuple(weight.shape) != (32, 8) or
            tuple(weight.stride()) != (8, 1) or
            tuple(output_size) != (720, 1280) or
            a.dtype != torch.float16 or weight.dtype != torch.float16 or
            a.device.type != "xpu" or weight.device != a.device):
        raise ValueError("Native post K8 requires the active 720p post contract")
    out_h, out_w = output_size
    result = torch.empty((out_h, out_w, 4), device=a.device,
                         dtype=torch.float16)
    _post_half_dot[(triton.cdiv(out_h * out_w, 32),)](
        a, weight, result, out_h * out_w, out_w, *a.stride(),
        *weight.stride(), 32, num_warps=4, enable_fp_fusion=False)
    return result


@contextmanager
def installed(session, arm: str):
    if arm not in ("pre", "post", "both"):
        raise ValueError("Expected pre, post or both")
    if session._stack.provider.mode != "fp16_xmx":
        raise RuntimeError("Active FP16 provider required")
    original = active_post._contiguous_cropped_head

    def scoped_post(a, weight, output_size):
        calls["post"] += 1
        return post_head(a, weight, output_size)

    with ExitStack() as scopes:
        if arm in ("pre", "both"):
            # Share the provider's live dispatch counters with the probe so
            # each reset/history graph capture can be checked independently.
            calls = scopes.enter_context(
                pre_kernel.installed(session, "pre-only"))
        else:
            calls = {"pre": 0, "post": 0}
        if arm in ("post", "both"):
            if active_post._contiguous_cropped_head is not original:
                raise RuntimeError("Post K8 injection was already changed")
            active_post._contiguous_cropped_head = scoped_post
            try:
                yield calls
            finally:
                if active_post._contiguous_cropped_head is not scoped_post:
                    raise RuntimeError("Post K8 injection changed during test")
                active_post._contiguous_cropped_head = original
        else:
            yield calls
