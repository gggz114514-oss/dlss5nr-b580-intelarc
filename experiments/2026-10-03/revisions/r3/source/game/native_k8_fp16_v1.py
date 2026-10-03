"""Session-local native FP16 K8 candidates for the installed RE8 matrix provider.

Only the two owned buffers are eligible. The pre result remains unquantized
FP16 for the existing MLP/residual path; the post result retains all eight
channels, including channel 3 used by temporal history. No module, provider
class, model forward, or global arithmetic dispatch is replaced.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch
import triton
import triton.language as tl


ARMS = {
    "pre-only": frozenset(("pre",)),
    "post-only": frozenset(("post",)),
    "both": frozenset(("pre", "post")),
}


@triton.jit
def _native_k8_fp16(A, W, OUT, H: tl.constexpr, WIDTH: tl.constexpr,
                    K: tl.constexpr, N: tl.constexpr,
                    A0: tl.constexpr, A1: tl.constexpr, A2: tl.constexpr,
                    W0: tl.constexpr, W1: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    kk = tl.arange(0, BK)
    a = tl.load(A + (rows // WIDTH)[:, None] * A0 +
                (rows % WIDTH)[:, None] * A1 + kk[None, :] * A2,
                (rows[:, None] < H * WIDTH) & (kk[None, :] < K), other=0)
    w = tl.load(W + kk[:, None] * W0 + cols[None, :] * W1,
                (kk[:, None] < K) & (cols[None, :] < N), other=0)
    value = tl.dot(a, w, out_dtype=tl.float32)
    tl.store(OUT + rows[:, None] * N + cols[None, :], value.to(tl.float16),
             (rows[:, None] < H * WIDTH) & (cols[None, :] < N))


def dot(a: torch.Tensor, w: torch.Tensor, *, site: str) -> torch.Tensor:
    """One full HWC projection, with FP32 dot accumulation and one FP16 store."""
    expected = {"pre": (16, 32), "post": (32, 8)}
    if site not in expected or tuple(w.shape) != expected[site]:
        raise ValueError("Unexpected owned K8 projection")
    if (a.ndim != 3 or a.shape[-1] != w.shape[0] or a.device.type != "xpu"
            or a.device != w.device or a.dtype != torch.float16
            or w.dtype != torch.float16 or a.shape[0] < 1 or a.shape[1] < 1):
        raise ValueError("Native K8 requires nonempty same-device FP16 HWC input and weight")
    h, width, k = a.shape
    n = w.shape[1]
    out = torch.empty((h, width, n), dtype=torch.float16, device=a.device)
    _native_k8_fp16[(triton.cdiv(h * width, 32),)](
        a, w, out, h, width, k, n, *a.stride(), *w.stride(), 32, 32, 32,
        num_warps=4, enable_fp_fusion=False)
    return out


@contextmanager
def installed(session, arm: str):
    """Patch only this session's provider instance, before its graph captures.

    ``session`` is the FullsizeSession inside a newly selected game mode. The
    existing provider installation still owns nr_backend.triton_math.fused_dot;
    all non-whitelisted calls delegate to its original bound ``dense`` method.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown native K8 arm: {arm}")
    stack = session._stack
    provider, model = stack.provider, stack.model
    if provider.mode != "fp16_xmx" or "dense" in provider.__dict__:
        raise RuntimeError("Expected the unmodified installed FP16 XMX provider instance")
    weights = {"pre": model.pre.front_weight, "post": model.post.head_weight}
    if (tuple(weights["pre"].shape) != (16, 32) or
            tuple(weights["post"].shape) != (32, 8) or
            provider.owned.get(id(weights["pre"]), (None,))[0] is not weights["pre"] or
            provider.owned.get(id(weights["post"]), (None,))[0] is not weights["post"]):
        raise RuntimeError("Session provider does not own the pinned K8 buffers")

    original = provider.dense
    calls = {site: 0 for site in ("pre", "post")}
    selected = ARMS[arm]

    def dense(self, a, w, *, chunk_k, initial=None, **kwargs):
        for site in selected:
            if w is weights[site]:
                if chunk_k != 8 or initial is not None or kwargs or self.mode != "fp16_xmx":
                    raise RuntimeError(f"Unexpected {site} K8 dispatch contract")
                result = dot(a, w, site=site)
                self.record("native_k8_fp16_" + site)
                calls[site] += 1
                return result
        return original(a, w, chunk_k=chunk_k, initial=initial, **kwargs)

    replacement = MethodType(dense, provider)
    provider.dense = replacement
    try:
        yield calls
    finally:
        if provider.dense is not replacement:
            raise RuntimeError("Native K8 provider method changed during its scope")
        del provider.dense
