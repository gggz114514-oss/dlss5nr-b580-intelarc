"""Fast-only C32-family ablation of independent E4M3 activation boundaries.

Use only after the installed three-structure combo has selected a 480/540
session.  The seven ordinary C32 blocks keep their existing fused attention
and all weights.  This scope removes the independent input, MLP, skip, pool,
and down E4M3 roundtrips for that family; fused-kernel rounding remains.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
from types import MethodType


_FUSION_SHA256 = "15da805d71caff5b5e518c97ef68b24658e129f0d1763df10540449211ee6106"
_BLOCK_SHA256 = "6083c1c3dbd2b01900524c927a4bf872486cec8ef72f6c2c103ce11636ac5e0f"


def _pinned(source, expected: Path, digest: str) -> None:
    actual = Path(source.__file__).resolve()
    if actual != expected.resolve() or hashlib.sha256(actual.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"Installed C32 source differs from audited runtime: {actual}")


@contextmanager
def installed(stack, *, height: int, runtime: Path):
    import torch
    import nr_backend.c32_block as blocks_source
    import c32_repeated_tail_fusion_v1 as fusion_source

    if height not in (480, 540):
        raise ValueError("C32 independent-rounding ablation requires the installed 480/540 combo")
    root = Path(runtime).resolve()
    _pinned(blocks_source, root / "fast/backend/nr_backend/c32_block.py", _BLOCK_SHA256)
    _pinned(fusion_source, root / "game/c32_repeated_tail_fusion_v1.py", _FUSION_SHA256)
    ordinary = fusion_source.ordinary_blocks(stack.model)
    if len(ordinary) != 7 or any("forward_unquantized" not in b.__dict__ for b in ordinary):
        raise RuntimeError("Installed C32 fusion is absent from an ordinary block")
    if any("forward" in b.__dict__ or "forward_outputs" in b.__dict__ for b in ordinary):
        raise RuntimeError("C32 output methods already have an unknown override")

    saved_fusion_q = fusion_source.q
    if saved_fusion_q is not blocks_source.quantize_fp8:
        raise RuntimeError("C32 conversion owner differs from the audited source")
    calls = {"fusion_input": 0, "by_block": {str(i): {"forward": 0, "outputs": 0}
                                            for i in range(7)}}

    def identity_activation(x):
        if not isinstance(x, torch.Tensor) or x.dtype != torch.float16 or x.device.type != "xpu":
            raise ValueError("Expected XPU FP16 C32-family activation")
        calls["fusion_input"] += 1
        return x

    replacements = []
    try:
        fusion_source.q = identity_activation
        for index, block in enumerate(ordinary):
            key = str(index)

            def forward(self, features, *, _key=key):
                calls["by_block"][_key]["forward"] += 1
                return self.forward_unquantized(features)

            def forward_outputs(self, features, *, _key=key):
                calls["by_block"][_key]["outputs"] += 1
                full = self.forward_unquantized(features)
                if self.down_weight is None:
                    return full, None
                top = (full[0::2, 0::2] + full[0::2, 1::2]).half()
                bottom = (full[1::2, 0::2] + full[1::2, 1::2]).half()
                pooled = ((top + bottom).half() * .25).half()
                down = blocks_source.sm89_f16_dot(pooled, self.down_weight, chunk_k=16)
                return full, down

            bound_forward = MethodType(forward, block)
            bound_outputs = MethodType(forward_outputs, block)
            block.forward = bound_forward
            block.forward_outputs = bound_outputs
            replacements.append((block, bound_forward, bound_outputs))
        yield calls
    finally:
        for block, bound_forward, bound_outputs in reversed(replacements):
            if (block.__dict__.get("forward") is not bound_forward or
                    block.__dict__.get("forward_outputs") is not bound_outputs):
                raise RuntimeError("C32 ablation override changed during its scope")
            del block.forward
            del block.forward_outputs
        if fusion_source.q is not identity_activation:
            raise RuntimeError("C32 fusion activation converter changed during its scope")
        fusion_source.q = saved_fusion_q
