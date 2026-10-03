"""Isolated fast-only ablation of pre attention's output E4M3 roundtrip.

Install on a newly selected RE8 model before its first graph capture.  This
scope changes one activation boundary; it does not touch weights, the INT8 FFN,
the exact backend, or any installed G: source file.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
from types import MethodType


_PRE_SOURCE_SHA256 = "06fc9e27fec003f126b0cba9a83ab8233b641c4f374e6867648538184a287fe9"
_CANVASES = {360: (384, 768), 480: (512, 896),
             540: (640, 1024)}


@contextmanager
def installed(stack, *, height: int, runtime: Path):
    """Replace only ``q(attended)`` inside this session's pre block.

    The caller must close this scope before closing the model session. Captured
    graphs keep the selected path; Python hot toggles cannot change a replay.
    """
    import torch
    import nr_backend.pre_block as pre_source

    if height not in _CANVASES:
        raise ValueError("Unsupported pre ablation game mode")
    source = Path(pre_source.__file__).resolve()
    expected = (Path(runtime).resolve() / "fast/backend/nr_backend/pre_block.py")
    if source != expected or hashlib.sha256(source.read_bytes()).hexdigest() != _PRE_SOURCE_SHA256:
        raise RuntimeError("Installed pre source differs from the audited RE8 runtime")
    pre = stack.model.pre
    if type(pre) is not pre_source.PreBlock or "forward_features_unquantized" in pre.__dict__:
        raise RuntimeError("Pre block is already overridden or has an unknown owner")
    original = pre_source.PreBlock.forward_features_unquantized
    calls = {"candidate_calls": 0, "shape": None, "attended_contiguous": None}

    def replacement(self, features):
        canvas = _CANVASES[height]
        if (tuple(features.shape) != (*canvas, 16) or features.dtype != torch.float16 or
                features.device.type != "xpu" or not features.is_contiguous()):
            raise ValueError("Unexpected pre input dtype/shape/layout")
        projected = pre_source.project_front_features(features, self.front_weight)
        mlp = self.mlp.forward_unquantized(projected)
        attended = self.attention(pre_source.quantize_fp8(mlp))
        if (tuple(attended.shape) != (*canvas, 32) or attended.dtype != torch.float16 or
                attended.device.type != "xpu"):
            raise ValueError("Unexpected pre attention output")
        calls["attended_contiguous"] = bool(attended.is_contiguous())
        if not calls["attended_contiguous"]:
            raise RuntimeError("Pre attention output is not contiguous; account for copy before ablation")
        calls["candidate_calls"] += 1
        calls["shape"] = (*canvas, 32)
        return pre_source.sm89_f16_dot(
            attended, self.output_weight, chunk_k=16,
            initial=(mlp * self.skip_scale).half())

    bound = MethodType(replacement, pre)
    pre.forward_features_unquantized = bound
    try:
        yield calls
    finally:
        if pre.__dict__.get("forward_features_unquantized") is not bound:
            raise RuntimeError("Pre ablation override changed during its scope")
        del pre.forward_features_unquantized
        if pre.forward_features_unquantized.__func__ is not original:
            raise RuntimeError("Pre ablation failed to restore installed method")
