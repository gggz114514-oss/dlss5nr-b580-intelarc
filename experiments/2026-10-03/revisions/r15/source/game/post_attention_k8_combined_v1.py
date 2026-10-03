"""Post attention fusion plus the installed cropped K8 RGB/history head.

Install after the existing three-structure combo, replace only its post
instance for a new graph capture, and restore it before closing the combo.
The game runtime and exact branch are unchanged. Output must match bytes.
"""
from __future__ import annotations

from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import MethodType

import torch
import triton

import post_k8_strided_v1 as k8
from nr_backend.post import ResetPostBlock


_ATTENTION_SOURCE = Path(__file__).with_name("post_attention_fusion_v1.py")
_SPEC = spec_from_file_location("post_attention_fusion_combo_local_v1", _ATTENTION_SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Local post attention source is unavailable")
attention = module_from_spec(_SPEC)
_SPEC.loader.exec_module(attention)


def _contiguous_cropped_head(a: torch.Tensor, weight: torch.Tensor,
                             output_size: tuple[int, int]) -> torch.Tensor:
    internal_to_output = {
        (512, 896): (480, 864),
        (640, 1024): (544, 960),
        (768, 1280): (720, 1280),
    }
    if (a.ndim != 3 or internal_to_output.get(tuple(a.shape[:2])) != output_size
            or a.shape[2] != 32
            or not a.is_contiguous()
            or a.dtype != torch.float16 or a.device.type != "xpu"
            or tuple(weight.shape) != (32, 8) or tuple(weight.stride()) != (8, 1)
            or weight.device != a.device or weight.dtype != torch.float16):
        raise ValueError("Unexpected post attention/K8 fused interface")
    out_h, out_w = output_size
    output = torch.empty((out_h, out_w, 4), dtype=a.dtype, device=a.device)
    k8._post_k8_strided[(triton.cdiv(out_h * out_w, 4), 1)](
        a, weight, output, out_h * out_w, out_w, a.stride(0),
        4, 4, 4, num_warps=1, enable_fp_fusion=False)
    return output


@contextmanager
def installed(module: ResetPostBlock, *, output_size: tuple[int, int],
              post_calls: dict[str, int], entry_fn=None,
              attention_fn=None,
              post_attention_calls: dict[str, int] | None = None):
    if not isinstance(module, ResetPostBlock) or output_size not in (
            (480, 864), (544, 960), (720, 1280)):
        raise ValueError("Unsupported post combo output canvas")
    if "forward_head" not in module.__dict__ or set(post_calls) != {"k8"}:
        raise RuntimeError("The installed K8 structure scope is absent")
    if entry_fn is not None and (post_attention_calls is None or
                                 not {"attention", "k8"}.issubset(post_attention_calls)):
        raise RuntimeError("The installed post attention capture gate is absent")
    previous = module.__dict__["forward_head"]
    counters = {"attention": 0, "k8": 0, "entry": 0}

    def run(self, features, skip):
        counters["attention"] += 1
        if entry_fn is not None:
            counters["entry"] += 1
        if post_attention_calls is not None:
            post_attention_calls["attention"] += 1

        def head_dot(a, weight):
            result = _contiguous_cropped_head(a, weight, output_size)
            counters["k8"] += 1
            post_calls["k8"] += 1  # Preserve the existing combo capture gate.
            if post_attention_calls is not None:
                post_attention_calls["k8"] += 1
            return result

        return attention.fused_forward_head(self, features, skip,
                                            head_dot=head_dot, entry_fn=entry_fn,
                                            attention_fn=attention_fn)

    replacement = MethodType(run, module)
    module.forward_head = replacement
    try:
        yield counters
    finally:
        if module.__dict__.get("forward_head") is replacement:
            module.forward_head = previous
        else:
            raise RuntimeError("Post combo method changed during the experiment")
