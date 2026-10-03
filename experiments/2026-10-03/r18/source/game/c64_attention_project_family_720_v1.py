"""Isolated 720p C64 attention/output-projection fusion for eight game blocks.

The active C64 QKV producer, residual MLP and output crop are unchanged. Scope
entry is after the installed structure combo and before either graph capture.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch

from c64_qkv_direct_pack_all_v1 import _SIZES, _SHIFTS, direct_pack
from c64_attention_project_fused_720_v1 import forward as fused
from current_dense_tiled_provider_v1 import POLICY
from nr_backend.execution import current_arithmetic_backend, record_arithmetic_dispatch
from nr_backend.multihead_block import MultiHeadSwinBlock
from nr_backend.unround_policy import round_multihead
from window_blocks_v3 import WindowBlocks


class Counter:
    def __init__(self, targets):
        self.calls = {name: 0 for name in targets}
        self.capture_calls = {name: 0 for name in targets}
        self.producer_launches = 0

    @contextmanager
    def capture(self):
        before = dict(self.capture_calls)
        yield
        missed = {name: self.capture_calls[name] - value
                  for name, value in before.items()
                  if self.capture_calls[name] - value != 1}
        if missed:
            raise RuntimeError(f"C64 fused family capture incomplete: {missed}")


def _targets(stack):
    model = stack.model
    targets = {}
    for index, shape in enumerate(_SIZES[720]):
        for side, group in (("encoder", model.encoder[1]),
                            ("decoder", model.decoder[2])):
            outer = group[index]
            block = outer.body if side == "decoder" and index == 0 else outer
            if (type(block) is not MultiHeadSwinBlock or block.channels != 64 or
                    block.attention.heads != 2 or block.window_shift != _SHIFTS[index]):
                raise ValueError(f"Unexpected {side} C64 block {index}")
            name = f"{side}[{index}]"
            if id(block) in targets:
                raise ValueError("Duplicate C64 block")
            targets[id(block)] = (block, name, shape)
    if len(targets) != 8:
        raise ValueError("Expected eight C64 blocks")
    return targets


def compile_preflight(stack, *, variant: str):
    if variant not in ("standard", "unrounded"):
        raise ValueError("Unknown game variant")
    targets = _targets(stack)
    resources = {}
    for block, _, shape in targets.values():
        if shape in resources:
            continue
        hp, wp = shape
        windows = (2, hp // 8, wp // 8, 64, 32)
        q, k, v = [torch.empty(windows, dtype=torch.float16, device="xpu")
                   for _ in range(3)]
        residual = torch.empty((hp, wp, 64), dtype=torch.float16, device="xpu")
        sy, sx = block.window_shift
        _, selection = fused(
            q, k, v, block.attention.bias, block.output_weight,
            residual, block.skip_scale, block.attention.pixel_order,
            height=192, width=320, shift=(sy, sx),
            round_weights=variant == "standard")
        if selection["attempts"][-1]["spills"] != 0:
            raise RuntimeError(f"C64 fusion spills at {shape}")
        resources[f"{hp}x{wp}"] = selection
    return resources


@contextmanager
def installed(stack, *, combo_calls: dict, variant: str):
    if variant not in ("standard", "unrounded"):
        raise ValueError("Unknown game variant")
    window_blocks = stack.window_blocks
    if not isinstance(window_blocks, WindowBlocks) or "apply" in window_blocks.__dict__:
        raise ValueError("Expected one unmodified WindowBlocks.apply")
    if not isinstance(combo_calls, dict) or "c64" not in combo_calls:
        raise ValueError("Current structure combo counters required")
    targets = _targets(stack)
    expected_names = {row[1] for row in targets.values()}
    c64_calls = combo_calls["c64"]
    if set(c64_calls["by_block"]) != expected_names:
        raise ValueError("Current C64 combo names changed")
    if window_blocks.probe is not None:
        raise ValueError("C64 fusion cannot service the packed-tensor debug probe")
    counter = Counter(expected_names)
    original = window_blocks.apply

    def selected_apply(self, module, features):
        row = targets.get(id(module))
        if row is None:
            return original(module, features)
        selected, name, expected_shape = row
        if (module is not selected or tuple(features.shape) != (192, 320, 64) or
                current_arithmetic_backend() != "triton" or
                self.provider.mode != "fp16_xmx" or
                not self.layout.native_normalize or not self.layout.native_swin or
                features.device.type != "xpu" or features.dtype != torch.float16):
            raise RuntimeError(f"C64 fused boundary changed: {name}")
        sy, sx = module.window_shift
        padded = torch.nn.functional.pad(
            round_multihead(64, features),
            (0, 0, sx, (-320 - sx) % 8, sy, (-192 - sy) % 8))
        mlp = module.mlp(padded)
        if tuple(mlp.shape) != (*expected_shape, 64):
            raise RuntimeError(f"C64 padded geometry changed: {name}")
        q, k, v = direct_pack(round_multihead(64, mlp), module.attention)
        result, selection = fused(
            q, k, v, module.attention.bias, module.output_weight,
            mlp, module.skip_scale, module.attention.pixel_order,
            height=192, width=320, shift=(sy, sx),
            round_weights=variant == "standard")

        # Preserve the existing logical dispatch and capture receipts. Physical
        # attention + projection now share one kernel, but combo gates still
        # require each of the same eight model blocks to have executed.
        for _ in range(2):
            record_arithmetic_dispatch("attention_normalize_c32")
        for _ in range(3):
            record_arithmetic_dispatch("fp8")
        windows = (expected_shape[0] // 8) * (expected_shape[1] // 8)
        for _ in range(2 * ((windows + 1023) // 1024)):
            for key in ("batched", "attention_exp_swin", "attention_weights", "batched"):
                record_arithmetic_dispatch(key)
        for key, n in (("multi", 1), ("batched_heads", 2), ("qkv_pack", 1)):
            self.layout.calls[key] = self.layout.calls.get(key, 0) + n
        record_arithmetic_dispatch("fp8")
        record_arithmetic_dispatch("dense")
        self.provider.record("fp16_dense")
        policy_key = (expected_shape[0] * expected_shape[1], 64, 64, True,
                      (True, True, True))
        if policy_key in POLICY:
            self.provider.record("fp16_tiled")
            label = f"{expected_shape[0] * expected_shape[1]}x64x64:initial=True:contiguous={(True,True,True)}"
            self.provider.tiled_calls[label] = self.provider.tiled_calls.get(label, 0) + 1
        self.calls.append(dict(module=self.modules[id(module)],
                               shape=list(features.shape),
                               padded_shape=list(mlp.shape), shift=[sy, sx],
                               fusion="c64_attention_projection",
                               fusion_selection=selection))
        c64_calls["direct_pack"] += 1
        c64_calls["by_block"][name] += 1
        counter.calls[name] += 1
        counter.producer_launches += 1
        if torch.xpu.is_current_stream_capturing():
            counter.capture_calls[name] += 1
        return result

    bound = MethodType(selected_apply, window_blocks)
    window_blocks.apply = bound
    try:
        yield counter
    finally:
        if window_blocks.__dict__.get("apply") is not bound:
            raise RuntimeError("C64 fused scope changed before teardown")
        del window_blocks.apply
