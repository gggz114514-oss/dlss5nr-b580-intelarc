"""Isolated twelve-block C128 attention/projection fusion on the active game chain.

Installs after the C64 fusion and C128 pairwise scopes. Non-C128 calls delegate
to the exact bound method supplied by those scopes; teardown restores it.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch

from c128_qkv_direct_pack_all_v1 import _SHIFTS, direct_pack
from c128_attention_project_fused_720_v1 import forward as fused
from current_dense_tiled_provider_v1 import POLICY
from nr_backend.execution import current_arithmetic_backend, record_arithmetic_dispatch
from nr_backend.multihead_block import MultiHeadSwinBlock
from nr_backend.unround_policy import round_multihead
from window_blocks_v3 import WindowBlocks


class Counter:
    def __init__(self, labels):
        self.calls = {label: 0 for label in labels}
        self.capture_calls = {label: 0 for label in labels}

    @contextmanager
    def capture(self):
        before = dict(self.capture_calls)
        yield
        missed = {label: self.capture_calls[label] - value
                  for label, value in before.items()
                  if self.capture_calls[label] - value != 1}
        if missed:
            raise RuntimeError(f"C128 attention/projection capture incomplete: {missed}")


def _targets(stack):
    targets = {}
    for side, group in (("encoder", stack.model.encoder[2]),
                        ("decoder", stack.model.decoder[1])):
        for index in range(6):
            outer = group[index]
            block = outer.body if side == "decoder" and index == 0 else outer
            shift = _SHIFTS[(index + (2 if side == "decoder" else 0)) % 4]
            if (type(block) is not MultiHeadSwinBlock or block.channels != 128 or
                    block.attention.heads != 4 or block.window_shift != shift):
                raise ValueError(f"Unexpected {side} C128 block {index}")
            sy, sx = shift
            shape = (96 + (8 if sy else 0), 160 + (8 if sx else 0))
            label = f"{side}[{index}]"
            if id(block) in targets:
                raise ValueError("Duplicate C128 block")
            targets[id(block)] = (block, label, shape)
    if len(targets) != 12:
        raise ValueError("Expected twelve C128 blocks")
    return targets


def compile_preflight(stack, *, variant: str):
    if variant not in ("standard", "unrounded"):
        raise ValueError("Unknown game variant")
    resources = {}
    for block, _, shape in _targets(stack).values():
        if shape in resources:
            continue
        hp, wp = shape
        windows = (4, hp // 8, wp // 8, 64, 32)
        q, k, v = [torch.empty(windows, dtype=torch.float16, device="xpu")
                   for _ in range(3)]
        residual = torch.empty((hp, wp, 128), dtype=torch.float16, device="xpu")
        _, selection = fused(q, k, v, block.attention.bias, block.output_weight,
                             residual, block.skip_scale, block.attention.pixel_order,
                             height=96, width=160, shift=block.window_shift,
                             round_weights=variant == "standard")
        if selection["attempts"][-1]["spills"] != 0:
            raise RuntimeError(f"C128 fusion spills at {shape}")
        resources[f"{hp}x{wp}"] = selection
    return resources


@contextmanager
def installed(stack, *, combo_calls: dict, variant: str):
    if variant not in ("standard", "unrounded"):
        raise ValueError("Unknown game variant")
    window_blocks = stack.window_blocks
    if not isinstance(window_blocks, WindowBlocks) or "apply" not in window_blocks.__dict__:
        raise ValueError("Expected active C64 instance method to delegate")
    if not isinstance(combo_calls, dict) or "c128" not in combo_calls:
        raise ValueError("Current structure combo counters required")
    targets = _targets(stack)
    labels = {row[1] for row in targets.values()}
    c128_calls = combo_calls["c128"]
    if set(c128_calls["by_block"]) != labels:
        raise ValueError("Current C128 combo names changed")
    if window_blocks.probe is not None:
        raise ValueError("C128 fusion cannot service a packed-tensor debug probe")
    original = window_blocks.__dict__["apply"]
    counter = Counter(labels)

    def selected_apply(self, module, features):
        row = targets.get(id(module))
        if row is None:
            return original(module, features)
        selected, label, expected_shape = row
        if (module is not selected or tuple(features.shape) != (96, 160, 128) or
                current_arithmetic_backend() != "triton" or
                self.provider.mode != "fp16_xmx" or
                not self.layout.native_normalize or not self.layout.native_swin or
                features.device.type != "xpu" or features.dtype != torch.float16):
            raise RuntimeError(f"C128 fused boundary changed: {label}")
        sy, sx = module.window_shift
        padded = torch.nn.functional.pad(
            round_multihead(128, features),
            (0, 0, sx, (-160 - sx) % 8, sy, (-96 - sy) % 8))
        mlp = module.mlp(padded)
        if tuple(mlp.shape) != (*expected_shape, 128):
            raise RuntimeError(f"C128 padded geometry changed: {label}")
        q, k, v = direct_pack(round_multihead(128, mlp), module.attention,
                              height=720, shift=(sy, sx))
        result, selection = fused(
            q, k, v, module.attention.bias, module.output_weight,
            mlp, module.skip_scale, module.attention.pixel_order,
            height=96, width=160, shift=(sy, sx),
            round_weights=variant == "standard")

        # Mirror the logical receipts of the two original kernels while the
        # physical attention/projection is fused into one GPU launch.
        for _ in range(2):
            record_arithmetic_dispatch("attention_normalize_c32")
        for _ in range(3):
            record_arithmetic_dispatch("fp8")
        windows = (expected_shape[0] // 8) * (expected_shape[1] // 8)
        for _ in range(4 * ((windows + 1023) // 1024)):
            for key in ("batched", "attention_exp_swin", "attention_weights", "batched"):
                record_arithmetic_dispatch(key)
        for key, count in (("multi", 1), ("batched_heads", 4), ("qkv_pack", 1)):
            self.layout.calls[key] = self.layout.calls.get(key, 0) + count
        record_arithmetic_dispatch("fp8")
        record_arithmetic_dispatch("dense")
        self.provider.record("fp16_dense")
        policy_key = (expected_shape[0] * expected_shape[1], 128, 128,
                      True, (True, True, True))
        if policy_key in POLICY:
            self.provider.record("fp16_tiled")
            tag = f"{expected_shape[0] * expected_shape[1]}x128x128:initial=True:contiguous={(True,True,True)}"
            self.provider.tiled_calls[tag] = self.provider.tiled_calls.get(tag, 0) + 1
        self.calls.append(dict(module=self.modules[id(module)],
                               shape=list(features.shape),
                               padded_shape=list(mlp.shape), shift=[sy, sx],
                               fusion="c128_attention_projection",
                               fusion_selection=selection))
        c128_calls["direct_pack"] += 1
        c128_calls["by_block"][label] += 1
        counter.calls[label] += 1
        if torch.xpu.is_current_stream_capturing():
            counter.capture_calls[label] += 1
        return result

    bound = MethodType(selected_apply, window_blocks)
    window_blocks.apply = bound
    try:
        yield counter
    finally:
        if window_blocks.__dict__.get("apply") is not bound:
            raise RuntimeError("C128 fused scope changed before teardown")
        window_blocks.apply = original
