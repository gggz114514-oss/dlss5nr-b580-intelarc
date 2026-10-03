"""Scoped decoder C512 packed-window projection for full-size game NR.

The installed fullsize session owns a c512 closure and frame-count contract.
Wrap its existing installed scope, replace selected decoder512 forward calls,
and preserve the original boundaries method and all other blocks. Experimental.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch

import nr_backend.split_block as split
import nr_backend.multihead_block as blocks
from nr_backend.execution import current_arithmetic_backend, record_arithmetic_dispatch
from nr_backend.unround_policy import ENABLED, round_multihead
from c512_window_projection_v1 import forward as project
from fused_qkv_pack_native_half_v1 import forward as prepare_packed_qkv
from fused_swin_heads_native_half_v1 import forward as native_heads
from quantization_dataflow_v1 import CONTRACTS


_CONTRACT = "c512_window_projection_v1._project"
_RECEIPT = (("OUT",), ())


def _packed_with_baseline_probabilities(stack, module, features):
    """Keep the unrounded C512 baseline's rounded probabilities, but not its output rounding."""
    if module.channels != 512 or "c512" not in ENABLED:
        raise ValueError("Expected unrounded C512 attention")
    height, width, _ = features.shape
    projected = blocks.sm89_f16_dot(
        round_multihead(512, features), module.qkv, chunk_k=16
    ).reshape(height, width, module.heads, 3, 32)
    (query, key, value), _ = prepare_packed_qkv(
        projected, module.scale, module.pixel_order,
        rows=stack.window_blocks.layout.rows, round_qkv=False
    )
    for _ in range(2):
        record_arithmetic_dispatch("attention_normalize_c32")
    for _ in range(3):
        record_arithmetic_dispatch("fp8")
    # The installed native-head kernel rounds probabilities before AV even
    # when the C512 activation policy is unrounded. Its output stays FP16.
    packed, kernel = native_heads(query, key, value, module.bias)
    stack.window_blocks.last_attention_kernel = kernel
    stack.window_blocks.last_attention_selection = {"route": "native_heads"}
    for _ in range(module.heads * ((height // 8 * (width // 8) + 1023) // 1024)):
        for kind in ("batched", "attention_exp_swin", "attention_weights", "batched"):
            record_arithmetic_dispatch(kind)
    for kind, count in (("multi", 1), ("batched_heads", module.heads), ("qkv_pack", 1)):
        calls = stack.window_blocks.layout.calls
        calls[kind] = calls.get(kind, 0) + count
    return packed


@contextmanager
def installed(session, *, height: int, block_indices=(1,)):
    if height not in (480, 540, 720) or not hasattr(session, "_installed"):
        raise ValueError("Expected selected full-size game session")
    stack = session._stack
    block_indices = tuple(block_indices)
    if not block_indices or len(set(block_indices)) != len(block_indices) or \
            any(index not in range(8) for index in block_indices):
        raise ValueError("Expected unique decoder512 block indices 0..7")
    expected_shifts = ((0, 0), (4, 4), (0, 4), (4, 0))
    targets = {}
    for index in block_indices:
        module = stack.model.decoder512[index]
        if module.window_shift != expected_shifts[index % 4] or \
                module.final_weight is not None:
            raise ValueError(f"Unexpected decoder512[{index}] layout")
        targets[id(module)] = index
    if _CONTRACT in CONTRACTS:
        raise RuntimeError("C512 direct projection contract already active")
    original_installed = session._installed
    calls = {f"decoder512_{index}": 0 for index in block_indices}
    calls["resources"] = {}

    @contextmanager
    def wrapper():
        with original_installed():
            if stack.provider.mode != "fp16_xmx":
                raise ValueError("C512 projection candidate requires pinned XMX path")
            previous_forward = split.SplitSwinBlock.forward
            previous_boundaries = split.SplitSwinBlock.forward_boundaries
            CONTRACTS[_CONTRACT] = _RECEIPT

            def replacement(module, features):
                index = targets.get(id(module))
                if index is None:
                    return previous_forward(module, features)
                if current_arithmetic_backend() != "triton":
                    raise ValueError("C512 candidate requires Triton arithmetic")
                if tuple(features.shape) != ({480: (16, 28, 512),
                                              540: (20, 32, 512),
                                              720: (24, 40, 512)}[height]):
                    raise ValueError(f"Unexpected decoder512[{index}] feature canvas")
                name = stack.c512_int8.modules[id(module)]
                mlp = stack.c512_int8.ffn(name, features)
                h, w = mlp.shape[:2]
                sy, sx = module.window_shift
                padded = torch.nn.functional.pad(
                    mlp, (0, 0, sx, (-w-sx) % 8, sy, (-h-sy) % 8))
                packed = (_packed_with_baseline_probabilities(
                    stack, module.attention, padded
                ) if height == 720 and "c512" in ENABLED else
                    stack.window_blocks.windows(module.attention, padded))
                full, kernel, _ = project(
                    packed, module.projection.weight, split.q(mlp),
                    module.projection.skip_scale,
                    module.attention.pixel_inverse, shift=(sy, sx))
                if kernel.n_spills not in (0, None):
                    raise RuntimeError("C512 projection kernel spills")
                calls["resources"][kernel.hash] = {
                    "spills": kernel.n_spills, "registers": kernel.n_regs,
                    "shared_bytes": kernel.metadata.shared}
                stack.provider.record("fp16_dense")
                record_arithmetic_dispatch("dense")
                for _ in range(2):
                    record_arithmetic_dispatch("fp8")
                session._counts["c512"] += 1
                calls[f"decoder512_{index}"] += 1
                return split.q(full)

            split.SplitSwinBlock.forward = replacement
            try:
                yield
            finally:
                valid = (split.SplitSwinBlock.forward is replacement and
                         split.SplitSwinBlock.forward_boundaries is previous_boundaries and
                         CONTRACTS.get(_CONTRACT) == _RECEIPT)
                split.SplitSwinBlock.forward = previous_forward
                CONTRACTS.pop(_CONTRACT, None)
                if not valid:
                    raise RuntimeError("C512 candidate scope changed during capture")

    session._installed = wrapper
    try:
        yield calls
    finally:
        if session._installed is not wrapper:
            raise RuntimeError("Fullsize installed scope changed during C512 candidate")
        session._installed = original_installed
