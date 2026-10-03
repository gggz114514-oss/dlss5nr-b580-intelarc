"""Isolated 720p fast ViT output projection: one full-K FP32 accumulator.

Install on a freshly selected, uncaptured C512+K8 720p unrounded game mode.
Keep the scope through capture/replay; reselect after leaving it. Python hits
count eager/capture calls, never graph replays. This is
lossy relative to the four half-rounded partitions; image quality is untested.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import triton
import triton.language as tl


TOKENS = 240
WIDTH = 1024


@triton.jit
def _project(X, W, INITIAL, OUT, BM: tl.constexpr, BN: tl.constexpr):
    TOKENS: tl.constexpr = 240
    WIDTH: tl.constexpr = 1024
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    lane = tl.arange(0, 32)
    valid = (row[:, None] < TOKENS) & (col[None, :] < WIDTH)
    total = tl.full((BM, BN), 0, tl.float32)
    for block in range(WIDTH // 32):
        kk = block * 32 + lane
        x = tl.load(X + row[:, None] * WIDTH + kk[None, :],
                    row[:, None] < TOKENS, other=0)
        w = tl.load(W + kk[:, None] * WIDTH + col[None, :])
        total = tl.dot(x, w, total, out_dtype=tl.float32)
    residual = tl.load(INITIAL + row[:, None] * WIDTH + col[None, :],
                       valid, other=0)
    tl.store(OUT + row[:, None] * WIDTH + col[None, :],
             (total + residual.to(tl.float32)).to(tl.float16), valid)


def project(attended, weight, initial, *, bm=16, bn=32):
    """Produce [240,1024] FP16 from the original half operands and weight."""
    if (bm, bn) != (16, 32):
        raise ValueError("Only the reviewed BM16/BN32 projection tile is supported")
    for name, tensor, shape in (("attended", attended, (TOKENS, WIDTH)),
                                ("weight", weight, (WIDTH, WIDTH)),
                                ("initial", initial, (TOKENS, WIDTH))):
        if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or
                tensor.dtype != torch.float16 or tensor.device.type != "xpu" or
                not tensor.is_contiguous()):
            raise ValueError(f"{name}: expected contiguous XPU FP16 {shape}")
        if tensor.device != attended.device:
            raise ValueError(f"{name}: expected the attention device")
    out = torch.empty((TOKENS, WIDTH), dtype=torch.float16, device=attended.device)
    kernel = _project[(triton.cdiv(TOKENS, bm), triton.cdiv(WIDTH, bn))](
        attended, weight, initial, out, bm, bn, num_warps=4,
        num_stages=1, enable_fp_fusion=False)
    return out, kernel


def capture_gate(before, after, new_entries):
    """GraphFront v5/v6 runs two warmups and one capture per new entry."""
    return (new_entries > 0 and set(before) == set(after) == set(range(8)) and
            all(after[i] - before[i] == 3 * new_entries for i in range(8)))


@contextmanager
def installed(modes):
    """Wrap only this selected session's installed scope before graph capture.

    Use ``with installed(modes) as calls:`` around the 720p capture and replay.
    ``calls['by_block']`` has eight Python hit counters; ``capture_gates`` records
    the automatic eight-block gate for each new graph entry.
    """
    from fused_vit_projection_v3 import FusedVitProjection
    from nr_backend.execution import current_arithmetic_backend, record_arithmetic_dispatch

    session = getattr(modes, "session", None)
    if (session is None or getattr(modes, "height", None) != 720 or
            getattr(modes, "variant", None) != "unrounded" or
            not getattr(modes, "controlled", False) or
            not getattr(modes, "c512_qkv_library_720", False) or
            not getattr(modes, "native_k8_720", False) or
            getattr(modes, "vit_head_720", False) or
            getattr(modes, "c512_library_calls", None) is None or
            getattr(modes, "native_k8_calls", None) is None):
        raise RuntimeError("Select the controlled 720p unrounded C512+K8 session first")
    session._ready()
    if (tuple(session.fullsize_geometry) != (720, 1280) or
            tuple(session.fullsize_padding["padding"]) != (768, 1280)):
        raise RuntimeError("Expected the unchanged 720p model and internal canvas")
    stack = session._stack
    if stack.graph.entries or stack.provider.mode != "fp16_xmx":
        raise RuntimeError("ViT projection requires a fresh uncaptured FP16 XMX session")
    blocks = tuple(stack.model.vit)
    if len(blocks) != 8 or len({id(block) for block in blocks}) != 8:
        raise RuntimeError("Expected eight distinct ViT blocks")
    fused = [item for item in stack.components if isinstance(item, FusedVitProjection)]
    if len(fused) != 1 or fused[0].provider is not stack.provider:
        raise RuntimeError("Expected the installed ViT four-part projection owner")
    fused = fused[0]
    projections = {}
    contracts = {}
    for index, block in enumerate(blocks):
        for name, weight, shape, owner in (
                ("projection", block.projection, (WIDTH, WIDTH), projections),
                ("contract", block.contract, (4096, WIDTH), contracts)):
            if (tuple(weight.shape) != shape or weight.dtype != torch.float16 or
                    weight.device.type != "xpu" or not weight.is_contiguous() or
                    fused.weights.get(id(weight)) is not weight or id(weight) in owner):
                raise RuntimeError(f"ViT block {index} has an unexpected {name} weight")
            owner[id(weight)] = (weight, index)
    if len(projections) != 8 or len(contracts) != 8 or set(projections) & set(contracts):
        raise RuntimeError("ViT projection/contract buffers must be distinct")

    if "_select" in modes.__dict__:
        raise RuntimeError("Game mode selector already has a scoped override")
    original_scope = session._installed
    vit = session._vit
    original_select = modes._select
    selected_source = tuple(modes.source)
    calls = {"by_block": {index: 0 for index in range(8)},
             "capture_gates": [], "kernel_resources": {}}

    def guarded_select(height, source, *, variant):
        if (modes.session is not session or height != 720 or
                tuple(source) != selected_source or variant != "unrounded"):
            session._failed = True
            raise RuntimeError("ViT candidate cannot switch the selected session or geometry")
        return original_select(height, source, variant=variant)

    @contextmanager
    def wrapped_scope():
        if modes.session is not session or session._stack is not stack:
            raise RuntimeError("ViT candidate session changed")
        before = dict(calls["by_block"])
        entries_before = len(stack.graph.entries)
        with original_scope():
            previous = vit.split_k_projection
            if getattr(previous, "__self__", None) is not fused or previous != fused.apply:
                raise RuntimeError("ViT projection owner changed during session install")

            def replacement(features, weight, initial, parts=4):
                try:
                    if (modes.session is not session or session._stack is not stack or
                            stack.provider.mode != "fp16_xmx" or parts != 4 or
                            current_arithmetic_backend() != "triton"):
                        raise RuntimeError("ViT candidate left its scoped FP16 contract")
                    target = projections.get(id(weight))
                    if target is not None and target[0] is weight:
                        output, kernel = project(features, weight, initial)
                        record_arithmetic_dispatch("dense")
                        calls["by_block"][target[1]] += 1
                        key = getattr(kernel, "hash", None)
                        if key is not None and key not in calls["kernel_resources"]:
                            metadata = getattr(kernel, "metadata", None)
                            calls["kernel_resources"][key] = {
                                "registers": getattr(kernel, "n_regs", None),
                                "spills": getattr(kernel, "n_spills", None),
                                "shared_bytes": getattr(metadata, "shared", None),
                            }
                        return output
                    contract = contracts.get(id(weight))
                    if contract is None or contract[0] is not weight:
                        raise RuntimeError("Unknown ViT split projection weight")
                    return previous(features, weight, initial, parts)
                except BaseException:
                    session._failed = True
                    raise

            vit.split_k_projection = replacement
            try:
                yield
            except BaseException:
                session._failed = True
                raise
            finally:
                unchanged = vit.split_k_projection is replacement
                vit.split_k_projection = previous
                new_entries = len(stack.graph.entries) - entries_before
                if new_entries:
                    passed = capture_gate(before, calls["by_block"], new_entries)
                    calls["capture_gates"].append({"new_entries": new_entries,
                                                   "by_block": dict(calls["by_block"]),
                                                   "passed": passed})
                    if not passed:
                        session._failed = True
                        raise RuntimeError("New ViT graph missed the eight projection blocks")
                if not unchanged:
                    session._failed = True
                    raise RuntimeError("ViT projection hook changed during session scope")

    session._installed = wrapped_scope
    modes._select = guarded_select
    try:
        yield calls
    finally:
        unchanged = (session._installed is wrapped_scope and
                     modes._select is guarded_select)
        session._installed = original_scope
        del modes._select
        # Captured commands survive Python hooks. Reject a replay after removal.
        if stack.graph.entries and not session._closed:
            session._failed = True
        if not unchanged:
            session._failed = True
            raise RuntimeError("ViT session hook changed during experiment")
