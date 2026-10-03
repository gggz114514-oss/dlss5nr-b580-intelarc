"""Scoped, byte-preserving game fast-chain structure candidate.

Install before a new graph session's first model call.  Keep the scope alive
until that session is closed; already captured graphs cannot be hot-switched.
The candidate is opt-in and leaves the exact backend untouched.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from types import MethodType

import c64_qkv_direct_pack_all_v1 as c64
import c128_qkv_direct_pack_all_v1 as c128
import c256_qkv_direct_pack_all_v1 as c256
import c512_window_projection_game_v1 as c512_window
import c32_repeated_tail_fusion_v1 as c32
import post_k8_strided_v1 as post
import post_attention_k8_combined_v1 as post_attention_k8
import pre_attention_tail_fusion_v1 as pre_tail


@contextmanager
def installed(stack, *, height: int, output_size: tuple[int, int], session=None):
    """Fuse the three already measured boundaries in capture order.

    Counters count Python calls during eager execution/graph capture.  A graph
    replay does not call Python, so callers must check its replay counter too.
    """
    model_canvas = {360: (360, 640), 480: (480, 864),
                    540: (544, 960), 720: (720, 1280)}
    if height not in model_canvas or output_size != model_canvas[height]:
        raise ValueError("Unexpected game NR height/model canvas pair")

    c128_scope = (c128.installed(stack.window_blocks, stack.model, height=height)
                  if height in (480, 540, 720) else nullcontext(None))
    c256_scope = (c256.installed(stack.window_blocks, stack.model,
                                height=height, block_m=32)
                  if height in (480, 540, 720) else nullcontext(None))
    pre_scope = (pre_tail.installed(stack.model, height=height)
                 if height in (480, 540, 720) else nullcontext(None))
    if height in (480, 540, 720) and session is None:
        raise ValueError("Full-size combo requires its session for C512 direct windows")
    c512_scope = (c512_window.installed(session, height=height,
                                        block_indices=tuple(range(8)))
                  if height in (480, 540, 720) else nullcontext(None))
    with c64.installed(stack.window_blocks, stack.model, height=height) as c64_calls, \
            c128_scope as c128_calls, c256_scope as c256_calls, \
            pre_scope as pre_calls, c512_scope as c512_calls:
        with c32.installed(stack.model):
            blocks = c32.ordinary_blocks(stack.model)
            if len(blocks) != 7:
                raise RuntimeError("Expected seven ordinary C32 blocks")
            c32_calls = {f"block_{index}": 0 for index in range(7)}
            originals = []
            try:
                for index, block in enumerate(blocks):
                    original = block.forward_unquantized
                    key = f"block_{index}"

                    def counted(self, features, *, _original=original, _key=key):
                        c32_calls[_key] += 1
                        return _original(features)

                    replacement = MethodType(counted, block)
                    block.forward_unquantized = replacement
                    originals.append((block, original, replacement))
                with post.installed(stack.model.post, output_size=output_size,
                                    output_channels=4) as post_calls:
                    extra_scope = (post_attention_k8.installed(
                        stack.model.post, output_size=output_size, post_calls=post_calls)
                        if height in (480, 540, 720) else nullcontext(None))
                    with extra_scope as post_attention_calls:
                        yield {"c64": c64_calls, "c128": c128_calls,
                               "c256": c256_calls, "c512": c512_calls,
                               "pre": pre_calls,
                               "c32": c32_calls,
                               "post": post_calls, "post_attention": post_attention_calls}
            finally:
                for block, original, replacement in reversed(originals):
                    if block.forward_unquantized is not replacement:
                        raise RuntimeError("C32 capture counter changed during combo")
                    block.forward_unquantized = original


def capture_gate(calls: dict | None) -> dict[str, bool]:
    """Check all candidate Python paths were used in this session's capture."""
    if calls is None:
        return {"c64_all_eight": False, "c32_all_seven": False,
                "post_k8": False}
    by_block = calls["c64"]["by_block"]
    c32_by_block = calls["c32"]
    result = {
        "c64_all_eight": len(by_block) == 8 and all(v > 0 for v in by_block.values()),
        "c32_all_seven": (len(c32_by_block) == 7 and
                           all(v > 0 for v in c32_by_block.values())),
        "post_k8": calls["post"]["k8"] > 0,
    }
    if calls.get("post_attention") is not None:
        result["post_attention"] = calls["post_attention"]["attention"] > 0
        result["post_contiguous_k8"] = calls["post_attention"]["k8"] > 0
    if calls.get("c128") is not None:
        c128_by_block = calls["c128"]["by_block"]
        result["c128_all_twelve"] = (len(c128_by_block) == 12 and
                                      all(value > 0 for value in c128_by_block.values()))
    if calls.get("c256") is not None:
        c256_by_block = calls["c256"]["by_block"]
        result["c256_all_sixteen"] = (len(c256_by_block) == 16 and
                                      all(value > 0 for value in c256_by_block.values()))
    if calls.get("pre") is not None:
        result["pre_attention_tail"] = calls["pre"]["pre_fused"] > 0
    if calls.get("c512") is not None:
        result["c512_decoder_all_eight"] = all(
            calls["c512"][f"decoder512_{index}"] > 0 for index in range(8))
    return result


def snapshot_calls(calls: dict | None) -> dict | None:
    if calls is None:
        return None
    post_attention = calls.get("post_attention")
    return {"c64_total": int(calls["c64"]["direct_pack"]),
            "c64_by_block": dict(calls["c64"]["by_block"]),
            "c128_by_block": (None if calls.get("c128") is None else
                              dict(calls["c128"]["by_block"])),
            "c256_by_block": (None if calls.get("c256") is None else
                              dict(calls["c256"]["by_block"])),
            "pre_fused": (None if calls.get("pre") is None else
                          int(calls["pre"]["pre_fused"])),
            "c512_decoder": (None if calls.get("c512") is None else
                             {index: int(calls["c512"][f"decoder512_{index}"])
                              for index in range(8)}),
            "c32_by_block": dict(calls["c32"]),
            "post_k8": int(calls["post"]["k8"]),
            "post_attention": None if post_attention is None else int(post_attention["attention"]),
            "post_contiguous_k8": None if post_attention is None else int(post_attention["k8"])}


def capture_delta_gate(before: dict, after_calls: dict) -> dict[str, bool]:
    """Require every fused boundary in a newly captured graph, excluding eager hits."""
    after = snapshot_calls(after_calls)
    result = {
        "c64_all_eight": (set(after["c64_by_block"]) == set(before["c64_by_block"])
                          and len(after["c64_by_block"]) == 8
                          and all(after["c64_by_block"][key] > value
                                  for key, value in before["c64_by_block"].items())),
        "c32_all_seven": (set(after["c32_by_block"]) == set(before["c32_by_block"])
                          and len(after["c32_by_block"]) == 7
                          and all(after["c32_by_block"][key] > value
                                  for key, value in before["c32_by_block"].items())),
        "post_k8": after["post_k8"] > before["post_k8"],
    }
    if after["post_attention"] is not None:
        result["post_attention"] = after["post_attention"] > before["post_attention"]
        result["post_contiguous_k8"] = after["post_contiguous_k8"] > before["post_contiguous_k8"]
    if after["c128_by_block"] is not None:
        result["c128_all_twelve"] = (
            set(after["c128_by_block"]) == set(before["c128_by_block"])
            and len(after["c128_by_block"]) == 12
            and all(after["c128_by_block"][key] > value
                    for key, value in before["c128_by_block"].items()))
    if after["c256_by_block"] is not None:
        result["c256_all_sixteen"] = (
            set(after["c256_by_block"]) == set(before["c256_by_block"])
            and len(after["c256_by_block"]) == 16
            and all(after["c256_by_block"][key] > value
                    for key, value in before["c256_by_block"].items()))
    if after["pre_fused"] is not None:
        result["pre_attention_tail"] = after["pre_fused"] > before["pre_fused"]
    if after["c512_decoder"] is not None:
        result["c512_decoder_all_eight"] = (
            set(after["c512_decoder"]) == set(before["c512_decoder"]) == set(range(8))
            and all(after["c512_decoder"][index] > before["c512_decoder"][index]
                    for index in range(8)))
    return result
