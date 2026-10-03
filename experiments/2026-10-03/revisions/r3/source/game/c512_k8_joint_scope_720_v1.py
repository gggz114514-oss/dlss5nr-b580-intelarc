"""Opt-in 720p fast candidate combining byte-exact C512 QKV and lossy K8.

The C512 library scope owns the provider first. A narrower pre-K8 wrapper then
delegates all other matrices to that existing scope; post K8 has a separate
active call site. Keep this file out of the exact backend and out of game
defaults until the combined reset/history video and frame timing are reviewed.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import c512_qkv_library_16_v1 as c512
import native_k8_fp16_v1 as pre_k8
import native_k8_active_720_v1 as post_k8
import post_attention_k8_combined_v1 as active_post


@contextmanager
def installed(session):
    provider = session._stack.provider
    model = session._stack.model
    if provider.mode != "fp16_xmx" or "dense" in provider.__dict__:
        raise RuntimeError("Joint scope requires an unmodified FP16 provider")
    if tuple(model.pre.front_weight.shape) != (16, 32):
        raise RuntimeError("Unexpected pre K8 weight")
    original_post = active_post._contiguous_cropped_head
    k8_calls = {"pre": 0, "post": 0}

    with c512.installed(session) as c512_calls:
        delegated_dense = provider.dense

        def dense(self, a, w, *, chunk_k, initial=None, **kwargs):
            if w is model.pre.front_weight:
                if chunk_k != 8 or initial is not None or kwargs:
                    raise RuntimeError("Unexpected pre K8 dispatch")
                result = pre_k8.dot(a, w, site="pre")
                self.record("native_k8_fp16_pre")
                k8_calls["pre"] += 1
                return result
            return delegated_dense(a, w, chunk_k=chunk_k,
                                   initial=initial, **kwargs)

        def post_head(a, weight, output_size):
            k8_calls["post"] += 1
            return post_k8.post_head(a, weight, output_size)

        replacement = MethodType(dense, provider)
        provider.dense = replacement
        if active_post._contiguous_cropped_head is not original_post:
            provider.dense = delegated_dense
            raise RuntimeError("Post head was already replaced")
        active_post._contiguous_cropped_head = post_head
        try:
            yield c512_calls, k8_calls
        finally:
            if active_post._contiguous_cropped_head is not post_head or \
                    provider.dense is not replacement:
                raise RuntimeError("Joint C512/K8 scope changed during capture")
            active_post._contiguous_cropped_head = original_post
            provider.dense = delegated_dense
