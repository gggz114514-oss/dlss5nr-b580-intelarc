"""Fast-only rounding policy; game selection is scoped to one NR session.

The environment remains the default for independent offline experiments. A game
mode switch closes its old model and graph before selecting another policy.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Set
from .pre_mlp import quantize_fp8 as _native_round

FAMILIES = frozenset({"pre", "c32", "c64", "c128", "c256", "c512", "vit", "post"})
_raw = os.environ.get("NR_FAST_UNROUND", "").strip().lower()
_parts = frozenset(part.strip() for part in _raw.split(",") if part.strip())
if "all" in _parts:
    if _parts != {"all"}:
        raise ValueError("NR_FAST_UNROUND=all cannot be combined with family names")
    _default = FAMILIES
else:
    if _parts - FAMILIES:
        raise ValueError(f"Unknown NR_FAST_UNROUND families: {sorted(_parts - FAMILIES)}")
    _default = _parts


_active = ContextVar("nr_fast_unround_families", default=_default)


class _ActiveFamilies(Set):
    def __contains__(self, item):
        return item in _active.get()

    def __iter__(self):
        return iter(_active.get())

    def __len__(self):
        return len(_active.get())


# Importers hold this object, so selection remains visible to their runtime
# launch arguments without replacing already-imported module globals.
ENABLED = _ActiveFamilies()


@contextmanager
def selected_game_variant(variant: str):
    if variant not in ("standard", "unrounded"):
        raise ValueError("Unknown game fast backend variant")
    token = _active.set(FAMILIES if variant == "unrounded" else frozenset())
    try:
        yield
    finally:
        _active.reset(token)


def round_activation(family: str, value):
    if family not in FAMILIES:
        raise ValueError(f"Unknown activation family: {family}")
    if family not in ENABLED:
        return _native_round(value)
    # The experiment removes a second E4M3 quantize/dequantize trip.  FP16
    # model tensors and the native half arithmetic are left unchanged.
    import torch
    if not isinstance(value, torch.Tensor) or value.dtype not in (torch.float16, torch.float32) or value.device.type != "xpu":
        raise ValueError(f"Unrounded {family} activation must be floating XPU tensor")
    # Native E4M3 round returns FP16. Keep that type boundary even when the
    # extra E4M3 lattice conversion is removed.
    return value.half()


def round_pre(value): return round_activation("pre", value)
def round_c32(value): return round_activation("c32", value)
def round_c512(value): return round_activation("c512", value)
def round_vit(value): return round_activation("vit", value)
def round_post(value): return round_activation("post", value)


def round_multihead(channels: int, value):
    if channels not in (64, 128, 256, 512):
        raise ValueError(f"Unexpected multihead channels: {channels}")
    return round_activation("c512" if channels == 512 else f"c{channels}", value)


def cubic_activation_family(family: str, value):
    if family not in FAMILIES:
        raise ValueError(f"Unknown cubic family: {family}")
    if family not in ENABLED:
        from .pre_mlp import cubic_quantize
        return cubic_quantize(value)
    import torch
    if not isinstance(value, torch.Tensor) or value.dtype not in (torch.float16, torch.float32) or value.device.type != "xpu":
        raise ValueError(f"Unrounded {family} cubic requires floating XPU tensor")
    from .triton_cubic_fp8 import direct_cubic_fp8
    from .execution import record_arithmetic_dispatch
    output = direct_cubic_fp8(value.half(), round_output=False)
    record_arithmetic_dispatch("cubic_fp8")
    return output


def describe():
    return {"enabled": sorted(ENABLED), "selection": ",".join(sorted(ENABLED)) or "baseline"}


_attention_owner = ContextVar("nr_fast_attention_owner", default=None)


@contextmanager
def attention_owner(family: str):
    if family not in FAMILIES:
        raise ValueError(f"Unexpected attention owner: {family}")
    token = _attention_owner.set(family)
    try:
        yield
    finally:
        _attention_owner.reset(token)


def current_attention_owner():
    return _attention_owner.get()
