"""Immutable second C32 cubic table for the isolated FP8-unround fastline.

The existing cubic+E4M3 table stays in the model for other consumers. Register
this independent table before GraphFront collects model buffers; no live table
or graph is ever mutated after capture.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch


BUFFER = "_cubic_half_lut_bits"
_runtime_root = os.environ.get("NR_FAST_UNROUND_RUNTIME_ROOT")
ROOT = ((Path(_runtime_root).resolve() / "data/reference") if _runtime_root else
        Path(__file__).resolve().parents[1] / "data/reference")
TABLE = ROOT / "experimental/cubic-half-lut-v1.npy"
TABLE_SHA256 = "3867ff3c68c5f0af6d8c51c8f5eb4a8957edcbad04a2fdce3f54d8724249fef5"


def register(model):
    if hasattr(model, BUFFER):
        raise ValueError("Unrounded cubic LUT already registered")
    if torch.is_inference_mode_enabled():
        raise ValueError("Register table before inference/graph construction")
    source = TABLE.resolve(strict=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != TABLE_SHA256:
        raise ValueError("Unrounded cubic LUT SHA256 mismatch")
    table = np.load(source, allow_pickle=False)
    if table.shape != (65536,) or table.dtype != np.dtype("<f2"):
        raise ValueError("Unrounded cubic LUT must contain 65536 little-endian FP16 values")
    value = torch.from_numpy(table.view("<i2").copy()).to(model.pre.front_weight.device)
    model.register_buffer(BUFFER, value)
    return value


class Constant:
    def __init__(self, model):
        self.model = model
        self.value = getattr(model, BUFFER)
        if self.value.dtype != torch.int16 or self.value.shape != (65536,) or not self.value.is_contiguous():
            raise ValueError("Expected an owned contiguous int16 half-cubic LUT")
        self.version = self.value._version
        self.pointer = self.value.data_ptr()

    def require(self):
        value = getattr(self.model, BUFFER)
        if value is not self.value or value._version != self.version or value.data_ptr() != self.pointer:
            raise RuntimeError("Unrounded cubic LUT changed after graph construction")
        return value
