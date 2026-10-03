"""Owned, versioned copy of the previously authenticated cubic+FP8 half-bit LUT."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

_runtime_root = os.environ.get('NR_FAST_UNROUND_RUNTIME_ROOT')
ROOT = ((Path(_runtime_root).resolve() / 'data/reference') if _runtime_root else
        Path(__file__).resolve().parents[1] / 'data/reference')
TABLE = ROOT / 'experimental/cubic-fp8-lut-v3.npy'
TABLE_SHA = '7eedd128e03576a99a21e52cf618aec3be445bad134a4dd62002ae024bbbd651'
BUFFER = '_cubic_fp8_lut_bits'


def register(model):
    """Call on a fresh model before constant sharing and graph construction."""
    if hasattr(model, BUFFER):
        raise ValueError('LUT already registered; do not replace immutable constants')
    if torch.is_inference_mode_enabled():
        raise ValueError('Register constants outside inference mode for version tracking')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(TABLE) == TABLE_SHA
    for name, digest in [('cubic-fp8-fused-cpu-v3.json', 'e392b661add80f01b0db4de802bc0a50ed5ea05b098d48354c2a13e611654f63'),
                         ('cubic-fp8-fused-xpu-v3.json', 'c79a820724e0d7e467b8cd963971990fa163971da4490bc497ecdac4aca4ae1e')]:
        path = ROOT / name
        assert sha(path) == digest
        report = json.loads(path.read_text())
        assert report['all_byte_equal'] and report['table_file_sha256'] == TABLE_SHA
    table = np.load(TABLE, allow_pickle=False)
    assert table.shape == (65536,) and table.dtype == np.dtype('<f2')
    value = torch.from_numpy(table.view('i2').copy()).to(model.pre.front_weight.device)
    model.register_buffer(BUFFER, value)
    return value


class Constant:
    def __init__(self, model):
        self.model = model
        self.value = getattr(model, BUFFER)
        value = self.value
        if value.dtype != torch.int16 or value.shape != (65536,) or not value.is_contiguous():
            raise ValueError('Expected owned contiguous int16 cubic LUT')
        self.version = value._version
        self.pointer = value.data_ptr()

    def require(self):
        value = getattr(self.model, BUFFER)
        if value is not self.value or value._version != self.version or value.data_ptr() != self.pointer:
            raise RuntimeError('Cubic LUT changed; create a new immutable session')
        return value
