"""Sidecar: instrumented ``VitBlock.forward_boundaries`` for the fast line only.

This is a **copy** of ``nr-b580-int8/backend/nr_backend/vit_block.py::
VitBlock.forward_boundaries``. It lives outside ``nr_backend/`` on purpose: the
shared file stays byte-identical in both trees, so
``rows_paths_v1.duplicate_source_agreement()`` and the exact line's precompile
package (``exact-v1/catalog.json``, 35 exact-tree sources) are both untouched.

What it is for
--------------
The landed cut (``quantize_fp8_grid``, ``reference/quanttrim_land_v1/``) proved
that at the block-input / residual / skip call sites the incoming tensor is
already on the E4M3FN grid, so the conversion is a value-level no-op. The ViT
block has six further ``q(`` sites (``vit_block.py`` 74/75/77/78x2/81) that the
landing did **not** cover, because nobody had measured whether they are
grid-resident too. This sidecar measures exactly that, on the product path,
without touching any shared file.

Mode
----
``NR_VIT_BLOCK_GRID`` unset / ``off`` -> probe. Arithmetic is **identical** to the
original (it still calls the real ``quantize_fp8``); only counters are added, so
the picture cannot change and no 243-frame picture gate is owed.

``NR_VIT_BLOCK_GRID=on`` -> the two cheap preconditions the landed cut uses
(``dtype == float16`` and contiguous) short-circuit the conversion and record the
``fp8`` arithmetic dispatch instead, so the ledger stays unchanged. Only turn it
on once the probe has shown the sites are eligible.

Staleness guard
---------------
``SOURCE_SHA256`` is the digest of the shared file this copy was taken from.
``install()`` refuses to bind when the live file differs, so a copy that has
drifted from its original never silently runs.

Name resolution (a copy must resolve names the way the original does)
--------------------------------------------------------------------
``vit_block.py`` calls ``split_k_projection(...)`` as a **module global**, so a
component installed inside a scope can replace
``nr_backend.vit_block.split_k_projection`` and the original body picks the
replacement up **at call time**. ``FusedVitProjection`` does exactly that
(``fused_vit_projection_v1.py:81``). Binding that name at import time would
silently bypass the replacement: the copy would run the unfused projection,
which changes the arithmetic **and** asks for kernel artifacts the reviewed
cache does not hold -- observed as ``Missing fast artifact: _matmul`` when this
copy ran with ``NR_BACKEND_SIDECARS=sidecar_vit_block_v1``. Every name the
original loads as a global is therefore resolved through ``_shared`` here too.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import torch
from torch import Tensor

import nr_backend.vit_block as _shared
from nr_backend.execution import record_arithmetic_dispatch

VIT_BLOCK = Path(_shared.__file__).resolve()

# Digest of the shared source this copy was taken from.
#   nr-b580-int8/backend/nr_backend/vit_block.py
SOURCE_SHA256 = 'd6d7c30208ffe94990eb60d47e882d0945d94316fb842891616d6c1d390e1b4d'

ENV = 'NR_VIT_BLOCK_GRID'
# Check value identity every Nth call per site. A frame gives 8 calls per site
# (one per ViT block), so 8 means roughly one check per site per frame -- enough
# to separate "always identical" from "sometimes identical" without paying a
# device sync on every call.
IDENTITY_SAMPLE = 8

SITES = ('features', 'mlp', 'query', 'key', 'value', 'output')
STATS = {name: dict(calls=0, fp16_contiguous=0, identity_checked=0,
                    identity_equal=0, short=0) for name in SITES}

_ORIGINAL = None


def _say(message):
    print('[sidecar:vit_block] %s' % message, file=sys.stderr, flush=True)


def _grid_on():
    return os.environ.get(ENV, 'off').strip().lower() in ('on', '1', 'true', 'yes')


def _q(site, value):
    """``quantize_fp8`` at one ViT call site: counted, optionally short-circuited."""
    row = STATS[site]
    row['calls'] += 1
    eligible = value.dtype == torch.float16 and value.is_contiguous()
    if eligible:
        row['fp16_contiguous'] += 1
    if _grid_on() and eligible:
        row['short'] += 1
        record_arithmetic_dispatch('fp8')
        return value
    # ``_shared.q``, not a module-level binding: the original resolves ``q`` as a
    # global at call time, so the copy has to as well (see "Name resolution").
    result = _shared.q(value)
    # The probe's evidence: how often the conversion returns the input unchanged.
    # Sampled, because torch.equal forces a device sync. Only meaningful while
    # short-circuiting is off -- once it is on this path is not taken.
    if row['calls'] % IDENTITY_SAMPLE == 1:
        row['identity_checked'] += 1
        row['identity_equal'] += int(bool(torch.equal(result, value)))
    return result


def forward_boundaries(self, features: Tensor) -> tuple[Tensor, ...]:
    if features.ndim != 2 or features.shape[0] not in (64, 96, 128, 640, 960) or features.shape[1] != 1024:
        raise ValueError('Expected observed 64/96/128/640/960 token contract, C1024')
    tokens = features.shape[0]
    x = _q('features', features); hidden = _shared.cubic_quantize(_shared.dot(x, self.expand, chunk_k=16))
    mlp = _q('mlp', _shared.split_k_projection(hidden, self.contract, (x * self.ffn_skip).half()))
    z = (_shared.dot(mlp[:, :512], self.qkv_weight[:512], chunk_k=16) + _shared.dot(mlp[:, 512:], self.qkv_weight[512:], chunk_k=16)).half().reshape(tokens, 32, 3, 32)
    query = _q('query', (_shared.normalize_c32(z[:, :, 0]) * 5.65625).half() * self.query_scale[None, :, None])
    key = _q('key', _shared.normalize_c32(z[:, :, 1])); value = _q('value', z[:, :, 2])
    Q, K, V = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
    attended = _shared.vit_attention(Q, K, V).transpose(0, 1).reshape(tokens, 1024)
    output = _q('output', _shared.split_k_projection(attended, self.projection, (mlp * self.attn_skip).half()))
    return hidden, mlp, query, key, value, attended, output


def install(verbose=True):
    """Bind ``forward_boundaries`` onto ``VitBlock``. Raises when the copy is stale."""
    global _ORIGINAL
    if _ORIGINAL is not None:
        return report()
    digest = hashlib.sha256(VIT_BLOCK.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError('shared %s changed (%s != %s); this copy is stale'
                           % (VIT_BLOCK.name, digest[:12], SOURCE_SHA256[:12]))
    # ``_shared.VitBlock``: the class is a module global of the shared file too.
    original = _shared.VitBlock.forward_boundaries
    _shared.VitBlock.forward_boundaries = forward_boundaries
    if _shared.VitBlock.forward_boundaries is not forward_boundaries:
        _shared.VitBlock.forward_boundaries = original
        raise RuntimeError('VitBlock.forward_boundaries could not be rebound')
    _ORIGINAL = original
    if verbose:
        _say('bound %s.%s (mode=%s, source=%s)'
             % (_shared.VitBlock.__name__, forward_boundaries.__name__,
                'grid' if _grid_on() else 'probe', digest[:12]))
    return report()


def uninstall():
    """Restore the shared implementation and return whether anything was bound."""
    global _ORIGINAL
    if _ORIGINAL is None:
        return False
    _shared.VitBlock.forward_boundaries = _ORIGINAL
    _ORIGINAL = None
    return True


def report():
    """Counters for the frame report. ``installed`` is False when never bound."""
    return dict(installed=_ORIGINAL is not None, patched='VitBlock.forward_boundaries',
                source=str(VIT_BLOCK), source_sha256=SOURCE_SHA256,
                mode='grid' if _grid_on() else 'probe',
                identity_sample=IDENTITY_SAMPLE,
                sites={name: dict(STATS[name]) for name in SITES})
