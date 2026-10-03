"""Install backend sidecars: int8-tree-only replacements for shared ``nr_backend`` code.

Why a sidecar
-------------
``nr-b580/backend/nr_backend/*.py`` and ``nr-b580-int8/backend/nr_backend/*.py``
are kept byte-identical on purpose (``rows_paths_v1.duplicate_source_agreement``),
and the exact line's precompile package lists the 35 *exact-tree* sources by
sha256 (``exact-v1/catalog.json``, asserted at ``exact_kernel_package_v2.py:212``).
A change written into the shared file has to be written twice, and that second
write invalidates the package: the 2026-09-19 quanttrim landing did exactly that
and left 11 exact-line entries unable to start.

A sidecar keeps the shared file untouched. It lives in the int8 tree only, holds
the modified function, and is bound onto the shared class at import time of the
public runtime. Both trees stay byte-identical, so the duplicate-tree invariant
and the package manifest are both untouched by construction.

Why the binding survives an earlier import
------------------------------------------
``executor.py`` binds the class at import time (``from .vit_block import
VitBlock``), so rebinding the *class* would not reach it. Method lookup does:
``block.forward_boundaries`` resolves through the class ``__dict__`` at call
time. A sidecar therefore replaces the function attribute on the class object and
asserts the identity of what it replaced -- the same idiom the scoped adapters in
``experimental/fused_vit_*_adapter_v1.py`` already use.

Staleness guard
---------------
A sidecar is a copy, so it can go stale. Each sidecar records the sha256 of the
shared source it shadows and refuses to bind when the live file differs. A stale
sidecar never silently runs; the original path stays in use.

Gating
------
``NR_BACKEND_SIDECARS`` names the sidecar modules to install. Empty / ``off``
installs nothing, and an unknown name installs nothing either -- a typo must not
silently enable one. Installation failure is reported and swallowed: the product
keeps running on the original path.
"""

from __future__ import annotations

import importlib
import os
import sys

ENV = 'NR_BACKEND_SIDECARS'
DEFAULT = 'off'

_INSTALLED = {}


def _say(message):
    print('[sidecar] %s' % message, file=sys.stderr, flush=True)


def install(verbose=True):
    """Install every sidecar named by ``NR_BACKEND_SIDECARS``. Never raises."""
    raw = os.environ.get(ENV, DEFAULT).strip()
    if raw.lower() in ('', 'off', 'none'):
        if verbose:
            _say('off (%s=%r)' % (ENV, raw))
        return dict(_INSTALLED)
    names = tuple(item.strip() for item in raw.split(',') if item.strip())
    for name in names:
        if name in _INSTALLED:
            continue
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            _say('%s not importable (%s: %s); original path in use'
                 % (name, type(exc).__name__, exc))
            continue
        hook = getattr(module, 'install', None)
        if hook is None:
            _say('%s exposes no install(); skipped' % name)
            continue
        try:
            record = hook(verbose=verbose)
        except Exception as exc:  # noqa: BLE001
            _say('%s failed to install (%s: %s); original path in use'
                 % (name, type(exc).__name__, exc))
            continue
        _INSTALLED[name] = record or {}
        if verbose:
            _say('on: %s -> %s (mode=%s)' % (name, _INSTALLED[name].get('patched', '?'),
                                             _INSTALLED[name].get('mode', '?')))
    return dict(_INSTALLED)


def report():
    """Installed sidecars and their live counters, for the frame report."""
    rows = []
    for name in sorted(_INSTALLED):
        module = sys.modules.get(name)
        hook = getattr(module, 'report', None) if module is not None else None
        try:
            row = hook() if hook is not None else dict(_INSTALLED[name])
        except Exception as exc:  # noqa: BLE001
            row = dict(error='%s: %s' % (type(exc).__name__, exc))
        rows.append(dict(module=name, **(row or {})))
    return dict(installed=bool(_INSTALLED), env=ENV, names=sorted(_INSTALLED),
                sidecars=rows)
