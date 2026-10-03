"""Select the RE8 fast-only source overlay before loading the NR model.

Call after the portable runtime's ``bootstrap()`` and before any nr_backend or
runtime module import.  This does not replace sys.modules entries or patch a
live backend; a new process and new graph are required for each selection.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FAMILIES = frozenset({"pre", "c32", "c64", "c128", "c256", "c512", "vit", "post"})


def _manifest(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        name, digest = line.split()
        if name in entries or len(digest) != 64:
            raise RuntimeError(f"Malformed source manifest: {path}")
        entries[name] = digest
    return entries


def _verify_source(directory: Path, manifest: Path) -> None:
    if not directory.is_dir():
        raise RuntimeError(f"Audited fast source missing: {directory}")
    for name, digest in _manifest(manifest).items():
        source = directory / name
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Fast source has changed since overlay snapshot: {source}")


def activate(runtime: str | Path, selected: str = "") -> dict:
    runtime = Path(runtime).resolve()
    selected = selected.strip().lower()
    flags = frozenset(x.strip() for x in selected.split(",") if x.strip())
    if flags == {"all"}:
        flags = FAMILIES
    elif flags - FAMILIES:
        raise ValueError(f"Unknown unround families: {sorted(flags - FAMILIES)}")
    if any(name == "nr_backend" or name.startswith("nr_backend.") for name in sys.modules):
        raise RuntimeError("NR backend was imported before fast overlay selection")
    # The installed runtime is evidence, not a write target.  Verify every
    # copied Python source still matches the snapshot before running a frame.
    _verify_source(runtime / "fast/backend/nr_backend", HERE / "baseline-sha256.txt")
    _verify_source(runtime / "modules", HERE / "modules-baseline-sha256.txt")
    if not (runtime / "exact/backend/nr_backend").is_dir() or not (runtime / "data").is_dir():
        raise RuntimeError("RE8 runtime exact comparison tree or data directory missing")
    os.environ["NR_FAST_UNROUND_RUNTIME_ROOT"] = str(runtime)
    os.environ["NR_FAST_UNROUND"] = ",".join(sorted(flags))
    for directory in (HERE / "nr_backend", HERE / "modules", HERE.parent.parent / "game"):
        if not directory.is_dir():
            raise RuntimeError(f"Experiment source missing: {directory}")
    # Imported nr_backend is a package whose parent is HERE, not the package
    # directory itself.  Keep all three experiment paths ahead of G runtime.
    for directory in (HERE.parent.parent / "game", HERE / "modules", HERE):
        sys.path.insert(0, str(directory))
    origin = importlib.util.find_spec("nr_backend")
    expected = (HERE / "nr_backend/__init__.py").resolve()
    if origin is None or Path(origin.origin).resolve() != expected:
        raise RuntimeError(f"NR overlay not selected: {getattr(origin, 'origin', None)}")
    for module_name in ("native_cubic_adapters_v1", "three_structure_combo_v1"):
        spec = importlib.util.find_spec(module_name)
        root = HERE / "modules" if module_name.startswith("native_") else HERE.parent.parent / "game"
        if spec is None or Path(spec.origin).resolve() != root / f"{module_name}.py":
            raise RuntimeError(f"Experiment module {module_name} resolved outside worktree")
    return {"runtime": str(runtime), "backend": str(expected),
            "selected": sorted(flags), "game": str(HERE.parent.parent / "game")}
