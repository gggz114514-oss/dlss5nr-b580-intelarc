"""Use the verified idle-game one-file installer for the scoped local observer."""
from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("flash_native_installer", PROJECT / "tools/install_periodic_flash_state_proof_v1_cpu.py")
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
native.SOURCE = PROJECT / "artifacts/periodic-flash-local-decl-v4-source-20261001"
native.BUILD = native.PERF / "periodic-flash-local-decl-v4-native-build"
native.EXPECTED_BEFORE = "3f0667c5490bee16a5a721a8154a445c5546d6c7cc7eaca46e8416e76ab45647"
native.BACKUP = native.PERF / "periodic-flash-local-decl-v4-installed-20261001"


if __name__ == "__main__":
    native.main()
