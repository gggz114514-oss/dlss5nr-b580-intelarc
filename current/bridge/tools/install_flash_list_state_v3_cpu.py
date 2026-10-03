"""Use the verified idle-game one-file installer for the local-state refresh."""
from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("flash_native_installer", PROJECT / "tools/install_periodic_flash_state_proof_v1_cpu.py")
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
native.SOURCE = PROJECT / "artifacts/periodic-flash-list-state-v3-source-20261001"
native.BUILD = native.PERF / "periodic-flash-list-state-v3-native-build"
native.EXPECTED_BEFORE = "8332bd30394266c1d81c25d4390e8332e25e05807b44cdb55b230197f6c3a3ed"
native.BACKUP = native.PERF / "periodic-flash-list-state-v3-installed-20261001"


if __name__ == "__main__":
    native.main()
