"""Use the verified one-file installer for the recorded-state correction."""
from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("flash_native_installer", PROJECT / "tools/install_periodic_flash_state_proof_v1_cpu.py")
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
native.SOURCE = PROJECT / "artifacts/periodic-flash-alias-state-v2-source-20261001"
native.BUILD = native.PERF / "periodic-flash-alias-state-v2-native-build"
native.EXPECTED_BEFORE = "6ee8c543e4284bb23763a55e644c25fa45fc0c672601d8996e75d0fc96c91a7e"
native.BACKUP = native.PERF / "periodic-flash-alias-state-v2-installed-20261001"


if __name__ == "__main__":
    native.main()
