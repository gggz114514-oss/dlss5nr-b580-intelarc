"""Install the normal-launch profile after the scoped v4 has live acceptance."""
from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "flash_native_installer", PROJECT / "tools/install_periodic_flash_state_proof_v1_cpu.py")
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
native.SOURCE = PROJECT / "artifacts/periodic-flash-cp-default-v5-source-20261001"
native.BUILD = native.PERF / "periodic-flash-cp-default-v5-native-build"
native.EXPECTED_BEFORE = "797e8c6fbee2f39017fccd4319e8e32b52d190180dac9299f34e724f6840a7a6"
native.BACKUP = native.PERF / "periodic-flash-cp-default-v5-installed-20261001"


if __name__ == "__main__":
    native.main()
