"""CPU-only retry with NOMINMAX; preserves the frozen runner and its source."""
import hashlib
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / 'artifacts/terminal-sync-qualified-gpu-runner-v1-20261002/build_sidecar_cpu.py'
EXPECTED = '09afc205a8ade4977a2bbe9615078d6da82fe18d6dc3b2bdb728c3b884ebddb0'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def main():
    original = SOURCE.read_bytes()
    if digest(original) != EXPECTED:
        raise RuntimeError('Frozen CPU build driver changed')
    source = original.decode('utf-8')
    old = "'-fsycl','-std=c++20','-O2','-fexceptions','-fcxx-exceptions',"
    new = "'-fsycl','-std=c++20','-O2','-fexceptions','-fcxx-exceptions','-DNOMINMAX',"
    marker = "report=dict(status='CPU_SIDECAR_BUILD_FAILED_GPU_UNTESTED',jobs_cap=14,commands=commands,"
    attribution = "report=dict(main_build_driver=" + repr(dict(
        path=str(Path(__file__).resolve()), sha256=digest(Path(__file__).read_bytes()),
        original_driver_path=str(SOURCE), original_driver_sha256=EXPECTED,
        repair='Add -DNOMINMAX at compile time: Windows max macro collided with numeric_limits<uint64_t>::max()',
        frozen_runner_modified=False, GPU_executed=False, native_DLL_loaded=False)) + ",status='CPU_SIDECAR_BUILD_FAILED_GPU_UNTESTED',jobs_cap=14,commands=commands,"
    if source.count(old) != 1 or source.count(marker) != 1:
        raise RuntimeError('CPU build repair anchors changed')
    source = source.replace(old,new).replace(marker,attribution)
    exec(compile(source,str(SOURCE),'exec'),{'__name__':'__main__','__file__':str(SOURCE)})

if __name__ == '__main__':
    main()
