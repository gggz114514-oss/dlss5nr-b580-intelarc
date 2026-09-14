"""Keep host Comfy/oneAPI tooling out of the dedicated NR worker."""
import ctypes
import os
from pathlib import Path
import sys


def isolate():
    runtime_bin = Path(sys.executable).resolve().parent / 'Library/bin'
    if not (runtime_bin / 'sycl9.dll').is_file():
        raise RuntimeError(f'NR 运行库不完整：{runtime_bin / "sycl9.dll"}')
    # Triton finds icpx on PATH before checking the pinned intel-sycl-rt wheel.
    # Exclude compiler directories as well as system oneAPI runtime directories.
    kept = []
    removed = []
    for entry in os.environ.get('PATH', '').split(os.pathsep):
        path = Path(entry.strip('"'))
        if not entry or 'oneapi' in entry.lower() or (path / 'icpx.exe').is_file():
            removed.append(entry)
        else:
            kept.append(entry)
    os.environ['PATH'] = os.pathsep.join([str(runtime_bin), *kept])
    keys = ('ONEAPI_DEVICE_SELECTOR', 'ONEAPI_ROOT', 'TRITON_INTEL_SYCL_COMPILER',
            'TRITON_INTEL_DEVICE_EXTENSIONS')
    cleared = [key for key in keys if key in os.environ]
    for key in keys:
        os.environ.pop(key, None)
    # Hold both handles until process exit. Resolve SYCL and its dependencies
    # from the verified wheel before OpenVINO or any host directory is added.
    directory = os.add_dll_directory(str(runtime_bin))
    library = ctypes.WinDLL(str(runtime_bin / 'sycl9.dll'))
    return directory, library, dict(runtime_bin=str(runtime_bin), removed_path=removed, cleared_keys=cleared)


def loaded_libraries():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel.GetModuleHandleW.restype = ctypes.c_void_p
    kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
    result = {}
    for name in ('sycl9.dll', 'ur_loader.dll', 'ur_adapter_level_zero_v2.dll',
                 'ur_adapter_level_zero.dll', 'ur_adapter_opencl.dll'):
        handle = kernel.GetModuleHandleW(name)
        if handle:
            buffer = ctypes.create_unicode_buffer(32768)
            kernel.GetModuleFileNameW(handle, buffer, len(buffer))
            result[name] = buffer.value
    return result
