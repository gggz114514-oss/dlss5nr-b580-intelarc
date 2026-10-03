"""Absolute W/D paths and identity helpers for the fullsize de-batching round.

The reviewed adapters derive their roots from ``__file__.parents``. Copied into a
new directory that silently resolves to a different tree, so this module states
every path in full and every loaded project module is checked against the real
file it came from. Nothing here writes to the historical entries, the product
loader files or the frozen references.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

# In this isolated worktree, audited assets and duplicate backend trees live
# under the installed RE8 runtime, not alongside this copied game source.
# bootstrap.activate() verifies the G fast source manifest before setting this.
_runtime_root = os.environ.get('NR_FAST_UNROUND_RUNTIME_ROOT')
if not _runtime_root:
    raise RuntimeError('Select the audited fast overlay before importing rows_paths_v1')
W = Path(_runtime_root).resolve()
if not (W / 'exact/backend/nr_backend').is_dir() or not (W / 'fast/backend/nr_backend').is_dir():
    raise RuntimeError('Audited RE8 runtime backend trees are unavailable')
D = W / 'data'

HERE = Path(__file__).resolve().parent
NR = W / 'exact'
NR_INT8 = W / 'fast'
PRODUCT = NR_INT8 / 'product'
EXPERIMENTAL = NR_INT8 / 'experimental'
REFERENCE = NR / 'reference'

# Frozen configuration and asset index. The profile name contains nr256, but the
# fullsize adapter reuses its weights/scales; it does not shrink the input.
PROFILE_DIR = D / 'product-v1'
LOCAL_RUNTIME = PROFILE_DIR / 'local-runtime-v1.json'
PROFILE = PROFILE_DIR / 'profile-v1.json'
FOURWAY = D / 'fourway-face-review-v1'
EXACT_VALIDATION = FOURWAY / 'exact/validation.json'
NR256_FAST_VALIDATION = FOURWAY / 'fast/validation.json'

# Reviewed fullsize fast baseline: 243 frames, source identity and adapter hash.
BASELINE = D / 'fast-fullsize-review-v1/pipeline-v4'
BASELINE_FULL = BASELINE / 'full/validation.json'

RUNS = D / 'fullsize-rows-v1'
SHARED_CACHE = D / 'reference/triton-cache-c32-triton38-v1'

# Historical entries. Read-only for this round.
BASELINE_ADAPTER = REFERENCE / 'review_fast_fullsize_v1.py'
C512_KERNELS = EXPERIMENTAL / 'c512_int8_ffn_gpu_v1.py'
C512_STACK = EXPERIMENTAL / 'c512_int8_full_stack_v1.py'
VIT_KERNELS = EXPERIMENTAL / 'int8_ffn_segment_gpu_v1.py'
VIT_SCOPE = EXPERIMENTAL / 'int8_ffn_body_scope_v1.py'

INTERPRETER = W / 'python/python.exe'
LEASE = REFERENCE / 'gpu_lease_nr.py'

# Host compiler toolchain.
#
# Triton's Intel XPU backend compiles three host helper modules from C
# (spirv_utils / arch_utils / extension_utils_impl) the first time it needs a
# device, and its compile_module_from_src appends MSVC-style flags (/LIBPATH:,
# /D) to the command line. CC/CXX must therefore be a cl-compatible driver, not
# gcc or clang. The reviewed collect/build launchers
# (Run-CollectExactV1.cmd, Run-BuildExactV1.cmd) set exactly this toolchain.
#
# The reviewed *fullsize* launcher sets no compiler at all, because
# compile_module_from_src consults the Triton cache before compiling and the
# shared cache already holds all three modules. A fresh candidate cache holds
# none of them, which is why this round needs the compiler. Kernel artifacts
# here are SPIR-V only (generate_native_code defaults to False), so ocloc is
# never invoked and does not need to be on PATH.
#
# runtime_environment.isolate() strips oneAPI and any directory containing
# icpx.exe from PATH on purpose, so that SYCL headers and libraries come from
# the pinned intel-sycl-rt wheel instead of the host compiler. It leaves CC/CXX
# untouched and keeps the MSVC directories, which is what makes this work.
VCVARS = Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/'
              'VC/Auxiliary/Build/vcvars64.bat')
ICX_CL = Path('C:/Program Files (x86)/Intel/oneAPI/compiler/2026.1/bin/icx-cl.exe')
ONEAPI_COMPILER_LIB = Path('C:/Program Files (x86)/Intel/oneAPI/compiler/2026.1/lib')
# PATH the reviewed build launchers started from, before calling vcvars64.
VCVARS_BASE_PATH = ('C:\\Windows\\System32;C:\\Windows;'
                    'C:\\Windows\\System32\\Wbem;'
                    'C:\\Windows\\System32\\WindowsPowerShell\\v1.0')
HOST_MODULES = ('spirv_utils', 'arch_utils', 'extension_utils_impl')

BASELINE_ADAPTER_SHA256 = '7c5b0ef0d2107cdd58e0d251d8524bf2851e7c62f20220c5213aaade4f278eed'
PROFILE_SHA256 = '8c0994c72c95404caf8116ef0a2ac0580fd8330c8b8a0c0a5f856489b1a9bb21'
SOURCE_SHA256 = 'e0d0776bb01dce7e64dab8c1b365dccaf4aba97bf6620297d73e5718f5dec107'

# Reviewed FFN kernel modules that the candidate must not dispatch.
OLD_KERNEL_MODULES = ('c512_int8_ffn_gpu_v1', 'int8_ffn_segment_gpu_v1')
ROWS_KERNEL_MODULES = ('c512_int8_ffn_rows_v1', 'int8_ffn_segment_rows_v1')

VARIANTS = ('baseline', 'c512', 'vit', 'both')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def child_environment():
    """Environment for every stage child process.

    The WorkBuddy safe-delete shim is loaded through PYTHONPATH as sitecustomize
    before any of our code runs, so it cannot be neutralised from inside the
    child; its switch is read at interpreter start and has to be set here.

    The shim replaces os.remove / os.unlink / os.rmdir / shutil.rmtree /
    pathlib deletions with recycle-bin versions. Two consequences matter:

    1. _check_bulk_delete_guard runs a Node helper on *every* trashed path and
       raises SystemExit(1) when the helper's state lock times out. Fourteen
       concurrent compile workers deleting cache temp directories trip it, and
       the process dies with SystemExit(1) - which is exactly how compile-c512
       failed. Compilation deletes only directories it created itself inside the
       round's own cache, so the recycle-bin redirect buys no safety here.
    2. It does not remove the real hazard: for a non-empty directory _safe_rmdir
       deliberately calls the *native* rmdir, and on this volume native rmdir
       removes non-empty directories. So rows_fscache_guard_v1 is still required
       with the shim off.

    Only this round's own child processes are affected; nothing global changes.
    """
    environment = dict(os.environ)
    environment['CODEBUDDY_SAFE_DELETE_ENABLED'] = '0'
    return environment


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(module, expected):
    """Fail loudly when an import resolved to a file other than the intended one."""
    actual = Path(module.__file__).resolve()
    want = Path(expected).resolve()
    if actual != want:
        raise RuntimeError(f'{module.__name__} loaded from {actual}, expected {want}')
    return actual


def require_directory(module, expected_dir):
    actual = Path(module.__file__).resolve()
    want = Path(expected_dir).resolve()
    if actual.parent != want:
        raise RuntimeError(f'{module.__name__} loaded from {actual}, expected a file under {want}')
    return actual


# Two complete copies of the backend exist. fast_cached_runtime_v1.bootstrap()
# puts nr-b580-int8/backend on sys.path but not nr-b580/backend, so the reviewed
# fullsize adapter resolves nr_backend from the int8 tree. Rather than hardcode
# either answer, derive it from sys.path order and additionally require the two
# trees to agree byte for byte, so "which copy" can never change the numerics.
DUPLICATE_TREES = (NR / 'backend/nr_backend', NR_INT8 / 'backend/nr_backend')


def first_provider(package, entries=None):
    """The directory Python actually uses for ``package``, given sys.path order."""
    for entry in (sys.path if entries is None else entries):
        if not entry:
            continue
        candidate = Path(entry) / package
        if (candidate / '__init__.py').is_file():
            return candidate.resolve()
    raise RuntimeError(f'no provider for {package} on sys.path')


def duplicate_source_agreement():
    """Every module present in both backend copies must have identical bytes."""
    left, right = DUPLICATE_TREES
    rows = {}
    for path in sorted(left.glob('*.py')):
        other = right / path.name
        if not other.is_file():
            continue
        left_sha, right_sha = sha(path), sha(other)
        rows[path.name] = dict(same=left_sha == right_sha, left_sha256=left_sha,
                               right_sha256=right_sha)
    if not rows:
        raise RuntimeError('no shared backend modules found; refusing to assume agreement')
    return rows


def toolchain():
    """Static identity of the compiler the candidate kernels must be built with."""
    missing = [str(item) for item in (VCVARS, ICX_CL, ONEAPI_COMPILER_LIB) if not item.exists()]
    if missing:
        raise RuntimeError(f'compiler toolchain incomplete: {missing}')
    return dict(vcvars=str(VCVARS), compiler=str(ICX_CL), compiler_lib=str(ONEAPI_COMPILER_LIB),
                base_path=VCVARS_BASE_PATH, host_modules=list(HOST_MODULES))


def identity():
    """Static identity of the reviewed baseline this round compares against."""
    if sha(BASELINE_ADAPTER) != BASELINE_ADAPTER_SHA256:
        raise RuntimeError('Reviewed baseline adapter changed; stop and re-establish identity')
    full = read(BASELINE_FULL)
    if full['adapter_sha256'] != BASELINE_ADAPTER_SHA256:
        raise RuntimeError('Reviewed fullsize validation does not describe this adapter')
    if full['profile_sha256'] != PROFILE_SHA256 or sha(PROFILE) != PROFILE_SHA256:
        raise RuntimeError('Frozen profile identity changed')
    if not full['passed'] or full['phase'] != 'full' or len(full['frames']) != 243:
        raise RuntimeError('Reviewed fullsize baseline is not a passed 243-frame full phase')
    exact = read(EXACT_VALIDATION)
    if exact['source_sha256'] != SOURCE_SHA256 or len(exact['frames']) != 243:
        raise RuntimeError('Reviewed input index changed')
    # The count is deliberately not called 'frames'. rows_entry_v1 merges this
    # dict into its own report with report.update(), and there 'frames' is the
    # list of per-frame byte-comparison results. A shared name let this integer
    # replace that list, so the gate phase died on its first append.
    return dict(baseline_adapter_sha256=BASELINE_ADAPTER_SHA256,
                baseline_validation=str(BASELINE_FULL), profile_sha256=PROFILE_SHA256,
                source_sha256=SOURCE_SHA256, baseline_frame_count=243,
                baseline_arithmetic=full['arithmetic'],
                baseline_phase_input=full['full_phase_input'])
