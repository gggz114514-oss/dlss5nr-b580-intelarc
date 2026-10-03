"""Load pinned prebuilt Triton host helpers independent of installation path."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def fast_cache_options(options):
    """Preserve the published cache identity after validating local libdevice.

    The old path is only a cache-key token. DiskOnly requires a hit before
    calling Triton, so it must never be opened for compilation.
    """
    identity = json.loads((ROOT / 'host-helpers/fast-cache-identity.json').read_text())
    local = ROOT / 'toolchain/triton/backends/intel/lib/libsycl-spir64-unknown-unknown.bc'
    if hashlib.sha256(local.read_bytes()).hexdigest() != identity['libdevice_sha256']:
        raise RuntimeError('Portable libdevice checksum mismatch')
    values = dict(options or {})
    libraries = dict(values.get('extern_libs') or {})
    current = libraries.get('libdevice')
    if current is not None and current not in (str(local), identity['libdevice_key']):
        raise RuntimeError('Unexpected fast libdevice override')
    libraries['libdevice'] = identity['libdevice_key']
    values['extern_libs'] = tuple(libraries.items())
    return values


def install_helpers():
    from triton import knobs
    # libtriton getenv decodes Windows ANSI bytes as UTF-8. Set the public
    # Python descriptor directly to keep non-ASCII cache paths as Unicode.
    knobs.cache.dir = os.environ['TRITON_CACHE_DIR']
    from triton.backends.intel import driver
    if getattr(driver, '_nr_portable_helpers', False):
        return
    manifest = json.loads((ROOT / 'host-helpers/manifest.json').read_text())
    for row in manifest.values():
        path = ROOT / 'host-helpers' / row['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError('NR host helper checksum mismatch: ' + path.name)

    def load(src, name):
        row = manifest.get(name)
        if row is None or hashlib.sha256(src.encode()).hexdigest() != row['source_sha256']:
            raise RuntimeError('Unprepared Triton host helper; compilation is disabled: ' + name)
        path = str(ROOT / 'host-helpers' / row['file'])
        factories = {'arch_utils': driver.ArchParser, 'spirv_utils': driver.SpirvUtils,
                     'extension_utils_impl': driver.ExtensionUtils}
        return factories[name](path)

    driver.compile_module_from_src = load
    from triton.backends.intel import extension_utils, compiler
    extension_utils.compile_module_from_src = load
    # Both modules import this function by value before bootstrap installs us.
    compiler.compile_module_from_src = load
    driver._nr_portable_helpers = True
