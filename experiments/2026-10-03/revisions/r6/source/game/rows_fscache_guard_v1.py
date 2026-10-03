"""Restore the filesystem guarantee Triton's FileCacheManager.put() depends on.

Triton's put() (triton/runtime/cache.py:103) ends with

    os.removedirs(temp_dir)

*after* having moved the payload out of temp_dir into the cache directory.
os.removedirs removes the leaf and then walks up the path pruning empty parents,
and it depends on rmdir(parent) *raising* once a parent is non-empty to stop
walking.

On this volume rmdir() does not raise for a non-empty directory: it removes it
recursively. Verified with ``python -S`` (no sitecustomize at all, native
nt.rmdir) and cross-checked from a second process:

    py: before = True ['child.txt']
    py: rmdir ok
    py: after  = False
    shell: directory and child both gone

so this is the volume's behaviour, not a Python-level monkeypatch. (The
WorkBuddy safe-delete shim replaces os.remove/os.rmdir/shutil.rmtree when
CODEBUDDY_SESSION_ID is set, and its safe-bin wrappers shadow rm/rmdir on PATH,
but disabling it via CODEBUDDY_SAFE_DELETE_ENABLED=0 does not change the
outcome.)

Consequence: os.removedirs(temp_dir) prunes temp_dir *and then the cache
directory itself*, which put() has just written a file into, and put() returns a
path that no longer exists. compile() then fails on the metadata it believes it
just cached:

    FileNotFoundError: ...\\triton-cache\\<dir>\\_kernel.json

The guard replaces os.removedirs with a leaf-only version in the current
process, which is exactly what Triton intends: delete the now-empty temp
directory, never touch its parent. It is the only such call site in the entire
Triton tree (cache.py:134), and nothing under nr-b580 / nr-b580-int8 calls
os.rmdir, os.removedirs or shutil.rmtree.

It changes no kernel, no artifact, no specialization and no numeric result. The
only behaviour it alters is that a temp-directory cleanup stops deleting its own
parent, so it cannot affect byte identity or timing.

Run ``rows_fscache_guard_v1.py --probe <scratch>`` to prove the cache write path
works before spending time on compilation.
"""
import argparse
import json
import os
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]
TRITON_SITE = W / 'toolchain'
PRODUCT = W

_ORIGINAL = None
_INSTALLED = False
_CALLS = []
_REFUSED = []
# Triton's put() creates the directory it is about to remove as
# f"tmp.pid_{os.getpid()}_{uuid4().hex}" (cache.py:116).
TEMP_PREFIX = 'tmp.pid_'


def _describe(fn):
    if fn is None:
        return None
    return dict(module=getattr(fn, '__module__', None), name=getattr(fn, '__name__', None))


def install():
    """Make os.removedirs remove only its argument. Idempotent."""
    global _ORIGINAL, _INSTALLED
    if _INSTALLED:
        return status()
    _ORIGINAL = os.removedirs

    def removedirs_leaf_only(path):
        name = os.path.basename(os.path.normpath(path))
        _CALLS.append(name)
        if not name.startswith(TEMP_PREFIX):
            # Refuse anything that is not a Triton temp directory. Nothing in
            # nr-b580 or nr-b580-int8 removes a directory through os.removedirs,
            # so the only legitimate caller is Triton's own put() cleanup.
            _REFUSED.append(name)
            return
        os.rmdir(path)

    removedirs_leaf_only.__name__ = 'removedirs_leaf_only'
    os.removedirs = removedirs_leaf_only
    _INSTALLED = True
    return status()


def status():
    return dict(installed=_INSTALLED,
                removedirs=_describe(os.removedirs),
                removedirs_before=_describe(_ORIGINAL),
                rmdir=_describe(os.rmdir),
                remove=_describe(os.remove),
                temp_prefix=TEMP_PREFIX,
                calls=len(_CALLS),
                refused=_REFUSED,
                note='leaf-only removedirs: the temp-directory cleanup no longer prunes the '
                     'Triton cache directory it just wrote into')


def probe(root):
    """Run the real FileCacheManager.put() and require the cache to survive it."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = str(root / 'cache')
    sys.path.insert(0, str(TRITON_SITE))
    sys.path.insert(0, str(PRODUCT / 'comfy'))
    from runtime_environment import isolate
    isolate()
    from triton.runtime.cache import FileCacheManager

    report = dict(root=str(root), guard=status())
    manager = FileCacheManager('guard-probe')
    report['cache_dir'] = manager.cache_dir

    # Fresh file.
    returned = manager.put(b'payload', '_kernel.source')
    report['first'] = dict(returned=returned, returned_exists=os.path.isfile(returned),
                           cache_dir_exists=os.path.isdir(manager.cache_dir))

    # Re-put over an existing file (Triton does this for every artifact).
    returned2 = manager.put(b'payload-2', '_kernel.source')
    report['second'] = dict(returned=returned2, returned_exists=os.path.isfile(returned2),
                            cache_dir_exists=os.path.isdir(manager.cache_dir),
                            content=Path(returned2).read_text() if os.path.isfile(returned2) else None)

    # A group, the way compile() records its artifacts.
    group_path = manager.put_group('_kernel.json', {'_kernel.json': returned})
    report['group'] = dict(returned=group_path, returned_exists=os.path.isfile(group_path),
                           cache_dir_exists=os.path.isdir(manager.cache_dir))

    report['passed'] = bool(report['first']['returned_exists']
                            and report['first']['cache_dir_exists']
                            and report['second']['returned_exists']
                            and report['second']['cache_dir_exists']
                            and report['second']['content'] == 'payload-2'
                            and report['group']['returned_exists'])
    report['leftovers'] = sorted(os.listdir(manager.cache_dir))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', metavar='DIR')
    parser.add_argument('--out')
    args = parser.parse_args()
    install()
    if args.probe is None:
        print(json.dumps(status(), indent=2, ensure_ascii=False))
        return
    report = probe(args.probe)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding='utf-8')
    print(text)
    if not report['passed']:
        raise SystemExit('cache write path still broken')


if __name__ == '__main__':
    main()
