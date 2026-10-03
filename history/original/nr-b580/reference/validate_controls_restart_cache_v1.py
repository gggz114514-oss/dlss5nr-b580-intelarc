"""Fresh-process replay: fail on Triton cache miss or native helper build.

Process-local guards leave installed files and cached artifacts untouched.
Driver binary loading/JIT below Triton is not disabled by these guards.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

BASE = Path(__file__).resolve().parents[2]
DATA = Path('D:/Codex-NR-Experiments/nr-b580')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not sys.flags.utf8_mode:
        raise RuntimeError('Launch with python -X utf8')
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = dict(passed=False, pid=os.getpid(), groups=[], helper_reads=[],
                  blocked=[], scope='Fresh process, same research cache, 13 cases x 2 frames',
                  driver_internal_jit_excluded=False)

    def save():
        report['elapsed_seconds'] = time.perf_counter() - start
        (args.out / 'cache-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    def reject(kind, detail):
        report['blocked'].append(dict(kind=kind, detail=str(detail)))
        save()
        raise RuntimeError(f'Restart cache-only guard: {kind}: {detail}')

    save()
    try:
        os.environ['PYTHONUTF8'] = '1'
        os.environ['TRITON_CACHE_DIR'] = str(DATA / 'reference/triton-cache-c32-triton38-v1')
        sys.path.insert(0, str(BASE / 'nr-b580-int8/product/comfy'))
        from runtime_environment import isolate
        handles = isolate()
        sys.path.insert(0, str(DATA / 'reference/toolchains/triton-xpu-3.8.0-git1e2d42a0/site'))
        import triton
        from triton import knobs
        from triton.runtime.cache import FileCacheManager
        import triton.backends.intel.driver as driver
        if knobs.compilation.always_compile:
            reject('always_compile_enabled', True)
        report.update(triton=triton.__version__, effective_cache=str(knobs.cache.dir))
        original_group = FileCacheManager.get_group
        original_file = FileCacheManager.get_file

        def group(cache, name):
            value = original_group(cache, name)
            row = dict(name=name, directory=cache.cache_dir, hit=bool(value))
            if value:
                row['artifacts'] = value
            report['groups'].append(row)
            if not value:
                reject('kernel_group_miss', row)
            return value

        def file(cache, name):
            value = original_file(cache, name)
            if name.endswith(('.pyd', '.so')):
                report['helper_reads'].append(dict(name=name, path=value, hit=value is not None))
            return value

        def put(cache, data, filename, binary=True):
            reject('cache_write', filename)

        def build(*args, **kwargs):
            reject('native_helper_build', args[0] if args else kwargs)

        FileCacheManager.get_group = group
        FileCacheManager.get_file = file
        FileCacheManager.put = put
        driver._build = build
        save()
        import validate_controls_session_v1 as validator
        sys.argv = [str(Path(validator.__file__)), '--out', str(args.out / 'validation')]
        validator.main()
        result = json.loads((args.out / 'validation/validation.json').read_text(encoding='utf-8'))
        assert result['passed'] and len(result['frames']) == 26
        assert all(r['byte_equal'] and r['private_byte_equal'] and r['seed_equal'] for r in result['frames'])
        assert report['groups'] and report['helper_reads'] and not report['blocked']
        assert all(r['hit'] for r in report['groups'] + report['helper_reads'])
        report.update(passed=True, first_frame_seconds=result['frames'][0]['seconds'],
                      remaining_frame_seconds=sum(r['seconds'] for r in result['frames'][1:]),
                      validation_sha256=hashlib.sha256((args.out / 'validation/validation.json').read_bytes()).hexdigest())
        save()
        print(json.dumps({k: v for k, v in report.items() if k not in ('groups', 'helper_reads')}), flush=True)
    except BaseException:
        report['error'] = traceback.format_exc()
        save()
        raise


if __name__ == '__main__':
    main()
