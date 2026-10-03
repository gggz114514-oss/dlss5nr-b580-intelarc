"""Authenticate one native reset smoke after the user restarted the 4060.

This is readiness evidence, not a B580 comparison, geometry probe or speed test.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental/reconnect-smoke-v1')
TARGET = OUT / 'saved-audit-v2.json'
assert not TARGET.exists()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def js(path):
    b = Path(path).read_bytes()
    return json.loads(b.decode('utf-16' if b.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'))


manifest = js(OUT / 'manifest.json')
assert all(sha(p) == h for p, h in manifest['sources'].items())
assert sha(manifest['input']['path']) == manifest['input']['sha256']
assert js(OUT / 'ssh-exit.json')['exitCode'] == 0
report = js(OUT / 'native/run.json')
remote_stdout = json.loads((OUT / 'ssh-stdout.json').read_bytes().decode('utf-8-sig', errors='replace'))
# Console transport can replace non-ASCII driver diagnostics. Validate the
# actual report downloaded intact and compare the decisive structured fields.
for key in ('runDirectory', 'exitCode', 'completedAllInputs', 'outputs', 'arguments'):
    assert report[key] == remote_stdout[key]
assert report['exitCode'] == 0 and not report['timedOut'] and report['completedAllInputs']
assert report['expectedImages'] == 1 and report['reset'] == 1
assert not report['tensorCapture'] and not report['nativeKernelTrace']
assert report['executableSha256'].lower() == '730a5f893ae0b461eda7bdd68be71d3962b079d7951b4a54839f29d8445761e2'
assert report['runtimeSha256'].lower() == '6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927'
expected = {'gradient.png_nr.png', 'gradient.png_nr_diff.png', 'gradient.png_orig.png'}
assert {v['name'] for v in report['outputs']} == expected
pixels = {}
for meta in report['outputs']:
    path = OUT / 'native/output' / meta['name']
    assert path.stat().st_size == meta['bytes'] and sha(path) == meta['sha256'].lower()
    with Image.open(path) as im:
        assert im.size == (256, 256)
        pixels[meta['name']] = np.asarray(im.convert('RGB')).copy()
assert np.any(pixels['gradient.png_nr.png'] != pixels['gradient.png_orig.png'])
after = js(OUT / 'after.json')
assert not after['nrProcesses'] and not after['reconnectTasks']
saved = dict(passed=True, scope=__doc__, complete_migration=False,
             files={str(p): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
             auditor_sha256=sha(Path(__file__)), image_count=1,
             output_png_bytes=sum(v['bytes'] for v in report['outputs']),
             temporary_task_removed=True, native_process_exited=True,
             native_report_sha256=sha(OUT / 'native/run.json'))
TARGET.write_text(json.dumps(saved, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in saved.items() if k != 'files'}, indent=2))
