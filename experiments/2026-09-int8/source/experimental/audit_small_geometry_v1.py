"""Authenticate all native frames, unchanged traces and actual work geometry."""
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
R = HERE.parent.parent / 'nr-b580/reference'
D = Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental/small-geometry-v1')
TARGET = D / 'native-audit-v1.json'
assert not TARGET.exists()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
js = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
sources = {str(Path(__file__)): sha(Path(__file__))}
manifest = js(D / 'manifest.json')
assert all(sha(p) == h for p, h in manifest['sources'].items())
sources.update(manifest['sources'])
sources[str(D / 'manifest.json')] = sha(D / 'manifest.json')
verified = {}


def authenticate(run):
    report = js(run / 'run.json')
    assert report['completedAllInputs'] and report['exitCode'] == 0 and not report['timedOut']
    assert report['runtimeSha256'].lower() == '6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927'
    verified[str(run / 'run.json')] = sha(run / 'run.json')
    for group, folder in [('outputs', run / 'output'), ('traceFiles', run / 'output/nvapi-trace')]:
        for value in report[group]:
            p = folder / value['name']
            assert p.stat().st_size == value['bytes'] and sha(p) == value['sha256'].lower()
            verified[str(p)] = value['sha256'].lower()
    return report


def trace(folder):
    events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
    launches = [e for e in events if e['event'] == 'launch']
    statuses = {e['sequence']: e['status'] for e in events if e['event'] == 'launch_result'}
    assert all(e['written'] and statuses[e['sequence']] == 0 for e in launches)
    assert sum(e['event'] == 'shutdown' for e in events) == 1
    pre = [e for e in launches if 'fused_pre_block' in e['name']]
    post = [e for e in launches if 'fused_post_block' in e['name']]
    return events, launches, pre, post


baseline = R / 'results/nvapi-trace-256-20260907-223414'
authenticate(baseline)
bt = baseline / 'output/nvapi-trace'
be, bl, bp, bpost = trace(bt)


def signature(folder, launches, first, last):
    return [(e['name'], e['grid'], e['block'], e['shared_bytes'], (folder / e['file']).stat().st_size)
            for e in launches if first <= e['sequence'] <= last]


baseline_signature = signature(bt, bl, bp[0]['sequence'], bpost[0]['sequence'])
baseline_modules = {}
for e in be:
    if e['event'] == 'module' and e['written']:
        raw = (bt / e['file']).read_bytes()
        magic, version, header, size = struct.unpack_from('<IHHQ', raw)
        assert magic == 0xba55ed50 and header == 16
        baseline_modules[hashlib.sha256(raw[:header + size]).hexdigest()] = e['file']
cases = []
for dim in ['128x128', '192x192', '256x144']:
    w, h = map(int, dim.split('x'))
    native = D / (dim + '-sequence/native')
    traced = D / (dim + '-trace/native')
    reports = [authenticate(run) for run in (native, traced)]
    fixture_dir = D / 'stage/inputs' / dim
    fixture = js(fixture_dir / 'manifest.json')
    assert sha(fixture['source']) == fixture['source_sha256']
    for kind, report in zip(('sequence', 'trace'), reports):
        assert report['dimension'] == dim and report['motionProbeMode'] == 'fine'
        assert report['resetPattern'] == [1, 0, 0, 1] and report['auxiliaryTextureCase'] == 'none'
        assert report['runnerSha256'].lower() == sha(HERE / 'Run-NrSmallGeometryV1.ps1')
        assert report['fixtureManifestSha256'].lower() == sha(fixture_dir / 'manifest.json')
        assert report['executableSha256'].lower() == 'ad69b6c840ccaa3adb977d10ccdda6df12bb68cce11c03904a313bc6edd89b53'
        transport = D / (dim + '-' + kind) / 'transport.json'
        assert js(transport)['returncode'] == 0
        sources[str(transport)] = sha(transport)
    assert reports[1]['traceDllSha256'].lower() == '1d1a002c01195d20dd40e950c4c5333827a619ceff1919751eb52223d6df7620'
    t = traced / 'output/nvapi-trace'
    events, launches, pre, post = trace(t)
    assert len(pre) == len(post) == 4
    modules = []
    for e in events:
        if e['event'] == 'module' and e['written']:
            raw = (t / e['file']).read_bytes()
            magic, version, header, size = struct.unpack_from('<IHHQ', raw)
            assert magic == 0xba55ed50 and header == 16
            digest = hashlib.sha256(raw[:header + size]).hexdigest()
            assert digest in baseline_modules, e['file']
            modules.append(digest)
    rows = []
    for i, (a, b, spec) in enumerate(zip(pre, post, fixture['frames'])):
        reset = i in (0, 3)
        name = f'frame{i:02d}.png'
        assert spec['file'] == name and spec['reset'] == reset and sha(fixture_dir / name) == spec['sha256']
        pixels = np.asarray(Image.open(fixture_dir / name).convert('RGB'), dtype='f4') / 255
        assert pixels.shape == (h, w, 3)
        capture = js(native / 'output' / (name + '_capture.json'))
        expected = dict(width=w, height=h, reset=int(reset), style=0, intensity=1, local_tone=1,
                        local_structure=1, preset=0, skin_structure=-1, global_tone=-1, auto_mask=0,
                        ui_correction=0, control_mask_bound=False, ui_bound=False,
                        ui_alpha_bound=False, backbuffer_bound=False, depth_bound=True, depth_inverted=False)
        assert all(capture[k] == v for k, v in expected.items())
        for suffix in ('input.rgba32f.bin', 'depth.r32f.bin', 'motion.rg32f.bin', 'output.rgba32f.bin',
                       'control-mask.rgba32f.bin', 'ui.rgba32f.bin', 'ui-alpha.r32f.bin', 'backbuffer.rgba32f.bin'):
            f = name + '_' + suffix
            assert (native / 'output' / f).read_bytes() == (traced / 'output' / f).read_bytes()
        inp = np.fromfile(native / 'output' / (name + '_input.rgba32f.bin'), '<f4').reshape(h, w, 4)
        assert pixels.astype('f2').astype('f4').tobytes() == inp[..., :3].copy().tobytes()
        y, x = np.indices((h, w), dtype='i4')
        mv = np.zeros((h, w, 2), dtype='f4')
        if not reset:
            mv[..., 0] = ((x * 131 + y * 17 + i * 997) % 2048 - 1024).astype('f4') / 1024
            mv[..., 1] = ((x * 73 + y * 97 + i * 431) % 2048 - 1024).astype('f4') / 2048
        assert mv.tobytes() == (native / 'output' / (name + '_motion.rg32f.bin')).read_bytes()
        assert not np.fromfile(native / 'output' / (name + '_depth.r32f.bin'), '<f4').any()
        out = np.fromfile(native / 'output' / (name + '_output.rgba32f.bin'), '<f4').reshape(h, w, 4)
        assert np.isfinite(out).all() and out.astype('f2').astype('f4').tobytes() == out.tobytes()
        pa, pb = (t / a['file']).read_bytes(), (t / b['file']).read_bytes()
        assert struct.unpack_from('<II', pa, 208) == (h, w)
        assert struct.unpack_from('<II', pb, 172) == (w, h)
        padded = struct.unpack_from('<II', pa, 240)
        assert padded == (320, 320)
        seed = struct.unpack_from('<I', pa, 200)[0]
        assert seed == (0 if reset else i)
        assert bool(struct.unpack_from('<Q', pa, 8)[0]) == bool(struct.unpack_from('<Q', pb, 88)[0]) == (not reset)
        sig = signature(t, launches, a['sequence'], b['sequence'])
        assert sig == baseline_signature
        vit = [e for e in launches if a['sequence'] < e['sequence'] < b['sequence'] and e['name'].startswith('cc_vit_1d_')]
        assert len(vit) == 42 and all(struct.unpack_from('<II', (t / e['file']).read_bytes(), (t / e['file']).stat().st_size - 8) == (8, 8) for e in vit)
        rows.append(dict(frame=i, padded_hw=padded, vit_tokens=64, seed=seed, reset=reset,
                         native_output_sha256=hashlib.sha256(out.tobytes()).hexdigest(),
                         body_dispatches=len(sig), body_geometry_equals_256=True,
                         pre_parameters=str(t / a['file']), post_parameters=str(t / b['file'])))
    assert (native / 'output/frame00.png_output.rgba32f.bin').read_bytes() == (native / 'output/frame03.png_output.rgba32f.bin').read_bytes()
    cases.append(dict(dimension=dim, input_dir=str(fixture_dir), native_dir=str(native / 'output'),
                      trace_dir=str(t), frames=rows, modules_equal_original_256=modules,
                      traced_and_untraced_all_arrays_equal=True, last_reset_equals_first=True))
    print(dim, 'all 4 frames authenticated, padded320x320, 64 tokens, original body geometry', flush=True)
report = dict(scope=__doc__, passed=True, sources=sources, files=verified, cases=cases,
              native_frame_evaluations=24, distinct_sequence_frames=12, complete_migration=False,
              limitation='Only three measured sizes and default SDR controls. No B580 validation or speed claim yet.')
TARGET.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print('Native audit passed:', sha(TARGET), flush=True)
