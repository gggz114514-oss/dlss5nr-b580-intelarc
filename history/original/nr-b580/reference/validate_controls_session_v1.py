"""Bounded candidate check against authenticated, existing native captures.

Two frames per fixed control: reset plus temporal. This does not certify the
installer, all resolutions, face quality, live changes, or community composites.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

BASE = Path(__file__).resolve().parents[2]
EXACT = BASE / 'nr-b580'
REF = EXACT / 'reference'
DATA = Path('D:/Codex-NR-Experiments/nr-b580')
DLL = '6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927'


def js(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def authenticated(tag, version, evidence):
    metadata_path = REF / f'4060-{tag}-fine-sequence-v{version}.json'
    meta = js(metadata_path)
    assert meta['completedAllInputs'] and meta['exitCode'] == 0 and not meta['timedOut']
    assert meta['runtimeSha256'].lower() == DLL
    directory = DATA / 'reference/results' / Path(meta['runDirectory']).name / 'output'
    evidence[str(metadata_path)] = sha(metadata_path)
    entries = {item['name']: item for item in meta['outputs']}
    for i in (0, 1):
        for suffix in ('capture.json', 'input.rgba32f.bin', 'motion.rg32f.bin',
                       'depth.r32f.bin', 'output.rgba32f.bin'):
            name = f'frame{i:02d}.png_{suffix}'
            path = directory / name
            assert sha(path) == entries[name]['sha256'].lower(), path
            evidence[str(path)] = sha(path)
    return directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(BASE / 'nr-b580-int8/product/comfy')]
    from runtime_environment import isolate
    runtime = isolate()
    sys.path[:0] = [str(DATA / 'reference/toolchains/triton-xpu-3.8.0-git1e2d42a0/site'),
                   str(EXACT / 'backend'), str(BASE / 'nr-b580-int8/product')]
    # Reuse the research cache; no product cache manifest or installation changes.
    os.environ['TRITON_CACHE_DIR'] = str(DATA / 'reference/triton-cache-c32-triton38-v1')
    import numpy as np
    from PIL import Image
    from nr_exact_controls_candidate_v1 import Session, parse_controls
    report = dict(passed=False, scope='256x256 SDR zero depth; 13 fixed cases x 2 frames',
                  product_accepted=False, complete_migration=False, frames=[], evidence={},
                  runtime=str(runtime), cache=os.environ['TRITON_CACHE_DIR'])
    paths = list((EXACT / 'backend/nr_backend').glob('*.py')) + [Path(__file__),
        BASE / 'nr-b580-int8/product/nr_exact_controls_candidate_v1.py',
        BASE / 'nr-b580-int8/product/nr_exact_runtime_v1.py']
    report['sources'] = {str(p): sha(p) for p in paths}

    def save():
        (args.out / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    save()
    try:
        cases = [('default', {}, 'depth-zero', 1), ('auto', {'auto_mask': True}, 'auxflags-auto', 1)]
        for field, label in [('skin_structure', 'skin'), ('local_tone', 'tone'),
                             ('local_structure', 'structure')]:
            for value in (0, 2):
                name = f'auto-{label}{value}'
                cases.append((name, {'auto_mask': True, field: value}, f'auxflags-{name}', 2))
        for value, label in [(0, '0'), (.5, '05'), (2, '2')]:
            cases.append((f'intensity{label}', {'intensity': value}, f'controls-intensity{label}', 1))
        for value in (1, 2):
            cases.append((f'style{value}', {'style': value}, f'controls-style{value}', 1))
        default = authenticated('controls-default', 1, report['evidence'])
        inputs = REF / 'inputs/temporal-v1'
        manifest = js(inputs / 'manifest.json')
        fixtures = []
        for name, controls, tag, version in cases:
            parsed = parse_controls(controls)
            native = authenticated(tag, version, report['evidence'])
            for i, spec in enumerate(manifest['frames'][:2]):
                image = inputs / spec['file']
                assert sha(image) == spec['sha256']
                report['evidence'][str(image)] = sha(image)
                prefix = native / spec['file']
                cap = js(Path(str(prefix) + '_capture.json'))
                expected_controls = dataclasses.asdict(parsed)
                expected_controls['skin_structure'] = -1 if parsed.skin_structure is None else parsed.skin_structure
                assert all(cap[key] == value for key, value in expected_controls.items()), (name, cap)
                assert cap['reset'] == int(spec['reset'])
                pixels = np.asarray(Image.open(image).convert('RGB'), dtype='<f4') / 255
                def array(suffix, channels, dtype='<f4'):
                    return np.fromfile(str(prefix) + '_' + suffix, dtype).reshape(256, 256, channels).copy()
                assert pixels.astype('<f2').astype('<f4').tobytes() == array('input.rgba32f.bin', 4)[..., :3].copy().tobytes()
                assert not array('depth.r32f.bin', 1).any()
                target = array('output.rgba32f.bin', 4)[..., :3].copy()
                private = target.astype('<f2')
                if name.startswith('intensity'):
                    private = np.fromfile(default / f'frame{i:02d}.png_output.rgba32f.bin', '<f4').reshape(256, 256, 4)[..., :3].astype('<f2')
                elif name.startswith('style'):
                    stage = DATA / 'reference/downloads/style-peripheral-replay-v1-stage'
                    rel = f'inputs/{name}-frame{i:02d}/neural.rgba16f.bin'
                    entry = next(e for e in js(stage / 'manifest.json')['files'] if e['name'].replace('\\', '/') == rel)
                    assert sha(stage / rel) == entry['sha256'].lower()
                    report['evidence'][str(stage / rel)] = sha(stage / rel)
                    private = np.fromfile(stage / rel, '<f2').reshape(256, 256, 4)[..., :3].copy()
                fixtures.append((name, controls, i, spec['reset'], pixels,
                                 array('motion.rg32f.bin', 2), target, private))
        for bad in ({'unknown': 1}, {'auto_mask': 1}, {'skin_structure': -1},
                    {'style': True}, {'local_tone': float('nan')}, {'intensity': 3}):
            try:
                parse_controls(bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f'Accepted invalid controls: {bad}')
        report.update(preflight_passed=True, planned_frames=len(fixtures), invalid_controls_rejected=True)
        save()
        print('Preflight passed: 13 cases / 26 authenticated frames', flush=True)
        if args.preflight:
            return
        import torch
        report['environment'] = dict(torch=torch.__version__, device=torch.xpu.get_device_name())
        session = None
        last = None
        try:
            for name, controls, i, reset, pixels, motion, target, private in fixtures:
                if name != last:
                    if session is not None:
                        session.close()
                    session = Session.create(EXACT, controls=controls)
                    last = name
                print(f'Start {name} frame {i}', flush=True)
                started = time.perf_counter()
                value = session.process(torch.from_numpy(pixels).to('xpu'),
                                        torch.from_numpy(motion).to('xpu'), reset=reset)
                actual = value.color.cpu().numpy()
                history = session._model._previous.cpu().numpy()
                row = dict(case=name, frame=i, controls=dataclasses.asdict(session._model.controls),
                           byte_equal=actual.tobytes() == target.tobytes(),
                           private_byte_equal=history.tobytes() == private.tobytes(),
                           next_seed=value.sequence, seed_equal=value.sequence == i + 1,
                           seconds=time.perf_counter() - started,
                           actual_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                           target_sha256=hashlib.sha256(target.tobytes()).hexdigest())
                report['frames'].append(row)
                save()
                print(json.dumps(row), flush=True)
                assert row['byte_equal'] and row['private_byte_equal'] and row['seed_equal'], row
                # A caller may mutate the displayed tensor without changing temporal state.
                with torch.inference_mode():
                    value.color.zero_()
                assert session._model._previous.cpu().numpy().tobytes() == history.tobytes()
        finally:
            if session is not None:
                session.close()
        assert report['sources'] == {str(p): sha(p) for p in paths}
        report['passed'] = True
        save()
    except BaseException:
        report['error'] = traceback.format_exc()
        save()
        raise


if __name__ == '__main__':
    main()
