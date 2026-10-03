"""Compare complete independent B580 small-size sequences to native 4060 RGB."""
import hashlib
import json
import os
import sys
import time
import traceback
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXACT = ROOT.parent / 'nr-b580'
R = EXACT / 'reference'
D = Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental/small-geometry-v1')
OUT = Path('D:/Codex-NR-Experiments/nr-b580/reference/results/small-geometry-exact-v1')
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR'] = 'D:/Codex-NR-Experiments/nr-b580/reference/triton-cache-sm89-v1'
sys.path[:0] = [str(R), str(ROOT / 'backend')]
from attention_weights_main_gate_v1 import authenticate_main
gate = authenticate_main()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
native_path = D / 'native-audit-v1.json'
assert sha(native_path) == '5edc154dd436108cee49e0bd38017bb05e3f83d8bb8418421de989f6839bcfb5'
native = json.loads(native_path.read_text())
assert native['passed'] and all(sha(p) == h for p, h in native['sources'].items())
assert all(sha(p) == h for p, h in native['files'].items())
with urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=5) as response:
    queue = json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
import numpy as np
import torch
from PIL import Image
from nr_backend.execution import use_arithmetic_backend
from small_geometry_motion_v1 import SmallMotionNR
import compressed_arrays_v1 as arrays

paths = [Path(__file__), HERE / 'small_geometry_motion_v1.py', HERE / 'Run-SmallGeometryExactV1.cmd',
         HERE / 'compressed_arrays_v1.py', native_path, *sorted((ROOT / 'backend/nr_backend').glob('*.py'))]
frozen = {str(p): sha(p) for p in paths}
for p in (ROOT / 'backend/nr_backend').glob('*.py'):
    assert p.read_bytes() == (EXACT / 'backend/nr_backend' / p.name).read_bytes()
OUT.mkdir()
report = dict(scope=__doc__, sources=frozen, exact_gate=gate, native_audit_sha256=sha(native_path),
              passed=False, frames=[], complete_migration=False,
              timing_scope='Diagnostic eager exact model time with synchronizations; not a paired speed test.',
              history_source='Only each B580 stream own complete prior output; reference is comparison-only')


def save():
    assert all(sha(p) == h for p, h in frozen.items())
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


try:
    torch.set_num_threads(2)
    model = SmallMotionNR.from_assets(EXACT / 'model-assets/sf-v2/WEIGHTS_HT.bin',
                                      EXACT / 'model-assets/noise-sm89-v2',
                                      EXACT / 'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    held = []
    for case in native['cases']:
        w, h = map(int, case['dimension'].split('x'))
        model.reset()
        first = None
        for i in range(4):
            name = f'frame{i:02d}.png'
            reset = i in (0, 3)
            pixels = np.asarray(Image.open(Path(case['input_dir']) / name).convert('RGB'), dtype='f4') / 255
            y, x = np.indices((h, w), dtype='i4')
            mv = np.zeros((h, w, 2), dtype='f4')
            if not reset:
                mv[..., 0] = ((x * 131 + y * 17 + i * 997) % 2048 - 1024).astype('f4') / 1024
                mv[..., 1] = ((x * 73 + y * 97 + i * 431) % 2048 - 1024).astype('f4') / 2048
            assert mv.tobytes() == (Path(case['native_dir']) / (name + '_motion.rg32f.bin')).read_bytes()
            rgb, motion = torch.from_numpy(pixels).to('xpu'), torch.from_numpy(mv).to('xpu')
            torch.xpu.synchronize()
            started = time.perf_counter()
            with use_arithmetic_backend('triton') as dispatch:
                value = model(rgb, motion, reset=reset,
                              progress=lambda s: print(f'{case["dimension"]} frame{i} {s}', flush=True) if s in ('pre', 'ViT', 'RGB') else None)
            torch.xpu.synchronize()
            seconds = time.perf_counter() - started
            actual = value.cpu().numpy()
            expected = np.fromfile(Path(case['native_dir']) / (name + '_output.rgba32f.bin'), '<f4').reshape(h, w, 4)[..., :3].copy()
            assert np.isfinite(actual).all()
            equal = actual.astype('f4').tobytes() == expected.tobytes()
            private = model._previous.cpu().numpy()
            row = dict(dimension=case['dimension'], frame=i, reset=reset, next_seed=model.next_seed,
                       byte_equal_native=equal, max_abs=float(np.abs(actual.astype('f4') - expected).max()),
                       half_components_different=int(np.count_nonzero(actual.view('u2') != expected.astype('f2').view('u2'))),
                       output=arrays.save(actual), private_equals_output=private.tobytes() == actual.tobytes(),
                       seconds=seconds, dispatch=dict(dispatch))
            report['frames'].append(row)
            save()
            print(json.dumps({k: v for k, v in row.items() if k not in ('output', 'dispatch')}), flush=True)
            assert equal and row['private_equals_output']
            assert model.next_seed == (1 if reset else i + 1)
            assert rgb.cpu().numpy().tobytes() == pixels.tobytes() and motion.cpu().numpy().tobytes() == mv.tobytes()
            if i == 0:
                first = actual.tobytes()
                held.append((value.clone(), first))
            if i == 3:
                assert actual.tobytes() == first
            with torch.inference_mode():
                value.zero_()
            assert model._previous.cpu().numpy().tobytes() == private.tobytes()
        seed, history = model.next_seed, model._previous
        try:
            model(rgb[:h-1], motion[:h-1])
        except ValueError:
            pass
        else:
            raise AssertionError('Unsupported geometry accepted')
        assert model.next_seed == seed and model._previous is history
    assert all(t.cpu().numpy().tobytes() == b for t, b in held)
    model.reset()
    assert model.next_seed == 0 and model._previous is None
    report.update(passed=True, frames_verified=12, caller_mutation_preserves_history=True,
                  held_outputs_unchanged=True, unsupported_geometry_preserves_state=True,
                  reset_clears_history=True, all_bytes_equal_native=True)
    save()
except BaseException as exc:
    report['error'] = repr(exc)
    save()
    traceback.print_exc()
    raise
