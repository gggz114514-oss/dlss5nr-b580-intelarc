"""Capture one real operand set per dense geometry from an authenticated 480p reset.

FP16 and W8A8 use independent model runs. Full output must match the prior same
precision output. This is operand collection, not a performance measurement.
"""
import argparse, hashlib, json, os, sys, traceback, urllib.request
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXACT = ROOT.parent / 'nr-b580'
R = EXACT / 'reference'
DREF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['fp16_xmx', 'int8_fused_v2'], required=True)
MODE = p.parse_args().mode
OUT = DREF / f'experimental/dense-operands-{MODE}-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR'] = str(DREF / 'triton-cache-sm89-v1')
sys.path[:0] = [str(R), str(ROOT / 'backend')]
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import js, sha
gate = authenticate_main()
prior_path = DREF / 'results/fast-precision-864x480-v1/validation.json'
assert sha(prior_path) == '7a43ff73d452cabe3e68ea1a15db1537417384a57c9a2443fcb16b095d5a5110'
prior = js(prior_path)
with urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=5) as response:
    queue = json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
import numpy as np
from PIL import Image
import torch
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from strided_batched_v2 import StridedMatrices
from fused_cached_matrices_v2 import geometry
from window_layout_v1 import WindowLayout
from swin_scheduling_v1 import SwinScheduling
import compressed_arrays_v1 as arrays
names = ['capture_dense_operands_v1.py', 'Run-DenseOperandsV1.cmd', 'strided_batched_v2.py',
         'strided_batched_v1.py', 'fused_cached_matrices_v2.py', 'fused_cached_matrices_v1.py',
         'fused_activation_int8_v1.py', 'fast_matrices_v3.py', 'static_weight_cache_v2.py',
         'static_weight_cache_v1.py', 'swin_scheduling_v1.py', 'window_layout_v1.py',
         'compressed_arrays_v1.py', 'immutable_artifacts_v1.py']
inputs = R / 'inputs/flow-full-864x480-v2'
manifest = js(inputs / 'manifest.json')
paths = [*(HERE / n for n in names), prior_path, inputs / 'manifest.json',
         *sorted((ROOT / 'backend/nr_backend').glob('*.py'))]
frozen = {str(path): sha(path) for path in paths}
OUT.mkdir()
report = dict(scope=__doc__, mode=MODE, sources=frozen, passed=False, exact_gate=gate,
              cases={}, operand_raw_bytes=0, maximum_operand_raw_bytes=512 * 2**20,
              complete_migration=False)

def save():
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')

class Collector(StridedMatrices):
    def dense(self, a, w, *, chunk_k, initial=None, **kwargs):
        if chunk_k == 16:
            k, n = w.shape
            m = a.numel() // k
            key = f'{m}x{k}x{n}:initial={initial is not None}'
            if key not in report['cases']:
                bn = geometry(m, k, n) if MODE == 'int8_fused_v2' else None
                row = dict(shape=[m, k, n], initialized=initial is not None, count=0,
                           fused_bn=bn, captured=False)
                report['cases'][key] = row
                size = sum(t.numel() * 2 for t in (a, w, initial) if t is not None)
                if report['operand_raw_bytes'] + size <= report['maximum_operand_raw_bytes']:
                    row['operands'] = {}
                    for name, t in [('a', a), ('w', w), ('initial', initial)]:
                        value = None if t is None else t.half().cpu().numpy()
                        assert value is None or np.isfinite(value).all()
                        row['operands'][name] = None if value is None else arrays.save(value)
                    row['captured'] = True
                    report['operand_raw_bytes'] += size
                    print(json.dumps(dict(captured=key, raw_bytes=size, fused_bn=bn)), flush=True)
            report['cases'][key]['count'] += 1
        return super().dense(a, w, chunk_k=chunk_k, initial=initial, **kwargs)

try:
    torch.set_num_threads(2)
    model = MotionNR.from_assets(EXACT / 'model-assets/sf-v2/WEIGHTS_HT.bin',
        EXACT / 'model-assets/noise-sm89-v2', EXACT / 'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    experiment = Collector()
    experiment.select(MODE)
    spec = manifest['frames'][0]
    rgb_path, motion_path = inputs / spec['file'], inputs / spec['motion_file']
    report['inputs'] = {str(path): sha(path) for path in (rgb_path, motion_path)}
    pixels = np.asarray(Image.open(rgb_path).convert('RGB'), dtype='f4') / 255
    flow = np.fromfile(motion_path, '<f4').reshape(480, 864, 2)
    rgb, motion = torch.from_numpy(pixels).to('xpu'), torch.from_numpy(flow).to('xpu')
    with torch.inference_mode(), experiment.installed(), SwinScheduling(enabled=True).installed(), WindowLayout().installed():
        if MODE.startswith('int8'):
            experiment.static_cache.prepack(model)
        with use_arithmetic_backend('triton') as dispatch:
            value = model(rgb, motion, reset=True)
        torch.xpu.synchronize()
    target = prior['runs']['fp16_xmx' if MODE == 'fp16_xmx' else 'int8_dense'][0]
    meta = target['actual']
    assert sha(Path(meta['path'])) == meta['sha256']
    expected = np.load(meta['path'], allow_pickle=False).astype('f2')
    actual = value.cpu().numpy()
    report.update(byte_equal_prior=actual.tobytes() == expected.tobytes(), output=meta,
                  private_equal=model._previous.cpu().numpy().tobytes() == actual.tobytes(),
                  dispatch=dict(dispatch), matrix_calls=experiment.calls, next_seed=model.next_seed)
    assert report['byte_equal_prior'] and report['private_equal'] and model.next_seed == 1
    assert dict(dispatch) == target['dispatches']
    assert rgb.cpu().numpy().tobytes() == pixels.tobytes() and motion.cpu().numpy().tobytes() == flow.tobytes()
    report['passed'] = True
except Exception as error:
    report.update(error=repr(error), traceback=traceback.format_exc())
    raise
finally:
    assert all(sha(Path(path)) == digest for path, digest in frozen.items())
    authenticate_main()
    save()
print(json.dumps(dict(passed=report['passed'], shapes=len(report['cases']), raw_bytes=report['operand_raw_bytes'])), flush=True)
