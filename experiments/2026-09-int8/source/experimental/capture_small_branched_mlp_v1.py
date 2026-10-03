"""Capture every real branch MLP in the currently fastest temporal NR256 body.

Uses authenticated saved body inputs, including recorded history. This is an
operand capture, not a new independent temporal sequence validation. The entire
body output must match the existing reference before these records are accepted.
"""
import hashlib
import json
import os
import sys
import traceback
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXACT = ROOT.parent / 'nr-b580'
R = EXACT / 'reference'
D = Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT = D / 'experimental/small-branched-mlp-operands-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR'] = str(D / 'triton-cache-sm89-v1')
sys.path[:0] = [str(R), str(ROOT / 'backend')]
from attention_weights_main_gate_v1 import authenticate_main
gate = authenticate_main()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
js = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
path = D / 'experimental/c32-lut-body-stages-v1/validation.json'
assert sha(path) == '7f299ccf6679c63524d7cafefae38415ab262d6f23ae7fb06958ba5be2f1cf9f'
prior = js(path)
assert prior['passed'] and js(path.parent.with_suffix('.log.lease.json'))['returncode'] == 0
assert all(sha(p) == h for p, h in prior['sources'].items())
sources = dict(prior['sources'])
for p in (path, Path(__file__), HERE / 'Run-CaptureSmallBranchedMLPV1.cmd'):
    sources[str(p)] = sha(p)
with urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=5) as response:
    queue = json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
import numpy as np
import torch
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from nr_backend.multihead_block import BranchedMLP
from strided_batched_v2 import StridedMatrices
from fused_branched_pairs_v2 import FusedPairs
from fused_split_ffwd_v2 import FusedSplit
from fused_c32_mlp_lut_v1 import FusedC32
from cubic_lut_constant_v1 import register, Constant, TABLE
from fused_vit_projection_v3 import FusedVitProjection
from fused_swin_scheduling_v1 import FusedSwin
from c32_chunk_layout_v2 import ChunkedHeadLayout
import capture_body_v1 as body
import compressed_arrays_v1 as arrays

OUT.mkdir()
report = dict(scope=__doc__, sources=sources, exact_gate=gate, passed=False,
              cases=[], source_frame=181, body_inputs=prior['body_inputs'],
              expected_output=prior['expected_output'], raw_bytes=0,
              maximum_raw_bytes=128*2**20, complete_migration=False)


def save():
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


class CapturePairs(FusedPairs):
    def apply(self, module, features):
        result = super().apply(module, features)
        if id(module) not in self.modules:
            return result
        name = module_names[id(module)]
        assert name not in seen
        seen.add(name)
        values = dict(features=features, expand=module.expand, reduce=module.reduce,
                      project=module.project, skip_scale=module.skip_scale, output=result)
        size = sum(t.numel()*t.element_size() for t in values.values())
        assert report['raw_bytes'] + size <= report['maximum_raw_bytes']
        metadata = {}
        for key, tensor in values.items():
            a = tensor.cpu().numpy()
            assert np.isfinite(a).all()
            metadata[key] = arrays.save(a)
        report['raw_bytes'] += size
        report['cases'].append(dict(name=name, shape=list(features.shape), channels=module.channels,
                                    rows=features.numel()//module.channels, arrays=metadata))
        print(json.dumps(dict(module=name, shape=list(features.shape))), flush=True)
        return result


try:
    torch.set_num_threads(2)
    model = MotionNR.from_assets(EXACT / 'model-assets/sf-v2/WEIGHTS_HT.bin',
        EXACT / 'model-assets/noise-sm89-v2', EXACT / 'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    lut = register(model)
    constant = Constant(model)
    module_names = {id(m): name for name, m in model.named_modules() if isinstance(m, BranchedMLP)}
    seen = set()
    provider = StridedMatrices()
    provider.select('fp16_xmx')
    pairs, split = CapturePairs(model, provider), FusedSplit(model, provider)
    c32, vit = FusedC32(model, provider), FusedVitProjection(model, provider)
    scheduler, layout = FusedSwin(provider), ChunkedHeadLayout(model, provider)
    inputs = {n: torch.from_numpy(arrays.load(m)).to('xpu') for n, m in prior['body_inputs'].items()}
    with torch.inference_mode(), provider.installed(), pairs.installed(), split.installed(), \
            c32.installed(), vit.installed(), scheduler.installed(), layout.installed(), \
            body.installed(), use_arithmetic_backend('triton') as dispatch:
        value = body.forward_front(model, **inputs, sigmoid=model.sigmoid,
                                    blend_scale=model.blend_scale, return_float32=False)
    assert value.cpu().numpy().tobytes() == arrays.load(prior['expected_output']).tobytes()
    assert seen == set(module_names.values()) and len(seen) == 36
    assert constant.require() is lut and lut.cpu().numpy().tobytes() == np.load(TABLE, allow_pickle=False).view('i2').tobytes()
    assert all(t.cpu().numpy().tobytes() == arrays.load(prior['body_inputs'][n]).tobytes() for n, t in inputs.items())
    report.update(passed=True, entire_body_byte_equal=True, all_modules_captured=True,
                  inputs_and_lut_unchanged=True, dispatch=dict(dispatch))
except BaseException as error:
    report.update(error=repr(error), traceback=traceback.format_exc())
    raise
finally:
    assert all(sha(p) == h for p, h in sources.items())
    authenticate_main()
    save()
print(json.dumps(dict(passed=report['passed'], cases=len(report['cases']), raw_bytes=report['raw_bytes'])), flush=True)
