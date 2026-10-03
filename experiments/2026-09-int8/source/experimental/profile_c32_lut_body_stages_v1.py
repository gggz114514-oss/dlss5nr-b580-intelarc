"""Bounded graph-segment timings for the current fastest C32-LUT FP16 NR256 body (wide LUT is not selected).

Uses real temporal-frame inputs from the authenticated residual256 sequence.
Segments preserve all blocks, arithmetic, heads, skips and original stage order.
Static isolated replays are diagnostics, not frame latency or exact kernel-cycle
attribution: segment sums include extra launches/synchronizations and cache effects.
No unstable XPU profiler and no production patch. No new NR image semantics.
"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXACT = ROOT.parent / 'nr-b580'
R = EXACT / 'reference'
DREF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT = DREF / 'experimental/c32-lut-body-stages-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR'] = str(DREF / 'triton-cache-sm89-v1')
sys.path[:0] = [str(R), str(ROOT / 'backend')]
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import js, sha
gate = authenticate_main()
prior_path = DREF / 'results/residual-scale-fp16_xmx-256-v1/validation.json'
assert sha(prior_path) == '056251b4e2a839fcc44845d9646d2cd8ae1e9ed7f99fd1858dc7e3d51e3e50f9'
prior = js(prior_path)
pipeline_path = DREF / 'results/c32-lut-residual256-v2/validation.json'
assert sha(pipeline_path) == '00156f5e5e01c2918f6fbca4c03449ffa5045feae45f111eec8e20e87e997782'
pipeline = js(pipeline_path)
assert pipeline['passed'] and all(sha(Path(p)) == h for p, h in pipeline['sources'].items())
with urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=5) as response:
    queue = json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']

import numpy as np
from PIL import Image
import torch
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from strided_batched_v2 import StridedMatrices
from fused_branched_pairs_v2 import FusedPairs
from fused_c32_mlp_lut_v1 import FusedC32
from cubic_lut_constant_v1 import register, Constant, TABLE
from fused_vit_projection_v3 import FusedVitProjection
from fused_swin_scheduling_v1 import FusedSwin
from fused_dynamic_front_v1 import FusedFront
from graph_history_warp_v2 import GraphHistoryWarp
from residual_scale_v1 import ResidualScale
from graph_front_v6 import GraphFront
from c32_chunk_layout_v2 import ChunkedHeadLayout
from fused_split_ffwd_v2 import FusedSplit
import capture_body_v1 as body
import compressed_arrays_v1 as arrays

inputs = R / 'inputs/flow-full-1920x1080-v3'
manifest_path = inputs / 'manifest.json'
manifest = js(manifest_path)
paths = [Path(__file__), HERE / 'Run-C32LutBodyStagesV1.cmd', prior_path, pipeline_path, manifest_path,
         *[Path(p) for p in pipeline['sources']]]
frozen = {str(p): sha(p) for p in paths}
OUT.mkdir()
report = dict(scope=__doc__, sources=frozen, exact_gate=gate, passed=False, stages=[],
              full_body_samples_seconds=[], input_frame=1, source_frame=181, complete_migration=False,
              timing='7 rounds, rotated segment order, 10 static replays then completion per sample; '
                     'whole captured body measured separately with same repetition count. No dynamic front/warp/upload/flow/composition.',
              output_semantics='All full intermediate graph outputs compared against same-arithmetic eager segment outputs; '
                               'assembled whole output compared against authenticated residual256 temporal frame1.')
def save():
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')

def values(result):
    return result if isinstance(result, tuple) else (result,)

def cpu_bytes(t):
    return t.cpu().numpy().tobytes()

stages, adapter, warp = [], None, None
try:
    torch.set_num_threads(2)
    model = MotionNR.from_assets(EXACT / 'model-assets/sf-v2/WEIGHTS_HT.bin',
        EXACT / 'model-assets/noise-sm89-v2', EXACT / 'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    lut = register(model); constant = Constant(model)
    provider = StridedMatrices(); provider.select('fp16_xmx')
    adapter = GraphFront(model, arithmetic=provider)
    scheduler, layout = FusedSwin(provider), ChunkedHeadLayout(model, provider)
    pairs = FusedPairs(model, provider)
    split = FusedSplit(model, provider)
    vit_fusion = FusedVitProjection(model, provider)
    c32 = FusedC32(model, provider, bm=32, warps=4, stages=1)
    front, warp, scaler = FusedFront(model), GraphHistoryWarp(model), ResidualScale(256)
    with torch.inference_mode(), provider.installed(), scheduler.installed(), layout.installed(), \
            adapter.installed(), pairs.installed(), split.installed(), c32.installed(), vit_fusion.installed(), front.installed(), warp.installed(), \
            body.installed(), use_arithmetic_backend('triton'):
        for i in (0, 1):
            spec = manifest['frames'][i]
            rgb = torch.from_numpy(np.asarray(Image.open(inputs/spec['file']).convert('RGB'), dtype='f4')/255).to('xpu')
            motion = torch.from_numpy(np.fromfile(inputs/spec['motion_file'], '<f4').reshape(1080,1920,2)).to('xpu')
            low_rgb, low_motion = scaler.prepare(rgb, motion)
            low_nr = model(low_rgb, low_motion, reset=i == 0)
            assert cpu_bytes(low_nr) == arrays.load(prior['runs'][i]['low_nr']).tobytes()
            assert cpu_bytes(scaler.composite(rgb, low_rgb, low_nr)) == arrays.load(prior['runs'][i]['output']).tobytes()
        entry = adapter.last_entry
        private = cpu_bytes(model._previous)
        next_seed = model.next_seed
        static = {name: None if t is None else t.clone() for name, t in entry.inputs.items()}
        report['body_inputs'] = {name: None if t is None else arrays.save(t.cpu().numpy()) for name, t in static.items()}
        report['expected_output'] = prior['runs'][1]['low_nr']

        def stage(name, fn, *args):
            owned = tuple(None if t is None else t.clone() for t in args)
            result = values(fn(*owned))
            expected = tuple(cpu_bytes(t) for t in result)
            row = dict(name=name, input_shapes=[None if t is None else list(t.shape) for t in owned],
                       output_shapes=[list(t.shape) for t in result],
                       output_sha256=[hashlib.sha256(b).hexdigest() for b in expected], samples_seconds=[])
            stages.append(dict(fn=fn, inputs=owned, expected=expected, row=row))
            report['stages'].append(row)
            return result

        pre_skip, x = stage('pre', lambda a: model.pre.forward_features_outputs(a), static['front'])
        skips = []
        def encoder(group, features):
            for block in group:
                skip, down = block.forward_outputs(features)
                features = down if down is not None else skip
            return skip, features
        for channels, group in zip((32,64,128,256), model.encoder):
            skip, x = stage(f'encoder_C{channels}', lambda a, g=group: encoder(g,a), x)
            skips.append(skip)
        def encoder512(features):
            for i, block in enumerate(model.encoder512):
                result = block.forward_boundaries(features)
                features = result[-1]
                if i == 7: skip = result[3]
            return skip, features
        skip512, x = stage('encoder_C512', encoder512, x)
        def vit(features):
            shape = features.shape
            features = features.reshape(-1,1024)
            for block in model.vit: features = block(features)
            return features.reshape(shape)
        report['vit_tokens'] = x.numel() // 1024
        x, = stage('ViT_8_blocks', vit, x)
        def decoder512(features, skip):
            features = model.decoder_input(features, skip)
            for block in model.decoder512: features = block(features)
            return features
        x, = stage('decoder_C512_with_input', decoder512, x, skip512)
        def decoder(group, features, skip):
            features = group[0](features, skip)
            for block in group[1:]: features = block(features)
            return features
        for channels, group, skip in zip((256,128,64,32), model.decoder, reversed(skips)):
            x, = stage(f'decoder_C{channels}', lambda a,b,g=group: decoder(g,a,b), x, skip)
        def post(features, pre, color, previous, reciprocal):
            return model.post(features,pre,color,previous=previous,history_reciprocal=reciprocal,
                              sigmoid=model.sigmoid,blend_scale=model.blend_scale,return_float32=False)
        result, = stage('post',post,x,pre_skip,static['rgb'],static['previous'],static['history_reciprocal'])
        assert cpu_bytes(result) == arrays.load(report['expected_output']).tobytes()
        report['assembled_byte_equal_prior'] = True
        stream = torch.xpu.Stream()
        for item in stages:
            fn, args = item['fn'], item['inputs']
            with stream:
                for _ in range(2): warm = values(fn(*args))
            torch.xpu.synchronize()
            public = tuple(torch.empty_like(t) for t in warm)
            del warm
            graph = torch.xpu.XPUGraph()
            item['graph'], item['outputs'] = graph, public
            with torch.xpu.graph(graph,stream=stream):
                temporary = values(fn(*args))
                for dst, src in zip(public,temporary): dst.copy_(src)
            del temporary
            graph.replay()
            torch.xpu.synchronize()
            assert tuple(cpu_bytes(t) for t in public) == item['expected'], item['row']['name']
            item['row']['capture_byte_equal'] = True
            print('Captured '+item['row']['name'],flush=True)
        for repetition in range(7):
            order = stages[repetition:] + stages[:repetition]
            for item in order:
                torch.xpu.synchronize(); started = time.perf_counter()
                for _ in range(10): item['graph'].replay()
                torch.xpu.synchronize()
                item['row']['samples_seconds'].append((time.perf_counter()-started)/10)
                assert tuple(cpu_bytes(t) for t in item['outputs']) == item['expected']
            torch.xpu.synchronize(); started = time.perf_counter()
            for _ in range(10): entry.graph.replay()
            torch.xpu.synchronize()
            report['full_body_samples_seconds'].append((time.perf_counter()-started)/10)
            assert cpu_bytes(entry.output) == arrays.load(report['expected_output']).tobytes()
            print('Measured graph-stage round '+str(repetition),flush=True)
        for row in report['stages']:
            row['median_seconds'] = statistics.median(row['samples_seconds'])
        report['full_body_median_seconds'] = statistics.median(report['full_body_samples_seconds'])
        report['segment_median_sum_seconds'] = sum(row['median_seconds'] for row in report['stages'])
        assert cpu_bytes(model._previous) == private and model.next_seed == next_seed
        adapter._validate()
        report['history_unchanged'] = True
        assert constant.require() is lut and cpu_bytes(lut)==np.load(TABLE,allow_pickle=False).view('i2').tobytes()
        report['lut_unchanged'] = True
        report['layout_calls'] = dict(layout.calls)
        report['vit_projection_calls'] = vit_fusion.calls
        report['passed'] = True
except Exception as error:
    report.update(error=repr(error),traceback=traceback.format_exc())
    raise
finally:
    for item in stages:
        if 'graph' in item: item['graph'].reset()
    if warp is not None: warp.close()
    if adapter is not None: adapter.close()
    assert all(sha(Path(p)) == h for p,h in frozen.items())
    authenticate_main()
    save()
print(json.dumps(dict(passed=report['passed'],full_body=report['full_body_median_seconds'],
    stages={r['name']:r['median_seconds'] for r in report['stages']}),indent=2),flush=True)
