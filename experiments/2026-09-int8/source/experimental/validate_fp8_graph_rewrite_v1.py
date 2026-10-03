"""Validate proof-driven elimination of redundant FP8 graph operations, without timing.

Count Triton launch requests and ATen tensor operations in each eager stage,
then compare every complete result against uninstrumented execution and the
approved final frame. ATen calls include views and allocation: they are NOT a
native GPU kernel count. Logical tensor bytes are NOT measured memory traffic.
This audit motivates execution architecture changes; it changes no model math.
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
OUT = DREF / 'experimental/fp8-graph-rewrite-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR'] = str(DREF / 'triton-cache-c32-triton38-v1')
TOOLCHAIN = DREF / 'toolchains/triton-xpu-3.8.0-git1e2d42a0'
sys.path[:0] = [str(TOOLCHAIN / 'site'), str(R), str(ROOT / 'backend')]
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import js, sha
gate = authenticate_main()
provision_path = TOOLCHAIN / 'provision-v1.json'
assert sha(provision_path) == 'e94757afcb5962cacbe208cbb808b3516de210059335f1ee7ab7b40d6bfe2c07'
assert all(sha(TOOLCHAIN / 'site' / p) == h for p,h in js(provision_path)['files'].items())
prior_path = DREF / 'results/residual-scale-fp16_xmx-256-v1/validation.json'
assert sha(prior_path) == '056251b4e2a839fcc44845d9646d2cd8ae1e9ed7f99fd1858dc7e3d51e3e50f9'
prior = js(prior_path)
pipeline_path = DREF / 'results/vit-attention64-residual256-v2/validation.json'
assert sha(pipeline_path) == '8ff85699fdf7c35e7273787e2f56fd3917c83cbbc1694011354b44f7a33b69b3'
pipeline = js(pipeline_path)
assert pipeline['passed'] and all(sha(Path(p)) == h for p, h in pipeline['sources'].items())
with urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=5) as response:
    queue = json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']

import numpy as np
from PIL import Image
import torch
import triton
assert Path(triton.__file__).is_relative_to(TOOLCHAIN / 'site') and triton.__version__.startswith('3.8.0')
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from current_dense_tiled_provider_v1 import CurrentDenseTiledMatrices as K8TiledMatrices
from native_cubic_adapters_v1 import FusedBatched,FusedC32,FusedSplit
from fused_vit_attention64_adapter_v2 import FusedVitQKV
from cubic_lut_constant_v1 import register, Constant, TABLE
from fused_vit_projection_v3 import FusedVitProjection
from fused_swin_native_half_v1 import FusedSwin
from fused_dynamic_front_v1 import FusedFront
from fused_graph_history_warp_v2 import FusedHistoryWarp
from residual_scale_v1 import ResidualScale
from graph_front_v6 import GraphFront
from native_half_head_layout_v1 import HeadLayout
import capture_body_v1 as body
from quantization_dataflow_v1 import FP8
from fp8_graph_rewrite_v1 import ProofDataflow,RewritingDataflow,FP8GraphRewrite
from collections import Counter
import compressed_arrays_v1 as arrays

inputs = R / 'inputs/flow-full-1920x1080-v3'
manifest_path = inputs / 'manifest.json'
manifest = js(manifest_path)
paths = [HERE/'fp8_graph_rewrite_v1.py',HERE/'quantization_dataflow_v1.py',Path(__file__), HERE / 'Run-FP8GraphRewriteV1.cmd', provision_path, prior_path, pipeline_path, manifest_path,
         *[Path(p) for p in pipeline['sources']]]
frozen = {str(p): sha(p) for p in paths}
OUT.mkdir()
report = dict(scope=__doc__, sources=frozen, exact_gate=gate, passed=False, stages=[],
              input_frame=1, source_frame=181, complete_migration=False,
              timing='No timing measurements under instrumentation.',
              output_semantics='Every complete traced stage output compared against uninstrumented execution; '
                               'assembled whole output compared against approved temporal frame1.')

def save():
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')

def values(result):
    return result if isinstance(result, tuple) else (result,)

def cpu_bytes(t):
    return t.cpu().numpy().tobytes()

stages, adapter, warp = [], None, None
report['runtime'] = dict(triton_version=triton.__version__,triton_file=triton.__file__,torch_version=torch.__version__,isolated=True)
report['tail_fusion_installed'] = False
try:
    torch.set_num_threads(2)
    import gc
    import nr_backend.triton_fp8 as fp8_function
    bits=np.arange(65536,dtype=np.uint16).view(np.float16)
    all_half=torch.from_numpy(bits.copy()).to('xpu')
    once=fp8_function.quantize_fp8(all_half);twice=fp8_function.quantize_fp8(once)
    assert cpu_bytes(once)==cpu_bytes(twice)
    report['all_half_fp8_idempotent']=dict(passed=True,inputs=arrays.save(bits),once=arrays.save(once.cpu().numpy()),twice=arrays.save(twice.cpu().numpy()))
    source=torch.arange(32,device='xpu',dtype=torch.float16).reshape(4,8)*.13
    tracker=RewritingDataflow()
    with tracker.installed():
        changed=fp8_function.quantize_fp8(source)
        changed.add_(.125)
        assert tracker.ref(changed)['domain'] is None
        rounded=fp8_function.quantize_fp8(changed)
        assert not tracker.elisions
        borrowed=fp8_function.quantize_fp8(rounded)
        assert len(tracker.elisions)==1
        rejected=False
        try:borrowed.zero_()
        except RuntimeError as error:
            assert 'borrowed FP8' in str(error);rejected=True
        assert rejected
    assert cpu_bytes(rounded)==cpu_bytes(fp8_function.quantize_fp8(changed))==cpu_bytes(borrowed)
    with RewritingDataflow().installed() as strided:
        packed=fp8_function.quantize_fp8(source)
        sliced=packed[:,::2]
        converted=fp8_function.quantize_fp8(sliced)
        assert converted.is_contiguous() and len(strided.elisions)==1
    assert cpu_bytes(converted)==cpu_bytes(fp8_function.quantize_fp8(sliced))
    with RewritingDataflow().installed() as lifetime:
        temporary=fp8_function.quantize_fp8(source)
        weak=lifetime.state(temporary)['weak'];generation=lifetime.ref(temporary)['storage']
        del temporary;gc.collect();assert weak.expired()
        fresh=torch.empty_like(source);ref=lifetime.ref(fresh)
        assert ref['storage']!=generation and ref['domain'] is None
    report['rewrite_guards']=dict(mutation_invalidates_proof=True,borrowed_alias_write_rejected_before_mutation=True,strided_result_contiguous_and_equal=True,expired_storage_not_reused_as_proof=True)

    model = MotionNR.from_assets(EXACT / 'model-assets/sf-v2/WEIGHTS_HT.bin',
        EXACT / 'model-assets/noise-sm89-v2', EXACT / 'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    lut = register(model); constant = Constant(model)
    provider = K8TiledMatrices(model); provider.select('fp16_xmx')
    adapter = GraphFront(model, arithmetic=provider)
    scheduler, layout = FusedSwin(provider), HeadLayout(model, provider)
    pairs = FusedBatched(model, provider, workload='small')
    split = FusedSplit(model, provider)
    vit_fusion = FusedVitProjection(model, provider)
    qkv_fusion = FusedVitQKV(model, provider)
    c32 = FusedC32(model, provider, bm=32, warps=4, stages=1)
    front, warp, scaler = FusedFront(model), FusedHistoryWarp(model), ResidualScale(256)
    rewrite=FP8GraphRewrite(model,provider)
    with torch.inference_mode(), provider.installed(), scheduler.installed(), layout.installed(), \
            adapter.installed(), pairs.installed(), split.installed(), c32.installed(), vit_fusion.installed(), front.installed(), warp.installed(), \
            qkv_fusion.installed(), rewrite.installed(), body.installed(), use_arithmetic_backend('triton'):
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
        report['graph_rewrite_builds']=rewrite.builds

        def stage(name, fn, *args):
            owned = tuple(None if t is None else t.clone() for t in args)
            result = values(fn(*owned))
            expected = tuple(cpu_bytes(t) for t in result)
            row = dict(name=name, input_shapes=[None if t is None else list(t.shape) for t in owned],
                       output_shapes=[list(t.shape) for t in result],
                       output_sha256=[hashlib.sha256(b).hexdigest() for b in expected])
            row['inputs'] = [None if t is None else arrays.save(t.cpu().numpy()) for t in owned]
            row['outputs'] = [arrays.save(t.cpu().numpy()) for t in result]
            inventory=ProofDataflow()
            with inventory.installed():observed=values(fn(*owned))
            assert tuple(cpu_bytes(t) for t in observed)==expected,name
            row['instrumented_byte_equal']=True
            row['all_redundancy_inputs_equal_quantized_outputs']=inventory.verify_proofs()
            rewriting=RewritingDataflow()
            with rewriting.installed():rewritten=values(fn(*owned))
            assert tuple(cpu_bytes(t) for t in rewritten)==expected,name
            row['rewritten_byte_equal']=True
            row['rewrite']=rewriting.rewrite_summary()
            row['execution']=dict(summary=inventory.summary(),events=inventory.events)
            stages.append(dict(fn=fn, inputs=owned, expected=expected, row=row))
            print(name+' '+str(row['execution']['summary']),flush=True)
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
        report['execution_totals']={key:sum(row['execution']['summary'][key] for row in report['stages']) for key in ('triton_calls','aten_calls','standalone_fp8','proven_redundant_fp8')}
        report['all_instrumented_stage_outputs_match']=all(row['instrumented_byte_equal'] for row in report['stages'])
        report['stage_rewrite_totals']={key:sum(row['rewrite'][key] for row in report['stages']) for key in ('triton_calls','standalone_fp8','elided_fp8')}
        assert cpu_bytes(model._previous) == private and model.next_seed == next_seed
        adapter._validate()
        report['history_unchanged'] = True
        assert constant.require() is lut and cpu_bytes(lut)==np.load(TABLE,allow_pickle=False).view('i2').tobytes()
        report['lut_unchanged'] = True
        report['layout_calls'] = dict(layout.calls)
        report['vit_projection_calls'] = vit_fusion.calls
        report['vit_qkv_calls']=qkv_fusion.calls
        report['native_cubic_calls']={f:dict(c.native_calls) for f,c in [('c32',c32),('batched',pairs),('split',split)]}
        assert qkv_fusion.calls>0 and all(report['native_cubic_calls'].values())
        report['k8_calls'] = dict(provider.k8_calls)
        report['fused_history_builds'] = warp.fused_builds
        assert all(provider.k8_calls.get(name,0)>0 for name in ('pre','post')) and warp.fused_builds==1
        assert all(None if t is None else cpu_bytes(t)==arrays.load(m).tobytes()
            for item in stages for t,m in zip(item['inputs'],item['row']['inputs']) if t is not None)
        report['stage_inputs_unchanged'] = True
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
print(json.dumps(dict(passed=report['passed'],execution_totals=report['execution_totals']),indent=2),flush=True)
