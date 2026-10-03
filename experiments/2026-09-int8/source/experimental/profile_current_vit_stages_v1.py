"""Bounded graph-segment timings for the first actual ViT block within current K8/native-half FP16 NR256.

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
OUT = DREF / 'experimental/current-vit-stages-v1'
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
pipeline_path = DREF / 'results/fused-history-residual256-v1/validation.json'
assert sha(pipeline_path) == '799eeeaa68a82ab4ee62d0493f1a286dfb69f6f3d41287b3a7a1ded9c7162e26'
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
from k8_tiled_provider_v1 import K8TiledMatrices
from batched_branched_mlp_v2 import FusedBatched
from fused_c32_mlp_lut_v1 import FusedC32
from cubic_lut_constant_v1 import register, Constant, TABLE
from fused_vit_projection_v3 import FusedVitProjection
from fused_swin_native_half_v1 import FusedSwin
from fused_dynamic_front_v1 import FusedFront
from fused_graph_history_warp_v2 import FusedHistoryWarp
from residual_scale_v1 import ResidualScale
from graph_front_v6 import GraphFront
from native_half_head_layout_v1 import HeadLayout
from fused_split_ffwd_v2 import FusedSplit
import capture_body_v1 as body
import compressed_arrays_v1 as arrays

stage_path=DREF/'experimental/current-body-stages-v1/validation.json'
assert sha(stage_path)=='b4561ad572f0818140e5e41bd7b7795dd0b45b896bfd8e3f424f19b012c65b00'
stage_reference=js(stage_path)
assert stage_reference['passed'] and js(stage_path.parent.with_suffix('.log.lease.json'))['returncode']==0
import nr_backend.vit_block as vb

inputs = R / 'inputs/flow-full-1920x1080-v3'
manifest_path = inputs / 'manifest.json'
manifest = js(manifest_path)
paths = [Path(__file__), HERE / 'Run-CurrentVitStagesV1.cmd', provision_path, prior_path, pipeline_path, manifest_path, stage_path,
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
report['runtime'] = dict(triton_version=triton.__version__,triton_file=triton.__file__,torch_version=torch.__version__,isolated=True)
report['tail_fusion_installed'] = False
try:
    torch.set_num_threads(2)
    model = MotionNR.from_assets(EXACT / 'model-assets/sf-v2/WEIGHTS_HT.bin',
        EXACT / 'model-assets/noise-sm89-v2', EXACT / 'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    lut = register(model); constant = Constant(model)
    provider = K8TiledMatrices(model); provider.select('fp16_xmx')
    adapter = GraphFront(model, arithmetic=provider)
    scheduler, layout = FusedSwin(provider), HeadLayout(model, provider)
    pairs = FusedBatched(model, provider, workload='small')
    split = FusedSplit(model, provider)
    vit_fusion = FusedVitProjection(model, provider)
    c32 = FusedC32(model, provider, bm=32, warps=4, stages=1)
    front, warp, scaler = FusedFront(model), FusedHistoryWarp(model), ResidualScale(256)
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
            row['inputs'] = [None if t is None else arrays.save(t.cpu().numpy()) for t in owned]
            row['outputs'] = [arrays.save(t.cpu().numpy()) for t in result]
            stages.append(dict(fn=fn, inputs=owned, expected=expected, row=row))
            report['stages'].append(row)
            return result

        original_stage=stage_reference['stages'][6]
        assert original_stage['name']=='ViT_8_blocks'
        features=torch.from_numpy(arrays.load(original_stage['inputs'][0])).to('xpu').reshape(-1,1024)
        report['vit_tokens']=features.shape[0]
        module=model.vit[0]
        original=tuple(cpu_bytes(t) for t in module.forward_boundaries(features))
        def expand(a):
            x=vb.q(a)
            return x,vb.cubic_quantize(vb.dot(x,module.expand,chunk_k=16))
        def contract(hidden,x):
            return vb.q(vb.split_k_projection(hidden,module.contract,(x*module.ffn_skip).half()))
        def qkv(mlp):
            z=(vb.dot(mlp[:,:512],module.qkv_weight[:512],chunk_k=16)+vb.dot(mlp[:,512:],module.qkv_weight[512:],chunk_k=16)).half().reshape(-1,32,3,32)
            query=vb.q((vb.normalize_c32(z[:,:,0])*5.65625).half()*module.query_scale[None,:,None])
            key=vb.q(vb.normalize_c32(z[:,:,1]));value=vb.q(z[:,:,2])
            return query,key,value
        def attention(q,k,v):
            return vb.vit_attention(q.transpose(0,1),k.transpose(0,1),v.transpose(0,1)).transpose(0,1).reshape(-1,1024)
        def projection(attended,mlp):
            return vb.q(vb.split_k_projection(attended,module.projection,(mlp*module.attn_skip).half()))
        x,hidden=stage('ffn_expand',expand,features)
        mlp,=stage('ffn_contract',contract,hidden,x)
        query,key,value=stage('qkv_projection_and_prepare',qkv,mlp)
        attended,=stage('attention',attention,query,key,value)
        result,=stage('output_projection',projection,attended,mlp)
        assert tuple(cpu_bytes(t) for t in (hidden,mlp,query,key,value,attended,result))==original
        report['first_block_all_seven_boundaries_match']=True
        for module_remaining in model.vit[1:]:result=module_remaining(result)
        assert cpu_bytes(result)==arrays.load(original_stage['outputs'][0]).tobytes()
        report['eight_blocks_match_prior']=True
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
            pool = graph.pool()
            assert not any(s['address'] <= t.data_ptr() < s['address']+s['total_size']
                for s in torch.xpu.memory_snapshot(pool) for t in [*public,*[a for a in args if a is not None]])
            item['row']['persistent_io_outside_pool'] = True
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
print(json.dumps(dict(passed=report['passed'],full_body=report['full_body_median_seconds'],
    stages={r['name']:r['median_seconds'] for r in report['stages']}),indent=2),flush=True)
