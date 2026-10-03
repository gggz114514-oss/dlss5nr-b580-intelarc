"""Actual eight-block QKV input comparisons and paired static replay timings under Triton3.8.

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
OUT = DREF / 'experimental/fused-vit-qkv-v1'
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
from fused_vit_qkv_v1 import forward as fused_qkv

inputs = R / 'inputs/flow-full-1920x1080-v3'
manifest_path = inputs / 'manifest.json'
manifest = js(manifest_path)
paths = [Path(__file__), HERE / 'Run-FusedVitQkvV1.cmd', provision_path, prior_path, pipeline_path, manifest_path, stage_path, HERE/'fused_vit_qkv_v1.py',
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
report['cases']=[]
report['scope']='Eight actual ViT QKV inputs; old full QKV outputs versus four fusion tiles. Five rotated static graph timing rounds, four calls per graph and ten replays per sample. No whole NR speed claim.'
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

        original_stage=stage_reference['stages'][6]
        features=torch.from_numpy(arrays.load(original_stage['inputs'][0])).to('xpu').reshape(-1,1024)
        configs=[(16,32),(16,64),(32,32),(32,64)]
        def previous(a,w,scale):
            z=(vb.dot(a[:,:512],w[:512],chunk_k=16)+vb.dot(a[:,512:],w[512:],chunk_k=16)).half().reshape(-1,32,3,32)
            q=vb.q((vb.normalize_c32(z[:,:,0])*5.65625).half()*scale[None,:,None])
            return q,vb.q(vb.normalize_c32(z[:,:,1])),vb.q(z[:,:,2])
        for index,module in enumerate(model.vit):
            boundaries=module.forward_boundaries(features)
            a=boundaries[1].clone();w=module.qkv_weight;scale=module.query_scale
            expected=tuple(cpu_bytes(t) for t in boundaries[2:5])
            assert tuple(cpu_bytes(t) for t in previous(a,w,scale))==expected
            row=dict(index=index,operands={name:arrays.save(t.cpu().numpy()) for name,t in [('a',a),('w',w),('scale',scale)]},
                     outputs=[arrays.save(t.cpu().numpy()) for t in boundaries[2:5]],candidates={},samples_ms={})
            report['cases'].append(row)
            variants={'previous':lambda:previous(a,w,scale)}
            variants.update({f'b{bm}n{bn}':lambda bm=bm,bn=bn:fused_qkv(a,w,scale,bm=bm,bn=bn)[0] for bm,bn in configs})
            entries={}
            for name,fn in variants.items():
                result=fn();equal=tuple(cpu_bytes(t) for t in result)==expected
                row['candidates'][name]=dict(eager_byte_equal=equal)
                if not equal:
                    row['candidates'][name]['actual']=[arrays.save(t.cpu().numpy()) for t in result]
                    continue
                stream=torch.xpu.Stream()
                with stream:
                    for _ in range(2):warm=fn()
                torch.xpu.synchronize();outputs=tuple(torch.empty_like(t) for t in warm)
                graph=torch.xpu.XPUGraph()
                stages.append(dict(graph=graph))
                with torch.xpu.graph(graph,stream=stream):
                    for _ in range(4):temporary=fn()
                    for dest,src in zip(outputs,temporary):dest.copy_(src)
                del temporary,warm
                assert not any(seg['address']<=t.data_ptr()<seg['address']+seg['total_size'] for seg in torch.xpu.memory_snapshot(graph.pool()) for t in (a,w,scale,*outputs))
                entries[name]=(graph,outputs);row['samples_ms'][name]=[]
                graph.replay();torch.xpu.synchronize()
                assert tuple(cpu_bytes(t) for t in outputs)==expected
                row['candidates'][name]['graph_byte_equal']=True
                row['candidates'][name]['persistent_io_outside_pool']=True
            names=list(entries)
            for repetition in range(5):
                order=names[repetition:]+names[:repetition]
                if repetition%2:order.reverse()
                for name in order:
                    graph,outputs=entries[name]
                    torch.xpu.synchronize();start=time.perf_counter()
                    for _ in range(10):graph.replay()
                    torch.xpu.synchronize();row['samples_ms'][name].append((time.perf_counter()-start)*1000/40)
                    assert tuple(cpu_bytes(t) for t in outputs)==expected
            row['median_ms']={name:statistics.median(ts) for name,ts in row['samples_ms'].items()}
            row['inputs_unchanged']=all(cpu_bytes(t)==arrays.load(row['operands'][name]).tobytes() for name,t in [('a',a),('w',w),('scale',scale)])
            assert row['inputs_unchanged']
            for graph,outputs in entries.values():graph.reset()
            stages.clear()
            features=boundaries[-1]
            print(json.dumps(dict(index=index,median_ms=row['median_ms'],rejected=[name for name,v in row['candidates'].items() if not v['eager_byte_equal']])),flush=True)
            save()
        assert cpu_bytes(features)==arrays.load(original_stage['outputs'][0]).tobytes()
        report['eight_blocks_match_prior']=True
        report['all_candidates_byte_equal']=all(v['eager_byte_equal'] and v.get('graph_byte_equal',False) for row in report['cases'] for v in row['candidates'].values())
        report['sum_median_ms']={name:sum(row['median_ms'][name] for row in report['cases']) for name in variants if all(name in row['median_ms'] for row in report['cases'])}
        assert cpu_bytes(model._previous)==private and model.next_seed==next_seed
        report['history_unchanged']=True
        adapter._validate();constant.require()
        report['passed']=True
except Exception as error:
    report.update(error=repr(error),traceback=traceback.format_exc())
    raise
finally:
    for item in stages:
        if 'graph' in item:item['graph'].reset()
    if warp is not None:warp.close()
    if adapter is not None:adapter.close()
    assert all(sha(Path(p))==h for p,h in frozen.items())
    authenticate_main();save()
print(json.dumps(dict(passed=report['passed'],sum_median_ms=report['sum_median_ms'],all_candidates_byte_equal=report['all_candidates_byte_equal'])),flush=True)
