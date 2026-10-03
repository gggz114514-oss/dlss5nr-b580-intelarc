"""Measure all remaining actual NR256 dense geometries under isolated Triton3.8.

Reconstruct original tensor strides, retain BK32 and original FP16 arithmetic,
compare every complete output against the captured reference, then measure
five rotated/reversed rounds, four calls per graph and ten replays per sample.
Use same-round >=2% wins and >=5% median savings to select a tile. Local dense
measurements weighted by call count are estimates, not complete NR timings.
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
OUT = DREF / 'experimental/current-dense-tiles-v1'
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
pipeline_path = DREF / 'results/vit-qkv-residual256-v2/validation.json'
assert sha(pipeline_path) == '928bb145fbd02023f4abaaa6b1e214ddbdc75d643a6df553385a7ffd9a5803c7'
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
from fused_vit_qkv_adapter_v1 import FusedVitQKV
from fused_swin_native_half_v1 import FusedSwin
from fused_dynamic_front_v1 import FusedFront
from fused_graph_history_warp_v2 import FusedHistoryWarp
from residual_scale_v1 import ResidualScale
from graph_front_v6 import GraphFront
from native_half_head_layout_v1 import HeadLayout
from fused_split_ffwd_v2 import FusedSplit
import capture_body_v1 as body
import compressed_arrays_v1 as arrays

inputs = R / 'inputs/flow-full-1920x1080-v3'
manifest_path = inputs / 'manifest.json'
manifest = js(manifest_path)
paths = [Path(__file__), HERE / 'Run-CurrentDenseTilesV1.cmd', provision_path, prior_path, pipeline_path, manifest_path,
         *[Path(p) for p in pipeline['sources']]]
capture_path=DREF/'experimental/current-dense-operands-v2/validation.json'
assert sha(capture_path)=='46a16ba64e9f84425928d0a0fd93862e99574910b031ba098c789f0ca69e7290'
capture=js(capture_path)
assert capture['passed'] and js(capture_path.parent.with_suffix('.log.lease.json'))['returncode']==0
assert all(sha(Path(p))==h for p,h in capture['sources'].items())
paths.extend([capture_path,HERE/'dense_tiles_v1.py',HERE/'fast_matrices_v3.py',*[Path(p) for p in capture['sources']]])
frozen = {str(p): sha(p) for p in paths}
from dense_tiles_v1 import dot as candidate
from fast_matrices_v3 import dot as previous

OUT.mkdir()
report=dict(scope=__doc__,sources=frozen,exact_gate=gate,passed=False,cases=[],nr_input=[256,256],source_canvas=[1920,1080],
            complete_migration=False,rounds=5,calls_per_graph=4,replays_per_sample=10)
report['runtime']=dict(triton_version=triton.__version__,torch_version=torch.__version__,triton_file=triton.__file__,isolated=True)
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
def raw(t):return t.cpu().numpy().tobytes()
configs=[(16,32,32,4),(16,64,32,4),(32,32,32,4),(32,64,32,4),(32,128,32,4),(64,32,32,4),(64,64,32,4),(64,128,32,4)]
entries={}
try:
    torch.set_num_threads(2)
    with torch.inference_mode():
        cases=sorted(capture['cases'].items(),key=lambda pair:pair[1]['count']*int(np.prod(pair[1]['shape'])),reverse=True)
        for key,old in cases:
            operands={};cpu={}
            for name,meta in old['operands'].items():
                if meta is None:operands[name]=None;cpu[name]=None;continue
                value=arrays.load(meta);desc=old['descriptors'][name]
                tensor=torch.empty_strided(tuple(desc['shape']),tuple(desc['stride']),dtype=torch.float16,device='xpu')
                tensor.copy_(torch.from_numpy(value.copy()).to('xpu'))
                assert tuple(tensor.stride())==tuple(desc['stride']) and raw(tensor)==value.tobytes()
                operands[name]=tensor;cpu[name]=value
            a,w,initial=(operands[k] for k in ('a','w','initial'))
            expected=arrays.load(old['output']).tobytes()
            assert raw(previous(a,w,initial=initial)[0])==expected
            row=dict(key=key,shape=old['shape'],count=old['count'],layout=old['layout'],initialized=old['initialized'],
                     operands=old['operands'],descriptors=old['descriptors'],expected=old['output'],candidates={},samples_ms={},orders=[])
            report['cases'].append(row);entries={}
            for config in configs:
                label='x'.join(map(str,config))
                fn=(lambda:previous(a,w,initial=initial)) if config==configs[0] else (lambda c=config:candidate(a,w,initial=initial,tile=c))
                value,kernel=fn();equal=raw(value)==expected
                row['candidates'][label]=dict(tile=list(config),byte_equal=equal)
                if not equal:
                    row['candidates'][label]['actual']=arrays.save(value.cpu().numpy());continue
                assert 'ttig.dpas' in str(kernel.asm['ttgir'])
                row['candidates'][label]['ttgir_sha256']=hashlib.sha256(str(kernel.asm['ttgir']).encode()).hexdigest()
                stream=torch.xpu.Stream()
                with stream:
                    for _ in range(2):warm=fn()[0]
                torch.xpu.synchronize();public=torch.empty_like(warm);del warm
                graph=torch.xpu.XPUGraph()
                with torch.xpu.graph(graph,stream=stream):
                    for _ in range(4):temporary=fn()[0]
                    public.copy_(temporary)
                del temporary
                assert not any(seg['address']<=t.data_ptr()<seg['address']+seg['total_size'] for seg in torch.xpu.memory_snapshot(graph.pool()) for t in (a,w,initial,public) if t is not None)
                entries[label]=(graph,public);row['samples_ms'][label]=[]
                graph.replay();torch.xpu.synchronize();assert raw(public)==expected
                row['candidates'][label].update(graph_byte_equal=True,persistent_io_outside_pool=True)
            names=list(entries);baseline='x'.join(map(str,configs[0]));assert baseline in entries
            for repetition in range(5):
                offset=repetition%len(names);order=names[offset:]+names[:offset]
                if repetition%2:order.reverse()
                row['orders'].append(order)
                for name in order:
                    graph,value=entries[name]
                    torch.xpu.synchronize();start=time.perf_counter()
                    for _ in range(10):graph.replay()
                    torch.xpu.synchronize();row['samples_ms'][name].append((time.perf_counter()-start)*1000/40)
                    assert raw(value)==expected
            medians={name:statistics.median(ts) for name,ts in row['samples_ms'].items()}
            eligible=[name for name in names if name!=baseline and medians[name]<medians[baseline]*.95 and all(b/c>1.02 for b,c in zip(row['samples_ms'][baseline],row['samples_ms'][name]))]
            selected=min(eligible,key=medians.get) if eligible else baseline
            row.update(baseline=baseline,selected=selected,median_ms=medians,speedup=medians[baseline]/medians[selected],weighted_saving_ms=old['count']*(medians[baseline]-medians[selected]))
            assert all(t is None or raw(t)==cpu[name].tobytes() for name,t in operands.items())
            row['inputs_unchanged']=True
            for graph,value in entries.values():assert raw(value)==expected;graph.reset()
            entries={};save()
            print(json.dumps(dict(key=key,count=old['count'],selected=selected,baseline_ms=medians[baseline],selected_ms=medians[selected],weighted_saving_ms=row['weighted_saving_ms'],rejected=[name for name,c in row['candidates'].items() if not c['byte_equal']])),flush=True)
        report['selected_changes']=sum(c['baseline']!=c['selected'] for c in report['cases'])
        report['weighted_saving_ms']=sum(c['weighted_saving_ms'] for c in report['cases'])
        report['all_original_dense_calls_accounted']=sum(c['count'] for c in report['cases'])==167
        assert report['all_original_dense_calls_accounted']
        report['passed']=True
except Exception as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    for graph,value in entries.values():graph.reset()
    try:
        assert all(sha(Path(p))==h for p,h in frozen.items());authenticate_main()
    except Exception as error:report.update(passed=False,finalization_error=repr(error));raise
    finally:save()
print(json.dumps(dict(passed=report['passed'],selected_changes=report['selected_changes'],weighted_saving_ms=report['weighted_saving_ms'])),flush=True)
