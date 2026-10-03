"""Capture actual current NR256 fused MLP inputs and complete outputs.

Keep one complete case per family and geometry, count all actual calls, and
verify the complete temporal body. Capturing CPU arrays is outside timing.
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
OUT = DREF / 'experimental/current-cubic-operands-v1'
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
pipeline_path = DREF / 'results/current-dense-residual256-v1/validation.json'
assert sha(pipeline_path) == '238bf5ba349a8bb1092fa842d4356fa7ddf40c87e647664eb5a8cbacfa55bc38'
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
from current_dense_tiled_provider_v1 import CurrentDenseTiledMatrices
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
paths = [Path(__file__), HERE / 'Run-CurrentCubicOperandsV1.cmd', provision_path, prior_path, pipeline_path, manifest_path,
         *[Path(p) for p in pipeline['sources']]]
frozen = {str(p): sha(p) for p in paths}
OUT.mkdir()
report=dict(scope=__doc__,sources=frozen,exact_gate=gate,passed=False,cases={},complete_migration=False,
            nr_input=[256,256],source_canvas=[1920,1080],operand_raw_bytes=0,maximum_operand_raw_bytes=128*2**20)
report['runtime']=dict(triton_version=triton.__version__,torch_version=torch.__version__,triton_file=triton.__file__,isolated=True)
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
def raw(t):return t.cpu().numpy().tobytes()

from contextlib import contextmanager
import inspect
import fused_split_ffwd_v2 as split_module
import batched_branched_mlp_v1 as batched_module
import fused_c32_mlp_lut_v1 as c32_module

@contextmanager
def collecting():
    originals=[]
    for family,module in [('split',split_module),('batched',batched_module),('c32',c32_module)]:
        original=module.forward
        signature=inspect.signature(original)
        def wrapped(*args,_family=family,_original=original,_signature=signature,**kwargs):
            result=_original(*args,**kwargs)
            bound=_signature.bind(*args,**kwargs);bound.apply_defaults()
            first=args[0];channels=first.shape[-1];rows=first.numel()//channels
            key=f'{_family}:{rows}x{channels}'
            if key not in report['cases']:
                tensors={n:t for n,t in bound.arguments.items() if isinstance(t,torch.Tensor)}
                options={n:t for n,t in bound.arguments.items() if not isinstance(t,torch.Tensor)}
                size=sum(t.numel()*t.element_size() for t in [*tensors.values(),result[0]])
                assert report['operand_raw_bytes']+size<=report['maximum_operand_raw_bytes']
                report['cases'][key]=dict(family=_family,rows=rows,channels=channels,count=0,options=options,
                    operands={n:arrays.save(t.cpu().numpy()) for n,t in tensors.items()},
                    descriptors={n:dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype)) for n,t in tensors.items()},
                    expected=arrays.save(result[0].cpu().numpy()))
                report['operand_raw_bytes']+=size
                print(json.dumps(dict(captured=key,raw_bytes=size)),flush=True)
            report['cases'][key]['count']+=1
            return result
        module.forward=wrapped;originals.append((module,original,wrapped))
    try:yield
    finally:
        for module,original,wrapped in reversed(originals):
            assert module.forward is wrapped;module.forward=original

adapter=warp=None
try:
    torch.set_num_threads(2)
    model=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    lut=register(model);constant=Constant(model)
    provider=CurrentDenseTiledMatrices(model);provider.select('fp16_xmx')
    adapter=GraphFront(model,arithmetic=provider);warp=FusedHistoryWarp(model);scaler=ResidualScale(256)
    from contextlib import ExitStack
    components=[provider,FusedSwin(provider),HeadLayout(model,provider),adapter,FusedBatched(model,provider,workload='small'),
                FusedSplit(model,provider),FusedC32(model,provider,bm=32),FusedVitProjection(model,provider),FusedFront(model),warp,FusedVitQKV(model,provider)]
    with torch.inference_mode(),ExitStack() as stack:
        for component in components:stack.enter_context(component.installed())
        stack.enter_context(body.installed());stack.enter_context(use_arithmetic_backend('triton'))
        for i in (0,1):
            spec=manifest['frames'][i]
            rgb=torch.from_numpy(np.asarray(Image.open(inputs/spec['file']).convert('RGB'),dtype='f4')/255).to('xpu')
            motion=torch.from_numpy(np.fromfile(inputs/spec['motion_file'],'<f4').reshape(1080,1920,2)).to('xpu')
            low_rgb,low_motion=scaler.prepare(rgb,motion)
            nr=model(low_rgb,low_motion,reset=i==0)
            assert raw(nr)==arrays.load(prior['runs'][i]['low_nr']).tobytes()
            assert raw(scaler.composite(rgb,low_rgb,nr))==arrays.load(prior['runs'][i]['output']).tobytes()
        entry=adapter.last_entry
        static={name:None if t is None else t.clone() for name,t in entry.inputs.items()}
        private=raw(model._previous);seed=model.next_seed
        provider.calls={};components[-1].calls=0
        with collecting(), use_arithmetic_backend('triton') as dispatch:
            value=body.forward_front(model,**static,sigmoid=model.sigmoid,blend_scale=model.blend_scale,return_float32=False)
        torch.xpu.synchronize()
        assert raw(value)==arrays.load(prior['runs'][1]['low_nr']).tobytes()
        assert raw(model._previous)==private and model.next_seed==seed
        assert components[-1].calls==8
        assert {row['family'] for row in report['cases'].values()}=={'split','batched','c32'}
        report.update(output=prior['runs'][1]['low_nr'],byte_equal_prior=True,history_unchanged=True,
                      matrix_calls=dict(provider.calls),dispatch=dict(dispatch),vit_qkv_calls=components[-1].calls)
        adapter._validate();assert constant.require() is lut
        report['passed']=True
except Exception as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    if warp is not None:warp.close()
    if adapter is not None:adapter.close()
    try:
        assert all(sha(Path(p))==h for p,h in frozen.items())
        authenticate_main()
    except Exception as error:
        report.update(passed=False,finalization_error=repr(error));raise
    finally:save()
print(json.dumps(dict(passed=report['passed'],shapes=len(report['cases']),calls=report['matrix_calls'],raw_bytes=report['operand_raw_bytes'])),flush=True)
