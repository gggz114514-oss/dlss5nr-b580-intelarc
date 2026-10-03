"""Capture reset/temporal complete value graphs after the selected FP8 rewrite.

The reusable stack is compared in full against the approved first two frames.
Keep scalar operation metadata and the escaping final result for conservative
producer/consumer rewrites. No performance timing or additional frame dumps.
"""
import hashlib,json,os,sys,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580';R=EXACT/'reference'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'experimental/full-body-dataflow-v1';assert not OUT.exists()
TOOLCHAIN=D/'toolchains/triton-xpu-3.8.0-git1e2d42a0'
os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-c32-triton38-v1')
sys.path[:0]=[str(TOOLCHAIN/'site'),str(R),str(ROOT/'backend')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main()
provision=TOOLCHAIN/'provision-v1.json'
assert sha(provision)=='e94757afcb5962cacbe208cbb808b3516de210059335f1ee7ab7b40d6bfe2c07'
assert all(sha(TOOLCHAIN/'site'/p)==h for p,h in js(provision)['files'].items())
prior_path=D/'results/residual-scale-fp16_xmx-256-v1/validation.json'
assert sha(prior_path)=='056251b4e2a839fcc44845d9646d2cd8ae1e9ed7f99fd1858dc7e3d51e3e50f9'
prior=js(prior_path)
pipeline_path=D/'results/fp8-graph-residual256-v1/validation.json'
assert sha(pipeline_path)=='7a45f2bdf93e9cfc45778df3703cf37172c1cb84667459af178b877c52c28af7'
pipeline=js(pipeline_path);assert pipeline['passed'] and js(pipeline_path.parent.with_suffix('.log.lease.json'))['returncode']==0
assert js(pipeline_path.with_name('saved-audit-v1.json'))['passed']
assert all(sha(p)==h for p,h in pipeline['sources'].items())
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:queue=json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
import numpy as np
from PIL import Image
import torch
import triton
assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site') and triton.__version__.startswith('3.8.0')
from nr256_selected_stack_v1 import Stack
from full_body_dataflow_v1 import FullTraceGraph
from nr_backend.execution import use_arithmetic_backend
from cubic_lut_constant_v1 import Constant,TABLE
from residual_scale_v1 import ResidualScale
import compressed_arrays_v1 as arrays
inputs=R/'inputs/flow-full-1920x1080-v3';manifest_path=inputs/'manifest.json';manifest=js(manifest_path)
sources=dict(pipeline['sources'])
for p in (Path(__file__),HERE/'Run-FullBodyDataflowV1.cmd',HERE/'nr256_selected_stack_v1.py',HERE/'full_body_dataflow_v1.py',provision,prior_path,pipeline_path,manifest_path):sources[str(p)]=sha(p)
OUT.mkdir();report=dict(scope=__doc__,sources=sources,exact_gate=gate,passed=False,complete_migration=False,frames=[],timing_measured=False)
stack=None
raw=lambda t:t.cpu().numpy().tobytes()
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
try:
    torch.set_num_threads(2);stack=Stack(EXACT,rewrite_type=FullTraceGraph);scaler=ResidualScale(256)
    with torch.inference_mode(),stack.installed(),use_arithmetic_backend('triton'):
        for i in (0,1):
            spec=manifest['frames'][i]
            pixels=np.asarray(Image.open(inputs/spec['file']).convert('RGB'),dtype='f4')/255
            flow=np.fromfile(inputs/spec['motion_file'],'<f4').reshape(1080,1920,2)
            rgb=torch.from_numpy(pixels).to('xpu');motion=torch.from_numpy(flow).to('xpu')
            low_rgb,low_motion=scaler.prepare(rgb,motion)
            low=stack.model(low_rgb,low_motion,reset=i==0);output=scaler.composite(rgb,low_rgb,low)
            expected=prior['runs'][i]
            assert raw(low)==arrays.load(expected['low_nr']).tobytes()
            assert raw(output)==arrays.load(expected['output']).tobytes()
            assert raw(stack.model._previous)==raw(low) and stack.model.next_seed==i+1
            assert raw(rgb)==pixels.tobytes() and raw(motion)==flow.tobytes()
            report['frames'].append(dict(frame=i,low_nr=expected['low_nr'],output=expected['output'],byte_equal=True,history_equal=True,next_seed=i+1,inputs_unchanged=True))
    report['graphs']=stack.rewrite.records;report['builds']=stack.rewrite.builds
    assert set(report['graphs'])=={'reset','temporal'} and len(report['builds'])==6
    assert all(b['triton_calls']==755 and b['standalone_fp8']==232 and b['elided_fp8']==231 for b in report['builds'])
    assert raw(Constant(stack.model).require())==np.load(TABLE,allow_pickle=False).view('i2').tobytes()
    with stack.installed():stack.graph._validate()
    report.update(passed=True,lut_unchanged=True,graph_metadata=stack.graph.metadata(),runtime=dict(triton=triton.__version__,torch=torch.__version__))
except BaseException as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    if stack is not None:stack.close()
    assert all(sha(p)==h for p,h in sources.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'],counts={k:{n:v['summary'][n] for n in ('triton_calls','standalone_fp8','elided_fp8')} for k,v in report.get('graphs',{}).items()}),indent=2),flush=True)
