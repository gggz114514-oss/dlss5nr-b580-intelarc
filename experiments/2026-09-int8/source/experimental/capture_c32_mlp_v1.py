"""Real C32 whole-MLP operands from the authenticated same-precision full480 reset.

Collect the first instance per M,C geometry, including complete module weights
and unquantized output. Collection does not alter the original module execution.
"""
import hashlib,json,os,sys,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580';R=EXACT/'reference'
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=DREF/'experimental/c32-mlp-operands-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-sm89-v1');sys.path[:0]=[str(R),str(ROOT/'backend')]
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import js,sha as original_sha
def sha(path):return original_sha(Path(path))
gate=authenticate_main()
prior_path=DREF/'results/fast-precision-864x480-v1/validation.json'
assert sha(prior_path)=='7a43ff73d452cabe3e68ea1a15db1537417384a57c9a2443fcb16b095d5a5110'
prior=js(prior_path)
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:queue=json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
import numpy as np
from PIL import Image
import torch
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from nr_backend.pre_mlp import C32MLP as BranchedMLP
from strided_batched_v2 import StridedMatrices
from swin_scheduling_v1 import SwinScheduling
from window_layout_v1 import WindowLayout
import compressed_arrays_v1 as arrays
inputs=R/'inputs/flow-full-864x480-v2';manifest=js(inputs/'manifest.json')
names=['capture_c32_mlp_v1.py','Run-CaptureC32MLPV1.cmd','strided_batched_v2.py','strided_batched_v1.py','swin_scheduling_v1.py','window_layout_v1.py','fused_cached_matrices_v2.py','fused_cached_matrices_v1.py','fused_activation_int8_v1.py','static_weight_cache_v2.py','static_weight_cache_v1.py','fast_matrices_v3.py','compressed_arrays_v1.py','immutable_artifacts_v1.py']
paths=[*(HERE/name for name in names),prior_path,inputs/'manifest.json',*sorted((ROOT/'backend/nr_backend').glob('*.py'))]
frozen={str(path):sha(path) for path in paths}
OUT.mkdir();report=dict(scope=__doc__,sources=frozen,exact_gate=gate,passed=False,cases={},raw_bytes=0,maximum_raw_bytes=256*2**20,complete_migration=False)
original=BranchedMLP.forward_unquantized
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
def collect(module,features):
    key=f'{features.numel()//32}x32'
    result=original(module,features)
    if features.numel()>32768*32:return result
    if key not in report['cases']:
        row=dict(shape=list(features.shape),channels=32,count=0,captured=False,module_name=module_names[id(module)])
        report['cases'][key]=row
        values=dict(features=features,expansion=module.expansion,contraction=module.contraction,skip_scale=module.skip_scale,output=result)
        size=sum(t.numel()*t.element_size() for t in values.values())
        if report['raw_bytes']+size<=report['maximum_raw_bytes']:
            row['arrays']={}
            for name,value in values.items():
                a=value.cpu().numpy();assert np.isfinite(a).all()
                row['arrays'][name]=arrays.save(a)
            row['captured']=True;report['raw_bytes']+=size
            print(json.dumps(dict(captured=key,module=row['module_name'],raw_bytes=size)),flush=True)
    report['cases'][key]['count']+=1
    return result
try:
    torch.set_num_threads(2)
    model=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    module_names={id(module):name for name,module in model.named_modules() if isinstance(module,BranchedMLP)}
    experiment=StridedMatrices();experiment.select('fp16_xmx')
    spec=manifest['frames'][0];pixels=np.asarray(Image.open(inputs/spec['file']).convert('RGB'),dtype='f4')/255
    flow=np.fromfile(inputs/spec['motion_file'],'<f4').reshape(480,864,2)
    rgb,motion=torch.from_numpy(pixels).to('xpu'),torch.from_numpy(flow).to('xpu')
    BranchedMLP.forward_unquantized=collect
    with torch.inference_mode(),experiment.installed(),SwinScheduling(enabled=True).installed(),WindowLayout().installed(),use_arithmetic_backend('triton') as dispatch:
        value=model(rgb,motion,reset=True)
    actual=value.cpu().numpy();old=prior['runs']['fp16_xmx'][0];meta=old['actual']
    assert sha(meta['path'])==meta['sha256']
    assert actual.tobytes()==np.load(meta['path'],allow_pickle=False).astype('f2').tobytes()
    assert model._previous.cpu().numpy().tobytes()==actual.tobytes() and dict(dispatch)==old['dispatches']
    assert rgb.cpu().numpy().tobytes()==pixels.tobytes() and motion.cpu().numpy().tobytes()==flow.tobytes()
    report.update(passed=True,full_output=meta,full_output_byte_equal_prior=True,private_equal=True,next_seed=model.next_seed,dispatch=dict(dispatch),total_mlp_calls=sum(row['count'] for row in report['cases'].values()))
except Exception as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    assert BranchedMLP.forward_unquantized in (original,collect)
    BranchedMLP.forward_unquantized=original
    assert all(sha(path)==digest for path,digest in frozen.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'],geometries=len(report['cases']),raw_bytes=report['raw_bytes'])),flush=True)
