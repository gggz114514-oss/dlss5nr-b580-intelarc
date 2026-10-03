"""Independent exact/FP16/W8A8 complete NR streams; no reduced resolution."""
import argparse,hashlib,json,os,runpy,statistics,sys,time,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
EXACT=ROOT.parent/'nr-b580'
R=EXACT/'reference'
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference')
p=argparse.ArgumentParser()
p.add_argument('--dimension',choices=['864x480','1920x1080'],required=True)
args=p.parse_args()
dimension=args.dimension
w,h=map(int,dimension.split('x'))
OUT=DREF/f'results/fast-precision-{dimension}-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-sm89-v1')
sys.path.insert(0,str(R))
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import authenticated,js,sha
gate=authenticate_main()
preflight=R/'validate_attention_weights_main_motion_v1.py'
sys.argv=[str(preflight),'--dimension',dimension,'--version','1','--preflight-only']
try:runpy.run_path(str(preflight),run_name='__main__')
except SystemExit as error:assert error.code==0
primitive_path=DREF/'experimental/fast-matrices-v1/primitive-v5.json'
primitive=js(primitive_path)
assert primitive['passed'] and len(primitive['cases'])==16
assert all(sha(Path(name))==digest for name,digest in primitive['sources'].items())
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:queue=json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import torch
sys.path.insert(0,str(ROOT/'backend'))
import nr_backend
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from fast_matrices_v3 import FastMatrices
import immutable_artifacts_v1 as artifacts
assert Path(nr_backend.__file__).resolve().parent==ROOT/'backend/nr_backend'
for source in (EXACT/'backend/nr_backend').glob('*.py'):
    assert source.read_bytes()==(ROOT/'backend/nr_backend'/source.name).read_bytes()
version=2 if dimension=='864x480' else 3
inputs=R/f'inputs/flow-full-{dimension}-v{version}'
manifest=js(inputs/'manifest.json')
native_meta=R/f'4060-real-flow-{dimension}-sequence-v{version}.json'
native=authenticated(js(native_meta))
stage=js(DREF/f'results/attention-weights-stage-{dimension}-rgb-b580-v1/validation.json')
assert stage['all_byte_equal'] and len(stage['frames'])==13
frozen={str(path):sha(path) for path in [Path(__file__),HERE/'Run-FullPrecisionV1.cmd',HERE/'fast_matrices_v3.py',HERE/'immutable_artifacts_v1.py',primitive_path,inputs/'manifest.json',native_meta,*sorted((ROOT/'backend/nr_backend').glob('*.py'))]}
modes=('baseline','fp16_xmx','int8_dense')
orders=[list(modes[i%3:]+modes[:i%3]) for i in range(13)]
OUT.mkdir(parents=True)
report=dict(scope=__doc__,dimension=dimension,sources=frozen,exact_gate=gate,runs={mode:[] for mode in modes},execution_orders=orders,passed=False,promoted=False,complete_migration=False,human_review='pending',timing_scope='Three simultaneously resident models with independent histories. Rotating execution order. Synchronized complete model call; excludes upload, validation, disk IO, startup weight packing and first-frame JIT. Mean frames1..12 includes repeated reset.',initial_queue=queue,downsampling=False)
experiment=FastMatrices()
models={}
last_error={}

def save():
    (OUT/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def metrics(actual,target,mode,reset):
    error=actual.astype('f4')-target
    mse=float(np.mean(error.astype('f8')**2))
    value=dict(psnr_db=None if mse==0 else float(-10*np.log10(mse)),ssim=float(structural_similarity(target,actual,data_range=1,channel_axis=2)),mae=float(np.mean(np.abs(error))),max_abs=float(np.max(np.abs(error))),error_p99=float(np.quantile(np.abs(error),.99)),byte_equal=actual.tobytes()==target.tobytes())
    # This is unwarped successive error change, NOT an optical-flow flicker score.
    value['successive_error_delta_mae']=None if reset or mode not in last_error else float(np.mean(np.abs(error-last_error[mode])))
    last_error[mode]=error.copy()
    return value

try:
    torch.set_num_threads(2)
    with torch.inference_mode(),experiment.installed():
        for mode in modes:
            print('Loading independent model '+mode,flush=True)
            models[mode]=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
        started=time.perf_counter()
        experiment.prepack(models['int8_dense'])
        report['prepack_seconds']=time.perf_counter()-started
        report['prepacked_buffers']=len(experiment.packed)
        report['prepacked_bytes']=sum(q.numel()*q.element_size()+s.numel()*s.element_size() for _,q,s,_ in experiment.packed.values())
        for i,spec in enumerate(manifest['frames']):
            pixels=np.asarray(Image.open(inputs/spec['file']).convert('RGB'),dtype='f4')/255
            flow=np.fromfile(inputs/spec['motion_file'],'<f4').reshape(h,w,2)
            native_path=native/f"{spec['file']}_output.rgba32f.bin"
            target=np.fromfile(native_path,'<f4').reshape(h,w,4)[...,:3].copy()
            for mode in orders[i]:
                print(f'{dimension} frame{i} {mode} start',flush=True)
                model=models[mode]
                experiment.select(mode)
                rgb=torch.from_numpy(pixels).to('xpu')
                motion=torch.from_numpy(flow).to('xpu')
                torch.xpu.synchronize()
                torch.xpu.reset_peak_memory_stats()
                started=time.perf_counter()
                with use_arithmetic_backend('triton') as dispatch:
                    result=model(rgb,motion,reset=spec['reset'])
                    torch.xpu.synchronize()
                seconds=time.perf_counter()-started
                actual=result.cpu().float().numpy()
                private=model._previous.cpu().numpy()
                row=dict(frame=i,reset=spec['reset'],seconds=seconds,next_seed=model.next_seed,dispatches=dict(dispatch),matrix_calls=experiment.calls.copy(),peak_allocated_bytes=torch.xpu.max_memory_allocated(),native_path=str(native_path),native_file_sha256=sha(native_path),rgb_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),private=artifacts.array(private),finite=bool(np.isfinite(actual).all()))
                compact=actual.astype('f2')
                lossless=compact.astype('f4').tobytes()==actual.tobytes()
                row['actual']=artifacts.array(compact if lossless else actual)
                row['rgb32f_losslessly_stored_as_half']=lossless
                report['runs'][mode].append(row)
                save()
                assert row['finite'], f'Nonfinite frame: {mode}'
                row.update(metrics(actual,target,mode,spec['reset']))
                assert dict(dispatch)==stage['frames'][i]['arithmetic_dispatches']
                if mode=='baseline':
                    assert row['byte_equal'] and private.tobytes()==target.astype('f2').tobytes()
                else:
                    assert experiment.calls.get('fp16_batched',0)>0
                    assert experiment.calls.get('int8_dense' if mode=='int8_dense' else 'fp16_dense',0)>0
                assert model.next_seed==(1 if spec['reset'] else i+1)
                assert rgb.cpu().numpy().tobytes()==pixels.tobytes() and motion.cpu().numpy().tobytes()==flow.tobytes()
                result.zero_()
                assert model._previous.cpu().numpy().tobytes()==private.tobytes()
                save()
                print(json.dumps(dict(frame=i,mode=mode,seconds=seconds,psnr_db=row['psnr_db'],ssim=row['ssim'],matrix_calls=row['matrix_calls'])),flush=True)
        for mode,model in models.items():
            assert report['runs'][mode][0]['rgb_sha256']==report['runs'][mode][12]['rgb_sha256']
            experiment.select(mode)
            seed,previous=model.next_seed,model._previous
            saved=previous.cpu().numpy().tobytes()
            for invalid in (float('nan'),float('inf'),float('-inf'),65505.,-65505.):
                bad=motion.clone();bad[0,0,0]=invalid
                try:model(rgb,bad,reset=True)
                except ValueError as error:assert 'FP16 texture range' in str(error)
                else:raise AssertionError('Invalid motion accepted')
                assert model.next_seed==seed and model._previous is previous and previous.cpu().numpy().tobytes()==saved
        means={mode:statistics.mean(row['seconds'] for row in report['runs'][mode][1:]) for mode in modes}
        report.update(passed=True,matching_warm_mean_seconds=means,paired_speedup={mode:means['baseline']/means[mode] for mode in modes},inputs_unchanged=True,caller_output_mutation_preserves_history=True,invalid_motion_preserves_state=True,reset_reproduces_frame0=True)
        report['compiled']={kind:dict(llir=artifacts.text(str(kernel.asm['llir']),'llir'),ttgir=artifacts.text(str(kernel.asm['ttgir']),'ttgir')) for kind,kernel in experiment.compiled.items()}
except Exception as error:
    report['error']=repr(error);report['traceback']=traceback.format_exc()
    raise
finally:
    assert all(sha(Path(path))==digest for path,digest in frozen.items())
    authenticate_main()
    save()
print(json.dumps({key:report[key] for key in ('passed','matching_warm_mean_seconds','paired_speedup')}),flush=True)
