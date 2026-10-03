"""Pair selected WindowBlocks v3 with the specialized producer-packed ESIMD ViT.

Three rotated rounds of the frozen thirteen-frame 1080p residual sequence.
Timing includes GPU resize/composition, complete NR256 and completion. Decode,
flow estimation, upload, verification, presentation, JIT and game contention
are outside. All outputs and history are compared, with error/ownership guards.
"""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from contextlib import ExitStack,contextmanager
from pathlib import Path,PureWindowsPath
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580';R=EXACT/'reference'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'results/esimd-vit-residual256-v2';assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-c32-triton38-v1')
TOOLCHAIN=D/'toolchains/triton-xpu-3.8.0-git1e2d42a0'
sys.path[:0]=[str(TOOLCHAIN/'site'),str(R),str(ROOT/'backend')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main()
provision_path=TOOLCHAIN/'provision-v1.json'
assert sha(provision_path)=='e94757afcb5962cacbe208cbb808b3516de210059335f1ee7ab7b40d6bfe2c07'
assert all(sha(TOOLCHAIN/'site'/p)==h for p,h in js(provision_path)['files'].items())
legacy_path=D/'results/nr256-native-parity-v1/validation.json'
assert sha(legacy_path)=='5008dfcf5316aabe984e1e30d1db1c933bb3536c5e4a54a3020390c4845ff007'
legacy=js(legacy_path);assert legacy['passed'] and js(legacy_path.parent.with_suffix('.log.lease.json'))['returncode']==0
assert all(sha(p)==h for p,h in legacy['sources'].items())
legacy_hashes={(r['round'],r['mode'],r['sample']):r['output_raw_sha256'] for r in legacy['measured']['native_half']}
paired_path=D/'results/fp8-graph-residual256-v1/validation.json';paired=js(paired_path)
assert sha(paired_path)=='7a45f2bdf93e9cfc45778df3703cf37172c1cb84667459af178b877c52c28af7'
native_path=D/'experimental/native-steady-bench-v1/validation.json';native=js(native_path)
assert sha(native_path)=='70a86900dbde71d285f187cedc42a123bc7706cf27f21f0231e515b4d964659d'
assert paired['passed'] and js(paired_path.parent.with_suffix('.log.lease.json'))['returncode']==0 and native['passed']
assert all(sha(p)==h for p,h in paired['sources'].items()) and all(sha(p)==h for p,h in native['sources'].items())
old_path=R/'4060-real-flow-256x256-sequence-v3.json';old=js(old_path)
folder=R/'results'/PureWindowsPath(old['runDirectory']).name/'output'
input_path=folder/'frame00.png_input.rgba32f.bin';motion_path=folder/'frame00.png_motion.rg32f.bin'
reference_path=folder/'frame00.png_output.rgba32f.bin'
for p in (input_path,motion_path,reference_path):assert sha(p)==next(f['sha256'].lower() for f in old['outputs'] if f['name']==p.name)
sources={**paired['sources'],**native['sources'],**legacy['sources']}
for p in (Path(__file__),HERE/'Run-EsimdVitResidual256V2.cmd',provision_path,legacy_path,paired_path,native_path,old_path,input_path,motion_path,reference_path):sources[str(p)]=sha(p)

block_path=D/'experimental/window-blocks-v3/validation.json'
assert sha(block_path)=='e0b7a5f9bf996bc9e825d439761f1cd19f8ed6ff4fa768d2dd28839a0f63f8af'
block=js(block_path);assert block['passed'] and js(block_path.parent.with_suffix('.log.lease.json'))['returncode']==0
block_audit=block_path.with_name('saved-audit-v1.json')
assert sha(block_audit)=='ea46f04dea7d0760d622a3449aae517ea88d975f1eae3dcf6eab7b282a79dec3' and js(block_audit)['passed']
assert all(sha(p)==h for p,h in block['sources'].items())
sources.update(block['sources'])
for p in (block_path,block_audit):sources[str(p)]=sha(p)

selected_path=D/'results/window-blocks-native-parity-v3/validation.json'
assert sha(selected_path)=='6986ddfcaf1adbfe054ac415c69515327682fe40f3ba780d8e92e2786407a63d'
selected=js(selected_path);assert selected['passed'] and js(selected_path.parent.with_suffix('.log.lease.json'))['returncode']==0
assert all(sha(p)==h for p,h in selected['sources'].items());sources.update(selected['sources'])
esimd_path=D/'experimental/esimd-dense-v3/validation.json'
assert sha(esimd_path)=='8b8214b33aea961cab0655913748b9a42b655d4c38da6cbc4b42ba38474cd6eb'
esimd=js(esimd_path);assert esimd['passed'] and all(sha(p)==h for p,h in esimd['sources'].items());sources.update(esimd['sources'])
for p in (HERE/'esimd_vit_chain_v2.py',HERE/'esimd_vit_graph_v2.py',HERE/'esimd_vit_expand_v1.cpp',HERE/'Build-EsimdVitExpandV1.cmd',D/'experimental/esimd-vit-expand-build-v1/esimd_vit_expand_v1.dll',D/'experimental/esimd-vit-expand-build-v1/build.log',selected_path,esimd_path,HERE/'esimd_vit_chain_v1.py',HERE/'esimd_vit_graph_v1.py',HERE/'nr256_selected_stack_v2.py'):
    sources[str(p)]=sha(p)
candidate_path=D/'results/esimd-vit-native-parity-v2/validation.json'
assert sha(candidate_path)=='42dff3710e3db9b5f0312c63bd234acfe49c6aeafea1a2690040cb00db9092f3'
candidate=js(candidate_path);assert candidate['passed'] and js(candidate_path.parent.with_suffix('.log.lease.json'))['returncode']==0
assert all(sha(p)==h for p,h in candidate['sources'].items());sources.update(candidate['sources'])
prior_path=D/'results/residual-scale-fp16_xmx-256-v1/validation.json'
assert sha(prior_path)=='056251b4e2a839fcc44845d9646d2cd8ae1e9ed7f99fd1858dc7e3d51e3e50f9'
prior=js(prior_path);assert prior['passed']
inputs=R/'inputs/flow-full-1920x1080-v3';manifest_path=inputs/'manifest.json';manifest=js(manifest_path)
for p in (candidate_path,prior_path,manifest_path,HERE/'residual_scale_v1.py',HERE/'cubic_lut_graph_guards_v2.py'):
    sources[str(p)]=sha(p)
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:q=json.load(response)
assert not q['queue_running'] and not q['queue_pending']
import numpy as np
from PIL import Image
import torch
import triton
assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site')
from nr256_selected_stack_v2 import Stack
from esimd_vit_graph_v2 import EsimdVitGraph
from nr_backend.execution import use_arithmetic_backend
from residual_scale_v1 import ResidualScale
from serial_graph_workspace_v1 import share_before_capture
from cubic_lut_constant_v1 import Constant,TABLE
from cubic_lut_graph_guards_v2 import verify as verify_lut_guards
import compressed_arrays_v1 as arrays
OUT.mkdir();report=dict(scope=__doc__,passed=False,complete_migration=False,candidate_promoted=False,sources=sources,exact_gate=gate,
    runs={'selected':[],'esimd':[]},guards=[],progress_fallback=[])
save=lambda:(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
raw=lambda t:t.cpu().numpy().tobytes()
stacks={};held={};scaler=None

def tensors(index):
    spec=manifest['frames'][index]
    pixels=np.asarray(Image.open(inputs/spec['file']).convert('RGB'),dtype='f4')/255
    flow=np.fromfile(inputs/spec['motion_file'],'<f4').reshape(1080,1920,2)
    return pixels,flow,torch.from_numpy(pixels).to('xpu'),torch.from_numpy(flow).to('xpu')

def run(name,rgb,motion,reset):
    s=stacks[name]
    with s.installed(),use_arithmetic_backend('triton') as dispatch:
        torch.xpu.synchronize();start=time.perf_counter()
        low_rgb,low_motion=scaler.prepare(rgb,motion);low=s.model(low_rgb,low_motion,reset=reset)
        output=scaler.composite(rgb,low_rgb,low);torch.xpu.synchronize();ms=(time.perf_counter()-start)*1000
    effective=dict(dispatch);assert effective.pop('xpu_graph_replay')==1
    for key,count in s.graph.last_entry.dispatch.items():
        if key!='backend':effective[key]=effective.get(key,0)+count
    return output,low,ms,effective

try:
    torch.set_num_threads(2);scaler=ResidualScale(256)
    expected=[(arrays.load(row['output']),arrays.load(row['low_nr'])) for row in prior['runs'][:13]]
    assert len(manifest['frames'])==len(expected)==13
    stacks['selected']=Stack(EXACT)
    stacks['esimd']=Stack(EXACT,rewrite_type=EsimdVitGraph,share_with=stacks['selected'])
    report['shared_constants']=stacks['esimd'].shared
    report['shared_transient_pool']=share_before_capture([s.graph for s in stacks.values()])
    with torch.inference_mode():
        engine=stacks['esimd'].rewrite.chain.engine
        a=torch.zeros((8,64,8,16),device='xpu',dtype=torch.float16)
        w=next(iter(stacks['esimd'].rewrite.chain.weights.values()))
        initial=torch.ones((64,4096),device='xpu',dtype=torch.float16);out=torch.full_like(initial,7)
        before=raw(out)
        for label,aa,ini,oo,m in (('initial_forbidden',a,initial,out,64),('shape_forbidden',a[:4],None,out[:32],32)):
            try:engine.into(aa,w,ini,oo,m=m,k=1024,n=4096,packed=True)
            except RuntimeError as error:assert 'Invalid dense contract' in str(error)
            else:raise AssertionError('Native specialized contract accepted unsupported call')
            assert raw(out)==before
            report['guards'].append(dict(case=label,rejected_before_launch=True,output_unchanged=True))
        for name,s in stacks.items():
            for i in (0,1):
                _,_,rgb,motion=tensors(i);output,low,_,_=run(name,rgb,motion,i==0)
                assert raw(output)==expected[i][0].tobytes() and raw(low)==expected[i][1].tobytes()
            s.model.reset();print('Prewarmed '+name,flush=True)
        packed_before={k:hashlib.sha256(raw(v)).hexdigest() for k,v in stacks['esimd'].rewrite.chain.weights.items()}
        for r in range(3):
            for i,spec in enumerate(manifest['frames']):
                pixels,flow,rgb,motion=tensors(i);order=list(stacks);shift=(r+i)%2;order=order[shift:]+order[:shift]
                dispatches={}
                for name in order:
                    output,low,ms,effective=run(name,rgb,motion,spec['reset']);s=stacks[name]
                    assert raw(output)==expected[i][0].tobytes() and raw(low)==expected[i][1].tobytes()
                    assert raw(s.model._previous)==raw(low) and s.model.next_seed==(1 if spec['reset'] else i+1)
                    assert np.isfinite(output.cpu().numpy()).all()
                    row=dict(round=r,frame=i,reset=spec['reset'],order=order,host_ms=ms,output=prior['runs'][i]['output'],low_nr=prior['runs'][i]['low_nr'],byte_equal=True,
                             private_byte_equal_low_nr=True,next_seed=s.model.next_seed,effective_dispatch=effective)
                    report['runs'][name].append(row);dispatches[name]=effective
                    if name not in held:held[name]=(output,raw(output))
                    else:
                        private=raw(s.model._previous);output.zero_();low.zero_();assert raw(s.model._previous)==private
                assert dispatches['selected']==dispatches['esimd']
                for value,previous in held.values():assert raw(value)==previous
                assert raw(rgb)==pixels.tobytes() and raw(motion)==flow.tobytes()
                save();print(json.dumps(dict(round=r,frame=i,host_ms={n:report['runs'][n][-1]['host_ms'] for n in stacks},all_equal=True)),flush=True)
        for name,s in stacks.items():
            _,_,rgb0,flow0=tensors(0);run(name,rgb0,flow0,True)
            _,_,rgb,motion=tensors(1);low_rgb,low_motion=scaler.prepare(rgb,motion)
            before=s.graph.replays;marks=[];chain_before=getattr(s.rewrite,'chain',None)
            calls=0 if chain_before is None else chain_before.calls
            with s.installed(),use_arithmetic_backend('triton'):
                low=s.model(low_rgb,low_motion,reset=False,progress=marks.append)
                output=scaler.composite(rgb,low_rgb,low)
            assert raw(output)==expected[1][0].tobytes() and s.model.next_seed==2 and s.graph.replays==before
            assert marks==['pre','encoder C32','encoder C64','encoder C128','encoder C256','encoder C512','ViT','decoder C512','decoder C256','decoder C128','decoder C64','decoder C32','RGB']
            assert chain_before is None or chain_before.calls==calls
            report['progress_fallback'].append(dict(name=name,marks=marks,byte_equal=True,native_chain_bypassed=True))
            private=s.model._previous;private_bytes=raw(private);state=(s.model.next_seed,s.graph.replays,s.warp.replays)
            with s.installed(),use_arithmetic_backend('triton'):
                bad=low_motion.clone();bad[0,0,0]=float('nan')
                try:s.model(low_rgb,bad,reset=False)
                except ValueError as error:assert 'Motion must be finite' in str(error)
                else:raise AssertionError('Invalid motion accepted')
                original=s.model.reciprocal.values
                with torch.inference_mode(False):replacement=original.clone()
                s.model.reciprocal.values=replacement
                try:
                    try:s.model(low_rgb,low_motion,reset=False)
                    except RuntimeError as error:assert 'reciprocal table changed' in str(error)
                    else:raise AssertionError('Modified table accepted')
                finally:s.model.reciprocal.values=original
            assert s.model._previous is private and raw(private)==private_bytes
            assert (s.model.next_seed,s.graph.replays,s.warp.replays)==state
            report['guards'].append(dict(name=name,invalid_motion_and_table_rejected=True,history_seed_and_replays_unchanged=True))
            assert raw(Constant(s.model).require())==np.load(TABLE,allow_pickle=False).view('i2').tobytes()
        s=stacks['esimd']
        with s.installed(),use_arithmetic_backend('triton'):
            report['lut_graph_guards']=verify_lut_guards(s.model,s.graph,lambda:s.model(low_rgb,low_motion,reset=False))
        assert packed_before=={k:hashlib.sha256(raw(v)).hexdigest() for k,v in s.rewrite.chain.weights.items()}
        report['builds']={n:s.rewrite.builds for n,s in stacks.items()}
        assert all(len(s.rewrite.builds)==6 for s in stacks.values())
        assert all(b['native_dense_calls']==8 for b in report['builds']['esimd'])
        report['summary']={n:{mode:dict(mean_host_ms=statistics.mean(row['host_ms'] for row in rows if mode=='all' or not row['reset']),
             round_mean_host_ms=[statistics.mean(row['host_ms'] for row in rows if row['round']==r and (mode=='all' or not row['reset'])) for r in range(3)]) for mode in ('all','temporal')} for n,rows in report['runs'].items()}
        report.update(passed=True,full_outputs_verified=78,packed_weights_unchanged=True,held_outputs_survive_replay=True,caller_ownership_guards_passed=True,
            inputs_unchanged=True,lut_bytes_unchanged=True,graphs={n:s.graph.metadata() for n,s in stacks.items()})
except BaseException as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    for s in stacks.values():s.close()
    try:assert all(sha(p)==h for p,h in sources.items());authenticate_main()
    except BaseException as error:report.update(passed=False,finalization_error=repr(error));raise
    finally:save()
print(json.dumps(dict(passed=report['passed'],summary=report.get('summary'))),flush=True)
