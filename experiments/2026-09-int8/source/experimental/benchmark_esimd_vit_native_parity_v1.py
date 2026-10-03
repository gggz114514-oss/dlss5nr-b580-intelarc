"""Paired complete NR256: selected WindowBlocks v3 vs producer-packed ESIMD ViT.

Same resident native256 fixture and frozen reset/temporal hashes as the native
4060 comparison. Warmup additionally saves and compares all seven boundaries
for each of eight ViT blocks, including the packed final output. Neither native
matrix-only timing nor sums of disjoint diagnostics are whole-model evidence.
"""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from contextlib import ExitStack,contextmanager
from pathlib import Path,PureWindowsPath
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580';R=EXACT/'reference'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'results/esimd-vit-native-parity-v1';assert not OUT.exists()
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
for p in (Path(__file__),HERE/'Run-EsimdVitNativeParityV1.cmd',provision_path,legacy_path,paired_path,native_path,old_path,input_path,motion_path,reference_path):sources[str(p)]=sha(p)

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
for p in (selected_path,esimd_path,HERE/'esimd_vit_chain_v1.py',HERE/'esimd_vit_graph_v1.py',HERE/'nr256_selected_stack_v2.py'):
    sources[str(p)]=sha(p)
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:q=json.load(response)
assert not q['queue_running'] and not q['queue_pending']
import numpy as np
import torch
import triton
assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site') and triton.__version__.startswith('3.8.0')
from nr256_selected_stack_v2 import Stack
from esimd_vit_graph_v1 import EsimdVitGraph
from nr_backend.execution import use_arithmetic_backend
from cubic_lut_constant_v1 import Constant,TABLE
from serial_graph_workspace_v1 import share_before_capture
import compressed_arrays_v1 as arrays
OUT.mkdir();report=dict(scope=__doc__,sources=sources,exact_gate=gate,passed=False,complete_migration=False,candidate_promoted=False,
    measured={'selected':[],'esimd':[]},boundaries=[],nr_input=[256,256],native_4060_reference=native['cases']['256x256']['statistics'],
    timing_scope='Complete resident-input NR256 host call and completion; excludes decode/flow/upload/display/JIT/residual scaling/game contention')
save=lambda:(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
raw=lambda t:t.cpu().numpy().tobytes()
digest=lambda a:hashlib.sha256(a.tobytes()).hexdigest()
stacks={};held={};reference_boundaries={};candidate_boundaries=set()

def run(name,reset):
    stack=stacks[name]
    with stack.installed(),use_arithmetic_backend('triton') as dispatch:
        torch.xpu.synchronize();start=time.perf_counter();value=stack.model(rgb,motion,reset=reset)
        torch.xpu.synchronize();elapsed=(time.perf_counter()-start)*1000
    effective=dict(dispatch);assert effective.pop('xpu_graph_replay')==1
    for key,count in stack.graph.last_entry.dispatch.items():
        if key!='backend':effective[key]=effective.get(key,0)+count
    return value,elapsed,effective

try:
    torch.set_num_threads(2)
    pixels=np.fromfile(input_path,'<f4').reshape(256,256,4)[...,:3].copy()
    flow=np.fromfile(motion_path,'<f4').reshape(256,256,2).copy();assert not flow.any()
    rgb=torch.from_numpy(pixels).to('xpu');motion=torch.from_numpy(flow).to('xpu')
    stacks['selected']=Stack(EXACT)
    stacks['esimd']=Stack(EXACT,rewrite_type=EsimdVitGraph,share_with=stacks['selected'])
    report['shared_constants']=stacks['esimd'].shared
    report['shared_transient_pool']=share_before_capture([s.graph for s in stacks.values()])
    report['resources']=stacks['esimd'].rewrite.chain.engine.resources
    report['runtime']=dict(torch=torch.__version__,triton=triton.__version__,device=torch.xpu.get_device_name())
    selected_adapter=stacks['selected'].components[-1];original_apply=selected_adapter.apply
    selected_indices={id(b):i for i,b in enumerate(stacks['selected'].model.vit)}

    def reference_probe(module,features):
        out=original_apply(module,features);index=selected_indices[id(module)]
        if index not in reference_boundaries:
            reference_boundaries[index]=dict(input=arrays.save(features.cpu().numpy()),outputs=[arrays.save(t.cpu().numpy()) for t in out])
        return out

    def candidate_probe(index,features,outputs,packed):
        if index in candidate_boundaries:return
        ref=reference_boundaries[index]
        row=dict(block=index,input=ref['input'],references=ref['outputs'],outputs=[],passed=False)
        report['boundaries'].append(row)
        assert raw(features)==arrays.load(ref['input']).tobytes(),('ViT input',index)
        for j,(t,meta) in enumerate(zip(outputs,ref['outputs'])):
            actual=t.cpu().numpy();row['outputs'].append(arrays.save(actual))
            assert actual.tobytes()==arrays.load(meta).tobytes(),('ViT boundary',index,j)
        p=packed.cpu().numpy();logical=p.transpose(0,2,1,3).reshape(64,1024)
        row['packed']=arrays.save(p)
        assert logical.tobytes()==raw(outputs[-1]),('Producer packed output',index)
        row['passed']=True;candidate_boundaries.add(index);save()

    selected_adapter.apply=reference_probe;stacks['esimd'].rewrite.chain.probe=candidate_probe
    with torch.inference_mode():
        first={}
        for name in stacks:
            value,_,_=run(name,True);first[name]=value.cpu().numpy().copy()
            held[name]=(value,raw(value));run(name,False)
            print('Prewarmed '+name,flush=True)
        assert len(reference_boundaries)==len(candidate_boundaries)==8
        assert first['selected'].tobytes()==first['esimd'].tobytes()==arrays.load(legacy['reset_output']).tobytes()
        report['reset_output']=arrays.save(first['esimd'])
        selected_adapter.apply=original_apply;stacks['esimd'].rewrite.chain.probe=None
        for r in range(3):
            for position in range(2):
                mode=('reset','temporal')[(position+r)%2];reset=mode=='reset'
                for name in stacks:
                    run(name,True)
                    for _ in range(20):warm,_,_=run(name,reset)
                    if name=='selected':warm_bytes=raw(warm)
                    else:assert raw(warm)==warm_bytes
                for i in range(60):
                    order=list(stacks);shift=(r+i)%2;order=order[shift:]+order[:shift];values={};dispatches={}
                    for name in order:
                        value,ms,dispatch=run(name,reset);a=value.cpu().numpy()
                        assert a.shape==(256,256,3) and a.dtype==np.dtype('f2') and np.isfinite(a).all()
                        assert raw(stacks[name].model._previous)==a.tobytes() and stacks[name].model.next_seed==(1 if reset else 22+i)
                        assert digest(a)==legacy_hashes[(r,mode,i)],('Frozen output',name,r,mode,i)
                        values[name]=a.copy();dispatches[name]=dispatch
                        report['measured'][name].append(dict(round=r,mode=mode,sample=i,host_ms=ms,output_raw_sha256=digest(a),seed=stacks[name].model.next_seed,private_equal=True,caller_independent=True))
                        value.zero_();assert raw(stacks[name].model._previous)==a.tobytes()
                    assert values['selected'].tobytes()==values['esimd'].tobytes()
                    assert dispatches['selected']==dispatches['esimd'],(dispatches['selected'],dispatches['esimd'])
                for v,data in held.values():assert raw(v)==data
                save();print(json.dumps(dict(round=r,mode=mode,all60outputs_match=True,host_mean_ms={n:statistics.mean(x['host_ms'] for x in report['measured'][n][-60:]) for n in stacks})),flush=True)
        for name,stack in stacks.items():
            value,_,_=run(name,True);assert raw(value)==first[name].tobytes() and stack.model.next_seed==1
            assert raw(Constant(stack.model).require())==np.load(TABLE,allow_pickle=False).view('i2').tobytes()
            assert len(stack.rewrite.builds)==6 and len(stack.window_blocks.calls)==216
        assert raw(rgb)==pixels.tobytes() and raw(motion)==flow.tobytes()
        report['summary']={name:{mode:dict(mean_host_ms=statistics.mean(s['host_ms'] for s in rows if s['mode']==mode),median_host_ms=statistics.median(s['host_ms'] for s in rows if s['mode']==mode),round_mean_host_ms=[statistics.mean(s['host_ms'] for s in rows if s['mode']==mode and s['round']==r) for r in range(3)]) for mode in ('reset','temporal')} for name,rows in report['measured'].items()}
        report['builds']={name:s.rewrite.builds for name,s in stacks.items()}
        assert all(b['triton_calls']==683 and b['standalone_fp8']==196 and b['elided_fp8']==231 for b in report['builds']['selected'])
        assert all(b['native_dense_calls']==8 for b in report['builds']['esimd'])
        report.update(passed=True,complete_outputs_compared=720,all_candidate_outputs_byte_equal=True,all_outputs_match_frozen_372=True,
            inputs_unchanged=True,reset_reproduces_first=True,held_outputs_unchanged=True,lut_bytes_unchanged=True,
            chain_calls=stacks['esimd'].rewrite.chain.calls,preflight=stacks['esimd'].rewrite.chain.selection,
            graphs={n:s.graph.metadata() for n,s in stacks.items()},full_boundaries_compared=56,packed_outputs_compared=8)
except BaseException as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    for stack in stacks.values():stack.close()
    try:assert all(sha(p)==h for p,h in sources.items());authenticate_main()
    except BaseException as error:report.update(passed=False,finalization_error=repr(error));raise
    finally:save()
print(json.dumps(dict(passed=report['passed'],summary=report.get('summary'))),flush=True)

