"""Paired complete resident-input NR256 with continuous INT8 FFNs.

360 reset/temporal calls per variant including input checks, noise/warp/front,
body, completion and private history commit. No decode, flow estimation, upload,
residual reconstruction, presentation, JIT or game contention. New approximate
arithmetic needs separate real-motion video validation and user review.
"""
from layout_crop_validation_env_v1 import *
from pathlib import PureWindowsPath
import statistics,time,traceback
OUT=D/'results/int8-ffn-nr-v1';assert not OUT.exists()
body_result=receipt(D/'experimental/int8-ffn-body-v1/validation.json',
    'fad7c67aca46b879d185fde61280d7d8c871537713362be4f1318cb85c5a0496')
assert body_result['sequential_eight_ffn_test'] and not body_result['full_nr_call_test']
cpu=receipt(D/'experimental/int8-ffn-segment-cpu-v1/validation.json',
    '1f1bdd147c386245dad5701407f65f5ac663a31a36647c4aeae6e11c9b799130')
legacy=receipt(D/'results/nr256-native-parity-v1/validation.json',
    '5008dfcf5316aabe984e1e30d1db1c933bb3536c5e4a54a3020390c4845ff007')
native=receipt(D/'experimental/native-steady-bench-v1/validation.json',
    '70a86900dbde71d285f187cedc42a123bc7706cf27f21f0231e515b4d964659d')
legacy_hashes={(r['round'],r['mode'],r['sample']):r['output_raw_sha256'] for r in legacy['measured']['native_half']}
for p in (Path(__file__),HERE/'Run-Int8FfnNrV1.cmd',HERE/'int8_ffn_nr_stack_v1.py'):
    sources[str(p)]=sha(p)
old_path=R/'4060-real-flow-256x256-sequence-v3.json';old=js(old_path)
folder=R/'results'/PureWindowsPath(old['runDirectory']).name/'output'
input_path=folder/'frame00.png_input.rgba32f.bin';motion_path=folder/'frame00.png_motion.rg32f.bin'
sources[str(old_path)]=sha(old_path)
for p in (input_path,motion_path):
    assert sha(p)==next(f['sha256'].lower() for f in old['outputs'] if f['name']==p.name)
    sources[str(p)]=sha(p)
import numpy as np
import torch,triton
from nr256_selected_stack_v4 import Stack
from int8_ffn_nr_stack_v1 import Stack as CandidateStack
from nr_backend.execution import use_arithmetic_backend
from serial_graph_workspace_v1 import share_before_capture
from cubic_lut_constant_v1 import Constant,TABLE
import compressed_arrays_v1 as arrays
import int8_ffn_segment_oracle_v1 as oracle
import capture_body_v1 as body
assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site')
OUT.mkdir()
report=dict(scope=__doc__,sources=sources,exact_gate=gate,passed=False,phase='initialized',
    complete_migration=False,candidate_promoted=False,new_quantization=True,
    full_nr_call_test=True,nr_input=[256,256],video_test=False,
    queue_check=queue_check,measured={'selected':[],'int8_p4':[]},guards=[],
    dispatch_by_mode={},eager_graph_checks=[],sample_outputs=[],
    native_4060_historical_reference=native['cases']['256x256']['statistics'])


def save():
    (OUT/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')


raw=lambda t:t.cpu().numpy().tobytes()
digest=lambda a:hashlib.sha256(a.tobytes()).hexdigest()
stacks={};held={};save()


def run(name,reset):
    s=stacks[name];before=s.rewrite.scopes
    with s.installed(),use_arithmetic_backend('triton') as dispatch:
        torch.xpu.synchronize();start=time.perf_counter()
        value=s.model(rgb,motion,reset=reset)
        torch.xpu.synchronize();ms=(time.perf_counter()-start)*1000
    effective=dict(dispatch)
    assert effective.pop('xpu_graph_replay')==1
    for key,count in s.graph.last_entry.dispatch.items():
        if key!='backend':effective[key]=effective.get(key,0)+count
    if before>=8:assert s.rewrite.scopes==before,'unexpected body rebuild during steady replay'
    s.rewrite.verify_restored()
    return value,ms,effective


def candidate_guards():
    s=stacks['int8_p4'];scope=s.int8_vit
    private=s.model._previous;private_bytes=raw(private)
    state=(s.model.next_seed,s.graph.replays,s.warp.replays,s.rewrite.scopes)
    def check(label,call,expected):
        try:call()
        except (ValueError,RuntimeError,AssertionError) as error:
            assert expected in str(error),(label,str(error))
        else:raise AssertionError('Guard did not reject '+label)
        assert s.model._previous is private and raw(private)==private_bytes
        assert (s.model.next_seed,s.graph.replays,s.warp.replays,s.rewrite.scopes)==state
        report['guards'].append(dict(name=label,rejected=True,history_seed_graph_warp_unchanged=True))
    with s.installed(),use_arithmetic_backend('triton'):
        marks=[]
        check('progress',lambda:s.model(rgb,motion,progress=marks.append),'does not support progress')
        assert not marks
        check('dimensions',lambda:s.model(rgb[:,:255],motion[:,:255]),'requires XPU NR256')
        bad=motion.clone();bad[0,0,0]=float('nan')
        check('invalid_motion',lambda:s.model(rgb,bad),'Motion must be finite')
        row=s.model._int8_ffn_buffers[0];original=row.expand_int8
        with torch.inference_mode(False):replacement=original.clone()
        row.expand_int8=replacement
        try:
            check('registered_weight_replacement',lambda:s.model(rgb,motion),'registered constants changed')
            try:s.graph._validate()
            except RuntimeError as error:assert 'Model constants changed' in str(error)
            else:raise AssertionError('Graph signature omitted registered INT8 weights')
        finally:row.expand_int8=original
        scope.enabled=False
        try:check('route_disabled',lambda:s.model(rgb,motion),'route changed')
        finally:scope.enabled=True
        original_row=scope.packed[0]
        scope.packed[0]=(replacement,*original_row[1:])
        try:check('packed_pointer_replacement',lambda:s.model(rgb,motion),'INT8 packed constants changed')
        finally:scope.packed[0]=original_row
        s.graph._validate();s.call_guard.validate()


try:
    torch.set_num_threads(2)
    pixels=np.fromfile(input_path,'<f4').reshape(256,256,4)[...,:3].copy()
    flow=np.fromfile(motion_path,'<f4').reshape(256,256,2).copy();assert not flow.any()
    rgb,motion=torch.from_numpy(pixels).to('xpu'),torch.from_numpy(flow).to('xpu')
    scales=[arrays.load(next(c for c in cpu['cases'] if c['name']==f'vit.{i}.ffn' and c['margin']==1.25)['hidden_scale']) for i in range(8)]
    stacks['selected']=Stack(EXACT)
    stacks['int8_p4']=CandidateStack(EXACT,hidden_scales=scales,share_with=stacks['selected'])
    s=stacks['int8_p4']
    constants=[hashlib.sha256(raw(t)).hexdigest() for row in s.int8_vit.packed for t in row]
    report['shared_constants']=s.shared
    report['shared_transient_pool']=share_before_capture([s.graph for s in stacks.values()])
    report['runtime']=dict(torch=torch.__version__,triton=triton.__version__,device=torch.xpu.get_device_name())
    sources.update(s.rewrite.fork.sources)
    with torch.inference_mode():
        first={}
        for name in ('int8_p4','selected'):
            value,_,_=run(name,True);first[name]=value.cpu().numpy().copy()
            held[name]=(value,raw(value));run(name,False)
            assert stacks[name].rewrite.scopes==6 and len(stacks[name].graph.entries)==2
            print('Prewarmed '+name,flush=True)
        assert first['selected'].tobytes()==arrays.load(legacy['reset_output']).tobytes()
        report['reset_outputs']={name:arrays.save(value) for name,value in first.items()}
        report['capture_builds']={name:list(s.rewrite.builds) for name,s in stacks.items()}
        # Recompute both captured bodies directly from their actual static inputs.
        # History remains untouched; this checks the complete graph composition.
        for name,s in stacks.items():
            previous=s.model._previous;seed=s.model.next_seed;old_bytes=raw(previous)
            for number,entry in enumerate(s.graph.entries.values()):
                with s.installed(),body.installed(),use_arithmetic_backend('triton'):
                    value=body.forward_front(s.model,**entry.inputs,sigmoid=s.model.sigmoid,
                        blend_scale=s.model.blend_scale,return_float32=False)
                    assert raw(value)==raw(entry.output),(name,number,'graph/eager body mismatch')
                report['eager_graph_checks'].append(dict(name=name,entry=number,byte_equal=True))
            assert s.model._previous is previous and raw(previous)==old_bytes and s.model.next_seed==seed
            assert s.rewrite.scopes==8
        # All packed constants and live IO must stay outside the shared transient pool.
        segments=torch.xpu.memory_snapshot(stacks['selected'].graph.capture_pool)
        persistent=[rgb,motion,*[t for row in stacks['int8_p4'].int8_vit.packed for t in row]]
        for s in stacks.values():
            persistent.extend(t for entry in s.graph.entries.values() for t in (entry.output,*[v for v in entry.inputs.values() if v is not None]))
        assert not any(seg['address']<=t.data_ptr()<seg['address']+seg['total_size'] for seg in segments for t in persistent)
        report['persistent_buffers_outside_pool']=True
        candidate_hashes={};report.update(phase='timing');save()
        print('sampling_started: complete NR256 reset/temporal calls',flush=True)
        for repeat in range(3):
            for position in range(2):
                mode=('reset','temporal')[(position+repeat)%2];reset=mode=='reset'
                for name in stacks:
                    run(name,True)
                    for _ in range(20):run(name,reset)
                for sample in range(60):
                    order=list(stacks)
                    if (repeat+sample)%2:order.reverse()
                    values={};dispatches={}
                    for name in order:
                        value,ms,effective=run(name,reset);s=stacks[name];a=value.cpu().numpy().copy()
                        assert a.shape==(256,256,3) and a.dtype==np.dtype('f2') and np.isfinite(a).all()
                        assert raw(s.model._previous)==a.tobytes()
                        assert s.model.next_seed==(1 if reset else 22+sample)
                        hashed=digest(a)
                        if name=='selected':assert hashed==legacy_hashes[(repeat,mode,sample)]
                        elif repeat==0:candidate_hashes[mode,sample]=hashed
                        else:assert hashed==candidate_hashes[mode,sample],('candidate repeatability',repeat,mode,sample)
                        report['measured'][name].append(dict(round=repeat,mode=mode,sample=sample,order=order,
                            host_ms=ms,output_raw_sha256=hashed,next_seed=s.model.next_seed))
                        values[name]=a;dispatches[name]=effective
                        value.zero_();assert raw(s.model._previous)==a.tobytes()
                    if mode not in report['dispatch_by_mode']:report['dispatch_by_mode'][mode]=dispatches
                    assert report['dispatch_by_mode'][mode]==dispatches
                    if repeat==0 and sample in (0,29,59):
                        report['sample_outputs'].append(dict(mode=mode,sample=sample,
                            outputs={n:arrays.save(a) for n,a in values.items()},
                            rgb_error=oracle.error_metrics(values['int8_p4'],values['selected'])))
                for value,data in held.values():assert raw(value)==data
                save();print(json.dumps(dict(round=repeat,mode=mode,samples=60,
                    mean_ms={n:statistics.mean(v['host_ms'] for v in report['measured'][n][-60:]) for n in stacks})),flush=True)
        for name,s in stacks.items():
            value,_,_=run(name,True)
            assert raw(value)==first[name].tobytes() and s.model.next_seed==1
            assert raw(Constant(s.model).require())==np.load(TABLE,allow_pickle=False).view('i2').tobytes()
            with s.installed():s.graph._validate()
        candidate_guards()
        s=stacks['int8_p4'];s.call_guard.verify_restored()
        assert constants==[hashlib.sha256(raw(t)).hexdigest() for row in s.int8_vit.packed for t in row]
        assert raw(rgb)==pixels.tobytes() and raw(motion)==flow.tobytes()
        for value,data in held.values():assert raw(value)==data
        report['summary']={n:{mode:dict(mean_host_ms=statistics.mean(v['host_ms'] for v in rows if v['mode']==mode),
            round_mean_host_ms=[statistics.mean(v['host_ms'] for v in rows if v['mode']==mode and v['round']==r) for r in range(3)])
            for mode in ('reset','temporal')} for n,rows in report['measured'].items()}
        report['change_percent']={mode:(report['summary']['int8_p4'][mode]['mean_host_ms']/report['summary']['selected'][mode]['mean_host_ms']-1)*100 for mode in ('reset','temporal')}
        report['builds']={n:s.rewrite.builds for n,s in stacks.items()}
        for name,s in stacks.items():
            assert s.rewrite.scopes==len(s.rewrite.builds)==8
            expected=(651,180,395,215) if name=='selected' else (643,172,395,223)
            assert all(tuple(b[k] for k in ('triton_calls','standalone_fp8','quantization_calls','elided_fp8'))==expected for b in s.rewrite.builds)
            assert len(s.graph.entries)==2 and s.graph.replays==489
            s.rewrite.verify_restored()
        report['candidate']=stacks['int8_p4'].metadata()
        report['selected_resources']=stacks['selected'].rewrite.metadata()['resources']
        for resources in (report['selected_resources'],report['candidate']['ffn_resources'],report['candidate']['capture']['resources']):
            assert resources and all(v['spills']==0 for v in resources.values())
        report.update(passed=True,phase='completed',baseline_outputs_match_frozen=360,
            candidate_outputs_repeatable=360,caller_ownership_passed=True,held_outputs_unchanged=True,
            inputs_lut_int8_constants_unchanged=True,reset_reproduces_first_frame=True,
            no_dependency_swaps_on_steady_replay=True,graphs={n:s.graph.metadata() for n,s in stacks.items()})
except BaseException:
    report.update(phase='failed',error=traceback.format_exc());raise
finally:
    try:
        for s in stacks.values():s.close()
        finalize_sources()
    except BaseException:
        report.update(passed=False,phase='failed',finalization_error=traceback.format_exc());raise
    finally:save()
print(json.dumps(dict(passed=report['passed'],summary=report.get('summary'),change_percent=report.get('change_percent'),video_test=False)),flush=True)
