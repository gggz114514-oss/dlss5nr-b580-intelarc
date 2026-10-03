"""Projection-store fusion screen on the accepted repaired INT8 route.

Compare repaired baseline, branched-store fusion, and branched+C512 fusion.
All must reproduce frozen car and all243 face outputs exactly. Static-body
timings and resident pipeline timings are separate; neither is whole-tool FPS.
v2 fixes only construction-time sharing topology with a fresh base-only donor.
No recalibration, new arithmetic approximation, new video or default promotion.
"""
from layout_crop_validation_env_v1 import *
import statistics,subprocess,time,traceback
OUT=D/'results/quantized-projection-store-v2';assert not OUT.exists()
repair=receipt(D/'results/int8-ffn-range-repair-v1/validation.json','bd53a2e2b818886e79fed05e836646bdd89814a434aa932ae5bdd4f7944ce2aa')
full=receipt(D/'results/long-precision-480-v1/validation.json','858b4dd646c3be61263bc3fac6fabe400912686af672861638619c63be3be8a3')
paired=receipt(D/'results/layout-crop-residual256-v1/validation.json','b0d5b7e984ca530d36459b3c20651b3989f37719aae57bcee3bb6eb64e265fe8')
review_path=D/'results/int8-ffn-range-repair-v1/user-review-v1.json'
assert sha(review_path)=='8629fdc089e24eeb9b968e65d6a85d80f89268437a488a17d0c11497360ce214'
assert js(review_path)['status']=='accepted_face_color_for_range_repair_v1'
import numpy as np
import torch,triton
from PIL import Image
from nr256_selected_stack_v4 import Stack as DonorStack
from int8_ffn_calibrated_stack_v1 import Stack as BaseStack
from quantized_projection_stack_v1 import Stack as CandidateStack
from serial_graph_workspace_v1 import share_before_capture
from nr_backend.execution import use_arithmetic_backend
from residual_scale_v1 import ResidualScale
from face480_residual_scale_v1 import Face480Scale
from nr_review_video_finalize_v1 import read_exact
import capture_body_v1 as body
import compressed_arrays_v1 as arrays

assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site')
NAMES=('repaired','branched','combined')
car_inputs=R/'inputs/flow-full-1920x1080-v3';manifest_path=car_inputs/'manifest.json';manifest=js(manifest_path)
assert sha(manifest_path)==paired['sources'][str(manifest_path)]
expected={r['frame']:r for r in repair['paired']['runs']['repaired_int8'] if r['round']==0}
assert len(expected)==len(manifest['frames'])==13
source=Path(full['source']);ffmpeg=Path(next(p for p in full['sources'] if Path(p).name=='ffmpeg.exe'))
for p in (Path(__file__),HERE/'quantized_projection_store_v1.py',HERE/'quantized_projection_stack_v1.py',
          HERE/'audit_quantized_projection_v2.py',HERE/'Run-QuantizedProjectionV2.cmd',review_path,manifest_path,source,ffmpeg):sources[str(p)]=sha(p)
OUT.mkdir();stacks={};held={};decoder=None;stderr=None;donor=None
# Preserve and authenticate the failed setup; it was not a kernel/performance result.
FAILURE_PINS={'D:\\Codex-NR-Experiments\\nr-b580\\reference\\results\\quantized-projection-store-v1\\validation.json': 'dfc7135581ab18de960c4c2b2d762ca08fbebabb36c872b7b3b6762bc2259e73', 'D:\\Codex-NR-Experiments\\nr-b580\\reference\\results\\quantized-projection-store-v1.log': '9655fb7728ced3c3d9a5afc7fbbf1034ca24fae6dea6cd9b6adbe786997ce1e1', 'D:\\Codex-NR-Experiments\\nr-b580\\reference\\results\\quantized-projection-store-v1.log.lease.json': '9bc6045cf96f97e2a3ad37b82baae8c7ac779f79c291ac8faf83d8d9ed29909c', 'E:\\ComfyUI-aki-v3-IntelArc_20260722\\nr-b580-int8\\experimental\\benchmark_quantized_projection_v1.py': '35a4855b643b8bbcec3a2c231222f6d84a61cfa4e70de0b1e0d9f5228dd5fe52', 'E:\\ComfyUI-aki-v3-IntelArc_20260722\\nr-b580-int8\\experimental\\audit_quantized_projection_v1.py': '33c6027c286fa92f5da683f04ab4b68eb76ad4d8abe0c859c1d851ccc6ea1ffa', 'E:\\ComfyUI-aki-v3-IntelArc_20260722\\nr-b580-int8\\experimental\\Run-QuantizedProjectionV1.cmd': '3b4139076a98f5f4b7f9240c2c9701d09b00622703cb2ed1858611c81612acec'}
for p,h in FAILURE_PINS.items():
    assert sha(p)==h,p
    sources[p]=h
report=dict(scope=__doc__,passed=False,phase='initializing',sources=sources,exact_gate=gate,queue_check=queue_check,
    candidate_promoted=False,new_quantization=False,recalibration=False,new_video=False,default_unchanged=True,
    previous_failure=dict(report='D:\\Codex-NR-Experiments\\nr-b580\\reference\\results\\quantized-projection-store-v1\\validation.json',sha256='dfc7135581ab18de960c4c2b2d762ca08fbebabb36c872b7b3b6762bc2259e73',setup_only=True,returncode=1),
    accepted_reference=dict(path=str(review_path),sha256=sha(review_path)),
    speed_scope='Resident1080p prepare,NR256,history,residual,host completion; excludes decode,flow,upload,encode,JIT',
    body_scope='Actual captured temporal body graph at car sample1; no front/warp/history commit/scaling; separate nonadditive diagnostic',
    body={n:dict(samples_ms=[]) for n in NAMES},paired=dict(runs={n:[] for n in NAMES}),frames=[])
raw=lambda t:t.cpu().numpy().tobytes()
digest=lambda a:hashlib.sha256(a.tobytes()).hexdigest()


def save():
    p=OUT/'progress.tmp';p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');p.replace(OUT/'validation.json')


def car_pair(i):
    spec=manifest['frames'][i];p,f=car_inputs/spec['file'],car_inputs/spec['motion_file']
    for path in (p,f):assert sha(path)==paired['sources'][str(path)];sources[str(path)]=sha(path)
    a=np.asarray(Image.open(p).convert('RGB')).astype('f4')/255;m=np.fromfile(f,'<f4').reshape(1080,1920,2)
    return a,m,torch.from_numpy(a).to('xpu'),torch.from_numpy(m).to('xpu')


def run(name,scaler,rgb,motion,reset):
    s=stacks[name];before=s.rewrite.scopes;entries=len(s.graph.entries)
    with s.installed(),use_arithmetic_backend('triton') as dispatch:
        torch.xpu.synchronize();start=time.perf_counter()
        canvas,flow=scaler.prepare(rgb,motion);low=s.model(canvas,flow,reset=reset)
        out=scaler.composite(rgb,canvas,low);torch.xpu.synchronize();ms=(time.perf_counter()-start)*1000
    assert dispatch.get('xpu_graph_replay')==1
    assert len(s.graph.entries)>entries or s.rewrite.scopes==before
    s.rewrite.verify_restored()
    return out,low,ms


save()
try:
    torch.set_num_threads(2)
    scales=[arrays.load(row['hidden_scale']) for row in repair['calibration']['layers']]
    # Each INT8 constructor shares its base buffers before registering its own
    # 40 packed buffers. The donor must therefore have only the base topology.
    donor=DonorStack(EXACT)
    donor_buffers=dict(donor.model.named_buffers(remove_duplicate=False))
    assert not any(n.startswith('_int8_ffn_buffers.') for n in donor_buffers)
    stacks['repaired']=BaseStack(EXACT,hidden_scales=scales,share_with=donor)
    stacks['branched']=CandidateStack(EXACT,hidden_scales=scales,share_with=donor,c512=False)
    stacks['combined']=CandidateStack(EXACT,hidden_scales=scales,share_with=donor,c512=True)
    assert not donor.graph.entries and donor.graph.replays==0
    assert donor.model._previous is None and donor.model.next_seed==0
    pointers=set();sharing={}
    for name,s in stacks.items():
        own=dict(s.model.named_buffers(remove_duplicate=False))
        extra={n for n in own if n.startswith('_int8_ffn_buffers.')}
        assert len(extra)==40 and set(own)==set(donor_buffers)|extra
        assert all(own[n] is t for n,t in donor_buffers.items())
        packed={t.data_ptr() for row in s.int8_vit.packed for t in row[:5]}
        assert len(packed)==40 and pointers.isdisjoint(packed)
        pointers.update(packed)
        sharing[name]=dict(base_buffers_shared=len(donor_buffers),owned_packed_buffers=40,
            base_buffers_identical_objects=True,packed_buffers_disjoint=True,receipt=s.shared)
    report['sharing_setup']=dict(donor_never_executed=True,unchanged_topology_guard=True,routes=sharing)
    save();print('Sharing setup validated: base-only donor and three disjoint INT8 registries',flush=True)
    report['shared_transient_pool']=share_before_capture([s.graph for s in stacks.values()])
    report['runtime']=dict(torch=torch.__version__,triton=triton.__version__,device=torch.xpu.get_device_name())
    car_scaler,face_scaler=ResidualScale(256),Face480Scale()
    constants={n:[digest(t.cpu().numpy()) for row in s.int8_vit.packed for t in row] for n,s in stacks.items()}
    tables={n:[digest(t.cpu().numpy()) for row in sc.tables.values() for t in row] for n,sc in [('car',car_scaler),('face',face_scaler)]}
    for s in stacks.values():sources.update(s.rewrite.fork.sources)
    with torch.inference_mode():
        resident=[car_pair(i) for i in range(13)]
        report['phase']='warmup';save()
        for name,s in stacks.items():
            for i in (0,1):
                _,_,rgb,motion=resident[i];out,low,_=run(name,car_scaler,rgb,motion,i==0)
                assert digest(out.cpu().numpy())==expected[i]['full_raw_sha256']
                assert digest(low.cpu().numpy())==expected[i]['low_raw_sha256']
            assert s.rewrite.scopes==len(s.rewrite.builds)==6 and len(s.graph.entries)==2
            print(json.dumps(dict(warmup=name,byte_equal=True)),flush=True)
        report['capture_build_counts']={n:[{k:b[k] for k in ('triton_calls','standalone_fp8','quantization_calls','elided_fp8')} for b in s.rewrite.builds] for n,s in stacks.items()}
        for i in range(6):
            base=report['capture_build_counts']['repaired'][i];branch=report['capture_build_counts']['branched'][i];combined=report['capture_build_counts']['combined'][i]
            assert combined['standalone_fp8']<branch['standalone_fp8']<base['standalone_fp8']
            assert combined['triton_calls']<branch['triton_calls']<base['triton_calls']
            assert combined['quantization_calls']==branch['quantization_calls']==base['quantization_calls']
        assert stacks['combined'].rewrite.layout.quantized_calls==15*6
        assert stacks['combined'].rewrite.layout.pooling_calls==6
        report['fewer_physical_calls_same_logical_rounding']=True
        # Full eager/body-graph equivalence and private-state isolation for each
        # actual captured mode, before timing. Does not call public model again.
        report['phase']='body_validation';save();report['eager_graph_checks']=[]
        for name,s in stacks.items():
            previous=s.model._previous;old=raw(previous);seed=s.model.next_seed
            for entry in s.graph.entries.values():
                inputs={k:None if t is None else raw(t) for k,t in entry.inputs.items()}
                with s.installed(),body.installed(),use_arithmetic_backend('triton'):
                    value=body.forward_front(s.model,**entry.inputs,sigmoid=s.model.sigmoid,
                        blend_scale=s.model.blend_scale,return_float32=False);torch.xpu.synchronize()
                assert raw(value)==raw(entry.output)
                assert inputs=={k:None if t is None else raw(t) for k,t in entry.inputs.items()}
                report['eager_graph_checks'].append(dict(route=name,byte_equal=True,inputs_unchanged=True))
            assert s.model._previous is previous and raw(previous)==old and s.model.next_seed==seed
        report['steady_scope_counts']={n:s.rewrite.scopes for n,s in stacks.items()}
        body_inputs={n:{k:None if t is None else raw(t) for k,t in s.graph.last_entry.inputs.items()} for n,s in stacks.items()}
        histories={n:(s.model._previous,raw(s.model._previous),s.model.next_seed) for n,s in stacks.items()}
        body_expected=expected[1]['low_raw_sha256'];report['phase']='static_body_timing';save()
        for repeat in range(7):
            shift=repeat%3;order=NAMES[shift:]+NAMES[:shift]
            for name in order:
                entry=stacks[name].graph.last_entry
                torch.xpu.synchronize();start=time.perf_counter()
                for _ in range(10):entry.graph.replay()
                torch.xpu.synchronize();ms=(time.perf_counter()-start)*100
                assert digest(entry.output.cpu().numpy())==body_expected
                report['body'][name]['samples_ms'].append(ms)
        for name,s in stacks.items():
            report['body'][name]['median_ms']=statistics.median(report['body'][name]['samples_ms'])
            assert body_inputs[name]=={k:None if t is None else raw(t) for k,t in s.graph.last_entry.inputs.items()}
            prev,b,seed=histories[name];assert s.model._previous is prev and raw(prev)==b and s.model.next_seed==seed
            s.model.reset()
        report['body_inputs_and_private_history_unchanged']=True
        report['phase']='resident_paired_timing';save()
        for repeat in range(3):
            for i,(a,m,rgb,motion) in enumerate(resident):
                shift=(repeat+i)%3;order=NAMES[shift:]+NAMES[:shift]
                reset=bool(manifest['frames'][i]['reset']);seed=expected[i]['next_seed']
                for name in order:
                    s=stacks[name];out,low,ms=run(name,car_scaler,rgb,motion,reset)
                    o,b=out.cpu().numpy(),low.cpu().numpy()
                    assert digest(o)==expected[i]['full_raw_sha256'] and digest(b)==expected[i]['low_raw_sha256']
                    assert raw(s.model._previous)==b.tobytes() and s.model.next_seed==seed
                    report['paired']['runs'][name].append(dict(round=repeat,frame=i,reset=reset,order=list(order),host_ms=ms,
                        full_raw_sha256=digest(o),low_raw_sha256=digest(b),next_seed=seed,frozen_byte_equal=True,private_history_matches_low=True))
                    out.zero_();low.zero_();assert raw(s.model._previous)==b.tobytes()
                assert raw(rgb)==a.tobytes() and raw(motion)==m.tobytes()
            save();print(json.dumps(dict(paired_round=repeat,all_byte_equal=True)),flush=True)
        report['paired']['summary']={n:{mode:dict(mean_host_ms=statistics.mean(r['host_ms'] for r in rows if mode=='all' or not r['reset']),
            round_mean_host_ms=[statistics.mean(r['host_ms'] for r in rows if r['round']==k and (mode=='all' or not r['reset'])) for k in range(3)])
            for mode in ('all','temporal')} for n,rows in report['paired']['runs'].items()}
        report['paired']['passed']=True
        report['phase']='full_face_byte_validation';save()
        for s in stacks.values():s.model.reset()
        del resident
        stderr=(OUT/'face-decode.stderr.txt').open('wb')
        decoder=subprocess.Popen([str(ffmpeg),'-v','error','-threads','2','-i',str(source),'-vf',full['decode_filter'],
            '-frames:v','243','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],
            stdout=subprocess.PIPE,stderr=stderr,creationflags=0x08000000)
        first=None
        for i in range(243):
            pixels=np.frombuffer(read_exact(decoder.stdout,480*864*3),dtype='u1').reshape(480,864,3).copy()
            ref=repair['frames'][i];assert digest(pixels)==ref['input_rgb8_sha256']
            a=pixels.astype('f4')/255;m=arrays.load(ref['motion']);rgb,motion=torch.from_numpy(a).to('xpu'),torch.from_numpy(m).to('xpu')
            low_bytes=arrays.load(ref['low']).tobytes();row=dict(frame=i,reset=i==0,runs={})
            shift=i%3;order=NAMES[shift:]+NAMES[:shift]
            for name in order:
                s=stacks[name];out,low,_=run(name,face_scaler,rgb,motion,i==0)
                o,b=out.cpu().numpy(),low.cpu().numpy()
                assert digest(o)==ref['full_raw_sha256'] and b.tobytes()==low_bytes,(i,name,'accepted face output changed')
                assert raw(s.model._previous)==low_bytes and s.model.next_seed==i+1
                row['runs'][name]=dict(full_raw_sha256=digest(o),low_raw_sha256=digest(b),byte_equal=True,next_seed=i+1,private_history_matches_low=True)
                if i in (0,120):held[name,i]=(out,low,o.tobytes(),b.tobytes())
                else:out.zero_();low.zero_();assert raw(s.model._previous)==low_bytes
            for out,low,o,b in held.values():assert raw(out)==o and raw(low)==b
            assert raw(rgb)==a.tobytes() and raw(motion)==m.tobytes()
            if i==0:first=pixels.copy()
            row.update(inputs_unchanged=True,held_outputs_unchanged=True);report['frames'].append(row)
            if i%48==0 or i==242:save();print(json.dumps(dict(face_frame=i,total=243,all_byte_equal=True)),flush=True)
        assert decoder.stdout.read(1)==b'' and decoder.wait(timeout=30)==0
        rgb=torch.from_numpy(first.astype('f4')/255).to('xpu');motion=torch.from_numpy(arrays.load(repair['frames'][0]['motion'])).to('xpu')
        for name,s in stacks.items():
            out,low,_=run(name,face_scaler,rgb,motion,True)
            assert digest(out.cpu().numpy())==repair['frames'][0]['full_raw_sha256'] and raw(low)==arrays.load(repair['frames'][0]['low']).tobytes()
            assert len(s.graph.entries)==2 and s.graph.replays==285 and s.model.next_seed==1
            assert s.rewrite.scopes==report['steady_scope_counts'][name]
            assert constants[name]==[digest(t.cpu().numpy()) for row in s.int8_vit.packed for t in row]
            with s.installed():s.graph._validate()
        for n,sc in [('car',car_scaler),('face',face_scaler)]:assert tables[n]==[digest(t.cpu().numpy()) for row in sc.tables.values() for t in row]
        for out,low,o,b in held.values():assert raw(out)==o and raw(low)==b
        report['candidates']={n:s.metadata() for n,s in stacks.items() if n!='repaired'}
        for candidate in report['candidates'].values():
            resources=candidate['projection_stores']['resources'];assert resources and all(v['spills']==0 for v in resources.values())
        report['graphs']={n:s.graph.metadata() for n,s in stacks.items()}
        assert not donor.graph.entries and donor.graph.replays==0
        assert donor.model._previous is None and donor.model.next_seed==0
        assert dict(donor.model.named_buffers(remove_duplicate=False)).keys()==donor_buffers.keys()
        assert all(dict(donor.model.named_buffers(remove_duplicate=False))[n] is t for n,t in donor_buffers.items())
        with donor.installed():donor.graph._validate()
        report['donor_remained_fresh_and_unchanged']=True
    report.update(passed=True,phase='completed',frames_completed=243,all_frozen_car_and_face_outputs_byte_equal=True,
        reset_reproduces_first_frame=True,inputs_constants_history_and_held_outputs_unchanged=True,
        reused_approved_video=repair['video'])
except BaseException:
    report.update(passed=False,phase='failed',error=traceback.format_exc());raise
finally:
    try:
        if decoder is not None and decoder.poll() is None:decoder.kill();decoder.wait()
        if stderr is not None:stderr.close()
        for s in stacks.values():s.close()
        if donor is not None:donor.close()
        finalize_sources()
    except BaseException:
        report.update(passed=False,phase='failed',finalization_error=traceback.format_exc());raise
    finally:save()
print(json.dumps(dict(passed=True,frames=243,all_byte_equal=True,body=report['body'],paired=report['paired']['summary'])),flush=True)
