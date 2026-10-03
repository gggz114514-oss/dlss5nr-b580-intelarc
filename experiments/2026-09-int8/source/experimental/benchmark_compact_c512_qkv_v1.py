"""Screen compact C512 QKV on the accepted repaired INT8 route.

Same QKV values including padding; all keys/values, normalization, attention,
FP8 rules and pooling retained. Compare reference with BM64 and BM32 compact
projection policies. All car/243face outputs must match frozen repair exactly.
Separate body and resident pipeline timings; no new ranges/video/promotion.
"""
from layout_crop_validation_env_v1 import *
import statistics,subprocess,time,traceback
OUT=D/'results/compact-c512-qkv-v1';assert not OUT.exists()
breakdown=receipt(D/'results/c512-breakdown-v1/validation.json','707ca3af8f6542c6eb9dad5f739906b1719968987202e005052815084d7a1b57')
repair=receipt(D/'results/int8-ffn-range-repair-v1/validation.json','bd53a2e2b818886e79fed05e836646bdd89814a434aa932ae5bdd4f7944ce2aa')
full=receipt(D/'results/long-precision-480-v1/validation.json','858b4dd646c3be61263bc3fac6fabe400912686af672861638619c63be3be8a3')
paired=receipt(D/'results/layout-crop-residual256-v1/validation.json','b0d5b7e984ca530d36459b3c20651b3989f37719aae57bcee3bb6eb64e265fe8')
review_path=D/'results/int8-ffn-range-repair-v1/user-review-v1.json'
assert sha(review_path)=='8629fdc089e24eeb9b968e65d6a85d80f89268437a488a17d0c11497360ce214'
assert js(review_path)['status']=='accepted_face_color_for_range_repair_v1'
import numpy as np
import torch,triton
from PIL import Image
import nr_backend.multihead_block as multi
from fused_qkv_pack_native_half_v1 import forward as reference_pack
from nr256_selected_stack_v4 import Stack as DonorStack
from int8_ffn_calibrated_stack_v1 import Stack as BaseStack
from compact_c512_qkv_stack_v1 import Stack as CandidateStack
from serial_graph_workspace_v1 import share_before_capture
from nr_backend.execution import use_arithmetic_backend
from residual_scale_v1 import ResidualScale
from face480_residual_scale_v1 import Face480Scale
from nr_review_video_finalize_v1 import read_exact
import capture_body_v1 as body
import compressed_arrays_v1 as arrays

assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site')
NAMES=('repaired','compact64','compact32')
car_inputs=R/'inputs/flow-full-1920x1080-v3';manifest_path=car_inputs/'manifest.json';manifest=js(manifest_path)
assert sha(manifest_path)==paired['sources'][str(manifest_path)]
expected={r['frame']:r for r in repair['paired']['runs']['repaired_int8'] if r['round']==0}
assert len(expected)==len(manifest['frames'])==13
source=Path(full['source']);ffmpeg=Path(next(p for p in full['sources'] if Path(p).name=='ffmpeg.exe'))
for p in (Path(__file__),HERE/'compact_c512_qkv_pack_v1.py',HERE/'compact_c512_qkv_stack_v1.py',
          HERE/'audit_compact_c512_qkv_v1.py',HERE/'Run-CompactC512QkvV1.cmd',review_path,manifest_path,source,ffmpeg):sources[str(p)]=sha(p)
OUT.mkdir();stacks={};held={};decoder=None;stderr=None;donor=None
report=dict(scope=__doc__,passed=False,phase='initializing',sources=sources,exact_gate=gate,queue_check=queue_check,
    candidate_promoted=False,new_quantization=False,recalibration=False,new_video=False,default_unchanged=True,
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
    stacks['compact64']=CandidateStack(EXACT,hidden_scales=scales,share_with=donor,bm=64)
    stacks['compact32']=CandidateStack(EXACT,hidden_scales=scales,share_with=donor,bm=32)
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
            base=report['capture_build_counts']['repaired'][i]
            a=report['capture_build_counts']['compact64'][i];b=report['capture_build_counts']['compact32'][i]
            assert a==b
            assert a['standalone_fp8']<=base['standalone_fp8'] and a['triton_calls']<=base['triton_calls']
            assert a['quantization_calls']==base['quantization_calls']
        for name in NAMES[1:]:
            assert stacks[name].rewrite.layout.compact_calls==16*6
        report['smaller_qkv_projection_same_logical_rounding']=True
        report['qkv_checks']=[]
        # Full eager/body-graph equivalence and private-state isolation for each
        # actual captured mode, before timing. Does not call public model again.
        report['phase']='body_validation';save();report['eager_graph_checks']=[]
        for name,s in stacks.items():
            previous=s.model._previous;old=raw(previous);seed=s.model.next_seed
            for entry in s.graph.entries.values():
                inputs={k:None if t is None else raw(t) for k,t in entry.inputs.items()}
                candidate=name!='repaired'
                mode='temporal' if entry.inputs.get('previous') is not None else 'reset'
                blocks_by_attention={id(block.attention):f'{group}.{i}' for group in ('encoder512','decoder512')
                    for i,block in enumerate(getattr(s.model,group))}
                def probe(module,features,shift,actual):
                    sy,sx=shift
                    padded=torch.nn.functional.pad(features,(0,0,sx,4-sx,sy,4-sy))
                    z=multi.sm89_f16_dot(multi.quantize_fp8(padded),module.qkv,chunk_k=16).reshape(16,16,16,3,32)
                    reference,_=reference_pack(z,module.scale,module.pixel_order,rows=s.window_blocks.layout.rows)
                    actual_bytes=[raw(t) for t in actual];reference_bytes=[raw(t) for t in reference]
                    assert actual_bytes==reference_bytes,(name,mode,blocks_by_attention[id(module)],'full QKV mismatch')
                    report['qkv_checks'].append(dict(route=name,mode=mode,block=blocks_by_attention[id(module)],
                        shift=list(shift),full_qkv_byte_equal=True,includes_all_padding_rows=True,
                        qkv_raw_sha256=[hashlib.sha256(v).hexdigest() for v in actual_bytes],
                        shape=list(actual[0].shape),valid_pixels=144,padded_pixels=256))
                if candidate:s.rewrite.layout.qkv_probe=probe
                try:
                    with s.installed(),body.installed(),use_arithmetic_backend('triton'):
                        value=body.forward_front(s.model,**entry.inputs,sigmoid=s.model.sigmoid,
                            blend_scale=s.model.blend_scale,return_float32=False);torch.xpu.synchronize()
                finally:
                    if candidate:s.rewrite.layout.qkv_probe=None
                assert raw(value)==raw(entry.output)
                assert inputs=={k:None if t is None else raw(t) for k,t in entry.inputs.items()}
                report['eager_graph_checks'].append(dict(route=name,byte_equal=True,inputs_unchanged=True))
            assert s.model._previous is previous and raw(previous)==old and s.model.next_seed==seed
        assert len(report['qkv_checks'])==64
        report['all_full_qkv_including_padding_byte_equal']=True
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
            resources=candidate['compact_qkv']['resources'];assert resources and all(v['spills']==0 for v in resources.values())
            assert candidate['compact_qkv']['calls']==128
            assert candidate['compact_qkv']['dense_rows']==144 and candidate['compact_qkv']['full_keys_and_values_preserved']
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
