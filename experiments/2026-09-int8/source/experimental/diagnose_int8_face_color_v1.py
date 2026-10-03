"""Causal face-color investigation using frozen frames96/192 and their histories.

First authenticate reconstruction of each reviewed FP16/INT8 output. Cross the
current body arithmetic with the two stored previous outputs (2x2 design).
Then hold FP16 front/history fixed and restore each FFN or one arithmetic
component. CPU oracle must exactly reproduce the GPU INT8 output. These
interventions are diagnostic, not a speed test, new video or promotion.
"""
from layout_crop_validation_env_v1 import *
from contextlib import contextmanager
import traceback
OUT=D/'experimental/int8-face-color-cause-v1';assert not OUT.exists()
face=receipt(D/'results/int8-ffn-face480-finish-v1/validation.json','b9590f1d70607ca0e3a20869f85ee7149af08ca0a9d4c8d005138e4bb1751e24')
full=receipt(D/'results/long-precision-480-v1/validation.json','858b4dd646c3be61263bc3fac6fabe400912686af672861638619c63be3be8a3')
cpu=receipt(D/'experimental/int8-ffn-segment-cpu-v1/validation.json','1f1bdd147c386245dad5701407f65f5ac663a31a36647c4aeae6e11c9b799130')
import numpy as np
import torch,triton
from PIL import Image
from nr256_selected_stack_v4 import Stack
from int8_ffn_nr_stack_v1 import Stack as CandidateStack
from int8_face_color_scope_v1 import ColorProbe
from face480_residual_scale_v1 import Face480Scale
from nr_backend.execution import use_arithmetic_backend
import capture_body_v1 as body
import compressed_arrays_v1 as arrays
import int8_ffn_segment_oracle_v1 as oracle
assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site')
for p in (Path(__file__),HERE/'int8_face_color_scope_v1.py',HERE/'Run-Int8FaceColorCauseV1.cmd',HERE/'audit_int8_face_color_v1.py'):
    sources[str(p)]=sha(p)
OUT.mkdir();report=dict(scope=__doc__,passed=False,phase='saved_output_analysis',sources=sources,exact_gate=gate,
    targets=[96,192],saved_output_analysis=[],cases=[],candidate_promoted=False,performance_benchmark=False,
    new_video=False,quality_approval=False,queue_check=queue_check,
    roi_scope='Fixed 384x384 display ROI, includes skin and background; not a segmented skin-only statistic',
    calibration_scope='Existing scales fitted to 48 interleaved tokens in one other fixed frame; no recalibration')
ROI=(256,16,640,400)
Y=np.array([.2126,.7152,.0722],dtype='f8')
raw=lambda t:t.cpu().numpy().tobytes()
digest=lambda a:hashlib.sha256(a.tobytes()).hexdigest()
stacks={};diag=None

def save():
    p=OUT/'progress.tmp';p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');p.replace(OUT/'validation.json')

def region(a):return a[16:400,256:640].astype('f8')
def metrics(a,reference):
    d=region(a)-region(reference);rgb=d.mean((0,1));yp=d@Y
    return dict(mean_delta_RGB_8bit=(rgb*255).tolist(),mean_delta_Yprime_8bit=float(rgb@Y*255),
        mean_delta_blue_minus_red_8bit=float((rgb[2]-rgb[0])*255),
        rgb_rmse_8bit=float(np.sqrt(np.mean(d*d))*255),mae_8bit=float(np.abs(d).mean()*255),
        yprime_rmse_8bit=float(np.sqrt(np.mean(yp*yp))*255))
def mean_color(a):
    rgb=region(a).mean((0,1));return dict(mean_RGB_8bit=(rgb*255).tolist(),mean_Yprime_8bit=float(rgb@Y*255))

def pixels_for(index):
    p=D/'results/int8-ffn-face480-review-v1'/f'frame{index:03d}-full-and-face.png'
    assert sha(p)==face['images'][str(p)]
    im=np.asarray(Image.open(p).convert('RGB'))
    pixels=im[56:536,:864].copy()
    assert digest(pixels)==face['frames'][index]['input_rgb8_sha256']
    return pixels,im

save()
try:
    torch.set_num_threads(2)
    for index in (48,72,96,144,192,242):
        pixels,im=pixels_for(index)
        values={'source':pixels.astype('f4')/255,'nr256_fp16':im[56:536,864:1728].astype('f4')/255,
                'nr256_int8':im[56:536,1728:2592].astype('f4')/255}
        for key,name in [('baseline','full480_exact'),('fp16_xmx','full480_fp16')]:
            values[name]=np.rint(np.clip(arrays.load(full['frames'][index]['runs'][key]['output']).astype('f4'),0,1)*255)/255
        report['saved_output_analysis'].append(dict(frame=index,mean_colors={k:mean_color(v) for k,v in values.items()},
            full_fp16_vs_exact=metrics(values['full480_fp16'],values['full480_exact']),
            reduced_fp16_vs_full_fp16=metrics(values['nr256_fp16'],values['full480_fp16']),
            int8_vs_reduced_fp16=metrics(values['nr256_int8'],values['nr256_fp16'])))
    scales=[arrays.load(next(c for c in cpu['cases'] if c['name']==f'vit.{i}.ffn' and c['margin']==1.25)['hidden_scale']) for i in range(8)]
    scaler=Face480Scale();stacks['selected']=Stack(EXACT)
    stacks['int8_p4']=CandidateStack(EXACT,hidden_scales=scales,share_with=stacks['selected'])
    selected=stacks['selected'];selected_scope=selected.rewrite.vit_layout
    diag=ColorProbe(selected.model,selected.provider,scales)
    packed=[hashlib.sha256(raw(t)).hexdigest() for row in diag.packed for t in row]
    sources.update(selected.rewrite.fork.sources)

    @contextmanager
    def mode_scope(mode,block=None):
        assert selected.rewrite.vit_layout is selected_scope
        target=selected_scope if mode=='selected' else diag
        diag.mode=mode;diag.restore_block=block;selected.rewrite.vit_layout=target
        try:yield
        finally:
            assert selected.rewrite.vit_layout is target;selected.rewrite.vit_layout=selected_scope

    with torch.inference_mode():
        for index in report['targets']:
            report['phase']=f'frame_{index}';save()
            pixels,_=pixels_for(index);rgb_cpu=pixels.astype('f4')/255
            flow_cpu=arrays.load(face['frames'][index]['motion'])
            rgb=torch.from_numpy(rgb_cpu).to('xpu');motion=torch.from_numpy(flow_cpu).to('xpu')
            inputs={};normal={};canvas_for=None
            case=dict(frame=index,factorial={},interventions={},ffn_clip_statistics=[],reproduction={})
            report['cases'].append(case)
            for name,s in stacks.items():
                previous=arrays.load(face['frames'][index-1]['runs'][name]['low'])
                s.model.reset();s.model._previous=torch.from_numpy(previous.copy()).to('xpu');s.model._next_seed=index
                with s.installed(),use_arithmetic_backend('triton'):
                    canvas,low_flow=scaler.prepare(rgb,motion)
                    low=s.model(canvas,low_flow,reset=False);output=scaler.composite(rgb,canvas,low)
                    torch.xpu.synchronize()
                actual=low.cpu().numpy();a=output.cpu().numpy()
                reference=arrays.load(face['frames'][index]['runs'][name]['low'])
                assert actual.tobytes()==reference.tobytes(),(index,name,'reviewed low output reproduction')
                assert digest(a)==face['frames'][index]['runs'][name]['full_raw_sha256']
                assert raw(s.model._previous)==actual.tobytes() and s.model.next_seed==index+1
                inputs[name]={k:None if t is None else t.clone() for k,t in s.graph.last_entry.inputs.items()}
                normal[name]=(a.copy(),actual.copy())
                if canvas_for is None:canvas_for=canvas.clone()
                else:assert raw(canvas_for)==raw(canvas)
                case['reproduction'][name]=dict(low_byte_equal=True,full_hash_equal=True,prior_seed=index,
                    supplied_previous=face['frames'][index-1]['runs'][name]['low'])
            stable_inputs={name:{k:None if t is None else raw(t) for k,t in values.items()} for name,values in inputs.items()}
            stable_history={n:(s.model._previous,raw(s.model._previous),s.model.next_seed) for n,s in stacks.items()}
            captured={}
            def probe(block,x,mlp,output,qh):
                assert qh is not None and block not in captured
                captured[block]=(x.cpu().numpy().copy(),mlp.cpu().numpy().copy(),qh.cpu().numpy().copy())
            def compute(mode,history='selected',block=None,capture=False):
                diag.int8_probe=probe if capture else None
                try:
                    with mode_scope(mode,block),selected.installed(),body.installed(),use_arithmetic_backend('triton'):
                        low=body.forward_front(selected.model,**inputs[history],sigmoid=selected.model.sigmoid,
                            blend_scale=selected.model.blend_scale,return_float32=False)
                        out=scaler.composite(rgb,canvas_for,low);torch.xpu.synchronize()
                    a,b=out.cpu().numpy().copy(),low.cpu().numpy().copy()
                    assert np.isfinite(a).all() and np.isfinite(b).all()
                    return a,b
                finally:diag.int8_probe=None
            factorial={}
            for mode,history,label in [('selected','selected','A_fp16_body_fp16_history'),
                                       ('int8','selected','B_int8_body_fp16_history'),
                                       ('selected','int8_p4','C_fp16_body_int8_history'),
                                       ('int8','int8_p4','D_int8_body_int8_history')]:
                a,b=compute(mode,history,capture=label.startswith('D_'));factorial[label]=a
                if label.startswith('A_'):assert b.tobytes()==normal['selected'][1].tobytes()
                if label.startswith('D_'):assert b.tobytes()==normal['int8_p4'][1].tobytes()
                case['factorial'][label]=dict(low=arrays.save(b),full_raw_sha256=digest(a),versus_fp16=metrics(a,normal['selected'][0]))
            A,B,C,E=[factorial[k] for k in ('A_fp16_body_fp16_history','B_int8_body_fp16_history','C_fp16_body_int8_history','D_int8_body_int8_history')]
            interaction=E.astype('f8')-B.astype('f8')-C.astype('f8')+A.astype('f8')
            case['factorial_decomposition']=dict(current_body=metrics(B,A),prior_history=metrics(C,A),
                interaction=metrics(interaction,np.zeros_like(interaction)),total=metrics(E,A))
            assert set(captured)==set(range(8))
            for block,(x,mlp,qh) in sorted(captured.items()):
                we,wc,skip,sh=diag.cpu_weights[block]
                hidden,_=oracle.expansion(x,we)
                initial=(x.astype('f4')*skip.astype('f4')).astype('f2')
                _,expected,details=oracle.contraction(hidden,sh,wc,initial)
                assert expected.tobytes()==mlp.tobytes() and details['qh'].tobytes()==qh.tobytes(),(index,block,'GPU/CPU FFN contract')
                unit=hidden.astype('f4')/sh
                clipped=np.abs(unit)>127;clip_error=np.clip(hidden.astype('f4'),-127*sh,127*sh)-hidden.astype('f4')
                case['ffn_clip_statistics'].append(dict(block=block,cpu_matches_gpu=True,
                    input=arrays.save(x),mlp=arrays.save(mlp),hidden_int8=arrays.save(qh),hidden_before_quantization=arrays.save(hidden),
                    clip_fraction=float(clipped.mean()),positive_clip_fraction=float((unit>127).mean()),negative_clip_fraction=float((unit<-127).mean()),
                    max_range_multiple=float(np.max(np.abs(unit))/127),p99_range_multiple=float(np.percentile(np.abs(unit),99)/127),
                    mean_signed_clip_error=float(clip_error.mean(dtype='f8')),clip_error_rmse=float(np.sqrt(np.mean(clip_error.astype('f8')**2)))))
            cpu_out,cpu_low=compute('cpu_oracle')
            assert cpu_low.tobytes()==arrays.load(case['factorial']['B_int8_body_fp16_history']['low']).tobytes()
            case['cpu_replacement_reproduces_gpu']=True
            for mode in ('fp16_all','unclipped_hidden','fp16_expand','restore_internal_fp8','unrounded_contract'):
                a,b=compute(mode)
                if mode=='fp16_all':assert b.tobytes()==normal['selected'][1].tobytes()
                case['interventions'][mode]=dict(low=arrays.save(b),full_raw_sha256=digest(a),versus_fp16=metrics(a,A))
                save();print(json.dumps(dict(frame=index,intervention=mode,done=True)),flush=True)
            for block in range(8):
                a,b=compute('restore_block',block=block)
                case['interventions'][f'restore_ffn_{block}']=dict(low=arrays.save(b),full_raw_sha256=digest(a),versus_fp16=metrics(a,A))
            baseline_error=case['factorial']['B_int8_body_fp16_history']['versus_fp16']['rgb_rmse_8bit']
            for row in case['interventions'].values():
                row['roi_rgb_rmse_change_percent_vs_int8_same_history']=(row['versus_fp16']['rgb_rmse_8bit']/baseline_error-1)*100 if baseline_error else None
            for n,s in stacks.items():
                ref,history_bytes,seed=stable_history[n]
                assert s.model._previous is ref and raw(ref)==history_bytes and s.model.next_seed==seed
                s.rewrite.verify_restored()
                with s.installed():s.graph._validate()
            for name,values in inputs.items():
                for k,t in values.items():assert (None if t is None else raw(t))==stable_inputs[name][k]
            assert raw(rgb)==rgb_cpu.tobytes() and raw(motion)==flow_cpu.tobytes()
            case['diagnostic_inputs_histories_seed_unchanged']=True
            save();print(json.dumps(dict(frame=index,completed=True,body_error=case['factorial_decomposition']['current_body'],history_error=case['factorial_decomposition']['prior_history'])),flush=True)
    assert packed==[hashlib.sha256(raw(t)).hexdigest() for row in diag.packed for t in row]
    diag.verify_restored();sources.update(selected.rewrite.fork.sources)
    report.update(passed=True,phase='completed',all_reviewed_outputs_reproduced=True,all16_gpu_ffns_match_cpu=True,
        both_cpu_body_replacements_match_gpu=True,no_default_changes=True,no_recalibration=True)
except BaseException:
    report.update(passed=False,phase='failed',error=traceback.format_exc());raise
finally:
    try:
        if diag is not None:diag.verify_restored()
        for s in stacks.values():s.close()
        finalize_sources()
    except BaseException:
        report.update(passed=False,phase='failed',finalization_error=traceback.format_exc());raise
    finally:save()
print(json.dumps(dict(passed=report['passed'],frames=report['targets'],causal_diagnostics=True,performance_benchmark=False)),flush=True)
