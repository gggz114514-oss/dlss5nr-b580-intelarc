"""CPU-only continuous INT8 ViT FFN feasibility, not GPU speed or visual approval.

Use all eight frozen real 64-token FFN boundaries. Fit static hidden scales from
48 interleaved tokens, and separately report the other 16 of the SAME FRAME.
These held-out tokens are correlated and are not unseen-frame/video validation.
No GPU framework import, original model edits, or optimized runtime promotion.
"""
import hashlib,importlib.util,json,os
from pathlib import Path
import sys,traceback
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT=D/'experimental/int8-ffn-segment-cpu-v1';assert not OUT.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
prior_path=D/'experimental/fp8-partial-int8-projection-v3/validation.json'
assert sha(prior_path)=='0d181b2f97b7d145627f03042d44f3a45fd27c49d4bd913838538f2a4fbb8c28'
prior=js(prior_path);assert prior['passed'] and prior['phase']=='completed'
sources=dict(prior['sources']);sources.update(prior['exact_gate']);sources[str(prior_path)]=sha(prior_path)
for p in (Path(__file__),HERE/'int8_ffn_segment_oracle_v1.py',HERE/'Run-Int8FfnSegmentCpuV1.cmd'):
    sources[str(p)]=sha(p)
assert all(sha(p)==h for p,h in sources.items())
vit_path=D/'experimental/vit-head-layout-body-v1/validation.json'
assert sha(vit_path)=='95faf6bb6818627d30f1aa5cf53fd0b30c01050d0130e6705dce96af206a2c26'
vit=js(vit_path);assert vit['passed'] and len(vit['boundaries'])==8
sources[str(vit_path)]=sha(vit_path)
dense_path=D/'experimental/current-dense-operands-v2/validation.json'
assert sha(dense_path)=='46a16ba64e9f84425928d0a0fd93862e99574910b031ba098c789f0ca69e7290'
dense=js(dense_path);assert dense['passed'];sources[str(dense_path)]=sha(dense_path)
import numpy as np
import compressed_arrays_v1 as arrays
import int8_ffn_segment_oracle_v1 as oracle
spec=importlib.util.spec_from_file_location('pinned_weight_reader',ROOT/'backend/nr_backend/weights.py')
reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
weights_path=EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin'
records=reader.load_pinned_records(weights_path);sources[str(weights_path)]=sha(weights_path)
# Spread the held-out subset across the 8x8 token grid; trailing rows can be pad.
CAL=[i for i in range(64) if ((i//8)+(i%8))%4!=0]
HELD=[i for i in range(64) if ((i//8)+(i%8))%4==0]
assert len(CAL)==48 and len(HELD)==16 and set(CAL).isdisjoint(HELD)
OUT.mkdir()
report=dict(scope=__doc__,passed=False,no_gpu_execution=True,phase='initialized',cases=[],sources=sources,
    candidate_promoted=False,complete_migration=False,new_quantization=True,full_model_test=False,
    calibration_tokens=CAL,heldout_same_frame_tokens=HELD,
    scale_margins=[1.0,1.25],weights_saved=False,
    scheme='dynamic INT8 row entry -> W8 expansion -> cubic half -> static per-channel INT8 hidden -> folded W8 contraction -> half residual -> outer FP8',
    semantic_changes=['INT8 entry and weights are approximate.',
        'The internal FP8 hidden boundary is replaced by signed uniform INT8 with static channel scales.',
        'Contraction uses one INT32 reduction rather than four FP16 partial-output merges.',
        'The cubic half arithmetic, scaled residual and outer FP8 boundary are retained.'],
    limitations=['One fixed frame only; hidden scales fitted to 48 correlated tokens per layer.',
        'Each FFN starts from its own frozen selected input, not a chained whole-model rollout.',
        'CPU arrays include dequantized intermediates for diagnostics, not the planned GPU storage.',
        'No latency measurement, GPU kernel validation, temporal quality test or visual approval.'])


def scalar(value):
    if isinstance(value,(np.bool_,np.integer,np.floating)):return value.item()
    raise TypeError(type(value).__name__)


def save():
    (OUT/'validation.json').write_text(json.dumps(report,indent=2,default=scalar,allow_nan=False)+'\n',encoding='utf-8')


def get(meta):
    value=arrays.load(meta)
    assert value.dtype==np.dtype('f2') and np.isfinite(value).all()
    return value


def numeric_guards():
    positive=oracle._positive.astype(np.float16)
    values=np.concatenate((positive,-positive))
    assert oracle.fp8_boundary(values).tobytes()==values.tobytes()
    mid=((positive[:-1].astype(np.float32)+positive[1:].astype(np.float32))*.5).astype(np.float16)
    expected=positive[np.arange(126)+np.arange(126)%2]
    assert oracle.fp8_boundary(mid).tobytes()==expected.tobytes()
    assert oracle.fp8_boundary(np.array([65000,-65000],dtype=np.float16)).tolist()==[448.,-448.]
    q,s=oracle.quantize_axis(np.zeros((3,32),dtype=np.float16),1)
    assert (q==0).all() and (s==1).all()
    assert oracle.quantize(np.array([-.5,.5,-127.6,127.6],dtype=np.float32),np.float32(1)).tolist()==[-1,1,-127,127]
    a=np.array([[127,-127,1,0]],dtype=np.int8);w=np.array([[127],[-127],[1],[2]],dtype=np.int8)
    assert oracle.integer_dot(a,w).item()==32259
    report['numeric_guards']=dict(finite_fp8_encodings=254,positive_midpoint_ties=126,
        signed_zero_preserved=True,saturation_checked=True,zero_scale_and_int8_ties_checked=True,int32_dot_checked=True)


save()
try:
    numeric_guards()
    for row in sorted(vit['boundaries'],key=lambda r:r['block']):
        index=int(row['block']);record=lambda layer:records[f'block{31+index}.layer{layer}.layer']
        x=get(row['input']);href=get(row['reference'][0]);expected=get(row['reference'][1])
        assert x.shape==(64,1024) and href.shape==(64,4096) and expected.shape==(64,1024)
        assert oracle.fp8_boundary(x).tobytes()==x.tobytes()
        wexp=oracle.decode_weights(record(0)[:4194304],1024,4096)
        wcontract=oracle.decode_weights(record(1)[:4194304],4096,1024)
        if index==0:
            captured=next(c['operands']['w'] for c in dense['cases'].values() if c.get('weight_owner_names')==['vit.0.expand'])
            assert wexp.tobytes()==get(captured).tobytes()
            report['decode_layout_matches_captured_weight']=captured
        assert np.any(x[CAL]!=0) and np.any(x[HELD]!=0),'Calibration subsets contain no nonzero input signal'
        skip=np.frombuffer(record(1)[4194304:],dtype=np.float16).copy()
        assert skip.shape==(1024,)
        initial=(x.astype(np.float32)*skip.astype(np.float32)).astype(np.float16)
        hidden,entry=oracle.expansion(x,wexp)
        for margin in (1.0,1.25):
            sh=oracle.hidden_scale(href[CAL],margin)
            unquant,out,packed=oracle.contraction(hidden,sh,wcontract,initial)
            dequant_hidden=packed['qh'].astype(np.float32)*sh
            # Demonstrate the scale-fold algebra independently of weight rounding.
            sample_q=packed['qh'][:2,:32].astype(np.float64)
            sample_s=sh[0,:32].astype(np.float64);sample_w=wcontract[:32,:8].astype(np.float64)
            left=(sample_q*sample_s)@sample_w;right=sample_q@(sample_s[:,None]*sample_w)
            assert np.allclose(left,right,rtol=1e-12,atol=1e-12)
            clipped=np.abs(hidden.astype(np.float32)/sh)>127
            metrics={}
            for name,section in (('calibration',CAL),('heldout_same_frame',HELD),('all',slice(None))):
                metrics[name]=dict(mlp=oracle.error_metrics(out[section],expected[section]),
                    hidden=oracle.error_metrics(dequant_hidden[section],href[section]),
                    hidden_clipped_fraction=float(clipped[section].mean()),
                    input_nonzero_tokens=int(np.any(x[section]!=0,axis=1).sum()))
            changed=int(np.count_nonzero(out.view(np.uint16)!=expected.view(np.uint16)))
            case=dict(name=f'vit.{index}.ffn',margin=margin,metrics=metrics,
                mlp_byte_equal=changed==0,mlp_different_half_words=changed,mlp_total_half_words=out.size,
                input_quantization=oracle.error_metrics(entry['qx'].astype(np.float32)*entry['sx'],x),
                expansion_weight_quantization=oracle.error_metrics(entry['qw'].astype(np.float32)*entry['sw'],wexp),
                folded_weight_quantization=oracle.error_metrics(packed['qw'].astype(np.float32)*packed['sw'],packed['folded']),
                hidden_scale=arrays.save(sh),mlp_output=arrays.save(out),mlp_unquantized=arrays.save(unquant),
                reference_input=row['input'],reference_hidden=row['reference'][0],reference_mlp=row['reference'][1],
                static_scale_fold_check=True,
                intended_hidden_storage_bytes=int(packed['qh'].nbytes),selected_hidden_storage_bytes=int(href.nbytes),
                hidden_scale_bytes=int(sh.nbytes),weight_packing_cpu_only=True)
            report['cases'].append(case);report.update(phase='calibrating',active_case=case['name']);save()
            print(json.dumps(dict(case=case['name'],margin=margin,
                heldout=metrics['heldout_same_frame'],mlp_byte_equal=case['mlp_byte_equal'])),flush=True)
    assert len(report['cases'])==16
    report['summary']={}
    for margin in (1.0,1.25):
        rows=[r for r in report['cases'] if r['margin']==margin]
        report['summary'][str(margin)]=dict(blocks=len(rows),
            mean_heldout_relative_rmse=float(np.mean([r['metrics']['heldout_same_frame']['mlp']['relative_rmse'] for r in rows])),
            worst_heldout_relative_rmse=max(r['metrics']['heldout_same_frame']['mlp']['relative_rmse'] for r in rows),
            max_heldout_clip_fraction=max(r['metrics']['heldout_same_frame']['hidden_clipped_fraction'] for r in rows))
    assert not any(name=='torch' or name.startswith('triton') for name in sys.modules)
    report.update(passed=True,phase='completed')
except BaseException:
    report.update(phase='failed',error=traceback.format_exc());raise
finally:
    if not all(sha(p)==h for p,h in sources.items()):
        report.update(passed=False,phase='failed',finalization_error='Source hash changed')
        save();raise RuntimeError('Source hash changed')
    save()
print(json.dumps(dict(passed=True,no_gpu_execution=True,summary=report['summary'])),flush=True)
