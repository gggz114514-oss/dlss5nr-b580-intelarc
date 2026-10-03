"""Paired NR256 static network body, with eight sequential approximate INT8 FFNs.

Includes encoder/C512/ViT/decoder/post. Excludes MotionNR input validation,
noise/warp/front construction, private history update, IO, packing and capture.
One frozen temporal body's inputs; no independent video quality approval.
"""
from layout_crop_validation_env_v1 import *
from contextlib import contextmanager
import statistics,time,traceback
OUT=D/'experimental/int8-ffn-body-v1';assert not OUT.exists()
local=receipt(D/'experimental/int8-ffn-segment-gpu-v1/validation.json',
    '5e2b26af3c53214dca6cf1413bec736e6acc38afa6438a36aab0ed8e8828d5fe')
assert len(local['cases'])==8 and not local['full_model_test']
cpu=receipt(D/'experimental/int8-ffn-segment-cpu-v1/validation.json',
    '1f1bdd147c386245dad5701407f65f5ac663a31a36647c4aeae6e11c9b799130')
checkpoint=receipt(D/'experimental/layout-crop-checkpoint-v1/saved-audit-v1.json',
    'f56a78d9260e8606e7a0faa280226c8308e770c66651266d72d258218d84fd05')
for p,h in checkpoint['new_files'].items():
    path=str(ROOT/p)
    assert sha(path)==h
    if path in sources:assert sources[path]==h
    sources[path]=h
for p in (Path(__file__),HERE/'Run-Int8FfnBodyV1.cmd',HERE/'int8_ffn_body_scope_v1.py',HERE/'nr256_selected_stack_v4.py'):
    sources[str(p)]=sha(p)
fixture_path=D/'experimental/selected-body-stages-v1/validation.json'
assert sha(fixture_path)=='f2553253c77b49e535c50f176afec1246abf5ee44784643bde1d25a820661e66'
fixture=js(fixture_path);sources[str(fixture_path)]=sha(fixture_path)
import numpy as np
import torch,triton
from nr256_selected_stack_v4 import Stack
from nr_backend.execution import use_arithmetic_backend
from int8_ffn_body_scope_v1 import Int8VitLayout
import int8_ffn_segment_oracle_v1 as oracle
import compressed_arrays_v1 as arrays
import capture_body_v1 as body
assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site')
OUT.mkdir()
report=dict(scope=__doc__,passed=False,phase='initialized',sources=sources,exact_gate=gate,
    candidate_promoted=False,complete_migration=False,full_nr_call_test=False,
    sequential_eight_ffn_test=True,new_quantization=True,calibration_margin=1.25,
    queue_check=queue_check,body_inputs=fixture['body_inputs'],expected_output=fixture['expected_output'],
    boundaries=[],outputs={},builds={},samples_ms={'selected':[],'int8_p4':[]},orders=[],
    changed_input_checks=[],torch_version=torch.__version__,triton_version=triton.__version__)


def scalar(value):
    if isinstance(value,(np.bool_,np.integer,np.floating)):return value.item()
    raise TypeError(type(value).__name__)


def save():
    (OUT/'validation.json').write_text(json.dumps(report,indent=2,default=scalar,allow_nan=False)+'\n',encoding='utf-8')


raw=lambda t:t.cpu().numpy().tobytes()
graphs=[];stack=None;candidate=None;save()


@contextmanager
def route(name):
    assert name in ('selected','int8_p4')
    original=stack.rewrite.vit_layout
    assert original is selected_scope
    if name=='int8_p4':stack.rewrite.vit_layout=candidate
    try:yield
    finally:
        assert stack.rewrite.vit_layout is (candidate if name=='int8_p4' else original)
        stack.rewrite.vit_layout=original


try:
    torch.set_num_threads(2)
    stack=Stack(EXACT);selected_scope=stack.rewrite.vit_layout
    scales=[arrays.load(next(c for c in cpu['cases'] if c['name']==f'vit.{i}.ffn' and c['margin']==1.25)['hidden_scale']) for i in range(8)]
    candidate=Int8VitLayout(stack.model,stack.provider,scales)
    constant_hashes=[hashlib.sha256(raw(t)).hexdigest() for values in candidate.packed for t in values]
    static={n:None if meta is None else torch.from_numpy(arrays.load(meta)).to('xpu') for n,meta in fixture['body_inputs'].items()}
    expected=arrays.load(fixture['expected_output'])
    options=dict(sigmoid=stack.model.sigmoid,blend_scale=stack.model.blend_scale,return_float32=False)
    references={};actual_boundaries={};entries={};saved={}
    with torch.inference_mode(),stack.installed(),body.installed(),use_arithmetic_backend('triton'):
        def compute():return body.forward_front(stack.model,**static,**options)
        def selected_probe(index,features,outputs,heads):
            assert index not in references
            references[index]=dict(input=arrays.save(oracle.fp8_boundary(features.cpu().numpy())),
                mlp=arrays.save(outputs[1].cpu().numpy()),output=arrays.save(outputs[-1].cpu().numpy()))
        def int8_probe(index,x,mlp,output,qh):
            assert index not in actual_boundaries
            actual_boundaries[index]=dict(input=arrays.save(x.cpu().numpy()),mlp=arrays.save(mlp.cpu().numpy()),
                output=arrays.save(output.cpu().numpy()),hidden=arrays.save(qh.cpu().numpy()))
        selected_scope.probe=selected_probe
        try:
            baseline=compute().cpu().numpy()
            assert baseline.tobytes()==expected.tobytes()
        finally:selected_scope.probe=None
        candidate.int8_probe=int8_probe
        try:
            with route('int8_p4'):approx=compute().cpu().numpy()
        finally:candidate.int8_probe=None
        assert set(references)==set(actual_boundaries)==set(range(8))
        assert actual_boundaries[0]['input']['raw_sha256']==references[0]['input']['raw_sha256']
        assert np.isfinite(approx).all() and approx.shape==expected.shape
        report['rgb_error_vs_selected']=oracle.error_metrics(approx,expected)
        report['outputs']={'selected':arrays.save(baseline),'int8_p4':arrays.save(approx)}
        report['outputs_differ']=baseline.tobytes()!=approx.tobytes()
        # New input to each block includes all preceding approximate blocks. Check
        # the fused FFN against its CPU oracle again on these actual chained inputs.
        for index in range(8):
            ref=references[index];actual=actual_boundaries[index];module=stack.model.vit[index]
            x=arrays.load(actual['input']);sh=scales[index]
            hidden,_=oracle.expansion(x,module.expand.cpu().numpy())
            initial=(x.astype(np.float32)*module.ffn_skip.cpu().numpy().astype(np.float32)).astype(np.float16)
            _,mlp,packed=oracle.contraction(hidden,sh,module.contract.cpu().numpy(),initial)
            assert mlp.tobytes()==arrays.load(actual['mlp']).tobytes(),(index,'chained FFN CPU mismatch')
            assert packed['qh'].tobytes()==arrays.load(actual['hidden']).tobytes(),(index,'chained hidden CPU mismatch')
            row=dict(block=index,reference=ref,actual=actual,chained_ffn_matches_cpu=True,
                hidden_matches_cpu=True,hidden_clip_fraction=float(np.mean(np.abs(hidden.astype(np.float32)/sh)>127)),
                input_error=oracle.error_metrics(x,arrays.load(ref['input'])),
                mlp_error=oracle.error_metrics(mlp,arrays.load(ref['mlp'])),
                block_output_error=oracle.error_metrics(arrays.load(actual['output']),arrays.load(ref['output'])))
            report['boundaries'].append(row)
        report.update(phase='capturing',baseline_frozen_bytes_match=True);save()
        pool,stream=torch.xpu.graph_pool_handle(),torch.xpu.Stream()
        for name in ('selected','int8_p4'):
            before=len(stack.rewrite.builds)
            with route(name):
                with stream:
                    for _ in range(2):warm=compute()
                torch.xpu.synchronize()
                owned=torch.empty_like(warm,memory_format=torch.contiguous_format);del warm
                graph=torch.xpu.XPUGraph();graphs.append(graph)
                with torch.xpu.graph(graph,stream=stream,pool=pool):
                    temporary=compute();owned.copy_(temporary)
                del temporary
            graph.replay();torch.xpu.synchronize()
            saved[name]=arrays.load(report['outputs'][name]).tobytes()
            assert raw(owned)==saved[name],(name,'captured/eager mismatch')
            entries[name]=(graph,owned)
            report['builds'][name]=stack.rewrite.builds[before:]
            assert len(report['builds'][name])==3
            for build in report['builds'][name]:
                if name=='selected':
                    assert (build['triton_calls'],build['standalone_fp8'],build['quantization_calls'],build['elided_fp8'])==(651,180,395,215)
                else:
                    assert build['triton_calls']>0 and build['elided_fp8']>0
            print('Captured complete static body '+name,flush=True);save()
        segments=torch.xpu.memory_snapshot(pool)
        persistent=[*[v[1] for v in entries.values()],*[v for v in static.values() if v is not None],
                    *[t for values in candidate.packed for t in values]]
        assert not any(s['address']<=t.data_ptr()<s['address']+s['total_size'] for s in segments for t in persistent)
        report['persistent_io_and_int8_constants_outside_pool']=True
        # Each graph must read a changed live input and agree with its OWN eager
        # path. Approximate and selected paths are intentionally allowed to differ.
        front_backup=static['front'].clone();static['front'].zero_()
        try:
            for name,(graph,owned) in entries.items():
                graph.replay();torch.xpu.synchronize();changed=raw(owned)
                assert changed!=saved[name]
                with route(name):fresh=compute().cpu().numpy()
                assert changed==fresh.tobytes(),(name,'changed input eager mismatch')
                assert np.isfinite(fresh).all()
                report['changed_input_checks'].append(dict(name=name,output=arrays.save(fresh),matches_own_eager=True,differs_original=True))
        finally:static['front'].copy_(front_backup)
        # Restore and warm both graphs before sampling; no oracle/probes in timing.
        for graph,owned in entries.values():graph.replay()
        torch.xpu.synchronize()
        report.update(phase='timing');save();print('sampling_started: paired complete static bodies',flush=True)
        for repeat in range(7):
            order=['selected','int8_p4'] if repeat%2==0 else ['int8_p4','selected']
            report['orders'].append(order)
            for name in order:
                graph,owned=entries[name]
                torch.xpu.synchronize();started=time.perf_counter()
                for _ in range(20):graph.replay()
                torch.xpu.synchronize()
                report['samples_ms'][name].append((time.perf_counter()-started)*1000/20)
                assert raw(owned)==saved[name],(name,'unstable replay output')
            print('Paired static body round '+str(repeat),flush=True)
        report['median_ms']={n:statistics.median(v) for n,v in report['samples_ms'].items()}
        report['change_percent']=(report['median_ms']['int8_p4']/report['median_ms']['selected']-1)*100
        report['all_replays_match_own_eager']=True
        assert all(raw(t)==arrays.load(fixture['body_inputs'][n]).tobytes() for n,t in static.items() if t is not None)
        assert stack.model._previous is None and stack.model.next_seed==0
        stack.graph._validate();candidate.validate_constants()
        assert constant_hashes==[hashlib.sha256(raw(t)).hexdigest() for values in candidate.packed for t in values]
        report['inputs_constants_model_history_seed_unchanged']=True
    candidate.verify_restored();stack.rewrite.verify_restored()
    assert candidate.calls==selected_scope.calls==40
    assert candidate.blocks==selected_scope.blocks==dict.fromkeys(range(8),5)
    report['vit_calls']={'selected':selected_scope.calls,'int8_p4':candidate.calls}
    report['resources']=candidate.ffn_resources
    report['selections']=candidate.selections
    report['attention_projection_resources']=candidate.resources
    report['stack_metadata']=stack.rewrite.metadata()
    for resources in (candidate.ffn_resources,candidate.resources,stack.rewrite.resources):
        assert resources and all(v['spills']==0 for v in resources.values())
    sources.update(stack.rewrite.fork.sources)
    report.update(passed=True,phase='completed',bindings_restored=True)
except BaseException:
    report.update(phase='failed',error=traceback.format_exc());raise
finally:
    for graph in graphs:graph.reset()
    try:
        if candidate is not None:candidate.verify_restored()
        if stack is not None:stack.close()
        finalize_sources()
    except BaseException:
        report.update(passed=False,phase='failed',finalization_error=traceback.format_exc());raise
    finally:save()
print(json.dumps(dict(passed=True,median_ms=report['median_ms'],change_percent=report['change_percent'],
    full_nr_call_test=False,new_quantization=True)),flush=True)
