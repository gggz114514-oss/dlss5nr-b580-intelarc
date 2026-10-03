"""Whole FFN segment screen: current FP16 kernels versus fused continuous INT8.

Eight frozen real inputs checked; paired timings on blocks 0 and 7. This excludes
attention, the rest of NR, video and IO. Static calibration is not visual approval.
"""
from layout_crop_validation_env_v1 import *
import statistics,time,traceback
OUT=D/'experimental/int8-ffn-segment-gpu-v1';assert not OUT.exists()
cpu=receipt(D/'experimental/int8-ffn-segment-cpu-v1/validation.json',
    '1f1bdd147c386245dad5701407f65f5ac663a31a36647c4aeae6e11c9b799130')
assert cpu['no_gpu_execution'] and len(cpu['cases'])==16
for p in (Path(__file__),HERE/'Run-Int8FfnSegmentGpuV1.cmd',HERE/'int8_ffn_segment_gpu_v1.py'):
    sources[str(p)]=sha(p)
import numpy as np
import torch,triton
import compressed_arrays_v1 as arrays
import int8_ffn_segment_oracle_v1 as oracle
from int8_ffn_segment_gpu_v1 import Segment
from nr_backend.weights import load_pinned_records
import fast_matrices_v3 as dense
import nr_backend.triton_cubic_fp8 as cubic_module
import nr_backend.triton_fp8 as fp8_module
import fused_vit_projection_v2 as projection
from fork_fp8_jit_v2 import Fork
from spill_preflight_v1 import select
assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site')
OUT.mkdir()
report=dict(scope=__doc__,passed=False,phase='initialized',sources=sources,exact_gate=gate,
    cases=[],resources={},candidate_promoted=False,full_model_test=False,new_quantization=True,
    complete_migration=False,queue_check=queue_check,calibration_margin=1.25,
    timing_scope='All entry quantization, expansion+cubic+hidden quantization, contraction and exit handling; no IO/weight packing/JIT.',
    baseline_scope='Preallocated current selected FP16 FFN kernels: BM64/BN32 expand, ShortFP8 cubic, scaled skip, BM32/BN32 K4 projection, ordered half merge, ShortFP8 exit; frozen FP8 input.',
    candidate_variants={'int8_p1':'3 kernels, serial K4096 INT32 contraction',
        'int8_p4':'4 kernels, four INT32 partials and merged exit'},
    torch_version=torch.__version__,triton_version=triton.__version__)


def scalar(value):
    if isinstance(value,(np.bool_,np.integer,np.floating)):return value.item()
    raise TypeError(type(value).__name__)


def save():
    (OUT/'validation.json').write_text(json.dumps(report,indent=2,default=scalar,allow_nan=False)+'\n',encoding='utf-8')


raw=lambda t:t.cpu().numpy().tobytes()
to=lambda a:torch.from_numpy(np.ascontiguousarray(a)).to('xpu')
graphs=[];save()
fork=Fork(ROOT)
baseline_cubic=fork.fork(cubic_module._direct)
baseline_fp8=fork.fork(fp8_module._kernel)
sources.update(fork.sources)


def bind(label,jit,args,grid,**options):
    _,kernel,selection=select(jit,[(1,)],lambda _:args,lambda _:grid,
        num_warps=4,enable_fp_fusion=False,**options)
    report['resources'][label+':'+kernel.hash]=dict(spills=kernel.n_spills,registers=kernel.n_regs,
        shared_bytes=kernel.metadata.shared,selection=selection)
    def run():
        assert jit[grid](*args,num_warps=4,enable_fp_fusion=False,**options) is kernel
    return run


def baseline(x,wexp,wcontract,skip):
    expanded=torch.empty((64,4096),device=x.device,dtype=torch.float16)
    hidden=torch.empty_like(expanded)
    initial=torch.empty_like(x);partial=torch.empty((4,64,1024),device=x.device,dtype=x.dtype)
    merged=torch.empty_like(x);out=torch.empty_like(x)
    expand=bind('baseline_expand',dense._matmul,
        (x,wexp,x,x,x,expanded,64,4096,1024,False,False,False,64,32,32),(1,128,1))
    cubic=bind('baseline_cubic',baseline_cubic,(expanded,hidden,64*4096,512),(512,))
    contract=bind('baseline_contract',projection._parts,(hidden,wcontract,initial,partial,64,4096,1024,32,32),(2,32,4),num_stages=1)
    merge=bind('baseline_merge',projection._merge,(partial,merged,64*1024,512),(128,))
    quantize=bind('baseline_fp8',baseline_fp8,(merged,out,64*1024,512),(128,))
    def run():
        expand();cubic();torch.mul(x,skip,out=initial);contract();merge();quantize()
        return out
    return run,out,hidden


def assert_array(t,expected,label):
    actual=t.cpu().numpy()
    assert actual.shape==expected.shape and actual.dtype==expected.dtype,(label,'shape/dtype')
    assert actual.tobytes()==expected.tobytes(),(label,'byte mismatch',int(np.count_nonzero(actual!=expected)))


def run_case(index,records):
    spec=next(c for c in cpu['cases'] if c['name']==f'vit.{index}.ffn' and c['margin']==1.25)
    record=lambda layer:records[f'block{31+index}.layer{layer}.layer']
    x_cpu=arrays.load(spec['reference_input']);hidden_ref=arrays.load(spec['reference_hidden'])
    baseline_ref=arrays.load(spec['reference_mlp']);cpu_out=arrays.load(spec['mlp_output'])
    cpu_raw=arrays.load(spec['mlp_unquantized']);sh=arrays.load(spec['hidden_scale'])
    wexp=oracle.decode_weights(record(0)[:4194304],1024,4096)
    wcontract=oracle.decode_weights(record(1)[:4194304],4096,1024)
    skip=np.frombuffer(record(1)[4194304:],dtype=np.float16).copy()
    initial=(x_cpu.astype(np.float32)*skip.astype(np.float32)).astype(np.float16)
    hidden_cpu,entry=oracle.expansion(x_cpu,wexp)
    check_raw,check_out,packed=oracle.contraction(hidden_cpu,sh,wcontract,initial)
    assert check_raw.tobytes()==cpu_raw.tobytes() and check_out.tobytes()==cpu_out.tobytes()
    x=to(x_cpu);skip_gpu=to(skip);wexp_gpu=to(wexp);wcontract_gpu=to(wcontract)
    constants=[to(a) for a in (entry['qw'].T,entry['sw'].reshape(-1),sh.reshape(-1),packed['qw'].T,packed['sw'].reshape(-1))]
    saved_constants=[raw(t) for t in (skip_gpu,wexp_gpu,wcontract_gpu,*constants)]
    run_base,base_out,base_hidden=baseline(x,wexp_gpu,wcontract_gpu,skip_gpu)
    segment=Segment(x,*constants,skip_gpu)
    row=dict(name=spec['name'],timed=index in (0,7),cpu_reference=spec['mlp_output'],
        half_cpu_reference=spec['mlp_unquantized'],baseline_reference=spec['reference_mlp'],
        resources=segment.resources,selections=segment.selections,
        all_candidate_boundaries_match_cpu=False,samples_ms={v:[] for v in ('selected','int8_p1','int8_p4')},orders=[])
    report['cases'].append(row);report.update(phase='validating',active_case=row['name']);save()
    run_base();torch.xpu.synchronize()
    assert_array(base_hidden,hidden_ref,'current FP16 hidden');assert_array(base_out,baseline_ref,'current FP16 MLP')
    for parts in (1,4):
        segment.run(parts,True);torch.xpu.synchronize()
        assert_array(segment.qx,entry['qx'],'entry INT8');assert_array(segment.sx,entry['sx'].reshape(-1),'entry scale')
        assert_array(segment.hidden_raw,hidden_cpu,'cubic half');assert_array(segment.qh,packed['qh'],'INT8 hidden')
        assert_array(segment.raw_outputs[parts],cpu_raw,f'p{parts} half exit')
        assert_array(segment.outputs[parts],cpu_out,f'p{parts} FP8 exit')
        segment.run(parts);torch.xpu.synchronize()
        assert_array(segment.outputs[parts],cpu_out,f'p{parts} non-debug exit')
        assert_array(segment.qh,packed['qh'],f'p{parts} non-debug hidden')
    row.update(all_candidate_boundaries_match_cpu=True,baseline_boundaries_match_frozen=True,
        candidate_output=arrays.save(segment.outputs[1].cpu().numpy()),
        error_vs_selected=oracle.error_metrics(cpu_out,baseline_ref),
        selected_hidden_bytes=base_hidden.numel()*base_hidden.element_size(),
        candidate_hidden_bytes=segment.qh.numel()*segment.qh.element_size(),
        split_int32_partial_bytes=segment.partial.numel()*segment.partial.element_size())
    variants={'selected':(run_base,base_out),'int8_p1':(lambda:segment.run(1),segment.outputs[1]),
        'int8_p4':(lambda:segment.run(4),segment.outputs[4])}
    original=x.clone()
    # Zero-entry control uses fresh dynamic row scales; both INT32 partitions agree.
    x.zero_()
    for fn,_ in variants.values():fn()
    torch.xpu.synchronize()
    assert raw(base_out)==raw(segment.outputs[1])==raw(segment.outputs[4])
    assert not torch.count_nonzero(segment.qx).item() and raw(segment.sx)==np.ones(64,dtype=np.float32).tobytes()
    x.copy_(original);torch.xpu.synchronize()
    if row['timed']:
        entries={};pool=torch.xpu.graph_pool_handle();stream=torch.xpu.Stream();nodes=16
        saved={v:baseline_ref.tobytes() if v=='selected' else cpu_out.tobytes() for v in variants}
        for variant in ('int8_p4','selected','int8_p1'):
            fn,out=variants[variant]
            for _ in range(3):fn()
            torch.xpu.synchronize()
            graph=torch.xpu.XPUGraph();graphs.append(graph)
            with torch.xpu.graph(graph,stream=stream,pool=pool):
                for _ in range(nodes):fn()
            graph.replay();torch.xpu.synchronize();assert raw(out)==saved[variant]
            entries[variant]=(graph,out)
        x.zero_()
        for graph,out in entries.values():graph.replay()
        torch.xpu.synchronize()
        assert raw(base_out)==raw(segment.outputs[1])==raw(segment.outputs[4])!=saved['selected']
        x.copy_(original);torch.xpu.synchronize()
        report.update(phase='timing',active_case=row['name']);save()
        print(json.dumps(dict(phase='sampling_started',case=row['name'])),flush=True)
        labels=list(variants)
        for repeat in range(7):
            order=labels[repeat%3:]+labels[:repeat%3]
            if repeat%2:order=order[::-1]
            row['orders'].append(order)
            for variant in order:
                graph,out=entries[variant]
                torch.xpu.synchronize();started=time.perf_counter()
                for _ in range(20):graph.replay()
                torch.xpu.synchronize()
                row['samples_ms'][variant].append((time.perf_counter()-started)*1000/(20*nodes))
                assert raw(out)==saved[variant]
        row['median_ms']={v:statistics.median(samples) for v,samples in row['samples_ms'].items()}
        row['change_percent']={v:(row['median_ms'][v]/row['median_ms']['selected']-1)*100 for v in ('int8_p1','int8_p4')}
        row['captured_complete_segments']=nodes
        for graph,_ in entries.values():graph.reset();graphs.remove(graph)
        row['live_input_graph_checks']=True
    assert raw(x)==x_cpu.tobytes()
    assert [raw(t) for t in (skip_gpu,wexp_gpu,wcontract_gpu,*constants)]==saved_constants
    row.update(operands_unchanged=True,zero_entry_control=True)
    save();print(json.dumps(dict(case=row['name'],median_ms=row.get('median_ms'),change_percent=row.get('change_percent'))),flush=True)


try:
    torch.set_num_threads(2)
    records=load_pinned_records(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin')
    with torch.inference_mode():
        for index in range(8):run_case(index,records)
    assert len(report['cases'])==8 and sum(c['timed'] for c in report['cases'])==2
    report.update(passed=True,phase='completed')
except BaseException:
    report.update(phase='failed',error=traceback.format_exc());raise
finally:
    for graph in graphs:graph.reset()
    try:fork.verify_restored();finalize_sources()
    except BaseException:
        report.update(passed=False,phase='failed',finalization_error=traceback.format_exc());raise
    finally:save()
print(json.dumps(dict(passed=True,cases=8,timed_cases=2,full_model_test=False)),flush=True)
