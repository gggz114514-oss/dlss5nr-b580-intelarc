"""Local mixed INT8/FP16 projection screen, not full-body or model performance.

Preserves operand values with exact binary scales. Record half/FP8 boundary
differences instead of declaring accumulator parity from representation alone.
Time two real C512 projections with dynamic classification/conversion included;
static weight packing, outer FP8 rounding and IO are outside this primitive.
"""
from layout_crop_validation_env_v1 import *
import statistics,time,traceback
OUT=D/'experimental/fp8-partial-int8-projection-v3';assert not OUT.exists()
analysis=receipt(D/'experimental/fp8-int8-tiles-v1/validation.json',
    'f24f42e6313e24acb1d67260f50e1b8da7500e3f5bf86117485e19769ec0cf6e')
assert len(analysis['cases'])==96 and analysis['no_gpu_execution']
boundaries=receipt(D/'experimental/c512-window-layout-body-v1/validation.json',
    'd538d84f920fd39b31403d49996b6eceb02f4c0576a33cd670309a5d11b66268')
for p in (Path(__file__),HERE/'Run-Fp8PartialInt8ProjectionV3.cmd',HERE/'fp8_partial_int8_projection_v2.py'):
    sources[str(p)]=sha(p)
import numpy as np
import torch,triton
from nr_backend.weights import load_pinned_records
from nr_backend.split_block import SplitSwinBlock
from nr_backend.triton_fp8 import quantize_fp8
import fast_matrices_v3 as ordinary
import fp8_partial_int8_projection_v2 as mixed
from spill_preflight_v1 import select
import compressed_arrays_v1 as arrays
assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site')
OUT.mkdir()
report=dict(scope=__doc__,sources=sources,exact_gate=gate,passed=False,complete_migration=False,
    candidate_promoted=False,cases=[],resources={},queue_check=queue_check,
    weights_prepacked_once=True,dynamic_activation_conversion_timed=True,no_zero_skipping=True,
    torch_version=torch.__version__,triton_version=triton.__version__)
def json_scalar(value):
    if isinstance(value,(np.bool_,np.integer,np.floating)):return value.item()
    raise TypeError(f'Unsupported report value: {type(value).__name__}')


def save():
    payload=json.dumps(report,indent=2,default=json_scalar,allow_nan=False)+'\n'
    (OUT/'validation.json').write_text(payload,encoding='utf-8')
raw=lambda t:t.cpu().numpy().tobytes()
graphs=[]
report['phase']='initialized';save()


def pack_weight(value):
    k,n=value.shape;u=(value.astype(np.float32)*512).astype(np.int32)
    assert np.array_equal(u.astype(np.float32),value.astype(np.float32)*512)
    ored=np.bitwise_or.reduce(np.abs(u.reshape(k//32,32,n)),axis=1)
    factor=np.maximum(ored&-ored,1)
    shifts=np.log2(factor).astype(np.int32)
    q=u.reshape(k//32,32,n)>>shifts[:,None,:]
    assert q.min()>=-128 and q.max()<=127
    assert np.array_equal(q<<shifts[:,None,:],u.reshape(k//32,32,n))
    return q.astype(np.int8).reshape(k,n),(factor.astype(np.float32)/512).copy()


def expected_flags(value):
    m,k=value.shape
    padded=np.pad(value,((0,(-m)%16),(0,0)))
    scaled=padded.astype(np.float32)*512;u=scaled.astype(np.int32)
    blocks=u.reshape(-1,k//32,32)
    ored=np.bitwise_or.reduce(np.abs(blocks),axis=2)
    shifts=np.log2(np.maximum(ored&-ored,1)).astype(np.int32)
    q=blocks>>shifts[:,:,None]
    fits=(q.min(axis=2)>=-128)&(q.max(axis=2)<=127)
    fits&=(scaled==u.astype(np.float32)).reshape(-1,k//32,32).all(axis=2)
    return fits.reshape(-1,16,k//32).all(axis=1).astype(np.int8)


def resource(label,kernel):
    kernel._init_handles()
    assert isinstance(kernel.n_spills,int) and kernel.n_spills==0,(label,kernel.n_spills)
    report['resources'][label+':'+kernel.hash]=dict(spills=kernel.n_spills,registers=kernel.n_regs,
        shared_bytes=kernel.metadata.shared)


def difference(a,b):
    aa=a.cpu().numpy();bb=b.cpu().numpy()
    assert aa.shape==bb.shape and aa.dtype==bb.dtype==np.dtype('f2')
    assert np.isfinite(aa).all() and np.isfinite(bb).all()
    changed=int(np.count_nonzero(aa.view(np.uint16)!=bb.view(np.uint16)))
    delta=aa.astype(np.float64)-bb.astype(np.float64)
    return dict(byte_equal=changed==0,different_half_words=int(changed),total_half_words=aa.size,
        max_absolute_error=float(np.abs(delta).max()),rmse=float(np.sqrt(np.mean(delta*delta))))


def run_case(name,a_cpu,w_cpu,initial_cpu,*,expected=None,timed=False,required_flags=None):
    m,k=a_cpu.shape;n=w_cpu.shape[1]
    qw,sw=pack_weight(w_cpu)
    to=lambda value:torch.from_numpy(np.ascontiguousarray(value)).to('xpu')
    a,w,w8,scales,initial=map(to,(a_cpu,w_cpu,qw,sw,initial_cpu))
    selected=torch.empty((m,n),device='xpu',dtype=torch.float16)
    candidate=torch.empty_like(selected)
    flags=torch.full((triton.cdiv(m,16),k//32),-1,device='xpu',dtype=torch.int8)
    baseline_args=(a,w,a,a,initial,selected,m,n,k,True,False,False,16,32,32)
    grid=(triton.cdiv(m,16),triton.cdiv(n,32),1)
    _,base_kernel,base_selection=select(ordinary._matmul,[(16,32)],lambda _:baseline_args,lambda _:grid,
        num_warps=4,enable_fp_fusion=False)
    def base_launch():return ordinary._matmul[grid](*baseline_args,num_warps=4,enable_fp_fusion=False)
    debug_args,debug_grid,debug_kernel,debug_selection=mixed.prepare(a,w,w8,scales,initial,candidate,flags,debug=True)
    args,mix_grid,mix_kernel,mix_selection=mixed.prepare(a,w,w8,scales,initial,candidate,flags)
    for label,kernel in (('baseline',base_kernel),('debug',debug_kernel),('mixed',mix_kernel)):resource(label,kernel)
    assert base_launch() is base_kernel
    assert mixed.launch(debug_args,debug_grid) is debug_kernel
    torch.xpu.synchronize()
    actual_flags=flags.cpu().numpy()
    cpu_flags=expected_flags(a_cpu)
    assert np.array_equal(actual_flags,cpu_flags),'GPU branch classification differs from CPU'
    if required_flags is not None:assert (actual_flags==required_flags).all()
    debug_bytes=raw(candidate)
    assert mixed.launch(args,mix_grid) is mix_kernel
    torch.xpu.synchronize()
    assert raw(candidate)==debug_bytes,'Debug and measured kernels differ'
    half_check=difference(candidate,selected)
    selected_q,candidate_q=quantize_fp8(selected),quantize_fp8(candidate)
    fp8_check=difference(candidate_q,selected_q)
    if expected is not None:assert raw(selected_q)==expected.tobytes(),'Baseline differs from frozen FFWD boundary'
    if required_flags==0:assert half_check['byte_equal'],'All-FP16 fallback changed arithmetic'
    row=dict(name=name,shape=[m,k,n],timed=timed,half=half_check,fp8=fp8_check,
        gpu_branch_flags_match_cpu=True,int8_row_k32_tiles=int(actual_flags.sum()),
        fp16_row_k32_tiles=int(actual_flags.size-actual_flags.sum()),
        int8_dot_programs=int(actual_flags.sum())*triton.cdiv(n,32),
        fp16_dot_programs=int(actual_flags.size-actual_flags.sum())*triton.cdiv(n,32),
        branch_counts_include_zero_products=True,static_weight_exact=True,
        baseline_selection=base_selection,candidate_selection=mix_selection,debug_selection=debug_selection,
        baseline_output=arrays.save(selected.cpu().numpy()),candidate_output=arrays.save(candidate.cpu().numpy()),
        samples_ms={'selected':[],'mixed':[]},orders=[])
    report['cases'].append(row);save()
    if timed:
        report.update(phase='timing',active_case=name);save()
        print(json.dumps(dict(phase='timing_started',case=name)),flush=True)
        saved={'selected':raw(selected),'mixed':raw(candidate)}
        pool=torch.xpu.graph_pool_handle();stream=torch.xpu.Stream();entries={}
        nodes=64
        for variant in ('mixed','selected'):
            invoke=base_launch if variant=='selected' else lambda:mixed.launch(args,mix_grid)
            for _ in range(3):invoke()
            torch.xpu.synchronize()
            graph=torch.xpu.XPUGraph();graphs.append(graph)
            with torch.xpu.graph(graph,stream=stream,pool=pool):
                for _ in range(nodes):invoke()
            graph.replay();torch.xpu.synchronize()
            out=selected if variant=='selected' else candidate
            assert raw(out)==saved[variant]
            entries[variant]=(graph,out)
        # Both graphs must read changed live input, then reproduce original bytes.
        a.zero_()
        for graph,out in entries.values():graph.replay()
        torch.xpu.synchronize()
        assert raw(selected)==raw(candidate) and raw(selected)!=saved['selected']
        a.copy_(to(a_cpu));torch.xpu.synchronize()
        for repeat in range(7):
            order=['selected','mixed'] if repeat%2==0 else ['mixed','selected']
            row['orders'].append(order)
            for variant in order:
                graph,out=entries[variant]
                torch.xpu.synchronize();started=time.perf_counter()
                for _ in range(20):graph.replay()
                torch.xpu.synchronize()
                row['samples_ms'][variant].append((time.perf_counter()-started)*1000/(20*nodes))
                assert raw(out)==saved[variant]
        row['median_ms']={v:statistics.median(samples) for v,samples in row['samples_ms'].items()}
        row['change_percent']=(row['median_ms']['mixed']/row['median_ms']['selected']-1)*100
        row['captured_kernel_nodes']=nodes
        for graph,_ in entries.values():graph.reset();graphs.remove(graph)
    for actual,value in ((a,a_cpu),(w,w_cpu),(w8,qw),(scales,sw),(initial,initial_cpu)):
        assert raw(actual)==np.ascontiguousarray(value).tobytes()
    row['operands_unchanged']=True
    save();print(json.dumps(dict(case=name,half=half_check,fp8=fp8_check,median_ms=row.get('median_ms'),change_percent=row.get('change_percent'))),flush=True)


try:
    torch.set_num_threads(2)
    records=load_pinned_records(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin')
    fixtures=[]
    report['fixture_boundaries']=[]
    for index in (0,1):
        row=next(r for r in boundaries['boundaries'] if r['block']==f'encoder512.{index}')
        report['fixture_boundaries'].append(dict(block=row['block'],input=row['input'],
            ffwd=row['reference'][0],mlp=row['reference'][1]))
        block=SplitSwinBlock([records[f'block{23+index}.layer{layer}.layer'] for layer in range(4)])
        a=arrays.load(row['reference'][0]).reshape(144,512)
        residual=arrays.load(row['input']).reshape(144,512)
        weight=block.ffwd_projection.weight.numpy().copy()
        scale=block.ffwd_projection.skip_scale.numpy().copy()
        initial=(residual.astype(np.float32)*scale.astype(np.float32)).astype(np.float16)
        fixtures.append((a,weight,initial,arrays.load(row['reference'][1]).reshape(144,512)))
    with torch.inference_mode():
        # Controls precede real timings: whole-FP16 fallback and whole-INT8 route.
        a,w,initial,_=fixtures[0]
        non_fp8=np.full(a.shape,np.float16(0.003123),dtype=np.float16)
        run_case('non_fp8_fallback_control',non_fp8,w,initial,required_flags=0)
        small=np.resize(np.array([0,1,-1,0.5,-0.5,2,-2],dtype=np.float16),a.shape)
        run_case('all_int8_control',small,w,initial,required_flags=1)
        extreme=a.copy();extreme[0,0]=448;extreme[0,1]=np.float16(1/512);extreme[1,2]=np.float16(-0.0)
        run_case('mixed_extreme_control',extreme,w,initial)
        for i,(a,w,initial,expected) in enumerate(fixtures):
            run_case(f'encoder512.{i}.ffwd_projection',a,w,initial,expected=expected,timed=True)
    report['all_half_byte_equal']=all(c['half']['byte_equal'] for c in report['cases'])
    report['all_fp8_byte_equal']=all(c['fp8']['byte_equal'] for c in report['cases'])
    report['real_fp8_byte_equal']=all(c['fp8']['byte_equal'] for c in report['cases'] if c['timed'])
    report.update(passed=True,phase='completed')
except BaseException:
    report.update(phase='failed',error=traceback.format_exc());raise
finally:
    for graph in graphs:graph.reset()
    try:finalize_sources()
    except BaseException:
        report.update(passed=False,finalization_error=traceback.format_exc());raise
    finally:save()
print(json.dumps(dict(passed=True,all_half_byte_equal=report['all_half_byte_equal'],
    real_fp8_byte_equal=report['real_fp8_byte_equal'],full_model_test=False)),flush=True)
