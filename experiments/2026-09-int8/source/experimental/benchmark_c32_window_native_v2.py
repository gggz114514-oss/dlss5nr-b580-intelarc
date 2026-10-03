"""Screen native-cubic C32 whole-window fusion against the ShortFP8 stack.

Primitive graph timings are not complete NR speed. Only zero-spill candidates
are dispatched. Full fixture bytes and changed input are checked. A losing first
module ends the screen; meaningful wins qualify for subsequent broader testing.
"""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request,urllib.error
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580';R=EXACT/'reference'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'experimental/c32-window-native-v2'
assert not OUT.exists()
TOOLCHAIN=D/'toolchains/triton-xpu-3.8.0-git1e2d42a0'
sys.path[:0]=[str(TOOLCHAIN/'site'),str(R),str(ROOT/'backend')]
os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-c32-triton38-v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main()
audit_path=D/'experimental/short-fp8-checkpoint-v1/saved-audit-v1.json'
assert sha(audit_path)=='1413f26f3269ed8b7379763077314f3090de1467b76309d1af0ff063bb4db085'
audit=js(audit_path);assert audit['passed']
sources=dict(audit['sources']);sources[str(audit_path)]=sha(audit_path)
for p in audit['new_files']:sources[str(ROOT/p)]=audit['new_files'][p]
fixture_path=D/'experimental/c32-window-blocks-v2/validation.json'
assert sha(fixture_path)=='b99326949733fe839431341cc743ea5903abda8f95ef3ca795b23ed0f2ca08cd'
fixture=js(fixture_path);assert fixture['passed']
sources.update(fixture['sources']);sources[str(fixture_path)]=sha(fixture_path)
for p in (Path(__file__),HERE/'fused_c32_window_native_v1.py',HERE/'Run-C32WindowNativeV2.cmd'):sources[str(p)]=sha(p)
assert all(sha(p)==h for p,h in sources.items())
provision=TOOLCHAIN/'provision-v1.json'
assert sha(provision)=='e94757afcb5962cacbe208cbb808b3516de210059335f1ee7ab7b40d6bfe2c07'
assert all(sha(TOOLCHAIN/'site'/p)==h for p,h in js(provision)['files'].items())
try:
    with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=3) as response:queue=json.load(response)
    assert not queue['queue_running'] and not queue['queue_pending']
    queue_check='running_and_empty'
except urllib.error.URLError as error:
    assert getattr(error.reason,'winerror',None)==10061 or getattr(error.reason,'errno',None)==10061
    queue_check='connection_refused_service_not_running'
import numpy as np
import torch
import triton
import fused_c32_window_native_v1 as candidate
from nr256_selected_stack_v3 import Stack
from nr_backend.execution import use_arithmetic_backend
from cubic_lut_constant_v1 import Constant
from fp8_graph_rewrite_v1 import RewritingDataflow
from quantization_dataflow_v1 import FP8
import compressed_arrays_v1 as arrays
import immutable_artifacts_v1 as artifacts
assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site')
OUT.mkdir();report=dict(scope=__doc__,passed=False,complete_migration=False,candidate_promoted=False,
    sources=sources,exact_gate=gate,queue_check=queue_check,cases=[],selected_commit='0b6f201db1022690fd18aa5cd1b9c47a05593765')
save=lambda:(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
raw=lambda t:t.cpu().numpy().tobytes()
stack=None;graphs=[]
try:
    torch.set_num_threads(2);stack=Stack(EXACT);sources.update(stack.rewrite.fork.sources)
    lut=Constant(stack.model).require();modules=dict(stack.model.named_modules())
    # Complete CPU domain proof for fixture only, using exhaustive GPU rounder table.
    scalar=js(D/'experimental/short-fp8-body-v2/validation.json')['exhaustive']
    mapping=arrays.load(scalar['old']).view('u2')
    with torch.inference_mode(),stack.installed(),stack.rewrite.fork.installed(),use_arithmetic_backend('triton'):
        for spec in fixture['cases']:
            meta=spec['arrays'];module=modules[spec['name']]
            ts={n:torch.from_numpy(arrays.load(m)).to('xpu') for n,m in meta.items() if n!='output'}
            x=ts['features'];expected=arrays.load(meta['output']).tobytes()
            words=arrays.load(meta['features']).view('u2');assert np.array_equal(mapping[words],words)
            builds=[]
            def baseline():
                df=RewritingDataflow()
                df.events.append(dict(name='complete_fixture_fp8_table_proof',kind='fixture',id=0))
                df.state(x).update(domain=FP8,producer=0)
                with df.installed():result=module.forward_unquantized(x)
                builds.append(df.rewrite_summary())
                return result
            assert raw(baseline())==expected
            args=[ts[n] for n in ('features','expansion','contraction','mlp_scale','qkv','q_scale','order','bias','projection','skip_scale')]+[lut]
            row=dict(name=spec['name'],shape=spec['shape'],shift=spec['shift'],candidates=[],arrays=meta)
            report['cases'].append(row);functions={'selected':baseline}
            for warps,stages in ((4,1),(8,1),(4,2),(8,2)):
                record=dict(warps=warps,stages=stages);row['candidates'].append(record)
                try:value,kernel=candidate.forward(*args,shift=spec['shift'],warps=warps,stages=stages)
                except candidate.ResourceRejected as error:
                    record.update(rejected_before_dispatch=True,resources=error.resources);save();print(json.dumps(record),flush=True);continue
                actual=value.cpu().numpy();record.update(rejected_before_dispatch=False,byte_equal=actual.tobytes()==expected,
                    resources=dict(spills=kernel.n_spills,registers=kernel.n_regs,shared_bytes=kernel.metadata.shared),
                    different_half_words=int(np.count_nonzero(actual.view('u2')!=np.frombuffer(expected,'u2').reshape(actual.shape))),
                    output=arrays.save(actual),ir={k:artifacts.text(v,k) for k,v in kernel.asm.items() if k in ('ttgir','llir') and isinstance(v,str)})
                save();print(json.dumps({k:v for k,v in record.items() if k not in ('ir','output')}),flush=True)
                if record['byte_equal']:
                    functions[f'w{warps}s{stages}']=lambda warps=warps,stages=stages:candidate.forward(*args,shift=spec['shift'],warps=warps,stages=stages)[0]
            entries={};pool=torch.xpu.graph_pool_handle();stream=torch.xpu.Stream()
            for name,fn in functions.items():
                with stream:
                    for _ in range(2):warm=fn()
                torch.xpu.synchronize();owned=torch.empty_like(warm);del warm
                g=torch.xpu.XPUGraph();graphs.append(g)
                with torch.xpu.graph(g,stream=stream,pool=pool):
                    temporary=fn();owned.copy_(temporary)
                del temporary
                entries[name]=(g,owned)
            segments=torch.xpu.memory_snapshot(pool)
            assert not any(s['address']<=t.data_ptr()<s['address']+s['total_size'] for s in segments for t in [x,*[v[1] for v in entries.values()]])
            samples={n:[] for n in entries};row['orders']=[]
            for repeat in range(7):
                names=list(entries);offset=repeat%len(names);order=names[offset:]+names[:offset];row['orders'].append(order)
                for name in order:
                    g,owned=entries[name];torch.xpu.synchronize();start=time.perf_counter()
                    for _ in range(30):g.replay()
                    torch.xpu.synchronize();samples[name].append((time.perf_counter()-start)/30)
                    assert raw(owned)==expected
            # Another FP8-domain input must yield fresh, identical full results.
            x.zero_();changed={}
            for name,(g,owned) in entries.items():
                g.replay();torch.xpu.synchronize();changed[name]=raw(owned)
            assert all(v==changed['selected'] for v in changed.values()) and changed['selected']!=expected
            x.copy_(torch.from_numpy(arrays.load(meta['features'])).to('xpu'))
            row.update(samples_seconds=samples,median_ms={n:statistics.median(v)*1000 for n,v in samples.items()},
                changed_input_all_bytes_equal=True,persistent_io_outside_pool=True,baseline_builds=builds)
            assert all(raw(t)==arrays.load(meta[n]).tobytes() for n,t in ts.items())
            for g,_ in entries.values():g.reset();graphs.remove(g)
            print(json.dumps(dict(name=spec['name'],median_ms=row['median_ms'])),flush=True);save()
            candidates=[v for n,v in row['median_ms'].items() if n!='selected']
            if not candidates or min(candidates)>row['median_ms']['selected']*.95:
                report['screen_stopped']='No at-least-5-percent primitive win; do not broaden a losing screen.'
                break
        assert stack.model._previous is None and stack.model.next_seed==0
    stack.rewrite.fork.verify_restored()
    report.update(passed=True,screen_only=True,bindings_restored=True,inputs_unchanged=True)
except BaseException:
    report['error']=traceback.format_exc();raise
finally:
    for g in graphs:g.reset()
    if stack is not None:stack.close()
    try:assert all(sha(p)==h for p,h in sources.items());authenticate_main()
    except BaseException:report.update(passed=False,finalization_error=traceback.format_exc());raise
    finally:save()
print(json.dumps(dict(passed=report['passed'],cases=len(report['cases']),stop=report.get('screen_stopped'))),flush=True)
