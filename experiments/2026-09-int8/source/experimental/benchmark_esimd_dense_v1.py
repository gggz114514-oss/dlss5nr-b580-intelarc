"""B580 ESIMD operand-layout feasibility on captured NR matrices.

All complete outputs checked against captured fast FP16 arithmetic. Graph replay
must observe changed inputs. Includes dynamic activation packing; prepacked-only
numbers are diagnostic. No whole-model speedup or NVIDIA-exact claim.
"""
import hashlib,json,os,sys,time,statistics,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580';R=EXACT/'reference'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'experimental/esimd-dense-v1';assert not OUT.exists()
TC=D/'toolchains/triton-xpu-3.8.0-git1e2d42a0'
os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-c32-triton38-v1')
sys.path[:0]=[str(TC/'site'),str(R),str(ROOT/'backend')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main()
provision=TC/'provision-v1.json';assert sha(provision)=='e94757afcb5962cacbe208cbb808b3516de210059335f1ee7ab7b40d6bfe2c07'
assert all(sha(TC/'site'/p)==h for p,h in js(provision)['files'].items())
capture_path=D/'experimental/current-dense-operands-v2/validation.json'
assert sha(capture_path)=='46a16ba64e9f84425928d0a0fd93862e99574910b031ba098c789f0ca69e7290'
capture=js(capture_path);assert capture['passed'] and all(sha(p)==h for p,h in capture['sources'].items())
assert js(capture_path.parent.with_suffix('.log.lease.json'))['returncode']==0
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:queue=json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
import numpy as np
import torch
import triton
assert Path(triton.__file__).is_relative_to(TC/'site') and triton.__version__.startswith('3.8.0')
from esimd_dense_adapter_v1 import Dense
from current_dense_tiled_provider_v1 import POLICY
from dense_tiles_v1 import dot
import compressed_arrays_v1 as arrays
dll=D/'experimental/esimd-dense-build-v3/esimd_dense_v3.dll'
sources=dict(capture['sources'])
for p in (Path(__file__),HERE/'Run-EsimdDenseV1.cmd',HERE/'esimd_dense_adapter_v1.py',HERE/'esimd_dense_v2.cpp',
          HERE/'Build-EsimdDenseV3.cmd',HERE/'current_dense_tiled_provider_v1.py',HERE/'dense_tiles_v1.py',dll,
          dll.parent/'build.log',capture_path,provision):sources[str(p)]=sha(p)
OUT.mkdir();report=dict(scope=__doc__,passed=False,complete_migration=False,sources=sources,exact_gate=gate,cases=[],smoke=[])
save=lambda:(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
raw=lambda t:t.cpu().numpy().tobytes()
entries={}
try:
    torch.set_num_threads(2)
    with torch.inference_mode():
        engine=Dense(dll);report['resources']={str(k):v for k,v in engine.resources.items()}
        print(json.dumps(dict(resources=report['resources'])),flush=True)
        accepted=[p for p in (False,True) if engine.resources[p]['spill_bytes']==0 and engine.resources[p]['max_group']>=16]
        report['resource_accepted']=accepted
        assert accepted,'Both ESIMD kernels rejected before launch'
        rng=np.random.default_rng(751032)
        for m,k,n in ((8,16,16),(9,32,32),(31,128,64)):
            a=torch.from_numpy(rng.uniform(-1,1,(m,k)).astype('f2')).to('xpu')
            w=rng.uniform(-1,1,(k,n)).astype('f2');wt=torch.from_numpy(w).to('xpu');b=engine.pack_weight(w)
            initial=torch.from_numpy(rng.uniform(-.5,.5,(m,n)).astype('f2')).to('xpu')
            expected=dot(a,wt,initial=initial)[0];record=dict(shape=[m,k,n],routes={})
            for packed in accepted:
                av=engine.pack_activation(a) if packed else a;out=torch.empty_like(expected)
                engine.into(av,b,initial,out,m=m,k=k,n=n,packed=packed)
                equal=raw(out)==raw(expected)
                record['routes'][str(packed)]=dict(byte_equal=equal,actual=arrays.save(out.cpu().numpy()),reference=arrays.save(expected.cpu().numpy()))
            report['smoke'].append(record);save()
        report['smoke_byte_equal']=all(v['byte_equal'] for r in report['smoke'] for v in r['routes'].values())
        assert report['smoke_byte_equal'],'ESIMD layout/reduction does not match baseline'
        for key,old in capture['cases'].items():
            if not all(old['layout']):continue
            m,k,n=old['shape'];cpu={name:None if meta is None else arrays.load(meta) for name,meta in old['operands'].items()}
            if k%16 or n%16:continue
            operands={name:None if v is None else torch.from_numpy(v.copy()).to('xpu') for name,v in cpu.items()}
            a=operands['a'].reshape(m,k);w=operands['w'];initial=operands['initial']
            if initial is not None:initial=initial.reshape(m,n)
            b=engine.pack_weight(cpu['w']);packed_a=engine.pack_activation(a)
            tile=POLICY.get((m,k,n,initial is not None,tuple(old['layout'])),(16,32,32,4))
            expected=arrays.load(old['output']).tobytes();baseline=lambda:dot(a,w,initial=initial,tile=tile)[0]
            assert raw(baseline())==expected
            row=dict(key=key,shape=[m,k,n],operands=old['operands'],expected=old['output'],count=old['count'],tile=tile,routes={},samples_ms={},orders=[])
            report['cases'].append(row)
            route_functions={'triton':baseline}
            held=[]
            for p in accepted:
                out=torch.empty((m,n),dtype=torch.float16,device='xpu');held.append(out)
                if not p:route_functions['esimd_direct_a']=lambda out=out:engine.into(a,b,initial,out,m=m,k=k,n=n,packed=False)
                else:
                    route_functions['esimd_prepacked_only']=lambda out=out:engine.into(packed_a,b,initial,out,m=m,k=k,n=n,packed=True)
                    combined=torch.empty_like(out);held.append(combined)
                    route_functions['esimd_with_pack']=lambda out=combined:engine.into(engine.pack_activation(a),b,initial,out,m=m,k=k,n=n,packed=True)
            entries={}
            for name,fn in route_functions.items():
                value=fn();equal=raw(value)==expected
                row['routes'][name]=dict(byte_equal=equal)
                if not equal:
                    row['routes'][name]['actual']=arrays.save(value.cpu().numpy());continue
                stream=torch.xpu.Stream()
                with stream:
                    for _ in range(2):value=fn()
                torch.xpu.synchronize();public=torch.empty_like(value);graph=torch.xpu.XPUGraph()
                with torch.xpu.graph(graph,stream=stream):
                    for _ in range(4):value=fn()
                    public.copy_(value)
                entries[name]=(graph,public);row['samples_ms'][name]=[]
                graph.replay();torch.xpu.synchronize();assert raw(public)==expected
                # Capture must contain the native work, not merely a stale output copy.
                a.zero_();packed_a.zero_();graph.replay();torch.xpu.synchronize()
                changed_expected=baseline();assert raw(public)==raw(changed_expected)
                a.copy_(torch.from_numpy(cpu['a'].copy()).to('xpu').reshape(m,k))
                packed_a.copy_(engine.pack_activation(a));graph.replay();torch.xpu.synchronize();assert raw(public)==expected
                row['routes'][name].update(graph_captures_native_work=True,restored_input_replay=True)
            names=list(entries)
            for round_index in range(5):
                shift=round_index%len(names);order=names[shift:]+names[:shift]
                if round_index%2:order.reverse()
                row['orders'].append(order)
                for name in order:
                    graph,value=entries[name];torch.xpu.synchronize();start=time.perf_counter()
                    for _ in range(15):graph.replay()
                    torch.xpu.synchronize();row['samples_ms'][name].append((time.perf_counter()-start)*1000/60)
                    assert raw(value)==expected
            row['median_ms']={name:statistics.median(samples) for name,samples in row['samples_ms'].items()}
            row['inputs_unchanged']=all(v is None or raw(operands[name])==v.tobytes() for name,v in cpu.items())
            assert row['inputs_unchanged']
            for graph,value in entries.values():graph.reset()
            entries={};save();print(json.dumps(dict(shape=row['shape'],median_ms=row['median_ms'],correctness=row['routes'])),flush=True)
        report['passed']=True
except Exception as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    for graph,value in entries.values():graph.reset()
    try:
        assert all(sha(p)==h for p,h in sources.items());authenticate_main()
    except Exception as error:report.update(passed=False,finalization_error=repr(error));raise
    finally:save()
print(json.dumps(dict(passed=report['passed'],cases=len(report['cases']),smoke=report['smoke_byte_equal'])),flush=True)
