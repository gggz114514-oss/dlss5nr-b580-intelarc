"""Current NR256 fused MLP operands: native-half cubic versus selected baseline.

Capture source has 13 real family/geometries. Compare complete outputs, retain
original matrix reductions, and time five rotated/reversed groups (four calls
per graph, ten replays per sample). Only stable >=5% median / >=2% every-round
savings are eligible; weighted primitive savings are not full-model timings.
"""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'experimental/native-cubic-mlp-v1'
assert not OUT.exists()
TOOLCHAIN=D/'toolchains/triton-xpu-3.8.0-git1e2d42a0'
os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-c32-triton38-v1')
sys.path[:0]=[str(TOOLCHAIN/'site'),str(EXACT/'reference'),str(ROOT/'backend')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main();sources={};records={}
for name,folder,pin in [('capture','current-cubic-operands-v1','daa162d9ead33906ffb9bd70e2946ac2817fd48eb543671ab7e91cd5ef2ccc6a'),
                        ('unary','native-half-cubic-v1','e5467b8979a353db29e11956adfc5a4e575b41d2e65731821672c1cdd219a925')]:
    p=D/'experimental'/folder/'validation.json';assert sha(p)==pin
    record=js(p);assert record['passed'] and js(p.parent.with_suffix('.log.lease.json'))['returncode']==0
    assert all(sha(p)==h for p,h in record['sources'].items())
    sources.update(record['sources']);sources[str(p)]=pin;records[name]=record
provision=TOOLCHAIN/'provision-v1.json'
assert sha(provision)=='e94757afcb5962cacbe208cbb808b3516de210059335f1ee7ab7b40d6bfe2c07'
assert all(sha(TOOLCHAIN/'site'/p)==h for p,h in js(provision)['files'].items())
for name in ('benchmark_native_cubic_mlp_v1.py','Run-NativeCubicMLPV1.cmd','native_cubic_split_v1.py','native_cubic_c32_v1.py','native_cubic_batched_v1.py'):
    sources[str(HERE/name)]=sha(HERE/name)
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:q=json.load(response)
assert not q['queue_running'] and not q['queue_pending']
import numpy as np
import torch,triton
from nr_backend.execution import use_arithmetic_backend
from fused_split_ffwd_v1 import forward as old_split
from fused_c32_mlp_lut_v1 import forward as old_c32
from batched_branched_mlp_v1 import forward as old_batched
from native_cubic_split_v1 import forward as new_split
from native_cubic_c32_v1 import forward as new_c32
from native_cubic_batched_v1 import forward as new_batched
import compressed_arrays_v1 as arrays
import immutable_artifacts_v1 as artifacts
assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site')
OUT.mkdir();report=dict(scope=__doc__,passed=False,complete_migration=False,sources=sources,exact_gate=gate,cases=[],nr_input=[256,256],rounds=5,calls_per_graph=4,replays_per_sample=10)
report['runtime']=dict(torch=torch.__version__,triton=triton.__version__,triton_file=triton.__file__,isolated=True)
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
def raw(t):return t.cpu().numpy().tobytes()
graphs=[]
try:
    torch.set_num_threads(2)
    with torch.inference_mode(),use_arithmetic_backend('triton'):
        for key,old in records['capture']['cases'].items():
            family=old['family'];cpu={n:arrays.load(meta) for n,meta in old['operands'].items()};operands={}
            for name,value in cpu.items():
                desc=old['descriptors'][name];dtype=torch.int16 if value.dtype==np.dtype('<i2') else torch.float16
                tensor=torch.empty_strided(tuple(desc['shape']),tuple(desc['stride']),dtype=dtype,device='xpu')
                tensor.copy_(torch.from_numpy(value.copy()).to('xpu'));assert raw(tensor)==value.tobytes()
                operands[name]=tensor
            expected=arrays.load(old['expected']).tobytes();options=dict(old['options'])
            baseline={'split':old_split,'c32':old_c32,'batched':old_batched}[family]
            candidate={'split':new_split,'c32':new_c32,'batched':new_batched}[family]
            if family=='split':configs=[dict(options,bm=bm,bn=bn) for bm,bn in [(16,32),(16,64),(32,32),(32,64)]]
            elif family=='c32':configs=[dict(options,bm=bm) for bm in (16,32)]
            else:configs=[dict(options,pair_bm=bm) for bm in (16,32)]
            functions={'previous':lambda:baseline(**operands,**options)}
            functions.update({str(i):lambda config=config:candidate(**operands,**config) for i,config in enumerate(configs)})
            row=dict(key=key,family=family,rows=old['rows'],channels=old['channels'],count=old['count'],operands=old['operands'],descriptors=old['descriptors'],expected=old['expected'],baseline_options=options,configs=configs,candidates={},samples_ms={},orders=[])
            report['cases'].append(row);entries={}
            for label,fn in functions.items():
                value,kernels=fn();equal=raw(value)==expected
                if not isinstance(kernels,tuple):kernels=(kernels,)
                kernel=kernels[0];llir=str(kernel.asm['llir']);ir=str(kernel.asm['ttgir'])
                row['candidates'][label]=dict(byte_equal=equal,spills=kernel.n_spills,native_half_fma='llvm.fma.f16' in llir,
                    llir=artifacts.text(llir,'llir'),ttgir_sha256=hashlib.sha256(ir.encode()).hexdigest())
                assert 'ttig.dpas' in ir
                if label!='previous':assert row['candidates'][label]['native_half_fma']
                if not equal:
                    row['candidates'][label]['actual']=arrays.save(value.cpu().numpy())
                    if label=='previous':raise AssertionError('Captured baseline no longer matches')
                    continue
                stream=torch.xpu.Stream()
                with stream:
                    for _ in range(2):warm=fn()[0]
                torch.xpu.synchronize();out=torch.empty_like(warm);del warm
                g=torch.xpu.XPUGraph()
                with torch.xpu.graph(g,stream=stream):
                    for _ in range(4):temporary=fn()[0]
                    out.copy_(temporary)
                del temporary;graphs.append(g);entries[label]=(g,out);row['samples_ms'][label]=[]
                assert not any(s['address']<=t.data_ptr()<s['address']+s['total_size'] for s in torch.xpu.memory_snapshot(g.pool()) for t in [*operands.values(),out])
                g.replay();torch.xpu.synchronize();assert raw(out)==expected
                row['candidates'][label].update(graph_byte_equal=True,persistent_io_outside_pool=True)
            labels=list(entries)
            for repeat in range(5):
                offset=repeat%len(labels);order=labels[offset:]+labels[:offset]
                if repeat%2:order.reverse()
                row['orders'].append(order)
                for label in order:
                    g,out=entries[label];torch.xpu.synchronize();start=time.perf_counter()
                    for _ in range(10):g.replay()
                    torch.xpu.synchronize();row['samples_ms'][label].append((time.perf_counter()-start)*1000/40)
                    assert raw(out)==expected
            row['median_ms']={label:statistics.median(v) for label,v in row['samples_ms'].items()}
            eligible=[label for label in entries if label!='previous' and row['median_ms'][label]<=.95*row['median_ms']['previous'] and all(a<=.98*b for a,b in zip(row['samples_ms'][label],row['samples_ms']['previous']))]
            row['selected']=min(eligible,key=lambda label:row['median_ms'][label]) if eligible else 'previous'
            row['weighted_saving_ms']=old['count']*(row['median_ms']['previous']-row['median_ms'][row['selected']])
            assert all(raw(operands[n])==a.tobytes() for n,a in cpu.items());row['operands_unchanged']=True
            for g in graphs:g.reset()
            graphs.clear();save()
            print(json.dumps(dict(key=key,count=old['count'],median_ms=row['median_ms'],selected=row['selected'],weighted_saving_ms=row['weighted_saving_ms'])),flush=True)
    report.update(passed=True,weighted_local_saving_ms=sum(c['weighted_saving_ms'] for c in report['cases']),selected_changes=sum(c['selected']!='previous' for c in report['cases']))
except BaseException as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    for g in graphs:g.reset()
    try:
        assert all(sha(p)==h for p,h in sources.items());authenticate_main()
    except BaseException as error:
        report.update(passed=False,finalization_error=repr(error));save();raise
    save()
print(json.dumps(dict(passed=report['passed'],selected=report['selected_changes'],weighted_local_saving_ms=report['weighted_local_saving_ms'])),flush=True)
