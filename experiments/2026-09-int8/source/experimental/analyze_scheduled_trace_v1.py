"""Analyze raw device trace intervals without double-counting parent CPU events."""
import collections,hashlib,json
from pathlib import Path
ROOT=Path('D:/Codex-NR-Experiments/nr-b580/reference/profiles/scheduled-fast-480-v1')
OUT=ROOT/'trace-analysis-v1.json';assert not OUT.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=ROOT/'validation.json';report=json.loads(source.read_text())
lease=ROOT.with_name(ROOT.name+'.log.lease.json');execution=json.loads(lease.read_text())
analysis=dict(scope=__doc__,sources={str(Path(__file__).resolve()):sha(Path(__file__)),str(source):sha(source),str(lease):sha(lease)},modes={},inference_outputs_checked=report['passed'],process_success=execution.get('returncode')==0,process_returncode=execution.get('returncode'),limitation='Trace files completed and outputs matched, but profiler process crashed during native teardown. Windows event1000 names ze_tracing_layer.dll, exceptions0xc0000005 and0xc0000409. Use traces only as diagnostic evidence; wall/union ratios are instrumented and are NOT production utilization or throughput.')
def union(intervals):
    merged=[]
    for a,b in sorted(intervals):
        if not merged or a>merged[-1][1]:merged.append([a,b])
        else:merged[-1][1]=max(merged[-1][1],b)
    return sum(b-a for a,b in merged)
for mode,row in report['modes'].items():
    path=Path(row['trace']['path']);assert sha(path)==row['trace']['sha256']
    analysis['sources'][str(path)]=sha(path)
    events=json.loads(path.read_text())['traceEvents']
    frame=next(e for e in events if e.get('name')=='NR_scheduled_'+mode)
    start,end=frame['ts'],frame['ts']+frame['dur']
    device=[e for e in events if e.get('ph')=='X' and e.get('cat') in ('kernel','gpu_memcpy')]
    assert {e['args']['device'] for e in device}=={0}
    clipped=[(max(start,e['ts']),min(end,e['ts']+e['dur'])) for e in device if e['ts']<end and e['ts']+e['dur']>start]
    kernels=collections.defaultdict(lambda:dict(count=0,total_us=0.0))
    copies=collections.defaultdict(lambda:dict(count=0,bytes=0,total_us=0.0))
    for e in device:
        if e['cat']=='kernel':
            entry=kernels[e['name']];entry['count']+=1;entry['total_us']+=e['dur']
        else:
            entry=copies[e['name']];entry['count']+=1;entry['bytes']+=e['args'].get('bytes',0);entry['total_us']+=e['dur']
    item=dict(kernel_calls=sum(k['count'] for k in kernels.values()),memcpy_calls=sum(k['count'] for k in copies.values()),instrumented_frame_us=frame['dur'],clipped_device_union_us=union(clipped),instrumented_device_union_fraction=union(clipped)/frame['dur'],raw_device_duration_sum_us=sum(e['dur'] for e in device),kernels=sorted((dict(name=k,**v) for k,v in kernels.items()),key=lambda v:-v['total_us']),copies=dict(copies))
    analysis['modes'][mode]=item
    print(json.dumps(dict(mode=mode,kernel_calls=item['kernel_calls'],instrumented_frame_us=item['instrumented_frame_us'],device_union_us=item['clipped_device_union_us'],copies=item['copies'],top_kernels=item['kernels'][:6])),flush=True)
OUT.write_text(json.dumps(analysis,indent=2)+'\n',encoding='utf-8')
