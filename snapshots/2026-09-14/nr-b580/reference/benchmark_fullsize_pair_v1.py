"""Paired 864x480 timings of reviewed fullsize fast and reviewed exact routes.

GPU-resident common inputs. Time synchronized processing calls, not IO/encoding.
Check every measured output against the corresponding saved reviewed frame.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
PRODUCT=BASE/'nr-b580-int8/product'
DATA=Path('D:/Codex-NR-Experiments/nr-b580')
FOUR=DATA/'fourway-face-review-v1'
ACCEPTED=DATA/'fast-fullsize-review-v1/pipeline-v4'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def measure(process,mode):
    import numpy as np
    import torch
    source=read(FOUR/'exact/validation.json')
    full=read(ACCEPTED/'full/validation.json')
    assert source['passed'] and full['passed']
    inputs=[]
    for i,e in enumerate(source['frames'][:32]):
        assert sha(e['file'])==e['sha256']
        with np.load(e['file'],allow_pickle=False) as a:
            rgb=torch.from_numpy(a['rgb'].astype('f4')).to('xpu')
            flow=torch.from_numpy(a['motion'].astype('f4')).to('xpu')
            ref=a['exact'].astype('f4')
        if mode=='fast':
            f=full['frames'][i];assert f['input_sha256']==e['sha256'] and sha(f['file'])==f['sha256']
            with np.load(f['file'],allow_pickle=False) as a:ref=a['fast'].astype('f4')
        inputs.append((rgb,flow,ref))
    # Warm both reset and temporal paths; restart sequence for timed run.
    warm=[]
    for i,(rgb,flow,_) in enumerate(inputs[:4]):
        torch.xpu.synchronize();start=time.perf_counter()
        process(rgb,flow,i==0);torch.xpu.synchronize()
        warm.append((time.perf_counter()-start)*1000)
    rows=[]
    for i,(rgb,flow,ref) in enumerate(inputs):
        torch.xpu.synchronize();start=time.perf_counter()
        output=process(rgb,flow,i==0);torch.xpu.synchronize()
        elapsed=(time.perf_counter()-start)*1000
        actual=output.cpu().numpy().astype('f4')
        assert actual.tobytes()==ref.tobytes(),f'{mode} benchmark output changed at frame {i}'
        rows.append(dict(index=i,ms=elapsed,byte_equal=True))
    steady=[r['ms'] for r in rows[4:]]
    return dict(passed=True,mode=mode,size=[480,864],warmup_ms=warm,frames=rows,
        measured_steady_indices=[4,31],median_ms=statistics.median(steady),mean_ms=statistics.mean(steady),
        min_ms=min(steady),max_ms=max(steady),scope='synchronized processing call; includes Python/scopes/guards; excludes upload, readback, IO, motion generation and codec')


def worker(mode,out):
    if mode=='fast':
        adapter=HERE/'review_fast_fullsize_v1.py'
        assert sha(adapter)==read(ACCEPTED/'full/validation.json')['adapter_sha256']
        text=adapter.read_text(encoding='utf-8')
        marker="        if args.phase=='collect':"
        assert text.count(marker)==1
        insertion="""        if args.phase=='benchmark':
            def timed_process(rgb,motion,reset):
                with installed(),torch.inference_mode(),use_arithmetic_backend('triton'):
                    output=stack.model(rgb,motion,reset=reset).float()
                    torch.xpu.synchronize()
                    return output
            with DiskOnly():
                report.update(measure(timed_process,'fast'))
            return
"""
        text=text.replace(marker,insertion+marker)
        ns=dict(__name__='benchmark_fast_adapter',__file__=str(adapter),measure=measure)
        exec(compile(text,str(adapter),'exec'),ns)
        ns['main'](SimpleNamespace(phase='benchmark',out=out,gate=None))
    else:
        out.mkdir(parents=True,exist_ok=False)
        result=dict(passed=False);session=None
        try:
            sys.path.insert(0,str(PRODUCT/'comfy'))
            from runtime_environment import isolate
            handles=isolate()
            from precompile_reuse_v2 import worker_init
            worker_init(DATA/'reference/triton-cache-c32-triton38-v1')
            sys.path.insert(0,str(PRODUCT/'precompile'))
            from fast_cached_runtime_v1 import DiskOnly
            from nr_exact_runtime_v1 import Session
            with DiskOnly():
                session=Session.create(BASE/'nr-b580')
                result=measure(lambda rgb,flow,reset:session.process(rgb,flow,reset=reset).color,'exact')
        except BaseException:
            result['error']=traceback.format_exc();raise
        finally:
            try:
                if session:session.close()
            finally:(out/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    report=dict(passed=False,runs=[],order=['fast','exact','exact','fast'])
    try:
        for i,mode in enumerate(report['order']):
            run=out/f'{i}-{mode}'
            with (out/f'{i}-{mode}.stdout.log').open('wb') as stdout,(out/f'{i}-{mode}.stderr.log').open('wb') as stderr:
                p=subprocess.Popen([sys.executable,'-X','utf8','-u',__file__,'--mode',mode,'--out',str(run)],stdout=stdout,stderr=stderr,creationflags=0x08000000)
                print(json.dumps(dict(mode=mode,pid=p.pid)),flush=True)
                if p.wait()!=0:raise RuntimeError(f'{mode} failed; inspect {run}')
            result=read(run/'validation.json');assert result['passed']
            report['runs'].append(dict(mode=mode,median_ms=result['median_ms'],mean_ms=result['mean_ms'],path=str(run)))
        report['summary']={mode:statistics.mean(r['median_ms'] for r in report['runs'] if r['mode']==mode) for mode in ('fast','exact')}
        report['exact_over_fast']=report['summary']['exact']/report['summary']['fast']
        report['passed']=True
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:(out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['fast','exact']);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.mode:worker(a.mode,a.out)
    else:main(a.out)
