"""Offline decoder-gather specializations for the isolated fullsize experiment.

Fourteen compiler workers; dtype-only warmup, no model/kernel execution.
Cover both the NR256 control and original 864x480 decoder geometry.
"""
import argparse
import concurrent.futures as futures
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

BASE=Path(__file__).resolve().parents[2]
PRODUCT=BASE/'nr-b580-int8/product'


def init():
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    sys.path.insert(0,str(PRODUCT/'comfy'))
    from runtime_environment import isolate
    global handles
    handles=isolate()
    sys.path.insert(0,str(PRODUCT/'precompile'))
    from fast_cached_runtime_v1 import bootstrap
    bootstrap()


def compile_one(job):
    import torch
    import decoder_gather_merge_v1 as kernels
    shape,h,w,c,ph,pw=job
    ps=(pw*c,c,1);ss=(w*c,c,1)
    options=dict(num_warps=4,enable_fp_fusion=False)
    if c==32:
        # Decoder C32 shift is (0,0) in the frozen model; additionally compile
        # all observed 0/4 combinations so no speculative runtime compile occurs.
        variants=[]
        for sy,sx in ((0,0),(0,4),(4,0),(4,4)):
            oh=(h+sy+7)//8*8;ow=(w+sx+7)//8*8
            variants.append((kernels._raw_padded,(h,w,c,oh,ow,sy,sx,ps,ss,1,256),oh*ow*c))
    else:
        variants=[(kernels._quantized,(h,w,c,ps,ss,1,256),h*w*c)]
    rows=[]
    for kernel,args,numel in variants:
        result=kernel.warmup(torch.float16,torch.float16,torch.float16,torch.float16,
            *args,grid=((numel+255)//256,),**options)
        rows.append(dict(name=kernel.fn.__name__,args=args,hash=result.hash,
            files={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in result.metadata_group.values()}))
    return dict(shape=shape,geometry=[h,w,c,ph,pw],pid=os.getpid(),kernels=rows)


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for label,shapes in (
        ('nr256',[(12,12,512,8,8),(20,20,256,12,12),(40,40,128,20,20),(80,80,64,40,40),(160,160,32,80,80)]),
        ('480x864',[(16,28,512,8,16),(32,56,256,16,28),(64,112,128,32,56),(128,224,64,64,112),(256,448,32,128,224)])):
        jobs.extend((label,*s) for s in shapes)
    report=dict(passed=False,workers=14,gpu_dispatches=0,completed=[],scope='decoder gather only; runtime cache guard remains enabled')
    try:
        with futures.ProcessPoolExecutor(max_workers=14,initializer=init) as pool:
            pending=[pool.submit(compile_one,j) for j in jobs]
            for f in futures.as_completed(pending):
                report['completed'].append(f.result())
                (out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                print(json.dumps(dict(completed=len(report['completed']),total=len(jobs))),flush=True)
        report['passed']=len(report['completed'])==len(jobs)
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        (out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
