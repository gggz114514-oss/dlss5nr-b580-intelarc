"""Isolated full-size fast arithmetic diagnostic, with a NR256 control gate.

No downsampling/reconstruction in full phase. Reuses frozen quantized weights
and row-independent INT8 kernels in padded row batches; never tiles attention.
Generic attention is checked against the product at NR256 before full testing.
This is eager diagnostic execution, not a performance candidate.
"""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import sys
import traceback

BASE=Path(__file__).resolve().parents[2]
PRODUCT=BASE/'nr-b580-int8/product'
ROOT=Path('D:/Codex-NR-Experiments/nr-b580/fourway-face-review-v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    report=dict(passed=False,phase=args.phase,frames=[],performance_test=False,
        low_resolution_input=False,full_phase_input=[480,864],
        arithmetic='Frozen product INT8 weights/scales, row batching only; generic full attention; eager diagnostic')
    session=None
    try:
        sys.path.insert(0,str(PRODUCT/'comfy'))
        from runtime_environment import isolate
        isolate()
        sys.path.insert(0,str(PRODUCT/'precompile'))
        from fast_cached_runtime_v1 import bootstrap,DiskOnly
        bootstrap()
        import numpy as np
        import torch
        import nr_backend.split_block as split
        import nr_backend.vit_block as vit
        from nr_backend.execution import use_arithmetic_backend
        from nr_runtime_v1 import Session
        from face480_residual_scale_v1 import Face480Scale
        config=read(ROOT.parent/'product-v1/local-runtime-v1.json')
        profile=ROOT.parent/'product-v1/profile-v1.json'
        capture=read(ROOT/'exact/validation.json')
        fast=read(ROOT/'fast/validation.json')
        report.update(source=capture['source'],source_sha256=capture['source_sha256'])
        assert capture['passed'] and fast['passed'] and len(capture['frames'])==243
        report['profile_sha256']=sha(profile)
        for r in (capture,fast):
            for p,h in r['loaded_sources'].items():assert sha(p)==h,p
        session=Session.create(BASE/'nr-b580',profile,config['profile_sha256'])
        stack=session._stack
        counts=dict(c512=0,vit=0)

        def batch_rows(x, width, rows, compute):
            flat=x.reshape(-1,width);parts=[]
            for start in range(0,len(flat),rows):
                n=min(rows,len(flat)-start)
                chunk=flat[start:start+n].contiguous()
                if n<rows:chunk=torch.nn.functional.pad(chunk,(0,0,0,rows-n))
                parts.append(compute(chunk).reshape(rows,width)[:n])
            return torch.cat(parts,dim=0).reshape(x.shape)

        def c512(module,x):
            name=stack.c512_int8.modules[id(module)]
            mlp=batch_rows(x,512,144,lambda t:stack.c512_int8.ffn(name,t.view(12,12,512)))
            h,w=x.shape[:2];sy,sx=module.window_shift
            padded=torch.nn.functional.pad(mlp,(0,0,sx,(-w-sx)%8,sy,(-h-sy)%8))
            attended=split.q(module.attention(padded)[sy:sy+h,sx:sx+w])
            full=module.projection.forward_unquantized(attended,mlp)
            output=split.q(full);counts['c512']+=1
            if module.final_weight is None:return (mlp,mlp,attended,output)
            top=(full[0::2,0::2]+full[0::2,1::2]).half()
            bottom=(full[1::2,0::2]+full[1::2,1::2]).half()
            pool=split.q(((top+bottom).half()*.25).half())
            pool=torch.nn.functional.pad(pool,(0,0,0,(-pool.shape[1])%4,0,(-pool.shape[0])%4))
            final=split.q(split.dot(pool,module.final_weight,chunk_k=16))
            return (mlp,mlp,attended,output,pool,final)

        def vforward(module,x):
            index=stack.int8_vit.modules[id(module)]
            x=vit.q(x)
            mlp=vit.q(batch_rows(x,1024,64,lambda t:stack.int8_vit.ffn(index,t)[0]))
            tokens=x.shape[0]
            z=(vit.dot(mlp[:,:512],module.qkv_weight[:512],chunk_k=16)+vit.dot(mlp[:,512:],module.qkv_weight[512:],chunk_k=16)).half().reshape(tokens,32,3,32)
            query=vit.q((vit.normalize_c32(z[:,:,0])*5.65625).half()*module.query_scale[None,:,None])
            key=vit.q(vit.normalize_c32(z[:,:,1]));value=vit.q(z[:,:,2])
            attended=vit.vit_attention(query.transpose(0,1),key.transpose(0,1),value.transpose(0,1)).transpose(0,1).reshape(tokens,1024)
            counts['vit']+=1
            return vit.q(vit.split_k_projection(attended,module.projection,(mlp*module.attn_skip).half()))

        @contextmanager
        def installed():
            # Keep generic provider/front/warp/fusions, omit NR256 graph ownership
            # and NR256-only rewrite/call guard. Recheck constant guards explicitly.
            excluded=(stack.graph,stack.rewrite,stack.call_guard,stack.compact_queries)
            stack.call_guard.validate()
            with ExitStack() as scopes:
                for c in stack.components:
                    if all(c is not e for e in excluded):scopes.enter_context(c.installed())
                old=(split.SplitSwinBlock.forward,split.SplitSwinBlock.forward_boundaries,vit.VitBlock.forward)
                split.SplitSwinBlock.forward=lambda m,x:c512(m,x)[-1]
                split.SplitSwinBlock.forward_boundaries=c512
                vit.VitBlock.forward=vforward
                try:yield
                finally:
                    split.SplitSwinBlock.forward,split.SplitSwinBlock.forward_boundaries,vit.VitBlock.forward=old
            stack.call_guard.validate()

        if args.phase=='collect':
            from collect_fast_fullsize_v1 import collect
            report.update(collect(stack,installed,args.out))
            return
        if args.phase=='full':
            gate=read(args.gate/'validation.json')
            assert gate['passed'] and gate['phase']=='gate' and gate['adapter_sha256']==sha(__file__)
        rows=capture['frames'][:1] if args.phase=='gate' else capture['frames']
        arrays=args.out/'frames';arrays.mkdir()
        with DiskOnly() as cache:
            for row in rows:
                assert sha(row['file'])==row['sha256']
                with np.load(row['file'],allow_pickle=False) as f:
                    rgb=torch.from_numpy(f['rgb'].astype('f4')).to('xpu')
                    motion=torch.from_numpy(f['motion'].astype('f4')).to('xpu')
                    reference=f['exact'].astype('f4')
                scaler=Face480Scale() if args.phase=='gate' else None
                a,b=scaler.prepare(rgb,motion) if scaler else (rgb,motion)
                before=dict(counts)
                with installed(),torch.inference_mode(),use_arithmetic_backend('triton'):
                    low=stack.model(a,b.float(),reset=row['reset'])
                    color=scaler.composite(rgb,a,low) if scaler else low.float()
                    torch.xpu.synchronize()
                assert counts['c512']-before['c512']==16 and counts['vit']-before['vit']==8,counts
                output=color.cpu().numpy()
                assert output.shape==(480,864,3) and np.isfinite(output).all()
                result=dict(index=row['index'],input_sha256=row['sha256'],counts=dict(counts))
                if args.phase=='gate':
                    fr=fast['frames'][row['index']];assert sha(fr['file'])==fr['sha256']
                    with np.load(fr['file'],allow_pickle=False) as f:expected=f['fast']
                    result['control_max_abs']=float(np.max(np.abs(output-expected)))
                    result['control_byte_equal']=output.tobytes()==expected.tobytes()
                    report['frames'].append(result)
                    assert result['control_byte_equal'],'Adapter differs from existing fast NR256: diagnose before full-size comparison'
                else:
                    delta=(output.astype('f8')-reference)*255
                    result['rmse']=float(np.sqrt(np.mean(delta*delta)))
                    path=arrays/f"{row['index']:04d}.npz"
                    np.savez_compressed(path,fast=output)
                    result.update(file=str(path),sha256=sha(path))
                    report['frames'].append(result)
                print(json.dumps(result),flush=True)
            report['cache_hits']=cache.hits
        report['passed']=True
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        report['adapter_sha256']=sha(__file__)
        try:
            if session is not None:session.close()
        except BaseException:
            report['passed']=False
            report['close_error']=traceback.format_exc()
            raise
        finally:
            (args.out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['gate','full','collect'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--gate',type=Path)
    main(p.parse_args())
