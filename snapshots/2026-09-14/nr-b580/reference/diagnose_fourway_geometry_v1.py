"""Separate NR256 geometry from current fast arithmetic, with complete histories.

Uses the authenticated fourway inputs and accepted full-size exact outputs.
This is an ordered intervention, not an additive percentage attribution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

BASE=Path(__file__).resolve().parents[2]
ROOT=Path('D:/Codex-NR-Experiments/nr-b580/fourway-face-review-v1')
PRODUCT=BASE/'nr-b580-int8/product'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    report=dict(passed=False,frames=[],performance_test=False,
                intervention='Full exact -> exact arithmetic at NR256 with the current residual scaler -> current fast NR256',
                history='Independent continuous history; reset only at frame0; no reference outputs injected into state',
                roi=[256,16,640,400],roi_is_skin_segmentation=False)
    session=None
    try:
        exact=read(ROOT/'exact/validation.json');fast=read(ROOT/'fast/validation.json')
        assert exact['passed'] and fast['passed'] and len(exact['frames'])==len(fast['frames'])==243
        assert exact['source_sha256']==fast['source_sha256']
        # These are current recorded product outputs, not obsolete pre-repair INT8.
        for r in (exact,fast):
            for path,digest in r['loaded_sources'].items():assert sha(path)==digest,path
        sys.path.insert(0,str(PRODUCT/'comfy'))
        from runtime_environment import isolate
        isolate()
        from precompile_reuse_v2 import worker_init
        worker_init(Path('D:/Codex-NR-Experiments/nr-b580/reference/triton-cache-c32-triton38-v1'))
        sys.path[:0]=[str(BASE/'nr-b580-int8/experimental'),str(PRODUCT/'precompile')]
        import numpy as np
        import torch
        from PIL import Image,ImageDraw,ImageFont
        from fast_cached_runtime_v1 import DiskOnly
        from nr_exact_runtime_v1 import Session
        from face480_residual_scale_v1 import Face480Scale
        font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
        def measure(a,b):
            d=(a.astype('f8')-b.astype('f8'))*255
            return dict(rmse=float(np.sqrt(np.mean(d*d))),mae=float(np.abs(d).mean()),
                        mean_rgb=d.mean((0,1)).tolist(),max_abs=float(np.abs(d).max()))
        with torch.inference_mode(),DiskOnly() as cache:
            session=Session.create(BASE/'nr-b580')
            scaler=Face480Scale()
            report['scaler']=scaler.metadata()
            for i,(e,f) in enumerate(zip(exact['frames'],fast['frames'])):
                assert e['index']==f['index']==i and e['reset']==(i==0)
                assert sha(e['file'])==e['sha256'] and sha(f['file'])==f['sha256'] and f['input_sha256']==e['sha256']
                with np.load(e['file'],allow_pickle=False) as d:
                    source=d['rgb'].astype('f4');flow=d['motion'].astype('f4');full=d['exact'].astype('f4')
                with np.load(f['file'],allow_pickle=False) as d:quick=d['fast'].astype('f4')
                rgb=torch.from_numpy(source).to('xpu');motion=torch.from_numpy(flow).to('xpu')
                canvas,low_motion=scaler.prepare(rgb,motion)
                result=session.process(canvas,low_motion.float(),reset=i==0)
                geometry=scaler.composite(rgb,canvas,result.color).cpu().numpy()
                assert np.isfinite(geometry).all()
                row=dict(index=i,regions={})
                for name,sl in [('full',(slice(None),slice(None))),('face_roi',(slice(16,400),slice(256,640)))]:
                    a,b,c=full[sl],geometry[sl],quick[sl]
                    g=(b.astype('f8')-a)*255;p=(c.astype('f8')-b)*255;t=(c.astype('f8')-a)*255
                    cross=float(2*np.mean(g*p))
                    assert np.isclose(np.mean(t*t),np.mean(g*g)+np.mean(p*p)+cross)
                    row['regions'][name]=dict(total_fast_vs_full=measure(c,a),geometry_vs_full=measure(b,a),
                        fast_vs_geometry=measure(c,b),mse_cross_term=cross)
                report['frames'].append(row)
                if i in (0,48,96,144,192,242):
                    path=out/f'frame-{i:03d}.npz'
                    np.savez_compressed(path,source=source,full_exact=full,low_exact_composite=geometry,current_fast=quick,
                                        low_input=canvas.cpu().numpy(),low_exact=result.color.cpu().numpy())
                    row['artifact']=dict(path=str(path),sha256=sha(path))
                    pictures=[source,full,geometry,quick]
                    labels=['原视频','全尺寸精确（已对照4060）','仅改NR256＋现有残差合成','当前快速NR256（现行量化）']
                    image=Image.new('RGB',(1728,1024),(18,22,28));draw=ImageDraw.Draw(image)
                    for j,(label,picture) in enumerate(zip(labels,pictures)):
                        x=(j%2)*864;y=(j//2)*512
                        draw.text((x+8,y+4),label,font=font,fill='white')
                        image.paste(Image.fromarray(np.rint(np.clip(picture,0,1)*255).astype('u1')),(x,y+32))
                    image.save(out/f'comparison-{i:03d}.png')
                if i%24==0:print(json.dumps(dict(frame=i,roi=row['regions']['face_roi'])),flush=True)
            report['cache_hits']=cache.hits
        report['summary']={}
        for region in ('full','face_roi'):
            report['summary'][region]={}
            for key in ('total_fast_vs_full','geometry_vs_full','fast_vs_geometry'):
                rows=[f['regions'][region][key] for f in report['frames']]
                report['summary'][region][key]=dict(aggregate_rmse=float(np.sqrt(np.mean([r['rmse']**2 for r in rows]))),
                    mean_mae=float(np.mean([r['mae'] for r in rows])),mean_rgb=np.mean([r['mean_rgb'] for r in rows],axis=0).tolist())
            report['summary'][region]['mean_mse_cross_term']=float(np.mean([f['regions'][region]['mse_cross_term'] for f in report['frames']]))
        report['limitations']=['Geometry includes input scaling, padding, motion scaling, residual composite and their temporal interaction.',
                              'Fast-vs-low-exact includes all arithmetic approximations and their own histories; not INT8 alone.',
                              'RMSE terms are not additive; the cross term may cancel or amplify error.',
                              'No output color-filter switch exists in the range-repair intervention.']
        report['passed']=True
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        if session is not None:session.close()
        (out/'validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
