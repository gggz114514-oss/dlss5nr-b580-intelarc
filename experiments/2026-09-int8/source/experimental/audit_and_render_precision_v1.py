"""Reread saved complete arrays, check evidence, and render a short visual review."""
import argparse,hashlib,json,math,statistics,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
HERE=Path(__file__).resolve().parent
EXACT=HERE.parent.parent/'nr-b580'
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference')
p=argparse.ArgumentParser()
p.add_argument('--dimension',choices=['864x480','1920x1080'],required=True)
a=p.parse_args()
w,h=map(int,a.dimension.split('x'))
OUT=DREF/f'results/fast-precision-{a.dimension}-v1'
path=OUT/'validation.json'
report=json.loads(path.read_text(encoding='utf-8'))
assert report['passed'] and not report['promoted']
assert not (OUT/'saved-audit-v1.json').exists()
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
for name,digest in report['sources'].items():assert sha(name)==digest
checked={}

def load(meta):
    item=Path(meta['path'])
    assert item.is_relative_to(DREF/'experimental/immutable-artifacts-v1')
    if str(item) not in checked:
        assert sha(item)==meta['sha256'] and item.stat().st_size==meta['stored_bytes']
        checked[str(item)]=meta['sha256']
    value=np.load(item,allow_pickle=False)
    assert list(value.shape)==meta['shape'] and value.dtype.str==meta['dtype']
    assert hashlib.sha256(value.tobytes()).hexdigest()==meta['raw_sha256']
    return value

summary={}
for mode,rows in report['runs'].items():
    assert len(rows)==13
    for i,row in enumerate(rows):
        actual=load(row['actual']).astype('f4')
        private=load(row['private'])
        assert actual.shape==(h,w,3) and np.isfinite(actual).all()
        assert hashlib.sha256(actual.tobytes()).hexdigest()==row['rgb_sha256']
        assert private.tobytes()==actual.astype('f2').tobytes()
        assert sha(row['native_path'])==row['native_file_sha256']
        expected=np.fromfile(row['native_path'],'<f4').reshape(h,w,4)[...,:3].copy()
        equal=actual.tobytes()==expected.tobytes()
        assert equal==row['byte_equal']
        if mode=='baseline':assert equal
        delta=actual.astype('f8')-expected.astype('f8')
        mse=float(np.square(delta).mean())
        if mse:assert abs(-10*math.log10(mse)-row['psnr_db'])<1e-5
        else:assert row['psnr_db'] is None
        assert row['reset']==(i in (0,12)) and row['next_seed']==(1 if i in (0,12) else i+1)
    assert rows[0]['rgb_sha256']==rows[12]['rgb_sha256']
    mean=statistics.mean(r['seconds'] for r in rows[1:])
    assert mean==report['matching_warm_mean_seconds'][mode]
    summary[mode]=dict(warm_mean_seconds=mean,speedup=report['paired_speedup'][mode],mean_psnr_db=None if mode=='baseline' else statistics.mean(r['psnr_db'] for r in rows[:12]),mean_ssim=statistics.mean(r['ssim'] for r in rows[:12]),max_abs=max(r['max_abs'] for r in rows),all_byte_equal=all(r['byte_equal'] for r in rows))
for kind,entries in report['compiled'].items():
    for ir_kind,meta in entries.items():
        assert sha(meta['path'])==meta['sha256']
        text=Path(meta['path']).read_text(encoding='utf-8')
        assert ('ttig.dpas' if ir_kind=='ttgir' else 'SubgroupMatrixMultiplyAccumulateINTEL') in text
fixture=EXACT/'reference'/f"inputs/flow-full-{a.dimension}-v{2 if a.dimension=='864x480' else 3}"
manifest=json.loads((fixture/'manifest.json').read_text(encoding='utf-8'))
review=OUT/'review'
review.mkdir(exist_ok=False)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
view_w=min(w,864)
view_h=round(h*view_w/w)
view_h+=view_h%2
title_h=54
video_path=review/'comparison-slow-6fps.mp4'
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
command=[ffmpeg,'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{view_w*2}x{(view_h+title_h)*2}','-r','6','-i','-','-an','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart','-n',str(video_path)]
process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
def image(value):return Image.fromarray(np.round(np.clip(value,0,1)*255).astype('u1'))
try:
    for i in range(12):
        original=Image.open(fixture/manifest['frames'][i]['file']).convert('RGB')
        rows={mode:report['runs'][mode][i] for mode in report['runs']}
        pictures=[original]+[image(load(rows[mode]['actual']).astype('f4')) for mode in ('baseline','fp16_xmx','int8_dense')]
        labels=['Original input','4060 reference = B580 exact','B580 FP16 XMX (candidate)','B580 INT8 dense + FP16 attention (candidate)']
        canvas=Image.new('RGB',(view_w*2,(view_h+title_h)*2),(20,24,32))
        draw=ImageDraw.Draw(canvas)
        for j,(picture,label) in enumerate(zip(pictures,labels)):
            x=(j%2)*view_w;y=(j//2)*(view_h+title_h)
            draw.text((x+10,y+5),label,font=font,fill='white')
            draw.text((x+10,y+30),f'Frame {i:02d} | 6 fps slow replay; not inference FPS',font=small,fill=(182,194,210))
            if picture.size!=(view_w,view_h):picture=picture.resize((view_w,view_h),Image.Resampling.LANCZOS)
            canvas.paste(picture,(x,y+title_h))
        if i in (0,5,11):
            canvas.save(review/f'comparison-frame{i:02d}.png')
            for mode in ('baseline','fp16_xmx','int8_dense'):
                image(load(rows[mode]['actual']).astype('f4')).save(review/f'{mode}-frame{i:02d}.png')
        process.stdin.write(np.asarray(canvas).tobytes())
    process.stdin.close()
    error=process.stderr.read().decode('utf-8',errors='replace')
    assert process.wait(timeout=60)==0,error
finally:
    if process.poll() is None:process.kill();process.wait()
audit=dict(report_sha256=sha(path),audit_source_sha256=sha(Path(__file__)),summary=summary,unique_arrays_verified=len(checked),unique_array_bytes=sum(Path(p).stat().st_size for p in checked),saved_rgb_and_private_verified=True,reset_reproduces_frame0=True,compiler_matrix_ir_verified=True,complete_migration=False,human_review='pending',review_files={str(p):sha(p) for p in review.iterdir()},video_scope='12 consecutive source frames slowed to 6fps; no interpolation; display RGB clamped/rounded to8bit; raw float arrays retained separately. Only 0.2-0.5s of original footage depending on source FPS, not a long temporal validation.')
(OUT/'saved-audit-v1.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
print(json.dumps(audit['summary'],indent=2),flush=True)
