"""Authenticate all four streams, compare native/exact before encoding, render review."""
import hashlib
import json
from pathlib import Path
import subprocess
import traceback
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont

BASE=Path(__file__).resolve().parents[2]
ROOT=Path('D:/Codex-NR-Experiments/nr-b580/fourway-face-review-v1')
FF=BASE/'xess-tools/work/d3d12-media-pipeline/deps/ffmpeg-lgpl-shared-9.0/ffmpeg-n9.0-latest-win64-lgpl-shared-9.0/bin/ffmpeg.exe'
ENCODER=Path(imageio_ffmpeg.get_ffmpeg_exe())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    out=ROOT/'review-v2';out.mkdir(exist_ok=False)
    report=dict(passed=False,frames=[],visual_review='pending',performance_test=False,encoder=str(ENCODER),encoder_sha256=sha(ENCODER),remuxer=str(FF),remuxer_sha256=sha(FF))
    processes=[]
    try:
        encoders=subprocess.check_output([str(ENCODER),'-hide_banner','-encoders'],text=True,stderr=subprocess.STDOUT)
        if 'libx264 ' not in encoders:raise RuntimeError('Selected encoder has no libx264')
        exact=read(ROOT/'exact/validation.json');fast=read(ROOT/'fast/validation.json');native=read(ROOT/'native/validation.json')
        assert exact['passed'] and fast['passed'] and native['passed']
        assert len(exact['frames'])==len(fast['frames'])==native['frames']==243
        assert exact['source_sha256']==fast['source_sha256']==sha(exact['source'])
        nroot=Path(native['results']);nreport=read(nroot/'run.json')
        assert sha(nroot/'run.json')==native['native_report_sha256']
        nhashes={r['relative']:r['sha256'].lower() for r in nreport['files']}
        # First finish byte acceptance of the whole stream before publishing a video.
        for i,(e,f) in enumerate(zip(exact['frames'],fast['frames'])):
            assert e['index']==f['index']==i and e['reset']==(i==0)
            assert sha(e['file'])==e['sha256'] and sha(f['file'])==f['sha256'] and f['input_sha256']==e['sha256']
            npth=nroot/f'{i:04d}.png_output.rgba32f.bin'
            assert sha(npth)==nhashes[npth.name]
            values=np.fromfile(npth,dtype='<f4').reshape(480,864,4)[:,:,:3]
            meta=read(nroot/f'{i:04d}.png_capture.json')
            assert sha(nroot/f'{i:04d}.png_capture.json')==nhashes[f'{i:04d}.png_capture.json']
            assert meta['reset']==int(i==0) and meta['style']==0 and meta['intensity']==1
            assert meta['local_structure']==meta['local_tone']==1
            assert np.isfinite(values).all() and np.array_equal(values,values.astype('f2').astype('f4'))
            with np.load(e['file'],allow_pickle=False) as data:
                a=data['exact'];b=values.astype('f2')
                equal=a.tobytes()==b.tobytes()
                row=dict(index=i,byte_equal=equal,max_abs=float(np.max(np.abs(a.astype('f4')-b.astype('f4')))))
                report['frames'].append(row)
                if not equal:raise AssertionError(f'Native/exact mismatch at frame {i}: {row}')
        labels=['原视频（未经过 NR）','4060 原版 NR · 864×480','B580 精确版 · 864×480','B580 快速版 · NR256＋色调修正']
        font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
        small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
        def start(name,w,h):
            path=out/('encoded-'+name)
            cmd=[str(ENCODER),'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','24','-i','pipe:0','-i',exact['source'],
                 '-map','0:v:0','-map','1:a:0?','-frames:v','243','-c:v','libx264','-threads','4','-preset','fast','-crf','14',
                 '-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
                 '-c:a','copy','-t','10.125','-movflags','+faststart+write_colr',str(path)]
            log=(out/(name+'.log')).open('wb')
            process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log,creationflags=0x08000000)
            processes.append((process,log))
            return process,path
        full,fullpath=start('comparison-fourway-24fps.mp4',1728,1056)
        face,facepath=start('comparison-face-24fps.mp4',1536,1632)
        for i,(e,f) in enumerate(zip(exact['frames'],fast['frames'])):
            with np.load(e['file'],allow_pickle=False) as d:source=d['rgb'].astype('f4');ex=d['exact'].astype('f4')
            with np.load(f['file'],allow_pickle=False) as d:quick=d['fast'].astype('f4')
            nv=np.fromfile(nroot/f'{i:04d}.png_output.rgba32f.bin',dtype='<f4').reshape(480,864,4)[:,:,:3]
            pictures=[Image.fromarray(np.rint(np.clip(a,0,1)*255).astype('u1')) for a in (source,nv,ex,quick)]
            for proc,w,h,crop in ((full,864,480,False),(face,768,768,True)):
                canvas=Image.new('RGB',(w*2,(h+48)*2),(19,24,32));draw=ImageDraw.Draw(canvas)
                for j,picture in enumerate(pictures):
                    x=(j%2)*w;y=(j//2)*(h+48)
                    draw.text((x+8,y+2),labels[j],font=font,fill='white')
                    draw.text((x+8,y+27),f'{i:03d}/242 · {i/24:.3f}s · 原速24fps'+(' · 同一区域2倍近邻放大' if crop else ''),font=small,fill=(190,200,215))
                    if crop:picture=picture.crop((256,16,640,400)).resize((768,768),Image.Resampling.NEAREST)
                    canvas.paste(picture,(x,y+48))
                proc.stdin.write(canvas.tobytes())
                if i in (96,192):canvas.save(out/f"{'face' if crop else 'full'}-{i:03d}.png")
        for proc,log in processes:
            proc.stdin.close();assert proc.wait(timeout=120)==0;log.close()
        # Finalize MP4 color tags with the modern muxer; retain encoded video bytes.
        def finalize(path):
            target=path.with_name(path.name.removeprefix('encoded-'))
            subprocess.run([str(FF),'-hide_banner','-loglevel','error','-n','-i',str(path),
                            '-map','0','-c','copy','-bsf:v','h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1:video_full_range_flag=0',
                            '-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
                            '-movflags','+faststart+write_colr',str(target)],check=True,timeout=120)
            return target
        fullpath=finalize(fullpath);facepath=finalize(facepath)
        ffprobe=FF.with_name('ffprobe.exe')
        videos=[]
        for path,w,h in ((fullpath,1728,1056),(facepath,1536,1632)):
            meta=json.loads(subprocess.check_output([str(ffprobe),'-v','error','-count_frames','-show_streams','-of','json',str(path)],timeout=120))
            stream=next(s for s in meta['streams'] if s['codec_type']=='video')
            assert int(stream['nb_read_frames'])==243 and stream['avg_frame_rate']=='24/1'
            assert (stream['width'],stream['height'])==(w,h)
            assert all(stream.get(k)=='bt709' for k in ('color_space','color_primaries','color_transfer'))
            source_meta=json.loads(subprocess.check_output([str(ffprobe),'-v','error','-show_streams','-of','json',exact['source']]))
            assert any(s['codec_type']=='audio' for s in meta['streams'])==any(s['codec_type']=='audio' for s in source_meta['streams'])
            videos.append(dict(path=str(path),sha256=sha(path),metadata=meta))
        report.update(passed=True,videos=videos,all_native_exact_byte_equal=True,source_sha256=exact['source_sha256'],
                      layout=labels,face_crop=[256,16,640,400],input_crop=False,frames_count=243)
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        for proc,log in processes:
            if proc.poll() is None:proc.kill();proc.wait()
            if not log.closed:log.close()
        (out/'validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')


if __name__=='__main__':main()
