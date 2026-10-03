"""Audit the full reduced review, rereading new lossless low NR and decoded video.

Both GPU paths compared complete full FP32 images during inference. Full images
are represented by hashes here; this audit does not independently recompute NR
or residual composition. Existing reference panels were authenticated and fully
loaded by the renderer. Human approval is a separate, pending decision.
"""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import compressed_arrays_v1 as arrays
D=Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT=D/'results/batched-residual-long1080-review-v2'
target=OUT/'saved-audit-v1.json';assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
path=OUT/'validation.json';r=js(path);lease=OUT.with_suffix('.log.lease.json')
assert r['passed'] and js(lease)['returncode']==0 and not r['complete_migration']
assert r['original_render_exit_code']==1 and r['original_render_error']=="KeyError('color_transfer')"
assert r['compressed_pictures_unchanged'] and not r['reencoded']
assert all(sha(p)==h for p,h in r['sources'].items())
assert r['frames_completed']==len(r['frames'])==390
assert r['all_reduced_outputs_byte_equal'] and r['independent_history'] and r['reset_reproduces_first_frame']
assert r['caller_ownership_guards_passed'] and r['held_outputs_survive_replay'] and r['lut_bytes_unchanged']
assert r['human_review']=='pending' and not r['new_quality_approved']
refs={mode:js(D/f'results/long-precision-1080-{mode}-v1/validation.json') for mode in ('baseline','fp16_xmx')}
verified={}
for i,row in enumerate(r['frames']):
    assert row['frame']==i and row['reset']==(i==0)
    assert row['full_byte_equal_previous'] and row['low_byte_equal_previous']
    left,right=row['runs']['previous'],row['runs']['batched']
    assert left['full_raw_sha256']==right['full_raw_sha256']
    assert left['low_raw_sha256']==right['low_raw_sha256']
    assert left['effective_dispatch']==right['effective_dispatch']
    assert all(v['private_byte_equal_low'] and v['next_seed']==i+1 and v['seconds']>0 for v in (left,right))
    meta=row['low_nr'];a=arrays.load(meta)
    assert a.shape==(256,256,3) and a.dtype==np.dtype('f2') and np.isfinite(a).all()
    assert hashlib.sha256(a.tobytes()).hexdigest()==right['low_raw_sha256']
    verified[meta['path']]=meta['stored_bytes']
    for mode,ref in refs.items():
        old=ref['frames'][i]
        assert row['reference_outputs'][mode]==old['output']
        assert row['motion']==old['motion'] and row['input_rgb8_sha256']==old['input_rgb8_sha256']
    assert len(row['presentation_rgb_sha256'])==64
pools=set()
for name,entries in r['graphs'].items():
    assert len(entries)==2
    assert all(e['persistent_inputs_and_output_verified_outside_pool'] for e in entries)
    pools.update(e['pool'] for e in entries)
assert len(pools)==1
for name,h in r['images'].items():assert sha(name)==h
video=Path(r['video']['path']);assert sha(video)==r['video']['sha256'] and video.stat().st_size==r['video']['bytes']
ffprobe=next(Path(p) for p in r['sources'] if Path(p).name=='ffprobe.exe')
probe=json.loads(subprocess.check_output([str(ffprobe),'-v','error','-count_frames','-select_streams','v:0',
    '-show_entries','stream=width,height,avg_frame_rate,nb_read_frames,duration,color_space,color_range,color_transfer,color_primaries',
    '-of','json',str(video)],creationflags=0x08000000))
assert probe==r['probe']
s=probe['streams'][0]
assert (s['width'],s['height'],s['avg_frame_rate'],int(s['nb_read_frames']))==(3840,2304,'60000/1001',390)
assert s['color_space']==s['color_transfer']==s['color_primaries']=='bt709' and s['color_range']=='tv'
assert abs(float(s['duration'])-6.5065)<.02
ffmpeg=ffprobe.with_name('ffmpeg.exe')
keyframes=(0,65,130,195,260,325,389)
selector='+'.join(f'eq(n\\,{i})' for i in keyframes)
decoder=subprocess.Popen([str(ffmpeg),'-v','error','-threads','2','-i',str(video),
    '-vf','select='+selector,'-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],
    stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=0x08000000)
presentation=[]
try:
    for i in keyframes:
        data=bytearray();remaining=3840*2304*3
        while remaining:
            part=decoder.stdout.read(remaining)
            if not part:raise EOFError('Incomplete encoded review decode')
            data.extend(part);remaining-=len(part)
        decoded=np.frombuffer(data,dtype='u1').reshape(2304,3840,3)
        expected_path=next(Path(p) for p in r['images'] if Path(p).name==f'frame{i:03d}-full.png')
        expected=np.asarray(Image.open(expected_path).convert('RGB'))
        difference=decoded.astype('f4')-expected.astype('f4')
        mae=float(np.abs(difference).mean());mse=float(np.square(difference).mean())
        assert mae<5,(i,mae)
        presentation.append(dict(frame=i,mean_absolute_rgb8_error=mae,
                                 psnr_db=None if mse==0 else float(10*np.log10(255**2/mse))))
        if i==195:Image.fromarray(decoded).save(OUT/'decoded-frame195.png')
    assert decoder.stdout.read()==b'' and decoder.wait(timeout=30)==0
finally:
    if decoder.poll() is None:decoder.kill();decoder.wait()
audit=dict(passed=True,report_sha256=sha(path),auditor_sha256=sha(__file__),lease_sha256=sha(lease),
           complete_frames_verified=390,new_low_arrays_reread=len(verified),new_low_array_bytes=sum(verified.values()),
           full_output_byte_comparisons_in_gpu_run=390,independent_full_cpu_replay=False,video=r['video'],
           encoded_presentation_checks=presentation,decoded_preview_sha256=sha(OUT/'decoded-frame195.png'),
           human_review='pending',new_quality_approved=False,complete_migration=False,limitation=__doc__)
target.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
print(json.dumps(audit,indent=2),flush=True)
