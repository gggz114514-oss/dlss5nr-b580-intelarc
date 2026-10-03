"""Complete the NR256 review after repairing only missing H264 colour tags.

The v1 GPU job completed390 comparisons, encoding and post-sequence guards, then
failed its ffprobe colour-tag check. Keep its report/exit1 unchanged. This CPU
finalizer copies the compressed stream and verifies unchanged VCL picture data.
No model replay, new frame encoding or visual acceptance is implied.
"""
import hashlib,json,re,statistics,subprocess,traceback
from pathlib import Path
D=Path('D:/Codex-NR-Experiments/nr-b580/reference')
OLD=D/'results/batched-residual-long1080-review-v1'
OUT=D/'results/batched-residual-long1080-review-v2';assert not OUT.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
old_path=OLD/'validation.json';old=js(old_path);old_lease=OLD.with_suffix('.log.lease.json')
assert not old['passed'] and old['error']=="KeyError('color_transfer')"
assert js(old_lease)['returncode']==1 and len(old['frames'])==390
assert all(sha(p)==h for p,h in old['sources'].items())
assert 'assert s[\'color_space\']' in old['traceback']
assert set(old['graphs'])=={'previous','batched'} and all(len(v)==2 for v in old['graphs'].values())
for i,row in enumerate(old['frames']):
    assert row['frame']==i and row['full_byte_equal_previous'] and row['low_byte_equal_previous']
    a,b=row['runs']['previous'],row['runs']['batched']
    assert a['next_seed']==b['next_seed']==i+1 and a['private_byte_equal_low'] and b['private_byte_equal_low']
    assert a['full_raw_sha256']==b['full_raw_sha256'] and a['low_raw_sha256']==b['low_raw_sha256']
    assert a['effective_dispatch']==b['effective_dispatch']
old_video=OLD/'comparison-full-59.94fps.mp4'
images=sorted(OLD.glob('frame*-full.png'));assert len(images)==7
frozen=dict(old['sources'])
for p in (Path(__file__),Path(__file__).with_name('Run-FinalizeBatchedResidualReviewV2.cmd'),
          old_path,old_lease,OLD.with_suffix('.log'),old_video,*images):frozen[str(p)]=sha(p)
ffmpeg=next(Path(p) for p in old['sources'] if Path(p).name=='ffmpeg.exe')
ffprobe=ffmpeg.with_name('ffprobe.exe')
OUT.mkdir();video=OUT/old_video.name
report=dict(old)
report.pop('error');report.pop('traceback')
report.update(sources=frozen,finalizer_scope=__doc__,original_render_report=str(old_path),
              original_render_exit_code=1,original_render_error=old['error'],reencoded=False,
              passed=False,images={str(p):sha(p) for p in images})

def command(args):return subprocess.check_output(args,stderr=subprocess.PIPE,creationflags=0x08000000)

def slices(path):
    stream=command([str(ffmpeg),'-v','error','-i',str(path),'-map','0:v:0','-c:v','copy',
                    '-bsf:v','h264_mp4toannexb','-f','h264','pipe:1'])
    payloads=[nal for nal in re.split(b'\x00\x00(?:\x00)?\x01',stream) if nal and (nal[0]&31) in (1,5)]
    assert payloads;digest=hashlib.sha256()
    for nal in payloads:digest.update(len(nal).to_bytes(8,'little'));digest.update(nal)
    return dict(vcl_nal_count=len(payloads),bytes=sum(map(len,payloads)),sha256=digest.hexdigest())

try:
    fix='h264_metadata=video_full_range_flag=0:colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1'
    command([str(ffmpeg),'-v','error','-i',str(old_video),'-map','0:v:0','-c:v','copy','-bsf:v',fix,
             '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-color_range','tv',
             '-movflags','+faststart','-n',str(video)])
    probe=json.loads(command([str(ffprobe),'-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=width,height,avg_frame_rate,nb_read_frames,duration,color_space,color_range,color_transfer,color_primaries',
        '-of','json',str(video)]))
    s=probe['streams'][0]
    assert (s['width'],s['height'],s['avg_frame_rate'],int(s['nb_read_frames']))==(3840,2304,'60000/1001',390)
    assert s['color_space']==s['color_transfer']==s['color_primaries']=='bt709' and s['color_range']=='tv'
    assert abs(float(s['duration'])-6.5065)<.02
    before,after=slices(old_video),slices(video);assert before==after
    report.update(passed=True,frames_completed=390,all_reduced_outputs_byte_equal=True,independent_history=True,
        reset_reproduces_first_frame=True,caller_ownership_guards_passed=True,held_outputs_survive_replay=True,
        lut_bytes_unchanged=True,post_sequence_guards_evidence='V1 authenticated source reached only the subsequent colour-tag assertion failure.',
        probe=probe,video=dict(path=str(video),sha256=sha(video),bytes=video.stat().st_size),
        compressed_picture_payload=after,compressed_pictures_unchanged=True,
        diagnostic_temporal_mean_ms={name:statistics.mean(f['runs'][name]['seconds'] for f in old['frames'][1:])*1000 for name in ('previous','batched')})
except BaseException as error:
    report.update(passed=False,error=repr(error),traceback=traceback.format_exc());raise
finally:
    assert all(sha(p)==h for p,h in frozen.items())
    (OUT/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=report['passed'],frames=report.get('frames_completed'),video=report.get('video'))),flush=True)
