"""Repair missing H264 VUI colour tags without re-encoding the completed review."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import traceback
import numpy as np

DREF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
OLD = DREF / 'results/long-precision-1080-review-v1'
OUT = DREF / 'results/long-precision-1080-review-v2'
assert not OUT.exists()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
js = lambda path: json.loads(Path(path).read_text(encoding='utf-8-sig'))
old_report = OLD / 'validation.json'
r = js(old_report)
assert not r['passed'] and r['error'] == "KeyError('color_transfer')" and len(r['frames']) == 390
assert all(sha(path) == digest for path, digest in r['sources'].items())
ffmpeg = next(path for path in r['sources'] if Path(path).name == 'ffmpeg.exe')
ffprobe = str(Path(ffmpeg).with_name('ffprobe.exe'))
old_video = OLD / 'comparison-full-59.94fps.mp4'
images = sorted(OLD.glob('*.png'))
assert len(images) == 14
frozen = {str(path): sha(path) for path in [Path(__file__), old_report, Path(str(OLD) + '.log'), old_video, *images]}
OUT.mkdir()
video = OUT / old_video.name
report = dict(scope=__doc__, passed=False, sources=frozen, render_sources=r['sources'],
    original_render_exit_code=1, original_render_error=r['error'], reencoded=False,
    native_capture_for_this_sequence=False, complete_migration=False, human_review='pending',
    images={str(path): sha(path) for path in images}, reference=r['reference'],
    metric_data=r['metric_data'], temporal_metric_scope=r['temporal_metric_scope'])

def command(args):
    return subprocess.check_output(args, stderr=subprocess.PIPE, creationflags=0x08000000)

def slices(path):
    stream = command([ffmpeg, '-v', 'error', '-i', str(path), '-map', '0:v:0', '-c:v', 'copy',
                      '-bsf:v', 'h264_mp4toannexb', '-f', 'h264', 'pipe:1'])
    nals = re.split(b'\x00\x00(?:\x00)?\x01', stream)
    payloads = [nal for nal in nals if nal and (nal[0] & 31) in (1, 5)]
    assert payloads
    digest = hashlib.sha256()
    for nal in payloads:
        digest.update(len(nal).to_bytes(8, 'little'))
        digest.update(nal)
    return dict(vcl_nal_count=len(payloads), bytes=sum(map(len, payloads)), sha256=digest.hexdigest())

def decoded_rgb(path):
    process = subprocess.Popen([ffmpeg, '-v', 'error', '-threads', '2', '-i', str(path),
        '-vf', 'scale=in_color_matrix=bt709:in_range=tv,format=rgb24', '-filter_threads', '2',
        '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '2', 'pipe:1'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=0x08000000)
    digest, count = hashlib.sha256(), 0
    try:
        while True:
            data = process.stdout.read(1024 * 1024)
            if not data:
                break
            count += len(data)
            digest.update(data)
        assert process.wait(timeout=30) == 0, process.stderr.read().decode(errors='replace')
        assert count == 390 * 3840 * 2304 * 3
        return dict(bytes=count, sha256=digest.hexdigest(), decoded_frames=390)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

try:
    fix = 'h264_metadata=video_full_range_flag=0:colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1'
    command([ffmpeg, '-v', 'error', '-i', str(old_video), '-map', '0:v:0', '-c:v', 'copy',
        '-bsf:v', fix, '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
        '-color_range', 'tv', '-movflags', '+faststart', '-n', str(video)])
    probe = json.loads(command([ffprobe, '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames,duration,color_space,color_range,color_transfer,color_primaries',
        '-of', 'json', str(video)]))
    s = probe['streams'][0]
    assert (s['width'], s['height'], s['avg_frame_rate'], int(s['nb_read_frames'])) == (3840, 2304, '60000/1001', 390)
    assert s['color_space'] == s['color_transfer'] == s['color_primaries'] == 'bt709' and s['color_range'] == 'tv'
    assert abs(float(s['duration']) - 390 * 1001 / 60000) < .02
    vcl_before, vcl_after = slices(old_video), slices(video)
    assert vcl_before == vcl_after
    print('Colour metadata fixed; complete compressed picture payload unchanged', flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        before, after = list(pool.map(decoded_rgb, (old_video, video)))
    assert before == after
    quality = {}
    for mode in ('fp16_xmx', 'int8_fused_v2'):
        rows = [frame['modes'][mode] for frame in r['frames']]
        mse = statistics.mean(row['mse'] for row in rows)
        quality[mode] = dict(aggregate_psnr_db=None if mse == 0 else float(-10 * np.log10(mse)),
            worst_psnr_db=min(row['psnr_db'] for row in rows if row['psnr_db'] is not None),
            max_abs=max(row['max_abs'] for row in rows),
            mean_sampled_ssim=statistics.mean(row['ssim'] for row in rows if 'ssim' in row),
            mean_flow_compensated_error_change_mae=statistics.mean(row['flow_compensated_error_change_mae'] for row in rows if 'flow_compensated_error_change_mae' in row))
    report.update(passed=True, frames_completed=390, probe=probe, video=dict(path=str(video), sha256=sha(video), bytes=video.stat().st_size),
        compressed_picture_payload=vcl_after, complete_decoded_rgb=after, decoded_pixels_unchanged=True, quality=quality)
except Exception as error:
    report.update(passed=False, error=repr(error), traceback=traceback.format_exc())
    raise
finally:
    assert all(sha(path) == digest for path, digest in frozen.items())
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
print(json.dumps({key: report[key] for key in ('passed', 'frames_completed', 'quality', 'video', 'decoded_pixels_unchanged')}), flush=True)
