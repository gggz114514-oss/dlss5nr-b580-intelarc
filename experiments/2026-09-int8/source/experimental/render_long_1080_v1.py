"""Render a full-resolution four-panel390-frame review from audited lossless arrays."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import traceback
import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity
import compressed_arrays_v1 as arrays

HERE = Path(__file__).resolve().parent
DREF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT = DREF / 'results/long-precision-1080-review-v1'
assert not OUT.exists()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
js = lambda path: json.loads(Path(path).read_text(encoding='utf-8-sig'))
modes = ('baseline', 'fp16_xmx', 'int8_fused_v2')
paths = {mode: DREF / f'results/long-precision-1080-{mode}-v1/validation.json' for mode in modes}
reports = {mode: js(path) for mode, path in paths.items()}
for mode, r in reports.items():
    audit = js(paths[mode].with_name('saved-audit-v1.json'))
    assert r['passed'] and r['mode'] == mode and r['frames_completed'] == len(r['frames']) == 390
    assert audit['passed'] and audit['report_sha256'] == sha(paths[mode])
    assert not r['native_capture_for_this_sequence'] and not r['complete_migration']
    assert all(sha(path) == digest for path, digest in r['sources'].items())
base = reports['baseline']
source = Path(base['source'])
ffmpeg = Path(next(path for path in base['sources'] if Path(path).name == 'ffmpeg.exe'))
ffprobe = ffmpeg.with_name('ffprobe.exe')
encoder = Path(imageio_ffmpeg.get_ffmpeg_exe())
font_path = Path('C:/Windows/Fonts/msyh.ttc')
font = ImageFont.truetype(str(font_path), 36)
small = ImageFont.truetype(str(font_path), 24)
frozen_paths = [Path(__file__), HERE / 'compressed_arrays_v1.py', HERE / 'immutable_artifacts_v1.py',
    source, ffmpeg, ffprobe, encoder, font_path, *paths.values(), *[path.with_name('saved-audit-v1.json') for path in paths.values()]]
frozen = {str(path): sha(path) for path in frozen_paths}
OUT.mkdir()
video = OUT / 'comparison-full-59.94fps.mp4'
report = dict(scope=__doc__, sources=frozen, passed=False, frames=[], complete_migration=False,
    native_capture_for_this_sequence=False, human_review='pending',
    video_scope='390 source frames at60000/1001fps; each1920x1080 panel unscaled; offline inference, original-speed playback',
    reference='B580 exact35 arithmetic with its own uninterrupted history, not a new native4060 long capture',
    display_conversion='Raw half RGB clipped to[0,1], rounded toRGB8; H264 yuv420p BT709 limited, CRF12. This presentation does not replace lossless model data.',
    temporal_metric_scope='Change in fast-minus-exact error after current-to-previous DIS warp; spatially valid pixels only, no occlusion ground truth; diagnostic, not visual acceptance',
    metric_data='Original complete half RGB converted tofloat32, before display clipping or encoding')
decoder = writer = None
labels = ['原始画面', 'B580 精确后端（本片参考）', 'B580 FP16 XMX', 'B580 INT8']
keyframes = (0, 65, 130, 195, 260, 325, 389)
previous_error = {}
yy, xx = np.indices((1080, 1920), dtype='f4')
cv2.setNumThreads(2)
cv2.ocl.setUseOpenCL(False)

def save():
    temp = OUT / 'progress.tmp'
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    temp.replace(OUT / 'validation.json')

def read_frame():
    chunks, remaining = [], 1080 * 1920 * 3
    while remaining:
        data = decoder.stdout.read(remaining)
        if not data:
            raise EOFError('Incomplete source decode')
        chunks.append(data)
        remaining -= len(data)
    return np.frombuffer(b''.join(chunks), dtype='u1').reshape(1080, 1920, 3)

def grid(i, pictures, crop=None):
    w, h = (1920, 1080) if crop is None else (crop[2] - crop[0], crop[3] - crop[1])
    canvas = Image.new('RGB', (w * 2, (h + 72) * 2), (19, 24, 32))
    draw = ImageDraw.Draw(canvas)
    for j, (label, picture) in enumerate(zip(labels, pictures)):
        x, y = (j % 2) * w, (j // 2) * (h + 72)
        draw.text((x + 14, y + 1), label, font=font, fill='white')
        note = f'源帧 {i:03d}/389  |  {i * 1001 / 60000:.3f} / 6.507 秒  |  原速回放 59.94 fps'
        draw.text((x + 14, y + 42), note, font=small, fill=(182, 194, 210))
        im = Image.fromarray(picture)
        canvas.paste(im if crop is None else im.crop(crop), (x, y + 72))
    return canvas

try:
    decoder = subprocess.Popen([str(ffmpeg), '-v', 'error', '-threads', '2', '-i', str(source),
        '-vf', base['decode_filter'], '-frames:v', '390', '-fps_mode', 'passthrough', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', 'pipe:1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=0x08000000)
    writer = subprocess.Popen([str(encoder), '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '3840x2304', '-r', '60000/1001', '-i', '-', '-an',
        '-vf', 'scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p',
        '-c:v', 'libx264', '-threads', '4', '-preset', 'fast', '-crf', '12',
        '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709', '-color_range', 'tv',
        '-movflags', '+faststart', '-n', str(video)], stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=0x08000000)
    for i in range(390):
        pixels = read_frame()
        source_digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        outputs = {}
        flow_meta = base['frames'][i]['motion']
        for mode in modes:
            row = reports[mode]['frames'][i]
            assert row['input_rgb8_sha256'] == source_digest and row['motion'] == flow_meta
            outputs[mode] = arrays.load(row['output'])
            assert outputs[mode].dtype == np.dtype('f2') and outputs[mode].shape == (1080, 1920, 3)
        expected = outputs['baseline'].astype('f4')
        flow = arrays.load(flow_meta).astype('f4')
        fx, fy = xx + flow[..., 0], yy + flow[..., 1]
        valid = (fx >= 1) & (fx < 1919) & (fy >= 1) & (fy < 1079)
        metrics = dict(frame=i, input_rgb8_sha256=source_digest, flow_valid_fraction=float(valid.mean()), modes={})
        for mode in modes[1:]:
            difference = outputs[mode].astype('f4') - expected
            mse = float(np.square(difference.astype('f8')).mean())
            assert mse == reports[mode]['frames'][i]['mse']
            row = dict(mse=mse, psnr_db=None if mse == 0 else float(-10 * np.log10(mse)),
                max_abs=float(np.abs(difference).max()), mae=float(np.abs(difference).mean()))
            if i % 60 == 0 or i == 389:
                row['ssim'] = float(structural_similarity(expected, outputs[mode].astype('f4'), data_range=1, channel_axis=2))
            if mode in previous_error and valid.any():
                warped = cv2.remap(previous_error[mode], fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                row['flow_compensated_error_change_mae'] = float(np.abs(difference - warped)[valid].mean())
            previous_error[mode] = difference
            metrics['modes'][mode] = row
        pictures = [pixels] + [np.rint(np.clip(outputs[mode].astype('f4'), 0, 1) * 255).astype('u1') for mode in modes]
        canvas = grid(i, pictures)
        writer.stdin.write(np.asarray(canvas).tobytes())
        if i in keyframes:
            canvas.save(OUT / f'frame{i:03d}-full.png')
            grid(i, pictures, (192, 320, 1728, 896)).save(OUT / f'frame{i:03d}-crop.png')
        report['frames'].append(metrics)
        if i % 30 == 0 or i == 389:
            save()
            print(json.dumps(dict(frame=i, total=390, rendered=True)), flush=True)
    assert decoder.stdout.read() == b''
    assert decoder.wait(timeout=30) == 0, decoder.stderr.read().decode(errors='replace')
    writer.stdin.close()
    assert writer.wait(timeout=60) == 0, writer.stderr.read().decode(errors='replace')
    probe = json.loads(subprocess.check_output([str(ffprobe), '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames,duration,color_space,color_range,color_transfer,color_primaries',
        '-of', 'json', str(video)], creationflags=0x08000000))
    stream = probe['streams'][0]
    assert (stream['width'], stream['height'], stream['avg_frame_rate'], int(stream['nb_read_frames'])) == (3840, 2304, '60000/1001', 390)
    assert stream['color_space'] == stream['color_transfer'] == stream['color_primaries'] == 'bt709'
    assert stream['color_range'] == 'tv'
    assert abs(float(stream['duration']) - 390 * 1001 / 60000) < .02
    quality = {}
    for mode in modes[1:]:
        rows = [frame['modes'][mode] for frame in report['frames']]
        mse = statistics.mean(row['mse'] for row in rows)
        quality[mode] = dict(aggregate_psnr_db=None if mse == 0 else float(-10 * np.log10(mse)),
            worst_psnr_db=min(row['psnr_db'] for row in rows if row['psnr_db'] is not None),
            max_abs=max(row['max_abs'] for row in rows),
            mean_sampled_ssim=statistics.mean(row['ssim'] for row in rows if 'ssim' in row),
            mean_flow_compensated_error_change_mae=statistics.mean(row['flow_compensated_error_change_mae'] for row in rows if 'flow_compensated_error_change_mae' in row))
    report.update(passed=True, frames_completed=390, quality=quality, probe=probe,
        video=dict(path=str(video), sha256=sha(video), bytes=video.stat().st_size),
        images={path.name: sha(path) for path in sorted(OUT.glob('*.png'))})
except Exception as error:
    report.update(passed=False, error=repr(error), traceback=traceback.format_exc())
    raise
finally:
    for process in (decoder, writer):
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
    assert all(sha(path) == digest for path, digest in frozen.items())
    save()
print(json.dumps({key: report[key] for key in ('passed', 'frames_completed', 'quality', 'video')}), flush=True)
