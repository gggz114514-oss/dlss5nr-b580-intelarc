"""Local VIDEO endpoint adapter. All pixel processing stays in native-v4/XPU."""
import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

# The Comfy launcher restricts enumeration to Level Zero. Triton's capability
# probe also needs to see the OpenCL device; otherwise it reports missing XMX/
# block-IO features. This process is dedicated to the verified B580 backend.
# Keep the exact cache's hardware/driver checks intact.
from runtime_environment import isolate, loaded_libraries
_runtime_directory, _runtime_library, _runtime_environment = isolate()

import av

FAST = Path(__file__).resolve().parents[2]
BASE = FAST.parent
D = Path('D:/Codex-NR-Experiments/nr-b580')
NATIVE = D / 'product-worker/native-v4'
RUNTIME = BASE / 'xess-tools/XeSS-R4-Offline/runtime'


def inspect_video(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        codec = stream.codec_context.name
        if codec not in ('h264', 'hevc'):
            raise ValueError('当前节点需要 H.264 或 HEVC 视频。')
        if stream.codec_context.format.name not in ('yuv420p', 'nv12'):
            raise ValueError('当前 GPU 链路仅支持 8-bit 4:2:0 视频。')
        if stream.codec_context.color_trc in (16, 18):
            raise ValueError('当前节点仅支持 SDR 视频。')
        fps = stream.average_rate
        if not fps or fps <= 0:
            raise ValueError('无法确定视频帧率。')
        pts = sorted(packet.pts for packet in container.demux(stream) if packet.pts is not None)
        if not pts:
            raise ValueError('视频没有可读取的帧。')
        step = 1 / (fps * stream.time_base)
        if any(abs((value - pts[0]) - index * step) > 1 for index, value in enumerate(pts)):
            raise ValueError('当前节点需要恒定帧率视频；不支持可变帧率。')
        return dict(width=stream.width, height=stream.height, codec=codec,
                    fps=str(fps), frames=len(pts), duration=float(len(pts) / fps))


def output_size(width, height, scale):
    if not math.isfinite(scale) or not 1 <= scale <= 2:
        raise ValueError('SR 倍率需要在 1 到 2 之间。')
    unit = math.gcd(width, height)
    count = round(unit * scale / 2) * 2
    return width // unit * count, height // unit * count


def run(request, job):
    source = Path(request['input']).resolve(strict=True)
    mode, combo = request['mode'], request['combo']
    if mode not in ('exact', 'fast') or type(combo) is not bool:
        raise ValueError('Invalid mode/pipeline')
    info = inspect_video(source)
    supported = {(256, 256), (864, 480), (1920, 1080)}
    if mode == 'exact':
        supported |= {(512, 512), (2559, 1439)}
    width, height = info['width'], info['height']
    if (width, height) not in supported or width % 2 or height % 2:
        raise ValueError(f'当前视频节点未准备 {width}×{height} 的缓存/编码支持。'
                         '快速版支持 256×256、864×480、1920×1080；精确版另支持 512×512。')
    ow, oh = output_size(width, height, float(request['scale'])) if combo else (width, height)
    ffmpeg = str(RUNTIME / 'media/ffmpeg.exe')
    raw = job / ('input.h264' if info['codec'] == 'h264' else 'input.hevc')
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-n', '-i', str(source),
                    '-map', '0:v:0', '-c:v', 'copy', '-an', '-bsf:v',
                    info['codec'] + '_mp4toannexb', str(raw)], check=True)
    sys.path[:0] = [str(FAST / 'product' / name) for name in ('worker', 'precompile', 'interop')]
    dll_dirs = [os.add_dll_directory(str(p)) for p in (
        RUNTIME / 'bin', BASE / 'xess-tools/venv/Lib/site-packages/openvino/libs',
        BASE / 'ComfyUI-aki-v3-IntelArc/python/Library/bin', NATIVE)]
    if mode == 'exact':
        from exact_kernel_package_v2 import CacheOnly, bootstrap
        context = CacheOnly(D / 'product-precompile/exact-v1/package.json')
        bootstrap(context.package['cache'])
        from triton.runtime import driver
        expected = json.loads(Path(context.package['catalog']).read_text(encoding='utf-8-sig'))['target']
        actual = driver.active.get_current_target().__dict__
        diagnostic = dict(actual=actual, expected=expected, libraries=loaded_libraries(), isolation=_runtime_environment, environment={
            key: value for key, value in os.environ.items()
            if key.startswith(('TRITON_', 'SYCL_', 'ONEAPI_', 'UR_', 'ZE_'))})
        (job / 'target-diagnostic.json').write_text(json.dumps(diagnostic, indent=2), encoding='utf-8')
        if actual != expected:
            raise RuntimeError('精确缓存设备配置不匹配，具体差异已记录到 target-diagnostic.json。')
    else:
        from fast_cached_runtime_v1 import bootstrap, DiskOnly
        bootstrap()
        context = DiskOnly()
    with context:
        if mode == 'exact':
            from nr_exact_runtime_v1 import Session
            session = Session.create(BASE / 'nr-b580')
        else:
            from nr_runtime_v1 import Session
            config = json.loads((D / 'product-v1/local-runtime-v1.json').read_text(encoding='utf-8-sig'))
            session = Session.create(BASE / 'nr-b580', D / 'product-v1/profile-v1.json', config['profile_sha256'])
        from native_callback_v2 import NativeCallback
        callback = NativeCallback(session, D / 'product-interop/nr-texture-v1-build/nr_texture_bridge_v1.dll',
                                  NATIVE / 'nr_color_v1.cso')
        scale = ow / width
        quality = 'performance' if scale >= 1.9 else 'balanced' if scale >= 1.6 else 'quality' if scale >= 1.4 else 'ultra-quality'
        args = ['nr-worker', '--input', str(raw), '--codec', info['codec'], '--max-frames', str(info['frames']),
                '--report', str(job / 'native-report.json'), '--shader-dir', str(RUNTIME / 'shaders/common'),
                '--gpu-mode', 'sr-fg' if combo else 'sr', '--motion-backend', 'gpu-block',
                '--fps', str(float(Fraction(info['fps']))),
                '--output-width', str(ow), '--output-height', str(oh), '--xess-quality', quality,
                '--gpu-post', 'off', '--qsv-out', str(job / 'output.h264'), '--terminal-encoder', 'h264_qsv',
                '--cancel-file', str(job / 'cancel.flag')]
        if combo:
            args += ['--depth-model', str(RUNTIME / 'models/depth-anything-v2-small/depth_anything_v2_small.xml')]
        try:
            result = callback.run(NATIVE / 'nr_worker_v1.dll', args, single=not combo)
        finally:
            callback.close()
        if not result['passed'] or result['nr_frames'] != info['frames'] or result['retired'] != info['frames']:
            raise RuntimeError('NR 未完整处理输入，详见 native-report.json。')
    fps = Fraction(info['fps']) * (2 if combo else 1)
    output = job / 'nr-video.mp4'
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-n', '-r', str(fps),
                    '-i', str(job / 'output.h264'), '-i', str(source), '-map', '0:v:0', '-map', '1:a:0?',
                    '-c', 'copy', '-t', str(info['duration']), '-movflags', '+faststart', str(output)], check=True)
    actual = inspect_video(output)
    expected = info['frames'] * 2 - 1 if combo else info['frames']
    if actual['frames'] != expected or (actual['width'], actual['height']) != (ow, oh) or Fraction(actual['fps']) != fps:
        raise RuntimeError(f'输出帧数/尺寸/帧率校验失败：{actual}，预期 {expected} 帧。')
    # Only our own disposable elementary streams, after successful mux verification.
    raw.unlink()
    (job / 'output.h264').unlink()
    return dict(passed=True, output=str(output), mode=mode, combo=combo, input=info,
                video=actual, actual_scale=ow / width, runtime_compilation=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    options = parser.parse_args()
    job = options.request.resolve().parent
    try:
        result = run(json.loads(options.request.read_text(encoding='utf-8')), job)
    except BaseException:
        result = dict(passed=False, error=traceback.format_exc())
        (job / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        traceback.print_exc()
        sys.exit(1)
    (job / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
