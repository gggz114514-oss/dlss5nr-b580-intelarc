"""CPU-only DIS budget on the same 11 continuous1080 reference pairs.

No NR output is generated or approved. Reduced optical flow changes model input;
endpoint/warp errors below are diagnostics against DIS, not engine ground truth.
Image decode, verification and storage are outside timing. No GPU work.
"""
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DREF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
INPUTS = DREF / 'inputs/flow-full-1920x1080-v3'
OUT = DREF / 'experimental/dis-budget-v1'
assert not OUT.exists(), 'Experiments are immutable'
sys.path.insert(0, str(HERE))
import compressed_arrays_v1 as arrays

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

manifest_path = INPUTS / 'manifest.json'
assert sha(manifest_path) == '093000e4222a2a0239e0782cda998dc733d6f86b71fa6c3447108eff433658d1'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
cv2.ocl.setUseOpenCL(False)
assert not cv2.ocl.useOpenCL()
variants = {
    'full_create_2t': (1920, 1080, 2, False),
    'full_reuse_2t': (1920, 1080, 2, True),
    'full_create_8t': (1920, 1080, 8, False),
    'half_reuse_2t': (960, 540, 2, True),
    '512_reuse_2t': (512, 288, 2, True),
    '256_reuse_2t': (256, 144, 2, True),
}
sources = [Path(__file__), manifest_path, HERE / 'compressed_arrays_v1.py', HERE / 'immutable_artifacts_v1.py',
           *sorted(Path(cv2.__file__).parent.glob('*.pyd'))]
pixels, original_flow = [], []
for frame in manifest['frames'][:12]:
    image_path, motion_path = INPUTS / frame['file'], INPUTS / frame['motion_file']
    assert sha(image_path) == frame['sha256'] and sha(motion_path) == frame['motion_sha256']
    pixels.append(np.asarray(Image.open(image_path).convert('RGB')).copy())
    original_flow.append(np.fromfile(motion_path, '<f4').reshape(1080, 1920, 2))
    sources += [image_path, motion_path]
report = dict(scope=__doc__, sources={str(p): sha(p) for p in sources}, passed=False,
              opencv=cv2.__version__, numpy=np.__version__, opencl=False, variants=variants,
              measurements=[], quality=[], semantics_changed=[],
              timing='CPU RGB to gray, optional AREA shrink, DIS create/reuse+calc, then FP16 cast; '
                     'previous grayscale cached. No decode, upload, model, validation or storage. '
                     '3 rounds x11 pairs, rotated variant order per frame, 2 warmup calls per variant.',
              model_grid_diagnostic='CPU AREA resize with pixel scaling to256x144 and zero padding to256x256; '
                                    'not byte equivalence to GPU area filter arithmetic.',
              production_changed=False, human_review='not requested; no new NR video', complete_migration=False)
OUT.mkdir()

def create():
    return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

def gray(rgb, width, height):
    value = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return value if width == 1920 else cv2.resize(value, (width, height), interpolation=cv2.INTER_AREA)

def model_grid(flow, width, height):
    active = cv2.resize(flow, (256, 144), interpolation=cv2.INTER_AREA)
    active *= np.array([256 / width, 144 / height], dtype='f4')
    result = np.zeros((256, 256, 2), dtype='f2')
    result[56:200] = active
    return result

def warp_error(cur, prev, flow):
    h, w = cur.shape
    yy, xx = np.indices((h, w), dtype='f4')
    px, py = xx + flow[..., 0], yy + flow[..., 1]
    valid = (px >= 1) & (px < w - 2) & (py >= 1) & (py < h - 2)
    warped = cv2.remap(prev, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return dict(valid_fraction=float(valid.mean()),
                warped_luma_mae=float(np.abs(cur.astype('f4') - warped.astype('f4'))[valid].mean()))

solvers, previous_gray, fingerprints = {}, {}, {}
for name, (width, height, threads, reuse) in variants.items():
    cv2.setNumThreads(threads)
    solvers[name] = create()
    a, b = gray(pixels[0], width, height), gray(pixels[1], width, height)
    for _ in range(2):
        (solvers[name] if reuse else create()).calc(b, a, None)
print('CPU DIS warmup complete', flush=True)
for repetition in range(3):
    previous_gray = {name: gray(pixels[0], w, h) for name, (w, h, _, _) in variants.items()}
    for index in range(1, 12):
        names = list(variants)
        offset = (repetition + index) % len(names)
        for name in names[offset:] + names[:offset]:
            w, h, threads, reuse = variants[name]
            cv2.setNumThreads(threads)
            started = time.perf_counter()
            current = gray(pixels[index], w, h)
            before_calc = time.perf_counter()
            solver = solvers[name] if reuse else create()
            flow = solver.calc(current, previous_gray[name], None)
            after_calc = time.perf_counter()
            half = flow.astype('f2')
            completed = time.perf_counter()
            assert np.isfinite(half).all()
            digest = hashlib.sha256(half.tobytes()).hexdigest()
            if repetition == 0:
                fingerprints[name, index] = digest
            else:
                assert fingerprints[name, index] == digest, (name, index, 'non-deterministic flow')
            equal_old = half.tobytes() == original_flow[index].astype('f2').tobytes() if w == 1920 else False
            report['measurements'].append(dict(round=repetition, frame=index, variant=name,
                prepare_seconds=before_calc-started, calc_seconds=after_calc-before_calc,
                half_seconds=completed-after_calc, cpu_path_seconds=completed-started,
                raw_half_sha256=digest, full_size_byte_equal_old=equal_old))
            if repetition == 0:
                grid = model_grid(flow, w, h)
                target = model_grid(original_flow[index], 1920, 1080)
                epe = np.linalg.norm(grid[56:200].astype('f4')-target[56:200].astype('f4'), axis=2)
                report['quality'].append(dict(variant=name, frame=index, low_motion=arrays.save(grid),
                    low_motion_epe_percentiles=np.percentile(epe, [50, 90, 99, 100]).tolist(),
                    low_motion_mean_epe=float(epe.mean()),
                    warp=warp_error(current, previous_gray[name], flow)))
            previous_gray[name] = current
        print(json.dumps(dict(round=repetition, frame=index)), flush=True)
    (OUT / 'progress.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
report['summary'] = {}
for name in variants:
    rows = [r for r in report['measurements'] if r['variant'] == name]
    report['summary'][name] = dict(
        **{field: dict(mean=statistics.mean(r[field] for r in rows),
                      median=statistics.median(r[field] for r in rows),
                      p95=float(np.percentile([r[field] for r in rows], 95)))
           for field in ('calc_seconds', 'cpu_path_seconds')},
        round_means=[statistics.mean(r['cpu_path_seconds'] for r in rows if r['round'] == j) for j in range(3)],
        all_full_size_byte_equal_old=all(r['full_size_byte_equal_old'] for r in rows),
        flow_grid_mean_epe=statistics.mean(r['low_motion_mean_epe'] for r in report['quality'] if r['variant'] == name))
    if not report['summary'][name]['all_full_size_byte_equal_old']:
        report['semantics_changed'].append(name)
assert report['summary']['full_create_2t']['all_full_size_byte_equal_old']
assert all(sha(p) == digest for p, digest in report['sources'].items())
report['passed'] = True
(OUT / 'validation.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report['summary'], indent=2), flush=True)
