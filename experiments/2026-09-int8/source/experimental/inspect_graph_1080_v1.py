"""Saved full-array quality summary and static review; no long-video approval claim."""
import hashlib, json, math, statistics
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import compressed_arrays_v1 as compressed

HERE = Path(__file__).resolve().parent
EXACT = HERE.parent.parent / 'nr-b580'
DREF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT = DREF / 'results/graph-native-1080-review-v1'
assert not OUT.exists()
js = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
modes = ('baseline', 'fp16_xmx', 'int8_fused_v2')
reports = {}
sources = {str(Path(__file__)): sha(Path(__file__))}
for mode in modes:
    path = DREF / f'results/graph-native-1920x1080-{mode}-v4/validation.json'
    audit_path = path.with_name('saved-audit-v1.json')
    report = js(path)
    audit = js(audit_path)
    assert report['passed'] and audit['passed'] and audit['report_sha256'] == sha(path)
    sources.update({str(path): sha(path), str(audit_path): sha(audit_path)})
    reports[mode] = report
inputs = EXACT / 'reference/inputs/flow-full-1920x1080-v3'
manifest_path = inputs / 'manifest.json'
manifest = js(manifest_path)
sources[str(manifest_path)] = sha(manifest_path)
OUT.mkdir()
font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 22)
small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
labels = ['原始画面', '4060 参考 = B580 精确输出', 'B580 FP16 XMX', 'B580 INT8 + FP16 注意力']
summary = {}

def load(meta):
    if meta['format'] == 'npy+zlib':
        return compressed.load(meta).astype('<f4')
    assert sha(meta['path']) == meta['sha256']
    return np.fromfile(meta['path'], '<f4').reshape(1080, 1920, 4)[..., :3].copy()

for mode in modes:
    rows = reports[mode]['runs']['graph'][:12]
    mse_values = []
    for row in rows:
        actual, expected = load(row['output']), load(row['native'])
        mse = float(np.square(actual.astype('f8') - expected.astype('f8')).mean())
        assert mse == row['mse']
        mse_values.append(mse)
    mse = statistics.mean(mse_values)
    summary[mode] = dict(aggregate_psnr_db=None if mse == 0 else -10 * math.log10(mse),
                         worst_psnr_db=None if mse == 0 else min(r['psnr_db'] for r in rows),
                         max_abs=max(r['max_abs'] for r in rows),
                         graph_mean_seconds=reports[mode]['matching_mean_seconds']['graph'])

for i in (0, 11):
    spec = manifest['frames'][i]
    assert sha(inputs / spec['file']) == spec['sha256']
    original = Image.open(inputs / spec['file']).convert('RGB')
    pictures = [original]
    for mode in modes:
        value = load(reports[mode]['runs']['graph'][i]['output'])
        pictures.append(Image.fromarray(np.rint(np.clip(value, 0, 1) * 255).astype('u1')))
    for cropped in (False, True):
        view_w, view_h = (1280, 400) if cropped else (960, 540)
        canvas = Image.new('RGB', (view_w * 2, (view_h + 58) * 2), (19, 24, 32))
        draw = ImageDraw.Draw(canvas)
        for j, (label, picture) in enumerate(zip(labels, pictures)):
            x, y = (j % 2) * view_w, (j // 2) * (view_h + 58)
            draw.text((x + 10, y + 3), label, font=font, fill='white')
            draw.text((x + 10, y + 33), f'源帧 {spec["source_frame"]} | ' + ('1:1 局部' if cropped else '1080p 缩略展示') + ' | 静态检查', font=small, fill=(183, 194, 209))
            picture = picture.crop((384, 368, 1664, 768)) if cropped else picture.resize((view_w, view_h), Image.Resampling.LANCZOS)
            canvas.paste(picture, (x, y + 58))
        canvas.save(OUT / f'frame{i:02d}-{"crop" if cropped else "full"}.png')
audit = dict(passed=True, sources=sources, quality=summary,
             pictures={str(p): sha(p) for p in OUT.glob('*.png')},
             scope='12 consecutive source frames 180..191 (about 0.2 seconds at 60000/1001fps); two static views. Full arrays independently verified. This is not long temporal or human approval.',
             human_review='not_requested_yet; prepare sufficiently long 1080p comparison first',
             complete_migration=False)
(OUT / 'inspection.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(summary), flush=True)
