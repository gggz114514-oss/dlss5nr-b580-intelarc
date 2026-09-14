"""CPU-only saved-frame attribution; no product changes or model inference.

Oracle is a diagnostic using the reference answer, not a deployable method.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import traceback
import numpy as np

BASE = Path(__file__).resolve().parents[2]
GEOMETRY = Path('D:/Codex-NR-Experiments/nr-b580/fast-difference-attribution-v1/geometry')


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, scope='six saved frames, no new temporal inference', frames=[])
    try:
        # Read only the pure table builder; never import GPU runtime or initialize GPU.
        source = BASE/'nr-b580-int8/experimental/residual_scale_v1.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'filter_table')
        ns = dict(np=np, math=math)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), ns)
        table = ns['filter_table']
        report['filter_source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        geometry = json.loads((GEOMETRY/'validation.json').read_text(encoding='utf-8'))
        assert geometry['passed']

        def axis(x, dst, kind, ax, half):
            indices, weights = table(x.shape[ax], dst, kind)
            shape = list(x.shape); shape[ax] = dst
            y = np.zeros(shape, dtype='f4')
            for tap in range(indices.shape[1]):
                wshape = [1, 1, 1]; wshape[ax] = dst
                y += np.take(x, indices[:, tap], axis=ax).astype('f4') * weights[:, tap].reshape(wshape)
            return y.astype('f2').astype('f4') if half else y

        def down(x, half=False):
            return axis(axis(x, 142, 'lanczos2', 0, half), 256, 'lanczos2', 1, half)

        def up(x, half=True):
            return axis(axis(x, 864, 'catmull', 1, half), 480, 'catmull', 0, False)

        def metrics(a, b):
            result = {}
            for name, sl in [('full', (slice(None), slice(None))), ('face_roi', (slice(16,400), slice(256,640)))]:
                d = (a[sl].astype('f8')-b[sl].astype('f8'))*255
                result[name] = dict(rmse=float(np.sqrt(np.mean(d*d))), mean_rgb=d.mean((0,1)).tolist(), max_abs=float(np.max(np.abs(d))))
            return result

        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
        for row in geometry['frames']:
            if 'artifact' not in row:
                continue
            p = Path(row['artifact']['path'])
            assert hashlib.sha256(p.read_bytes()).hexdigest() == row['artifact']['sha256']
            with np.load(p, allow_pickle=False) as a:
                original=a['source'].astype('f4'); exact=a['full_exact'].astype('f4')
                low=a['low_input'].astype('f4')[57:199]
                nr=a['low_exact'].astype('f4')[57:199]
                saved=a['low_exact_composite'].astype('f4')
            residual = nr-low
            replay = np.clip(original+up(residual.astype('f2').astype('f4')), 0, 1)
            replay_error = metrics(replay, saved)
            # CPU replay must closely reproduce saved GPU output before attribution.
            assert replay_error['full']['max_abs'] <= 0.02, replay_error
            down_delta = (down(original, True).astype('f8')-low)*255
            down_error = dict(max_abs=float(np.max(np.abs(down_delta))), rmse=float(np.sqrt(np.mean(down_delta*down_delta))))
            assert down_error['max_abs'] <= 0.02, down_error
            precise_storage = np.clip(original+up(residual, False), 0, 1)
            reference_residual = exact-original
            oracle_low = down(reference_residual)
            oracle = np.clip(original+up(oracle_low.astype('f2').astype('f4')), 0, 1)
            direct_up = np.clip(up(nr), 0, 1)
            results = dict(replay_vs_saved=replay_error,
                low_input_replay=down_error,
                current_geometry_vs_exact=metrics(saved, exact),
                reference_residual_roundtrip_vs_exact=metrics(oracle, exact),
                fp32_residual_storage_vs_current=metrics(precise_storage, saved),
                direct_upsampled_low_nr_vs_exact=metrics(direct_up, exact))
            # Low-res domain comparison avoids attributing everything to upsampling.
            delta = (residual.astype('f8')-oracle_low)*255
            results['low_domain_residual_mismatch'] = dict(rmse=float(np.sqrt(np.mean(delta*delta))), mean_rgb=delta.mean((0,1)).tolist())
            assert all(np.isfinite(x).all() for x in (replay,oracle,precise_storage,direct_up))
            report['frames'].append(dict(index=row['index'], metrics=results))
            img=Image.new('RGB',(1728,1024),(18,22,28)); draw=ImageDraw.Draw(img)
            labels=['全尺寸精确参考','精确NR256＋当前重建','参考增强差值缩小后重建（诊断）','低分辨率NR直接放大（诊断）']
            for j,(label,x) in enumerate(zip(labels,[exact,saved,oracle,direct_up])):
                xx=(j%2)*864; yy=(j//2)*512
                draw.text((xx+8,yy+4),label,font=font,fill='white')
                img.paste(Image.fromarray(np.rint(np.clip(x,0,1)*255).astype('u1')),(xx,yy+32))
            img.save(out/f"comparison-{row['index']:03d}.png")
        assert len(report['frames']) == 6
        report['limitations'] = ['Six selected frames only, not full-clip statistics.',
            'Reference residual oracle uses unavailable reference outputs; not a proposed product.',
            'Low-domain mismatch includes resolution, padding, motion and temporal interactions.',
            'FP32 variant changes residual/intermediate storage only; does not restore full input detail.',
            'Direct upscale also removes original high-frequency detail; not an isolated filter comparison.',
            'RMSE values are nonadditive; no causal percentage attribution.']
        report['passed'] = True
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        (out/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
