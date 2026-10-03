"""开关自证：这个臂的 `NR_MOTION_SCOPE` 到底读到什么、有没有真的换实现。

为什么需要
----------
ABBA 的帧墙差只有在"**这个臂确实在跑它声称的那套代码**"时才有意义。
`allmerge_land` 的自证方式是数整数输入下的整图归约数；本轮没有计数器，
所以换一种等价的自证：

  在**近似退化**输入上（x0 = x3 = 2.5749e-05、gx = 1 - 2*eps，即 XPU 实测残差量级），
  比较**本臂的 `_sample_five_axes`** 与 **AST 抽取的落地前原函数**：

    on  臂 ⇒ 短路生效 ⇒ 与落地前**不同**（max_abs ≈ 5.04e-05，> 0）
    off 臂 ⇒ 逐字复现落地前 ⇒ 与落地前**逐位相同**（max_abs == 0）

⇒ `on` 臂必须 > 0、`off` 臂必须 == 0，否则这个臂的帧墙差值不可用（臂没跑成它声称的形态）。

⚠️ 纯 CPU，不占 GPU、不建租约、不改任何文件。**不写 `validation.json`、不参与门槛。**
⚠️ CPU 的 `fma32` 走 double ⇒ **恰好**整数 motion 在 CPU 上残差为 0、`on ≡ off`；
   要复现"会变画面"的形态必须手工注入 XPU 残差量级（见 `motion_scope_land_cpu_check_v1.py`）。

用法（必须与臂**同一个环境**）：
    NR_MOTION_SCOPE=on python motion_scope_land_gate_probe_v1.py --out <臂目录>
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BACKUP = HERE / 'backup' / 'sampling_nr_b580_orig.py'
BACKEND = Path('E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580/backend')
TARGET = '_sample_five_axes'
ENV = 'NR_MOTION_SCOPE'
ENV_TOL = 'NR_MOTION_SCOPE_TOL'
XPU_RESIDUAL = 2.5749e-05


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _extract_from_backup(name: str):
    tree = ast.parse(BACKUP.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            mod = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(mod)
            return mod
    raise RuntimeError('备份里找不到 %s' % name)


def _fresh_import(backend: Path):
    for name in [n for n in list(sys.modules) if n.startswith('nr_backend')]:
        del sys.modules[name]
    entry = str(backend)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return importlib.import_module('nr_backend.sampling')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True, help='臂目录（写 gate_probe.json）')
    args = parser.parse_args()

    import torch

    sm = _fresh_import(BACKEND)
    ns = dict(vars(sm))
    exec(compile(_extract_from_backup(TARGET), '<backup>', 'exec'), ns)   # noqa: S102
    orig = ns[TARGET]

    h = w = 64
    image = torch.rand(h, w, 3, dtype=torch.float32)
    x = torch.arange(w, dtype=torch.float32)
    y = torch.arange(h, dtype=torch.float32)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    z = torch.zeros_like(X)
    o = torch.ones_like(X)

    # 近似退化（门应当开火）
    e = XPU_RESIDUAL
    axn, ayn = (X + .5, z + e, o - 2 * e, z + e, X + .5), (Y + .5, z, o, z, Y + .5)
    # 恰好退化
    axe, aye = (X + .5, z, o, z, X + .5), (Y + .5, z, o, z, Y + .5)
    # 亚像素（门不应当开火）
    s = 1e-2
    axs, ays = (X + .5, z + s, o - 2 * s, z + s, X + .5), (Y + .5, z, o, z, Y + .5)

    iw, ih = torch.tensor(1.0 / w), torch.tensor(1.0 / h)
    scale = (iw, ih)

    near_now = sm._sample_five_axes(image, axn, ayn, normalized_scale=scale)
    near_pre = orig(image, axn, ayn, normalized_scale=scale)
    sub_now = sm._sample_five_axes(image, axs, ays, normalized_scale=scale)
    sub_pre = orig(image, axs, ays, normalized_scale=scale)
    exact_now = sm._sample_five_axes(image, axe, aye, normalized_scale=scale)
    exact_pre = orig(image, axe, aye, normalized_scale=scale)

    env_value = os.environ.get(ENV)
    flag = bool(getattr(sm, '_MOTION_SCOPE_ON', False))
    tol = float(getattr(sm, '_MOTION_SCOPE_TOL', -1))
    near_d = float((near_now - near_pre).abs().max())
    sub_d = float((sub_now - sub_pre).abs().max())
    exact_d = float((exact_now - exact_pre).abs().max())

    ok = ((flag is True and near_d > 0) if env_value != 'off'
          else (flag is False and near_d == 0))
    ok = ok and (sub_d == 0.0) and (exact_d == 0.0)

    doc = {
        'probe': 'motion_scope_land_gate_probe_v1',
        'env_name': ENV,
        'env_value': env_value,
        'tol_env': os.environ.get(ENV_TOL),
        'flag': flag,
        'tol': tol,
        'backend': str(BACKEND),
        'backup_sha256': sha(BACKUP),
        'sampling_sha256': sha(BACKEND / 'nr_backend' / 'sampling.py'),
        'near_degenerate_max_abs_vs_pre': near_d,
        'subpixel_max_abs_vs_pre': sub_d,
        'exact_degenerate_max_abs_vs_pre': exact_d,
        'expect': 'on -> near>0 / off -> near==0；两者都要求 subpixel==0 与 exact==0',
        'ok': bool(ok),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    p = args.out / 'gate_probe.json'
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')

    print('[gate_probe] env %s=%r  flag=%s  tol=%r' % (ENV, env_value, flag, tol))
    print('[gate_probe] near   vs pre: max_abs=%.3e  (on 应 >0 / off 应 ==0)' % near_d)
    print('[gate_probe] subpix vs pre: max_abs=%.3e  (应 ==0)' % sub_d)
    print('[gate_probe] exact  vs pre: max_abs=%.3e  (应 ==0)' % exact_d)
    print('[gate_probe] wrote %s' % p)
    print('[gate_probe] %s' % ('OK' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
