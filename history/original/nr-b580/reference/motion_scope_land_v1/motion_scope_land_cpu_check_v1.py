"""落地后 CPU 预检：`NR_MOTION_SCOPE` 两种模式 vs 备份原版 / 单点采样。

不占 GPU、不建租约、不改任何文件。回答七件事：

  1. 两棵树的 `backend/nr_backend/` 是否**仍逐字节一致**
     （`rows_paths_v1.duplicate_source_agreement()` 要求；只改一棵会让两棵树不再一致）；
  2. 备份 sha256 是否等于落地记录里的**落地前**值 `fb13d07c…`；
  3. `off` 模式 vs **AST 抽取的备份原函数**：逐位相同 ⇒ `off` 分支是逐字复现（技能 129）；
  4. `on` 模式在退化输入上 vs **单点采样 oracle**：逐位相同 ⇒ 短路的语义正确；
  5. ★ `on` vs `off`：**近似退化时**不逐位相同（恰好退化时相同）—— 这把刀**不是**逐位恒等，
     写成一条**会 FAIL 的断言**（技能 137：反直觉的例外要写成会 FAIL 的断言，而不是注释）。
     ⚠️ 恰好整数 motion 在 **CPU** 上残差为 0（`fma32` 走 double）⇒ 必须**手工注入 XPU 残差
     2.5749e-05** 才能复现"会变画面"的那一形态 —— 否则这条断言会**假通过**；
  6. `NR_MOTION_SCOPE_TOL=0` ⇒ `on` ≡ `off` 逐位相同 ⇒ 证明**门**才是那个开关；
  7. 亚像素输入 ⇒ `on` ≡ `off`（回落生效，判据有区分力）。

⚠️ 这里测的是**落地后文件里的版本**，不是 `motion_scope_v1` 那个 monkey-patch。
   两者逻辑等价，但"等价"必须是**测量**出来的（`degen_land_v1/RESULT.md` §3）。
⚠️ `sampling.fma32` 在 CPU 上走 `(a.double()*b.double()+c.double()).float()`，
   在 XPU 上走 `torch.addcmul` ⇒ **本脚本证明的是结构与语义，不是 XPU 上的逐位行为**。
   XPU 逐位/画面必须另跑 GPU 全量（本轮 `motion_scope_land_full_v1.sh`）。

用法：
    python motion_scope_land_cpu_check_v1.py
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
TREES = {
    'nr-b580': Path('E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580'),
    'nr-b580-int8': Path('E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580-int8'),
}
BACKUPS = {
    'nr-b580': HERE / 'backup' / 'sampling_nr_b580_orig.py',
    'nr-b580-int8': HERE / 'backup' / 'sampling_nr_b580_int8_orig.py',
}
PRE_LANDING_SHA = 'fb13d07cdc0b0fe7636c210711f84d5f996102dd68762b10193602cc93b08b56'
TARGET = '_sample_five_axes'
ENV = 'NR_MOTION_SCOPE'
ENV_TOL = 'NR_MOTION_SCOPE_TOL'
DEFAULT_TOL = 1.0 / 256.0

FAIL = []
RES = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chk(cond, label, detail=''):
    print('  %s %s%s' % ('OK  ' if cond else 'FAIL', label, ('  | ' + detail) if detail else ''))
    RES[label] = {'pass': bool(cond), 'detail': detail}
    if not cond:
        FAIL.append(label)


def _extract_from_backup(backup: Path, name: str):
    """从备份源码里 AST 抽出原函数并编译（**不手抄**，技能 129）。"""
    tree = ast.parse(backup.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            mod = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(mod)
            return mod
    raise RuntimeError('备份里找不到 %s' % name)


def _fresh_import(backend: Path):
    """干净地导入某棵树的 `nr_backend.sampling`（两棵树同名 ⇒ 必须先清缓存）。"""
    for name in [n for n in list(sys.modules) if n.startswith('nr_backend')]:
        del sys.modules[name]
    entry = str(backend)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return importlib.import_module('nr_backend.sampling')


def _tree_agreement():
    """两棵树 backend/nr_backend/ 的逐文件比对。"""
    a, b = TREES['nr-b580'] / 'backend' / 'nr_backend', TREES['nr-b580-int8'] / 'backend' / 'nr_backend'
    fa = {p.relative_to(a).as_posix(): sha(p) for p in a.rglob('*.py')}
    fb = {p.relative_to(b).as_posix(): sha(p) for p in b.rglob('*.py')}
    same = [k for k in fa if k in fb and fa[k] == fb[k]]
    diff = [k for k in fa if k not in fb or fa[k] != fb[k]]
    extra = [k for k in fb if k not in fa]
    return len(fa), len(fb), same, diff, extra


def _center_oracle(sm, image, mx, my, normalized_scale):
    """独立复刻 `sample()` 在 `coordinates[2] == (mx, my)` 处的取值。

    这是**语义 oracle**：它来自原始 `sample()` 的定义，不是从本轮的短路分支抄的。
    """
    if normalized_scale is None:
        return sm.half_texture_linear(image, mx, my)
    h, w = image.shape[:2]
    iw, ih = normalized_scale
    u = sm.fma32(mx.clamp(.5, w - .5) * iw, w, 0) * (1 / w)
    v = sm.fma32(my.clamp(.5, h - .5) * ih, h, 0) * (1 / h)
    return sm.half_texture_normalized(image, u, v)


# XPU 上实测的残差量级（`motion_scope_v1/RESULT.md` §4：2.5749e-05 像素）。
# ⚠️ CPU 的 `fma32` 走 `(a.double()*b.double()+c.double()).float()` ⇒ **恰好整数 motion 在 CPU 上残差为 0**；
#    XPU 走 `torch.addcmul`（float32）⇒ 残差 2.5749e-05。
#    要复现"**近似**退化"（那才是画面会变的形态）必须手工注入这个量级。
XPU_RESIDUAL = 2.5749e-05


def _axes(torch, h, w, kind):
    """构造轴对。(bx, x0, gx, x3, mx) —— `kind` 决定门是否应当开火。"""
    x = torch.arange(w, dtype=torch.float32)
    y = torch.arange(h, dtype=torch.float32)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    z = torch.zeros_like(X)
    o = torch.ones_like(X)
    if kind == 'exact':                 # 恰好 (c, 0, 1, 0, c)：x0 + gx + x3 == 1
        return (X + .5, z, o, z, X + .5), (Y + .5, z, o, z, Y + .5)
    if kind == 'near':                  # 近似退化：x0 = x3 = eps、gx = 1 - 2eps（XPU 残差量级）
        e = XPU_RESIDUAL
        return (X + .5, z + e, o - 2 * e, z + e, X + .5), (Y + .5, z, o, z, Y + .5)
    if kind == 'subpixel':              # 1e-2 像素，远大于门 1/256 = 3.9e-3 ⇒ 必须回落
        e = 1e-2
        return (X + .5, z + e, o - 2 * e, z + e, X + .5), (Y + .5, z, o, z, Y + .5)
    raise ValueError(kind)


def main():
    import torch

    torch.manual_seed(0)
    h = w = 64
    image = torch.rand(h, w, 3, dtype=torch.float32)
    iw = torch.tensor(1.0 / w)
    ih = torch.tensor(1.0 / h)

    print('=' * 78)
    print('1. 两棵树 backend/nr_backend/ 逐字节一致？')
    print('=' * 78)
    na, nb, same, diff, extra = _tree_agreement()
    chk(not diff and not extra, '两棵树 %d/%d 文件一致' % (len(same), na),
        'diff=%r extra=%r' % (diff[:5], extra[:5]))
    RES['tree_files'] = {'a': na, 'b': nb, 'same': len(same), 'diff': diff, 'extra': extra}

    print()
    print('=' * 78)
    print('2. 备份 sha == 落地前 %s…' % PRE_LANDING_SHA[:8])
    print('=' * 78)
    for label, p in BACKUPS.items():
        s = sha(p)
        chk(s == PRE_LANDING_SHA, '%s 备份 sha' % label, s[:16] + '…')
        RES['backup_sha_' + label] = s

    print()
    print('=' * 78)
    print('3-5. 落地文件：off vs 备份原函数 / on vs 单点 oracle / on vs off')
    print('=' * 78)

    def _load(env_value, tol=None):
        """设好环境后**干净导入**，返回 (module, 绑定到该模块 globals 的函数)。

        ⚠️ **不能用 `importlib.reload`** —— reload 会**就地**改掉同一个模块对象，
        `sm_on is sm_off` 于是为真，两个模式变成自己跟自己比（本轮第一版就栽在这：
        `on ≠ off` 全部报 max_abs=0，看着像"恰好退化"，其实是没在测 on）。
        必须每次重新 import，拿到**不同的模块对象**；旧函数对象仍持有旧 globals。
        """
        os.environ[ENV] = env_value
        if tol is None:
            os.environ.pop(ENV_TOL, None)
        else:
            os.environ[ENV_TOL] = str(tol)
        sm = _fresh_import(TREES['nr-b580'] / 'backend')
        return sm, sm._sample_five_axes

    sm_on, fn_on = _load('on')
    ns = dict(vars(sm_on))
    exec(compile(_extract_from_backup(BACKUPS['nr-b580'], TARGET), '<backup>', 'exec'),
         ns)                                              # noqa: S102 - 受控命名空间
    orig = ns[TARGET]

    RES['on_flag'] = bool(getattr(sm_on, '_MOTION_SCOPE_ON', None))
    RES['on_tol'] = float(getattr(sm_on, '_MOTION_SCOPE_TOL', -1))
    chk(RES['on_flag'] is True and abs(RES['on_tol'] - DEFAULT_TOL) < 1e-12,
        'on 默认：开关 on、tol = 1/256', 'tol=%r' % RES['on_tol'])

    sm_off, fn_off = _load('off')
    RES['off_flag'] = bool(getattr(sm_off, '_MOTION_SCOPE_ON', None))
    chk(RES['off_flag'] is False, 'off 开关读到 False')
    chk(sm_on is not sm_off, '两个模式是**不同**的模块对象（reload 陷阱的守卫）')

    for kind in ('exact', 'near', 'subpixel'):
        ax, ay = _axes(torch, h, w, kind)
        v_on = fn_on(image, ax, ay, normalized_scale=(iw, ih))
        v_off = fn_off(image, ax, ay, normalized_scale=(iw, ih))
        v_org = orig(image, ax, ay, normalized_scale=(iw, ih))
        v_orc = _center_oracle(sm_off, image, ax[4], ay[4], (iw, ih))

        res = {'on_off_max_abs': float((v_on - v_off).abs().max())}
        res['off_orig_bit_equal'] = bool(torch.equal(v_off, v_org))
        res['off_orig_max_abs'] = float((v_off - v_org).abs().max())
        res['on_oracle_bit_equal'] = bool(torch.equal(v_on, v_orc))
        res['on_oracle_max_abs'] = float((v_on - v_orc).abs().max())
        RES['axes_' + kind] = res

        chk(res['off_orig_bit_equal'], '[%s] off ≡ 备份原函数（逐位）' % kind,
            'max_abs=%.3e' % res['off_orig_max_abs'])
        if kind == 'subpixel':
            chk(torch.equal(v_on, v_off), '[subpixel] 门不开火 ⇒ 回落，on ≡ off（逐位）',
                'max_abs=%.3e' % res['on_off_max_abs'])
        else:
            chk(res['on_oracle_bit_equal'], '[%s] 门开火 ⇒ on ≡ 单点 oracle（逐位）' % kind,
                'max_abs=%.3e' % res['on_oracle_max_abs'])
        if kind == 'exact':
            # 恰好退化 ⇒ 5-tap 的权重是精确的 0/1 ⇒ 每一步都精确 ⇒ 与短路**逐位相同**。
            chk(torch.equal(v_on, v_off), '[exact] 恰好退化 ⇒ on ≡ off（逐位）',
                'max_abs=%.3e' % res['on_off_max_abs'])
        if kind == 'near':
            # ★ 技能 137：把"这不是逐位恒等"写成一条**会 FAIL 的断言**。
            # 这是本刀**唯一**会让画面变的形态：残差非零 ⇒ 5-tap 带 O(1e-5) 扰动，
            # 而短路直接取中心 tap。CPU 上要手工注入 XPU 残差量级才能复现（见 XPU_RESIDUAL 注释）。
            chk(not torch.equal(v_on, v_off), '[near] ★ on ≠ off（本刀**不是**逐位恒等）',
                'max_abs=%.3e（应当 > 0）' % res['on_off_max_abs'])

    print()
    print('=' * 78)
    print('6. 门才是开关：NR_MOTION_SCOPE_TOL=0 ⇒ on ≡ off（逐位）')
    print('=' * 78)
    sm_t0, fn_t0 = _load('on', tol=0)
    RES['tol0'] = float(getattr(sm_t0, '_MOTION_SCOPE_TOL', -1))
    ax, ay = _axes(torch, h, w, 'near')
    v_t0 = fn_t0(image, ax, ay, normalized_scale=(iw, ih))
    v_off = fn_off(image, ax, ay, normalized_scale=(iw, ih))
    d_t0 = float((v_t0 - v_off).abs().max())
    RES['tol0_bit_equal'] = bool(torch.equal(v_t0, v_off))
    chk(RES['tol0'] == 0.0, 'tol 读到 0', 'tol=%r' % RES['tol0'])
    chk(RES['tol0_bit_equal'], '[near] tol=0 ⇒ 门不开火 ⇒ on ≡ off（逐位）',
        'max_abs=%.3e' % d_t0)
    # ★ 与 3-5 的 [near] 互为对照：**同一输入**，只把 tol 从 1/256 改成 0，
    #   行为就从"不同"变成"逐位相同" ⇒ 证明那个门真的是开关。
    RES['tol_switch_contrast'] = {'tol_1_256_max_abs': RES['axes_near']['on_off_max_abs'],
                                  'tol_0_max_abs': d_t0}
    chk(RES['axes_near']['on_off_max_abs'] > 0 and d_t0 == 0.0,
        '★ tol 对照：1/256 时不同、0 时逐位相同',
        'tol=1/256 → %.3e ｜ tol=0 → %.3e' % (RES['axes_near']['on_off_max_abs'], d_t0))
    os.environ.pop(ENV_TOL, None)

    print()
    print('=' * 78)
    print('7. 完整 warp_history_normalized（真实调用链，整数 + 亚像素 motion）')
    print('=' * 78)
    sm_won, _ = _load('on')
    sm_woff, _ = _load('off')
    dim_recip = lambda n: torch.tensor(1.0 / n)           # noqa: E731
    for mv in [(0, 0), (1, 0), (-1, 0), (0, 1), (3, -2), (0.37, 0)]:
        motion = torch.zeros(h, w, 2, dtype=torch.float32)
        motion[..., 0] = float(mv[0])
        motion[..., 1] = float(mv[1])
        a = sm_won.warp_history_normalized(image, motion, reciprocal_source=None,
                                           dimension_reciprocal=dim_recip)
        b = sm_woff.warp_history_normalized(image, motion, reciprocal_source=None,
                                            dimension_reciprocal=dim_recip)
        d = float((a - b).abs().max())
        integer = float(mv[0]).is_integer() and float(mv[1]).is_integer()
        key = 'warp_%s_%s_max_abs' % (mv[0], mv[1])
        RES[key] = d
        if integer:
            # ⚠️ CPU 上 `fma32` 走 double ⇒ 整数 motion 的残差**恰好为 0** ⇒ 精确退化 ⇒ 逐位相同。
            #    XPU 上走 `addcmul`（float32）⇒ 残差 2.5749e-05 ⇒ **会不同**
            #    （外挂轮实测 `identical 1/243`）。所以这里断言的只是 **CPU 行为**，
            #    并把它和 XPU 的差别写进 label —— 别把这条读成"落地在 XPU 上也逐位不变"。
            chk(d == 0.0, 'motion=%-10r CPU 无残差 ⇒ 恰好退化，on ≡ off' % (mv,),
                'max_abs=%.3e（XPU 上应为 2.57e-05 量级）' % d)
        else:
            chk(d == 0.0, 'motion=%-10r 亚像素 ⇒ 门不开火，on ≡ off（回落）' % (mv,),
                'max_abs=%.3e' % d)

    out = HERE / 'cpu_check.json'
    out.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding='utf-8')
    print()
    print('已写 %s' % out)
    print('RESULT: ' + ('ALL PASS' if not FAIL else 'FAILURES(%d) = %r' % (len(FAIL), FAIL)))
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
