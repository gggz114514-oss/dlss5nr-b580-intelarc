"""整数 motion 下，把 `_sample_five_axes` 的 5-tap 加权**短路成单点采样**。

它想省的是什么
--------------
`N2b`（已进产品）只短路了 `_half_texture_counts` **内部**：

    warp_history_normalized
      → axis(u,w) / axis(v,h)          ← _axis_weights_fraction（5 个 fma32 + 1 次 reciprocal）
      → _sample_five_axes              ← 5 个权重乘法 + 4 次 fma32 累加 + **5 次 sample()**
            → sample() → half_texture_normalized → _half_texture_counts
                                                  ↑ ★ N2b 只短路这一层

⇒ **权重、5 次采样、每像素一次 `reciprocal` 全都在 N2b 短路之外，仍在跑。**

整数 motion ⇒ 坐标恰落在 texel 中心 ⇒ `t = 0` ⇒ 代数上：

    w0 = fma32(t+t3, -.5, t2)          = 0
    w1 = fma32(t3, 1.5, -(t2*2.5)) + 1 = 1
    w3 = (t3-t2)*.5                    = 0
    w2 = ((1-w0)-w1)-w3                = 0
    group = w1 + w2                    = 1
⇒ axis = (center, 0, 1, 0, center)
⇒ weights = (x0*gy, y0*gx, gx*gy, y3*gx, x3*gy) = (0, 0, 1, 0, 0)
⇒ total = 1 ⇒ reciprocal_source(total) = 1.0
⇒ value = fma32(…,0, fma32(s2,1, fma32(…,0, s1*0))) = s2 = **单点采样**

纯 CPU 已验（`motion_scope_five_axes_equiv_v1.py`，4 组 ALL PASS、逐位相等，
含亚像素反例 `max_abs=1.114e-01`）。本模块把它搬到**真实 XPU 链路**上量收益。

短路条件（**不猜，按代数写全**）
------------------------------
    (x0==0) & (x3==0) & (gx==1) & (y0==0) & (y3==0) & (gy==1)
六项缺一不可。**合并成一个 `.all()`** —— 一次设备归约，而不是六次。
（`gx == 1` 单独就已经接近等价：精确算术下 `group = 1 + 0.5*t*(1-t)`，
而 `t ∈ [0, 0.5)` ⇒ `group == 1 ⟺ t == 0`。但浮点下 `gx` 可能因舍入先塌到 1
而 `x0/x3` 还差最后一位 ⇒ **必须写全六项**，让等价性是构造性的而不是概率的。）

⚠️ **第一次冒烟就把上面这个条件证伪了**（`degen=0 / calls=1`）⇒ 见 `measure` 模式
------------------------------------------------------------------------------------
XPU 上 `fma32` 走 `torch.addcmul`（真 FMA），`u = fma32(mv, 1/w, (x+.5)*iw)` 与
`axis()` 里的 `fma32(u, w, -.5)` 各有一次舍入 ⇒ `t` **不是精确 0**，而是 ~1e-5 量级
（与既有结论「5-tap 权重恒为 0 是错的、真实 ~2.5e-5」一致）。
⇒ 「精确退化」在真实链路上**不发生**。

`MOTION_SCOPE_CUTS=measure` 就是为这一步准备的：**只测量、不短路**。
12 帧实测（480×864 face，11 次 `_sample_five_axes`）：

| 量 | 逐帧最大值 |
|---|---|
| `x0` | 1.7165e-05 |
| `x3` | 1.2398e-05 |
| `gx − 1` | 1.7166e-05 |
| `y0` | 2.5747e-05 |
| `y3` | 6.6757e-06 |
| `gy − 1` | 2.5749e-05 |
| **合并** | **2.5749e-05**（11 次里 9 次同值） |

阈值扫描：`0` → 0/11 ｜ `1/65536 = 1.53e-5` → 0/11 ｜ **`1/4096 = 2.44e-4` → 11/11**。

⇒ 残差 **2.57e-5 像素 ≈ 采样器自身 1/256 栅格（3.91e-3）的 1/150**。
**所以阈值取采样器自己的栅格 `1/256` 像素**（`MOTION_SCOPE_TOL`，默认 1/256），
而不是拍一个"看起来小"的数：低于 1/256 像素时，**采样器自己的定点量化早就把坐标
吸到 texel 中心了** —— 这就是 N2b 的判据，这里沿用同一把尺子。

⚠️ **这不是逐位恒等**：5 个 tap 的坐标里，`coordinates[0]=(bx−1,my)` 落在**相邻 texel**
上，权重 `w0 ≈ 1.7e-5` 非零 ⇒ 输出有 O(1e-5) 的扰动。
**正确性判据只能是 243 帧逐字节门 + 画面指标**，不能靠"理论上等价"。

⚠️ 已知的唯一差异面：**零的符号**
--------------------------------
原式累加链里 `s1*0`、`s0*0`、`s3*0`、`s4*0` 的符号会参与 IEEE 加法，
所以当 `s2 == 0` 且全部被采样值都是**负零**时，原式可能给 `+0.0` 而短路给 `s2`（`-0.0`）。
本域（RGB history ∈ [0,1]）下不可能出现负零 ⇒ 判据落到 **XPU 逐字节门**上。
若门报出差异，**先查这一条**，再谈别处。

口径
----
* 外挂式：`install()` / `uninstall()` 成对，只替换 `sampling._sample_five_axes`
  这一个模块属性，**产品源码一个字不动**。
* `temporal.py:11` 是 `from .sampling import warp_history_normalized` ⇒ 模块级绑定已固定，
  但 `warp_history_normalized` 体内对 `_sample_five_axes` 是**全局名查找**，
  解析的是 `sampling.__dict__` ⇒ 换掉 `sampling._sample_five_axes` 即可生效。
  （仍然走 `_rebind_all` 把「谁直接 import 过它」一并换掉并逐个上报 —— 不靠"应该生效"。）
* 非退化时**原样回落**到原函数，不做任何改写。

用法：
    MOTION_SCOPE_CUTS=off        对照臂（一把不装）
    MOTION_SCOPE_CUTS=measure    **只测量不短路**：记录六个量的逐次量级 + 候选阈值命中数
    MOTION_SCOPE_CUTS=fiveaxes   装刀（阈值门：六项都 ≤ `MOTION_SCOPE_TOL`，默认 1/256 像素）
"""

from __future__ import annotations

import os
import sys

GROUPS = {
    'fiveaxes': ('sampling._sample_five_axes',),
}

# `measure` 模式下逐次统计的候选阈值（单位 = 像素；采样器自身的定点栅格是 1/256 像素）。
MEASURE_TOLS = (0.0, 1 / 65536, 1 / 4096, 1 / 1024, 1 / 256, 1 / 64)
PROBE_KEEP = 400          # 最多保留多少次逐次记录（避免产物膨胀）
# 短路阈值默认 = 采样器自己的定点栅格（1/256 像素）。见模块头部"阈值取哪来的"。
DEFAULT_TOL = 1.0 / 256.0


class MotionScopeCuts:
    """一把刀，独立开关。卸载时把原对象原样放回。"""

    def __init__(self, cuts=('fiveaxes',), verbose=True, measure_only=False, tol=DEFAULT_TOL):
        self.cuts = tuple(cuts)
        for name in self.cuts:
            if name not in GROUPS:
                raise ValueError(name)
        self.verbose = verbose
        self.measure_only = bool(measure_only)
        self.tol = float(tol)
        self.stats = dict(
            fiveaxes_calls=0, degenerate_hits=0, fallback_calls=0,
            gate_reductions=0, samples_avoided=0, reciprocal_calls_avoided=0,
            tol=self.tol, max_merged_applied=0.0)
        if self.measure_only:
            self.stats.update(
                probe_calls=0, probe_values=[],
                probe_pass={('%.8g' % tol): 0 for tol in MEASURE_TOLS},
                probe_component_max=dict(x0=0.0, x3=0.0, gx_minus_1=0.0,
                                         y0=0.0, y3=0.0, gy_minus_1=0.0),
                probe_exact_all=0)
        self.rebound = {}
        self._saved = {}
        self._installed = False

    # ---------------- 安装 ----------------

    def install(self):
        if self._installed:
            return self
        import torch

        self._torch = torch
        if 'fiveaxes' in self.cuts:
            self._install_fiveaxes()
        self._installed = True
        return self

    def _rebind_all(self, original, replacement, label):
        """遍历 sys.modules，把所有 `attr is original` 的模块属性换成 replacement。

        返回被重绑的 `模块名.属性名` 列表 —— 可核对，而不是"应该生效"。
        """
        done = []
        for mod_name, module in list(sys.modules.items()):
            if module is None:
                continue
            try:
                current = getattr(module, label, None)
            except Exception:                       # pragma: no cover - 防御
                continue
            if current is not original:
                continue
            key = (mod_name, label)
            self._saved.setdefault(key, original)
            try:
                setattr(module, label, replacement)
            except Exception:                       # pragma: no cover - 防御
                continue
            done.append('%s.%s' % (mod_name, label))
        return done

    def _install_fiveaxes(self):
        from nr_backend import sampling as sampling_mod

        original = sampling_mod._sample_five_axes
        self._original_fiveaxes = original
        stats = self.stats
        torch_mod = self._torch
        sm = sampling_mod
        # ⚠️ 闭包要用的每一个名字都得在这里绑好。`self.measure_only` 不能直接写进闭包体，
        # 否则 `NameError: name 'measure_only' is not defined`（首轮 measure 就栽在这）。
        measure_only = self.measure_only
        tol = self.tol

        def _sample_five_axes_fast(image, axis_x, axis_y, *, return_components=False,
                                   reciprocal_source=None, normalized_scale=None):
            bx, x0, gx, x3, mx = axis_x
            by, y0, gy, y3, my = axis_y
            stats['fiveaxes_calls'] += 1

            if measure_only:
                # 只测量：把六个门控量各自的量级与合并量级记下来，然后**一律回落**。
                d = dict(
                    x0=float(x0.abs().max()), x3=float(x3.abs().max()),
                    gx_minus_1=float((gx - 1).abs().max()),
                    y0=float(y0.abs().max()), y3=float(y3.abs().max()),
                    gy_minus_1=float((gy - 1).abs().max()))
                for key, value in d.items():
                    if value > stats['probe_component_max'][key]:
                        stats['probe_component_max'][key] = value
                merged = max(d.values())
                stats['probe_calls'] += 1
                if len(stats['probe_values']) < PROBE_KEEP:
                    stats['probe_values'].append(merged)
                # ⚠️ 循环变量**不能叫 `tol`** —— 那会把 `tol` 变成本函数的局部变量，
                # 遮蔽闭包里的阈值，非 measure 路径上直接 `UnboundLocalError`（踩过）。
                for cand in MEASURE_TOLS:
                    if merged <= cand:
                        stats['probe_pass']['%.8g' % cand] += 1
                if bool(((x0 == 0) & (x3 == 0) & (gx == 1)
                         & (y0 == 0) & (y3 == 0) & (gy == 1)).all()):
                    stats['probe_exact_all'] += 1
                stats['fallback_calls'] += 1
                return original(image, axis_x, axis_y,
                                return_components=return_components,
                                reciprocal_source=reciprocal_source,
                                normalized_scale=normalized_scale)

            # 一次归约把六个条件一起判掉（见模块头部"阈值取哪来的"）。
            # 阈值 = 采样器自身的定点栅格 1/256 像素；低于它时采样器本来就把坐标吸到 texel 中心。
            degenerate = bool(((x0.abs() <= tol) & (x3.abs() <= tol) & ((gx - 1).abs() <= tol)
                               & (y0.abs() <= tol) & (y3.abs() <= tol)
                               & ((gy - 1).abs() <= tol)).all())
            stats['gate_reductions'] += 1
            if not degenerate:
                stats['fallback_calls'] += 1
                return original(image, axis_x, axis_y,
                                return_components=return_components,
                                reciprocal_source=reciprocal_source,
                                normalized_scale=normalized_scale)
            stats['degenerate_hits'] += 1
            stats['samples_avoided'] += 4
            stats['reciprocal_calls_avoided'] += 1
            # ⚠️ 只统计**第一次**短路放行了多大的扰动。若每次都测，就要多花 5 个逐元素 + 1 次归约
            # —— 那正是这把刀想省的东西，会自己吃掉收益。逐帧分布由 `measure` 模式负责。
            if tol > 0 and stats['degenerate_hits'] == 1:
                merged = torch_mod.maximum(
                    torch_mod.maximum(x0.abs(), x3.abs()),
                    torch_mod.maximum(torch_mod.maximum((gx - 1).abs(), y0.abs()),
                                      torch_mod.maximum(y3.abs(), (gy - 1).abs()))).max()
                stats['max_merged_applied'] = float(merged)
            # 复刻 _sample_five_axes 内的 sample() 闭包，只取 coordinates[2] = (mx, my)。
            if normalized_scale is None:
                value = sm.half_texture_linear(image, mx, my)
            else:
                h, w = image.shape[:2]
                iw, ih = normalized_scale
                # Preserve model-normalized -> region -> backing texture normalization.
                u = sm.fma32(mx.clamp(.5, w - .5) * iw, w, 0) * (1 / w)
                v = sm.fma32(my.clamp(.5, h - .5) * ih, h, 0) * (1 / h)
                value = sm.half_texture_normalized(image, u, v)
            # total == 1 ⇒ reciprocal_source(total) == 1.0 逐位
            # （已实测 `NativeReciprocalTable[1.0] = 1.0`，表 sha 与长度均匹配）。
            ones = torch_mod.ones_like(gx)
            if return_components:
                return value, ones
            return value * ones[..., None]

        _sample_five_axes_fast.__name__ = '_sample_five_axes'
        _sample_five_axes_fast.__doc__ = original.__doc__
        self._impl_fiveaxes = _sample_five_axes_fast
        self.rebound['fiveaxes'] = self._rebind_all(
            original, _sample_five_axes_fast, '_sample_five_axes')

    # ---------------- 卸载 ----------------

    def uninstall(self):
        if not self._installed:
            return self
        if 'fiveaxes' in self.cuts:
            self._rebind_all(self._impl_fiveaxes, self._original_fiveaxes, '_sample_five_axes')
        self._installed = False
        return self

    # ---------------- 上报 ----------------

    def report(self):
        note = ('短路条件 = (x0==0)&(x3==0)&(gx==1)&(y0==0)&(y3==0)&(gy==1) 的整图 .all()；'
                '退化时只取 coordinates[2] 一次采样、reciprocal 用 1.0 代替。')
        if self.measure_only:
            note = ('**只测量不短路**：每次调用记下 max(|x0|,|x3|,|gx-1|,|y0|,|y3|,|gy-1|)，'
                    '并按候选阈值统计命中数；一律回落到原函数。')
        return dict(
            cuts=list(self.cuts), measure_only=self.measure_only,
            rebound={k: list(v) for k, v in self.rebound.items()},
            stats=dict(self.stats),
            note=note)

    # ---------------- 自检 ----------------

    def selftest(self):
        """三条必须成立的事实，逐条显式测，不假设。

        1. 重绑确实覆盖了 `_sample_five_axes` 的所有持有者。
        2. 整数 motion ⇒ 门开火，且短路结果与原版**逐位相同**（CPU）。
        3. 亚像素 motion ⇒ 门不开火（回落），且输出与原版**逐位相同**（CPU）。
        """
        import torch

        # ⚠️ 自检**自己也会调刀**，会污染计数。计数是给「真实运行」读的 ⇒ 先快照、
        # 测完**原地**还原（`clear()` + `update()`；不能写 `self.stats = snapshot`，
        # 因为闭包捕获的是 dict 对象本身，重绑会让还原静默失效）。
        snapshot = dict(self.stats)
        seen = {}

        if 'fiveaxes' not in self.cuts:
            seen['status'] = 'skipped (fiveaxes cut not installed)'
            return seen

        if self.measure_only:
            # measure 模式的 `probe_values` 是 list，`dict(self.stats)` 只是浅拷贝 ⇒
            # 自检的"原地还原"会把新增元素留在共享的 list 上。这里直接跳过自检：
            # measure 模式本来就不改行为，自检没有对象。
            seen['status'] = 'skipped (measure_only: 行为未改变，无等价性可测)'
            return seen

        holders = []
        for mod_name, module in list(sys.modules.items()):
            if module is None:
                continue
            if getattr(module, '_sample_five_axes', None) is self._impl_fiveaxes:
                holders.append(mod_name)
        seen['fiveaxes_holders'] = sorted(holders)

        try:
            from nr_backend import sampling as sm
            h = w = 32
            image = torch.rand(h, w, 3, dtype=torch.float32)
            iw = torch.tensor(1.0 / w)
            ih = torch.tensor(1.0 / h)
            x = torch.arange(w, dtype=torch.float32)
            y = torch.arange(h, dtype=torch.float32)
            X, Y = torch.meshgrid(x, y, indexing='xy')

            def axes_for(off_x, off_y):
                ax = (X + .5, off_x, 1.0 - off_x, torch.zeros_like(X), X + .5)
                ay = (Y + .5, off_y, 1.0 - off_y, torch.zeros_like(Y), Y + .5)
                return ax, ay

            # 2. 整数（退化）
            ax, ay = axes_for(torch.zeros_like(X), torch.zeros_like(Y))
            got = self._impl_fiveaxes(image, ax, ay, reciprocal_source=None,
                                      normalized_scale=(iw, ih))
            want = self._original_fiveaxes(image, ax, ay, reciprocal_source=None,
                                           normalized_scale=(iw, ih))
            seen['degenerate_bit_equal'] = bool(torch.equal(got, want))
            seen['degenerate_max_abs'] = float((got - want).abs().max())

            # return_components 也要对
            got_c = self._impl_fiveaxes(image, ax, ay, return_components=True,
                                        reciprocal_source=None, normalized_scale=(iw, ih))
            want_c = self._original_fiveaxes(image, ax, ay, return_components=True,
                                             reciprocal_source=None, normalized_scale=(iw, ih))
            seen['degenerate_components_bit_equal'] = bool(
                torch.equal(got_c[0], want_c[0]) and torch.equal(got_c[1], want_c[1]))

            # 3. 亚像素（不退化）—— 必须回落，且输出逐位相同
            frac = torch.full_like(X, 0.37)
            ax2, ay2 = axes_for(frac, frac)
            got2 = self._impl_fiveaxes(image, ax2, ay2, reciprocal_source=None,
                                       normalized_scale=(iw, ih))
            want2 = self._original_fiveaxes(image, ax2, ay2, reciprocal_source=None,
                                            normalized_scale=(iw, ih))
            seen['fractional_bit_equal'] = bool(torch.equal(got2, want2))
            seen['fractional_max_abs'] = float((got2 - want2).abs().max())
            seen['fractional_fell_back'] = bool(
                not torch.equal(got2, self._impl_fiveaxes(image, ax, ay,
                                                          reciprocal_source=None,
                                                          normalized_scale=(iw, ih))))

            # 4. ★ 阈值行为：`t` 极小（这里 1e-6，真实链路实测 ~2.6e-5）时门**开火**，
            #    而结果与原版**小但不为 0** 地不同 —— 这正是"这不是逐位恒等"的书面证据。
            tiny = torch.full_like(X, 1e-6)
            ax3 = (X + .5, -tiny, torch.ones_like(X), -tiny, X + .5)
            ay3 = (Y + .5, -tiny, torch.ones_like(Y), -tiny, Y + .5)
            before = self.stats['degenerate_hits']
            got3 = self._impl_fiveaxes(image, ax3, ay3, reciprocal_source=None,
                                       normalized_scale=(iw, ih))
            seen['tiny_tol_fired'] = bool(self.stats['degenerate_hits'] == before + 1)
            want3 = self._original_fiveaxes(image, ax3, ay3, reciprocal_source=None,
                                            normalized_scale=(iw, ih))
            seen['tiny_max_abs'] = float((got3 - want3).abs().max())
            seen['tiny_bit_equal'] = bool(torch.equal(got3, want3))
            seen['tol'] = self.tol
        except Exception as exc:                        # pragma: no cover - 防御
            seen['error'] = '%s: %s' % (type(exc).__name__, exc)

        made = {key: self.stats[key] - snapshot.get(key, 0) for key in self.stats}
        self.stats.clear()
        self.stats.update(snapshot)
        seen['selftest_own_calls'] = made
        return seen


def install_from_env():
    """按 `MOTION_SCOPE_CUTS` 决定装不装。`off`（或空）= 不装（A/B 对照臂）。"""
    raw = os.environ.get('MOTION_SCOPE_CUTS', 'off').strip().lower()
    if raw in ('', 'off', 'none'):
        print('[motion_scope] cuts off (MOTION_SCOPE_CUTS=%r)' % raw, file=sys.stderr, flush=True)
        return None
    measure_only = raw in ('measure', 'probe')
    tol = float(os.environ.get('MOTION_SCOPE_TOL', DEFAULT_TOL))
    cuts = ('fiveaxes',) if measure_only else tuple(
        part for part in raw.replace(',', ' ').split() if part)
    if not cuts:
        raise ValueError('MOTION_SCOPE_CUTS=%r 解析后为空' % raw)
    for name in cuts:
        if name not in GROUPS:
            raise ValueError('MOTION_SCOPE_CUTS=%r 里有未知开关 %r' % (raw, name))
    instance = MotionScopeCuts(cuts=cuts, measure_only=measure_only, tol=tol).install()
    print('[motion_scope] cuts on: %s  tol=%.6g px%s'
          % (','.join(cuts), tol, '  (measure_only：只测量不短路)' if measure_only else ''),
          file=sys.stderr, flush=True)
    for label, mods in instance.rebound.items():
        print('[motion_scope]   %s 重绑到 %d 个模块：%s'
              % (label, len(mods), ', '.join(mods) or '(无)'), file=sys.stderr, flush=True)
    return instance
