"""原尺寸 NR 的**几何档位**：把 `fullsize_session_v1` 硬锁的 FULLSIZE 变成可指定。

背景（为什么需要这个文件）
--------------------------
`fullchain_timing_v1/fullsize_session_v1.py` 是全链实测（56.07 ms/输入帧）用的原尺寸
NR 会话，它把输入几何写成一个模块常量::

    FULLSIZE = (480, 864)          # (H, W)

`fullchain_timing_v1/RESULT.md` §7.7 第 7 条把"`NR_FULLSIZE` 在 960×540 / 864×486
下能否成立（当前会话硬锁 `FULLSIZE=(480,864)`）"列为**未做项**。本文件就是关这一条。

为什么用运行期覆盖而不是改那个文件
----------------------------------
两条约束同时成立：

1. 铁律「**新改动另起一轮**」—— 上一轮的装置要保字节不变，否则 56.07 那个数失去可比性；
2. `FullsizeSession` 的 `process()` / `warmup()` / `report()` 三处都在**调用时**读模块
   全局 `FULLSIZE`（`size != FULLSIZE`、`h, w = FULLSIZE`、`list(FULLSIZE)`），
   所以覆盖该全局在语义上完全等价于把常量改掉 —— 零改动、可逆、原文件 sha256 不动。

⚠️ **覆盖必须活到 `close()`，不能只在构造期**（这是探针第一版踩的坑）：
在构造之后立刻还原，会让 `warmup()` 用旧几何编内核、`process()` 又用旧几何校验，
报 `Fullsize NR expects HWC RGB (864, 480), got (480, 640)`。本模块因此把覆盖的
生命周期绑到会话对象上，并在 `close()` 时释放。

几何的硬约束（来自后端，不是本文件的约定）
------------------------------------------
* `nr_backend/c32_block.py:30`、`attention.py:148`、`multihead_block.py:61`、
  `window_layout_v1.py:76` —— **H 与 W 必须能被 8 整除**（8×8 注意力窗口）；
* `nr_backend/vit_block.py:38` —— ViT 注意力 `tokens = key.shape[-2]; padding = (-tokens) % 64`
  会把 token 数**自动补齐到 64 的倍数**，所以 ViT 侧不要求 64 整除；
* `fullsize_rows_v1/rows_scopes_v1.py:32,83` —— 两个 INT8 FFN 的行数
  `rows = features.shape[0] * features.shape[1]` / `rows = x.shape[0]`
  是**从张量形状派生**的，不写死，因此任意合法几何都能跑，只是**内核要重新编译**。
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

# 让 `import fullsize_session_v1` 成立（它在 fullchain_timing_v1/，与本文件不同目录）。
_FULLCHAIN = Path(__file__).resolve().parent
if str(_FULLCHAIN) not in sys.path:
    sys.path.insert(0, str(_FULLCHAIN))

import fullsize_session_v1 as _base

# 已审核的档位。新增几何**必须**先在这里登记，避免拼写错误静默跑到别的形状。
GEOMETRIES = {
    '480x864': (480, 864),   # 全链实测用的原始档（56.07 ms/输入帧）
    '480x640': (480, 640),   # 4:3 —— 生化危机4 的客户区（640x480）
    '540x960': (540, 960),   # 16:9 480p 级的候选
    '480x854': (480, 854),   # 16:9 近似；854 不是 8 的倍数，登记只为**显式拒绝**
    '486x864': (486, 864),   # 16:9 精确源（864x486）
}

# 本档位的默认几何 = 上游原尺寸档，保证"不指定就等于改动前"。
DEFAULT = '480x864'

# ---------------------------------------------------------------------------
# ★ 真正的门：`nr_backend/executor.py:31` / `nr_backend/temporal.py:91` 的
#   PADDED_SIZES = 输入几何 -> **内部补齐画布**。
#
# 那张表的原注（executor.py:29-30）是决定性的：
#   "Native captures show asymmetric padding at 512.  Do not extrapolate a
#    padding formula or silently resize arbitrary inputs to these contracts."
# 即：**每个几何的补齐画布是原生采集"观测"出来的，不是公式推的**。
# 所以"支持一个分辨率"= 加一条契约 + 预编译该几何的内核集（这才是
# 「nr 支持分辨率需要预先编译」的确切含义）。
#
# 这张换算也同时解释了两个已记录的行数（本条为交叉校验，非新结论）：
#   c512 行数 = padH * padW / 1024
#     (480,864)->(512,896): 512*896/1024 = 448   ← rows RESULT §4.3 记的 448
#     (256,256)->(320,320): 320*320/1024 = 100   ← rows RESULT §4.3 记的 100
# ---------------------------------------------------------------------------
NATIVE_PADDING = {
    (480, 864): (512, 896),      # 原生观测（全链实测档）
    (256, 256): (320, 320),      # 原生观测（NR256 画布）
    (512, 512): (512, 576),
    (1080, 1920): (1152, 1920),
    (1439, 2559): (1472, 2560),
}

# **非原生**补充契约：本条**不是**采集观测值。
#
# 先把原生契约里的关系写出来（这是实测得到的，不是猜的）——
#   c512 网格 == 补齐画布 / 32       （(512,896) -> (16,28)，(512/32=16, 896/32=28)）
# 再按"每轴 +1 个 32 的格子"给 (480,640)（15x20 格）推：16x21 格 = (512,672)。
#
# 实测把这条路堵掉了：`(512,672)` 下网络**自己**又在 /16 一级补了一层
#   （672/16 = 42 不是 8 的倍数 ⇒ 补到 48 ⇒ /2 后 c512 网格成 24），
# 于是 c512 网格变成 (16,24) 而 pad/32 = (16,21) —— **不同构**。
# 而 768/16 = 48 本来就是 8 的倍数 ⇒ 网格 (16,24) == pad/32 ⇒ **同构**。
#
# 所以最终采用 **(512, 768)**：它与原生契约遵守同一条 "c512 网格 = pad/32"，实测
# `c512_grid_equals_pad_over_32 == True`，且网格(16,24)/行数384/ViT token 96
# 与 (512,672) 那次逐项相同（印证"网络把 672 内部当 768 用"）。
#
# ⚠️ 本条**没有原生对拍基准**：DLSS 5 原生只在上面 NATIVE_PADDING 的 5 个几何被采集过，
# 而 `executor.py:29-30` 明确禁止外推补白公式。因此本档产物**不能**当作
# "DLSS 5 在该分辨率下的忠实重现"，只能说"同架构、同权重、补齐画布由我选择的运行档"。
# 任何画面对拍结论都必须带这个限定（本仓铁律：诚实边界）。
# 另：该画布**有语义**——(512,672) 与 (512,768) 的网格/token 相同但输出不同
# （mean 0.516665 vs 0.504689）⇒ 它不是装饰参数。
DERIVED_PADDING = {
    (480, 640): (512, 768),
}

PADDING = {**NATIVE_PADDING, **DERIVED_PADDING}


def parse_geometry(text):
    """接受 '480x640' / '480X640' / '480,640' / 'HxW' 并返回 (H, W)。

    不在这里做整除校验 —— 校验交给 `require_divisible()`，好在报错里说清是哪一维。
    """
    if text is None:
        return GEOMETRIES[DEFAULT]
    cleaned = str(text).strip().lower().replace(' ', '').replace(',', 'x')
    if cleaned in GEOMETRIES:
        return GEOMETRIES[cleaned]
    parts = cleaned.split('x')
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f'geometry must look like HxW, got {text!r}')
    return int(parts[0]), int(parts[1])


def require_divisible(geometry):
    """后端硬门：H、W 必须能被 8 整除（8×8 窗口）。"""
    height, width = geometry
    bad = [name for name, value in (('H', height), ('W', width)) if value % 8]
    if bad:
        raise ValueError(
            'fullsize NR needs dimensions divisible by 8 (8x8 attention windows); '
            f'{" and ".join(bad)} of {height}x{width} is not')
    return geometry


def geometry_from_env(variable='NR_GEOM'):
    return require_divisible(parse_geometry(os.environ.get(variable) or DEFAULT))


class GeometryConflict(RuntimeError):
    """同一进程里同时要求两个**不同**几何的原尺寸会话。"""


# 当前被原尺寸会话占用的几何；`FullsizeSession` 读的是模块全局，同一时刻只能有一个值。
_OWNER = None


def active_geometry():
    return getattr(_OWNER, 'fullsize_geometry', None)


class _GeometrySession(_base.FullsizeSession):
    """几何被覆盖后构造出来的原尺寸会话；`close()` 时释放覆盖。"""

    def close(self):
        global _OWNER
        try:
            super().close()
        finally:
            if _OWNER is self:
                _OWNER = None
                _base.FULLSIZE = GEOMETRIES[DEFAULT]

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self, *_):
        self.close()


def install_runtime_contract(session, size, padding=None):
    """把"输入几何 -> 补齐画布"这条契约打到**该模型实例自己的类**上。

    为什么打在实例的类而不是改源：`PADDED_SIZES` 是类属性，`forward()` 里按
    `self.PADDED_SIZES[tuple(rgb.shape[:2])]` 在**调用时**查表 ⇒ 只要在 `warmup()`
    之前把它补上，行为与改源等价。这样：
      * 不动 `nr_backend/*`（那是精确线预编译包的受保护源，改一行整包失效）；
      * 只影响本进程，进程退出即消失。

    ⚠️ 不能去打 `MotionNR.PADDED_SIZES` 之类的**基类**：`ControlledMotionNR`
    （`controlled_temporal.py:48`）是 `MotionNR.PADDED_SIZES.copy()`，基类改了它看不到。
    打在实例的类上就不会踩这个坑。
    """
    model = session._stack.model
    owner = type(model)
    pad = tuple(padding) if padding else PADDING.get(tuple(size))
    if pad is None:
        raise KeyError(f'no padding contract registered for {tuple(size)}')
    existing = owner.PADDED_SIZES.get(tuple(size))
    contract = dict(
        geometry=list(size), padding=list(pad), owner=f'{owner.__module__}.{owner.__name__}',
        origin='native-observed' if tuple(size) in NATIVE_PADDING else 'derived-not-observed',
        already_present=existing is not None,
        previous=list(existing) if existing else None,
        c512_rows_expected=pad[0] * pad[1] // 1024)
    if existing is not None and tuple(existing) != pad:
        raise GeometryConflict(
            f'{contract["owner"]} already maps {tuple(size)} to {existing}, '
            f'refusing to overwrite it with {pad}')
    owner.PADDED_SIZES[tuple(size)] = pad
    session.geometry_contract = contract
    return contract


def open_session(exact_root, profile_path, profile_sha256, *,
                 geometry=None, padding=None, verify_paths=True):
    """在指定几何下打开一个原尺寸会话。

    `geometry` 可以是 (H, W)、'480x640'，或 None（取 `NR_GEOM` 环境变量 / 默认档）。
    `padding` 若不指定，取本模块的 PADDING 表。返回对象与
    `fullsize_session_v1.FullsizeSession` **同型同接口**。
    """
    global _OWNER
    if geometry is None:
        size = geometry_from_env()
    elif isinstance(geometry, str):
        size = require_divisible(parse_geometry(geometry))
    else:
        size = require_divisible(tuple(geometry))

    held = active_geometry()
    if held is not None and tuple(held) != size:
        raise GeometryConflict(
            f'an open fullsize session already holds {held}; FULLSIZE is a module '
            f'global of fullsize_session_v1, so close it before opening {size}')

    previous = _base.FULLSIZE
    _base.FULLSIZE = size
    try:
        session = _GeometrySession(exact_root, profile_path, profile_sha256,
                                   verify_paths=verify_paths)
        # 模型已建好 ⇒ 可以给它的类打补齐契约（必须在 warmup/process 之前）。
        contract = install_runtime_contract(session, size, padding)
    except BaseException:
        _base.FULLSIZE = previous
        raise
    # 形状是构造期读的；固化到实例上，给日志/报告用。
    session.fullsize_geometry = size
    session.fullsize_padding = contract
    _OWNER = session
    return session


# 兼容旧写法：Re4Session(...) == open_session(...)
Re4Session = open_session


def describe():
    return dict(
        module=__file__, default=DEFAULT, registered=sorted(GEOMETRIES),
        rule='H % 8 == 0 and W % 8 == 0; ViT tokens are padded to 64 internally; '
             'the model also needs a PADDED_SIZES entry (observed contract)',
        active=active_geometry(),
        native_padding={f'{h}x{w}': list(p) for (h, w), p in NATIVE_PADDING.items()},
        derived_padding={f'{h}x{w}': list(p) for (h, w), p in DERIVED_PADDING.items()})
