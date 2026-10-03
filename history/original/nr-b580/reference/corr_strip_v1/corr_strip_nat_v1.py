"""corr_strip_nat_v1.py —— 修正层「原生等价物」原语集 + 运行期替换。

零产品源码改动：补丁只活在实验进程内存里。机制见 corr_strip_v1/PLAN.md §1、§4：

* 快速线的「修一下」是五个可点名的原语（本轮靶面四个，_tiled 已切除不动）。
* 这些原语都是 ``@triton.jit`` 设备函数，被热内核**内联**调用；Triton 在**编译期**
  从「定义该内核的那个函数的 ``__globals__``」解析被调符号。
* ⇒ 只要在第一次编译之前，把目标模块里那些**模块级全局名**换成原生版本，
  重新编译出来的内核就是原生版。

本模块提供：
* 四个原生原语（签名与被替换者完全一致，便于直接换名）；
* ``install(mode)``：在 ``Session.create`` 之后、第一帧之前调用，
  遍历 ``sys.modules`` 把快速线模块（路径在 ``nr-b580-int8/experimental/`` 或
  ``nr-b580-int8/backend/nr_backend/`` 下）的对应全局名换成原生版，
  并在定义模块上同步替换；返回一份**留痕**（哪些模块、哪些名、各换几处）。
* ``mode='nat'``：替换四个原语。
* ``mode='alg'``：在 nat 基础上再把 ``_normalize`` / ``normalize_c32`` 换成
  ``tl.sum`` 版本（XOR 归约树 → 一行求和；clamp 仍保留）。
* 回退：若 ``tl.float8e4nv`` 转换在 XPU 上不可用或抛错，退回「更少整数指令」的
  简化实现，并在留痕里显式标注 ``hardware_cvt=unavailable, fell back``，**绝不静默降级**。
* CPU-only 友好：本模块在 import 时不触碰 torch / triton（二者均惰性导入）；
  无 triton 时原语退化为纯 Python 桩，仅用于 ``--check`` 验证替换逻辑真的改了 ``__dict__``。
"""
from __future__ import annotations

import os
import sys
import types

# ----------------------------------------------------------------- 惰性取 triton
try:
    import triton.language as tl  # noqa: F401
    from triton import jit as _triton_jit
    from triton.runtime.jit import JITFunction as _JITFunction
    HAVE_TRITON = True
except Exception:  # CPU-only / 无 GPU 环境：退化为纯 Python 桩
    tl = None  # type: ignore[assignment]
    HAVE_TRITON = False
    _JITFunction = ()  # type: ignore[assignment]  —— isinstance(x, ()) 恒 False，安全

    def _triton_jit(fn):  # 无 triton 时装饰器退化为恒等，函数体只在其被调用时才触碰 tl
        return fn


# ----------------------------------------------------------------- 快速线根标记
_PRODUCT_ROOT_MARKS = (
    'nr-b580-int8/experimental/',
    'nr-b580-int8/backend/nr_backend/',
)

_PRIMITIVE_NAMES = ('_nan_left', 'rsqrt_half_clamped', '_round_fp8_half', '_half_fma_value')
_STRUCT_NAMES = ('_normalize', 'normalize_c32')

# 运行期被改写的状态（供 install 选择 _round_fp8_half 变体 + 留痕）
_ROUND_FP8_HARDWARE_CVT = 'not_probed'

# 硬件 cvt 探测开关：自检（--check / CPU-only）期间关闭，避免在任意机器上真的编译
# XPU 内核占用 GPU；真跑（有租约）时打开。
_ALLOW_HW_PROBE = True


def _is_fastline_module(mod):
    """判断模块是否属于快速线（experimental 或 backend/nr_backend）。"""
    f = getattr(mod, '__file__', None)
    if not f:
        return False
    n = f.replace('\\', '/')
    return any(mark in n for mark in _PRODUCT_ROOT_MARKS)


# ★ 替换后是否把受影响内核的**编译缓存**失效（默认开；置 0 可复现旧行为做对照）。
#   不失效 ⇒ 替换只改了模块字典，而内核的 ``cache_key`` 因 ``hash`` 缓存不变
#   ⇒ 命中旧产物、跑的还是含修正层的内核，**整个 nat 臂等于没测**。
INVALIDATE_KERNEL_KEYS = os.environ.get('NR_CORR_INVALIDATE_KEYS', '1') != '0'


def _invalidate_kernel_caches():
    """让快速线模块里所有 ``JITFunction`` 的**编译缓存**失效；返回清掉的内核数。

    ⚠️ **不做这一步，替换等于没做。** 两处缓存都会挡住新源码：

    1. ``JITFunction.cache_key`` 是「算过就缓存」——``triton/runtime/jit.py`` 的
       ``cache_key`` property 开头就是 ``if self.hash is not None: return self.hash``。
       而替换只改**模块字典**、不改 ``JITFunction`` 对象本身，所以任何在替换**之前**
       被访问过 ``cache_key`` 的内核（Session 预热、首帧探查等）会**永远**返回旧值。
       实测（纯 CPU）：同一内核在替换前先取一次 key 得 ``b773a381…``，替换后仍是
       ``b773a381…``；手动清 ``fn.hash`` 再取才是 ``42207208…`` —— 可见替换本应改 key。
       哪些内核凑巧在替换前被访问过纯属时序偶然，于是「只有个别内核真的换了」。
    2. ``JITFunction.device_caches``（``defaultdict``）按 device 缓存**已编内核对象**；
       若替换前该内核已经跑过，``run`` 会直接命中它，连 ``_do_compile`` 都不进。

    两处都清掉，内核才会按替换后的 ``__globals__`` 重算 key 并重新取产物。
    ``used_global_vals`` 不必手动清 —— ``cache_key`` 重算时会整份覆盖。
    """
    seen, cleared = set(), 0
    for mod in list(sys.modules.values()):
        if not _is_fastline_module(mod):
            continue
        for value in list(vars(mod).values()):
            if id(value) in seen or not isinstance(value, _JITFunction):
                continue
            seen.add(id(value))
            value.hash = None
            value.device_caches.clear()
            cleared += 1
    return cleared


# _round_fp8_half 的两种变体必须存在于**模块级全局**，供 _select_round_fp8_half
# 在运行期通过 globals() 重新绑定主路径名。
if HAVE_TRITON:
    @_triton_jit
    def _round_fp8_half_cvt(x):
        # 硬件 cvt：一条转换换掉约 20 条整数指令。
        return x.to(tl.float8e4nv).to(tl.float16)

    @_triton_jit
    def _round_fp8_half_fallback(x):
        # 回退：更少整数指令的简化 E4M3 编码（就近舍入，跳过 subnormal 与
        # 规范 NaN 修正）。仅当 XPU 不支持 float8e4nv 转换时启用，且留痕标注。
        bits = x.to(tl.uint16, bitcast=True).to(tl.int32)
        sign = bits & 0x8000
        raw = bits & 0x7fff
        mag = tl.minimum(raw, 0x5f00)  # 448，最大有限 E4M3FN 幅度
        exponent = mag >> 10
        sig = (mag & 1023) + tl.where(exponent != 0, 1024, 0)
        shift = tl.maximum(16 - tl.maximum(exponent, 1), 1)
        q = (sig + (1 << (shift - 1))) >> shift  # 就近舍入（含中点上取）
        normal = ((mag + 63 + ((mag >> 7) & 1)) & 0x7f80) | sign
        return normal.to(tl.uint16).to(tl.float16, bitcast=True)
else:  # CPU-only 桩
    def _round_fp8_half_cvt(x):
        return x

    def _round_fp8_half_fallback(x):
        return x


# ----------------------------------------------------------------- 原生原语
def _make_natives():
    """构造四个原生原语。签名与被替换者完全一致。

    返回 (dict, info)：dict 把名字映射到本模块级全局里的原生函数；info 记录能力。
    """
    global _round_fp8_half  # 主路径默认指向 cvt 变体；install 时可能切到 fallback
    info = dict(triton_present=HAVE_TRITON, torch_present=False, xpu_present=False,
                hardware_cvt=_ROUND_FP8_HARDWARE_CVT)

    if HAVE_TRITON:
        @_triton_jit
        def _nan_left(result, left, right):
            # 原语模拟 PyTorch XPU 二元的 NaN payload 传播；
            # 对有限数据零影响 ⇒ 直接返回 result（左操作数原样）。
            return result

        @_triton_jit
        def rsqrt_half_clamped(x):
            # 保留 clamp（算法、防除零）；去掉手写 NaN 图案修正（|512）。
            value = tl.maximum(x.to(tl.float32), 6.198883056640625e-05)
            return tl.rsqrt(value).to(tl.float16)

        @_triton_jit
        def _half_fma_value(av, bv, cv):
            # 原生 FP16 FMA（09-09 已落地的 native_half_attention_fma_v1.half_fma_attention）。
            return tl.fma(av.to(tl.float16), bv.to(tl.float16), cv.to(tl.float16)).to(tl.float16)

        # 默认主路径；install 时按探测结果决定是否切到 fallback。
        _round_fp8_half = _round_fp8_half_cvt
    else:
        def _nan_left(result, left, right):  # noqa: F811
            return result

        def rsqrt_half_clamped(x):  # noqa: F811
            return x

        def _half_fma_value(av, bv, cv):  # noqa: F811
            return av

        def _round_fp8_half(x):  # noqa: F811
            return x

    natives = dict(
        _nan_left=_nan_left,
        rsqrt_half_clamped=rsqrt_half_clamped,
        _round_fp8_half=_round_fp8_half,
        _half_fma_value=_half_fma_value,
    )
    return natives, info


def _probe_hardware_cvt():
    """探测 XPU 是否真能编译/运行 ``tl.float8e4nv`` 转换。

    返回 ``True``（可用）/ ``False``（不可用或异常）/ ``'skip'``（自检期禁止探测）。
    """
    if not _ALLOW_HW_PROBE:
        return 'skip'
    if not HAVE_TRITON:
        return False
    try:
        import torch
        import triton
    except Exception:
        return False
    if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
        return False
    try:
        @triton.jit
        def _probe(x):
            return x.to(tl.float8e4nv).to(tl.float16)

        inp = torch.zeros(16, dtype=torch.float16, device='xpu')
        _probe[inp.numel():](inp, num_warps=1)
        torch.xpu.synchronize()
        return True
    except Exception:
        return False


def _select_round_fp8_half(info):
    """按探测结果选择 _round_fp8_half 变体；写入留痕字段 hardware_cvt。"""
    global _round_fp8_half, _ROUND_FP8_HARDWARE_CVT  # noqa: PLW0603
    if not HAVE_TRITON:
        _ROUND_FP8_HARDWARE_CVT = 'not_probed (triton absent)'
        info['hardware_cvt'] = _ROUND_FP8_HARDWARE_CVT
        return
    if _ROUND_FP8_HARDWARE_CVT != 'not_probed':
        info['hardware_cvt'] = _ROUND_FP8_HARDWARE_CVT
        return
    verdict = _probe_hardware_cvt()
    if verdict == 'skip':
        _ROUND_FP8_HARDWARE_CVT = 'not_probed (self-check)'
    elif verdict:
        _round_fp8_half = globals()['_round_fp8_half_cvt']
        _ROUND_FP8_HARDWARE_CVT = 'available'
    else:
        _round_fp8_half = globals()['_round_fp8_half_fallback']
        _ROUND_FP8_HARDWARE_CVT = 'unavailable, fell back'
    info['hardware_cvt'] = _ROUND_FP8_HARDWARE_CVT


# ----------------------------------------------------------------- alg：tl.sum 版 _normalize
def _make_struct_natives():
    """构造 _normalize / normalize_c32 的 tl.sum 版本（替换 XOR 归约树，clamp 保留）。"""
    if HAVE_TRITON:
        @_triton_jit
        def _normalize(X, Y, ROWS: tl.constexpr, BR: tl.constexpr):
            rows = tl.program_id(0) * BR + tl.arange(0, BR)
            lanes = tl.arange(0, 8)
            offsets = rows[:, None] * 32 + lanes[None, :]
            valid = rows[:, None] < ROWS
            x0 = tl.load(X + offsets, valid, other=0)
            x8 = tl.load(X + offsets + 8, valid, other=0)
            x16 = tl.load(X + offsets + 16, valid, other=0)
            x24 = tl.load(X + offsets + 24, valid, other=0)
            # 四向量平方和（原生 mul/add；本刀不动算法结构，只换归约树）。
            s = (x0.to(tl.float32) * x0.to(tl.float32)
                 + x8.to(tl.float32) * x8.to(tl.float32)
                 + x16.to(tl.float32) * x16.to(tl.float32)
                 + x24.to(tl.float32) * x24.to(tl.float32)).to(tl.float16)
            denominator = tl.sum(s, axis=1)                 # 一行求和，替换 XOR 归约树
            scale = rsqrt_half_clamped(denominator[:, None])  # clamp 仍保留
            sc = scale.to(tl.float32)
            tl.store(Y + offsets, (x0.to(tl.float32) * sc).to(tl.float16), valid)
            tl.store(Y + offsets + 8, (x8.to(tl.float32) * sc).to(tl.float16), valid)
            tl.store(Y + offsets + 16, (x16.to(tl.float32) * sc).to(tl.float16), valid)
            tl.store(Y + offsets + 24, (x24.to(tl.float32) * sc).to(tl.float16), valid)

        def normalize_c32(x, *, rows=16, warps=4):
            import torch
            import triton
            if x.device.type != 'xpu' or x.shape[-1] != 32:
                raise ValueError('C32 XPU vectors required')
            source = x.half().contiguous()
            output = torch.empty_like(source)
            count = source.numel() // 32
            if count:
                _normalize[(triton.cdiv(count, rows),)](source, output, count, rows,
                                                         num_warps=warps, enable_fp_fusion=False)
            return output
    else:
        def _normalize(X, Y, ROWS, BR):  # noqa: F811
            return None

        def normalize_c32(x, *, rows=16, warps=4):  # noqa: F811
            return x

    globals()['_normalize'] = _normalize
    globals()['normalize_c32'] = normalize_c32
    return _normalize, normalize_c32


# ----------------------------------------------------------------- 安装入口
def install(mode='nat'):
    """运行期替换快速线模块里的修正原语。

    Parameters
    ----------
    mode : 'off' | 'nat' | 'alg'
        off  —— 不替换任何原语（基准臂）；
        nat  —— 替换四个原语；
        alg  —— nat + 把 ``_normalize`` / ``normalize_c32`` 换成 tl.sum 版本。

    Returns
    -------
    dict  留痕：mode、各模块替换明细、各名计数、hardware_cvt 标注、capabilities。
    """
    if mode not in ('off', 'nat', 'alg'):
        raise ValueError(f'NR_CORR_STRIP must be one of off|nat|alg, got {mode!r}')

    trace = dict(mode=mode, modules=[], struct_modules=[],
                 counts={n: 0 for n in _PRIMITIVE_NAMES},
                 struct_counts={n: 0 for n in _STRUCT_NAMES},
                 hardware_cvt=None, capabilities=None, note=None)
    if mode == 'off':
        trace['note'] = 'off: no primitive replaced (baseline arm)'
        return trace

    natives, info = _make_natives()
    _select_round_fp8_half(info)
    # ⚠️ 必须把探测结果**回写进 natives 字典**。
    # `_make_natives()` 在返回前就把 `_round_fp8_half` 绑成了 cvt 变体（L169/L186），
    # 而 `_select_round_fp8_half()` 只重绑**模块全局**、不碰这个字典 ⇒ 若此处不回写，
    # 下面 `d[name] = natives[name]` 装进去的**永远是 cvt**，而留痕字段却写着
    # 'unavailable, fell back' —— r9 就是这样：报告说回落了、实际装的是 cvt。
    # 字段与实现必须一致，否则「换了什么」无法从落盘证据里判定。
    natives['_round_fp8_half'] = _round_fp8_half
    info['round_fp8_half_installed'] = getattr(_round_fp8_half, '__name__',
                                               repr(_round_fp8_half))
    trace['round_fp8_half_installed'] = info['round_fp8_half_installed']
    trace['hardware_cvt'] = info['hardware_cvt']
    trace['capabilities'] = {k: info[k] for k in ('triton_present', 'torch_present', 'xpu_present')}

    # 1) 替换四个原语（含定义模块，因同模块内调用走同一份 globals）。
    per_module = []
    for mod in list(sys.modules.values()):
        if not _is_fastline_module(mod):
            continue
        d = mod.__dict__
        touched = {}
        for name in _PRIMITIVE_NAMES:
            if name in d:
                d[name] = natives[name]
                trace['counts'][name] += 1
                touched[name] = touched.get(name, 0) + 1
        if touched:
            per_module.append(dict(module=getattr(mod, '__name__', str(mod)),
                                   file=getattr(mod, '__file__', None),
                                   replaced=touched))
    trace['modules'] = per_module

    # 2) alg：再替换结构性归约（_normalize / normalize_c32）。
    if mode == 'alg':
        _make_struct_natives()
        struct_natives = dict(_normalize=globals()['_normalize'],
                             normalize_c32=globals()['normalize_c32'])
        per_struct = []
        for mod in list(sys.modules.values()):
            if not _is_fastline_module(mod):
                continue
            d = mod.__dict__
            touched = {}
            for name in _STRUCT_NAMES:
                if name in d:
                    d[name] = struct_natives[name]
                    trace['struct_counts'][name] += 1
                    touched[name] = touched.get(name, 0) + 1
            if touched:
                per_struct.append(dict(module=getattr(mod, '__name__', str(mod)),
                                      file=getattr(mod, '__file__', None),
                                      replaced=touched))
        trace['struct_modules'] = per_struct

    # ★ 替换只改了模块字典；内联了这些原语的**热内核**其 key 与已编对象仍被缓存钉着。
    #   必须显式失效，否则内核不会重编、跑的仍是修正层版本（详见 _invalidate_kernel_caches）。
    trace['invalidate_enabled'] = INVALIDATE_KERNEL_KEYS
    trace['invalidated_kernels'] = (
        _invalidate_kernel_caches() if INVALIDATE_KERNEL_KEYS else None)

    return trace


# ----------------------------------------------------------------- CPU-only 自检辅助
def _selfcheck_install():
    """纯 Python 层验证 install 真的改了假模块的 __dict__（不 import torch / triton）。"""
    global _ALLOW_HW_PROBE  # noqa: PLW0603
    _ALLOW_HW_PROBE = False  # 自检期禁止任何 XPU 内核编译
    fake = types.ModuleType('fake_fastline')
    fake.__file__ = 'E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580-int8/experimental/fake_v1.py'
    fake.__dict__['_nan_left'] = 'ORIGINAL_NAN'
    fake.__dict__['_round_fp8_half'] = 'ORIGINAL_ROUND'
    fake.__dict__['_half_fma_value'] = 'ORIGINAL_FMA'
    fake.__dict__['rsqrt_half_clamped'] = 'ORIGINAL_RSQRT'
    sys.modules['fake_fastline'] = fake

    fake2 = types.ModuleType('fake_struct')
    fake2.__file__ = ('E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580-int8/backend/'
                      'nr_backend/triton_attention_normalize.py')
    fake2.__dict__['_normalize'] = 'ORIGINAL_NORMALIZE'
    fake2.__dict__['normalize_c32'] = 'ORIGINAL_NORMALIZE_C32'
    sys.modules['fake_struct'] = fake2

    # off 模式不应改任何东西
    t_off = install('off')
    assert fake.__dict__['_nan_left'] == 'ORIGINAL_NAN', 'off must not replace'
    # nat 应改四个原语
    t_nat = install('nat')
    assert fake.__dict__['_nan_left'] != 'ORIGINAL_NAN', 'nat did not replace _nan_left'
    assert fake.__dict__['_round_fp8_half'] != 'ORIGINAL_ROUND', 'nat did not replace _round_fp8_half'
    assert fake.__dict__['_half_fma_value'] != 'ORIGINAL_FMA', 'nat did not replace _half_fma_value'
    assert fake.__dict__['rsqrt_half_clamped'] != 'ORIGINAL_RSQRT', 'nat did not replace rsqrt'
    # alg 应额外改 _normalize / normalize_c32
    t_alg = install('alg')
    assert fake2.__dict__['_normalize'] != 'ORIGINAL_NORMALIZE', 'alg did not replace _normalize'
    assert fake2.__dict__['normalize_c32'] != 'ORIGINAL_NORMALIZE_C32', 'alg did not replace normalize_c32'

    for tag in ('off', 'nat', 'alg'):
        try:
            install(tag)
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f'install({tag!r}) raised: {exc}') from exc
    try:
        install('bogus')
        raise AssertionError('install accepted invalid mode')
    except ValueError:
        pass

    # 清理假模块，避免污染真实 sys.modules
    sys.modules.pop('fake_fastline', None)
    sys.modules.pop('fake_struct', None)
    return dict(off=t_off, nat=t_nat, alg=t_alg)


if __name__ == '__main__':
    out = _selfcheck_install()
    print('corr_strip_nat_v1 self-check: OK')
    print('  nat   :', {k: out['nat'][k] for k in ('mode', 'hardware_cvt', 'counts')})
    print('  alg   :', {k: out['alg'][k] for k in ('mode', 'struct_counts')})
