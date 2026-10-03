"""corr_strip_cvt_probe_v1.py —— E4M3 舍入实现的原位微基准（判定 P1 是否还有效）

## 为什么需要它

r9 的 nat 臂在报告里写着 ``hardware_cvt: 'unavailable, fell back'`` ⇒ 探针失败、
``_round_fp8_half`` 被换成了 ``_round_fp8_half_fallback``（**另一套整数模拟**）。
于是 P1 的 +1.7835 ms **不是**在测「硬件 cvt 替掉整数模拟」这条杠杆，
它测的是「拿一套整数模拟换另一套更差的整数模拟」。

本脚本在真设备上把三件事分开：

1. **原生 fp8 转换到底能不能编译/运行** —— 把真实异常原样打出来。
   （harness 里的 ``_probe_hardware_cvt`` 用 ``except Exception`` 吞掉了它，这是根因所在。）
2. **同形状内核里各实现各多少 µs** —— orig（复刻层）/ fallback / cvt × 2 变体。
3. **noop 基线** —— 只 load/store、不做舍入。这是判据的关键：
   若 ``noop ≈ orig``，说明这些内核**访存受限**、舍入指令被访存延迟遮住 ⇒
   「拿掉模拟层」在 ALU 上没有钱可捡，PLAN §3 第三条的前提就不成立。

## 口径声明（务必与结论一起引用）

* 这是**孤立微基准**，只用于**定价与定性**，**不得外推为帧墙收益**。
  帧墙收益只能由 ABBA 四臂给出（这是 `PROBE-CUTSWEEP` 一轮的直接教训）。
* 内核形状逐字复制 ``nr_backend/triton_fp8.py`` 的 ``_kernel``：``B=512``、
  ``enable_fp_fusion=False``。
* 被测实现**不自带副本**：``_round_fp8_half`` 与 ``_round_fp8_half_fallback``
  的源码是从各自文件里**逐字抽取**的，抽取失败即报错退出（不静默退回近似实现）。

用法::

    python corr_strip_cvt_probe_v1.py --check          # CPU-only：只验证源码抽取
    python corr_strip_cvt_probe_v1.py --out <dir>      # 真跑（需 XPU）
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import textwrap
import time
import traceback
from pathlib import Path

W = Path('E:/ComfyUI-aki-v3-IntelArc_20260722')
IMPL_FILE = W / 'nr-b580-int8/backend/nr_backend/triton_fp8.py'      # 复刻层原文
NAT_FILE = W / 'nr-b580/reference/corr_strip_v1/corr_strip_nat_v1.py'  # 回退版原文

# ======================= 环境守卫（务必在任何 `import triton` 之前执行） =======================
ROWS_DIR = W / 'nr-b580/reference/fullsize_rows_v1'

# 绝不允许把 Triton 缓存放在这些树**内部**（本卷 rmdir 上溯会连带删除整棵树）。
PROTECTED_ROOTS = (
    'e:/comfyui-aki-v3-intelarc_20260722',
    'd:/corr-strip-v1',
    'd:/fullsize-rows-v1',
    'g:/nr-4way-v1',
)


def _install_fs_guard() -> dict:
    """把 Triton 的 ``os.removedirs(tmp.pid_*)`` 换成「只删叶子」的版本。

    ⚠️ 本卷 ``rmdir(非空目录)`` 不抛错而是**递归删除**，于是 Triton
    ``FileCacheManager.put()`` 末尾的 ``os.removedirs(temp_dir)`` 会从
    ``<TRITON_CACHE_DIR>/<key>/tmp.pid_*`` 一路向上剪掉空父目录，**连整棵交付物树一起吃掉**：
    2026-09-22 吞过 ``E:`` 的 ``.git``；2026-09-23 00:56 又吞过 ``D:/corr-strip-v1`` 全树
    13,695 条 —— 那一轮正是因为**本脚本当时忘了装这个守卫**（旧驱动装了，新驱动漏了）。
    因此它必须在本进程任何 ``import triton`` 之前装好，并自检替换是否真的生效。
    """
    if str(ROWS_DIR) not in sys.path:
        sys.path.insert(0, str(ROWS_DIR))
    import rows_fscache_guard_v1 as g
    st = g.install()
    got = (st.get('removedirs') or {}).get('name')
    if not st.get('installed') or got != 'removedirs_leaf_only':
        raise SystemExit('[guard] rows_fscache_guard_v1 未生效：%r' % st)
    return st


def _assert_cache_outside_trees() -> str:
    """硬断言：TRITON_CACHE_DIR 不许落在任何交付物树内部。

    这是 2026-09-22 已写进纪律的规则（`G:/nr-4way-v1` 事故的根因之一），
    2026-09-23 的第二次事故说明「只写在文档里」不够 —— 这里做成进程内硬失败。
    """
    cache = Path(os.environ.get('TRITON_CACHE_DIR',
                                str(Path.home() / '.triton' / 'cache')))
    low = str(cache.resolve()).replace('\\', '/').lower()
    for root in PROTECTED_ROOTS:
        if low == root or low.startswith(root + '/'):
            raise SystemExit(
                '[guard] TRITON_CACHE_DIR 落在交付物树内部，拒绝运行：\n'
                '        cache = %s\n'
                '        树根  = %s\n'
                '        本卷 rmdir 会上溯删除 ⇒ 必须把缓存放到树外。' % (cache, root))
    return str(cache)


FS_GUARD = _install_fs_guard()
CACHE_DIR_RESOLVED = _assert_cache_outside_trees()
# ==============================================================================================


# --------------------------------------------------------------------- 源码抽取
def extract_func(path: Path, name: str) -> str:
    """从源文件里逐字抠出顶层（或缩进一层）函数 ``name``，含紧邻的装饰器，并 dedent。

    只做文本操作、不 import，故可在无 torch/triton 的机器上验证。
    """
    lines = path.read_text(encoding='utf-8').splitlines()
    head = None
    for i, raw in enumerate(lines):
        if re.match(rf'^(\s*)def {re.escape(name)}\s*\(', raw):
            head = i
            indent = len(lines[i]) - len(lines[i].lstrip())
            break
    if head is None:
        raise SystemExit(f'[extract] 找不到 def {name}() 于 {path}')

    # 向上吃装饰器（同缩进）与紧邻注释
    start = head
    while start > 0:
        prev = lines[start - 1]
        stripped = prev.strip()
        prev_indent = len(prev) - len(prev.lstrip())
        if stripped.startswith('@') and prev_indent == indent:
            start -= 1
            continue
        if stripped.startswith('#') and prev_indent == indent:
            start -= 1
            continue
        break

    # 向下吃到下一个同/更浅缩进的非空语句
    end = head + 1
    while end < len(lines):
        cur = lines[end]
        if cur.strip() and (len(cur) - len(cur.lstrip())) <= indent:
            break
        end += 1

    body = textwrap.dedent('\n'.join(lines[start:end])).rstrip() + '\n'
    if f'def {name}' not in body:
        raise SystemExit(f'[extract] {name}: 抽取结果异常')
    return body


# 判别标记：各实现必须能通过这几个字符串被认出来，否则视为抽取错位
MARKERS = {
    '_round_fp8_half': ('quotient', 'midpoint', '0x5f00', '0x7f80'),
    '_round_fp8_half_fallback': ('0x5f00', '0x7f80'),
}


def check_extraction() -> dict:
    """CPU-only 自检：两个原语都能抠出来，且带判别标记。"""
    out, problems = {}, []
    for path, name in ((IMPL_FILE, '_round_fp8_half'),
                       (NAT_FILE, '_round_fp8_half_fallback')):
        try:
            src = extract_func(path, name)
        except SystemExit as exc:
            problems.append(str(exc))
            continue
        missing = [m for m in MARKERS[name] if m not in src]
        out[name] = dict(file=str(path), lines=src.count('\n'),
                         chars=len(src), missing_markers=missing,
                         first_line=src.splitlines()[0], last_line=src.splitlines()[-1])
        if missing:
            problems.append(f'{name}: 缺标记 {missing}')
    # 抽取审计：复刻层应以 bitcast 返回收尾（防止只抠到半个函数）
    tail = out.get('_round_fp8_half', {}).get('last_line', '')
    if tail and 'bitcast=True' not in tail:
        problems.append(f'_round_fp8_half: 末行不像函数结尾: {tail!r}')
    # 复刻层必须比回退版长（回退版是简化实现）；若反了说明抠错了函数
    a = out.get('_round_fp8_half', {})
    b = out.get('_round_fp8_half_fallback', {})
    if a and b and not a['chars'] > b['chars']:
        problems.append(f"复刻层({a['chars']}) 应长于回退版({b['chars']})")
    if problems:
        raise SystemExit('[check] 源码抽取自检失败:\n  - ' + '\n  - '.join(problems))
    return out


# --------------------------------------------------------------------- 生成被测模块
CVT_VARIANTS = {
    'cvt': "def _impl(x):\n    return x.to(tl.float8e4nv).to(tl.float16)\n",
    'cvt_rtne': ("def _impl(x):\n"
                 "    return x.to(tl.float8e4nv, fp_downcast_rounding='rtne').to(tl.float16)\n"),
    'noop': "def _impl(x):\n    return x\n",
}

MODULE_HEAD = '''"""自动生成（corr_strip_cvt_probe_v1.py）—— 请勿手改。"""
import triton
import triton.language as tl

_triton_jit = triton.jit

'''


KERNEL_TEMPLATE = '''
@triton.jit
def apply_{tag}(X, Y, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < N
    x = tl.load(X + off, mask, other=0.0)
    y = {call}
    tl.store(Y + off, y, mask)
'''


def build_variants_module(out_dir: Path, fusion: bool) -> tuple:
    """把「原语源码 + 每个变体一个内核」写成真文件再 import（exec 会丢源码行）。"""
    origs = {
        '_round_fp8_half': extract_func(IMPL_FILE, '_round_fp8_half'),
        '_round_fp8_half_fallback': extract_func(NAT_FILE, '_round_fp8_half_fallback'),
    }
    calls = {
        'orig': '_round_fp8_half(x)',
        'fallback': '_round_fp8_half_fallback(x)',
    }
    for tag in CVT_VARIANTS:
        calls[tag] = f'_{tag}_impl(x)'

    # 每个变体各自一份 _impl 定义，避免同名互撞
    variant_defs = []
    for tag, src in CVT_VARIANTS.items():
        renamed = src.replace('def _impl(', f'def _{tag}_impl(')
        variant_defs.append(f'# ---- 变体 {tag} ----\n{renamed}\n')

    body = MODULE_HEAD + '\n'.join(
        [f'# ---- 逐字抽取: {n} ----\n{s}\n' for n, s in origs.items()]
        + variant_defs
        + [KERNEL_TEMPLATE.format(tag=t, call=c) for t, c in calls.items()])

    out_dir.mkdir(parents=True, exist_ok=True)
    mod_path = out_dir / ('_cvt_kernels_fusion%d.py' % int(bool(fusion)))
    mod_path.write_text(body, encoding='utf-8', newline='\n')

    spec = importlib.util.spec_from_file_location(mod_path.stem, mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_path.stem] = module
    spec.loader.exec_module(module)
    return module, calls, mod_path, origs


# --------------------------------------------------------------------- 计时
def bench_kernel(torch, fn, x, y, n, block, fusion, reps, warmup=3):
    import triton
    grid = (triton.cdiv(n, block),)

    def once():
        fn[grid](x, y, n, block, enable_fp_fusion=fusion)

    try:
        for _ in range(warmup):
            once()
        torch.xpu.synchronize()
    except Exception:
        return None, traceback.format_exc()

    times = []
    for _ in range(reps):
        start = torch.xpu.Event(enable_timing=True)
        stop = torch.xpu.Event(enable_timing=True)
        start.record()
        once()
        stop.record()
        torch.xpu.synchronize()
        times.append(start.elapsed_time(stop))
    times.sort()
    return times[len(times) // 2], None


def compile_only(torch, fn, n, block, fusion):
    """只做编译+一发，用来验证某个实现能否编译（不看时间）。"""
    import triton
    x = torch.zeros(n, dtype=torch.float16, device='xpu')
    y = torch.empty_like(x)
    try:
        fn[(triton.cdiv(n, block),)](x, y, n, block, enable_fp_fusion=fusion)
        torch.xpu.synchronize()
        return True, None
    except Exception:
        return False, traceback.format_exc()


def replicate_harness_probe() -> dict:
    """逐字复刻 ``corr_strip_nat_v1._probe_hardware_cvt``，但**不吞异常**。

    harness 的探针返回 False 时会切到 ``_round_fp8_half_fallback``，而它用
    ``except Exception`` 把原因丢掉了 —— r9 的 ``hardware_cvt='unavailable, fell back'``
    因此无法归因。这里把真实异常原文留下。

    ⚠️ 必须逐字复刻，包括 ``num_warps=1`` 与 ``torch.zeros(16)``：若失败原因正是
    ``num_warps=1``，用别的参数复现就会漏掉它。
    """
    import torch
    import triton
    out = dict(attempted=True, torch_xpu_available=None, ok=None, error=None)
    try:
        out['torch_xpu_available'] = bool(hasattr(torch, 'xpu') and torch.xpu.is_available())
    except Exception:
        out['torch_xpu_available'] = False
    if not out['torch_xpu_available']:
        out['ok'] = False
        out['error'] = 'torch.xpu.is_available() is False'
        return out

    try:
        @triton.jit
        def _probe(x):
            return x.to(tl.float8e4nv).to(tl.float16)

        inp = torch.zeros(16, dtype=torch.float16, device='xpu')
        _probe[inp.numel():](inp, num_warps=1)
        torch.xpu.synchronize()
        out['ok'] = True
    except Exception:
        out['ok'] = False
        out['error'] = traceback.format_exc()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--check', action='store_true', help='CPU-only：只验证源码抽取')
    parser.add_argument('--sizes', default='65536,1048576,4194304')
    parser.add_argument('--block', type=int, default=512)
    parser.add_argument('--reps', type=int, default=30)
    parser.add_argument('--check-reps', type=int, default=30)
    args = parser.parse_args()

    if args.check:
        info = check_extraction()
        print('corr_strip_cvt_probe --check: OK')
        for name, row in info.items():
            print(f'  {name:26s} {row["lines"]:>3d} 行 / {row["chars"]:>5d} 字符'
                  f'  首行={row["first_line"]}')
            print(f'  {"":26s} 末行={row["last_line"]}')
        print('  变体内联: ' + ', '.join(CVT_VARIANTS))
        return

    if args.out is None:
        parser.error('--out is required unless --check')

    try:
        import torch
        import triton
    except Exception:
        traceback.print_exc()
        raise SystemExit('[probe] 需要 torch/triton（真跑路径）')

    report = dict(
        kind='corr-strip-e4m3-impl-microbench',
        triton_version=getattr(triton, '__version__', '?'),
        torch_version=getattr(torch, '__version__', '?'),
        xpu_available=bool(getattr(torch, 'xpu', None) and torch.xpu.is_available()),
        block=args.block, sizes=[int(s) for s in args.sizes.split(',') if s.strip()],
        reps=args.reps,
        note=('孤立微基准，只用于定价与定性；不得外推为帧墙收益。'
              '内核形状逐字复制 nr_backend/triton_fp8.py 的 _kernel（B=512, fusion=False）。'),
        compile=dict(), timing=dict(), errors=dict(),
    )
    if not report['xpu_available']:
        print('[probe] XPU 不可用 ⇒ 只能做 CPU 自检', file=sys.stderr)
        report['compile']['_note'] = 'XPU unavailable'
        (args.out / 'cvt_probe.json').write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        return

    # 0) 先把 harness 那个吞异常的探针复刻一遍，拿到真实原因
    report['harness_probe'] = replicate_harness_probe()
    if report['harness_probe']['ok']:
        print('[harness-probe] 复刻成功：harness 探针本应返回 True（r9 却记录了 fell back）')
    else:
        tail = [l for l in str(report['harness_probe']['error'] or '').strip().splitlines()
                if l.strip()][-1:] or ['<无>']
        print('[harness-probe] 复刻失败：%s' % tail[0][:300])

    n_probe = int(report['sizes'][0])
    module = None
    for fusion in (False, True):
        tag = 'fusion=%d' % int(fusion)
        try:
            module, calls, mod_path, origs = build_variants_module(
                args.out / 'gen', fusion)
        except Exception:
            report['errors'][tag] = traceback.format_exc()
            print(f'[{tag}] 生成/导入被测模块失败:\n{traceback.format_exc()}', file=sys.stderr)
            continue
        report.setdefault('module', str(mod_path))
        report.setdefault('extracted', {k: dict(chars=len(v), lines=v.count('\n'))
                                        for k, v in origs.items()})

        row = {}
        for name, call in calls.items():
            fn = getattr(module, 'apply_' + name)
            ok, err = compile_only(torch, fn, n_probe, args.block, fusion)
            row[name] = ok
            if not ok:
                report['errors'].setdefault(name, {})[tag] = err
        report['compile'][tag] = row

        timing = {}
        for size in report['sizes']:
            x = torch.randn(size, dtype=torch.float16, device='xpu') * 0.5
            y = torch.empty_like(x)
            for name in calls:
                if not row.get(name):
                    continue
                fn = getattr(module, 'apply_' + name)
                med, err = bench_kernel(torch, fn, x, y, size, args.block,
                                        fusion, args.reps)
                if err is not None:
                    report['errors'].setdefault(name, {})[f'{tag}/size={size}'] = err
                    continue
                # noop 的字节吞吐（读+写各 2 字节/元素）
                gbs = (size * 2 * 2) / (med * 1e-3) / 1e9
                timing.setdefault(str(size), {})[name] = dict(ms=med, gbs=gbs)
            del x, y
        report['timing'][tag] = timing

    report['any_variant_compiled'] = any(
        v for row in report['compile'].values() for v in row.values())
    out_json = args.out / 'cvt_probe.json'
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    # ---- 打印 ----
    print('=' * 92)
    print('triton %s / torch %s / XPU=%s' %
          (report['triton_version'], report['torch_version'], report['xpu_available']))
    for name, row in report['timing'].items():
        print('\n[%s]' % name)
        sizes = sorted(row, key=int)
        names = [n for n in ('noop', 'orig', 'fallback', 'cvt', 'cvt_rtne')
                 if any(n in row[s] for s in sizes)]
        print('  %-12s %s' % ('size', ''.join('%14s' % n for n in names)))
        for size in sizes:
            cells = []
            for n in names:
                cell = row[size].get(n)
                cells.append('%9.1fus' % (cell['ms'] * 1000) if cell else '%14s' % '-')
            print('  %-12s %s' % (size, ''.join('%14s' % c for c in cells)))
        for size in sizes:
            noop = row[size].get('noop')
            if not noop:
                continue
            print('       ↑ size=%s  noop 吞吐 %.1f GB/s；'
                  'orig/noop = %.2fx，cvt/noop = %s' %
                  (size, noop['gbs'], (row[size]['orig']['ms'] / noop['ms'])
                   if 'orig' in row[size] else float('nan'),
                   ('%.2fx' % (row[size]['cvt']['ms'] / noop['ms']))
                   if 'cvt' in row[size] else '（未编成）'))
    print('\n[harness 探针复刻]')
    hp = report['harness_probe']
    print('  torch.xpu.is_available()=%s  ok=%s' % (hp['torch_xpu_available'], hp['ok']))
    if not hp['ok']:
        for line in str(hp['error'] or '').splitlines():
            print('  | %s' % line)
    print('\n[编译结果]')
    for tag, row in report['compile'].items():
        print('  %-10s %s' % (tag, row))
    if not any(v for row in report['compile'].values() for v in row.values()):
        print('\n⚠️ FATAL 口径：**一个变体都没编成** ⇒ 本轮的计时表为空、'
              '不构成任何结论。\n'
              '   最常见原因：TRITON_CACHE_DIR 指向全新空目录，导致 Triton 的运行时扩展\n'
              '   `spirv_utils*.pyd` 缺失（见 cvt_probe_run_v1.sh 的「播种运行时扩展」段）。')
    if report['errors']:
        print('\n[异常]')
        for name, per in report['errors'].items():
            keys = list(per) if isinstance(per, dict) else ['<root>']
            for k in keys:
                txt = (per[k] if isinstance(per, dict) else per) or ''
                last = [l for l in txt.strip().splitlines() if l.strip()][-1:]
                print('  %s / %s -> %s' % (name, k, last[0][:160] if last else ''))
    print('\n产物: %s' % out_json)


if __name__ == '__main__':
    main()
