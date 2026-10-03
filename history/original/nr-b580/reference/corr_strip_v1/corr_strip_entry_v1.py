"""corr_strip_v1 臂入口：把「修正层」四个原语在现行快速线上运行期替换成原生等价物，
同一条快速线、同一进程、同索引下量出设备侧能省多少。产品源码 0 改动、冻结缓存 0 改动。

臂设计（PLAN.md §4）：off（基准）/ nat（只换原语）/ alg（nat + 把 _normalize 的
XOR 归约树换成 tl.sum）。本入口通过环境变量 ``NR_CORR_STRIP`` ∈ {off, nat, alg} 选臂。

装置**逐字复用** ``a_insitu_v1``（同一把设备侧尺子：dense_ab 剖析遍 + l3_1 的
补丁串），只新增一个锚点：``Session.create`` 之后、第一帧之前注入
``ARM.apply(stack, report)``，在那里调 ``corr_strip_nat_v1.install(mode)``。

⚠️ 诊断跑：不产正确性产物、不进任何门槛、不写 fastpath；绝对毫秒只在本轮同索引间可比。
⚠️ 剖析遍可能在解释器拆卸期触发 Windows fail-fast（0xC0000409）
   ⇒ 驱动脚本**不能 set -e**，只以报告是否落盘判成败。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE.parent
GAPFILL = REF / 'gapfill_v1'
DENSE_AB = REF / 'dense_ab_v1'
MATERIALS = REF / 'materials_v1' / 'materials_entry_v1.py'
RULER_DIR = REF / 'l3_1_insitu_v1'
W = Path('E:/ComfyUI-aki-v3-IntelArc_20260722')
EXPERIMENTAL = W / 'nr-b580-int8/experimental'

for item in (str(HERE), str(RULER_DIR), str(GAPFILL), str(DENSE_AB)):
    if item not in sys.path:
        sys.path.insert(0, item)

import l3_1_insitu_entry_v1 as ruler  # noqa: E402  —— 同一把尺子：复用它的两个补丁串
import corr_strip_nat_v1 as nat  # noqa: E402  —— 原生原语集 + 运行期替换
import corr_strip_manifest_v1 as cman  # noqa: E402  —— 编译清单纯收集（不编译，供并行预编译）

PROFILED_TAIL = ruler.PROFILED_TAIL
ON = ruler.ON

# ---------------------------------------------------------------- 新锚点
ARM_ANCHOR_OLD = (
    "        session = Session.create(paths.NR, paths.PROFILE, config['profile_sha256'])\n"
    "        stack = session._stack\n"
)
ARM_ANCHOR_NEW = ARM_ANCHOR_OLD + "        ARM.apply(stack, report)\n"

MODES = ('off', 'nat', 'alg')

# ---------------------------------------------------------------- 清单收集专用补丁
# harness 在计时路径里设了两道「派发计数」断言（每帧必须正好 16 次 c512 + 8 次 vit）。
# 清单收集遍**刻意不派发任何内核**（只记录要编哪些），计数必然为 0，会当场把链路打断
# —— 实测：链路断在第 1 帧末尾，清单只收到 150/165 条。这两道断言只在**清单模式**下放开，
# 正式臂照旧带着它们跑（那是这条尺子的健全性检查，不能拿掉）。
MANIFEST_ASSERT_OLD = (
    "            assert counts['c512'] - before['c512'] == 16 and counts['vit'] - before['vit'] == 8, counts\n"
    "            totals = counters.totals()\n"
    "            assert totals == expected_counters, (totals, expected_counters)\n"
)
MANIFEST_ASSERT_NEW = (
    "            # [manifest mode] 内核不派发 ⇒ 派发计数必然不符，此处不设卡；\n"
    "            # 本遍只为收集编译清单，不产生任何设备结论。\n"
    "            totals = counters.totals()\n"
)


def resolve_mode(env_value):
    """把 NR_CORR_STRIP 解析成 off/nat/alg；非法值一律视为 off（默认）。"""
    v = (env_value or 'off').strip().lower()
    return v if v in MODES else 'off'


# 收尾前是否把 nat 的替换还原回去（理由见 ``Arm`` 的类文档）。置 0 可关掉做对照。
RESTORE_BEFORE_CLOSE = os.environ.get('NR_CORR_RESTORE_BEFORE_CLOSE', '1') != '0'

# nat 会改写的全部全局名（原语 + alg 模式的结构性归约）
_REPLACED_NAMES = tuple(nat._PRIMITIVE_NAMES) + tuple(nat._STRUCT_NAMES)


def _snapshot_primitives():
    """拍照：快线模块里那些将被 nat 改写的全局名的**当前值**。须在 ``nat.install`` 之前调。"""
    snapshot = []
    for mod in list(sys.modules.values()):
        if not nat._is_fastline_module(mod):
            continue
        namespace = mod.__dict__
        for name in _REPLACED_NAMES:
            if name in namespace:
                snapshot.append((mod, name, namespace[name]))
    return snapshot


def _restore_primitives(snapshot):
    """把拍下的值写回模块；返回还原处数。"""
    for mod, name, old in snapshot:
        mod.__dict__[name] = old
    return len(snapshot)


class Arm:
    """会话建立后、第一帧之前：运行期替换修正原语，并留痕。

    ⚠️ **收尾前必须还原**（``RESTORE_BEFORE_CLOSE``，默认开）。nat 替换的四个原语名
    （``_nan_left`` / ``rsqrt_half_clamped`` / ``_round_fp8_half`` / ``_half_fma_value``）
    与实验栈 ``fork_fp8_jit_v2.Fork`` 登记的 bindings 是**同一批全局名**。Fork 在
    ``close()`` 期的 ``verify_restored()`` 会断言「这些名字应已还原成原始 JITFunction」——
    nat 按设计不还原，断言必然失败，harness 遂把 ``passed`` 覆盖成 False（记 ``close_error``），
    而报告器拿 ``passed`` 当硬门槛 ⇒ 该臂被判 INVALID、直接中止出报告。

    测量期**必须**让替换生效（那正是被测对象）；但进程退出时应当干净 —— 这也正是 Fork 自检
    的本意。所以在 ``stack.close`` 之前把被替换的名字还原回快照值：**测量一分不受影响**，
    Fork 的还原断言按其本意通过。还原处数与人名一并留痕到 ``report['restore_before_close']``。
    """

    def apply(self, stack, report):
        mode = resolve_mode(os.environ.get('NR_CORR_STRIP'))
        snapshot = _snapshot_primitives()          # ← 必须在 install 之前拍照
        trace = nat.install(mode)
        info = dict(name=mode,
                    env=dict(NR_CORR_STRIP=os.environ.get('NR_CORR_STRIP', '<unset>')),
                    provider=type(stack.provider).__name__,
                    mode=mode,
                    install_trace=trace,
                    restore_before_close=bool(mode != 'off' and RESTORE_BEFORE_CLOSE))
        report['arm'] = info
        report['strip_mode'] = mode
        report['install_trace'] = trace

        if mode == 'off' or not RESTORE_BEFORE_CLOSE:
            report['restore_before_close'] = dict(
                enabled=False, count=0,
                why='off 臂不替换' if mode == 'off' else 'NR_CORR_RESTORE_BEFORE_CLOSE=0')
            return

        original_close = stack.close

        def close_after_restore(*args, **kwargs):
            report['restore_before_close'] = dict(
                enabled=True, count=_restore_primitives(snapshot),
                names=list(_REPLACED_NAMES),
                why='nat 与 fork_fp8_jit_v2.Fork 争同一批全局名，收尾前还原让 Fork 自检按其本意通过')
            return original_close(*args, **kwargs)

        stack.close = close_after_restore


def build_patches(manifest_mode=False):
    """四个剖析补丁逐字复用 `l3_1_insitu_v1`（同一把尺子），再加一个「会话后挂钩」锚点。

    ``manifest_mode=True`` 时**额外**放开 harness 里的两道派发计数断言 —— 那是清单
    收集遍特有的需要（详见 ``MANIFEST_ASSERT_OLD`` 上方注释），正式臂不加这个补丁。
    """
    import dense_ab_entry_v1 as da
    patches = (
        da.PATCHES[0],                                     # callsite_namer+trace+summarise
        ('profiled_frame_tail', da.OLD_SELECT, ruler.MY_NEW_SELECT),
        ('profile_pass_tail', da.OLD_CACHEHITS, ruler.MY_NEW_CACHEHITS),
        da.PATCHES[3],                                     # adapter_provenance
        ('arm_injection', ARM_ANCHOR_OLD, ARM_ANCHOR_NEW),
    )
    if manifest_mode:
        # ★ 必须排在**最前面**：`profiled_frame_tail` 补丁插入的那段剖析代码里也带着
        #   同样三行（剖析遍有自己的计时循环），等它落进去之后再匹配就会变成 2 处、
        #   被 `load_patched_materials` 当场拒绝。
        patches = (('manifest_dispatch_asserts',
                    MANIFEST_ASSERT_OLD, MANIFEST_ASSERT_NEW),) + patches
    return patches


def load_patched_materials(patches):
    """把打了补丁的 harness 塞进 `sys.modules`，**先于**任何 import 它的人。"""
    text = MATERIALS.read_text(encoding='utf-8')
    patched = text
    for label, old, new in patches:
        count = patched.count(old)
        if count != 1:
            raise SystemExit(
                f'patch {label!r}: anchor found {count} times in {MATERIALS}; refusing to patch')
        patched = patched.replace(old, new)
    provenance = dict(
        source=str(MATERIALS),
        source_sha256=hashlib.sha256(MATERIALS.read_bytes()).hexdigest(),
        patched_sha256=hashlib.sha256(patched.encode('utf-8')).hexdigest(),
        patches=[label for label, _, _ in patches],
        profile_tail=PROFILED_TAIL,
        ruler=str(Path(ruler.__file__).resolve()),
    )
    module = types.ModuleType('materials_entry_v1')
    module.__file__ = str(MATERIALS)
    # `adapter_provenance` 补丁会写 `report['dense_ab'] = DENSE_AB`；
    # 新锚点会调 `ARM.apply` —— exec 进模块时必须显式提供这两个全局。
    module.DENSE_AB = dict(kind='corr-strip-correction-removal',
                           wrapper=str(Path(__file__).resolve()), **provenance)
    module.ARM = Arm()
    sys.modules['materials_entry_v1'] = module
    exec(compile(patched, str(MATERIALS), 'exec'), module.__dict__)
    return module, provenance


def extract(validation: Path):
    """从 harness 的 validation.json 里只取分析需要的字段（不给本轮携带冗余）。"""
    if not validation.is_file():
        return {}, None
    try:
        payload = json.loads(validation.read_text(encoding='utf-8-sig'))
    except BaseException:                                    # noqa: BLE001
        return {}, traceback.format_exc()
    keep = ('passed', 'phase', 'cache_mode', 'cache_hits', 'selected_frames',
            'expected_counters', 'installed_ffn', 'timing', 'error', 'close_error',
            'arm', 'profiled', 'profile_error', 'strip_mode', 'install_trace',
            'restore_before_close')
    out = {key: payload.get(key) for key in keep if key in payload}
    out['frames'] = [
        {key: row.get(key) for key in ('index', 'ms', 'rmse', 'byte_equal', 'max_abs', 'ffn_calls')}
        for row in payload.get('frames') or []
    ]
    return out, None


def main():
    parser = argparse.ArgumentParser()
    # `--out` / `--exact-validation` 仅非 --check 路径必填；
    # 这样 `python ... --check` 才能裸跑（CPU-only 自检不碰 GPU / 不读素材）。
    parser.add_argument('--out', type=Path, required=False)
    parser.add_argument('--phase', choices=['probe', 'full'], default='probe')
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--frames', type=int, default=243)
    parser.add_argument('--c512-arith', dest='c512_arith', default='int8')
    parser.add_argument('--exact-validation', dest='exact_validation', type=Path, required=False)
    parser.add_argument('--baseline-validation', dest='baseline_validation', type=Path, default=None)
    parser.add_argument('--cache', type=Path, default=Path('D:/fullsize-rows-v1/r1/triton-cache'))
    parser.add_argument('--allow-compile', dest='allow_compile', action='store_true')
    parser.add_argument('--manifest', type=Path, default=None,
                        help='只收集「本臂要编哪些内核」清单并跳过实际编译（秒级跑完），'
                             '产物交给 corr_strip_parcompile_v1 多进程并行编译')
    parser.add_argument('--check', action='store_true',
                        help='只做锚点/语法自检后退出（CPU-only，不占 GPU、不 import torch）')
    args = parser.parse_args()

    if not args.check and (args.out is None or args.exact_validation is None):
        parser.error('--out and --exact-validation are required unless --check is given')

    # 1) 先注册打补丁的 harness（在任何 import materials_entry_v1 之前）
    patches = build_patches(manifest_mode=args.manifest is not None)
    materials, provenance = load_patched_materials(patches)

    if args.check:
        import dense_ab_entry_v1 as da
        assert callable(materials.main), 'patched harness has no main()'
        assert hasattr(materials, 'CallSiteNamer') and hasattr(materials, 'summarise'), \
            'profiler helpers missing from patched harness'
        assert hasattr(materials, 'ARM'), 'arm hook missing from patched harness'
        source = MATERIALS.read_text(encoding='utf-8')
        assert source.count(ARM_ANCHOR_OLD) == 1, 'arm anchor not unique in the harness'
        assert 'ARM.apply' not in source, 'harness already carries an arm hook'
        patched_text = material_text(patches)
        assert patched_text.count('ARM.apply(stack, report)') == 1, 'arm hook not injected exactly once'
        injected = ruler.MY_NEW_CACHEHITS + ruler.MY_NEW_SELECT
        assert 'selected[-3:]' in injected and 'profiled_frame' in injected
        assert da.NEW_SELECT != ruler.MY_NEW_SELECT, 'tail patch did not diverge from dense_ab'
        for label, old, new in patches:
            assert old != new, label
        # NR_CORR_STRIP 三档都能解析；非法值降级为 off
        for m in MODES:
            assert resolve_mode(m) == m, f'mode {m!r} not parsed'
        assert resolve_mode('garbage') == 'off', 'invalid mode should fall back to off'
        assert resolve_mode('') == 'off', 'empty mode should fall back to off'
        # corr_strip_nat_v1.install 可在纯 Python 层被调用（用假模块验证替换真实发生）
        nat._selfcheck_install()
        print('corr_strip --check: OK')
        print('  harness   :', provenance['source'])
        print('  ruler     :', provenance['ruler'])
        print('  patches   :', ','.join(provenance['patches']))
        print('  source_sha256 :', provenance['source_sha256'])
        print('  patched_sha256:', provenance['patched_sha256'])
        print('  profile_tail  :', provenance['profile_tail'])
        return

    # 1b) 清单收集模式：装上钩子（只记录、不编译、不派发），必须在第一帧之前。
    #     链路照常跑完，只是数值无意义 —— 本遍只为拿到「要编哪些内核」。
    if args.manifest is not None:
        cman.install(args.manifest)
        print(f'[corr-strip] manifest 收集已开（本遍不编译、不派发内核）: {args.manifest}',
              file=sys.stderr)

    # 2) 与 l3_1 / a_insitu 逐项一致的会话引导
    import gapfill_entry_v1 as gap
    args.out.mkdir(parents=True, exist_ok=True)
    guard, paths_ = gap.prologue(args.cache)
    del guard, paths_

    # 3) EXPERIMENTAL 必须先于 nr_backend 进 sys.path（与 l3_1 一致）
    if str(EXPERIMENTAL) not in sys.path[:1]:
        sys.path.insert(0, str(EXPERIMENTAL))
    mode = resolve_mode(os.environ.get('NR_CORR_STRIP'))

    entry_args = argparse.Namespace(
        phase=args.phase, c512_arith=args.c512_arith,
        exact_validation=args.exact_validation,
        baseline_validation=args.baseline_validation,
        out=args.out / 'run', cache=args.cache, limit=args.limit,
        frames=args.frames, allow_compile=args.allow_compile)

    error = None
    try:
        materials.main(entry_args)
    except BaseException:                                   # noqa: BLE001
        error = traceback.format_exc()
        print(error, file=sys.stderr)

    # 3b) 清单收集模式：把「这一臂碰到的每个内核」写盘后就可以收工了 ——
    #     本轮不产生任何设备结论，编译留给 corr_strip_parcompile_v1 多进程去做。
    if args.manifest is not None:
        summary = cman.dump(args.manifest, mode=mode)
        print('[corr-strip] manifest: ' + json.dumps(summary, ensure_ascii=False), file=sys.stderr)
        if summary['records'] == 0:
            print('[corr-strip] FATAL: 清单为空 —— 一个内核都没碰到，说明链路没跑起来',
                  file=sys.stderr)

    harness, payload_error = extract(args.out / 'run' / 'validation.json')

    report = dict(
        kind='corr-strip-correction-removal',
        arm=harness.get('arm'),
        strip_mode=harness.get('strip_mode', mode),
        install_trace=harness.get('install_trace'),
        env=dict(NR_CORR_STRIP=os.environ.get('NR_CORR_STRIP', '<unset>')),
        phase=args.phase, limit=args.limit, frames=args.frames,
        model_runs=None,
        allow_compile=bool(args.allow_compile),
        error=error.splitlines()[-1] if error else None,
        provenance=provenance,
        harness=harness,
        payload_error=payload_error,
        profiled_indices=[row.get('index') for row in (harness.get('profiled') or [])],
        profiled=harness.get('profiled') or [],
        note=('device track only; wall ms are inflated by the profiler and are not used. '
              'Absolute device ms is same-round/same-index comparable only. '
              'Product source and frozen cache are unchanged; the correction strip '
              'lives only in the profiled process memory.'),
    )
    out_json = args.out / 'corr_strip.json'
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    summary = dict(arm=report['arm'], strip_mode=report['strip_mode'],
                   profiled_indices=report['profiled_indices'],
                   gpu_busy_ms=[round(row.get('gpu_busy_ms', 0.0), 4) for row in report['profiled']],
                   gpu_instances=[row.get('gpu_instances') for row in report['profiled']],
                   harness_passed=harness.get('passed'),
                   cache_mode=harness.get('cache_mode'),
                   profile_error=harness.get('profile_error'),
                   install_trace=report['install_trace'],
                   error=report['error'])
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def material_text(patches):
    """自检辅助：返回补丁后被消费掉的原文拼接（用于断言锚点已被替换）。"""
    text = MATERIALS.read_text(encoding='utf-8')
    for _, old, new in patches:
        text = text.replace(old, new)
    return text


if __name__ == '__main__':
    main()
