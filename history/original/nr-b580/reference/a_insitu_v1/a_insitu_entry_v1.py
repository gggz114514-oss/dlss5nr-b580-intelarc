"""A 的设备侧在场判定：把快速线**已经落到** `FastMatrices.dense` 的调用强制走 W8A8 INT8。

要回答的那一问
--------------
`to_single_digit_v1/RESULT.md` §6 行 A 把「8 个 C32 dense/投影 走 INT8 XMX」列为杠杆，
`a_recheck_v1` 已把它的 9–14 ms 量级估计作废（同一实现上的既有实测方向相反）。
本轮把 A 放到**快速线本体**上、用**设备侧尺子**判一次生死。

刀的形状（见 `PLAN.md` §3–§4）
------------------------------
**不做** `select('int8_dense')` —— 那会同时翻转三处都以 `mode == 'fp16_xmx'` 为条件的闸门
（POLICY tile、K8 精确点、批量分支 MLP），计数变化集里会混进与 A 无关的大块位移。

**做**：运行期替换 `fast_matrices_v3.FastMatrices.dense` 为包装器，`mode` 保持 `fp16_xmx`，
只把**已经落到 `FastMatrices.dense` 的那批调用**改成 `int8=True`。
配 `provider.prepack(model)` 让权重只量化一次。

⇒ 唯一改变的是 `fast_matrices_v3._matmul` 人口：**115 次/帧、6.9212 ms/帧**（见 PLAN.md §2）。
⇒ 其余一切（K8 点、POLICY tile、批量 MLP、ATen 侧）两臂**逐字相同**。

装置
----
* 剖析装置与补丁串**逐字复用** `l3_1_insitu_entry_v1`（同一把尺子）；
* 新增一个锚点：`stack = session._stack` 之后注入 `ARM.apply(stack, report)`（预量化 + 留痕）。

⚠️ 诊断跑：不产正确性产物、不进任何门槛、不写 fastpath；绝对毫秒只在本轮同索引间可比。
⚠️ 剖析遍可能在解释器**拆卸期**触发 Windows fail-fast（0xC0000409）
   ⇒ 驱动脚本**不能 `set -e`**，只以报告是否落盘判成败。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
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

PROFILED_TAIL = ruler.PROFILED_TAIL
ON = ruler.ON

# ---------------------------------------------------------------- 新锚点
ARM_ANCHOR_OLD = (
    "        session = Session.create(paths.NR, paths.PROFILE, config['profile_sha256'])\n"
    "        stack = session._stack\n"
)
ARM_ANCHOR_NEW = ARM_ANCHOR_OLD + "        ARM.apply(stack, report)\n"

STATE = dict(enabled=False, dense_patched=False, prepack_seconds=None, packed_buffers=None,
             packed_bytes=None, packed_hit=0, packed_miss=0, int8_calls=0,
             kernels=[], provider=None, mode=None)


class Arm:
    """会话建立后、第一帧之前：给 provider 预量化权重，并留痕。"""

    def apply(self, stack, report):
        import torch
        enabled = STATE['enabled']
        info = dict(name='on' if enabled else 'off',
                    env=dict(NR_A_INT8=os.environ.get('NR_A_INT8', '<unset>')),
                    provider=type(stack.provider).__name__,
                    mode=stack.provider.mode,
                    dense_patched=STATE['dense_patched'])
        STATE['provider'] = type(stack.provider).__name__
        STATE['mode'] = stack.provider.mode
        if enabled:
            start = time.perf_counter()
            stack.provider.prepack(stack.model)
            torch.xpu.synchronize()
            elapsed = time.perf_counter() - start
            packed = stack.provider.packed
            total = sum(v[1].numel() * v[1].element_size() + v[2].numel() * v[2].element_size()
                        for v in packed.values())
            STATE.update(prepack_seconds=elapsed, packed_buffers=len(packed), packed_bytes=total)
            info.update(prepack_seconds=elapsed, packed_buffers=len(packed), packed_bytes=total)
            assert packed, 'prepack 一个权重都没打包 —— 装置没接上'
        report['arm'] = info


def build_patches():
    """四个剖析补丁逐字复用 `l3_1_insitu_v1`，再加一个「会话后挂钩」锚点。"""
    import dense_ab_entry_v1 as da
    patches = (
        da.PATCHES[0],                                     # callsite_namer+trace+summarise
        ('profiled_frame_tail', da.OLD_SELECT, ruler.MY_NEW_SELECT),
        ('profile_pass_tail', da.OLD_CACHEHITS, ruler.MY_NEW_CACHEHITS),
        da.PATCHES[3],                                     # adapter_provenance
        ('arm_injection', ARM_ANCHOR_OLD, ARM_ANCHOR_NEW),
    )
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
    module.DENSE_AB = dict(kind='a-insitu-int8-dense',
                           wrapper=str(Path(__file__).resolve()), **provenance)
    module.ARM = Arm()
    sys.modules['materials_entry_v1'] = module
    exec(compile(patched, str(MATERIALS), 'exec'), module.__dict__)
    return module, provenance


def install_knife():
    """把 `FastMatrices.dense` 换成强制 int8 的包装器；返回原函数以便还原。"""
    import fast_matrices_v3 as fm

    original = fm.FastMatrices.dense

    def forced_int8_dense(self, a, w, *, chunk_k, initial=None, **kwargs):
        if self.mode == 'baseline' or chunk_k == 8:
            return original(self, a, w, chunk_k=chunk_k, initial=initial, **kwargs)
        item = self.packed.get(id(w))
        if item is not None and item[0] is w:
            packed = (item[1], item[2])
            STATE['packed_hit'] += 1
        else:
            packed = None
            STATE['packed_miss'] += 1
        out, compiled = fm.dot(a, w, initial=initial, int8=True, packed=packed)
        STATE['int8_calls'] += 1
        STATE['kernels'].append(getattr(compiled, 'name', str(compiled)))
        self.record('int8_dense')
        if 'int8_dense' not in self.compiled:
            self.compiled['int8_dense'] = compiled
        return out

    forced_int8_dense.__a_int8__ = True
    fm.FastMatrices.dense = forced_int8_dense
    return original


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
            'arm', 'profiled', 'profile_error')
    out = {key: payload.get(key) for key in keep if key in payload}
    out['frames'] = [
        {key: row.get(key) for key in ('index', 'ms', 'rmse', 'byte_equal', 'max_abs', 'ffn_calls')}
        for row in payload.get('frames') or []
    ]
    return out, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', choices=['probe', 'full'], default='probe')
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--frames', type=int, default=243)
    parser.add_argument('--c512-arith', dest='c512_arith', default='int8')
    parser.add_argument('--exact-validation', dest='exact_validation', type=Path, required=True)
    parser.add_argument('--baseline-validation', dest='baseline_validation', type=Path, default=None)
    parser.add_argument('--cache', type=Path, default=Path('D:/fullsize-rows-v1/r1/triton-cache'))
    parser.add_argument('--allow-compile', dest='allow_compile', action='store_true')
    parser.add_argument('--check', action='store_true',
                        help='只做锚点/语法自检后退出（CPU-only，不占 GPU、不 import torch）')
    args = parser.parse_args()

    # 1) 先注册打补丁的 harness（在任何 import materials_entry_v1 之前）
    patches = build_patches()
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
        print('a_insitu --check: OK')
        print('  harness   :', provenance['source'])
        print('  ruler     :', provenance['ruler'])
        print('  patches   :', ','.join(provenance['patches']))
        print('  source_sha256 :', provenance['source_sha256'])
        print('  patched_sha256:', provenance['patched_sha256'])
        print('  profile_tail  :', provenance['profile_tail'])
        return

    # 2) 与 l3_1 逐项一致的会话引导
    import gapfill_entry_v1 as gap
    args.out.mkdir(parents=True, exist_ok=True)
    guard, paths_ = gap.prologue(args.cache)
    del guard, paths_

    # 3) 刀：provider 层。EXPERIMENTAL 必须先于 nr_backend 进 sys.path。
    if str(EXPERIMENTAL) not in sys.path[:1]:
        sys.path.insert(0, str(EXPERIMENTAL))
    STATE['enabled'] = os.environ.get('NR_A_INT8', 'off').strip().lower() in ON
    original_dense = None
    if STATE['enabled']:
        original_dense = install_knife()
        STATE['dense_patched'] = True

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

    if STATE['enabled']:
        import fast_matrices_v3 as fm
        fm.FastMatrices.dense = original_dense

    harness, payload_error = extract(args.out / 'run' / 'validation.json')

    # ⚠️ 口径：`profile_pass_tail` 补丁只替换了 `report['cache_hits']` 那一行，
    #    **原 harness 的 `for row in selected: one_frame(row)` 循环仍在** ⇒ 每个被选帧被处理**两次**
    #    （一次未剖析、一次剖析）。这对设备读数无害（每次剖析帧各自成行），
    #    但刀的计数器必须按真实模型运行次数摊分。
    selected_n = (args.frames if args.phase == 'full' else min(args.limit, args.frames))
    tail_n = selected_n if selected_n <= PROFILED_TAIL else PROFILED_TAIL
    model_runs = selected_n + tail_n
    per_run = (STATE['int8_calls'] / model_runs) if model_runs else None

    report = dict(
        kind='a-insitu-int8-dense',
        arm='on' if STATE['enabled'] else 'off',
        env=dict(NR_A_INT8=os.environ.get('NR_A_INT8', '<unset>')),
        phase=args.phase, limit=args.limit, frames=args.frames,
        model_runs=model_runs,
        allow_compile=bool(args.allow_compile),
        error=error.splitlines()[-1] if error else None,
        provenance=provenance,
        knife=dict(dense_patched=STATE['dense_patched'], provider=STATE['provider'],
                   mode=STATE['mode'], int8_calls=STATE['int8_calls'],
                   int8_calls_per_run=per_run,
                   packed_hit=STATE['packed_hit'], packed_miss=STATE['packed_miss'],
                   prepack_seconds=STATE['prepack_seconds'],
                   packed_buffers=STATE['packed_buffers'], packed_bytes=STATE['packed_bytes'],
                   kernels=sorted(set(STATE['kernels']))),
        harness=harness,
        payload_error=payload_error,
        profiled_indices=[row.get('index') for row in (harness.get('profiled') or [])],
        profiled=harness.get('profiled') or [],
        note=('device track only; wall ms are inflated by the profiler and are not used. '
              'Absolute device ms is same-round/same-index comparable only.'),
    )
    out_json = args.out / 'a_insitu.json'
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    summary = dict(arm=report['arm'], knife=report['knife'],
                   profiled_indices=report['profiled_indices'],
                   gpu_busy_ms=[round(row.get('gpu_busy_ms', 0.0), 4) for row in report['profiled']],
                   gpu_instances=[row.get('gpu_instances') for row in report['profiled']],
                   harness_passed=harness.get('passed'),
                   cache_mode=harness.get('cache_mode'),
                   profile_error=harness.get('profile_error'),
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
