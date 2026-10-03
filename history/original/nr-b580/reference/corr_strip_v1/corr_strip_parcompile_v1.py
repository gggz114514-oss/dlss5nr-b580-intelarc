"""corr_strip_parcompile_v1：拿编译清单，在多个进程里**并行**预编译（不占 GPU）。

配套 ``corr_strip_manifest_v1``：清单由「不编译地跑一遍链路」产出，本模块负责把
那些编译**真正做掉**、落进同一个 Triton 缓存副本里。因为用的是同一套
``ASTSource`` / ``target`` / ``options`` / 进程环境，算出的 cache key 与串行时
**逐位相同**，产物因此落在**同一个目录**里 —— 之后正式臂直接命中，无需再编。

## 为什么不占 GPU

``JITFunction.cache_key`` 来自**静态分析**（AST + 定义模块的 globals），
``ASTSource.hash()`` 只用它加上 signature/constants/attrs；整套 cache key
的计算不碰设备。编译本身走 icpx/ocloc 这类**子进程**，也不需要 GPU 上下文。
所以 16 个 worker 可以纯 CPU 并行。

## 需要小心的几处

1. **源码必须与主进程一致**：worker 里要先 ``import`` 清单涉及的模块，
   再 ``nat.install(mode)``（该函数遍历 ``sys.modules`` 替换全局名，
   模块没 import 就替换不到），然后才取 ``fn``。
2. **硬件 cvt 探测必须沿用主进程的结论**：``_probe_hardware_cvt`` 会真在 XPU 上
   编一个内核跑一次；worker 里禁止重探，直接把 ``_ROUND_FP8_HARDWARE_CVT``
   预置成主进程记录的值，并同步选好 ``_round_fp8_half`` 的变体。
3. **模块目录由清单携带**（``module_dirs``）：有些内核模块不在 backend/experimental
   之下（实测 ``c512_int8_ffn_rows_v1`` / ``int8_ffn_segment_rows_v1`` 在
   ``reference/fullsize_rows_v1/``），采集遍的 ``sys.path`` 是 harness 配的；
   worker 不照搬就会 ``ModuleNotFoundError``。
4. **``options`` / ``target`` 里的 tuple**：见 ``corr_strip_manifest_v1`` 的同名警告。
   裸存进 JSON 会让 list 顶替 tuple，cache key 长度不变、内容全错 —— 自检必须
   与清单逐条比对目录名，别只比「有没有编成功」。

## 判据

**成功 = 产物可用地落在清单记录的目录里**（``__grp__*.json`` + ``*.spv`` + 元数据
``*.json``，见 ``_is_done``），而不是「本进程里的 ``CompiledKernel`` 构造完成」。
实测有一例：``CompiledKernel.__init__`` 读 ``_matmul.ttir`` 时该文件已被并发清理，抛
``FileNotFoundError``，但 spv/llir/grp/json 齐全 —— 正式臂完全可用（四跑 164 命中证实）。
若按构造成功判，整条流水线会白等一次 7.5 分钟的串行重编。

顺带记下一个已知的 Triton 并发缺陷：``FileCacheManager.put`` 用
``os.replace`` 落盘，Windows 上撞到别人占着同名文件时抛 ``PermissionError``，
而它的处理是 ``os.remove(temp_path)`` —— **新内容被直接丢弃**，文件可能始终没落盘。
所以「worker 里编成功」不等于「文件一定在」，只有回到目录里核对才算数。

## 自检

每个任务在编译前用 ``get_cache_key`` 预先算一遍目录名，与清单里记录的
``cache_dir_name`` 比对。不一致 ⇒ 说明重建的 ``ASTSource`` / options 与原件不等价，
**报错而不是静默通过**（静默通过会让正式臂以为命中、实则重新编译），
并把 ``src_hash / backend_hash / options_hash / fn_cache_key`` 逐项记进报告，
下次直接定位到是哪一项漂的。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = Path('E:/ComfyUI-aki-v3-IntelArc_20260722')
BACKEND = W / 'nr-b580-int8/backend'
EXPERIMENTAL = W / 'nr-b580-int8/experimental'

sys.path.insert(0, str(HERE))
import corr_strip_manifest_v1 as manifest  # noqa: E402

_READY = {}


def _bootstrap_paths():
    for item in (str(BACKEND), str(EXPERIMENTAL)):
        if item not in sys.path:
            sys.path.insert(0, item)


def _resolve(module, qualname, name):
    """按 qualname/name 在模块里取回 ``JITFunction``。"""
    obj = module
    if qualname:
        for part in qualname.split('.'):
            if part == '<locals>':                      # 闭包内定义，取不到
                break
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                break
    if hasattr(obj, 'cache_key'):                       # 已经是 JITFunction
        return obj
    if name and hasattr(module, name):                  # 退回模块级名字
        return getattr(module, name)
    raise LookupError(f'cannot resolve {module.__name__}.{qualname} ({name})')


def _seal_removedirs():
    """把 ``os.removedirs`` 封成「只删这一层、绝不上溯」的 ``os.rmdir``。

    ``triton/runtime/cache.py::FileCacheManager.put()`` 结尾是
    ``os.removedirs(temp_dir)``，而 ``temp_dir`` 是 ``<cache>/<hash>/tmp.pid_*``。
    ``os.removedirs`` 会在删掉该目录之后**继续沿父目录往上删所有空目录** ——
    多进程同时往同一个缓存落盘时，这条上溯会把别人的目录一起带走。
    实测两次：16 进程跑批把一份 530 目录的副本削到 2 个、一份 362 目录的副本削到 46 个
    （残下的恰好是字母序最前的一批，正是「边跑边按序上溯删」的形状）。

    这里只保留它「删掉 temp_dir」的本意，上溯那半截直接砍掉 —— 缓存就再也不会
    被并发跑批擦掉。``put()`` 里的原子替换（temp 文件 + ``os.replace``）本身是安全的。
    """
    import os as _os

    if getattr(_os, '_corr_strip_sealed', False):
        return
    original = _os.removedirs

    def removedirs_shallow(name):
        try:
            _os.rmdir(name)
        except OSError:
            pass                      # 非空/不存在都无所谓；**绝不向父目录回溯**

    removedirs_shallow.__wrapped__ = original
    _os.removedirs = removedirs_shallow
    _os._corr_strip_sealed = True


def _init_worker(modules, mode, nat_state, module_dirs=()):
    """每个 worker 一次：封住上溯删除 → 路径 → import 模块 → 预置 cvt 结论 → install。

    缓存目录**沿用父进程的共享副本**（不另开），这样才能命中已经编好的
    ``arch_parser`` 之类的中间件（Intel 后端会把 ``arch_parser.c`` 编成 DLL 存在
    ``TRITON_CACHE_DIR`` 里；若每个 worker 各指一个空目录，每个进程都要重编一遍，
    而且实测加载会失败：``FileNotFoundError: Could not find module ...\\wXXXX\\...``）。
    并发的安全性由 ``_seal_removedirs`` + Triton 自身的原子替换保证。

    ``module_dirs`` 由清单携带（采集遍记下每个模块的 ``__file__`` 所在目录）：
    有些内核模块不在 backend/experimental 之下（如 ``reference/fullsize_rows_v1/``），
    没有这一步，worker 会以 ``ModuleNotFoundError`` 落空。
    """
    os.environ['CODEBUDDY_SAFE_DELETE_ENABLED'] = '0'
    _seal_removedirs()
    _bootstrap_paths()
    for item in reversed(list(module_dirs or ())):      # 清单自带的目录也上路径
        if item not in sys.path:
            sys.path.insert(0, item)
    import corr_strip_nat_v1 as nat

    nat._ALLOW_HW_PROBE = False                         # worker 绝不重探硬件
    verdict = (nat_state or {}).get('hardware_cvt')
    if verdict and verdict != 'not_probed':
        nat._ROUND_FP8_HARDWARE_CVT = verdict
        if verdict == 'available':
            nat._round_fp8_half = nat.__dict__['_round_fp8_half_cvt']
        elif 'fell back' in verdict:
            nat._round_fp8_half = nat.__dict__['_round_fp8_half_fallback']
    for name in modules:                                # 先 import，install 才替换得到
        try:
            importlib.import_module(name)
        except BaseException:                           # noqa: BLE001
            pass
    trace = nat.install(mode)
    _READY['installed'] = True
    _READY['mode'] = mode
    _READY['cache_dir'] = os.environ.get('TRITON_CACHE_DIR')
    _READY['hardware_cvt'] = nat._ROUND_FP8_HARDWARE_CVT
    _READY['trace_counts'] = dict(trace.get('counts') or {})


def _compile_one(task):
    """编译单个条目；返回 (目录名, 是否命中清单预期, 耗时, 错误)。"""
    import triton                                       # noqa: F401
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import (ASTSource, compile as tcompile,
                                          get_cache_key, get_cache_invalidating_env_vars,
                                          make_backend)

    started = time.perf_counter()
    result = dict(module=task['module'], name=task['name'],
                  cache_dir_name=task['cache_dir_name'],
                  expected=None, matched=None, ok=False, skipped=False,
                  error=None, note=None, attempts=0, seconds=None, who=None,
                  hardware_cvt=_READY.get('hardware_cvt'))
    # 清单是把闭包内定义的内核（``qualname`` 含 ``<locals>``）也记下来的 ——
    # 它们按名字取不回来（如 ``corr_strip_nat_v1._probe_hardware_cvt.<locals>._probe``）。
    # 那不是缺陷：这类内核是**探测用**的，worker 侧本来就禁探硬件，正式臂里也只编一个
    # 极小的内核。所以记为 skipped，不计入失败（否则会误报 FATAL）。
    if '<locals>' in (task.get('qualname') or ''):
        result.update(skipped=True, who='<locals> 闭包内核，按名取不到')
        result['seconds'] = round(time.perf_counter() - started, 3)
        return result

    cache_root = os.environ.get('TRITON_CACHE_DIR')
    last_error = None
    for attempt in (1, 2):
        result['attempts'] = attempt
        try:
            mod = importlib.import_module(task['module'])
            fn = _resolve(mod, task.get('qualname'), task.get('name'))
            backend_name, arch, warp_size = task['target']
            target = GPUTarget(backend_name, arch, warp_size)
            backend = make_backend(target)
            options = dict(task['options'])
            parsed = backend.parse_options(options)
            src = ASTSource(fn, task['signature'], task['constexprs'], task['attrs'])

            raw_key = get_cache_key(src, backend, parsed, get_cache_invalidating_env_vars())
            expected = manifest._b32(hashlib.sha256(raw_key.encode('utf-8')).hexdigest())
            result['expected'] = expected
            result['matched'] = (expected == task['cache_dir_name'])
            if not result['matched']:
                # 逐项留痕，供定位是哪一项漂的（清单侧同样记了这几项）。
                result['who'] = dict(
                    src_hash=src.hash(), src_hash_manifest=task.get('src_hash'),
                    backend_hash=backend.hash(), backend_hash_manifest=task.get('backend_hash'),
                    options_hash=parsed.hash(), options_hash_manifest=task.get('options_hash'),
                    key_len=len(raw_key),
                    env_vars=sorted(get_cache_invalidating_env_vars().items()),
                    fn_cache_key=fn.cache_key,
                    fn_cache_key_manifest=task.get('kernel_cache_key'),
                    target_arch=task['target'][1], options=parsed.__dict__,
                )

            try:
                tcompile(src, target=target, options=options)
                result['ok'] = True
            except BaseException as exc:                    # noqa: BLE001
                # ★ 判据是「产物是否**完整落在清单目录里**」，而不是「本进程里的
                #   CompiledKernel 能否构造完」。正式臂要的只是缓存里的文件。
                #   实测一例：`CompiledKernel.__init__` 读 ``_matmul.ttir`` 时该文件
                #   已被并发清理，但 spv/llir/grp 齐全 —— 那种内核正式臂完全可用，
                #   若判为失败，整条流水线会白等一次串行重编。
                if _is_done(cache_root, task['cache_dir_name']):
                    result['ok'] = True
                    result['note'] = (f'产物已完整落盘；__init__ 期读 asm 抛错（可忽略）: '
                                      f'{type(exc).__name__}: {exc}')
                    break
                raise
            break
        except BaseException as exc:                        # noqa: BLE001
            last_error = traceback.format_exc(limit=6)
            if _is_done(cache_root, task['cache_dir_name']):
                result['ok'] = True
                result['note'] = '重试前发现产物已完整落盘'
                break
    result['error'] = None if result['ok'] else last_error
    result['seconds'] = round(time.perf_counter() - started, 3)
    return result


def _is_done(cache, dir_name):
    """缓存目录是否**可用**：``__grp__*.json`` + 二进制 ``*.spv`` + 元数据 ``*.json`` 齐备。

    判据要对准**运行期真实需要的东西**，而不是「目录里文件多不多」：

    * ``compile()`` 的命中判定是 ``get_group(metadata_filename)`` 能不能给出
      ``metadata_path``；而 ``FileCacheManager.get_group`` 会**过滤掉已不存在的 child**
      （逐项 ``os.path.exists``）。所以缺一个中间 IR（``.ttir`` / ``.llir`` / ``.source``）
      **不影响命中**，那种目录照旧可用 —— 实测 ``343JY5BP…`` 就少了 ``_pack.ttir``，
      四跑照样 164 命中。
    * 真正必需的是元数据 ``*.json``（``CompiledKernel`` 靠它取 signature/num_warps 等）
      与二进制 ``*.spv``。少了这两个才会在正式臂上炸。

    只查「有 grp + 有 spv」会漏掉「连元数据 json 都没落盘」的半成品，所以这里补上。
    """
    if not dir_name:
        return False
    root = Path(cache) / dir_name
    if not root.is_dir():
        return False
    if not list(root.glob('__grp__*.json')):
        return False
    if not list(root.glob('*.spv')):
        return False
    return any(p.name != '__grp__' and p.suffix == '.json' for p in root.glob('*.json'))


def _harvest(shared_cache, worker_root):
    """把各 worker 一次性目录里的产物**只 copy 不删**地搬回共享缓存。

    ``__grp__*.json`` 里存的是**绝对路径**的 ``child_paths``，必须改写成共享缓存下的
    新位置，否则运行时按 grp 取文件会指回 worker 目录。
    """
    import shutil

    shared = Path(shared_cache)
    shared.mkdir(parents=True, exist_ok=True)
    harvested, files, regrp = [], 0, 0
    for wdir in sorted(Path(worker_root).glob('w*')):
        if not wdir.is_dir():
            continue
        for hdir in sorted(wdir.iterdir()):
            if not hdir.is_dir():
                continue
            dst = shared / hdir.name
            dst.mkdir(parents=True, exist_ok=True)
            for src in sorted(hdir.iterdir()):
                if src.name.startswith('tmp.') or not src.is_file():
                    continue
                out = dst / src.name
                shutil.copy2(src, out)
                files += 1
                if src.name.startswith('__grp__') and src.name.endswith('.json'):
                    try:
                        data = json.loads(out.read_text(encoding='utf-8'))
                        child = data.get('child_paths') or {}
                        data['child_paths'] = {
                            k: str(dst / Path(v).name) for k, v in child.items()}
                        out.write_text(json.dumps(data), encoding='utf-8')
                        regrp += 1
                    except BaseException:               # noqa: BLE001
                        pass
            harvested.append(dst.name)
    return harvested, files, regrp


def main():
    parser = argparse.ArgumentParser()
    # 只有 ``--check`` 路径允许裸跑（import-only 自检，不碰清单、不占 CPU 池）
    parser.add_argument('--manifest', type=Path, action='append')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--jobs', type=int, default=16)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--force', action='store_true', help='连已完成的也重编（自检用）')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()

    if args.check:
        print('corr_strip_parcompile --check: OK (import-only)')
        return 0
    if not args.manifest or args.cache is None or args.out is None:
        parser.error('--manifest / --cache / --out are required unless --check is given')

    os.environ['TRITON_CACHE_DIR'] = str(args.cache)    # 子进程继承
    _bootstrap_paths()

    tasks, modes, modules, nat_state = [], set(), set(), {}
    module_dirs = set()
    for path in args.manifest:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        mode = payload.get('mode') or 'off'
        modes.add(mode)
        nat_state.update(payload.get('nat_state') or {})
        module_dirs.update(payload.get('module_dirs') or ())
        for record in manifest.loads(path):
            record['mode'] = mode
            modules.add(record['module'])
            tasks.append(record)

    seen, unique = set(), []
    for task in tasks:                                  # 同目录只编一次
        key = task.get('cache_dir_name') or f"{task['module']}.{task['name']}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(task)

    pending = [t for t in unique if args.force or not _is_done(args.cache, t.get('cache_dir_name'))]
    print(f'[parcompile] 清单 {len(unique)} 条（模式 {"+".join(sorted(modes))}），'
          f'已完成 {len(unique) - len(pending)}，待编 {len(pending)}，jobs={args.jobs}', file=sys.stderr)
    print(f'[parcompile] 涉及模块 {len(modules)} 个，模块目录 {len(module_dirs)} 个',
          file=sys.stderr)
    if args.force:
        print('[parcompile] ⚠ --force：会重编**已存在**的目录，而 Triton 落缓存前要清理同名'
              '目录 —— 只应指向一次性副本，绝不要在共享缓存上用', file=sys.stderr)

    started = time.perf_counter()
    _seal_removedirs()                                  # 父进程也封上，worker 会再封一次
    os.environ['CODEBUDDY_SAFE_DELETE_ENABLED'] = '0'
    # 父进程先把 Intel 后端的中间件（arch_parser.c → DLL）编进共享缓存，
    # 免得 16 个 worker 同时首编同一个 DLL 打架。
    prewarmed = False
    for task in unique:
        if not task.get('target'):
            continue
        try:
            from triton.backends.compiler import GPUTarget
            from triton.compiler.compiler import make_backend
            b_name, arch, warp = task['target']
            make_backend(GPUTarget(b_name, arch, warp))
            prewarmed = True
            print(f'[parcompile] 后端中间件已预热（{b_name}）', file=sys.stderr)
        except BaseException as exc:                    # noqa: BLE001
            print(f'[parcompile] 预热后端失败（不致命，worker 会各自编）: {exc!r}', file=sys.stderr)
        break

    results = []
    # 一个 mode 一个进程池：`nat.install(mode)` 改写的是模块全局名，同一池里只能是一种。
    for mode in sorted(modes):
        group = [t for t in pending if t['mode'] == mode]
        if not group:
            continue
        print(f'[parcompile] mode={mode} 待编 {len(group)}', file=sys.stderr)
        try:
            with ProcessPoolExecutor(max_workers=args.jobs,
                                     initializer=_init_worker,
                                     initargs=(sorted(modules), mode, nat_state,
                                               sorted(module_dirs))) as pool:
                futures = [pool.submit(_compile_one, task) for task in group]
                for index, future in enumerate(as_completed(futures), 1):
                    try:
                        outcome = future.result()
                    except BaseException as exc:            # noqa: BLE001
                        outcome = dict(module='<worker>', name='<died>', cache_dir_name=None,
                                       matched=None, ok=False, seconds=None,
                                       error=f'worker 异常终止: {exc!r}')
                    results.append(outcome)
                    if outcome.get('skipped'):
                        flag = '-- '
                    elif outcome['ok'] and outcome['matched']:
                        flag = 'OK '
                    else:
                        flag = '!! '
                    if index % 20 == 0 or index == len(group) or flag == '!! ':
                        print(f'[parcompile] {flag}{index}/{len(group)} '
                              f'{outcome["module"]}.{outcome["name"]} {outcome["seconds"]}s',
                              file=sys.stderr)
        except BaseException as exc:                        # noqa: BLE001
            print(f'[parcompile] FATAL: mode={mode} 的进程池崩了: {exc!r}', file=sys.stderr)
    wall = round(time.perf_counter() - started, 2)

    ok = [r for r in results if r['ok']]
    skipped = [r for r in results if r.get('skipped')]
    noted = [r for r in results if r.get('note')]
    mismatched = [r for r in results if r['ok'] and r['matched'] is False]
    failed = [r for r in results if not r['ok'] and not r.get('skipped')]
    report = dict(
        kind='corr-strip-parallel-compile',
        cache=str(args.cache), jobs=args.jobs, modes=sorted(modes),
        manifests=[str(p) for p in args.manifest],
        module_dirs=sorted(module_dirs),
        total=len(unique), pending=len(pending), already_done=len(unique) - len(pending),
        compiled_ok=len(ok), hash_mismatch=len(mismatched), failed=len(failed),
        skipped=len(skipped), noted=len(noted),
        harvested=0, harvested_files=0, regrp=0,
        landed_after_harvest=sum(1 for r in results
                                 if r.get('cache_dir_name') and _is_done(args.cache, r['cache_dir_name'])),
        prewarmed_backend=prewarmed,
        wall_seconds=wall,
        nat_state=nat_state,
        worker=_READY,
        mismatched=mismatched, failures=failed, skipped_items=skipped,
        noted_items=noted, results=results,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')

    print(json.dumps({k: report[k] for k in
                      ('total', 'pending', 'already_done', 'compiled_ok',
                       'hash_mismatch', 'failed', 'skipped', 'noted', 'wall_seconds')},
                     ensure_ascii=False, indent=2))
    if failed or mismatched:
        print(f'[parcompile] FATAL: {len(failed)} 个编译失败、{len(mismatched)} 个 hash 不匹配'
              f' —— 正式臂会退化成串行重编，必须先修', file=sys.stderr)
        return 2
    if skipped:
        print(f'[parcompile] 注：{len(skipped)} 条为 <locals> 闭包内核（探测用），'
              f'不在本遍编译范围；正式臂首用时会各自编一次（极小）', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
