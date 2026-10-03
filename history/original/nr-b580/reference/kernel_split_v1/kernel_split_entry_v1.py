"""把合并的内核名 `_kernel` 拆回它来自的模块 —— 与 chrome trace 的设备时长 zip。

为什么需要这一轮
----------------
当前线的设备侧排序**全部**建立在 `prof.key_averages()` 的一张表上，而那张表里最大的一行
`_kernel` **是合并名**（profiler 只报 Triton 内核的**裸 Python 函数名**）：
不同模块的同名内核被算成一行。`nowsplit2_v1/RESULT.md` §5 尾注自己写了「**本轮仍未拆**」，
`where_to_cut_v1/ASSESSMENT.md` §3 把它列为必须**先测量**的第 1 项。

为什么不能事后从代码推断
------------------------
`nr-b580-int8/backend` 与两颗产品树里 `def _kernel` 至少两处（`triton_fp8.py:34`、
`triton_attention_exp.py:9`），而 `pre_mlp.py:42` 把 `triton_fp8._kernel` 别名成 `fp8_kernel`
再启动 ⇒ **按源码里的调用点数是数不出来的**（同 `l3_1_unwrap_v1` 的教训：12 个
`chunk_k=16` 调用点里只有 2 个真的在快速线上跑）。**只能实测。**

做法（**产品源码 0 改动**）
---------------------------
沿用本项目的「**源码补丁 + exec**」模式（同 `nowsplit2_v1/now_attrib_entry_v1.py`），
在 `reference/materials_v1/materials_entry_v1.py` 上只加四样：

  1. `KernelNamer`：包 `JITFunction.run`，在**启动时刻**按顺序追加
     `f'{inner.fn.__module__}.{inner.fn.__name__}'`（+ `grid`）。**只记顺序，不读时钟。**
  2. `export_trace`：把 chrome trace 导出到 `KERNEL_SPLIT_TRACE`。
  3. `summarise`：与 `nowsplit2` 逐字相同的设备侧汇总（含**全部**内核行，不留 top-N）。
  4. `profiled_frame`：最后一帧同时装 profiler 与名器；其余帧一律不装。

**方法出处**：`fullsize_devsplit2_v1/rows_devsplit2_v1.py`（82 ms 时代已解过同一问题）。
本文件是它在**当前线**上的挂载点，zip 逻辑见 `kernel_split_zip_v1.py`。

★ 两点必须先讲清（都在 PLAN.md §3.2/§3.3）

* **与产品「层 1」的关系**：`NR_LAUNCH_FASTPATH=run,...` 的 `run` 刀替换的正是
  `JITFunction.run`（`nr-b580-int8/experimental/launch_fastpath_v1.py:155-165`），
  但它**不绕过 `run` 本身**、只把准备换成查表 ⇒ 本名器在它之后安装、成为最外层包装。
  **这一条不靠推理，靠 zip 的 C1/C2 自检**。
* **比 `src` 反查更强**：82 ms 那份表用 `CompiledKernel.launch_metadata` 的 `kernel.src`
  反查名字，而 Triton 的编译缓存会让两个**同形** `JITFunction` 指向同一个 `CompiledKernel`
  ⇒ 有 **0.65%（8 次/帧）** 的归因歧义（`fullsize_rows_v1/FRAME_BUDGET_REPORT.md:286`）。
  本类记的是**真实被调用的那个 Python 函数**，没有这个歧义。

⚠️ 这是**诊断**，不是证据跑：不产出正确性产物、不参与任何门槛，
绝对毫秒**不作加速比证据**（技能：诊断只贡献占比与计数）。
⚠️ 本机剖析遍可能在解释器拆卸阶段触发 Windows fail-fast（0xC0000409），
报告在 `finally` 里已写出，驱动脚本据此判成败（同 `nowsplit2` 的注记）。
"""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE.parent
SOURCE = REF / 'materials_v1' / 'materials_entry_v1.py'

# ---------------------------------------------------------------- 补丁锚点
# 每个锚点断言在源文件里**恰好出现一次**；对不上就停，而不是静默补了个空。

# 1) 名器 + trace 导出 + 设备侧汇总。整块插在 `def main` 之前。
OLD_MAIN = """def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
"""

NEW_MAIN = '''class KernelNamer:
    """按启动顺序记录每次**真实** Triton 启动的 `<module>.<function>`（+ grid）。

    为什么必须按启动顺序记
    ----------------------
    `prof.key_averages()` 把 Triton 内核**按裸 Python 函数名**分组，于是不同模块的同名
    内核（`nr_backend.triton_fp8._kernel` / `nr_backend.triton_attention_exp._kernel` …）
    被合并成一行。chrome trace 里的内核事件**不带模块信息**：ATen 事件有 `External id`
    能链回 `cpu_op`，**Triton 事件没有任何可用的 `External id`**（它带的 correlation
    指向一个无关的 cpu_op）⇒ **trace 里没有从内核事件回到 Python 模块的路**。

    ⇒ 模块名只能在**启动时刻**记。顺序对齐的依据：全部内核事件 `pid=0 / tid=64`
    （**单流有序队列**）⇒ 设备执行顺序 = 启动顺序 ⇒ **第 k 个内核事件 = 第 k 次记录**。
    这个前提由 zip 的 C1/C2/C3 三条自检复核，不成立就不出表。

    与产品层 1 的关系（**已核实的漏记风险**）
    ----------------------------------------
    层 1 的 `run` 刀（`nr-b580-int8/experimental/launch_fastpath_v1.py:155-165`）替换的
    **就是** `JITFunction.run`，但它只是把准备阶段换成查表、**并不绕过 `run` 本身**
    ⇒ 本类在它**之后**安装，成为最外层包装，**看得见每一次启动**。

    与 `src` 反查相比更强的一点
    ---------------------------
    82 ms 那份表用 `CompiledKernel.launch_metadata` 的 `kernel.src` 反查名字，而 Triton
    的编译缓存会让两个**同形** `JITFunction` 指向同一个 `CompiledKernel` ⇒ 归属随首次
    编译顺序变（实测 0.65% = 8 次/帧，`fullsize_rows_v1/FRAME_BUDGET_REPORT.md:286`）。
    本类记的是**真实被调用的那个 Python 函数**（`inner.fn.__module__`），没有这个歧义。

    代价：每次启动多一次属性读 + 一次 append。**不读时钟**，不污染设备侧读数。
    """

    def __init__(self):
        self.order = []
        self.grids = []
        self._jit_class = None
        self._run = None
        self._installed = False
        self._fastpath_before = None
        self._fastpath_after = None

    def install(self):
        from triton.runtime.jit import JITFunction
        namer = self
        self._jit_class = JITFunction
        self._run = JITFunction.run
        self._fastpath_before = self.fastpath_stats()

        def run(inner, *args, **kwargs):
            if not kwargs.get('warmup'):
                # `inner` 是 JITFunction 实例；`inner.fn` 是 Python 函数，
                # 它的 __module__ 是"这个内核来自哪个模块"唯一活着的把手。
                fn = inner.fn
                namer.order.append(fn.__module__ + '.' + fn.__name__)
                namer.grids.append(str(kwargs.get('grid')))
            return namer._run(inner, *args, **kwargs)

        JITFunction.run = run
        self._installed = True
        return self

    def uninstall(self):
        if not self._installed:
            return self
        self._jit_class.run = self._run
        self._fastpath_after = self.fastpath_stats()
        self._installed = False
        return self

    def fastpath_stats(self):
        """层 1 自己的**累计**计数（独立第三源）。拿不到就返回 None。

        ⚠️ 它是**从安装起累计**的（`n4_land_product_report_v1.py` 里那句
        "`run hits 944.7/帧`"就是靠前后取差算出来的）⇒ 直接用会和**单帧**的
        `len(order)` 对不上。所以只在 install/uninstall 各取一次，用差。
        """
        try:
            import launch_fastpath_v1 as fastpath
            instance = getattr(fastpath, '_INSTANCE', None)
            if instance is None:
                return None
            return dict(getattr(instance, 'stats', None) or {})
        except BaseException:
            return None

    def fastpath_delta(self):
        before, after = self._fastpath_before, self._fastpath_after
        if before is None or after is None:
            return None
        keys = set(before) | set(after)
        return dict(before=before, after=after,
                    delta={key: after.get(key, 0) - before.get(key, 0) for key in sorted(keys)})


def export_trace(prof):
    """把 chrome trace 写出来 —— zip 里的**设备时长**只从这一份取。

    profiler 此时**已停**（本函数在 `with` 块之后被调用），导出是 PyTorch 的既有用法。
    """
    path = os.environ.get('KERNEL_SPLIT_TRACE')
    if not path:
        return dict(error='KERNEL_SPLIT_TRACE not set')
    try:
        prof.export_chrome_trace(path)
    except BaseException as error:
        return dict(error=repr(error))
    return dict(path=str(path), bytes=os.path.getsize(path))


def summarise(prof):
    """把被剖析的那一帧拆成设备忙时、内核族构成与 CPU 形状。

    设备侧取 `device_time_total`，求和**精确闭合**到 `gpu_busy_ms`（全部内核，不留 top-N）。
    CPU 侧**只作形状** —— profiler 把这一帧的墙钟抬高一个数量级，
    `self_cpu_time_total` 被放大很多倍，报它只为看构成与次数，绝不是毫秒。
    """
    from torch.autograd import DeviceType
    gpu, cpu = [], []
    gpu_busy_ms = 0.0
    cpu_self_ms = 0.0
    instances = dict(triton=0, aten=0)
    gpu_ms_by_family = dict(triton=0.0, aten=0.0)
    api = {}
    for item in prof.key_averages():
        if item.device_type == DeviceType.XPU:
            gpu_busy_ms += item.device_time_total / 1000.0
            family = 'aten' if item.key.startswith('_ZTS') else 'triton'
            instances[family] += item.count
            gpu_ms_by_family[family] += item.device_time_total / 1000.0
            gpu.append(dict(name=item.key, count=item.count,
                            device_ms=item.device_time_total / 1000.0))
        else:
            cpu_self_ms += item.self_cpu_time_total / 1000.0
            if item.key in ('urEnqueueKernelLaunchWithArgsExp',
                            'zeCommandListAppendLaunchKernelWithArguments',
                            'zeCommandListHostSynchronize',
                            'urEnqueueUSMMemcpy',
                            'zeMemGetAllocProperties'):
                api[item.key] = dict(count=item.count,
                                     self_cpu_ms=item.self_cpu_time_total / 1000.0)
            if item.self_cpu_time_total:
                cpu.append(dict(name=item.key, count=item.count,
                                self_cpu_ms=item.self_cpu_time_total / 1000.0))
    gpu.sort(key=lambda item: item['device_ms'], reverse=True)
    cpu.sort(key=lambda item: item['self_cpu_ms'], reverse=True)
    return dict(gpu_busy_ms=gpu_busy_ms,
                gpu_ms_by_family=gpu_ms_by_family,
                gpu_distinct_kernels=len(gpu),
                gpu_instances=sum(instances.values()),
                gpu_instances_by_family=instances,
                cpu_self_total_ms=cpu_self_ms,
                launch_api=api,
                gpu_kernels=gpu,
                cpu_top=cpu[:40])


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
'''

# 2) 剖析遍：最后一帧同时装 profiler 与名器，并导出 trace。
#    ★ 名器在 `with profiler` **内部**安装与卸载 ⇒ 记录到的每一次启动都落在被追踪的窗口里，
#      不存在"记了但没被 trace"或"被 trace 但没记"的窗口差。
OLD_SELECT = """        selected = capture['frames'] if args.phase == 'full' else capture['frames'][:args.limit]
"""

NEW_SELECT = """        def profiled_frame(row):
            \"\"\"最后一帧装 profiler + 名器。**只取设备侧**：墙钟被 profiler 抬高，不作数。\"\"\"
            assert paths.sha(row['file']) == row['sha256']
            with np.load(row['file'], allow_pickle=False) as handle:
                rgb = torch.from_numpy(handle['rgb'].astype('f4')).to('xpu')
                motion = torch.from_numpy(handle['motion'].astype('f4')).to('xpu')
            before = dict(counts)
            counters.reset()
            torch.xpu.synchronize()
            start = time.perf_counter()
            namer = KernelNamer()
            activity = [torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.XPU]
            with torch.profiler.profile(activities=activity) as prof:
                namer.install()
                try:
                    with rows_scopes.dispatch_guard(VARIANT), \\
                            installed(), torch.inference_mode(), use_arithmetic_backend('triton'):
                        low = stack.model(rgb, motion.float(), reset=row['reset'])
                        color = low.float()
                        torch.xpu.synchronize()
                finally:
                    namer.uninstall()
            finished = time.perf_counter()
            assert counts['c512'] - before['c512'] == 16 and counts['vit'] - before['vit'] == 8, counts
            totals = counters.totals()
            assert totals == expected_counters, (totals, expected_counters)
            out = summarise(prof)
            out['index'] = row['index']
            out['wall_ms_inflated'] = (finished - start) * 1000.0
            out['launch_order'] = namer.order
            out['launch_grids'] = namer.grids
            out['fastpath_stats'] = namer.fastpath_delta()
            out['trace'] = export_trace(prof)
            out['note'] = ('wall and CPU self times are inflated by the profiler; '
                           'only the device track, the kernel mix, the launch order '
                           'and the counts are used')
            return out

        selected = capture['frames'] if args.phase == 'full' else capture['frames'][:args.limit]
"""

# 3) 剖析遍挂在正常帧循环之后，仍在 cache_scope 内（DiskOnly 语义不变）
OLD_CACHEHITS = """            report['cache_hits'] = None if cache is None else cache.hits
"""
NEW_CACHEHITS = """            report['cache_hits'] = None if cache is None else cache.hits

            # ---- 诊断遍：只在最后一帧装 profiler + 名器 -------------------------
            # 帧墙一律取自上面**没有装任何剖析器**的那一遍；
            # 这一遍只贡献**设备侧**（gpu_busy_ms、内核族、全部内核明细）与**启动顺序**。
            profiled = None
            try:
                for row in selected[:-1]:
                    one_frame(row)
                profiled = profiled_frame(selected[-1])
            except BaseException:
                report['profile_error'] = traceback.format_exc()
            report['profiled'] = profiled
"""

# 4) 报告自证：跑的是哪份代码
OLD_ADAPTER = """        report['adapter_file'] = str(Path(__file__).resolve())
"""
NEW_ADAPTER = """        report['adapter_file'] = str(Path(__file__).resolve())
        report['kernel_split'] = KERNEL_SPLIT
        report['launch_fastpath_env'] = os.environ.get('NR_LAUNCH_FASTPATH')
        report['trace_env'] = os.environ.get('KERNEL_SPLIT_TRACE')
"""

PATCHES = (
    ('namer+trace+summarise', OLD_MAIN, NEW_MAIN),
    ('profiled_frame', OLD_SELECT, NEW_SELECT),
    ('profile_pass', OLD_CACHEHITS, NEW_CACHEHITS),
    ('adapter_provenance', OLD_ADAPTER, NEW_ADAPTER),
)


def main():
    text = SOURCE.read_text(encoding='utf-8')
    patched = text
    for label, old, new in PATCHES:
        count = patched.count(old)
        if count != 1:
            raise SystemExit(
                f'patch {label!r}: anchor found {count} times in {SOURCE}; refusing to patch')
        patched = patched.replace(old, new)

    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    patched_sha = hashlib.sha256(patched.encode('utf-8')).hexdigest()
    print('kernel_split: source=%s' % SOURCE, flush=True)
    print('kernel_split: source_sha256=%s' % source_sha, flush=True)
    print('kernel_split: patched_sha256=%s' % patched_sha, flush=True)
    print('kernel_split: patches=%s' % (','.join(label for label, _, _ in PATCHES),), flush=True)

    # exec 成 __main__，让它自己的 argparse 尾巴点火、建自己的 parser。
    # __file__ 指向**真实源路径**，这样它的 sys.path 引导与 rows_paths_v1 导入
    # 与原件运行时逐字一致（同 rows_devsplit2_v1.py）。
    namespace = dict(
        __name__='__main__',
        __file__=str(SOURCE),
        KERNEL_SPLIT=dict(
            kind='diagnostic-kernel-name-resolution',
            wrapper=str(Path(__file__).resolve()),
            wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            source=str(SOURCE),
            source_sha256=source_sha,
            patched_sha256=patched_sha,
            patches=[label for label, _, _ in PATCHES],
            method='launch-order zip: JITFunction.run wrapper records module.function in '
                   'order; chrome trace supplies per-kernel device duration; the queue is '
                   'single and in-order (pid=0/tid=64) so the k-th kernel event is the '
                   'k-th recorded launch (same method as fullsize_devsplit2_v1)',
        ),
    )
    exec(compile(patched, str(SOURCE), 'exec'), namespace)


if __name__ == '__main__':
    main()
