Main 从 READY.frozen_snapshot 合入 source_files 的相对文件并验证 SHA。该目录是 runner3 的完整私有副本，比较 backend 固定为现有 bundle CPU NumPy subprocess；没有更换默认 G: GPU runtime、GPU 模型来源或 Torch 环境。当前 target_manifest 是 r6 archive；原 configure 支持 Main 同 revision archive/显式 SHA，后续 r7 可用新 CPU 报告重新冻结，无须写 math source。

| 文件 | 实际作用 |
| --- | --- |
| metrics.py | 保留原 header API，导出 compare_pair/compare_frames 的 CPU subprocess 接口 |
| metrics_subprocess.py | 固定 bundle exe/SHA、五项 vendor 身份、隔离 CLI、单 CPU 线程、JSON 协议、退出/超时归还 |
| cpu_comparison/* | 原 stage1 四个源码加 CPU_RUNTIME.json 的精确字节副本；worker 内才导入 NumPy |
| metrics_streaming_reference.py | 原 runner3 metrics 精确源码，仅 CPU 对照/header；生产比较无 fallback |
| process_policy.py | 原规则保留，单独增加 exact_bundled_CPU_comparison 分支，必须匹配脚本、exe、SHA、全部 CLI |
| runner_common.py | 在私有嵌套位置找到实际 IMP；候选写域限私有 D，既有 r4 结果从 EVIDENCE_DATA 只读 |
| cpu_checks.py | 46 项回归在固定 CPU bundle 独立进程，目标准入在冷 namespace，原来源守卫保留 |
| phase2_runner.py | 两个比较消费函数原样；仅 CLI 增加 candidate_GPU_HOLD，未资格候选禁止 GPU opt-in |

调用链为 phase2_runner.measure_arm/comparison → metrics.compare_frames → metrics_subprocess._compare → 明确的 `C:/Users/REFERENCE_USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe -I -X utf8 -B <candidate>/cpu_comparison/metrics_cpu_worker.py` → 原 stage1 compare_frames。父进程全部 stdlib；即使父进程是同一个 bundle exe，仍启动不同 PID 的 CPU child。GPU child/worker/support/evidence/cache/retirement 文件保持 runner3 SHA，未增加 NumPy/CPU comparison 导入。

runtime_contract 固定 exe SHA `372c2eae555b344520bf147be0096e009069aeca4e7f78d6aecea6d53158056a`，CPU_RUNTIME SHA `52bf530a7ad6ecc78cc473a69f6bc6912af17a2aa33fe943ebaa5fa3cc4e9358`；后者固定四个源码及 NumPy 路径/入口 SHA/2.3.5 版本。worker SHA 为 `84f0677353b055915e4bc5c1197d4f29121707fce1fa5d107428e52b382b8106`。进程名单只有这一精确 worker 的新例外，普通 unknown Python 的规则未扩大。

NPY header/schema/dtype/shape、file/raw SHA、finite、截断/trailing、signed zero、逐帧 frame_id/reset/RGB/motion/seed 和完整 per_frame/aggregate 字段由固定 kernel 真实执行，误差是保持原加法顺序的 float64。NR timing/gain 仍来自原 comparison/pair_timing 的 GPU events；CPU LAST_CALL wall/命令/PID/退出记录只在本次 CPU 结果中单列。

比较失败保留原子进程错误及异常名于 LAST_CALL，向现有 fp.guard 抛 RuntimeError，成为 NUMERIC_FAILURE；不继续下一 arm、不把失败改成 skip、不触发清理。子进程 communicate 完成或超时 kill/wait 后才返回/抛错，NPY 全程只读。完整报告的 schema 不加入 CPU 元数据；因此 Main 原字段消费保持相同。

验证命令：普通 Python `-B cpu_checks.py --manifest <Main archive> --manifest-sha256 <SHA> --report <本私有D新JSON>`，以及 `-B runner4_cpu_tests.py --report <本私有D新JSON>`。46 项回归、13 项新增检查及实际 Main baseline13 self 对照均在 CPU 运行。历史 baseline completed=false，报告只用于格式/数值计算，不产生新 GPU qualification。

候选 READY.candidate_GPU_HOLD=true，Main 最终审核并运行新 manifest 的 CPU admission 后再冻结新 READY/授权 Luna。Base runner3 的 main_cpu_script_pins 保持原精确名单；部署比较模块时 READY.files 必须包含五项 cpu_comparison 资产及新 wrapper/policy SHA。SCRIPT_DIR 在私有 reviews 树时输出限对应 D: runner4-candidate；Main 放回官方 luna/phase2 时 DATA 自动保持官方原路径。EVIDENCE_DATA 和 PHASE1 seed 始终只读原位置。
