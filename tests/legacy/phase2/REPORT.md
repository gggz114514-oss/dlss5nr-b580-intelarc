Runner4 私有 CPU 候选已完成。目标为 Main PHASE2-r6.json，SHA 360757955fc7cd5d93d639a4c66e925db8e6d6fb9b211c7750e5597f21a92212，258 source/103 arms。复制的 runner3 READY 为 45b511eb8d53a272a1645f70b630077a7690131f8e616dcd4674e04b8a054cb3；live 文件保持原样。

actual API 是 metrics.compare_pair/compare_frames。phase2_runner 的同臂复测与 baseline comparison 继续使用该入口，完整 cold/warm 报告、manifest/control/sequence、byte preserving、capture/provider/resource、cache、owner/retirement 和 fail-stop 门保持原实现。metrics 包装层始终启动明确固定的 CPU bundle 子进程；即使父进程也是该 exe，也不会内联导入 NumPy。

CPU bundle python.exe SHA 为 372c2eae555b344520bf147be0096e009069aeca4e7f78d6aecea6d53158056a，NumPy 2.3.5。已有向量化 kernel/worker/metadata 五项逐字节复用 stage1；G: GPU runtime 常量、GPU child、GPU 环境默认保持原样。父 runner 未导入 Torch/Trition/NumPy；GPU child 没有新增比较模块依赖。

新增 process_policy 只允许 READY 中精确 CPU worker SHA、固定 exe 路径/SHA、完整 -I -X utf8 -B CLI。缺失/不同 exe、其他模式/附加参数、脚本 hash 漂移仍 fail closed；既有 unknown Python、GPU 作业和 Main stable PID 授权规则保留。

46 项 runner3 回归、r6 actual source/registry/fixture CPU admission、13 项新集成测试通过。r4 回归会留下旧模块绑定，曾使 r6 admission 正确触发 Foreign module binding；私有 cpu_checks 将回归放入独立 CPU 进程，实际 r6 admission 冷加载，原 import_file 守卫未改。失败 attempt01 留在 D: 证据中。

真实 r4 两帧 identity 为 0.234 秒，r4 对有差异 PHASE1 两帧为 0.324 秒；每项完整报告与已验收的冻结报告完全一致，浮点为 0 ULP、changed_values/pixels 精确相等。实际 Main comparison 的 baseline13 self cold+warm 完整报告与原实现一致，用时 19.805 → 0.918 秒。这些时间都是 CPU comparison 开销，NR gain/timing 字段保持原值。

真实坏 history raw SHA 经 CPU 子进程返回原错误，归类 NUMERIC_FAILURE；该子进程已退出、NPY 原字节保留。实际 CPU 超时也 kill/wait 自己的子进程；原 fake-child 串行失败审计验证失败后无新 launch/cleanup。未执行 GPU，历史失败 baseline 仍未获 GPU qualification。

只写 runner4-candidate 和对应 D: 小型结果，不修改 metrics-vectorized 原 READY/stage1、live runner/READY、r4/r5/r6/integrated 或 G。READY 提供 source_files SHA、不可变 stage1、CPU 报告和接口说明。候选 GPU HOLD；Main 审核最终 manifest 后自行冻结/资格测试，可沿用显式 r7 archive/SHA。
