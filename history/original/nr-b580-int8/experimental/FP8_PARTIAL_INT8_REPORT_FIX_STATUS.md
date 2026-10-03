# 局部 INT8 实验报告修复（2026-09-10）

用户指出测试没有启动后，主助手读取 Luna 的 v2 完成交接并核对日志、lease 和
三个执行源文件。v2 实际启动后运行 22.291 秒，保存首个控制案例报告时退出，
rc=1，没有进入正式计时，也没有生成 validation.json。不能据此给出速度或
完整精度结论。前次把命令已发出描述为重测已启动，没有交代后续提前退出。

原因是 difference 中 NumPy count_nonzero 的结果参与比较后产生了 NumPy
布尔标量，标准 json.dumps 拒绝该类型。异常处理中的保存操作再次触发同一
错误，因而连失败报告也未写出。该错误由主助手处理，Luna 没有自行修复重跑。

v3 将计数显式转换成 Python int，并为报告保存显式处理 NumPy bool、整数和
浮点标量；未知对象仍报错，非有限数值不允许写入。报告在初始化时即保存，并
记录后续阶段。日志 timing_started 表示进入测速准备代码，之后还需完成图
捕获和守卫检查；只有实际样本才能证明计时完成。

新 CPU 回归直接从 v3 runner 的 AST 提取实际 difference/json_scalar/save：
相同、不同和正负零输出均可正确写读 JSON；NumPy 标量转换、未知对象及 NaN
拒绝检查通过。未导入 Torch/Triton，也未执行 GPU。复用 v2 内核，计算规则、
测试案例和计时条件不变。

CPU 报告：D:/Codex-NR-Experiments/nr-b580/reference/experimental/fp8-partial-int8-report-cpu-v3/validation.json

SHA256 571c87ba25262f5f497cf154396138210da87a325884d85e27e481ff2ee10d56。

v2 日志 SHA256 2de154dba111e46c7f82d1f5e8f807fead0dbafb76db3ba7e25bf132fbece5f2。

v2 lease SHA256 453383f9040d443c071490420c83dd5e31c334f7c694472a3fac1dbb2c3ae519。

v3 GPU 重跑命令已提交，由 Luna max 监控并核实实际阶段；此记录不宣称其
编译、精度或计时已完成。默认仍是已验证 Stackv4，没有推广局部 INT8 原型。
