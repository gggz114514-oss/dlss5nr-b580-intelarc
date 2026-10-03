# 局部 FP8/INT8 投影 v1：编译失败记录（2026-09-10）

用户询问进展后，主助手读取 Luna 完成交接，独立核对结果、日志、lease 三个
哈希、767 项 sources 和 269 项 exact_gate。进程 rc=1，完成案例数和测速轮数
均为 0；没有速度、精度或迁移通过结论。默认仍是已验证的 Stackv4。

失败发生在第一个 non_fp8_fallback_control 的 debug 内核预编译。INT8 分支
和 FP16 分支都给局部变量 bv 赋值，Triton 在分支合并时要求同名变量类型相同，
报 Mismatched type for bv，分别是 int8[32,32] 与 fp16[32,32]。
这是主助手编写原型时的类型合并错误；未进入测速，也不是实测数值差异。
Luna 只记录并上报，没有修改环境、代码或重跑实验。

v1 的三个执行文件及失败产物保留。v2 将权重局部变量分别命名为 weight_int8
和 weight_fp16；AST 还原这两个名称后与 v1 完全相同，数学操作、tile、寄存器
检查和测试条件不变。v2 使用独立 runner、CMD 和输出目录；编译及运行仍待验证。

结果：D:/Codex-NR-Experiments/nr-b580/reference/experimental/fp8-partial-int8-projection-v1/validation.json

SHA256 ca43d455513d95b7a90646de79cc1e480845a2a9c5a020cd2438555d5164aa60。

日志 SHA256 679e00ca77a7a9c35fdfdf493340e6a81fd41a167a2f130d07afadc9e3594021。

Lease SHA256 fbd83723ce878cd7d02eec7857d23c71fac052755dce9b044d7d7e1a63c5d383。
