# C512 有效查询与紧凑投影状态

2026-09-10：完整c512-quad-full-v1通过，rc0。243帧连续人脸、128真实传播边界、
16组控制、10组eager/活输入和104管线记录均满足已接受floor16逐字节标准。
新内核零spill；原1011常量和CPU保护保留。Luna审核1189文件；主助手已认证
主结果/日志/lease、关键源文件、结果来源及精确后端保护清单。

本体8.199730→8.256865 ms（+0.697%，3/8轮较快）；时序管线
11.590577→11.610925 ms（+0.176%，2/4轮较快）。未测出提速，不提升默认。
小差值有波动，不能解释成确定退化。结果及main-decision-v1.json在：
D:/Codex-NR-Experiments/nr-b580/reference/results/c512-quad-full-v1/。

分段诊断c512-quad-parts-v1已通过，Luna审核896文件，主助手认证完成证据。
全部16块的attention 0.150928→0.142472 ms（省5.60%，11/12轮较快），
projection 0.187361→0.152413 ms（省18.65%，12/12），实际组合
0.370520→0.321798 ms（省0.048722 ms、13.15%，12/12）。全部边界逐字节一致。
局部两项都变快，本轮不支持“投影抵消收益”假设；尚未证明整帧变快，不提升默认。
分段中位数不可相加，隔离缓存/依赖不同于整模型。完成决策归档在
D:/Codex-NR-Experiments/nr-b580/reference/results/c512-quad-parts-v1/main-decision-v1.json。

c512-upstream-parts-v1已完成并通过固定审核V2，修正后的启动rc0，Luna审核908文件。
主助手已核对主结果/日志/lease/交接及源码来源和精确保护清单。
真实16块、12轮×32重放：FFN 0.561777 ms，QKV 0.572866 ms，
实际FFN→QKV连续段1.153873 ms。输出逐字节一致，10二进制匹配且零spill，
3组活输入检查通过，1011常量及历史/seed/静态输入输出不变。
本轮是当前成本诊断，没有新计算规则或新增提速。详细解释及下一步见
C512_UPSTREAM_RESULT_V1.md。完整实验设计见C512_UPSTREAM_PARTS_DESIGN_V1.md。

QKV直接归一化/写窗口的融合候选已完成，Luna审核913文件，主助手认证完成证据。
真实16块及zero/negated/重放字节检查通过，全部零spill，但同轮原版0.542281 ms，
direct32为0.652077 ms（慢20.25%，1/12轮较快），direct16为0.609838 ms
（慢12.46%，0/12轮较快）。不采用、不做完整集成。BM32报告256寄存器；N32使
工作组增加，各因素因果贡献未单独定位，不能把零spill当作速度保证。
结果及main-decision-v1.json归档在results/c512-qkv-direct-v1/，实现/设计冻结。
本轮源码提交a00afb7603e86e8500b798f04563cbb8a1d9bbfe；主助手通过GPU lease启动
nr-c512-qkv-direct-v1（900秒上限），初始10秒返回running session69594。
已交现有Luna/max监控和固定审核，主助手未轮询后续运行状态。日志/结果/交接位于
results/c512-qkv-direct-v1.log、c512-qkv-direct-v1/、c512-qkv-direct-v1-monitor-luna-v1/。

依用户此前约定，下一轮c512-qkv-int8-v1已实现、完成CPU预检并实际启动：保留当前
QKV的FP8边界、half归一化和窗口pack，内部改用已有W8A8矩阵核，允许数值变化。
输入按行量化计入每次重放；权重按列只打包一次。三个固定块配置先查spill。
全量输入量化对照CPU，每块48个整数点积抽查，误差仅统计144有效位置，padding
单独要求字节一致。质量尚未接受，若速度有价值再做完整视频并交用户审核。
CPU检查已验证144组调用参数/数据衔接、整数抽查能检出错误、四种偏移的有效区域
统计与padding错误检测。设计和前轮结论见C512_QKV_INT8_DESIGN_V1.md。
源码提交417d234be4c31d7bf297280b3bb7c30dc0c57a88；主助手通过GPU lease启动
nr-c512-qkv-int8-v1（900秒上限），初始10秒返回running session91519。
现有Luna/max接手监控和固定审核；主助手未轮询。结果/日志/交接分别位于
results/c512-qkv-int8-v1/、c512-qkv-int8-v1.log、c512-qkv-int8-v1-monitor-luna-v1/。
源码提交8f9bb7d5a2417feb85e8383babecb35626c6e582；本次固定审核V2提交
6479dc48b3e2f55214d6d47ee3cd49be57f6ec43。Luna/max已完成监控及固定审核。
首次启动因主助手传入cmd.exe路径格式错误，在Python模型执行前rc1；已用纯CPU
小脚本定位并验证Windows反斜杠路径修复，旧日志及失败决策冻结保留。修正后启动
初始10秒返回running，session64667；主助手未轮询后续状态。
本次日志c512-upstream-parts-v1.launch2.log，审核manifest/handoff位于
results/c512-upstream-parts-v1-monitor-luna-v2/，勿与第一次失败日志混用。

V1溢出失败、V2局部通过及完整版本的已执行源文件和结果均冻结保留。
默认栈、精确后端不变，没有发布或重新量化。主助手准备/修复/实际启动；
Luna/max仅监控并执行固定审核，异常立即报告，不自行修复或重试。


2026-09-10 最新：INT8 QKV V1完成通过，Luna和主助手均认证921个文件。
同轮原版1.749208 ms，INT8 16×64为1.335444 ms，省23.65%，12/12轮较快；
32×64省12.11%，16×32慢4.60%。8个候选二进制零spill，各INT8配置彼此字节一致。
原版此前约0.54 ms，本轮明显升高。用户随后明确说明当时在运行其他3D软件；
这轮属于已知并发负载结果，不能当作空闲性能或整模型节省0.414 ms的证据。
数值实现检查通过；QKV相对原版的量化差异不是外部3D负载造成的，画质仍待审核。
完成决策及user-load-context-v1.json留在results/c512-qkv-int8-v1/。

用户已确认“已经停止，可以复测”。V2保持原GPU算术和全部输入/输出证据，
先重复12×32，再每样本预热64+计时128，最后重复12×32，完整记录三个计时块。
CPU预检通过：3个Python AST/2个ASCII启动器；虚拟时钟证实预热与CPU检查不计入、
GPU完成等待计入，6类刻意损坏均检出。设计C512_QKV_INT8_TIMING_DESIGN_V2.md。
源码提交696a3eeee0b002a3d2df4c227aa9c78570825c9e；主助手已通过GPU lease实际启动
nr-c512-qkv-int8-v2（900秒上限），初始10秒返回running session2273，主助手未轮询。
现有Luna/max监控和固定审核，提交01a08bf3-54b9-7091-b1b8-cfc996276734。
结果/日志/manifest/handoff：results/c512-qkv-int8-v2/、c512-qkv-int8-v2.log、
c512-qkv-int8-v2-monitor-luna-v1/。默认和精确分支不变；未集成新候选或接受画质。


2026-09-10 最新：空闲复测c512-qkv-int8-v2完成通过，Luna和主助手均认证929文件。
原协议原版0.578206→INT8 16×64 0.441378 ms（省23.66%）；预热长批次
0.575192→0.442066（省23.14%）；末尾原协议0.590291→0.445163（省24.59%）。
三组36轮全部较快。全部输入/权重/二进制/误差/CPU抽查/输出匹配V1。
局部收益稳定，但仅省0.13～0.15 ms，不推导整模型快24%。完成决策归档在
results/c512-qkv-int8-v2/main-decision-v1.json；详细迁移思路见C512_QKV_INT8_FULL_DESIGN_V1.md。

完整候选c512-qkv-int8-full-v1已实现并实际启动。原floor16/native-query栈只替换
QKV，复用筛选后的16×64三核，6二进制哈希/零spill在派发前核对。新增32权重/
尺度注册入图常量，总1043项，原1011不变，保留CPU保护和每次调用独立中间缓冲。
测试包含真实传播CPU校验、双路线独立历史、完整本体与1080p常驻完整调用计时，
以及243帧10.125秒完整人脸三列视频。画质待用户审核，不提升默认/精确路线。
CPU预检通过：48次完整调用与冻结screen逐参数/布局一致，真实内核签名绑定；
注册引用/版本变化、二进制不符、spill和作用域异常恢复均已检验。无CPU阶段GPU执行。
源码8b411505b7a33f92cfa1b9eaa28bbc21136bdf78；主助手通过GPU lease实际启动
nr-c512-qkv-int8-full-v1（1800秒上限），初始10秒返回running session85061。
现有Luna/max监控与固定审核，提交01a08c01-eaaa-7953-8826-8c64035be798；主助手未轮询。
结果/日志/manifest/handoff在results/c512-qkv-int8-full-v1/、c512-qkv-int8-full-v1.log、
c512-qkv-int8-full-v1-monitor-luna-v1/。数据D，源码E，无原始张量/权重转储。
