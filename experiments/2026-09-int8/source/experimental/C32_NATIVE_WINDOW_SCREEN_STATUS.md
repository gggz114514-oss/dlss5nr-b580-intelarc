# C32 整窗口融合重新筛选（2026-09-10）

不采用。使用已选 ShortFP8 栈、原生 half cubic 和正常 Triton 3.8 编译流程，
重新测试 encoder.0.0 的 160×160×32 完整窗口融合。4-warp 两种配置在发射前
因 spill=3264 被拒绝；8-warp 两种配置零溢出且完整字节相同，但均比基线慢。

| 静态完整模块 | 中位数 ms |
| --- | ---: |
| 当前 ShortFP8 基线 | 0.120996667 |
| 整窗口 8 warps / 1 stage | 0.124203333 |
| 整窗口 8 warps / 2 stages | 0.125413333 |

候选相对基线分别慢约 2.65% 和 3.65%。Luna 最初总结的“提升2.58%/3.52%”
方向有歧义，那是基线相对候选更快的百分比；主助手核对原始数值后明确上述方向。
这不是完整 NR 测速，不能和历史 NR11.56ms 拼接计算整网收益。

7轮交替顺序、每轮30次图重放；原输入与全零输入完整输出一致，持有输出位于共享
图池之外，输入恢复校验通过。基线保留 FP8 数据流消除，fixture 输入的 FP8 域
由全部 half 位模式的既有舍入表逐元素验证，只用于该 fixture，不声称通用值域合同。
因首模块未获得至少5%收益，未继续余下6个模块或完整NR/长视频测试。

v1批处理在进入Python前报输入行/语法错误；v2只对实验子进程设置短PATH后启动成功。
未改系统环境，未启动关机后尚未运行的ComfyUI。batch根因未进一步证明。

结果由 Luna max 监控并确认returncode=0、passed=true、进程结束、465个source哈希一致。
主助手在用户通知“继续”后读取交接并作上述不采用决定；没有轮询测试。
运行数据：D:/Codex-NR-Experiments/nr-b580/reference/experimental/c32-window-native-v2。
validation.json SHA256：0c7fb2aab83f449ae4c58a90faeee47803233f089e173b0a58eeddda3b520c03。
日志与lease、Luna交接在相邻目录。已执行脚本保留原样。

已选栈仍为 nr256_selected_stack_v3.Stack（ShortFP8），精确后端和公开仓库未改动。
