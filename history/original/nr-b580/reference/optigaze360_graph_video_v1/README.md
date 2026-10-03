# 360p 全帧图重放视频实验

只在本目录开发；不改产品、精确后端、私有 GPU Block/DIS。冻结输入仍是同一段 864×480、24 fps、243 帧人脸视频和已存的逐帧 Block 运动向量。360p 为 648×360，内部补边 768×384；这个几何补边是实验推定值，不是 4060 原生观测契约。缩图、运动向量缩放、NR 权重、原图引导重建沿用 `../optigaze360_v1/`。仅 NR 的 `_forward_front` 主体使用现有 XPU 图重放；输入准备、运动/历史管理及输出重建照常执行。首帧和稳态的历史状态由模型按捕获帧的 `reset` 标志管理。

## 两个独立模式

从本目录运行（launcher 调用项目固定 Python，配置 14 个编译线程，缓存/临时文件在 D 盘）：

```powershell
python launch.py correctness --motion fractional
python launch.py speed-video --motion fractional
```

`correctness` 依次以 eager、图重放各跑完整 243 帧；每帧分别计算**模型 FP32 输出**的 SHA-256，逐帧比较。它要双算，不能用其时间作性能结论。完整通过后写 `D:/Codex-NR-Experiments/nr-b580/optigaze360-graph-video-v1/fractional/correctness.json`。失败或中断不会留下新的通过标志。

`speed-video` 必须先找到相同输入、运动模式、代码指纹且逐帧 243 帧通过的正确性报告；只执行图重放，随后把本次 243 帧模型 FP32 SHA 再和 eager SHA 逐帧核对。任何一帧不同即停止，不交付新视频或速度结论。视频仅采用**原图引导重建**这一路，没有把普通放大与帧间运动对齐两条实验分支也算进来。将原音频流直接复制进 MP4；完工后完整解码视频，要求恰好 243 帧，并比对原视频与成片的解码 PCM SHA。通过才把暂存文件更名为 `guide360.mp4` 并写 `speed-video.json`。视频是 CRF 13 的审核编码，模型 SHA 对拍取自编码前 FP32 张量。

计时和现有 480p 原尺寸报告同口径：每帧输入已上传 XPU，GPU 同步后的墙钟时间；`prepare` 为缩图与可选运动取整，`model` 为图重放 NR，`reconstruct` 为残差与原图引导重建，`total` 是三段之和。去掉前四帧，取后 239 帧中位数；编译、上传、回读、SHA、视频编解码和音频校验不计入。480p 基线读取已验证的完整同源报告；速度报告同时列相对耗时变化和吞吐变化。图首次建立如落在前四帧内，不计入稳态值，但可从 `graph_entries` 查看构图时间。

可选分支：

```powershell
python launch.py correctness --motion rounded
python launch.py speed-video --motion rounded
```

默认 `fractional` 是现有 360p 分数运动向量。`rounded` 在缩放 ×0.75 后用 `torch.round()` 取整，尝试恢复整数位移快路径；**它会改变帧间语义和画面，尚未经肉眼审核，只能标为候选**。两种模式分别存放在 D 盘 `fractional/` 与 `rounded/`，正确性报告不能交叉复用。`rounded` 的 eager 与图重放字节一致不代表它与 `fractional` 或 4060 原版一致。

## 报告结构与验收

两个 JSON 的完整字段及类型见 [report_schema.json](report_schema.json)。`correctness.json` 至少要求 `passed=true`、两组各 243 个 FP32 SHA 且逐项相同、`context.motion` 正确。`speed-video.json` 至少要求 `passed=true`、同一 context、243 个本次模型 SHA 与 eager 参考相同、`decoded_video_frames=243`、输入与输出音频 PCM SHA 相同、四段耗时各有 239 个稳态样本。`context.code_sha256` 固定本目录 runner/launcher、上游 360p 输入与适配、会话、图重放及 NR 后端源；代码变更后必须重做正确性模式。`quality_approved=false` 对 `rounded` 仅表示尚未人工审核，不是画质不合格结论。

静态检查（不初始化 GPU、不启动测试）：

```powershell
python -B -c "import ast,json,pathlib; p=pathlib.Path('.'); [ast.parse((p/f).read_text(encoding='utf-8')) for f in ('launch.py','run_video.py')]; json.loads((p/'report_schema.json').read_text(encoding='utf-8')); print('static syntax/schema OK')"
```

本任务只交付可运行代码与静态核对；完整正确性、速度、帧数、音频和视频画质均待 Luna 在 B580 上执行与人工审核。历史上 240p 12 帧图重放相对 eager 约 49.4→26.5 ms 的结果只能说明那一几何的探针已通过，不能当成本次 360p 全链实测。
