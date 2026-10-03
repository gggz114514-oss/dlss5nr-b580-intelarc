# hires_three_way_v1 —— 1080p 赛车素材：精确线 / INT8 / FP16 **三路画面对比**（不含 4060）

日期：2026-09-19　执行：Qoder（接手会话）　卡：Intel Arc B580（11.6 GiB 可见）
触发：用户 2026-09-19 指令 ——「赛车（1080p）不管 4060 那一路，给精确版，fp16，int8 三路对比」。
本轮**只出画面与像素差**，不测速、不判质量：画面过不过由用户看（`HANDOVER.md` §4 的验收口径）。

---

## 1. 结论（三条，第三条留给用户）

1. **三格齐了，而且是同一批输入。** 精确线本轮现跑（13 帧，`D:/hires-three-way-v1/exact/frames/E/`）；
   INT8 / FP16 两格**直接沿用 `D:/hires-rows-v1/itl1/frames/{A,B}/`（速度裁决的正源，未重跑）**，
   并在开跑前把盘上 26 个帧的摘要与 itl1 `validation.json` 逐帧对齐：**13/13 + 13/13 全中**。
2. **数值上 FP16 明显更贴近精确线，方向与 480p 那轮一致。**
   以精确线为参照：PSNR 中位 **E↔B = 49.585 dB**、**E↔A = 42.171 dB** ⇒ FP16 臂离精确线高
   **7.41 dB**；逐帧无一例外（13/13 帧都是 E↔B 更好）。超 1/255 的像素占比中位
   **6.0%（E↔B）vs 17.6%（E↔A）**；高频 RMSE 中位 **0.00849 vs 0.01387**。
   这与 480p 落地轮"ViT 前馈 FP16 相对 4060 锚点 +8.331 dB"（`fp16land_v1/RESULT.md` §7）**同向**。
3. **画面可不可接受 —— 本轮不判，等用户看。** 素材见 §5。

诊断性时间（**不作加速比承诺**，硬约束）：精确线稳态中位 **1935.4 ms/帧**（首帧 3254.2 ms），
itl1 两臂稳态中位 442.6 / 447.4 ms ⇒ 粗比 ≈ **4.4×**。这只说明"精确线在 1080p 比快速线慢一个数量级
的方向没变"，具体倍数要在同一租约下的正式配对测速档才能定。

---

## 2. 设计与"三路同源"是怎么成立的

| 格 | 来源 | 运行时 | 是否重跑 |
|---|---|---|---|
| E 精确线 | 本轮 `three_way_exact_v1.py` | `nr_exact_runtime_v1.Session`（= `product/comfy/runner.py:101` 产品 mode=exact 那支，**不是** `branched_exact_v1` 候选分支） | 新跑 |
| A INT8 | `D:/hires-rows-v1/itl1/frames/A/` | `nr_runtime_v1.Session` + `rows_scopes_v1`（ViT FFN = INT8 行分批） | **未重跑**，sha 绑定 |
| B FP16 | `D:/hires-rows-v1/itl1/frames/B/` | 同上，ViT FFN = FP16 前馈（已落地默认） | **未重跑**，sha 绑定 |

**输入同源**：三格都吃 `reference/inputs/flow-full-1920x1080-v3/` 的 13 对
（`frameNN.png` + `frameNN.rg32f` **DIS 估计光流**，源帧 180–191 + 第 13 帧回到 180 并重置）。
校验函数**直接 import** itl1 的 `load_inputs`（不留第二份校验逻辑），逐帧查 manifest 维度、
13 帧、12 个不同画面、png/flow 的 sha、flow 最大幅度、`effective_half_motion_sha256`；
像素/光流的上界转换与 itl1 主循环逐句一致。

**一个主动偏离，必须点名**：`materials_v1/exact_capture_v1.py:74-75` 在进 Session 前做
`rgb.half().float()` / `motion.half().float()` —— 那是**为对齐 4060 原生 R16 纹理传输**加的，
不是精确线 Session 的要求（`nr_exact_runtime_v1.py:67-70` 只要求 float32）。
itl1 两臂拿到的是未预舍入的 float32（快速线在 `sampling.py:121` 内部才 `motion.half().float()`）。
⇒ **本轮三路都不做入口预舍入**，三路拿到同一对 float32 张量，比的才是"路线"本身，
而不是"路线 × 传输口径"。用户这次明确不要 4060 那格，所以那个对齐没必要。

**内核：一次没编。** 走 `product/comfy/runner.py:83-93` 同一条 `CacheOnly` 路：
缓存包 `D:/Codex-NR-Experiments/nr-b580/product-precompile/exact-v1`（996 条特化，2026-09-11 建成）
的 `cases` 已含 `1920×1080` 的 `reset=true/false` 两种（各 7623 次调用）⇒ 无需采集、无需编译。
实测 `disk_hits = 996`、`Missing fast artifact` / `Kernel absent from precompiled package` 零命中。
开跑前还核了 `runtime_fingerprint`（python 3.13.11 / torch 2.13.0+xpu / triton_key）与包内记录**逐项相同**，
以及设备 target 断言（`Intel(R) Arc(TM) B580 Graphics`）通过。

---

## 3. 本轮自带的门（结论能不能信看这些，不看叙述）

| # | 门 | 结果 |
|---|---|---|
| 1 | **两格出处**：itl1 盘上 A/B 帧摘要 == 它 `validation.json` 第 0 轮记录 | A 13/13、B 13/13（口径 = `sha256(array.tobytes())`） |
| 2 | **只许命中**：`CacheOnly` + cases 覆盖 1080p | 通过，`disk_hits=996`，零编译 |
| 3 | **轮内确定性**：整条 13 帧前缀跑两遍 | 13 帧 × 2 遍逐字节相同，`mismatches=[]` |
| 4 | **重置等效**：帧 12 与帧 0 是同一张源画面、同为重置帧、光流同为全零 ⇒ 输出必须逐字节相同 | `max\|Δ\| = 0.0`；itl1 两臂同一对帧实测也是 `0.0` ⇒ 精确线的时域排程与它同源 |
| 5 | **状态排程**：`reset_applied` 与 `sequence` 逐帧核对（重置帧 seed=1，其余 = 帧号+1） | 13/13 通过 |
| 6 | **后端来源**：`nr_backend` 必须解析到 `nr-b580/backend`（精确树） | `…\nr-b580\backend\nr_backend\__init__.py` |
| 7 | 帧形状/值域：`(1080,1920,3)` float32、有限、0..1 附近 | 通过 |

**没有的门，以及为什么没有**：1080p 精确线**没有逐字节冻结参考** —— 4060 逐字节一致只在
480×864 那 243 帧上验过（`HANDOVER.md` §2.1）。所以本轮不提任何"对 4060 字节门"，
只提"三路输入同源"，并用门 1/3/4 实证。

**帧 0 不是透传参照**（记下来免得被误读）：无历史时三路都仍在改像素 ——
精确线帧 0 对输入 `max|Δ| = 0.2718 / mean 0.0100`，INT8 臂 `0.2485 / 0.0099`。

---

## 4. 数值（量纲 0..1；×255 即 8bit 级；指标出自 `verdict_metrics_v1`，`data_range=1.0`）

| 对比 | max\|Δ\| 最坏 | max\|Δ\| 中位 | mean\|Δ\| 最坏 | PSNR 最坏 | PSNR 中位 | SSIM 最坏 | 高频 RMSE 中位 | 超 1/255 占比中位 |
|---|---|---|---|---|---|---|---|---|
| E vs A（精确 ↔ INT8） | 0.21680 | 0.19482 | 0.003728 | 41.285（帧 1） | 42.171 | 0.99702 | 0.01387 | 0.17602 |
| E vs B（精确 ↔ FP16） | 0.10742 | 0.07666 | 0.001638 | 47.800（帧 0） | 49.585 | 0.99898 | 0.00849 | 0.05959 |
| A vs B（INT8 ↔ FP16） | 0.17627 | 0.14307 | 0.003477 | 43.266（帧 2） | 44.337 | 0.99778 | 0.01208 | 0.16834 |

* 13 帧里 **E↔B 逐帧都优于 E↔A**（无反例）；`identical_frames` 三对都是空集 ⇒ 没有任何两格逐字节相同。
* 帧 0 与帧 12 三列数值完全相同 —— 是同一张源画面 + 同为重置帧，属预期（门 4）。
* 逐帧表、SSIM/PSNR 全部明细：`D:/hires-three-way-v1/review/METRICS3_v1.md`
  与机器可读 `review/metrics3_v1.json`（含每格每帧的 sha）。

---

## 5. 看图素材（交给用户裁决的就是这些）

| 用途 | 路径 | 规格 |
|---|---|---|
| 三格原尺寸纵排静帧 | `D:/hires-three-way-v1/review/still/still_frame00..12.png` | 1920×3372（每格 1920×1080 + 48px 标题带） |
| 固定中心区 2× 细节 | `review/zoom/zoom_frameNN.png` | 中心 `box=[640,360,1280,720]` @2× 横排 3840×768 |
| 最差细节区 2× | `review/zoom/zoom-worst_frameNN.png` | `box=[837,400,1477,760]`（按整段 \|E−A\| 的 640×360 窗口和选**一次**、全程共用，规则入档） |
| 两两差异热图 | `review/diff/diff_frameNN.png` | \|E−A\| / \|E−B\| / \|A−B\| ×8 增益，1920×3372 |
| 视频 · 整幅横排 | `review/video/comparison-three-way-full.mp4` | 5760×1128、156 帧、6.50 s、h264/yuv420p、sha `69c87c73d974…`、10.04 MB |
| 视频 · 中心细节 2× | `review/video/comparison-three-way-zoom.mp4` | 3840×768、156 帧、6.50 s、sha `6ace552aa917…`、4.55 MB |
| 视频 · 差异热图横排 | `review/video/comparison-three-way-diff.mp4` | 5760×1128、156 帧、6.50 s、sha `3c6a93be698a…`、21.93 MB |

三条视频都用 ffprobe 复核过（moov 在位、可解码、帧数与时长对得上）。
**视频是逐帧慢放**（每帧停留 0.5 s）：素材只有 12 帧不同画面，所以能看单帧差异与相邻帧跳变，
**不能**当实时播放判时间稳定性/闪烁 —— 要判闪烁得跑更长的连续段（见 §7）。

---

## 6. 本轮过程中发现并修掉的缺陷（逐条登记，不沉默）

| # | 缺陷 | 现象 | 状态 |
|---|---|---|---|
| 1 | 我自己的"两格出处"门用错摘要口径：拿 **`.npy` 文件字节**去比 itl1 记的 `sha256(array.tobytes())`（文件多 128 字节 npy 头） | 第一版 `three_way_exact_v1.py` 在**还没上卡**时就把 A 臂 13/13 全判成"不符"并作废整档（`D:/hires-three-way-v1/stage-exact.log` 留着实跑栈） | **已修**：改成 `np.load(path).tobytes()` 同口径，并额外记 `npy_file_sha256` 供外部核对；修后 A 13/13、B 13/13 |
| 2 | `hf_rmse` 传 float64 | `cv2.cvtColor` 只吃 8u/16u/32f ⇒ `Unsupported depth of input image: depth is 6` | **已修**（三个指标一律原样 float32，与 `verdict_metrics_v1` 自己的调用口径一致）；由**跑前假数据自检**抓到，不是正式档抓到 |
| 3 | 编码中间件文件名用了 `encoded-<stem>`（丢了 `.mp4`） | ffmpeg `Unable to choose an output format … Invalid argument`，三路管道全废 | **已修**：中间件保留扩展名；同样由自检抓到 |
| 4 | 报告里"细节 `zoom_frameNN.png`"一行把 box 写成**最差选区**坐标（该图其实用固定中心区） | 数值不受影响，但会把人指到错的地方看图 | **已修**并重出报告；修正前的现场证据留在 `D:/hires-three-way-v1/_defect-r1-label/`（md + json + NOTE，≈8 KB）。重出的三条视频 sha 与 r1 **逐字节相同** ⇒ 编码路径确定性顺带得到一次实证 |

---

## 7. 未决 / 边界

1. **画面裁决未出**：§4 只是数值。用户看完 §5 的图/视频才算裁决。
   **⚠️ 截至 2026-09-19 交接时仍未出**（本轮素材是当天最后一步产出的）⇒ 接手者第一件事是请用户看，
   **不要**拿 §4 的 dB 自行判"可接受/不可接受"（本项目质量判定权在用户，`HANDOVER.md` §4）。
2. **精确线在 1080p 到底值不值这 ≈4.4×** —— **2026-09-19 用户明确「不用了」⇒ 本轮之后不再排期。**
   留方法给后人（若将来要判）：本轮判不了（诊断毫秒不算证据），要判得在**同一租约**下做精确线 ↔ 快速线的
   配对测速，且 1080p 机器状态是双峰台阶（`hires_rows_v1/RESULT.md` §4，台阶 91.6 ms）
   ⇒ **必须单帧 A-B-A-B 交替那一档的粒度**，整趟 ABBA 挡不住。
3. **时间稳定性/闪烁判不了**：13 帧、12 帧不同画面。要判需同一素材的连续长段（例如 243 帧那批），
   而 1080p 长段还要先补 DIS 光流包（`inputs/flow-full-1920x1080-v3` 只有 13 帧）。
4. **运动矢量仍是 DIS 估计值**，不是引擎真值 ⇒ 绝对毫秒当不了产品性能承诺（既有边界，未变）。
5. **精确线本轮用的是产品那支 `nr_exact_runtime_v1.Session`**，没带 `exact_pipeline_v1` 的 11 个
   候选 scope（那些是"让精确线自己变快"的另一条线，§2.1）。若用户想知道"候选全开的精确线"
   长什么样，那是另一轮。

---

## 8. 留存现状（2026-09-19）

| 路径 | 内容 | 体积 |
|---|---|---|
| `D:/hires-three-way-v1/exact/` | `validation.json` + `frames/E/frame00..12.npy`（13 × 23.7 MiB float32） | 309 MB |
| `D:/hires-three-way-v1/review/` | `METRICS3_v1.md` + `metrics3_v1.json` + still/zoom/diff PNG + 3 条 mp4 | 173 MB |
| `D:/hires-three-way-v1/_defect-r1-label/` | 缺陷 4 修正前的 md + json + NOTE | ≈8 KB |
| `D:/hires-three-way-v1/{stage-exact.log,stage-exact.log.lease.json,temp/}` | 租约日志（含缺陷 1 的失败现场） | ≈60 KB |
| `D:/hires-rows-v1/itl1/frames/{A,B}/` | **两格的出处，未动**（§9.2 冻结口径之外的本轮引用物） | 309 MB |

E 帧与 review/ 都可由 `bash three_way_run_v1.sh`（`STAGES=exact` / `STAGES=report`）重出：
`exact` 段约 1 分钟占卡（DiskOnly，无需再编译），`report` 段纯 CPU 约 2 分钟。
若按 B 档口径再清一次，删 `exact/frames/E/*.npy` 与 `review/{still,zoom,diff,video}` 即可，
`validation.json` / `metrics3_v1.json` / `METRICS3_v1.md` 必须留。

**绝不可删**：`D:/fullsize-rows-v1/{r1,r2}`（冻结参考）、两处 `triton-cache`、
`D:/Codex-NR-Experiments/nr-b580/product-precompile/exact-v1/`（本轮依赖的精确线缓存包）。
