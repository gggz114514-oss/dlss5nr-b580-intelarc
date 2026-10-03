# B580 Neural Rendering 后端研究

视频验证范围已修正：旧抽帧的时间戳导致首帧重复，原相邻视频帧声明撤回；实际输入的
字节一致比较仍有效，见 [VIDEO_FIXTURE_CORRECTION.md](VIDEO_FIXTURE_CORRECTION.md)。
最新已合入（9月9日）：原注意力64项求和及权重归一化融合后，完整1080p交错对照由
2.298秒降到2.189秒/帧，约快4.9%。正式55帧与主目录13帧及独立读盘审计全部通过。
主后端35个源码文件，原模型和舍入保留，见[ATTENTION_WEIGHTS_SPEED_STATUS.md](ATTENTION_WEIGHTS_SPEED_STATUS.md)。
此前C32归一化融合的成对实验约快6.6%，注意力指数融合约快8.1%；不同轮次绝对时间不混算。
此前已合入的速度优化：K16矩阵使用4×32分块/单warp，完整1080p交错对照由
6.22秒降至2.74秒，约2.27倍。正式包55帧、实际主目录13帧及独立读盘审计全部一致，
详见[SPEED_OPTIMIZATION_STATUS.md](SPEED_OPTIMIZATION_STATUS.md)。
主后端的可选triton路径现已接入注意力求和/权重融合、C32归一化、注意力指数融合、4×32单warp的K16分块复用、FP8转换及cubic激活融合，保留完整INT32
累加与舍入、半精度乘加融合、窗口分组及有限FP16运动范围；默认reference路径未变。
检查点按十二个优化源码版本分别记录372次历史参考比较和736次融合比较。
最新注意力权重融合版本68帧包括39帧运动测试、16帧四尺寸风格/UI，
以及13帧实际主目录真实视频回归；保存的显示和历史结果均已重新读回核验。
480p主目录后续帧约0.65秒，1080p相同源码独立目录约2.19秒，Apex控制/UI约3.57秒，仍非实时。
这些回归计时未交错运行旧版，小尺寸尚未证明提速；本轮收益依据为独立的1080p交错实验。
此前1080p交错对照支持cubic直接融合额外约6%的速度收益；没有改变模型精度或启用INT8。
两轮1080p成对实验支持分块约1.68–1.79倍收益；不同实验绝对耗时有波动。
历史控制/遮罩测试不直接算作本版覆盖，详见[SPEED_OPTIMIZATION_STATUS.md](SPEED_OPTIMIZATION_STATUS.md)。
INT8 XMX第一版变慢的定量诊断见[INT8_XMX_DIAGNOSIS.md](INT8_XMX_DIAGNOSIS.md)；
OptiScaler接口及显存共享缺口见[OPTISCALER_INTEGRATION_STATUS.md](OPTISCALER_INTEGRATION_STATUS.md)。
此前完整INT32版本在实际PyTorch队列完成D3D12共享缓冲区的FP32/FP16各32轮往返，随后完整NR也通过
480p和1080p各13帧共享输入/输出测试；D3D12读回RGB和私有历史全部逐字节一致。
26次图形共享比较按存档的完整INT32源码单独写入检查点，不算作当前注意力融合版本的图形回归；尚未接入调用方游戏纹理和命令队列，详见
[GPU_RESOURCE_INTEROP_STATUS.md](GPU_RESOURCE_INTEROP_STATUS.md)。

本机原清单约185.0GB实验数据已迁至D:/Codex-NR-Experiments/nr-b580，原路径联接已恢复；
源码留在E盘，新实验数据及缓存写D盘，见[STORAGE_MIGRATION_STATUS.md](STORAGE_MIGRATION_STATUS.md)。

参考机空间清理：四批累计70.70GB、97,032个文件已校验归档到本机。第四批100个已完成实验目录（12.51GB）放在
[本机D盘](D:/Codex-NR-Archive/nr-b580/20260908-d/README.md)，清理后当时参考机E盘剩33.21GB。
后续安装本轮运动输入时实测空闲161.63GB；这些空间记录各对应测量时刻。
当前大尺寸数据及运行环境保留；历史远端路径如需复用，应先从对应归档恢复。

历史参考算术版本通过512×512、864×480、1920×1080和2559×1439非默认控制/UI组合的16帧完整模型，
显示RGB和私有历史均逐字节一致；累计372次完整RGB比较（含重复重置输入），见[CONTROL_DIMENSION_STATUS.md](CONTROL_DIMENSION_STATUS.md)。
此前流内UI切换、Alpha回退和Backbuffer-only输入选择40帧，以及基础UI及六组固定控制
组合36帧也已通过。接口见[UI_BACKEND_USAGE.md](UI_BACKEND_USAGE.md)，流内证据见
[LIVE_UI_STATUS.md](LIVE_UI_STATUS.md)。
此前运行中自动遮罩、皮肤强度及显式遮罩绑定切换28帧也已通过，临时显式重置限制已解除。
切换规则见[LIVE_AUXILIARY_STATUS.md](LIVE_AUXILIARY_STATUS.md)。显式遮罩接口及范围见
[CONTROL_MASK_BACKEND_STATUS.md](CONTROL_MASK_BACKEND_STATUS.md)。自动遮罩见
[AUTO_MASK_BACKEND_STATUS.md](AUTO_MASK_BACKEND_STATUS.md)。略高于1的HSL饱和度标量域已穷举验证；
上述UI绑定切换和可选Alpha/Backbuffer已通过，其他控制交叉组合仍在研究。该历史版本完整模型约20秒/256×256帧，尚未完成
XMX/INT8加速和工具集成。

第一版可选融合路径接入主后端时，通过22组GPU算子比较、四尺寸16帧完整模型及主目录实际调用4帧。
20次融合RGB及私有历史比较全字节一致，与原参考路径372次比较分别记录。
编译后512约5.81秒、864×480约8.17秒、1080p约35.39秒、2559×1439约59.28秒；
同输入和控制下整帧约快5.2–5.5倍。首次新尺寸编译约230–263秒。默认仍为reference，
显式调用`use_arithmetic_backend('triton')`启用融合；尚非实时，也未使用XMX/INT8，
调用与范围见[FUSED_ARITHMETIC_STATUS.md](FUSED_ARITHMETIC_STATUS.md)。

后续独立候选已通过：1024窗口分组16帧，以及合并INT32对齐乘积求和的16帧，
均保持RGB和私有历史逐字节一致。合并版编译后512约3.18秒、864×480约4.21秒、
1080p约16.72秒、2559×1439约27.83秒；仍未使用XMX/INT8，已接入主目录并复验四帧。
落盘输出重新校验与范围见[INT32_ARITHMETIC_STATUS.md](INT32_ARITHMETIC_STATUS.md)，
分组独立收益见[WINDOW_GROUPING_STATUS.md](WINDOW_GROUPING_STATUS.md)。
社区INT8检索和跨工具复用范围见[COMMUNITY_INT8_AND_REUSE.md](COMMUNITY_INT8_AND_REUSE.md)。

此前INT32主后端又通过20组运行中辅助控制/UI和固定自动遮罩场景，共80帧RGB及历史
全部逐字节一致，见[FUSED_CONTROL_STATUS.md](FUSED_CONTROL_STATUS.md)。检查点将372次
历史参考比较与当时120次融合比较按源码版本分别记录。后续半精度乘加融合已通过
CPU10组/GPU11组算子检查和四尺寸16帧完整模型，1080p约13.33秒、2559×1439约22.78秒，
已推广进主目录并复验四帧，见[HALF_FMA_FUSION_STATUS.md](HALF_FMA_FUSION_STATUS.md)。

当前目标：实现可复用的 Intel Arc B580 NR 执行后端，由 OptiScaler 和现有 XeSS/ComfyUI 视频工具调用。256×256六份输入、512×512、完整864×480、1920×1080以及Apex原始2559×1439帧的最终RGB均已与固定RTX4060参考逐字节对齐。Apex的44,188,812个RGB字节全部一致；尺寸和显存修正见 [DIMENSION_BACKEND_STATUS.md](DIMENSION_BACKEND_STATUS.md)，半精度融合乘加修正见 [HALF_FMA_STATUS.md](HALF_FMA_STATUS.md)。256×256零运动四帧连续链路也已完全一致，包含历史累积与重置，见 [TEMPORAL_BACKEND_STATUS.md](TEMPORAL_BACKEND_STATUS.md)。非零运动时序、性能与工具集成仍在开发。

用户已明确以完整迁移为目标。完成条件包括固定参考运行库的单帧输出、时序状态和控制参数行为对齐，以及可复用的 B580 执行后端；INT8 优化在正确性成立之后进行。下述研究进展不表示已经完成迁移。

新增非零运动进展：`MotionNR256` 的整数、亚像素、跨边界及两组逐像素变化运动，
加上零运动回归，共24帧最终RGB全部逐字节一致。原生半纹理采样的权重精度、
有效位对齐、带符号输出、融合乘加与倒数舍入已纳入后端，
见 [MOTION_SAMPLING_STATUS.md](MOTION_SAMPLING_STATUS.md)。
长视频、其他尺寸时序和控制参数仍待扩大验证。

后续视频素材探针：两段用户视频各12个重复首帧的256裁剪，另加回到首帧的重置，
共25帧全部逐字节一致。运动为合成fine场、零深度；该结果不等于全尺寸视频或真实光流
链路已完成。报告：`reference/results/temporal-video-rgb-b580-v1/validation.json`。

`MotionNR` 另已通过512×512的零运动、逐像素fine运动各四帧，共8帧最终RGB全字节
一致，约32–33秒/帧；见 [调用接口](MOTION_BACKEND_USAGE.md)。256兼容接口仍保留。

全尺寸时序：864×480视频首帧重复输入，零运动及fine运动各四帧，共8帧最终RGB
逐字节一致，约43–45秒/帧。新增归一化坐标21位小数转换及尺寸倒数规则，见
[NORMALIZED_MOTION_STATUS.md](NORMALIZED_MOTION_STATUS.md)。1920×1080零运动/fine两组共8帧
也已全部逐字节通过，约181–183秒/帧，峰值已分配显存约2.93GB。2559×1439原始Apex
截图的零运动/fine两组共8帧也已全部逐字节通过，约303–305秒/帧，峰值约4.63GB。
此项为重复截图和合成运动探针；更长全尺寸视频及真实光流仍待验收。
非默认控制进展见 [CONTROL_BACKEND_STATUS.md](CONTROL_BACKEND_STATUS.md)。样式1/2已在
256×256原权重完整会话中各通过四帧，显示输出与私有历史均逐字节一致；
调用方式见 [STYLE_BACKEND_USAGE.md](STYLE_BACKEND_USAGE.md)。强度0/0.5/1/2也已各通过
四帧完整模型。局部色调0/2、结构0/2各四帧，以及六组连续取值/组合参数共24帧也已
全部逐字节一致；统一接口见 [CONTROL_BACKEND_USAGE.md](CONTROL_BACKEND_USAGE.md)。
运行中改参新增24帧也已全部逐字节一致，包含隐式重置和样式/局部色调联动，见
[LIVE_CONTROL_STATUS.md](LIVE_CONTROL_STATUS.md)。目前检查点收录372次完整RGB比较
（含重置复现），见`reference/backend-checkpoint.json`。更大控制尺寸、深度交互、UI与样式/遮罩交叉组合、
真实光流、性能和工具集成仍待完成。

**用户已完成本轮画面审核并要求恢复迁移工作。** 用户在查看真实素材对照后明确回复“没问题，开工咱一定要搞出来”。这通过了本轮三份素材的候选参考验收门槛，恢复 B580 算子翻译和原生内核独立重放。当前 SF-v2 仍是社区兼容运行库；用户视觉认可不等于证明与官方 RTX50 运行库逐位等价，也不代表已经完成 B580 迁移。

用户已提供 Apex 截图（2559×1439）、480p 视频（864×480 / 24 fps / 243 帧）、1080p 视频（1920×1080 / 60000/1001 fps / 390 帧）。`reference/inputs/visual-qa-01` 保存未修改原文件的副本、哈希及有限字段媒体清单；笔记本实验放在 `E:\Codex-NR-Reference\visual-qa-01`。三份素材已经完成4060运行，结果在 `reference/results/visual-qa-01`；`review/REVIEW.txt` 是审核入口说明。

本轮自检：截图尺寸保留，host 原图 RGB 与输入逐像素一致；截图原始模型 RGB 全有限且在 [0,1]。两段视频完整243/390帧，源解码与无损基线逐帧哈希一致；四个最终MP4观看副本的帧数/帧率/尺寸和原音轨解码哈希已核对。已抽查截图局部、全片分散视频帧、人物112–119帧及赛车170–177帧。人物肤色/阴影、皮肤纹理及赛车反射有变化；UI修正开/关的本张截图结果逐像素相同，功能是否有效仍未验证。原始浮点全有限检查仅针对截图，视频检查基于导出的RGB8。`review/analysis.json` 保存全片数值诊断，`review/delivery-checks.json` 保存交付验证。用户后续已通过本轮肉眼审核，记录见 `reference/visual-qa-user-acceptance.json`。

**最新用户审核规则：输出字节完全一致时无需再次肉眼审核。** 用户原话：“继续，如果输出的字节完全一致就不用找我肉眼审查”。256和512已有明确视觉认可；后续完全一致的输出通过技术验收后可继续，有差异的候选结果仍保留用户肉眼审核。原始RGB数值比较是依据，不能仅比较PNG或压缩视频文件。

画面验收先检查同源输入/输出的颜色、曝光、结构、边缘、细节、裁剪与缩放，然后检查连续帧闪烁、拖影、重置与运动信息。原图、关闭 NR 的基线、开启 NR 的结果必须在一致显示条件下可直接比较。视觉正常与正式运行路径的逐位等价是两项独立证据；当前两者均不能凭调用成功来认定。

## 最新进展：B580 独立 RGB→RGB 执行，用户画面审核通过（2026-09-08）

`ResetNR256` 直接输入 RGB，在 B580 上执行完整原始权重图，包含 front、编码器、八个 ViT、解码器及 RGB 输出。运行时无需 NVIDIA 设备，也不读取参考中间特征。六份 256×256 输入、四种种子均与固定 SF-v2 RTX4060 参考的最终 RGB 逐字节一致；人物、赛车、Apex 三组对照已由用户肉眼审核，回复“这版画面正常，继续”。证据、范围与调用方法见 [RESET_BACKEND_STATUS.md](RESET_BACKEND_STATUS.md)。

尺寸扩展的最新证据见 [DIMENSION_BACKEND_STATUS.md](DIMENSION_BACKEND_STATUS.md)。当前采用整数张量模拟原生计算的舍入规则，256每帧约17–20秒，512约29.6秒，864×480约40秒，尚非实时后端。时序、控制参数、XMX/INT8优化及OptiScaler/现有工具集成仍未完成。以下各节是历史门槛，不应当作当前完整状态。

## 历史门槛：完整编码器与八个 ViT 块逐字节对齐（2026-09-08）

最新门槛为 `reference/results/vit-holdouts-v2/encoder-prefix-validation.json`：四组素材、CPU/B580、每组120个输出缓冲区，共960次比较全部零差异。从相同原生front出发，连续运行全部编码器及八个ViT块，中途不替换计算张量。详见 [VIT_BACKEND_STATUS.md](VIT_BACKEND_STATUS.md)。当前推进解码器；完整NR、便携front噪声、时序、最终画面与性能仍未完成。

## 完整编码器至 ViT 输入逐字节对齐（2026-09-08）

最新门槛为`reference/results/c512-holdouts-v2/encoder-prefix-validation.json`：相同原生front→pre→C32×4→C64×4→C128×6→C256×8→C512×8→池化→C1024投影，四组素材、CPU/B580、每组62个原始输出缓冲区全部零差异，中途不替换计算张量。详见[C512_BACKEND_STATUS.md](C512_BACKEND_STATUS.md)。此前28缓冲区门槛及C256的10→12补零布局详见[C256_BACKEND_STATUS.md](C256_BACKEND_STATUS.md)。

C512的FFWD实际为全通道投影加八组64→256→64计算，八个块、6→8补零池化和最终投影均已纳入四组连续验证。

## pre 与四个 C32 编码块的验证（2026-09-08）

`reference/results/c32-holdouts-v1/pre-c32-chain-validation.json` 已通过：从相同原生front特征开始，由CPU/B580连续执行pre、四个C32 Swin块和下采样到C64，中途不替换计算张量。gradient/checker/gray/Apex四组素材、两设备、每组七个输出缓冲区全部零差异。范围仍是输入256、pre padding320、C32为160×160、C64为80×80，便携front噪声和完整NR不在此门槛内。

新增 `C32SwinBlock` 实现原生零扩展移窗、FP8/half边界及C32→C64池化投影，可复用已验证的MLP和注意力组件。72个holdout输入/输出文件哈希及其与原生pre输出的同源关系已检查。参考CUDA四调用在gradient的每一步完整15,711,232字节arena均与NGX捕获一致。详见 [C32_BACKEND_STATUS.md](C32_BACKEND_STATUS.md)。当前下一阶段是C64双头注意力，其后仍有更深编码器、ViT、解码器、时序和OptiScaler集成。

## pre 主体的两份输出均逐字节对齐（2026-09-08）

详细边界、算法和证据见 [PRE_BACKEND_STATUS.md](PRE_BACKEND_STATUS.md)。新增自有执行组件在 `backend/nr_backend/`，尚不是完整 NR 后端。

**最新完整门槛已通过：在相同原生 front 特征输入下，CPU/B580 的整个 pre 主体——投影、MLP、注意力、输出投影与2×2下采样——对四份输入均与未修改的原生 CUBIN 逐字节一致。** 每份输入包含3,276,800字节skip及819,200字节down。报告为 `reference/results/pre-holdouts-v1/pre-outputs-validation.json`。便携随机噪声生成仍有小误差，后续NR blocks与时序尚未移植，因此这不是完整NR迁移完成。

- RTX4060 独立 CUDA 重放原生 pre CUBIN，skip 与 down 两块输出和 NGX 原生捕获逐字节一致。以后可直接在 SSH 会话中隔离测试这些算子。
- 恢复 reset front 的 16 通道、坐标噪声哈希、颜色缩放和反射边界。13 个非噪声通道在四组素材上逐位一致；3 个噪声通道仍有便携超越函数误差。
- 恢复原生 HMMA/QMMA 的权重排列、共享指数截断和中间 FP16 舍入。给定相同 front，第一层投影在四组素材的全部 `320×320×32` 值上，CPU/B580 均与原生逐位一致。
- 第一层 MLP 的 128 通道扩展及 cubic 激活，在四组素材、每个 CTA 的 16 个采样像素上全部逐位一致。完整第一层 MLP（扩展、激活、收缩、残差、FP8 边界）从相同投影输入出发，在四组素材的全部像素/32通道上也全部逐位一致。
- 从原始 RGB 开始包含便携噪声的 B580 链路，每组 `3,276,800` 个 MLP 输出中仍有 `75/102/66/112` 个不同值，依次为 gradient/checker/gray/Apex(seed17)；MAE 为 `3.40e-7/4.99e-7/3.32e-7/4.06e-7`，最大绝对差均 `0.125`。不能以接近为由认定整体逐位对齐。
- 注意力的QKV、归一化、指数近似、行求和/倒数、V乘法与残差投影已纳入上述完整pre输出验证。FP8次正规操作数需保留格式指数-6，不能按数值重新规格化；这消除了最后5个skip字节差异。
- 下采样使用未量化输出的half 2×2行内配对求和，再乘0.25，最后量化FP8；不能从已量化skip做池化。下一项是后续Swin blocks和前端噪声精确性，再到最终输出及视频时序。当前实现用于验证正确性，尚未做 XMX/INT8 性能优化或接入 OptiScaler。

## 历史诊断：跨设备逐层记录与原生中间数据（2026-09-07 22:50）

- 复用笔记本 E 盘 ComfyUI 的 `python_embeded\python.exe`，CUDA PyTorch `2.13.0+cu130`；本机为 `2.13.0+xpu`。没有修改两个现有 Python 环境的依赖。笔记本语义图试验源码和权重在 `E:\Codex-NR-Reference\semantic-graph-v1`，缓存和输出也在 E 盘。
- `reference/trace_semantic_graph.py` 用同一固定输入、源文件、权重和量化实现，记录两张 GPU 的 97 个中间/输出张量。`compare_semantic_traces.py` 校验输入、源码、脚本、权重和张量哈希后比较。数据在 `reference/results/trace-xpu-portable-02`、`trace-cuda-portable-01`；汇总为 `reference/results/semantic-cross-device-01.json`。
- 枚举全部 65,536 种 FP16 位型后，CPU 原生 FP8 转换、CUDA 原生转换和两张卡上的 portable E4M3 转换逐位一致（按当前图明确的 NaN→0、饱和到 ±448 规则）。这仅验证该转换规则，不证明整个图与 NR 内核等价。
- 原版 NR 与恢复图的 gradient RGB MAE：B580 **1.64796**，4060 CUDA **1.74049**。因此缺陷不是 B580 特有。前端投影在两卡上有 99.9999% 值精确一致，但后续差异不断放大，最终跨卡 MAE 为 1.23047。
- 在 FP8 饱和处理之前额外检查，发现两卡均有 **16 处非有限中间结果**，位于形状 `[1,8,8,512]` 的阶段。先前仅检查最终输出有限会漏掉这个问题。尚未确定全部溢出的根因，不能简单将它归为硬件量化错误。
- 新增自有参考进程内的可选 NVAPI 跟踪模块：`reference/native-trace/nr_nvapi_trace.cpp`，通过 `src/nr_trace_loader.h` 在参考程序中显式加载。MinHook 固定为官方 v1.3.4、提交 `c3fcafdc10146beb5919319d0683e44e3c30d537`。未改动原 NR DLL、驱动或其他应用。原有未跟踪程序仍保留。
- 笔记本 `video2dlssnr-trace-v2` 记录 9 个实际加载模块和 156 次实际内核调用，参数已逐字节保存；`video2dlssnr-trace-v3` 进一步记录原生前端之后和 post 之前的两份 15,711,232 字节工作缓冲区。CPU 映射在自有 D3D12 队列 fence 完成之后执行。构建源码包及各版本哈希均保存在 `reference` 的 `.source.json` / `4060-*-build.json`。
- v2 数据：`reference/results/nvapi-trace-256-20260907-223414`；v3 数据：`reference/results/nvapi-trace-256-20260907-224616`。`analyze_native_trace.py` 验证全部捕获哈希，并确认两版采集的 gradient 输入、depth、motion、原始 RGBA 输出和 PNG 均与未进行内核跟踪的基准逐字节相同。
- 原生参数确认输入 256×256，pre 内部尺寸 **320×320**，down 为 **160×160**；当前语义图的尺寸和边界处理未与此对齐。pre skip 区由两个输出地址限定，长度 3,276,800 字节；下采样候选区域 819,200 字节。当前 buffer 地址是每次进程特有，不能写死到后端。
- v3 的 pre skip 在 after-pre 与 before-post 两份快照间完全一致，SHA-256 `ae940bce1d23edacac45271031d01f1350af38edcd31e534e29b2fb8ab682edb`。按 E4M3 解释范围 -4.5..7，无 NaN 编码。down 区后续被复用，两次快照不同，不能用最终缓冲区冒充早期输出。
- 独立重复运行 `nvapi-trace-256-20260907-225101` 的 pre skip、pre down 和最终原始输出均与第一遍逐字节一致；报告为 `reference/results/native-pre-repeatability.json`。这增强了捕获的可重复性证据，但还没有证明物理到逻辑张量的排列。
- 原生模块为压缩 fatbinary；按头部 `header_size + data_size` 提取容器后，可用官方 cuobjdump 解出 SM89 CUBIN。直接把包含尾部填充的完整模块交给 cuobjdump 会报 Invalid fatbin header。官方 CUDA 13.0.85 `cuobjdump`/`nvdisasm` 仅解压在本机研究 downloads 目录，校验了官方 manifest SHA 和 nvdisasm 签名，未安装 CUDA Toolkit 或驱动。前端 SASS 与完整分析在 v2 数据目录的 `disassembly`。
- `analyze_arena_snapshots.py` 提取物理字节并明确区分已知地址范围与候选张量形状。`probe_pre_pool_layout.py` 检查了五种布局组合，均未通过池化关系的数值对照（MAE 0.696–0.931）。**通道与像素的实际排列仍未恢复，当前原始 buffer 不是已验证的逻辑张量。**

当时的下一步骤是恢复front producer和独立重放，现已取得上节所列进展；pre skip/down物理布局已通过完整输出验证。不得通过裁剪错误输出、拟合单张参考图或用蒸馏模型替代原模型来宣称成功。

内核跟踪运行（当前 runner 选择 v3，单张 gradient，REFERENCE_USER 保持登录）：

```powershell
$null = & '.\nr-b580\reference\Invoke-Reference.ps1' `
  -ScriptPath '.\nr-b580\reference\Invoke-InteractiveSmoke.ps1' `
  -ScriptParameters @{Trace=$true} `
  -OutputPath '.\nr-b580\reference\next-native-trace.json'
```

跟踪过程大量读回且同步，所有计时均不应作为模型性能基准。

## 已确认的参考环境（2026-09-07）

- 参考电脑：`REFERENCE_HOST`，`REFERENCE_MACHINE`，SSH 用户 `REFERENCE_USER`。
- NVIDIA GeForce RTX 4060 **Laptop GPU**，8188 MiB 显存；驱动已由用户升级并实测为 **616.86**（最初为 591.74）。
- Windows 11 IoT Enterprise LTSC，build 26100；VS 2022 BuildTools 可用。
- SSH 主机 ED25519 指纹：`SHA256:REFERENCE_HOST_FINGERPRINT`，已与用户返回值比对并固定。
- 客户端任务私钥保存在本机用户的 `.ssh/REFERENCE_KEY`；不能复制到工程、参考机或日志。专用 known_hosts 为 `.ssh/REFERENCE_KNOWN_HOSTS`。
- 参考机 SSH 任务密钥和防火墙限制客户端为 `REFERENCE_CLIENT_HOST`。服务当前运行，启动类型 Manual。若重启后未运行，在参考机管理员 PowerShell 执行 `Start-Service sshd`。
- 初次配置用的本机 TCP 18746 临时 HTTP 服务和临时防火墙规则均已关闭/移除。

## 参考程序

本地源码：`vendor/video2dlssnr`，上游 https://github.com/DaniilSokolyuk/video2dlssnr ，固定提交 `55a4ceb588a419b9b56497aa0b563d0c9e2b6c77`。

已在 4060 电脑编译到：

```text
E:\Codex-NR-Reference\video2dlssnr-55a4ceb5\out
```

产物为 `video2dlssnr.exe`、`video2dlssnr_tests.exe` 和转发器 `nvngx.dll_dlssnr.dll`。转发器并不是包含模型的 `nvngx_dlssnr.dll`。

官方 NGX 构建依赖来自 https://github.com/NVIDIA/DLSS ，提交与 SHA-256 见 `reference/ngx-build-dependency.json`。构建输入清单见 `reference/reference-source.json`，产物哈希与验证结果见 `reference/4060-build.json`。

已验证：18 项 CPU 测试、1303 项断言通过；11 项上游 GPU 测试未执行；CLI `--help` 成功。另已实际执行 NR：三组 256×256 单帧图像均成功，原始张量参考数据已采集并用于 B580 对照。

远程构建和测试日志在 `E:\Codex-NR-Reference\logs`。**用户明确要求后续笔记本实验保存在 E 盘**；任务控制的临时文件、工作目录、输入、输出及进程缓存也已改到 `E:\Codex-NR-Reference`。原 C 盘任务目录已迁移、逐文件校验后删除，见 `reference/4060-migration-to-e.json`。

## 参考机当前状态

1. 用户已经手动安装驱动，实测 616.86，满足参考程序 616.56+ 要求。安装包现位于 E 盘 `Codex-NR-Reference\downloads`；`reference/4060-driver-package.json` 是下载时的历史记录，里面的 C 盘路径已迁移。本任务不需要再安装驱动或重启。
2. 已从 RHI 的公开发行包下载并部署 **310.8.SF-v2**，已在 RTX 4060 登录会话下完成单帧推理。
3. 固定使用的 DLL SHA-256：`6EB209E764F39872625DEBD6ABAF45E2BB6322F6F270F781F70C059AE30B3927`。交叉核对了 Wan2GP 文档及 `taowen/dlss5-as-inpainting` 固定提交的 Git LFS 对象 ID。来源与压缩包哈希见 `reference/nr-runtime-source.json`，部署回读结果见 `reference/4060-runtime.json`。

SF-v2 清单依据：https://github.com/deepbeepmeep/Wan2GP/blob/main/docs/DLSS5.md

SF-v2 研究仓库对象依据：https://github.com/taowen/dlss5-as-inpainting/blob/f8e18d366c5610bc5a51d562356655ac0885e21f/bin/nvngx_dlssnr.dll

驱动官方说明：https://nvidia.custhelp.com/app/answers/detail/a_id/5906 。这是可选 Hotfix 测试版，修复虚拟显示器和 RDP 问题；参考机装有 Zako 虚拟显示适配器。用户已完成安装。

最初查看的 OptiScaler 分支固定了另一份兼容版 `E67DEE...989A`，没有采用该二进制；不要混用校验值。该分支的跨代说明：https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass#gpu-and-runtime-compatibility

其指向的作者 RenoDX 线程：https://discord.com/channels/1408098019194310818/1543976771920330884

兼容版的说明不能代替实测：采集结果需标注它来自修改版运行库，不能宣称是 RTX 50 原版的逐位真值。本地还下载检查了 SF 初版和 video2dlssnr v1.3 自带版本，哈希均不同，未部署或执行；它们保存在忽略的 downloads 目录中。

只读检查：在本机工程根目录运行

```powershell
& '.\nr-b580\reference\Invoke-Reference.ps1' `
  -ScriptPath '.\nr-b580\reference\Check-ReferenceReady.ps1' `
  -OutputPath '.\nr-b580\reference\4060-readiness.json'
```

脚本不加载运行库，不安装驱动，不执行模型。`prerequisitesPresent` 只反映前置文件与版本检查，不代表 NR 推理成功。

## 已完成的原始张量参考采集

SSH 的 PowerShell 处于 Session 0：D3D12 设备创建成功，但 NGX 初始化返回 `FeatureNotSupported / 0xBAD00001`。同一原版程序、同一 DLL、同一输入在 REFERENCE_USER 已登录的 Session 1 下成功。对照记录为 `reference/4060-smoke-256.json`（失败）与 `reference/4060-smoke-256-interactive.json`（成功）。后续 NR GPU 实验用 `Invoke-InteractiveSmoke.ps1` 在用户登录会话执行；一次性计划任务不存密码、不弹窗口，结束后移除。不要把 Session 0 的初始化错误当作 GPU 本身不支持 NR。

增加了仅在 `CODEX_NR_CAPTURE=1` 时启用的张量采集，修改本地 vendor 的 `src/nr.cpp` 并新增 `src/nr_capture.h`。对应构建输入归档和清单见 `reference/video2dlssnr-capture-v1.zip`、`reference/capture-source.json`，远程编译结果见 `reference/4060-capture-build.json`。原版参考程序仍保留，采集版在 `E:\Codex-NR-Reference\video2dlssnr-capture-v1\out`。

参考数据保存在笔记本：

```text
E:\Codex-NR-Reference\runs\capture-256-20260907-220919
```

本机用于比较的下载副本位于 `reference/results/capture-256-20260907-220919`；报告也已回传上述笔记本 E 盘目录。

- 三组固定输入：gradient、checker、gray，均为 256×256、SR scale=1、Reset=1、preset/style=0、intensity/local structure/local tone=1、UI correction=0、auto mask=0，传给模型的颜色为 sRGB。
- 捕获实际 GPU 颜色输入、深度、运动矢量，以及原始模型输出（在主机颜色转换及合成之前）。颜色/运动纹理为 FP16，读取后无损扩展为紧密 HWC 排列的小端 FP32 `.bin`；深度原生为 FP32。格式、控制参数和模型/DLL/程序哈希均写入元数据。
- 三组所有捕获值均有限；depth/motion 实测全零；alpha 全为 1。共 24 个输出文件，传输后 SHA-256 验证全部通过。
- 开启采集后的 gradient PNG 与此前原版程序单图输出 SHA-256 完全相同（`BF6ECC...88AA`），未观察到采集改变最终预览结果。
- `reference/compare_b580_reference.py` 已将同一 GPU 输入送入当前 XPU 语义图：checker / gradient / gray 的 RGB MAE 分别为 **1.66551 / 1.64796 / 1.68586**，远大于参考输出自身相对输入的变化（约 0.01590 / 0.01034 / 0.00439）。模型输出未裁剪以掩盖问题。这明确说明当前语义图尚未与原 NR 对齐；不是可用后端。

后续运行命令（本机工程根目录，笔记本保持 REFERENCE_USER 登录）：

```powershell
& '.\nr-b580\reference\Invoke-Reference.ps1' `
  -ScriptPath '.\nr-b580\reference\Invoke-InteractiveSmoke.ps1' `
  -ScriptParameters @{Capture=$true; Batch=$true} `
  -OutputPath '.\nr-b580\reference\next-capture.json'
```

参考机根目录 `Run-NrSmoke.ps1` 为登录会话真正执行的脚本；修改本地同名脚本后需将更新上传到笔记本 E 盘根目录。SSH helper 上传的临时执行脚本不自动替换该持久文件。

## 初期语义图探测（历史记录）

新发现的公开恢复代码：https://github.com/taowen/dlss5-as-inpainting ，已固定到 `f8e18d366c5610bc5a51d562356655ac0885e21f`，本地 `vendor/dlss5-as-inpainting`。未下载子模块或执行其本地 DLL/CUBIN；使用的是可读的 Python 权重解析与算子图代码。

从 SF-v2 静态提取 `WEIGHTS_HT`：147,695,410 字节，SHA-256 `836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4`；153 条记录，71 个块。资产及完整清单保存在 `model-assets/sf-v2`，该目录不应提交或发布。

使用现有 ComfyUI 自带 Python（没有改变它的依赖）确认 `torch 2.13.0+xpu` 可访问 B580。`reference/probe_b580_graph.py` 实测了加载原权重的**不完整语义图**：

- 489 条已实现的权重映射，模型参数数目 145,754,963；映射报告无 skipped 不代表所有语义已恢复，未解决的内容仍在 metadata 和代码中。
- 64×64 输入，XPU FP16 算子执行，模拟 E4M3 SATFINITE 中间存储边界。
- 输出设备 `xpu:0`，12,288 个输出值均为有限值；显存峰值约 309 MiB。
- 非零 SASS 前端候选下，输出范围 **-5.234375 到 7.40625**，相对 [0,1] 输入平均绝对变化约 **2.1231**。没有原生参考对照，这些结果不能作为正确 NR 画面，不能宣传为 B580 已运行正确 DLSS 5。
- 一次冷启动探测约 1.24 秒，含未优化 Python 图开销；不属于性能基准，不能据此外推分辨率或帧率。

主要缺口：前端纹理/特征组装、down/up transition 精确空间排列、block39 动态插值、ViT 指数计算精度、post 输出排列与增益语义，均需原生输出或中间张量对照。

该仓库另附一个小型蒸馏 `PortableModel`，它是近似模型，**本任务没有用它冒充原 NR 模型**。本次探测运行的是加载提取权重的语义图。

复现命令：

```powershell
& '.\ComfyUI-aki-v3-IntelArc\python\python.exe' '.\nr-b580\reference\inspect_nr_weights.py'
& '.\ComfyUI-aki-v3-IntelArc\python\python.exe' '.\nr-b580\reference\probe_b580_graph.py'
```

静态资源解析依赖 `pefile 2024.8.26` 单独放在 `analysis_deps`；未安装到 ComfyUI 环境。

## 初期工作计划（历史记录，最新进度见文首报告）

1. 已完成小分辨率单帧、固定参数的原始张量采集。接下来优先定位前端/布局错误，再增加连续帧与场景切换样本；目前数据集只验证单帧 Reset=1，尚未验证时序正确性。
2. 使用已获得的权重、恢复代码和 B580 算子探测，逐项验证前端、权重布局、算子语义、归一化和时序状态。这仍是未解决的核心；不能把静态提取或不完整算子图运行成功等同于完整移植。
3. 在 B580 上完成正确的 FP16/BF16 对照执行路径，再评估 XMX INT8 W8A8、校准误差与时序误差。INT8 峰值更高不等于整网加速两倍，权重 INT8 解量化到 FP16 也不等于 INT8 矩阵执行。
4. 共用的后端接受纹理/张量、运动矢量、历史重置及模型控制参数。OptiScaler 提供游戏输入与合成链路，现有 XeSS 视频工具提供离线输入；两者复用同一模型后端。

OptiScaler 的工作分辨率、与 SR/FG 的执行顺序可以减少 NR 负担，但不能替代 Intel 模型执行后端。当前阶段不开展新 UI 或游戏注入功能。
