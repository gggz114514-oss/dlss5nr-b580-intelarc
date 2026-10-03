# AMD DLSSNR 最新代码深审：相对现役 Cyberpunk 720p 的移植空间

审查日期：2026-10-03（Asia/Shanghai）。作者仓库默认分支 main，联网取证时间 2026-10-03 10:51:59 UTC；固定 **9ec741522d267c4d2365af53081716fb2943068a**（作者时间 2026-10-03 14:50:14 +08:00）。所有 AMD 源码链接固定到该 SHA；不会用随后变化的 main 解释本次结果。历史起点 **7ef24e7c1498bce59738277e174249866608c4ed**。

**结论：可核查的作者归档显示，RX 9070 XT 的1080档网络已达到约9.5～10ms，这是明确的性能进展。** 10-01离线HIP span约9.50～9.56ms，10-02整网回放汇总约9.84ms；可移植的关键结构是完整窗口producer→consumer闭合、有效栅格QKV、输入/权重片段复用、ViT尾部流式消费和窗口依赖调度。**本报告六项新增建议全部相对游戏现役accepted_baseline；修复未采用r15候选的退化不计入新增收益。**

90fps的口径另行区分：9.84ms约等于102次网络回放/秒，支持“网络吞吐约90以上”的理解；归档尚不能确立完整游戏Present约90fps。网络档位、1088行、跳块、ViT复用、FG与真实历史配方均须跟对应日志说明，这些差异不否定已取得的网络结构优化成绩。

## 1. 比较边界和证据等级

本地唯一数学主对照是 PHASE2-r15.json 的 **arms.accepted_baseline.constructor**，清单 SHA256 已核为 **303b5ce21c938b681616622801644290db7dd11fb038b781ce27ff9d4f6e4d78**。固定 1280×720，内部 1280×768；C512 的有效特征 24×40×512，ViT 240×1024。游戏用户 RTSS 55–60ms 是现役端到端证据。离线 process/raw GPU 时间是另一口径，不以 r15 候选的 54–55ms 冒充现役。素材/帧契约按 PHASE1.json。

现役启用 C512 library QKV、native K8，以及全部六项：decoder gather unround、C32 hidden native、C512 probability unround、C128 pairwise、C64 attention+project、C128 attention+project；另有 C128 分支 FP32 累积、FP32 fractional 真实历史和 native_both front/noise。保留真实运动、控制、full K、帧数和分辨率；图重放、纹理复用及 GPU 栅栏接力已经存在。accepted_baseline 的 history hook 参数全 false，作共同审计准入；它没有安装 r15 Swin/ViT 全模块实验。入口证据：[FullsizeGameModes._select/安装链](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/nr_game_fullsize.py:501)、[运行帧](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/nr_game_fullsize.py:738)、[PHASE2-r15.json](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/PHASE2-r15.json:1049)。

证据分层：**C** = 本次可检查的固定源码/提交；**L** = 作者提交到仓库的日志或汇总，能独立阅读但本次没有重新运行；**A** = README/DevHistory 中作者报告，底层逐帧原序列未在审查目录出现；**I** = 基于代码的推导/待证伪预算。不将 A/L 写成我们实测。GitHub Releases API 返回空数组；仓库有 tag 和 README 指向网盘的 0.39 包，因此“没有 GitHub Release 对象”不等于“没有发布包”。当前树没有hip/src目录；设备代码主要在hip/*.hip与*.inc，宿主在src及Development/HIP。本次没有下载包、权重、巨大二进制，没有执行 GPU 或改动开发源码/游戏。

## 2. 1080p / 90fps 到底是哪一种速度

| 项目 | 固定 HEAD 可核查的实际口径 | 判断 |
|---|---|---|
| 发布版 0.39 | README 标日期 10-01；900 网络约 7.27→6.8ms，1080 约 10.05→9.5ms；剑星 2K 只报告 +1.5–2fps | A；不是 90fps 的游戏证明 |
| 1080 网络尺寸 | 1920×1152 默认，1920×1088 可选；ViT 两者都补到 32×20=640 tokens | C；1088 的收益不能记作内核同工作量收益 |
| 上采样前输入 | game render→网络→FSR→display；2K Quality 的 1707×961 输入自动落入 900 档 1600×960 | C；“2K/1080 显示”不自动等于 1080 网络 |
| 10-02 全网汇总 | full-N.txt：1080 三组 10.0291→9.8397、10.0163→9.8363、10.0311→9.8556ms；合并 10.0255→9.8439ms，p99 10.347→10.143 | L；是整网回放汇总，没有 Present/FG 证明 |
| wall 与 HIP span | wall.txt：1152 候选 wall 中位 9.988/10.014ms，span 9.460/9.464ms；1088 wall 9.684/9.704，span 9.153/9.162 | L；不同 harness/指标不要混算 |
| 10-03 最新默认 | all 71 blocks + FAST_NUMERIC=1；regular 仍开 ViT adaptive；MULTI_PASS 默认 1 | C；与 0.39 的 skip3/数值配方不同 |
| 10-02 跳块 | 默认 skip 42,43,46：71 个块图中的 3 块未算。归档旧配方的“full-network”不能译成“全 71 块” | C/A；必须跟 flags |
| 历史/运动 | regular pre-upscale 调用 reset=true；源码注释明确不采样历史/运动。RE9 caps history_supported=0、overlap_supported=0 | C；远弱于本地真实时序契约 |
| FG | Magpie 支持 optional XeSS FG；离线网络日志本身不包含 FG。没有 90fps claim 对应的 FG 开关/Present 原始日志 | C；不能排除传闻的显示帧包含 FG |
| 渲染重叠 | pre-upscale async 可拆宿主提交；网络仍竞争 GPU。另有 opt-in one-frame-behind OVERLAP，改变时延/帧对应；不是默认 90fps 的证明 | C；不能移植成偷用上一帧输出 |
| HDR | regular 接入 pre-tonemap buffer，有 codec/曝光/格式处理；不能由格式支持推断保留等价 HDR 动态范围。10-02 game probe 没记录完整 HDR/曝光/用户画质契约 | C/L；不可与本地 HDR/控制等价 |
| 更新的“全开” | DevHistory 10-03：fast numeric +1088 行、全块全算 9.29ms；加 AE 的运动序列 8.54ms | A；包含几何变化，后者还包含少算 |

引用：[README 配方与发布说明](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/README.md#L12-L80)、[几何与网络入口](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/HIP/hip_reference_network.h#L159)、[ViT 640 token 派发](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/HIP/hip_reference_network.h#L712)、[reset=true](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/src/native_pre_upscale.h#L220-L221)、[RE9 caps](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/src/LmxxfNrRuntime.cpp#L795-L796)、[重叠选项](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/src/native_game_frame.h#L153)、[原始归档汇总 full-N.txt](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/results/next-candidate-20261002/full-N.txt)、[wall.txt](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/results/next-candidate-20261002/wall.txt)。

**原始日志边界。** 10-02 game-probe-stellar-20261002.txt 有多组每 100 帧聚合的 encode/input/network/neural/decode/gpu_total/cpu_frame_ms，并反复出现 history=0、reset=100；例如 frames=900 的 network=9.446ms、gpu_total=9.7033ms、cpu_frame_ms=12.494。它不是逐 Present 帧时间，且多处测量会扰动队列。12.494ms 的倒数也不能直接当用户游戏 fps。full-N/wall 是汇总文本；作者在 DevHistory 描述的 1000 帧 CSV/槽原数组位于其 D:\\DLSSNR-Lab，不在这些已提交摘要中。本次没有获得完整原序列和对应截图/视频。可证“作者提交了这些数据”，不可证“我们独立复现了全部数值”。[game probe](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/results/outside-net-20261002/game-probe-stellar-20261002.txt)。

**ViT 复用不是普通历史融合。** AdaptiveVitGroup 跳过整个 31–38 组的计算，让各核在 reuse_gate 下 early-return。默认 mode1、global=.22、local=1、image=.35、period=4；token L1 相对锚点变化、最大局部比值和 RGB 32×32 tile 的连续 2×2 区域变化一起判定。复用输出为 anchor_out + gain[channel]×(current_in−anchor_in)，gain 是八块 contract/projection skip 的乘积；只在完整计算时更新锚点，并非每帧递归写新的近似锚点。通常最多连续复用 3 帧；若输入（含 padding/位模式）完全未变，可跨 period 继续复用。mode2 强制完整；mode3 更激进的固定周期复用。geometry/状态变化和空闲超时重置。现 HEAD 不允许此路径与 opt.graph、byte stream 或 ViT skip 混用；VIT-REUSE.md 的旧默认说明须让位于当前 flags/代码。[AdaptiveVitGroup:586–612](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/HIP/hip_reference_network.h#L586-L612)、[reuse_token_stats/reuse_decide/reuse_finish:1038–1061](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L1038-L1061)。本地保留 full frame/full calculation 的主线不采用它。

## 3. 从约 17ms 到约 10ms：关键跳变，不把异口径差值相加

| 日期/版本节点 | 内核结构或数学变化 | 作者可查数据 / 分类 |
|---|---|---|
| 09-23，7ef24e7 | prod6；已是 HIP、已存在量化/跳块配置 | NativeGameFrame 1080 约17.3ms；900约12.1ms。历史起点，不是今天 9.5ms span 的同口径基线 |
| 09-26 wave-owned | C32 整个 64-token 窗口 FFN→QKV→attention→projection 归一 wave；C64/C128 一 head 一 wave 完整块；C256 保留生产前段，仅 attention/projection 改 | 完整宿主同批1080 16.99276→15.87261ms（−1.12014，6.59%）；C32 window/hidden 滚动后 VGPR 从240上下到165/174，4KiB LDS。真实调度/数据归属收益 |
| 09-26 M32/N64 | C512 mix/QKV 同一 wave 同权重处理32 tokens；ViT projection 扩 N64，权重跨输出利用 | 各项整网约1%–1.9%；计算矩阵乘次数不减，供数/权重利用变 |
| 09-27 PACK8/输入布局 | 把成组 half/FP8 转换写成向量片段，改 wave-owned 输入坐标/RTZ 双值处理 | PACK8 两族组合作者报告整网−8.7%/−9.1%；0.35发布1080约12.716ms。是连续多次采用的累计，不仅换一条指令 |
| 09-28 FMA + 后续组织 | 激活 float FMA 改数学，随后融合/布局重排 | 单 FMA 1080 12.551→12.400；0.35→0.36同批12.715527→11.573710（−1.141817）。不能称这一步对0.35逐位 |
| 09-29 compact C512 | 只对有效栅格做 pointwise FFN/投影，window attention仍包含补边64 keys | 1080 11.57261→11.34437（−.22823）；该几何的13块统计点数33280→28080（−15.625%）。少算恒零边界，不是缩分辨率 |
| 09-29 C256 persistent | 固定层链的窗口就绪队列，写完窗口后发布至最多4个依赖子窗口 | 1080约−.064ms，900约−.16ms；C64/C128推广失败。依赖/派发优化，层间缓冲仍在 |
| 09-30 compact1088 | 1152→1088 行，ViT仍640 | 10.736→10.261（−.474ms）/10.764→10.283（−.481ms）。几何变化，另列有损 |
| 10-01 FUSEQKV/deep4 | C512 Q/K/V 同 K-loop 共用 A；边界指针/有效判断外提；四级片段环形预取，显式调度限制活跃变量 | 寄存器235→189；新一刀1080约0～.014ms，不能独占全天−.52ms |
| 10-01 FFN_ONE/F8W | C512 mix→expand→cubic→contract 寄存器内完成；真正 FP8 weights/WMMA 仅在输入/权重已是 E4M3 格点处采用 | ViT QKV 5-wave staged weight 也切 F8W；当天1080 span10.05→9.50–9.56ms。真实组织 + 硬件 FP8 供数/计算 |
| 10-01 Swin 5刀 | FFN 同权重连续喂4个16-token tile，C64/C128 QKV输入片段共用，Up/Down存储宽化/half/byte | 输出逐位是相对09-28作者黄金版本；没有增加新的跳块 |
| 10-02 compiler + outside IO | 部分模块 LLVM23/max-ilp 调度；修 fence；直接网络输入/输出去额外复制；post-signal query 促 Windows批次提交 | full-N1080约10.0255→9.8439ms；runtime整链 IO收益约.07–.09ms。GPU调度/IO，不能等同CPU包袱 |
| 10-02 负候选 | C512 FFN+projection 更大WG，LDS接力；地址偏移；部分RTZ候选 | FFN+proj虽省一个launch、8KiB中间LDS，整网反而约+.11～.21ms，未收。全融合不是单调更快 |
| 10-03 当前HEAD | FAST_NUMERIC + 全71块；可选叠层默认1，外推实验失败 | 相对旧skip3默认慢约.19ms/1080而画质改善；没有10-03全网再次大突破到“游戏90” |

关键提交：[wave-owned生产路径](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/6784ec0741d1efcef6206ebb0b827b66aa9f34a5)、[M32](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/17a95105ab41cb59f62bc18f3b8fb9d0875fa595)、[输入片段/RTZ](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/5be634cf9f0038deb3dfcae9c1e6e1e2fa79f5ec)、[批准FMA数学变化](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/328e1081b7d7d81da9fb82ba59b575a3d25e3edc)、[compact C512](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/9880d03016e08e5032cc743d7271b9afbfba1455)、[C256队列](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/3767f49a70fe57e6805673d5f2a62c72ffa723ea)。节点数据见 [DevHistory:434–502](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/DevHistory.md#L434-L502)、[累计/FMA/compact:927–1028](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/DevHistory.md#L927-L1028)、[10-01整网日报](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/results/day-summary-20261001.md#L3-L14)。没有逐族、同配方的全时段连续回放，无法给17.3→9.84的每一刀精确分摊。

10-01/02关键实现提交：[C512 QKV共用](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/e5069f0788f3398ea714a65a1b3db984f2bda79b)、[deep4调度](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/754acc445f3339f5c7dcbea248108a9a36ad39eb)、[FFN_ONE](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/e4c3ecb1c0949f5adef7891e3db04cfd9e94d7c8)、[FFN F8W](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/79114a61ad701af1d5fbd4252fc68c659677adde)、[ViT F8W](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/e22d15a242d661de76f28ca44452ec8aefe51ac2)、[FFN QT4](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/b37aa5b48231928d514b0542685640836bd1385d)、[max-ilp](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/82ce821f0ea1a12925d04c353cb2d1b9ee006c11)、[LLVM23](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/d48cdae3503cdf23e53bb646b6e6d0f428ce2c59)、[计时修复](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/a4c84272546a4ed712f9c747782b445780bd0061)、[outside IO](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/cc021a8774c0054c3fcd1beeb8d23a34e1b35ca4)、[post-signal query](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/87b745d1e77964999aa683d52e5b93028317107d)、[安装汇总](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/cc10337376bde2d0fc6e5e2fc0148acd6e8ae193)、[大融合负账](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/c49bdf1c5be857abc791d6105908cbf3b86d342f)、[今天fast数值](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/741e5e469427b3fb23834feb5d2e0e491d6619ff)、[今天全71默认](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/5f8c4212f56ebce58c29b66c202e65348d208dbf)。

## 4. 逐模块 producer→consumer 映射

“改成 HIP”不是移植方案。AMD wave32 的 WMMA 16×16×16、E4M3、目的 fragment 方向、VGPR 和 LDS 规则不能直接代入 Intel subgroup16 的 DPAS。B580 已有真实 INT8×INT8→INT32 DPAS；用该路径维持 FFN 整数合同。Swin/QKV/attention 的现役未量化 half 则用 FP16 DPAS，不为模仿 AMD 再量化。

| 模块 | AMD 当前关键归属 | 现役对应入口/差异 | Intel 映射/剩余缺口 |
|---|---|---|---|
| 前端/历史/prefix0 | cw_body 的 prefix 变体从噪声/历史构造输入，prefix MLP+attention+下采样闭合；输出按 byte/half 接后层 | numeric_cleanup front/noise、真实历史 sampler；FDP已有完整帧实测收益 | 借用 producer 直接写 consumer 所需布局；保留本地真实历史而不借 reset=true。FDP由Main继续，非本次新发明 |
| C32 ordinary 1–4/67–69 | 32线程一wave窗口，rolling hidden/window，4KiB残差LDS，AV直接供projection | seven ordinary 的 MLP独立；QKV pack已融合，attention+project已融合，但MLP和QKV仍全局落盘 | 新N3：整个64-token窗口归64/128线程WG，4个16-token query tile轮流消费；FP16阶段边界留在寄存器/SLM |
| C64/C128 | swin_wave2_body，head归wave，FFN QT4共用权重；Q/K/V同K-loop共享A，V用PV需要的方向 | MLP expand/reduce已有 fused，C128 pairwise 已接受；QKV按head×family独立CTA；tail已全输出C64/C128融合 | 新N4共用A；新N5跨4个token tile共享W。保留已融合tail，不重写成慢的全输出大tile |
| C256 | 全融合寄存器压力不划算；保留前段+局部attention，6层链可persistent | 现役direct-pack与窗口attention；Main的split-tail修复针对未收实验 | 新N6只尝试窗口就绪队列；不把r15输出列重复修复称新优化 |
| C512 FFN | one-wave mix→8组expand/cubic/reduce，结果byte直接供projection；native FP8仅格点处 | rows_scopes.c512_ffn：INT8 entry/linear→_groups已做expand/cubic/contract闭合→project；有效960行 | mix→groups间仍有buffer，但INT8动态量化/scale与AMD不同。先不盲做FFN+proj巨WG；AMD负账已说明不能只数launch |
| C512 QKV/attention | compact valid raster→窗口全64key；QKV同A三路，deep4；window内attention | 已用单个torch.mm计算拼接1536输出；输入却先pad；decoder投影已从packed window消费 | 新N1只改输入有效行和pack，留库GEMM。不能重复提出“3次QKV GEMM合1次” |
| ViT FFN31–38 | 1024→4096→1024，部分weights/输入格点nativeFP8，流式合同片段 | 现役真实INT8，entry独立行量化一次、P4 contract；新K64/行量化一次正由Main合并 | 修复附注R1；不算N1–N6新增，不建议FP16→INT8“升级” |
| ViT QKV/attention/projection | 5 waves同head跨80tokens共享weight；专用bit-map exponent在kernel内供PV/分母；输出供projection | fullsize_session.vforward实际调用两K512 GEMM→norm→全局exponent矩阵→4个K256投影half merge；240token专用head候选未默认安装 | 新N2完整尾部：保留两个独立half QKV积、64key分母序、padded16纠偏、P4投影顺序，在头布局内消费 |
| Up/Down/post70 | half/byte producer格式对应consumer、尾存储向量化；部分up与body合并 | decoder gather、nativeK8、post/pre现有融合及FDP；纹理复用/fence已经有 | 只做剩余实际buffer边界；不用AMDbyte压缩覆盖本地unrounded half数值 |
| 外围桥接 | direct IO去冗余拷贝，D3D→HIP→D3D fence；postsignalquery推动Windows提交 | 现役纹理复用、图重放、GPU栅栏接力已经存在 | 不是重做共享纹理或CPU等待。只有证明本地确有多余copy/submit才再做IO实验 |

AMD源：[cw_body:206/exports:531](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L206)、[swin_wave2_body:258、FFN QT4:418、QKV:563](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L258)、[C512_ONE:151/242](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/c512_m32_deep.inc#L151)、[compact QKV:96–155](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/c512_qkv_attention_compact.inc#L96-L155)、[ViT W5:90/152](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/vit_stream.inc#L90)、[ViT流式attention:363](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L363)、[persistent:38–103](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/swin_persistent.inc#L38-L103)、[HIP桥](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/HIP/hip_d3d12_bridge.h#L215)。

本地对应：[ordinary C32完整forward](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/c32_repeated_tail_fusion_v1.py:34)；[C64当前QKV](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/c64_qkv_direct_pack_v2.py:31)；[C128当前QKV](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/c128_qkv_direct_pack_all_v1.py:63)；[C256当前QKV](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/c256_qkv_direct_pack_one_v1.py:30)；[C512 FFN真实行数/调用链](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/rows_scopes_v1.py:23)；[C512 fused groups](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/c512_int8_ffn_rows_v1.py:106)；[ViT当前forward](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/fullsize_session_v1.py:131)；[C128已接受FP32 branch accum](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/branch_accum_native_720_kernel_v1.py:11)。

### 线程/布局的可执行约束

INT8的现有实际IR是 i8×i8→i32，IntelDPAS repeatCount=8、systolicDepth=8、executionSize=16、opsPerChan=4、threadsPerWarp=16。不是浮点乘后装成INT8。FP32 row scale、激活、残差仍存在，但不否定矩阵指令真实为INT8。新FFN排布可从K32的A[16,32]、B[32,16]、C[16,16]切片构造；K64应是两个有序K32片段，保持scale/half/cubic边界。FP16阶段按其实际DPAS lower结果，不能引用INT8的opsPerChan=4证明FP16布局。

AMD “每head一wave”在Intel上对应独立head的小tile和明确的SG角色，而非强求64个token全塞一个SG。初始64线程=4个SG16；按query16/32和column16/32分工。SLM只跨需共享的K/V/权重/残差；明确barrier和workgroup memory fence；host/kernel末尾到下游的device发布由现有图/fence契约承接。AMD的s_barrier、s_wait_dscnt、schedule_barrier不可原样替换；不同编译器导致AMD LDS可见性回归是代码级教训，不是Intel可盲用的优化flag。

权重列归属也不同：AMD W2_QKV_FUSE用part*C+head*32+channel组织Q/K/V；本地direct-pack和ViT head-major producer的列是head*96+family*32+channel。Main预打包时必须按实际权重语义重排，保留各自pixel_order/inverse的栅格→window token映射，不能照抄AMD地址公式。结构首轮的half边界只用于隔离收益；用户快速版允许其后改原生完整K累积/归约。

## 5. 六项相对现役的新收益候选

排序综合潜在收益、实现复杂度和隔离难度。以下ms都是 **I：立项预算**，没有本次GPU测量；正向区间包括零，可能回退，不能相加或换算成游戏fps。没有对应阶段当前实测耗时，不能伪装成roofline定量预测。先做小范围组件筛选，再用同一完整帧流程推翻/保留。Main新合并臂可以作为另一个配对基线，但主表必须同时报 accepted_baseline；不能把消除r15回退的8–9ms当新增收益。

### N1（优先1）：C512有效栅格 QKV，保留库 GEMM 和全窗口 K

**新增性：相对现役新增。** 旧compact_c512_qkv_stack_v1确有原型，但 fullsize_session.installed:148明确排除了stack.compact_queries；本地720编码器仍pad后attention，decoder的现役wrapper也pad后libraryQKV。因此源码存在紧凑原型不等于720已启用。证据：[编码器当前先pad](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/fullsize_session_v1.py:111)、[排除compact_queries](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/fullsize_session_v1.py:148)、[decoder先pad](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/c512_window_projection_game_v1.py:99)、[单torch.mm，已拼接QKV](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/c512_qkv_library_16_v1.py:37)。AMD对应[compact提交](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/9880d03016e08e5032cc743d7271b9afbfba1455)。

**修改界面。** Main工作副本新增compact_c512_qkv_pack_720模块；接fullsize_session_v1.c512和c512_window_projection_game_v1._packed_with_baseline_probabilities，或增加专用有效栅格入口；c512_qkv_library_16_v1提供同一library multiply。输入MLP[24,40,512] contiguous half、W[512,1536]；一次mm得到[960,16,3,32]，pack同时按shift/pixel_order映射到[16,HP/8,WP/8,64,32]的Q/K/V；越界token按当前pad→GEMM→norm的结果填零/保持零位语义。沿用native概率unround、bias及decoder packed projection。编码器最后池化块输出、pool求和顺序、skip不动。

**真实算量。** 四shift对应padded QKV行数960/1536/1152/1280，每种在16块中各4次：19712→15360，少4352行，**QKV GEMM MAC少3,422,552,064（22.08%，约6.845GFLOP）**。已是960行的FFN不再算一份“compact收益”；未跳attention padding keys。QKV中间写57.75→45MiB，pack读也少12.75MiB，合计最少25.5MiB逻辑流量；pad输入额外4.25MiB写/读可避免。最终窗口Q/K/V大小保持，不扣除其读写。不是缩小720画布。

**线程/DPAS。** mm继续库实现。只pack用SG16，按(head,window,token16,channel32)安排，真实token从960行读，invalid从确定零语义生成；不造C512 fullBN1536 Triton核。library改M可能改变tile/归约，必须全帧比对，不能假设torch.mm不同M逐位自然一致。

**预算/证伪。** 0～0.8ms，复杂度中低；物理DRAM节省可能被cache遮蔽，GEMM M变小还可能换到较差实现。最小Luna：仅此开关、全部16块，原13帧/历史/控制完整序列，ABBA；记录实际mm M、16块命中，13输出/内部历史差异、process和raw GPU。若确认4352行消失而完整帧无重复收益，停止扩大；不能靠单GEMM benchmark采用。

**结构收益通过后，原生fast数学后续：** 库GEMM已经完整K512，不再造FP8模拟舍入；可单独测试native norm/概率分母的浮点归约。现有完整C512 native encoder实验未纳入accepted，这类数学改动另报收益与时序画质。

### N2（优先2）：ViT完整 QKV→流式attention→head-major projection

**新增性：相对现役新增；与Main FFN K64/行量化一次互相独立。** 240-token的vit_head_dataflow_720_v1是未默认采用的骨架，其attention仍调用vit.vit_attention，指数矩阵照常落盘；audit_vit_native_dataflow也仍物化指数矩阵。应复用已有QKV/P4投影代码，新增stream64 consumer，不重做FFN。证据：[当前两GEMM/240token尾部](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/fullsize_session_v1.py:131)、[旧head候选仍调用原attention](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/vit_head_dataflow_720_v1.py:149)、[当前special exponent与纠偏](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/experimental/fp8_unround_overlay/nr_backend/vit_block.py:32)。AMD对应[W5 weight staging](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/vit_stream.inc#L90-L144)、[stream attention](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L363-L381)。

**接口/布局。** 新vit_tail_stream64_720(module,mlp[240,1024],out)接vforward的FFN之后。QKV kernel保留两个K512各自FP32累积→half，再half相加；寄存器norm后直写Q[32,240,32]、K/V[32,256,32]，末16keys生成zeros。中间不再写z或token-major大矩阵。attention按(head,query tile16/32)CTA，加载Q一次，循环4个key64块；QK累积/half→模型bit-map指数，寄存器或SLM产生PV所需operand，同时按原64项树归约累积分母。末16个padding仍贡献exp(0)，最后按原half边界减1.34375，而不是提前mask。分子维持原K32累计/舍入次序；别替换成标准softmax或flash_attention。输出head-major[32,240,32]，projection直接读它，四个K256各自half边界及part0 residual初值、顺序half merge保持。

**真正减少的工作。** 矩阵MAC不减少；减少整矩阵暂存及归约/merge launch。单ViT块score/exponent[32,240,256]为3.75MiB；消除score写+exp读至少7.5MiB，exp写+PV读至少7.5MiB，分母还需再读exp至少3.75MiB：**至少18.75MiB/块、八块150MiB逻辑流量**，不是相同数字的DRAM实测。两QKV积各1.40625MiB，两者写/merge读至少5.625MiB/块可在寄存器闭合；QKV norm及projection临时另列，不重复计入150MiB。现役FusedVitProjection_v3已把四个数学K256分区收进parts+merge两核，不能当成四个独立GPU launch；partial[4,240,1024]仍落盘，每份0.46875MiB。新增是头布局直读和这份partial/merge消除，K1024和全部240tokens保持。

**线程/DPAS。** QKV初始64/128线程，BM16；按head×token组让2～4个tile共享SLM的K32×N32 weight，采用两级prefetch；不硬搬AMD5-wave=160线程。attention初始64线程BM16、key64，FP16 DPAS QK/PV，16-key或32-key fragment细分；只保留当前score片段，不寄存整个240×256。projection BM16/BN32/64，按head-index计算地址，不做heads.contiguous()+transpose reshape的全局复制。动态NaN/符号零语义沿用现有helper。

**预算/证伪。** 0～1.5ms，复杂度中高；过长单CTA的串行链或高SLM会慢。最小Luna两臂：先仅stream attention替换，其次完整head tail；全八块、13帧输出/历史，ABBA，GPU捕获确认没有score/exponent全局buffer。若后者无收益，把QKV staging/projection单独留作完整帧配对；不把“没有exp tensor”当速度验收。

**结构收益通过后，原生fast数学后续：** 首轮两个K512/P4 half边界只是隔离结构收益；随后允许QKV完整K1024单累加、projection完整K1024单累加及native norm/exponent、FP32分母。[audit_vit_native_dataflow_720_v1.py:161](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/audit_vit_native_dataflow_720_v1.py:161)已有full_qkv/full_projection/native_norm/native_exp和fp32_reduction入口，[vit_projection_single_accum_720_v1.py:22](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/vit_projection_single_accum_720_v1.py:22)已有独立单累加投影；均未纳入现役accepted，不能视作已收收益，也不恢复FP8模拟舍入。

### N3（优先3）：C32完整窗口 MLP→QKV→attention→projection

**新增性：相对现役新增。** 现役只在尾部融合，MLP、QKV和tail三个阶段仍跨global；已有C32 native hidden不等于整个block驻留。先覆盖seven ordinary块，decoder66随后与Main FDP/gather分工协调；pre0/post70的控制/历史/RGB链不在首轮改。AMD完整窗口结构是17ms→10ms过程中重要且相对可移植的一刀，不是“把已有attention+project再融合一次”。证据：[三段现役forward](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/c32_repeated_tail_fusion_v1.py:45)；AMD [cw_body与exports](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L206)、[rolling window资源变化](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/DevHistory.md#L460-L482)。

**实现。** 新c32_window_resident_720_v1._block取HWC half输入、MLP expand/contract、QKV、bias、output weight、skip、shift/pixel_order，写HWC half输出；同一WG拥有64token。先从原features按shift gather输入，不分配F.pad；MLP hidden以32-channel滚动，保留cubic/contract/残差half点；MLP residual留SLM，QKV norm在register，K/V存SLM，query按16/32tile依次消费AV并立即project；最终crop store HWC。全部64key（包括补边的原结果）保留，MLP padding区的计算不能随便以“输入是零”删掉后续语义。

**算量/读写。** MAC不减少。仅省MLP边界写读和3路QKV边界写读，下限16×M×C bytes；C32普通有效M=384×640=245760时每块120MiB、七块840MiB逻辑边界流量（补边额外另算）；不保证对应显存总线同等减少。此前hidden已经fused，不能再计其巨tensor消失收益。SLM粗预算64×32×2=4KiB residual，三路64×32×2=12KiB，合16KiB；Q按tile计算可以减少，但需给权重prefetch预留，不将hidden128全展开到register。

**线程/DPAS。** 初始64线程/4SG16，16token query与16输出col fragment滚动；并测128线程分工。FP16 DPAS保持原K顺序，每个SG的fragment按lower后布局安排；跨SG K/V需要明确SLM barrier。不要把AMD无组barrier的单wave假设搬到4SG。输出layout暂保持现役HWC，不连带改层间格式。

**预算/证伪。** 0～2.5ms，复杂度高；寄存器、SLM和同步可能吃掉全部收益。Luna首个完整帧实验覆盖七块，只替换普通C32入口，记录7实际site/帧、输出/历史及raw/process；BM16/32、64/128线程只保留少量合理配置。单块成功之后才扩大decoder66，并避开FDP已融合的消费者，整帧对照不能只测one-window。

**结构收益通过后，原生fast数学后续：** 可把C32 native cubic/norm/分母归约作为独立fast臂，减少旧half拆分边界；保留全部K及真实帧输入。audit_swin_720_kernels已有FP32 norm/weights候选，当前完整实验未采纳，结构测量与数学测量分开。

### N4（优先4）：C64/C128 QKV同输入片段，分阶段改consumer布局

**新增性：相对现役新增。** direct-pack已省掉未归一化z中间buffer，但CTA仍按head×family分开：C64的program_id(1)决定family后独立读取X；C128的grid=头数×3同样如此。现役c128_dual_qkv=false。nr_game_fullsize.py:576的可选loader引用qkv_dual_segment_c128_family_720_v1，但immutable r15/source及其source_files不包含该模块，不能据此声称已验证dual实现或可直接开启；本项目标是有明确源码证据的小BM三路A共用+consumer匹配，避免多路大tile的live accumulator压力。[C64 family分工及load](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/c64_qkv_direct_pack_v2.py:31)、[C128 family分工](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/c128_qkv_direct_pack_one_v1.py:30)、[C128 grid](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/c128_qkv_direct_pack_all_v1.py:63)；AMD [W2_QKV_FUSE:563–600](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L563-L600)、[采用C64/C128，未采用C256的提交](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/96efbc3db4bf89698cf09b75dbf8f67ab77160e7)。

**接口/布局。** Main新增qkv_shared_a_720_v1._qkv3(X,W,scale,inverse,Q,K,V,M,Wp,C,BM=16)；接现役c64_qkv_direct_pack_all_v1.direct_pack与c128_qkv_direct_pack_all_v1.direct_pack，输出先完全沿用[head,HP/8,WP/8,64,32]。每(head,row tile)一次K32 load A，同时更新Q/K/V三个BM×32独立FP32 acc；每个acc的K顺序/half/norm/queryscale不变，Q/K归一化按现役helper。K32 W片段预打包/双缓冲只改变供数。**第一轮不碰已采用C64/C128 fused tail**，所以能判断新增A复用本身。

第二阶段有明确ISA证据再试V[head,window,channel32,token64]转置布局，改c64/c128_attention_project_fused kernel的V地址与consumer operand，使PV不用重复布局转换；metadata明确V的stride/layout，Q/K仍原布局。不要混用新V和旧consumer。AMD用反向WMMA直接产生V operand，但Intel tl.dot的fragment和store策略未必同样受益；若SLM转置/全局窄写增加，保持第一阶段即可。

**真正变化。** QKV MAC和W数量不减；同一个head的X矩阵逻辑读取3→1。不计pad时，8个C64块M61440、12个C128块M15360，按4×M×C×heads bytes估算**约600MiB/帧逻辑输入读取可省**；跨family/head的cache可能使实际DRAM节省远小于此。Q/K/V输出数量不变，现役已经没有raw z，不能再加一次“z buffer收益”。V阶段只有ISA确认转换/transaction变少才能申报额外收益。

**线程/DPAS。** 64线程4SG16，BM16、N32，三acc滚动；C128可先Q+K共用、V单独作资源对照，随后再测三路。K32顺序不变；norm尾只在K-loop结束后存活，尽早释放load/pointer向量；短预取2级起步，不机械复制deep4。C256暂不推广，AMD已有寄存器负账。

**预算/证伪。** 0～0.6ms，复杂度中；多acc可能压低occupancy。Luna两独立完整帧臂：仅C64三路、再C128三路，记录8/12实际site、IR/ISA中共用load及barrier资源、全部13输出/历史；ABBA。第二阶段V布局只在第一阶段有重复收益后启动，与相同第一阶段配对，仍同时报现役基线。

**结构收益通过后，原生fast数学后续：** QKV先同K序以隔离A复用，随后可单独开启native FP32 norm/概率分母或其他完整K累积；[audit_swin_720_kernels_v1.py:183](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/audit_swin_720_kernels_v1.py:183)与[audit_swin_720_kernels_v1.py:250](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/audit_swin_720_kernels_v1.py:250)已有FP32选项，未成为当前accepted。旧half拆分不构成永久逐位要求。

### N5（优先5）：Swin MLP跨4个token tile共享weights，保留现役pairwise和branch累积

**新增性：相对现役新增；不是Main gated-C64/BM8修复的重复任务。** 现役C64的batched._pairs和C128的_pairs_pairwise已经在register做expand→cubic→reduce。C128 pairwise把每次A喂两份W，也已接受。剩余是不同M tile重复取同一组expand/reduce weights。AMD W2_FFN_QT_BATCH=4让4个16-token tile轮流/并行消费同一weight fragment，隐藏维只保持1～2个tile。[C64当前_pairs](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/experimental/fp8_unround_overlay/modules/batched_branched_mlp_v1.py:22)、[C128已有pairwise](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/branched_mlp_pairwise_720_v1.py:35)；AMD [QT4生产者/消费者:418–444](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L418-L444)、[采用QT4提交](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/b37aa5b48231928d514b0542685640836bd1385d)。

**修改。** 新mlp_weight_group4_720._pairs_group4；(branch,rowgroup64)WG覆盖4×16真实行。4个SG16各有独立row tile的acc，协作读K32×N32 W到双缓冲SLM、各自DPAS消费；保持原part 0,1,2,3顺序，C128仍pairwise的两acc复用A；reduce先后次序、cubichalf、unround以及LATENT[branch,M,32]布局不动。沿用accepted branch projection；特别是C128的branch_accum_native_720_kernel_v1已经跨branch保持FP32，本项不能退回旧逐branch half merge。C64保持自己的现役half边界。

**算量/流量。** 不减少MAC，不去掉LATENT，不削K。每branch每16行原本读取expand C×128×2 bytes、reduce128×32×2 bytes；理想4tile一组后，这两类W逻辑global请求为原来的1/4（尾组另算），X/输出流量不变，并增加SLM写/读和barrier。cache可能已经复用了W，且AMD单wave寄存器共享比Intel跨SG SLM更便宜，不能直接期待4倍内核速度。

**线程/DPAS。** 初始64线程/4SG，各16行；N32拆16-column DPAS fragment，W缓冲每K32每part32约2KiB，pair两份4KiB、两级约8KiB。别提高BM到64然后把全hidden128向量展开给一个SG。Main新C64 gate稳定后再派独立worker做此跨tileW合同，防止混入BM8修复收益。

**预算/证伪。** 0～0.6ms，复杂度中高。Luna先C128十二块（已有pairwise主对照）only，再C64八块；保持project kernel原哈希，完整13帧ABBA。统计W的load指令/SLM bytes、资源和raw/process；若W请求少但整帧无重复收益，判cache/同步抵消，不再扩大C256。

**结构收益通过后，原生fast数学后续：** C128跨branch FP32已在现役，不再算新增；branch_accum_native_720_v1已有C64/C256扩展开关，当前constructor仅启用c128。其他家族可另测完整分支单累加/native cubic，full K不减，不重新加FP8模拟舍入。

### N6（优先6，暂后置）：C256窗口就绪队列，做GPU依赖收益实验

**新增性：相对现役新增，但立项把握最低。** 现役graph已消掉大部分CPU逐launch成本；AMD队列的真实可借点是producer某个window完成后即可启动consumer依赖窗口，降低全层dispatch边界的GPU尾巴，并非再造图重放。AMD只在C256中间链取得约.064ms/1080，推广C64/C128反而失败。引用 [sp_run_body:38–103](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/swin_persistent.inc#L38-L103)、[sp_global_group_sync:14](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/swin_persistent.inc#L14)、[采用提交](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/3767f49a70fe57e6805673d5f2a62c72ffa723ea)。

**接口/实现边界。** 新c256_window_queue_720_v1(queue,counts,nodes,layer_in/out,weights,status,epoch)作为可选graph body；先两块普通C256，最终才考虑encoder16–21/decoder49–54六块链。任务描述包含layer/stage/window，shift依赖预计算为最多4个上一层窗口；window×stage的device body维持现役前段及split tail资源选择，不强制C256全融合巨tile。先需要可复用的device body，当前没有现成可直接切换的Intel persistent module；此项不是“一行开启PDL”。

producer所有SG写完实际output后，workgroup fence/barrier，然后device release发布ready；consumer acquire读ready。全局队列仅执行已就绪任务，不能占满全部驻留WG后等待尚未运行的producer。epoch、计数复位在graph执行完的边界，timeout/异常状态触发整帧失败/安全回退，不能把旧frame buffer当成功输出。所有层输出仍保留global，真实history的换帧/fence不由该队列私自接管。

**算量/流量。** MAC、层间读写不減，还新增队列/atomic流量；只可能缩短GPU依赖尾巴/减少dispatch边界。AMD删18/6个launch的数字与我们的物理launch数不通用；capture逻辑counter不代表GPU launch。Intel graph中是否存在可省尾巴尚无本次证据。

**预算/证伪。** 0～0.25ms，复杂度很高；可能更慢，建议排在N1–N5之后。Luna仅“两块”完整帧臂与同两块普通图路径配对，13帧、多epoch与reset，超时/毒化guard一次；若raw GPU毫无重复改善，不推广六块。不能以CPU减少计数判采用。

**结构收益通过后，原生fast数学后续：** 队列先复用已资格device body隔离依赖收益；后续可换成经过完整帧验证的原生fast数学body，队列没有坚持旧half边界的数值要求。分别记录body数学收益与queue收益，真实history/motion/control/epoch契约保持。

## 6. 已有实现、不能照搬的内容和门控证据

**已有，不重做：** C512单concat QKV library GEMM；C512 INT8 _groups的expand→cubic→contract；ViT真实INT8与现役独立行量化；C32 hidden native；C32/C64/C128 attention→projection；C128 pairwise与FP32 branch sum；decoder gather unround、full nativeK8；前端真实历史/运动/控制；图重放、纹理复用、GPU栅栏接力。FDP重复完整帧45.410→43.610ms（1.800ms）是已存在的离线新收益证据，仍未装游戏；由Main继续，不能被AMD报告包装为新发现，更不能预言RTSS必减1.8ms。

**AMD FP8限定。** FFN_F8W/ViT_F8W在E4M3输入和weights格点测试下转换不改变数值；不是任意FP16网络免费换FP8。本地接受去FP8模拟舍入，很多activation正是非格点，原生FP8会改变数学；B580矩阵已经INT8 DPAS也不会自动得到“再切FP8两倍速度”。AMD窗口链byte储存“exact”的原因是它原数学定义会量化，不能用1-byte替换本地unrounded half；可以借fragment/rolling/prefetch，不能借其量化值。FMA/RTZ、FP16累加、快速norm/分母必须作为单独numeric fast arm记录，不夹带进布局收益。

**具体资源硬门。** [spill_preflight.select](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/experimental/fp8_unround_overlay/modules/spill_preflight_v1.py:1)逐配置warmup，遇到首个n_spills==0立即返回，其他配置性能不比较。[Swin preflight](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/audit_swin_720_v1.py:489)枚举BM32再16以及BN，[launch](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/audit_swin_720_v1.py:561)拒绝非零spill。r15真实C256 tail选择BM16/BN128：BN256约33KiB spill，BN128/BM32也有832/896B spill，故选首个zero；zero配置可达256寄存器，仍可能低occupancy。**这证明硬门限制搜索空间，不证明某个有spill配置一定更快。** 冷端可以保留小而明确的资源候选，允许独立fast实验测少量spill并以完整帧质量/速度验收；不随意全开高spill。主线Main已修split-tail/gated-C64，须避免撤销其选择。

**逐位与审计包装。** AMD9-28已批准FMA变更，10-03fast default也明确有损；其逐位目标相对作者自己的黄金版本，不是4060精确线。我们的主对照已经接受去FP8舍入/C128 FP32分支，无理由把未采用实验的逐位规则上升为“所有新fast path永久逐位”。用户快速版明确允许数值变化；结构首轮沿用现役数学是为了隔离收益，不是永久保留旧half拆分累加。full-K单累加、native norm/exponent/分母可以后续直接作原生fast臂，保留full K、真实motion/history/control和全部帧，按完整时序质量与重复性能采用；不重新加入FP8模拟舍入。若库M/tile/归约导致差异，记录差异和全部时序画质，由Main按当前fast策略判断；不能用输出逐位一样解释“矩阵INT8是假”。

[每帧owner检查入口](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/nr_game_fullsize.py:744)确会validate_frame_context；[history hook检查](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r15/source/game/audit_history_host_720_v1.py:101)转调validate，图capture会有准入/计数。代码存在host检查，不足以证明55–60ms主要耗在审计CPU。graph replay不会重走模型Pythonlaunch链；local ViT/Swin实验raw GPU也变慢，已排除“全是CPU”的归因。**没有查到需要为快路径移除现役每帧权重SHA重算的证据，也不提出删history/control guard作为加速方案。** 冷编译并行可用CPU16路、48GB；不纳入GPU帧收益。

**计时修复占篇幅仅到此。** AMD旧event立刻query导致数值坍塌：fix-go.txt中1080旧med约.079–.093ms、新约9.28–9.29ms，输出hash同；这是纠错，不是额外提速。postsignalquery在Windows HIP促提交是平台特例，不等于Intel添加host同步能快。[timing证据](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/results/net-timing-20261002/fix-go.txt)、[outside-net分析](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/results/outside-net-20261002/README.md)。六项新增方案的收益只能由Luna完整帧验证，不靠计时读数选项。

## 7. 对Main/Luna的最小交接与验收

唯一当前写集是本review目录；上面的修改文件均指Main未来工作副本，**immutable r15及当前游戏本次均未改**。PLAN.json同时给accepted主对照、每项完整接口/布局/计算/预算、既有实现和R1/R2修复附注。

统一实验：固定PHASE1素材/13标准帧/720p/真实history、motion、控制、fullK；同一accepted constructor，唯一candidate toggle；cold编译不计时，每臂独立模型/graph/状态；先正确性完整13帧和私有history，再ABBA四槽完整帧计时，使用现有raw GPU事件与process两口径；有正向信号再BAAB复核。记录原数组、均值/中位/尾部、baseline/candidate commit/source hash、kernel资源/IR、真实命中/graph replay数；输出读回/哈希不夹在计时body。没有1ms门：任意可重复正收益都可以保留。数值变化已获用户快速版授权，不以新方案必须永远逐位相同为门；若输出改变，完整运动/历史画质审查是后续采用条件，不能拿单静态PSNR替代；游戏部署和用户视觉复核由Main在本次约束之外另行处理。

若Main K64/Swin merge成为已资格新基线，再分别做“accepted vs新增”“新基线 vs新增”配对，避免把两者合并收益归因给AMD某一刀。N1/N2/N3潜在buffer收益不独立可加；FDP也可能覆盖部分前后端消费者，不再双计。

**另列有损候选，不是上述内核收益：** 降网络输入/削画布（含1088）、跳块、ViT跨帧近似复用、上一帧输出OVERLAP、FG、byte重新量化、改变half归约/快速数值。可以解释AMD为何更快或作为未来用户选项，但违反本次保留真实history/fullK/fullframe主线时不进入N1–N6。

附注R1/R2（用户最新消息已由开发代理修复、Main正合并，非新增任务）：r15融合ViT entry原按64个N tile重算rowmax/quant，P1长串行contract；r15 Swin C256 attention随两个N128输出tile重复、C64 tile资源不佳。原完整实验分别45.94→55.12、46.01→54.63ms；只说明这些候选退化，不能代表现役，更不能把恢复速度算为相对现役−8/−9ms。ViT13输出/历史逐位且IR真实INT8；Swin最大差.00488未采用。详见本地R15_FULL_MODULE_MAIN_REVIEW、VIT_INT8_ACTUAL_DPAS_MAIN及FDP_REPEAT_MAIN_REVIEW，EVIDENCE.json提供哈希。用户最新更新：Main已修第33项unsupported JIT In→or，r18已冻结交Luna；本报告收尾时新修复尚未测速。报告不重派这些开发工作。

本次结论的实用优先级：**先有效行QKV，再ViT完整尾部；另行安排C32窗口闭合。** AMD约9–10ms的原因已可从结构和配方解释，游戏90fps的严格口径仍未有原始证据确立。
