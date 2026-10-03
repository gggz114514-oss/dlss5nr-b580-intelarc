# NR 调参功能执行记录

2026-09-14；用户指令“你来跑吧”，随后收口为“先把官方支持的功能加上就行”。本轮只推进已有原生参考的六项NR控制及精确/快速产品接入；社区额外合成、多轮、工作倍率等暂缓，以下社区表仅保留研究记录，不是待实现清单。六项不代表已穷尽官方接口或获得官方支持承诺。性能优化仍暂停，尚未替换已安装节点或发布。

## 已落地的候选

### 新进程缓存复用验收（两轮通过）

主任务已核对Luna报告及两轮原始cache-audit、validation和lease JSON：独立PID16128/8580，每轮175/175内核group命中、4/4辅助模块命中，blocked=0；26/26显示/历史/seed通过，退出码0。首帧分别1.911秒、1.709秒，全测试进程分别33.147秒、28.717秒（含预检、加载和多会话验证，不能当作26帧纯推理时间）。旧首帧约388秒等待未重现。

结论：当前机器、当前工具链和缓存、当前256×256六项控制测试覆盖内，新进程可直接复用Triton缓存，无需重新执行被守卫的编译路径。该结论不覆盖换机器/驱动/工具链、清缓存、新尺寸或未测参数分支，亦不证明驱动内部没有JIT。研究级缓存阻塞已解除，下一步可继续产品接线；可分发运行包仍需独立验收。

用户要求先确认重启进程的缓存复用，再接节点。新增 `reference/validate_controls_restart_cache_v1.py`，仅当前验证进程安装守卫：kernel metadata group缺失、native helper build、cache put均报错；不改已安装Triton。记录实际缓存目录、group与helper读取命中、PID、首帧与后续耗时，复用13组26帧原生字节验收。源码已过CPU语法检查。

已交同一Luna max：依次启动两个独立Python进程，第一轮成功才开始第二轮，每轮lease限900秒且忙时立即退出。输出 `D:/Codex-NR-Experiments/nr-b580/controls-session-v1/restart-01`、`restart-02`；报告 `luna-restart-cache-report.md`。两轮均需cache-audit passed、无blocked、全部命中和26/26数值通过，不能仅凭耗时或rc0宣布成功。守卫不拦截驱动内部JIT/二进制加载，所以必须同时报告冷启动耗时。失败由主任务诊断，Luna不得解除守卫或重试。

**最新结果：run-02通过，主任务已复核完成报告和原始JSON。** `run-02/validation.json` 的26/26帧显示字节、私有历史字节、seed全部一致，`passed=true`，lease退出码0。六项固定控制在本轮256×256、零深度范围通过；产品节点接线和其他声明范围仍未验收。

首帧计时387.999秒，其余25帧合计7.239秒。Luna记录首帧等待期间Python进程CPU累计时间持续增加，之后正常完成；没有观察到独立编译器进程或所检查缓存的新写入。因此能确认首次执行准备成本集中在首帧，不能仅凭这份记录认定全部耗时都是编译、已经落盘缓存或下次启动不会重现。性能优化仍暂停；产品接入时需确保新增路径满足预编译运行合同。

完成报告：`D:/Codex-NR-Experiments/nr-b580/controls-session-v1/luna-report-02.md`。下文“尚未收到run-02结论”是此前派发时记录，已由本段结果取代。

- `../nr-b580-int8/product/nr_exact_controls_candidate_v1.py`：复用原精确 Session 的执行、尺寸、失败和关闭合同，创建 `ControlledMotionNR`。固定六项控制，拒绝未知字段和错误类型；`skin_structure=null` 表示继承，不接受研究层不支持的负数。旧产品入口没有改变。
- `reference/validate_controls_session_v1.py`：13 组 × 2 帧（重置帧与相邻时序帧），直接比较认证的原生显示和私有历史；检查 seed、调用方改显示结果不污染历史。参考数据只用于比较，不进入模型历史。
- 配置：默认、auto、auto 下 skin/tone/structure 各 0/2、intensity 0/0.5/2、style 1/2。前两帧原始输入、motion、零 depth、capture、原生输出均按采集清单验证哈希；style 私有输出按 peripheral manifest 验证。Intensity 私有参考使用原生 default 序列。
- CPU 预检通过：`D:/Codex-NR-Experiments/nr-b580/controls-session-v1/preflight-01/validation.json`。只认证此次使用的档案，不重读巨大 arena trace，不将 CPU 通过称为当前 GPU 通过。
- GPU 验证已交用户指定的 Luna max 单次执行，agent `01a09d8e-00e0-73f3-9fe7-54e81b346d34`。lease 限时1200秒、忙时不等待、不抢占；Luna只执行监控并上报，不能修复或重跑。主任务不轮询 GPU。
- 预定结果：`D:/Codex-NR-Experiments/nr-b580/controls-session-v1/run-01/validation.json`；日志 `run-01.log`；Luna报告 `luna-report-01.md`。此记录写入时尚未收到 GPU 结论，不预填通过。

run-01已由Luna报告失败：0/26帧、rc1，进程退出。主任务读取 traceback 确认是 Triton `driver.py` 使用默认GBK读取 `driver.c` 导致 UnicodeDecodeError，不是模型数值不一致。启动命令漏设UTF-8；使用相同Python `-X utf8` 的CPU定点读取已通过（51684字符）。主任务已授权Luna单次run-02：内外Python均 `-X utf8`，仅子进程环境 `PYTHONUTF8=1`，其余验收/lease不变；新日志/输出/报告后缀02，保留失败记录。尚未收到run-02结论。

当前验证仅覆盖256×256 SDR、零深度、固定参数。它不验收1080p、人脸皮肤效果、流内改参、产品安装、预编译包及快速版。测试使用隔离进程的现有研究 Triton 缓存；不改产品 cache-only 清单。下一步收到报告后由主任务诊断；通过再做精确产品接线及缺失分支，失败停在首个问题。

## 固定社区来源与差异

主对照：[OptiScaler 社区提交 e237f895](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/tree/e237f895623742b761f9e5f00067cb3dc62619f4)。35份相关源码快照和逐文件 URL/SHA256 在 `D:/Codex-NR-Experiments/nr-b580/controls-upstream-e237f895/sources.json`。自动交叉索引 `config-consumers.json` 覆盖 Config 中72个 DlssNr 标识；索引只证明引用，不能代替计算验证。

下表是消费链分类结果，未列为产品通过的项目均不能对用户宣称已可用。所有路径相对该固定上游提交。

| 功能 / 配置 | 范围、默认或条件 | 实际消费者 / 当前项目状态 / 最小下一步 |
| --- | --- | --- |
| Style、Intensity、LocalTone、LocalStructure、AutoMask、SkinStructure | style0..2；强度/tone/structure0..2；skin继承-1或0..2；上游auto默认true，本项目默认false | `dlssnr/PassProfiles.h` → `forwarder/dlssnr_forwarder.cpp` 创建及evaluate参数。研究有历史证据，本次候选复测中；先保持旧默认。 |
| 显式 mask、UI/alpha/backbuffer、输入有效区域及motion/depth | 依赖真实资源及坐标尺度 | forwarder参数绑定、`DlssNr_Guides.h`；研究已实现的 mask/UI 范围不等于视频产品已支持。缺输入合同的部分先标缺口。 |
| Detail strength / Colour strength | 默认1；菜单分别0..2、0..4 | `precompile/dlssnr.hlsl` resolve中的亮度、颜色处理；独立于网络强度。当前产品未迁移该合成合同，应另做默认/端点的 shader 对照。 |
| SkinProtection、SkinToneEnabled、SkinDetail/Colour、EnvironmentDetail/Colour、ShowSkinMask | protection默认false；四个系数0..1默认1；tone开关控制skin colour | 同一HLSL的最终合成依据原始画面的颜色构造遮罩，分开衰减亮度与色度变化；不是模型自动mask。产品缺失，需含脸/非皮肤相近颜色素材及0/1端点检查。 |
| ApplyModel | 默认true；false仍运行模型 | HLSL早退显示原始输入。不能把它实现成不执行NR并暗中改变历史。产品未接；固定视频场景是否需要此调试开关单独收口。 |
| 多pass及逐pass覆盖、UnlockPasses | 默认1；常规最多3，解锁30；后续tone默认0，其余按Profile继承 | `PassProfiles.h` 明确继承；DX12调度独立pass状态。产品没有多轮合同；不能重复调用同一个有历史Session来假装多轮。以后用2pass最小序列验证独立历史及继承。 |
| WorkingScale / ScalingDownscaler | 默认1 / Lanczos3；工作倍率限制0.25..2 | DX12调整模型工作大小并准备输入/输出转换。当前快速NR256只是一种固定工作尺寸策略，不能宣称覆盖该任意倍率。需要尺寸/motion比例/重建联合验收。 |
| Transfer、MaxRatio | 默认matched residual、highlight guard2；还有classic路径 | HLSL根据工作尺寸和transfer分支重建效果。当前残差路线有相似目的，不代表公式相同。先列明SDR可移植部分再实现。 |
| Preset及pass2/3 hint | 0..3；按Profile继承 | forwarder设置hint；上游日志也明确参数表读回不证明视觉效果。当前研究不提供这项效果保证；需匹配DLL最小差分采样才能判断。 |
| Precision | 原生FP8或NVFP4混合候选，依赖NVIDIA路径及资产 | `DlssNrNative.cpp` 替换特定FFN，其他算子保留FP8，有不支持时回退。不能映射为B580精确/快速开关；当前不做移植需求。 |
| RunBeforeSr、FinishedPicture、DeferredDlss、ResidualAcrossRr及blend、ResidualFg及近似camera | 游戏hook、RR/SR/FG资源与时序相关 | DX12主调度、`DlssNr_Late.inl`、`DlssNr_DeferredSr.inl`、ResidualFg；当前视频NR→XeSS链保持不变。逐项区分可用重建与游戏专属接缝，不默认整体迁移。 |
| ReversibleMode、white point/source/trim/scale、曝光扫描、anchors及inverted | HDR/曝光及显示映射，部分为兼容旧配置 | HLSL codec、ExposureScan及主调度。当前SDR产品不扩展HDR声明；要真实HDR输入与曝光参考后才能开项。 |
| HoldFrame、Compare/split/zoom/swap/tags、DebugView | 画面冻结和诊断呈现 | 主调度、HLSL及overlay相关。记录调试用途，暂不为离线节点新增游戏菜单。 |
| ProxyProbe/UseProxy/ProbeD3D11、ScanMeter、AutoCapture、ToggleKey等 | 原生driver路由、平台兼容或工具行为 | 不等于模型图像参数。该源码子集里AutoCapture、CompareTags、ScanMeter、TagScale、ToggleKey、WhitePointFromExposure尚无非菜单消费者命中，必须标“需在全仓追踪”，不能据此判为死代码或无效。 |

本轮消费者盘点仍有上述跨仓引用待收口；RenoDX补充对照未完成。不把这份表称为社区所有功能已经验证。

## 新发现：社区改参策略不等于原生语义

`DlssNr_Dx12.cpp` 的 `TuningMatchesFeature` 比较整份 pass tuning，再由主调度重建 feature。因此该封装在强度变化时也会重建。本项目旧 `LiveControlledMotionNR` 原生对照则允许仅Intensity变化保留历史。这是层级差异，不能据社区封装直接修改精确后端。当前一次视频固定一组参数避开该分歧；以后若开放流内调参，必须声明选择的状态合同并另验收。

**⇒ 三家（我们 / 大力喜鹊 / 社区封装）的 ApplyMode 分类与代价分级、我们的可控项逐项对照、以及「我们没有的」各自对应什么功能，见 `reference/ressweep_v1/RESULT.md` §6.1–§6.3（2026-09-18 新增）。**

固定4060运行库SHA256仍为 `6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927`。上游安装文档推荐的另一个DLL哈希不是本次参考；没有对新DLL做兼容声明，也未更换权重或驱动。
