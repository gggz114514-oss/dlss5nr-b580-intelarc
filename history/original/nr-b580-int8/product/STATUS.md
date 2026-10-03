## 最新：实际 Comfy 环境仍失败，已加入系统工具链隔离并启动真实服务回归

上一轮模拟测试通过不代表实际用户环境：fcfd1012d6814625b8219db01b53799a仍断言。Luna诊断复现fbc838a500654f7fb965b7344b4e3acb仅两项原生能力false→true。Triton find_sycl_icpx优先PATH系统icpx，Comfy启动系统oneAPI2026.1，不能只清设备选择器。

新增runtime_environment.py，NR子进程排除PATH系统oneAPI/icpx、清工具链覆盖、显式先加载Python/Library/bin/sycl9.dll；模型与target/driver/cache校验不变。注入oneAPI PATH/ROOT/selector的独立诊断完整target匹配且SYCL/UR库均为随Python库。

当前用户Comfy已关闭，第一次POST未启动（startup null），没有当作成功。主助手随后启动真实ComfyUI PID19552，全部现有节点、系统oneAPI PATH/ROOT/selector保留在宿主；已启动submit_live_v3.py helper PID11384，等待服务就绪再提交冻结完整LoadVideo→精确NR→SaveVideo。目录D:/Codex-NR-Experiments/nr-b580/product-worker/comfy-live-env-v3。Luna监控审核，主助手不poll。收到handoff前不标记通过。服务留给用户使用。不再向用户声称所有电脑可即用；可分发包仍需后续工作。

## 最新：精确节点环境修复完整视频复测通过

已读取comfy-exact-env-v2/handoff并核对全部证据与源hash，保存main-decision-v1.json。用户此次10s_480p.mp4精确单NR：worker rc0，输入/输出独立解码均243帧864x480@24，10.125秒，音轨保留；Block242、SR/FG/depth与CPU像素传输0。模拟Comfy启动过滤环境通过，父进程selector保持不变，无临时源码编译。

精确节点报错修复已生效，无需重启Comfy，用户可直接再次运行。此次是节点完整执行回归，不是新的4060逐字节比较或画质验收。输出D:/Codex-NR-Experiments/nr-b580/comfy-jobs/a23d958d29e3465b81a4c3d28231dcfa/nr-video.mp4。无新测试启动。

## 最新：修复实际 Comfy 环境下精确节点的 target 断言

用户作业4b9b9a92b68c4913849d3b0aabb7500f在CacheOnly进入前target断言失败。已复现启动脚本ONEAPI_DEVICE_SELECTOR=level_zero:0使Triton扩展探测的XMX/blockIO/bfloat16三项为false；SYCL_DEVICE_FILTER单独不会导致问题。runner只在独立NR进程导入GPU运行时前移除ONEAPI_DEVICE_SELECTOR，保留精确包完整target/driver/hash校验，不改模型/内核/缓存。不需要重启Comfy；新worker读取新runner。

模拟Comfy过滤环境的诊断已确认完整target与冻结catalog一致。已实际启动smoke_exact_env_v2.py（用户10s_480p.mp4、精确单NR），Luna监控审核，路径D:/Codex-NR-Experiments/nr-b580/product-worker/comfy-exact-env-v2。完整视频结果等待handoff，不能先声称通过。runner修改前副本保存在该目录runner-before-fix.py；旧测试证据保留。

上一轮comfy-node-v1快速单节点handoff已读取：243帧864x480@24与音频、Block242、CPU像素0技术通过；不代表精确节点实际环境通过。

## 最新：两个本机 VIDEO 节点已安装，快速版明确标为类 DLSS5

用户认为快速版差异明显，要求封装后自行体验；不再把快速版描述为精确等价。fast-sr-fg-full-v1/handoff技术审核通过，主助手核对全部证据hash并保存main-decision；原manifest scale=1.0为陈旧字段，实际native尺寸1728x960是2x，原记录不改。

新增 product/comfy/{nodes.py,runner.py,README.md}；安装 custom_nodes/NR-B580-Local，两个节点 NR 视频（B580）、NR + XeSS SR + FG（B580），精确/快速可选。GPU Block/native-v4与缓存执行复用，输入输出文件只在端点封装，作业与成片在D:/Codex-NR-Experiments/nr-b580/comfy-jobs。尚非可分发安装包。重启ComfyUI加载。CPU导入、两节点schema、243帧输入metadata、1.5x尺寸计算通过。

已实际启动 product/comfy/smoke_node_v1.py，从已安装节点处理完整480p视频。监控与固定解码审核由Luna执行，产物 D:/Codex-NR-Experiments/nr-b580/product-worker/comfy-node-v1；主助手不poll。此项在收到handoff之前不算通过。以后恢复先看handoff/用户通知，遇错误由主助手修复，不让Luna重试。

## 最新：快速1xAA+FG通过，快速2xSR+FG已启动

已读取fast-aa-fg-full-v1/handoff通过并核对来源/产物哈希，记录用户“可以”画面确认，保存main-decision。已核对预先冻结的2x清单全部源未变并实际启动nr-fast-sr-fg-full-v1，source3f0d6a9，exec43520，900秒。243源→485输出1728x960@48，快速NR+2xSR+FG，Block共享运动，DiskOnly禁止临时编译，左同倍率精确/右快速完整对照和音轨。

Luna监控audit_fast_sr_fg_full_v1.py固定审核，异常立即回报不修复重试，主助手不poll。路径D:/Codex-NR-Experiments/nr-b580/product-worker/fast-sr-fg-full-v1。2x新组合还没验收，不能沿用1x结论。精确和快速单NR/1x已验收，产品两个VIDEO节点、倍率UI、安装打包尚未完成；完成2x后进入产品封装。

## 最新：快速单NR通过；快速1xAA+FG运行中，2x测试已准备

主助手已读取fast-single-full-v1/handoff通过并核对来源/产物，保存main-decision-v1.json和用户“没问题继续”画面确认。快速单NR完整链路通过。

已实际启动nr-fast-aa-fg-full-v1，source3f0d6a9，exec1942，900秒；243源→485帧864x480@48快速NR+AA1x+FG、Block共享运动、DiskOnly禁止编译，左同倍率精确/右快速完整对照与音轨。Luna监控audit_fast_aa_fg_full_v1.py固定审计，遇错立即反馈不修复重试，主助手不poll。路径D:/Codex-NR-Experiments/nr-b580/product-worker/fast-aa-fg-full-v1。

2x快速NR+SR+FG脚本Run-FastSrFgFullV1.cmd、audit_fast_sr_fg_full_v1.py及fast-sr-fg-full-v1/manifest.json已冻结准备但未启动。先核对1x交接，用户新画面审核，再启动2x；节点封装/可搬迁包仍未完成。不要把准备说成已运行，或把旧单NR验收当FG组合通过。

## 最新：快速单NR完整GPU视频测试已启动

快速缓存验证已由Luna和主助手核验通过，6cases/154配置/0源编译，保存fast-cache-v1/main-decision-v1.json。现已实际启动nr-fast-single-full-v1，sourcee860bb4，exec62095，900秒。沿用reviewed NR256 Stack+色调修正，独立fast模块进程、DiskOnly禁止缓存缺失时编译。243帧864x480@24原始GPU Block→快速NR→QSV，SR/FG/depth应0；左精确单NR已接受成片，右新快速单NR，1/24显式索引拼接和音轨/243解码验收。新链路画质需要用户审核，不能沿用旧视频验收。

产物D:/Codex-NR-Experiments/nr-b580/product-worker/fast-single-full-v1。Luna监控audit_fast_single_full_v1.py固定审核，出错立即报告不修复重试；主助手不poll。下一步核对交接和画面，再快速版1x/2x SR+FG组合，随后两个VIDEO节点及打包。精确节点已验证的实现依赖native-v4；快速完整视频首次GPU Block接入正在验证，不宣称产品节点完成。

## 最新：单NR通过，快速版缓存执行准备已启动

单NR完整243帧技术handoff已核对全部来源/成片哈希，用户明确“可以了。推进快速版”，已保存exact-single-full-v1/main-decision-v1.json。精确两个目标模式已有实机链路，产品节点仍未封装。

已启动nr-fast-cache-v1，source4872f93，exec46574，900秒。沿用已接受快速NR256 Stack/色调修正profile，不重新量化；旧缓存D:/Codex-NR-Experiments/nr-b580/reference/triton-cache-c32-triton38-v1。实际GPU执行三尺寸256/480p/1080p各reset+temporal共6次，记录Triton配置/产物，DiskOnly guard禁止源编译，缺失即停止报告。快速Stack有常量校验/图捕获/寄存器筛选，使用真实缓存执行收集，不冒称纯shape trace。此步不是视频画质验收或完整可搬迁预编译包。Luna监控audit_fast_cache_v1.py固定审核，不修复重试；主助手不poll。产物D:/Codex-NR-Experiments/nr-b580/product-precompile/fast-cache-v1。下一步核对结果，接快速单NR及1x/2x组合视频，再两个节点封装。冻结源存在EOF额外空行style提示，不在运行中改源码；下版整理。

## 最新：1xAA+FG人工/技术通过；单NR完整视频测试已启动

主助手已读取exact-aa-fg-full-v1/handoff passed并核对全部来源和成片哈希，保存main-decision-v1.json，记录用户“可以没问题”人工反馈。1x自动AA106、完整485帧/音轨/计数通过。这只覆盖该精确1x组合样本。

native-v4 (966316f)编译通过且哈希检查完成。单NR显式入口禁用SR/FG/深度执行，仍用原始GPU Block源对运动、NR和QSV。已启动nr-exact-single-full-v1，source20d3b0c，exec20438，900秒。243帧864x480@24；Luna监控audit_exact_single_full_v1.py固定审核，验证SR/FG/depth计数0、NR243、源对242和双视频243帧含音轨；错误立即回报不修复重试，主助手不poll。D:/Codex-NR-Experiments/nr-b580/product-worker/exact-single-full-v1。单NR通过后再处理快速版预编译及实际链路、两个VIDEO节点和打包，GPU DIS已取消，倍率只开放验证过的范围。

## 最新收口：两个节点，仅GPU Block；1x AA+FG完整测试已启动

用户明确取消DIS接入，改两个节点：单NR，以及NR+可调倍率XeSS SR+FG；均保留精确/快速。SR1x需实际执行XeSS AA，而非未说明地绕过。其他倍率先由SDK尺寸范围检查并实测后开放，不承诺任意倍率均已验证。四节点/独立NR+FG/DIS计划被本决定替代。

主助手已读FG修复handoff passed并核对result、sources、视频hash，保存fix-v2/main-decision-v1.json；485帧修复技术验收完成，原失败记录保留。用户此前画面正常反馈保留，后续先看审核再宣称通过。

native-v3 (ea035bb)编译退出0，产物/来源核对保存main-build-receipt；增加同尺寸自动XESS_QUALITY_SETTING_AA=106和xessGetOptimalInputResolution范围/宽高比验证。已实际启动nr-exact-aa-fg-full-v1 (8d10519)，exec60648，900秒，243原始帧→485帧864x480@48。Luna监控audit_exact_aa_fg_full_v1.py固定审核，出错立即报告不修复重试；主助手不poll。产物D:/Codex-NR-Experiments/nr-b580/product-worker/exact-aa-fg-full-v1。比较制片使用修复后的显式1/48时间基和0..484索引。单NR、快速版预编译/接入、两个Comfy节点、倍率UI及打包仍待完成。下一步核对1x结果，完整视频新画质需用户审核。

## 最新纠正：FG原审核失败，不得称整轮通过

用户指出Luna审核comparison.mp4仅484帧，原生和preview为485。主助手此前未看handoff就把“少一帧”误解为243→485正常规则，已向用户承认并撤回整轮通过。用户“画面正常”保留为观感反馈，不能替代失败的技术验收。

已检查原始视频包数：SR243、FG preview485、旧comparison484。左fps48滤镜独立486帧、右独立485，错误位于对照合成时间基准/结束与输出同步处理。CPU修复统一两路1/48时间基与帧索引0..484，passthrough输出。fix-v1因-r与passthrough矛盾即时失败保留；fix-v2退出0，实际解码原preview及新comparison均485帧含音轨，保存末帧PNG且主助手已查看。未重跑GPU，旧源/成片/失败交接未改。修复路径D:/Codex-NR-Experiments/nr-b580/product-worker/exact-sr-fg-comparison-fix-v2，独立Luna审计audit_fg_comparison_fix_v2.py待执行。后续必须先读Luna结果再作通过声明。

## 2026-09-11 最新：SR完整视频人工通过，SR+FG完整测试运行中

用户明确“验收了没问题”，范围为exact-sr-full-v1完整243帧人脸NR+SR；主助手验证Luna交接/全部来源及产物哈希，保存main-decision-v1.json与user-review-v1.json（不改原审计）。该验收不覆盖FG/DIS/其他模式。

已实际启动nr-exact-sr-fg-full-v1，源44ab6e2，exec55156，900秒。243原始帧执行NR/SR/FG，期待485帧48fps输出；一次源帧对运动供三者消费，审计242次源对分析且motion_after_sr_count=0。完整preview保留音轨；comparison左已接受NR+SR（重复到48fps），右NR+SR+FG48fps。路径D:/Codex-NR-Experiments/nr-b580/product-worker/exact-sr-fg-full-v1。Luna运行audit_exact_sr_fg_full_v1.py固定审核，错误立即回报不修复重试，主助手不poll。完成后核对并请用户审核新FG画质；异常停止不隐瞒。单NR/NR+FG、快速包、DIS、四Comfy节点仍待实现/验收。

## 2026-09-11 最新：完整人脸视频已启动，待Luna审核

用户要求跑通后给完整视频。nr-exact-sr-v2的48帧真实链路已Luna及主助手验收，原生report CPU整帧上传/回读0，48帧全部NR/SR/编码完成；尚无画质或运动兼容验收。已启动nr-exact-sr-full-v1，source2a48011，exec5289，900秒上限。243帧、10.125秒、24fps，1728x960 NR+SR成片保留原音轨，另做左原图右NR+SR缩小到864x480的并排对照（复查制片在产品链外）。

D:/Codex-NR-Experiments/nr-b580/product-worker/exact-sr-full-v1：preview.mp4为完整分辨率，comparison.mp4为对照；result/manifest/handoff记录证据。Luna监控并audit_exact_sr_full_v1.py固定审核，两视频需CPU解码243帧且有音轨。出错立即回报主助手，不自行修复重试。主助手不poll；完成通知后核对哈希并展示两视频，要求用户审核脸部色调/运动闪烁/拖影。未验收前不宣称共享源运动兼容。单NR、FG、DIS、快速包和四节点产品化仍待完成。

## 2026-09-11 最新：真实视频精确NR+SR短测已启动

native-v2由Luna固定审计通过，主助手核对源码、生成文件、DLL及CSO哈希并写main-decision-v1.json。已启动nr-exact-sr-v2，source ddcb193，exec session37538，GPUlease900秒。48帧用户人脸视频864x480@24 → 精确NR+GPU Block源运动+XeSS SR 1728x960+QSV，成功后封装2秒带原音轨预览。不是完整画质/运动兼容验收；快速版、DIS、FG、单NR、四节点和完整视频仍待完成。

V1在native参数检查退出4：主助手漏传现有worker必需的--depth-model，未运行NR。V2补现有DepthAnythingV2Small共享GPU深度配置；精确NR仍零深度/default controls，原算术及996缓存未变。日志/证据/视频均D:/Codex-NR-Experiments/nr-b580/product-worker/exact-sr-v2；日志在上一层exact-sr-v2.log。Luna运行audit_exact_sr_v2.py启动/完成固定审计，异常立即回报不自行修复重试，主助手不监控。等待用户通知后核对交接，再开展完整视觉对照；发现源运动经NR后不兼容需停止报告。

## 2026-09-11 最新：原生NR回调接入，V2编译退出0，待Luna审计

nr-exact-texture-v2已由Luna及主助手验收：39帧归档4060字节一致、9纹理帧一致、996配置纯缓存加载。main-decision-v1.json位于该任务D盘monitor目录。没有SR/FG组合视频验收。

新增product/worker：同进程ctypes NR回调、每slot独立RGBA8增强资源、预编译GPU格式转换shader、明确producer/consumer完成同步。源color/flow不覆盖；只替换record_gpu_consumers下游颜色。原生worker从相邻项目生成私有构建副本，相邻源码未修改。当前入口先覆盖原生GPU Block，GPU DIS工厂接入及单NR路由、快速版预编译和Comfy节点仍待完成，不宣称四节点已实现。

V1编译失败原因是Windows min/max宏：新头文件比上游NOMINMAX定义更早引入Windows头。V2仅增加/DNOMINMAX，源commit3c300e2，实际编译已退出0；Luna负责固定audit_native_build_v2.py审计，主助手不poll。所有源码和失败证据保留，缓存/构建均D:/Codex-NR-Experiments/nr-b580/product-worker/native-v2。没有运行组合视频；下一步接受编译审计，配置同进程动态库依赖并启动有界组合测试。新增callbacks无CPU像素传输，但包含host完成等待和RGBA32F到RGBA8量化；待画质验收，不称精确端到端输出与4060一致。

## 2026-09-11 最新：预编译通过，缓存模式数值测试已启动

精确预编译 a86fd59：996规格、14编译进程，约167.7秒完成（含新进程加载检查）；Luna固定审核及主助手源码、证据、全部编译文件哈希核对通过。只验收编译/缓存加载，不代表数值通过。D:/Codex-NR-Experiments/nr-b580/product-precompile/exact-v1-monitor-luna-v1/main-decision-v1.json。

已实际启动 nr-exact-texture-v2，exec session87350，冻结b9d879752d994bff766ced39130ed654bbea2503。复用预编译包，运行中禁止新增Triton编译。39帧对照归档4060（256/480p/1080p，各13帧）及9帧GPU纹理对照。900秒上限，Luna监控和固定审计，遇错立即反馈主助手、不自行修复重试。此项待验收；没有宣称SR/FG兼容或四节点已完成。

原nr-exact-texture-v1已900秒超时，只有13帧原生256部分结果，未通过整轮；旧缓存完整保留。快速版预编译包尚未制作。所有实验输出/缓存留D盘。下一步核对V2交接后继续全GPU worker接入，运动信息不兼容必须停下，组合视频需用户审核。

# 产品收口进度

2026-09-11：本体研究暂停新增优化，转产品接入。

已完成：优化总结、统一 Session API、已接受的快速本体和五处无损 decoder
gather 整合；约325KB独立标定包在D盘。源码commit900a851。

已验收：nr-product-full-v2 的 Luna 固定审计通过，主助手已核对源码、结果、
哈希及 V1 调用方修正记录。完整243帧人脸保持已接受快速版输出，产品 API 的
重置/尺寸/关闭及1080p配对验证通过。V1 的 half 运动输入错误保留原始记录。

显存互操作：bridge-memory-v2 的18轮逐字节往返和 Luna 审计通过，主助手
已写独立 main-decision-v1.json。其范围是共享 buffer，仍有 host 完成等待。

新增已验收：nr-texture-v1 的Luna审计和主助手核验通过；48组纹理转换、
4组RGBA32F输入和3帧快速NR纹理往返均字节一致，168项合同拒绝通过。主助手
核对1433项来源和证据并写独立main-decision-v1.json。产品路径CPU像素上传为0，
参考上传/诊断回读单独计数；不代表SR/FG兼容性或最终产品已经验收。

当前实际运行：nr-exact-texture-v1，源码f65b2e17c60a13eb1600485e50a675fc5286c0c1，
GPU lease900秒，session70404，由原Luna/max监控和固定审计。精确Session在256、
480p、1080p各13帧对照存档4060原始RGB，再验证9帧精确NR纹理输出。主助手不轮询。
精确分支35个源码已与既有stage receipt完整哈希映射核验相同；不改本体算术。

尚未安装四个ComfyUI节点。下一步将精确/快速接口接原生全GPU worker，保留
源图给下一对运动分析，再验证SR/FG复用原始运动场的画质、成片和音轨。
若NR后原运动导致SR/FG异常，立即停止组合实验、保留证据并报告用户。
新的组合视频必须由用户肉眼审核。未发布GitHub；实验和构建均写D盘。
