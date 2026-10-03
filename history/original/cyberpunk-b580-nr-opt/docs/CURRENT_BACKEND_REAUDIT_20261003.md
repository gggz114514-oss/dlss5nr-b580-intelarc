# 现役 C512＋K8 后端重审与改造顺序

本轮以《赛博朋克》现役 720p C512＋K8 为唯一开发基线。四名 GPT‑6.1 Sol max 分别审计 Swin/MLP、C512/ViT、前端/Decoder/post、history/host/graph；主任务固定来源并独立核对调用、有效选项及引用。没有修改游戏安装或精确4060分支。

重审确认：**现役 FP16 矩阵已有 DPAS，仍有分块、读取复用和数值对齐方面的改进空间；去舍入开关也没有覆盖所有输出边界。** 不能再以“用了XMX”或“unrounded已开启”视为审计完成。每个模块均检查了计算、相邻非矩阵、生产者/消费者及图外开销，包括未发现FP8遗留的模块。

审计内容固定为九项：实际调用和有效开关；数值对齐及正常算法边界；矩阵类型、tile、K分块和资源；相邻非矩阵工作；生产者/消费者布局；临时缓冲与寿命；CPU、capture、replay边界；完整模块改写方案；最小成本与质量判别实验。逐项要求真实shape/stride、调用位置、源码SHA和编译证据，缺少实际成本的项目明确列为待测。完整要求见 [AUDIT_CONTRACT.md](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/b580-current-backend-audit-v1-20261002/AUDIT_CONTRACT.md)。

固定 identity 为 `7cf38a499a25d863db90a49d19c88ab3ad98d32511c72ee54c9efb973db17d30`。223份parent数学源、551份当前runtime源和9份front来源已与实际G安装核对；执行索引绑定137个实际compiler key及加载的SPIR‑V。四组共38项发现，A17/B17/C4。A表示确定发生、值得先验证，**不表示已测得加速**；其中共享helper和复制链有重叠，不能把38项当作38份可叠加收益。

主任务来源/覆盖回执：
[MAIN_SOURCE_AND_COVERAGE_REVIEW_FINAL.json](D:/Codex-NR-Experiments/cyberpunk-opt/b580-current-backend-audit-v1-20261002/MAIN_SOURCE_AND_COVERAGE_REVIEW_FINAL.json)，SHA `ee0187dfedfac630a19d785d3e93e9c3567c66021c7bc353485a872e46f2d957`。这是四组最终交付后的核验；较早回执原样保留。它验证SHA和字段；以下路线选择与重叠处理由主任务另行审阅，未把静态字节或指令数转成耗时。

| 当前模块 | 重审结果 | 改造边界及状态 |
| --- | --- | --- |
| C32 MLP | native half cubic；矩阵FP16 DPAS。已共享一次X加载，现役不读旧cubic LUT | 比较更宽expand片上排程；不要重复声称去掉四次X重载或当前LUT。新写：Swin‑10 |
| C64/C128/C256 MLP | C128已pairwise＋FP32分支合并；C64/C256四段expand重复加载X，分支projection保留有序half边界 | C64/C256分支FP32策略已有机制；更宽expand、latent生产/投影读取布局需新写。Swin‑01/02 |
| 窗口QKV和归一化 | 已直接打包QKV，FP16 DPAS；跨segment/head重复读取。Q/K仍有ordered half归约和NaN选择，half FMA已native | 合并读取；单独比较FP32归一化。不是所有norm都还用补偿FMA。Swin‑03/04 |
| 窗口attention/tail | C64、C128已融合，score/value为FP16 DPAS；权重denominator保留half链，近似指数是算法。C128 BM16、C256输出投影BN32 | native denominator数值候选与tile/布局分别测；首个decoder C32遗漏融合有具体调用入口。Swin‑05/06/07/08 |
| C512连续FFN | 现役已经是五段INT8 DPAS链，native cubic，无hidden FP8往返；scale、half/requantize及scratch仍有成本 | 当前C128 Q12探针不能代替此M960/floor16模块。须另导出此模块的真实pack/input，再判断内部融合或FP16整FFN替代。ViT‑05 |
| 全局ViT FFN | INT8 expand/contract；P4 partial与merge为INT32，并非half分段模拟 | PARTS1/2/4调度和scratch成本需完整FFN验证，检查真实pack的INT32范围。ViT‑06 |
| 全局ViT QKV/投影 | FP16 DPAS、K内部FP32累加；QKV两段输出、projection四段输出及合并仍有half边界，附带pack/partial存储 | full‑K候选已有但当前关闭；生产者/消费者直接读取、保序分段融合是另一类新实现。ViT‑01/04 |
| 全局ViT norm/exp/denominator | unrounded下补偿half FMA仍执行；四组64-key pack/ordered half归约和exp(0)重算仍在 | 复用native FMA候选，直接denominator消费者及常量预计算；保留近似指数算法、padding校正、floor。ViT‑02/03 |
| 普通stage down/up矩阵 | FP16 DPAS，实际720几何仍多用通用16×32 tile；noise/front已原生，pool/down仍多pass连接 | 以真实M/N/K/stride改policy，并测pool→down完整模块。不能重复移植已完成K8/QKV库路径。Front‑02/09、ViT‑07 |
| DecoderGather | 激活去舍入已开启；**四个非C32 merge的ROUND_OUTPUT仍为true，执行E4M3舍入** | 复用已有merge‑unround scope。merge/post entry补偿FMA独立比较；新写连续decoder输入投影可消pack/partial，但不能混淆数值变量。Front‑03/04/05 |
| post | sigmoid表索引、颜色/历史FMA、软件RTZ及多个中间张量仍执行；post entry可直接供native C32消费者 | 完整post-tail先融合当前算法，再单独比较native sigmoid/RTZ/FMA；直接写V6外部目的缓冲。Front‑05/06/07/08 |
| 历史与图外准备 | fractional写了一份未返回的normalized，调用者又乘一次；finite/range/near谓词多次读回；frame诊断列表持续增长 | 关闭无消费者store，合并扫描为一份状态摘要，限长记录；每帧动态状态仍需有效判断。History‑001/002/008 |
| 历史坐标/倒数/near | fractional仍table/table/reference；near有SM89纹理对齐计算。13帧fixture未命中near，游戏命中频次未知 | native inverse/direct-pixel已有选项；近整数必须补真实路由样本，不能当作“不执行”或当前已测热点。History‑004/010 |
| graph/资源生命周期 | body replay已绕过capture内Python；图外准备仍提交。publication、公有snapshot、private history有独立寿命 | 改producer目的缓冲与消费者接口，保留跨队列lease/错误回滚；不能直接删除clone、等待或暴露graphpool临时输出。History‑006/007、Front‑01/08 |

表中简写分别对应 `swin-mlp-01…10`、`c512-vit-01…07`、`front-decoder-post-F01…F11`、`history-host-graph-001…010`；以各组JSON中的完整ID为准。普通half存储、近似指数、cubic、尺度及NaN处理单列数值/算法合同，不能仅凭出现half或查表便归为FP8模拟。本轮执行证据为720p；360/480/540p共用代码的适用性可以据调用核对，收益及画质仍须各档验证。

发现的完整代码位置、SHA、effective constexpr、风险和最小实验分别保存在：

- [Swin/MLP报告](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/b580-current-backend-audit-v1-20261002/swin-mlp/REPORT.md)
- [C512/ViT报告](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/b580-current-backend-audit-v1-20261002/c512-vit/REPORT.md)
- [前端/Decoder/post报告](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/b580-current-backend-audit-v1-20261002/front-decoder-post/REPORT.md)
- [history/host/graph报告](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/b580-current-backend-audit-v1-20261002/history-host-graph/REPORT.md)

**开发先处理实际执行的数值对齐工作，再重组完整模块。** 矩阵继续按B580有效DPAS布局实现；归一化、采样、激活等采用适合的GPU向量实现。改dtype、换“native”名字或删除普通FP16存储cast均不能替代这一工作。

第一批并行数值候选：四处Decoder merge输出去FP8舍入；全局ViT及Decoder/post entry补偿FMA；C64/C256分支累加；ViT full‑K和denominator；post sigmoid/store；history inverse/coordinate。已有scope优先复用，新增策略固定到本基线。native FP32 FMA→half可与单次half FMA相差1 ULP，应作为明确的fast候选记录误差，不能因不逐字节相同而阻止快速版，也不能把它标为精确等价。近似指数/cubic、尺度、floor、完整运动/历史/控制语义保留。

第二批完整计算模块：C64/C256 expand复用及projection布局；QKV多segment读取；遗漏的首个decoder C32融合；C256输出投影；ViT QKV→norm→score与denominator→projection消费者链；真实720 stage tile及C512/ViT完整FFN。tile/数学/布局分别保存判别版本；不再重复同tile、同工作量的“原生attention”替换。C512/ViT整数链须有自己的当前pack/input探针，不能把C128结果移过去。

第三批接口与状态工作：front控制一次生产、pool→down、post entry→native C32、post-tail直接发布、history scratch/static input接口和图外摘要。按完整producer/consumer链改，合并以下重叠：front-control/normalized归同一准备链；post-tail目的缓冲与V6 publish归同一次发布改造；native FMA共用真值/lowering检查；stage矩阵统一policy但分别验几何。192MiB未消费noise表为冷驻留问题，单列内存项，不承诺热路径ms。

每个有数学差异的快速候选记录最终输出/私有历史误差并做用户画面审核；等语义布局候选可用字节验证。所有候选均在计时外标定、预分配、预编译，Luna串行GPU测试；普通提交和图重放分开，整模块与边界分开，保存raw/P50/P95/首尾漂移及实际key/IR/binary。取消1ms判别门；有可重复净收益即可合并。整帧/RTSS以《赛博朋克》同场景、FG关闭验收。

既有候选的负收益只约束其当时实现、布局和调用边界。复测须说明改变了哪个原因；小矩阵原生核、输入转换、内部激活/重缩放及下游消费分别记录，再用完整模块净耗时判定。没有结构或边界变化的旧候选不重复测试。

本轮INT8已完成的独立结果进一步限定了方向：真实`encoder128.0` M15360，BM32 graph16，FP16完整MLP约0.0994ms，旧INT8完整约0.2397ms，预量化后仍0.2333ms；裸expand为INT8约0.0175ms、FP16约0.0202ms。实际矩阵已有INT8 DPAS，入口不足以解释完整模块失利；内部复用、cubic LUT/Q12及投影需继续判别。该LUT不是FP8编码表。详见[INT8测量与数据流审计](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/docs/INT8_NATIVE_AND_DATAFLOW_AUDIT_20261002.md)。

当前已验收实机为55–60ms/帧；已保存离线38.8188ms仅为captured body＋布局＋V6发布，范围不同，不能相减当作纯桥/搬运。720p基础30/60FPS仍以整帧≤33.33/16.67ms衡量。本次审计没有每项成本证据或可加总的收益承诺，下一阶段通过上述完整模块测试收敛预算。

独立桥终端等待试验不计入这38项收益：第一轮真实TLS/完整模型OFF运行至第8帧，第9帧CPU‑prepared路径的测试审计错误读取空`_gpu_refs`，回执被拒。生产代码没有因此部署；修订测试将监听实际prepare返回值核对输入，并继续独立验证桥自身lease。此测试修复不改变模型数学。
