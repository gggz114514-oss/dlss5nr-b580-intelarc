# 现役基线证据（截点：2026-10-03）

本档由历史／证据 worker 只读整理。**唯一现役性能主线是 Cyberpunk 1280×720 C512＋K8／all6、去 FP8 激活模拟舍入、融合真实运动／历史、图重放，以及已验收的纹理复用／GPU 接力桥；用户实机基础帧时间验收为 55–60 ms。** NR256 产品与冻结 4060 精确分支是历史路线，不能替换这个默认叙述。

“去舍入”不能扩大成全网不含 FP8：原 E4M3 权重解码、必要 INT8 scale／激活和部分尚存 half／FP8／补偿 FMA 边界仍需逐核区分。选中配置、源码存在和实际执行是三种不同证据。

## 1. 已找到的 accepted 源与固定证据

本地索引保存实际路径和hash；公开伴随索引使用`@project`、`@experiments-D/E`别名，不发布个人目录／局域网／SSH内容：

- `P`＝`E:\ComfyUI-aki-v3-IntelArc_20260722\cyberpunk-b580-nr-opt\artifacts\b580-full-implementation-v1-20261003`。
- `R18`＝`P\phase2-complete\revisions\r18\source`。

| 证据 | 记录内容 | 身份／边界 |
| --- | --- | --- |
| `P\phase2-complete\PHASE2-r18.json` | revision 18；261 份源码；114 个测试档；`arms.accepted_baseline` 的 constructor、numeric identity、空 selectors／packages、baseline=true | 文件 SHA256 `699ee05cf1008610a4772dfeebe6398d7099c7c1ed141cbbaa91967da0a05f7f`；source manifest SHA `74d2bb6b2a0de19d095359f84365e96c07d283c4f1e25d643926891d348c3094`。261／114 是实验快照数量，不能当成现役文件数量或优化数量。 |
| `P\BASELINE.json` | 223 项 math sources、7 项 host sources、63 项 native sources，各有来源／目标／哈希 | 这是源身份清单，GPU_executed=false；原 math 来源为 `artifacts\vit-current-provider-gpu-runner-v4-20261002\source\parent`。 |
| `artifacts\b580-full-coverage-audit-v2-20261003\RUNTIME_SCOPE_720.json` | 835 项审计库存、223 项 parent 源、137 个实际 compiler keys；720p selector；accepted texture pool／GPU handoff | SHA `83c14cad54e28461e39a25515bebb30a769ab9372c0ab32114e9dcb7c97397de`。G 盘 runtime／安装回执是**报告中的引用**，本 worker 未读取／核验实际 G 部署。 |
| `P\TEST_PROGRESS_20261003.md` 与 `WORK.json` | accepted 当前720对照、各候选结果和未测范围；现役游戏未换版 | 55–60 ms 来自此前用户验收；本文没有新增实机测量。 |
| `P\AMD_LATEST_MAIN_REVIEW.json` | accepted constructor 与研究基准一致，保留全 K、真实运动／历史／控制 | six new plans 是未测方案，不是六项已实现收益。 |

模型身份在 RUNTIME_SCOPE 中记录：`WEIGHTS_HT` SHA256 `836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4`；profile SHA256 `336086117e4e24423a486265dcf9f08803bef1a5c4610954d179d3cfeeed3588`；runtime config SHA256 `26471f140205ebbdc249159f0ca1c0c35d786cd1172c809e01cfd0f2297ef94f`。这些是记录的身份，不表示本次重新打开模型／G 配置校验。

## 2. accepted constructor 的实际内容

以下来自 r18 `accepted_baseline`，不能用 10 月 1 日旧网页“四项”记录覆盖它：

```json
{
  "controlled": true,
  "graph_capture_policy": "all",
  "combo_modes": [480, 540, 720],
  "c512_qkv_library_720": true,
  "native_k8_720": true,
  "decoder_gather_unround_720": true,
  "c32_hidden_native_720": true,
  "c512_probability_unround_720": true,
  "c128_pairwise_720": true,
  "c128_dual_qkv_720": false,
  "c64_attention_project_720": true,
  "c128_attention_project_720": true,
  "numeric_cleanup_720": {
    "branch_accum_families": ["c128"],
    "history_value": "fp32_fractional",
    "front_noise": "native_both"
  }
}
```

numeric identity 为 `numeric-cleanup-720-v1:85549e9c272011a4ed4974a4d341f3e28280723afe3dabf29c9dd0d9af043938`。`combo_modes` 宣告其他尺寸入口，**本次现役优化验收对象仍固定720p**。运行调用还需选 `variant="unrounded"`、`history_warp="fused"`、`graph_replay=true` 和相应实际 controls／source geometry；只有 constructor 不能重现全部运行合同。

r18 accepted 对照另有 `audit_history_host_720_v1` before_numeric 测试 hook；其改变计算／launch／buffer 等开关均 false，仅使用 receipt_capacity=64。它是实验对照的记录包装，不能据此认定当前 G 运行时已安装该 audit hook。

## 3. 阅读 accepted 源的入口

`R18\game\nr_game_fullsize.py` 已静态核对：

- 36–40 行尺寸表：720p 有效／模型面 720×1280、内部 768×1280；540p 模型面 544×960、内部 640×1024；480p 模型面 480×864、内部 512×896；360p 内部 384×768。不能只改 grid 数量就认为相同 shape。
- 约 499–521 行：`re4_session_v1.open_session`，controlled 路线用 `GameLiveControlledNR`，在作用域结束恢复原工厂。
- 约 600 行：C512 library 与 native K8 同时启用时走 `c512_k8_joint_scope_720_v1`；两个开关单独启用有分开的入口。
- 后续 accepted 启用路径包括 `c32_hidden_native_720_v1`、`c512_probability_unround_scope_v1`、decoder gather overlay、C64/C128 attention/project、C128 pairwise 与 numeric cleanup。
- `process/_process` 保留实际 geometry、flow、reset/history/seed／controls；捕获与 replay 必须区分。退出先退役图消费者再释放会话与资源。

依赖在 `R18\experimental\fp8_unround_overlay\` 及其 `modules`／backend、`game` 工厂和 bridge binding 中。**不能只抽一个看起来最热的 Triton 文件就称为可运行现役工程，也不能把 r18 整目录全部候选默认启用。** 独立源／哈希回执在 `CURRENT_SOURCE_RECEIPTS.json`。

### 3.1 main实际runtime提取与本worker本地复核

主代理已完成实际G来源逐文件提取。初始`SOURCE_EXTRACTION_LOCAL.json`记录current449／r18候选260、合计709项、5,076,831字节，未复制weights／私有GPUBlock/DIS，没有GPU运行或游戏写入。原r18一项可再生pyc不入source发布目录。

公开工程选源分开`current/runtime`、`current/bridge`和`experiments/2026-10-03/r18/source`，后者仍未采用。`release-staging/evidence/2026-10-03/current-source-manifest.json`随后增加authoring来源：首次复核568条、随后索引重建571条，manifest明确installed_runtime_source_files=406；扩展总数不能都称G安装文件。worker对该时点stage按manifest逐文件重hash，带观察时间的匹配数与文件SHA见`CURRENT_SOURCE_RECEIPTS.json`。此处复核的是main提取的本地副本，没有重新访问G。

该manifest仍明确`bridge_source_to_binary_match_verified=false`。源身份、已验收trial二进制、canonical重编二进制与当前实际部署binary需要各自receipt，不能由Python/source相同直接推出binary build一致。

### 3.2 bridge promoted／build证据补齐

已读`@experiments-D/cyberpunk-opt/gpu-handoff-serial-owner-v2-20261002/source-integration-v2/MAIN_CANONICAL_SOURCE_RECEIPT.json`（SHA`1701e0769354368a5e6ec185d981897dea6f7c8175349d35f335154de7a6e19e`），状态为canonical源合入并CPU验证，包含25after-pins、promoted回执／plan hash、9项C++检查及48项Python检查、旧ViT freeze保持。

已验收trial ASI build-03 SHA`12fe0f89f183e614c3e989a15de05c1f3b85601b9a4d7d0cf26096508a149309`；canonical重编ASI SHA`368d973007ce84e888b346fce43dc32c9527d83c90745039e5e188a3f29611a9`。ABI/export不变不是二进制byte identity。本轮用户已说明main对实际current extraction另核验；公开review应以这两种身份和main最终release pins对齐，不能把旧canonical未合入状态当截点结论。

## 4. r18 与现役的采用边界

| 候选／记录 | 已有结果 | 采用状态 |
| --- | --- | --- |
| FDP 标准档 | 完整离线 process 45.410→43.610 ms，独立 BCCB 约快1.800 ms | 输出有变化，待游戏画面／实机收益；未部署 |
| FDP 自然／电影档 | 单轮约快8.126／7.994 ms | 待独立复测／画面；不可相加 |
| QKV、C512 FFN／encoder、Swin、原 ViT | 完整 process 均负收益；部分逐字节一致 | 未采用 |
| r18 ViT row-once K64 P4／full | 完整 process 分别慢0.893／1.003 ms，13帧输出／历史字节一致、零spill | 未采用；raw graph 约0.054–0.071 ms小差不是完整帧收益 |
| r16 hot admission | 相同未采用模块对照，均值微利但有反向cycle | 不是确认现役收益 |
| native HDR raw2 | 两帧 prepare／verify raw pack/export字节与消费者回收通过 | 未运行模型／游戏／Present；不是整桥验收 |
| Luna37–39 Swin 修复 | 截稿报告记为串行待测／precompile | 没有本 worker 新读 active attempts；未知／未完成 |
| AMD 六项新计划 | 源码和结构预算 | 未测；不是现役实现 |

## 5. 已闭合部分与独立复现仍缺内容

1. main已完成实际G来源源码提取，并给出逐文件manifest；本worker重验本地副本。完整ASI／DLL／config／profile／cache当日部署与构建证据仍按main/release-review记录，不能扩张本worker核验范围。
2. canonical25源合入／promoted／CPU build回执已经取得。已验收trial12fe与canonical368d build分别pin；二进制byte／build closure不由源提取替代。
3. 当前已启用 all6 与 numeric controls 的实际 settings／active／compiler key／capture hit 对照；分数历史在静止帧可能 conditional_unexercised。
4. 本轮用户明确授权gggz114514-oss/dlss5nr-b580，main已告知clone/publish完成。公开部分／私有GPUBlock/DIS／模型／第三方许可按main审查；本worker没有GitHub写入。
5. 隔离环境中的完整安装、冷启动与新进程只读缓存 smoke，及对应输入 fixture、参数和用户画面验收范围。

本档确认源码提取证据与历史闭环，不宣布所有资产完整独立重现，也不宣布r18装入游戏。
