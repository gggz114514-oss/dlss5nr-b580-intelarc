# 720p 现役候选：GPU 图内与图外成本

范围：实际《赛博朋克》DLSS→NR→XeSS 桥，720p、C512＋K8、去舍入、融合历史、真实运动、图重放，帧生成关闭。基准勾选旧三项及 `num_front_both`；候选只追加一项。用户保持同一实机场景和静止镜头，Luna执行测量，主线程复核完成回执及真实安装源码。

## 共用修复与扩大复测

用户要求覆盖所有删减后减速的候选，并单独安排 GPT‑6.1 Sol max 排查静止画面每隔约 5–10 秒闪一下的问题。之前 C64 和 ViT 是成本拆分的代表，两者不能代替其余候选的实测。

E 源码新增 `numeric_frame_validation_720_v1.py`：周期在真实 Cyberpunk adapter.process 的串行锁内建立，不跨帧缓存；首个合法阶段仍完整检查，同阶段入口复用成功结果，成功返回前清空周期并进行完整终检。父 suite 安装时对 decoder_input、branch_accum、ViT、history、post、front 六类绑定共同 validate 入口；`_SessionCounter` 的独立 fixed-owner 入口也接入。Sol 复核发现其它族绕过 validate 的直接重检查，主线程继续接入 History._validate_fixed_owner、ViT._guard、Post/Front._guard_fixed；保留外层原线程门和会改变语义的 fresh/dispatch/sources 参数，区分 in-frame/live/retired 与实际 front/post/sigmoid callee 阶段。每个阶段的首次检查和成功终检仍存在，不宣称全部扫描消失。

网页新增独立“合并重复校验”开关及 `/api/validation`；切换只改变 CPU 验证策略，不改变模型设置、历史、核、GPU 图或算术。适配器成功终检计入 process_wall_ms，线程交接前缀与探针读数尾段单列，不重复相加包含式 parent/child 校验。

已完成 CPU 检查：frame helper 扩展至 23 项；原有 suite/decoder/ViT/post-front 线程移交共 46 项；adapter/web 协议 16 项。包含 wrong-thread 原异常、失败回滚、无周期回退、持续的帧内权重变更终检、下一帧首检、合法 owner 退休、同 owner 身份变更失败、原 History/Post 外层线程门、不同 dispatch flags/帧阶段不误复用，以及开关不重置历史/图。fresh/source 显式重检查不缓存；dispatch 命中仍检查 live、in-frame、preflight、当前 policy 和 backend；projection_owner.apply 的变化单独进入阶段键。命中仍执行 child._require_fixed_session（有此接口时）；资源重核的拒绝时点从部分中间入口移至阶段首检和终检，短暂替换后恢复且元数据不变不能由首/终检查排除，不声称与逐入口扫描检测时点完全相同。本节不把 CPU 测试当作实际游戏通过。

`tools/compare_validation_batch_720_v1.py` 支持两层验证：同一捕获图 off/on/on/off，以直接完成帧及固定 owner epoch 验证 CPU 去重收益；`screen --candidates all-slow` 在修复启用时重新筛选 20 个此前减速选项，夹在前后 reference 四项之间，区分图内与图外成本。小收益可保留，不设置 1 ms 门槛，筛选结果不直接相加。GPU 执行、切档和监看由 Luna 串行承担。

备份安装器仅写显式的 Python 运行文件，拒绝游戏仍运行、目标重定向或原文件 SHA 改变；不写插件目录的 nr-runtime junction，不改核/缓存/模型/签名 DLL。用户正常退出后已完成 G 安装，备份和安装前后 SHA 回执位于 `live-web-perf-20261001/validation-batch-install-v1-20261001/`；尚无修复后实机结论。

最终 dispatch/projection 防护补丁的备份位于 `validation-batch-dispatch-fix-v1-20261001/`，实际 helper SHA 为 `b21625145f0efc5a33ca9aa9bbafb717d7b6fee457b4287ede4bbf4f53aca05e`。同一启动另装入被动闪动诊断，安装回执位于 `periodic-flash-v2-installed-20261001/`，仅四个诊断文件。对应 ASI SHA `2f8c9e8450f32da31efeb01f2e8697d2cb6e7a5e66a12a23d593620635e7be9d`；host reset、模型和纹理桥调用的 AST 保持不变。配对脚本把诊断 ASI/host/helpers 纳入源冻结。子进程以 `CYBERPUNK_NR_COST_METER=1`、`NR_DIAG_PERIODIC_FLASH_V2=1` 启动，GPU 测量/网页切换由 Luna 承担；目前等待用户进入同一静止场景，不把成功安装当作性能结论。

## 首轮实机校验开关配对：七组完成，第八组缓存失败

Luna 完成了 reference、C64、C128、C256、ViT FP32 分母、decoder full K、post sigmoid 的 OFF/ON/ON/OFF 配对，每臂 30 个唯一样本，每组 GPU cost epoch 不变，ON 快照与 cost frame ID 30/30 精确相同。OFF 原接口不提供 frame ID，回执保留此限制。完整数据位于 `validation-batch-live-v1-20261001/`；源/config 冻结，诊断开销双方共同存在。

下面是**同一候选内部校验 ON 减 OFF**，不是候选内核减现役基线；不能按此选择 C64/128/256 上线。完整适配器为顺序墙钟 `process_wall_ms + adapter_handoff_cpu_ms + observer_finish_cpu_ms`，不再叠加 GPU 或包含式 guard。

| 固定计算组合 | 适配器 ON−OFF，ms | 线程交接 ON−OFF，ms | GPU 图 ON−OFF，ms | 两个 ON 的适配器都低于两个 OFF |
| --- | ---: | ---: | ---: | --- |
| reference 四项 | -0.886 | -0.162 | -0.013 | 是 |
| 加 C64 | -2.894 | -0.896 | +0.098 | 是 |
| 加 C128 | -1.344 | -0.922 | +0.164 | 是 |
| 加 C256 | -2.380 | -0.822 | -0.043 | 是 |
| 加 ViT FP32 分母 | -0.189 | -0.264 | -0.073 | 否 |
| 加 decoder full K | +0.138 | -0.219 | -0.066 | 否 |
| 加 post sigmoid | -0.227 | -0.262 | -0.113 | 否 |

C64 四臂适配器分别 51.266、48.974、48.624、52.120 ms；OFF 漂移 0.855 ms，两个 ON 均明显低于两端。C128 为 50.967、49.601、49.907、51.230 ms，C256 为 52.057、48.834、49.529、51.066 ms。线程交接在三个 Branch 的两个 ON 都降低。不能仅因存在漂移就把明确的 CPU 减负判为无效；也不将一次短配对的整个差值保证为所有场景恒定收益。

`numeric_guard_cpu_ms` 的包含式值反而增 0.047–0.188 ms，新增成功终检约 0.23–0.69 ms 已包含在适配器中；只比较该单项会漏掉独立作用域与交接节省。GPU 图未更换，图区间差异不能当作校验开关改进 GPU 算术。剩余小差值受纹理准备和模型墙钟波动影响，尚未确认稳定净收益。

第八组 `num_history_direct_pixel` 初始化失败：`_select -> numeric_cleanup.preflight -> history.preflight -> spill_preflight.select -> jit.warmup -> disk_only` 缺 `_five_tap` specialization，触发原生产 `NR_PROCESS_FAILED` 锁存。不是校验 helper 的身份/线程拒绝，也不是数值错误。已保存 `stopped-checkpoint.json`，20 项 screen 和固定 90 秒自然闪动采集均未运行；切档异常期间的 raw fallback 不用于证明自然闪动。用户已正常退出，Luna 将一次离线准备全部待筛组合及需要的分支 specialization，经过 disk-only 自检后再重启继续；已完成七组不因新缓存重复重测。

离线准备首轮未启动 GPU：旧 v7 source-freeze 仍指向安装校验合并与被动诊断前的源码，被 `Frozen E source hash mismatch` 拦下。主任务已按已完成的安装回执生成 v8 冻结与当前安装清单，保留旧 v7；未直接接受任意源码变化，也未改内核运算。实际 host 固定到与 G 相同的 E v2 stage，不能把未安装的旧 E host 当作实际运行代码。原 runner 的 CPU 前置校验通过：36 份源码、38 个安装目标、665 个种子缓存文件；未导入 Torch/Triton。准备驱动也修复了失败后无法复用既有快照的问题，重试只使用新输出目录。

当前准备范围为现役四项加 20 个逐项候选，以及历史数值/坐标/倒数的 18 个兼容组合；CPU 注册表检查通过。v8 文件位于任务 D 根目录的 `source-freeze-validation-flash-v8-20261001.json` 与 `source-installed-validation-flash-v8-20261001.json`。Luna 已重新获离线 GPU GO；此处不把前置校验或任务派发当作预编译成功。

首个 v8 prepare 进程在只读的 `G_baseline_13` 缺 `_entry_kernel`，尚未到达候选。隔离缓存原先仅从候选导出集合构造，漏了现役 post 基线缓存；主任务已按实际 G 缓存逐文件 SHA，仅补 `_entry_kernel` / `_entry_mlp_kernel` 家族的四组、28 个文件、758,979 字节到 D 快照，新组的 child_paths 指向 D，原有组与 G 文件均不改。回执为快照目录的 `installed-baseline-seed-receipt.json`。已继续交 Luna 新进程准备，不在游戏内放开临时编译。

随后 v5 的精确查找记录证明缓存键无误：`eb2156eb8f0d7bcd15dabcadaa7ccd6399607804374fd69270eeb5f24bed14bc` 对应 720p post `_entry_kernel` 的 768×1280 / FP16 / ROUND_INPUT=False。真正遗漏原因是 Triton 失败查找已自动创建 `5MQVN24...` 空目录，而补缓存脚本仅检查目录存在就跳过；前述四组实际不含此组，不能把复制成功当作本次缺失组已补齐。脚本现在按可加载的 group marker 和全部 child 文件判断完整性，已有不同内容的文件仍拒绝覆盖。v2 补齐该组七个文件、158,725 字节，全部源/目标 SHA 留在 `installed-baseline-seed-receipt-v2.json`。使用实际 FileCacheManager 的 get_group 源码方法进行纯 stdlib 读取，六个 child、哈希和 D 路径均通过；证据为 `baseline-cache-loader-v2.json`。G 和内核运算均未改。Luna 获得 v6 顺序准备 GO；此时还没有 38 项新进程 GPU 验证结论。

v6 越过 post 后，baseline 的 C128 成对 preflight 又缺 `_project`：实际 key `7cfed67f6991451ed22c281826a37556df93d2d5b1d093617227ad7deca19dcb`，M=15360 / C=128 / BM=16 / BN=64，G 已有完整同键缓存。驱动现按实际缺失 key，在 **prepare 阶段**从 G 验证并复用完全同键的缓存到 D，而不只补一个核名家族；原 source、specialization、编译选项和 DiskOnly 守卫不变。逐文件 SHA、marker/metadata 目标路径和复制预算均入回执，已有不同文件拒绝覆盖；readonly 和 verify-G 不启用此功能。主任务纯 CPU 补齐该 `_project` 的七文件、199,058 字节，回执 `exact-group-seed-project-cpu-v1.json`。Luna 在 v7 新进程继续全部准备，成功与否仍以完成回执为准。

v7 完成全部准备：复用 116 个同键已安装组到 D，812 文件、34,828,397 字节；14 条实际缺失 key 由限定候选 JIT 编译补齐。prepare 的每行均为 `cache_prepared_unvalidated`，整体退出码 1 是原 runner 防止把准备当验收的约定，不作为本轮失败结论。随后 **fresh readonly 全部通过：20/20 候选、18/18 历史组合；0 缓存缺失、0 编译、缓存指纹未变化、基线 G/E 逐字节相同且重复稳定**。结果为 `readonly-slow-v7-20261001/RESULT.json`，SHA `8e066abe9a09b8ca38d8a8b97764441afc0b4e4038eef4dd20fa67eea687b0f7`。性能筛选和自然闪动采集仍待安装及游戏启动；不能把离线缓存验证当作速度收益。

并行 CPU 复核还确认 exporter 将同一已安装 group 的 JSON 重新紧凑序列化会造成无意义的安装 SHA 冲突：PT7NM `_project` 的 source/IR/binary 完全相同，metadata 除路径字段完全相同，marker 映射到 G 后结构相同，但空格/换行改变了文件 SHA。只在 compiler key、非路径 metadata、全部 source/IR/binary SHA 和完整目标 marker 路径均相同的条件下，导出器应复用 G 原 JSON 字节并记录证明；installer 的严格 SHA 和拒绝覆盖保持。该修复由 Sol 实施，export/install 暂缓到 CPU 检查完成。

canonical 修复已实现并由主任务审阅：真实 PT7NM 及篡改 binary/metadata/marker、缺损组等 10 项 CPU 检查通过，G/D 原缓存指纹不变；回执 `canonical-installed-reuse-cpu-v1-20261001/RESULT.json`。完整 readonly 报告的首次 CPU 导出越过原 JSON 冲突，但 front 专用解析器对组合条目错误限定 declared D root，将已审计 G 共享组判为路径逃逸，使 18 个组合显示 `missing_cache_evidence`。组合本身的 DiskOnly specialization/hash/file SHA 和实际 front reset/history launch 都在各自回执中，非缺少 GPU 执行证据。正在将此专用解析器与通用 exporter 的 approved shared-child-root 合同统一；不得通过取消 38 项验收门或借用其他候选回执绕过。GPU 已通过的 38 项不因纯导出修复重跑。

front 解析范围已按上述合同修正，主任务审阅通过。最终 CPU 导出 `cache-slow-v7-front-shared-root-export-cpu-v1-20261001/manifest.json`，SHA `10b4da56050aa996afdb2fafca8367c8db8805ae915ae3ec6b706f5a943093ad`：38/38 全部导出、96 组、674 文件、23,062,972 字节，missing 两表为空；82 组保留已安装 JSON 原字节。Luna 获得该清单的严格 additive 安装及 G 新进程 readonly 验证 GO，不重复准备或 D readonly。尚待此最后实际 G 验证完成后启动游戏。

**本轮缓存闭合**：`install-slow-v7-canonical-20261001/completed-receipt.json` 证明只追加 98 文件，576 文件复用原字节，674 个导出文件的目标 SHA 均匹配。`verify-g-slow-v7-20261001/RESULT.json` SHA `38eca2fe77c42360f6060c746000331566d73bf448b74dfbf5ae298443e021c9`：实际 G fresh process 的 20/20 和 18/18 全通过，0 缺失、0 编译、缓存指纹未变化；verify-g 没有 prepare seeding。主任务已复核完整回执。

2026-10-01 15:06 +08 主任务启动 Cyberpunk，子进程启用 cost meter 与 passive flash v2，PID 11056；启动回执 `live-web-perf-20261001/launch-after-cache-v7-20261001.json`。目前等待用户回到同一静止场景，FG 关闭、DLSS 性能档。Luna 下一轮只补第八个 history_direct_pixel 校验配对，再筛选 20 项，最后固定 reference 四项采集 90 秒自然闪动；此前七组保留。缓存阶段成功不构成新的实机提速/闪动根因结论。

## 修复启用后的 20 项实机筛选（2026-10-01 15:24 完成）

用户进入同一静止场景后，Luna 补完第八组 `num_history_direct_pixel` 校验 OFF/ON/ON/OFF：同图、每臂 30 个唯一完成样本。ON−OFF 的完整适配器为 -0.708 ms，线程交接 -0.110 ms；准备阶段波动也贡献约 -0.301 ms，不能将整个差值解释成校验代码的固定节省。

随后固定校验 ON 完成全部 20 项筛选，每次追加一项，至多四项夹在前后 reference 四项之间。每臂 30 个直接完成样本、恰好一次图重放；源/config 冻结，结束恢复 720p reference 四项及校验 ON，确认四个新完成帧、健康无失败。主任务只读复核完成回执，没有运行或监看 GPU。

差值均为候选减前后 reference 均值，负值表示更快。适配器是 `process_wall + handoff + observer_finish`，不是游戏基础帧时间；GPU 图包括内部访存，不能再相加到适配器墙钟中。

| 追加候选 | GPU 图差值 ms | 完整适配器差值 ms |
| --- | ---: | ---: |
| 解码合并原生乘加 | -0.037 | +0.301 |
| 输出 FP16 融合乘加 | +0.151 | +0.021 |
| 解码输入完整 K | +0.117 | +1.571 |
| C64 分支原生累加 | +0.123 | +3.023 |
| C128 分支原生累加 | -0.089 | +2.722 |
| C256 分支原生累加 | -0.007 | +1.700 |
| ViT 分母顺序融合 | -0.234 | +0.599 |
| ViT 分母 FP32 归约 | -0.024 | +0.611 |
| ViT 归一化融合乘加 | +0.043 | +0.393 |
| ViT 指数融合乘加 | -0.210 | +0.697 |
| ViT 归一化＋指数融合乘加 | -0.129 | +0.725 |
| ViT 输入投影完整 K | -0.597 | +0.431 |
| ViT 输出投影完整 K | +0.038 | +1.541 |
| ViT 零指数常量 | -0.079 | +0.688 |
| 历史尺寸倒数 | +0.047 | +0.745 |
| 历史直接像素坐标 | +0.039 | +0.407 |
| 历史权重倒数 | +0.201 | +1.889 |
| 输出亮度曲线原生计算 | +0.129 | +2.672 |
| 输出就近舍入 RNE | -0.499 | -0.761 |
| 输出截断舍入 RTZ | -0.286 | +0.634 |

ViT 输入投影、分母顺序融合、指数融合乘加及输出 RNE 的 GPU 图均低于所在组前后基准；这是图内有效实现收益的短筛选证据。RNE 也是唯一完整适配器配对均值降低的追加项，但该组适配器基准漂移 1.685 ms、准备阶段 -0.524 ms，尚未证明其 -0.761 ms 是稳定净收益。没有 1 ms 判别门，也没有将图内小收益丢弃；有益项需要剥离残余 CPU 开销及适当配对确认，个别差值不可直接相加。

C128 已降低部分重复校验，但相对 reference 的 branch guard 总量仍增加 1.093 ms、线程交接增加 0.530 ms；其 GPU 图实际快 0.089 ms，完整适配器仍慢 2.722 ms。这证明帧内复用修复没有消除全部实验作用域/元数据扫描，不能据此把这些核都判为 GPU 实现低效。主任务已安排独立 GPT-6.1 Sol max 在未安装 E stage 审查剩余同帧 phase 与局部校验复用；不改变本轮冻结代码、不进行跨帧缓存、不将 CPU 检查说成 XMX 矩阵计算或像素纠错。

完成回执：`validation-batch-screen-v7-20261001/completed-checkpoint.json`，SHA `8e9c90d7a3026c428db257f687eaec9e29a9d22b21e04b469d2c6b3b2416e4c7`。第八组位于 `validation-batch-history-directpixel-v1-20261001/`。固定 reference 的 90 秒自然闪动采集由 Luna 随后串行执行，筛选切档产生的历史重置不作为自然闪动根因。

## 已闭合结论

**已拆测候选相对基线增加的损失主要来自每帧重复CPU校验与线程交接，不是新增完整张量格式转换，也不能把候选的整帧回退都当作原生GPU核变慢。** 这是候选增量的归因，不表示整个 NR 模型主要耗时在 CPU。C64复测、ViT首轮分别给出了以下证据。GPU图区间仍包含内部读写；没有据此宣称全部原生核更快。

| 已闭合配对，候选减基准，ms | C64 独立复测 | ViT 分母首轮 |
| --- | ---: | ---: |
| 完整适配器顺序墙钟增量 | +3.014 | +0.640 |
| 网络图GPU区间增量 | +0.118 | -0.078 |
| 帧内候选validate总量（不再叠加numeric parent） | +1.391 | +0.245 |
| 线程交接前缀增量 | +1.174 | +0.253 |
| 纹理准备 / 回送增量 | +0.009 / +0.006 | +0.027 / +0.018 |
| 前后基准漂移：host墙钟 / GPU图区间 | +0.210 / +0.091 | -0.059 / -0.019 |

C64复测：四臂各60个直接完成样本，实际命中且单次replay；基准GPU40.669/40.760ms，候选40.787/40.878ms，候选均慢于两端但最小间距仅0.026ms。基准完整适配器48.024ms，候选51.038ms。新增候选validate与线程交接合计约2.565ms，占3.014ms整段增量的大部分；没有对包含式parent重复相加，也没有把GPU区间加到墙钟上。图内小幅回退仍保留在结论中，不冒称核完全持平。

复测四臂的图外static input-copy均为57,262,080逻辑字节，output.clone均为5,529,600逻辑字节；这些是既有共有开销。准备和回送只差约0.015ms，不能解释3ms损失。图内未测的copy/计算比例仍不能凭逻辑字节计数推定。

独立复测回执：`cost-decomposition-branch-repeat-v1-20261001/num_branch_c64-completed.json`及`completed-checkpoint.json`。源/config SHA冻结成立，末尾恢复reference四项及实际replay，三个新完成帧已确认。

## 首轮已完成拆计

每组B-C-C-B，每臂60个直接完成帧；所有臂均有实际候选命中和恰好一次网络图replay。表中差值为候选减基准的配对均值，单位ms。

| 项目 | C64 分支原生累加 | ViT 分母 FP32 归约 |
| --- | ---: | ---: |
| host.process墙钟（不含适配器线程交接前缀） | +2.606 | +0.385 |
| 网络图GPU区间，含图内计算与读写 | +0.965 | -0.078 |
| numeric validate_context CPU区间 | +0.451 | +0.123 |
| 候选child.validate帧内总区间 | +1.362 | +0.245 |
| 适配器线程交接前缀CPU区间 | +1.106 | +0.253 |
| 图常量校验CPU区间 | -0.054 | -0.041 |
| 纹理准备墙钟 | -0.006 | +0.027 |
| 纹理回送墙钟 | +0.013 | +0.018 |
| 观察器结束读数CPU区间 | +0.002 | +0.001 |
| 完整适配器顺序墙钟增量：前缀＋host.process＋结束读数 | +3.714 | +0.640 |
| 前后基准漂移：host墙钟 / GPU图区间 | +2.144 / +1.951 | -0.059 / -0.019 |

C64首轮的前后基准GPU由38.773升到40.724ms，而两个候选为40.690和40.736ms；不能把这组+0.965ms直接判为核实现必然更慢。上面的独立复测已闭合这一风险，主要CPU损失再次出现，GPU回退仅约0.118ms。首轮原始数据保持原样。

ViT组前后基准GPU为40.709和40.690ms，两个候选为40.680和40.563ms，图内有约0.078ms小收益；完整适配器反而慢约0.640ms。该候选至少不能被统一归类为GPU实现变慢：生产CPU校验/线程交接明确增加，吞掉图内小收益。现有结果不表示其他候选也都快，不表示纯矩阵性能提升。

## 原计时标签的解释修正

`process_wall_ms`从进入host.process开始，不含适配器的线程/stream交接前缀；首轮仅比较host墙钟会漏掉这一部分。前缀、host.process、结束读数是顺序的CPU墙钟，可合成适配器总墙钟。GPU时间及提交CPU时间可能重叠，不能再加到这个总数中。

`numeric_guard_cpu_ms`是validate_context自身的包含式时间；`child_*_guard_cpu_ms`是host.process内该child.validate的**所有调用总和**，既包含validate_context下的调用，也包含独立作用域入口/出口校验。安装探针原说明把child都称为nested不准确；child总量不能再与parent相加，不能把child大于parent视为设备计时故障。线程交接发生在计时frame之前，其内部child校验已包含在前缀墙钟中。

网络图时间戳不覆盖图外静态输入copy和返回output.clone，覆盖捕获图的内部内存操作。现有逻辑字节计数不能当作copy耗时/显存带宽；因此没有宣称已把图内纯算术与访存完全分开。

## 已定位的C64重复校验调用链

实际安装文件均位于 `G:/SteamLibrary/steamapps/common/Resident Evil Village BIOHAZARD VILLAGE/nr-runtime/game/`，没有用旧普通modules替代现役overlay。

- `nr_game_fullsize.py:672` 调用numeric validate_context；`numeric_cleanup_suite_720_v1.py:85`调用各child.validate。
- `decoder_input_full_k_720_v1.py:623`及`:636`是Branch继承的wrapped_scope入口/出口validate。故稳态host.process内至少三次完整Branch validate。
- `decoder_input_full_k_720_v1.py:523`在线程交接中直接调用_validate_fixed_owner，`:525`再调用validate；`numeric_cleanup_suite_720_v1.py:155`随后再做一次suite validate_context。线程更换时另加三次重检查。
- Branch的_validate_callees每次重建36个MLP目标并核对C128成对调度，保存144个权重槽；局部_scope入口还单独调用_validate_callees。上述为CPU元数据/绑定检查，并非CPU帮GPU纠正数值。

这说明完整候选实验夹带了可观的生产每帧验证成本。后续应先在保持必要线程/作用域/生命周期边界下合并重复检查，再测候选GPU和整帧；不能依据未剥离的整帧回退结论永久否定原生算术，也不能直接删掉所有线程与资源边界守卫。

### 后续实施边界（用户确认，2026-10-01）

CPU 元数据审计不会纠正输出像素，也不是 XMX 的计算要求。固定权重、内核源码/编译指纹和固定模块清单应在加载、切档、建图阶段完成审计；不能因它们曾用于精确分支验收就默认每帧重查。当前尚未把这项后续改动装入游戏，旧配对数据仍对应上述冻结版本。

图重放热路径只保留实际会变化且影响当前调用的条件，例如当前线程/序列锁、会话及图代次、资源生命周期和有效帧归属。需要区分“对已经固定内容的重复审计”和“运行时失效检测”，不得把两者统称为不可删除的安全检查。若候选精简导致减速，应拆计其 CPU/GPU 增量，不能继续用未剥离实验审计的整段耗时否定原生内核。

2026-10-01 第一批实现已交付到独立 `artifacts/replay-lifecycle-audit-base-v1-20261001/`，仅迁移 DecoderInput 与 BranchAccum 的固定审计。10 项 CPU 合成元数据用例通过：真实实际 descriptor key 的图 entry 命中及线程 handoff，固定 symbols/matrix/weights/binary/targets/schedule 扫描增量为 0；冷建图前后各完整审计，保留当前实际 caller、序列锁、线程、资源归属与退出回滚。GPU 数学、输入 copy、图重放和输出 clone 未改。显式诊断仍完整审计，默认关闭，尚未安装或测得毫秒收益；History/ViT/Post/Front 和 GraphFront 常量重查仍待后续迁移。见该 stage 的 `REPORT.md` 及 D 中 `replay-lifecycle-audit-base-v1-cpu-20261001/cpu-receipt.json`。此交付不混入当前桥闪动修正版的实机验收。

## 证据与安装边界

完成回执：`D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/live-web-perf-20261001/cost-decomposition-live-v1-20261001/`，含各臂样本、两个candidate-completed和completed-checkpoint。所有源/config SHA在配对前后相同；末尾已恢复reference四项、实际replay及三个新完成帧。

仅本次启动的子进程启用CYBERPUNK_NR_COST_METER；安装的是两份Python适配文件，没有改共享后端、缓存或签名DLL。九项CPU协议检查通过，Sol复核异常隔离和绑定。普通游戏启动默认不开该探针。
