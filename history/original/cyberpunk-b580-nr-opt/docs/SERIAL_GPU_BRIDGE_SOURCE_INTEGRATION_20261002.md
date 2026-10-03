# Serial GPU bridge 源码定点集成计划（2026-10-02）

本任务只新增本说明与 [CPU 集成工具](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/tools/promote_serial_gpu_bridge_source_v2_cpu.py)。canonical 源码、冻结 artifacts/tests、旧 driver、G 安装均未修改；不调用 GPU/API、不加载 ASI/helper/Torch，不更改数学、shader、controls/history/motion 合同。main 已明确选择 gpu-hooks-only，本版固定该 ready plan；实际 promote 必须等待当前 ViT 完整模型 GPU lane 结束，并由 main 明确执行。此交付后停止写入。

## 接受门与证据范围

工具固定 SHA 校验以下历史回执，交叉校验 built source pins、两个 D binary 和实际 run 的九个安装字节身份；G 路径只作为 JSON 字典键，从不读取或写入 G。

- [九文件 installed.json](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-serial-owner-v2-20261002/install-01/installed.json)：两个 checkbox 默认 OFF，源码是否默认开启不随当前运行双 ON 改变。
- [serial MAIN_CPU_REVIEW](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-serial-owner-v2-20261002/cpu-sidecar-02/MAIN_CPU_REVIEW.json)：build03 的 69 源 pin、sidecar02 真实 CPP 540 断言/双持久线程、新 host 23 项。保留原四文件和 distinct-callbacks 修正版原样。
- [owner CPU_BUILD_RECEIPT](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-owner-transfer-v1-20261002/cpu-build-03c41unu/CPU_BUILD_RECEIPT.json) 与 [owner MAIN_RUNTIME_REVIEW](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-owner-transfer-v1-20261002/runtime-2thread-02/MAIN_RUNTIME_REVIEW.json)：真实 transfer ABI、15 busy/4 迁移、byte/非零 motion/正常退出保持历史原范围。
- [run05 MAIN_PROTOCOL_REVIEW](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-live-pair-v1-20261002/serial-driver-run05/MAIN_PROTOCOL_REVIEW.json)：83 pins、270 unique timed frames、90.39 秒/1559 completed frames，串行资格稳定。
- [combined MAIN_PROTOCOL_REVIEW](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-live-pair-v1-20261002/combined-serial-run01/MAIN_PROTOCOL_REVIEW.json) 与 [MAIN_COMBINED_RAW_REVIEW](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-live-pair-v1-20261002/combined-serial-run01/MAIN_COMBINED_RAW_REVIEW.json)：86 pins/120 frames，CPU record 平均减少 1.12383 ms；record-to-retire span 平均减少 2.97105 ms。该 span 含 game 和等待，不能等同 GPU kernel 时间或 Present/FPS。
- [HUMAN_ACCEPTANCE](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-live-pair-v1-20261002/combined-serial-run01/HUMAN_ACCEPTANCE.json)：用户快速转镜头/Alt+Tab 后回复“画面和稳定性都正常”，RTSS“55-60ms”，约 17–18 基础 FPS，约定 FG OFF。接受门要求 human_visual_stability_accepted 与 keep_actual_combined_mode。
- [ON_FOR_USER_REVIEW_03](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-live-pair-v1-20261002/combined-serial-run01/ON_FOR_USER_REVIEW_03.json)：35224→35234、83 pins 不变、hit/bypass 各 10。保留实际双 ON；本工具不切换它。
- 没有自动 Present 逐帧配对；不把全部 RTSS 变化归因于约 3 ms。所有旧 GPU/九文件 byte 证据保持原标签与范围，本次 CPU check 不产生新的 GPU 证明。

ASI built SHA：12fe0f89f183e614c3e989a15de05c1f3b85601b9a4d7d0cf26096508a149309。
helper built SHA：8875a437b478a970a3c5b5da57b9e607d6d025ddd6ac54520e0dc22418e06908。
这两个二进制仅用于历史身份校验，均不进入 canonical source 映射。

## 完整最小映射与 EXPECTED_BEFORE

25 个 source 目标：21 个 serial/pool 集成项与 4 个 shared native/wrapper 项（adapter 取 owner-transfer 的已测版本，避免复制旧 adapter）。保留完整 CMake 依赖，包括 CMake 中 EXCLUDE_FROM_ALL 的 GPU probe 源文件；仅复制源码，不构建/运行该 GPU target。

表中 source/built SHA 均从固定 SHA 的 CPU build 回执读取并对实际候选字节重验。before 是 2026-10-02 读取的 canonical 基线，ABSENT 表示文件应不存在。工具硬编码同一组 EXPECTED_BEFORE；不会以执行时最新值自动重设基线。已有目标只允许 expected-before 或已经等于 proposed-after，其他差异必须阻止并保留。

project 根：E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt。
shared 根：E:/ComfyUI-aki-v3-IntelArc_20260722/.codex-worktrees/re8-fp8-unround-fast-20260928。
shared 仅协调 controls、pre-XeSS hook、handoff controller、wrapper 与三个 native 接口源；模型目录不复制。

| canonical 目标 | candidate 源 | EXPECTED_BEFORE SHA-256 | source/built SHA-256 |
|---|---|---|---|
| [project/CMakeLists.txt](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/CMakeLists.txt) | [serial/CMakeLists.txt](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-serial-owner-v2-20261002/payload/CMakeLists.txt) | ad3119ec1d923f0801466bfda0389f9ab1336d27c719a34535b14d118dd951f1 | f32c3b137824c719b73f533d358a310d1c67f0b57011c70a4bde92fcbe3e33e3 |
| [project/game/cyberpunk_nr_adapter.py](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/game/cyberpunk_nr_adapter.py) | [owner/plugins/cyberpunk_nr_adapter.py](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-owner-transfer-v1-20261002/payload/plugins/cyberpunk_nr_adapter.py) | e05c5749ff5c0978874ea3f2987469c70c0d3859650ac90012de2369f5209938 | 22da70eebf71a1692a2a5a372f220014a66ab13e659394b8a16de0352cacadeb |
| [project/game/cyberpunk_nr_web.py](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/game/cyberpunk_nr_web.py) | [serial/game/cyberpunk_nr_web.py](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-serial-owner-v2-20261002/payload/game/cyberpunk_nr_web.py) | 8589fbb00318c7300f7edd5205de2647345d10356c841ed45024256fb6729093 | 0da630ec90247fb68aafa2544193265031699a5f238be912a4c9e2b917c8a693 |
| [shared/game/nr_game_controls.py](E:/ComfyUI-aki-v3-IntelArc_20260722/.codex-worktrees/re8-fp8-unround-fast-20260928/game/nr_game_controls.py) | [serial/game/nr_game_controls.py](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-serial-owner-v2-20261002/payload/game/nr_game_controls.py) | 8e4a75d295eff4a184df539bda6c96f79415df275bc89c562deb0b25add38400 | 9ba18ac2256fa4459050d5cd98d5376afdf3466f15f187958ae87dee7ed72df2 |
| [shared/game/nr_game_pre_xess_host.py](E:/ComfyUI-aki-v3-IntelArc_20260722/.codex-worktrees/re8-fp8-unround-fast-20260928/game/nr_game_pre_xess_host.py) | [serial/game/nr_game_pre_xess_host.py](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-serial-owner-v2-20261002/payload/game/nr_game_pre_xess_host.py) | 1fec385b544fe42418c2a20b08970589c5fec46b1be5e1c17c2d471c2cdaefbd | 97982805d388eedad2c6c256c325fa2cb009f6316cca1b7dcbc877b4f0bd7338 |
| [shared/game/nr_gpu_handoff_host_v1.py](E:/ComfyUI-aki-v3-IntelArc_20260722/.codex-worktrees/re8-fp8-unround-fast-20260928/game/nr_gpu_handoff_host_v1.py) | [serial/game/nr_gpu_handoff_host_v1.py](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-serial-owner-v2-20261002/payload/game/nr_gpu_handoff_host_v1.py) | ABSENT | 53fd5ab81bb3d9b0b98f5d56def9eb80b6d99881c24cb241c4bf80e81aeeec40 |
| [project/include/nr_gpu_handoff_api.h](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/include/nr_gpu_handoff_api.h) | [serial/include/nr_gpu_handoff_api.h](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-serial-owner-v2-20261002/payload/include/nr_gpu_handoff_api.h) | ABSENT | f4571f153d6c966cdf370c349913334bc082009a0dfc07ded6e6e89cfa7e2ffb |
| [project/include/nr_hdr_resource_pool_api.h](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/include/nr_hdr_resource_pool_api.h) | [serial/include/nr_hdr_resource_pool_api.h](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-serial-owner-v2-20261002/payload/include/nr_hdr_resource_pool_api.h) | ABSENT | 2f927e7187cd440eef2d5ca26d9f311ac4afa78837d0daf7f4296a394b83c64f |
| [project/src/asi.cpp](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/src/asi.cpp) | [serial/src/asi.cpp](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-serial-owner-v2-20261002/payload/src/asi.cpp) | 1db0a22f54a46a80ff020415e5f6cadd72f1e675dc039ffa04ae6ecbca2011b8 | 2543c94aea2080634a1f3dd84987fb706dfdd9559cf788bfcaec542a23ff12b7 |
| [project/src/deferred_identity.cpp](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/src/deferred_identity.cpp) | [serial/src/deferred_identity.cpp](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-serial-owner-v2-20261002/payload/src/deferred_identity.cpp) | d425b66fa31e08129fa92b8da19b54a14a33af62bda641e560a5724a651a0b2e | 4509bf39dac87f9481ec9fc8d01e4a619b64e5c2f3d8ee93c1487b846655ffe7 |
| [project/src/nr_gpu_handoff.cpp](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/src/nr_gpu_handoff.cpp) | [serial/src/nr_gpu_handoff.cpp](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-serial-owner-v2-20261002/payload/src/nr_gpu_handoff.cpp) | ABSENT | 0ba577029b5ab79af64d917db444f5478e8d0326b01a69974cb09f0e2849bb28 |
| [project/src/nr_gpu_handoff.h](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/src/nr_gpu_handoff.h) | [serial/src/nr_gpu_handoff.h](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-serial-owner-v2-20261002/payload/src/nr_gpu_handoff.h) | ABSENT | b6898d17dd0e1d4dc35a51ea55c48fbd7f390386080cbdc7549ebb240f2af958 |
| [project/src/nr_gpu_handoff_reuse.h](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/src/nr_gpu_handoff_reuse.h) | [serial/src/nr_gpu_handoff_reuse.h](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-serial-owner-v2-20261002/payload/src/nr_gpu_handoff_reuse.h) | ABSENT | 23adeedbf06c8fa0b830e42be85213cd5b125187572534f82062e05b630d4ada |
| [project/src/nr_hdr_proxy.cpp](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/src/nr_hdr_proxy.cpp) | [serial/src/nr_hdr_proxy.cpp](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-serial-owner-v2-20261002/payload/src/nr_hdr_proxy.cpp) | 397303b0ca8739b90d5c5aa3b5894bbc44975ea9ab9f517e40c11fd657a2c8eb | 1bac36dfdb54d1a4209a03a3adf70f95a20ff14329659dc5dda32382fba49f4c |
| [project/src/nr_hdr_proxy.h](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/src/nr_hdr_proxy.h) | [serial/src/nr_hdr_proxy.h](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-serial-owner-v2-20261002/payload/src/nr_hdr_proxy.h) | 5f6a3cbc97adbef488c341e0a9485721d4994114ec1fa7f81ba4c5acec86e29e | d0f80bdbfda2d6d097acaaf461bf17306056849472fd9aebc7318dd058303650 |
| [project/src/nr_hdr_resource_pool.h](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/src/nr_hdr_resource_pool.h) | [serial/src/nr_hdr_resource_pool.h](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-serial-owner-v2-20261002/payload/src/nr_hdr_resource_pool.h) | ABSENT | 07eefa76c01d531e879c0dec93552033f3b3e9e6f8642673464d727d23f097eb |
| [project/tests/gpu_handoff_cpu.cpp](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/tests/gpu_handoff_cpu.cpp) | [serial/tests/gpu_handoff_cpu.cpp](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-serial-owner-v2-20261002/payload/tests/gpu_handoff_cpu.cpp) | ABSENT | 88ee1893872a4ffbec1c863aeb2dfe94383a05006b2bdda14ae856a8da18d2a2 |
| [project/tests/gpu_handoff_host_cpu.py](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/tests/gpu_handoff_host_cpu.py) | [serial/tests/gpu_handoff_host_cpu.py](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-serial-owner-v2-20261002/payload/tests/gpu_handoff_host_cpu.py) | ABSENT | 7bb9c6eec56fd18f761f3c5ece828707e8baf04232a8df74c77fb2c728b74f45 |
| [project/tests/hdr_resource_pool_controls_cpu.cpp](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/tests/hdr_resource_pool_controls_cpu.cpp) | [serial/tests/hdr_resource_pool_controls_cpu.cpp](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-serial-owner-v2-20261002/payload/tests/hdr_resource_pool_controls_cpu.cpp) | ABSENT | 85a1a18575946e938f5e1fc60d9cce398a4f40c2e06006febace673d0be7f6d1 |
| [project/tests/hdr_resource_pool_cpu.cpp](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/tests/hdr_resource_pool_cpu.cpp) | [serial/tests/hdr_resource_pool_cpu.cpp](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-serial-owner-v2-20261002/payload/tests/hdr_resource_pool_cpu.cpp) | ABSENT | 02dd176057ce2b91edd18dd504f9482d84a4ac76b6c584d7c555808b29081cbd |
| [project/tests/hdr_resource_pool_gpu.cpp](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/tests/hdr_resource_pool_gpu.cpp) | [serial/tests/hdr_resource_pool_gpu.cpp](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-serial-owner-v2-20261002/payload/tests/hdr_resource_pool_gpu.cpp) | ABSENT | be4c39eacabd03cf8309619b03966c744556e06d2a24f7caaea1386237552d6f |
| [shared/game/nr_texture_bridge_v1.py](E:/ComfyUI-aki-v3-IntelArc_20260722/.codex-worktrees/re8-fp8-unround-fast-20260928/game/nr_texture_bridge_v1.py) | [owner/game/nr_texture_bridge_v1.py](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-owner-transfer-v1-20261002/payload/game/nr_texture_bridge_v1.py) | 01bb3bc05594bfcdf769177595b9d7854136419996a1216abf8dfecddf004662 | 83b1b33f056dfad1fec28e78dfe6daf17414c0b443ddd886c480c491bbc0bb7b |
| [shared/native/nr_gpu_handoff_lease.h](E:/ComfyUI-aki-v3-IntelArc_20260722/.codex-worktrees/re8-fp8-unround-fast-20260928/native/nr_gpu_handoff_lease.h) | [owner/native/nr_gpu_handoff_lease.h](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-owner-transfer-v1-20261002/payload/native/nr_gpu_handoff_lease.h) | ABSENT | cb3cb7f0f4ce3539f02c47d286dd1245359b67491b5d73bd07f979f3f7067eaa |
| [shared/native/nr_texture_bridge_re8_v1.cpp](E:/ComfyUI-aki-v3-IntelArc_20260722/.codex-worktrees/re8-fp8-unround-fast-20260928/native/nr_texture_bridge_re8_v1.cpp) | [owner/native/nr_texture_bridge_re8_v1.cpp](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-owner-transfer-v1-20261002/payload/native/nr_texture_bridge_re8_v1.cpp) | e8bf06e5c6e37da4ab94c030ba8d8829bb8ab4152ca8b7a60b14f314f0cbc375 | e45f5af965912d47b4781672b8e06ab8d0808352a134ec646b342e76416b2ee1 |
| [shared/native/nr_texture_bridge_v1.h](E:/ComfyUI-aki-v3-IntelArc_20260722/.codex-worktrees/re8-fp8-unround-fast-20260928/native/nr_texture_bridge_v1.h) | [owner/native/nr_texture_bridge_v1.h](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-owner-transfer-v1-20261002/payload/native/nr_texture_bridge_v1.h) | fe66075ec1e32943fe58aad32021bc1d45471f49135ffcc2303083027c182076 | ca02d7e1e97054a1808c72833df1ea7de24a116638a61c0add1b3018ba4d7623 |

除 pre-XeSS host 外，proposed-after 与表中 source/built SHA 完全相同。pre-XeSS host 的 proposed-after 为 6d15d11881747b74901971c6df70a1bed870c94d7c78bc86cef9ee290f243117；before/after、四处 diff 和各 build/install 关联在 D PLAN.json/DIFF.patch 中记录，不能把这个投影 SHA 宣称为完整已测 host 的 byte pin。12 个现有文件将在未来 promote 时备份，13 个新文件记录原先 ABSENT；本任务 E source 实际写入数为零。

## 已解决的 pre-XeSS 范围差异

[canonical pre-XeSS host](E:/ComfyUI-aki-v3-IntelArc_20260722/.codex-worktrees/re8-fp8-unround-fast-20260928/game/nr_game_pre_xess_host.py) 的 before 为 1fec385b544fe42418c2a20b08970589c5fec46b1be5e1c17c2d471c2cdaefbd；
[完整候选](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-serial-owner-v2-20261002/payload/game/nr_game_pre_xess_host.py) 为 97982805d388eedad2c6c256c325fa2cb009f6316cca1b7dcbc877b4f0bd7338。
候选还包含当前 canonical 没有的 periodic-flash 诊断代码，来自此前 G 安装父版本，并非本次接力必需。main 已复核此差异并明确要求 gpu-hooks-only；最终 ready plan 只在上述 pinned canonical before 上投影四处 hook，保留原 host 的所有其他诊断、模型、控制和历史字节。

固定的四处实际已测调用为：
重建前 ensure_idle_before_reinitialize；取实际 source_handoff producer 点；
SourceFrame 传入 producer_fence/value；失败时 invalidate_after_failure。
工具独立验证这些四处 edit 应用到 trial 的 pinned 原安装父 host 后，与完整 candidate 一致（比较时只归一换行）；该父文件 SHA 为 d613d1ba88dd5c60833e48a2bc00778b9a274f775459319d46df4c548f0a7100。逆向 edit 恢复 canonical before 的每一个原字节，reverse SHA 为 1fec385b544fe42418c2a20b08970589c5fec46b1be5e1c17c2d471c2cdaefbd。PLAN 的 host_projection_proof 记录这两个布尔证明与完整 SHA，既有换行保留；失败 hook 明确锚定 process 的 output return/except，不进入 retire。

该投影只做 Python 内存编译与精确 scope 检查，尚无投影 canonical host 的独立 GPU 运行。ViT lane 结束后，main 合入时仍需对最终 canonical 集成做 CPU build/契约审查；这不会改写已有九文件 GPU receipt。本工具没有 force/full-host 绕过门。

当前主工程 CMake/asi/deferred/HdrProxy/adapter/web 与 shared controls before 均匹配此前 bridge-cache-integrated 候选，未发现它们另有不相关修改。未来任何 before 漂移会按目标、预期 SHA、实际 SHA 报冲突，不删除或覆盖。其余未映射编译依赖要求与 tested build 保持一致。所有路径拒绝 symlink/junction/reparse；canonical hardlink 也拒绝。

## Authored 输入与 CMake 构建输出闭包

实际审查的是 source/built SHA 为 f32c3b137824c719b73f533d358a310d1c67f0b57011c70a4bde92fcbe3e33e3 的 [候选 CMakeLists.txt](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-serial-owner-v2-20261002/payload/CMakeLists.txt:90)。它只有一个 add_custom_command(TARGET CyberpunkNRBridge POST_BUILD)，逐项内容是 copy_if_different 两个 Python authored 文件到 TARGET_FILE_DIR；没有 add_custom_command OUTPUT/DEPENDS，也没有生成 shader header 的规则。OUTPUT_VARIABLE（git commit 查询）与 OUTPUT_NAME（目标名称）不属于生成头文件声明。

| 输入/输出 | 分类与处理 | SHA-256/规则 |
|---|---|---|
| shaders/nr_hdr_proxy.hlsl | authored shader，保留 canonical 原字节，纳入未修改依赖 pin | 5aabd2fda6062983a8976a888ad90ec692997ce653dd8637489efa564fcae5ad |
| src/nr_hdr_proxy.h | authored header，nr_hdr_proxy.cpp 第一行 include，列入精确 source 映射 | d0f80bdbfda2d6d097acaaf461bf17306056849472fd9aebc7318dd058303650 |
| game/cyberpunk_nr_adapter.py 与 game/cyberpunk_nr_web.py | POST_BUILD 的 authored 输入；两者均保留 source/build pin | 上方映射对应 SHA |
| TARGET_FILE_DIR 内的两份 Python copy | 生成的 build 目录副本 | 不是新增 authored 依赖，不进入 source 映射 |
| vcxproj、obj、ASI/helper 二进制 | CMake/编译输出 | 本任务不生成或 promote；仅校验已有 D binary 身份 |
| shaders/nr_hdr_proxy.h | 不是此 recipe 的 authored 输入或生成输出 | 前次解析误截 `.hlsl` 后缀得到的路径，拒绝记录仍保留 |

工具以完整扩展名与词边界解析 40 个 local authored CMake 引用，要求每项都有 build source pin 并被映射或保持 canonical 原字节。69 个 serial build 源 pin 全部重验，未映射的 48 个 canonical 输入也全部重验；没有通过排除 shader/header 来消除错误。custom-command 的全部 tokens 必须与这份 pinned recipe 相同，任何新增 OUTPUT/DEPENDS/输入规则都拒绝并要求新的范围审查。

原有 optional re8_panel_smoke 的条件为 ../re8-b580-nr-xess/game/nr_game_controls.py；它保留为外部既有条件，没有误映射成本项目 game 文件。既有 OptiScaler SDK/header/Detours 依赖和 audited commit 7534ad00bf9e590eedb99e8dd9fd8c89dae3654f 保留在 recipe 中，本侧不配置 SDK 或执行 CMake。GPU pool probe 的 EXCLUDE_FROM_ALL/无 CTest 注册保持原样。

## main 的 CLI（本任务只调用 check）

PowerShell/stdlib Python 3.12：

```powershell
$serialPromoter = 'E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/tools/promote_serial_gpu_bridge_source_v2_cpu.py'
$serialPlanOut = 'D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-serial-owner-v2-20261002/source-integration-v2/cpu-ready-01'

# 本侧执行一次并冻结的最终 check。该目录已有交付时，复核请用 stdout 或新 D 目录。
python -B $serialPromoter check --pre-xess-policy gpu-hooks-only --output-dir $serialPlanOut

# 仅 ViT full-model GPU lane 完成后，main 明确授权 source 合入并核对 plan/diff 后调用：
python -B $serialPromoter promote --plan "$serialPlanOut/PLAN.json" --plan-sha256 '<main-reviewed-plan-SHA256>'
```


每次 output-dir 必须是新的 D 子目录，不重写旧回执。check 无 output-dir 时只输出 stdout；未显式传 gpu-hooks-only 的默认 pending-main 仍 exit 2，不把历史拒绝改成 pass。Python compile 只在内存进行，不 import/执行候选模块、不生成 pycache。C++ 编译留给 main；没有 API/G/GPU 操作或接受默认值改变。

promote 验证 tool/doc/所有证据与 exact plan SHA，以及 PLAN 内固定的 DIFF.patch SHA；白名单只含上述 25 个 E source。
所有更改项先在新的 D promotions/<UTC-UUID>/before 备份，并写 before.json；
重新检查 source/baseline/protected pins 后才原子替换 E，逐项与最终 readback。
失败则反序恢复旧字节或移除本任务新建的白名单文件，并写 rollback.json；
成功写 promoted.json。若发现外部并发修改，保留外部修改并记录 rollback_incomplete，
D 备份仍可恢复；不以“rollback”名义盲覆盖它。不递归删除/移动、不写 G、
不复制/install DLL/ASI、不启动 GPU。不支持 force、不自动改默认值。

## 本侧 CPU 结果

gpu-hooks-only 预检 exit 0：25 映射、12 existing/13 ABSENT、48 preserved canonical pins、40 authored CMake 引用、8 Python 内存 compile。四处 hook 的 tested-parent 等价与 reverse-to-original-byte 两项证明均 true；future promote 的 payload/闭包重验也只读通过。没有未解决的 canonical 冲突。CXX 未由本侧编译，投影 host GPU 未由本侧运行，actual promote 未执行。

最终不可覆盖交付位于 [cpu-ready-01/PLAN.json](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-serial-owner-v2-20261002/source-integration-v2/cpu-ready-01/PLAN.json) 与 [DIFF.patch](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-serial-owner-v2-20261002/source-integration-v2/cpu-ready-01/DIFF.patch)。PLAN 包含完整 source/built/before/after pins、接收门、tool/doc SHA 与 DIFF SHA；该目录的 CPU_READY_RECEIPT.json 固定最终 check stdout/exit、输出 SHA 和只读 readback 结果。main 可在不写 E/G 的前提下复核这些文件。

[CPU_REJECTED_HISTORY.json](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-serial-owner-v2-20261002/source-integration-v2/rejected-checks-01/CPU_REJECTED_HISTORY.json) 原样保留四次前期 tool 输出：错 exception hook 锚点、optional sibling 误分类、.hlsl 被误截成 .h、默认 pending-main。前三次是开发迭代中的本侧拒绝，第四次是真实范围决策门；没有伪造重跑或将旧失败回执重标成通过。各迭代 source SHA 当时没有捕获，记录中明示此限制；第四次 PowerShell 外层没有传递 Python exit 2，原 tool exit 1 与真实 JSON status 一并保留。

本任务最终仅两个 E 新文件和本任务 D 计划/拒绝证据新增；canonical/shared/G、冻源、driver、ViT、原 83/155 pins 未写入。交付后严格冻结。
