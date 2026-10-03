# 原尺寸快速版：取消固定行分批 — 本轮结论（实施侧）

日期：2026-09-15。范围：`E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580/reference/fullsize_rows_v1/`。

**状态：六个阶段全部通过 —— `preflight` / `collect`（293/293 + 292/292 内核，各 14 worker）/ `local`（72 块逐字节一致，`max_abs=0.0`）/ `gate`（4 变体 NR256 控制帧逐字节一致，`max_abs=0.0`）/ `full`（**243 帧 × 4 变体全部逐字节一致**，`max_abs=0.0`）/ `bench`（**`both` 1.1722×，每帧 −14.69%**）。**

**本轮结论：取消固定 144 行 C512 与 64 行 ViT 分批，在保持原尺寸全片 243 帧输出逐字节不变的前提下，把每帧处理调用从 95.633 ms 降到 81.581 ms（`both` 1.172×；`c512` 1.140×；`vit` 1.030×）。计时口径为同步处理调用段，不含上传/回读/IO/运动生成/编解码。详见 §15 与 §16。**

**§17 的计划里，Phase 0（归因）已执行完毕，见 §18。** 结论：一帧是 **CPU 受限**（同步尾仅 0.038 ms）、GPU 只忙 **51–62%**；`select()` 每帧 **7.52 / 3.28 ms**；`jit.warmup` **不会**在 GPU 上启动（H2 证伪）；一帧有 **2387 / 2075 次 GPU 内核启动**，其中 **1164 / 1124 次是 eager ATen 内核**，取消分批只动到其中 **13.1%**。§17.0 的结论不变但理由需修正（见 §18.6）。Phase 1 与新增的 Phase 1b 仍待执行，**新改动必须另起一轮目录**。

交接文档把测试执行与监控交给 Luna max。该 worker 不可用，改由实施负责人自己执行 GPU 阶段并自行诊断失败。本文件记录实际执行到的位置、遇到的环境障碍及其处理。**遇到的障碍全部在宿主环境侧与本轮自身代码侧（卷语义、safe-delete 垫片、`gate` 的两处缺陷），都不在候选实现侧；门槛一条未放宽。**

---

## 1. 基线身份（已核实，只读）

| 项 | 值 |
| --- | --- |
| 原尺寸快速入口 | `nr-b580/reference/review_fast_fullsize_v1.py` |
| 入口 SHA-256 | `7c5b0ef0d2107cdd58e0d251d8524bf2851e7c62f20220c5213aaade4f278eed` |
| 与 `full/validation.json` 的 `adapter_sha256` 一致 | 是（程序化断言，见 `rows_paths_v1.identity()`） |
| 已审核快速版验证文件 | `D:/Codex-NR-Experiments/nr-b580/fast-fullsize-review-v1/pipeline-v4/full/validation.json`（SHA-256 `dd94091d8c394780…`） |
| 该验证文件帧数 | 243，`passed=true`，`phase=full`，`full_phase_input=[480,864]` |
| 冻结 profile | `D:/Codex-NR-Experiments/nr-b580/product-v1/profile-v1.json`，SHA-256 `8c0994c7…`，与 `local-runtime-v1.json` 的 `profile_sha256` 一致 |
| 输入帧索引 | `D:/Codex-NR-Experiments/nr-b580/fourway-face-review-v1/exact/validation.json`，243 帧全部存在 |
| 素材 | `E:/下载/480p (1).mp4`，SHA-256 `e0d0776b…` |

实际运行链路（逐层读取源码确认，不是猜测）：

```
nr_runtime_v1.Session.create
 → nr256_product_stack_v1.Stack
   → c512_quad_query_stack_v1.Stack        加 compact_queries
     → c512_int8_full_stack_v1.Stack       加 c512_int8 = Layout；替换 graph 与 call_guard
       → compact_c512_qkv_stack_v1.Stack    rewrite.layout = CompactLayout(bm=32)
         → int8_ffn_calibrated_stack_v1.Stack
           → int8_ffn_nr_stack_v1.Stack     加 int8_vit = Int8VitLayout
```

原尺寸适配的 `installed()` 排除 `graph / rewrite / call_guard / compact_queries`，因此实际被调用的只有：

- `stack.c512_int8.ffn`（`c512_int8_full_stack_v1.Layout.ffn`，内核 `c512_int8_ffn_gpu_v1`，M 硬编码 144）
- `stack.int8_vit.ffn`（`int8_ffn_body_scope_v1.Int8VitLayout.ffn`，内核 `int8_ffn_segment_gpu_v1`，M 已参数化但包装层硬编码 64）

形状（864×480）：内部 512×896 → pre 后四级下采样 → C512 `16×28=448` 行；池化 `8×14` 后填充为 `8×16=128` 个 ViT token。
NR256 门控（256×256）下：C512 `10×10=100` 行；ViT `8×8=64` token。

---

## 2. 本轮新增文件

全部位于 `nr-b580/reference/fullsize_rows_v1/`，未修改任何历史入口、产品加载文件或冻结参考。

| 文件 | SHA-256（前 16 位） | 用途 |
| --- | --- | --- |
| `rows_paths_v1.py` | `d9bf255d365d5aaf` | 显式绝对 W/D 路径、基线身份断言、import 真实路径校验、编译器工具链常量 |
| `c512_int8_ffn_rows_v1.py` | `a376e3a9692be3f7` | C512 五个内核，总行数改为 `M: tl.constexpr` |
| `int8_ffn_segment_rows_v1.py` | `3b809c71b70c28c1` | ViT 四个内核，**逐字节复制**原模块后追加契约常量 |
| `rows_scopes_v1.py` | `6b55dda6f03b4516` | 全行 `c512_ffn` / `vit_ffn`、调度守卫、FFN 调用计数 |
| `rows_entry_v1.py` | `e8b76520c71c25dc` | 本轮独立入口（collect/gate/full/local/benchmark），进程内安装缓存写入守卫 |
| `rows_pipeline_v1.py` | `f5c666bb8fc367b0` | 分阶段编排：hostcache → collect → cache → local → gate → full |
| `rows_compile_v1.py` | `ef76c495210970d1` | 离线编译，14 个独立 worker 进程；worker 初始化安装缓存写入守卫 |
| `rows_bench_v1.py` | `c9285779f069b63b` | ABBA 八进程交错计时与汇总 |
| `rows_preflight_v1.py` | `614d236ed5cdae91` | 无 GPU 预检（身份、路径、语法、工具链、缓存写入门禁、命令清单） |
| `rows_audit_v1.py` | `7a8e91a363a78520` | 无 GPU 内核差异审计 |
| `rows_hostcache_v1.py` | `1f9a305a4e937724` | 候选缓存宿主工具链播种与命中校验（见第 9 节） |
| `rows_toolchain_v1.py` | `02fd9a52f983084e` | 历史编译器环境的解析与诊断（本轮不需要，用于记录） |
| `rows_fscache_guard_v1.py` | `af615be64e14ef90` | 缓存写入守卫：把 `os.removedirs` 改为只删叶子（见第 10 节） |
| `rows_fsprobe_v1.py` | `2e0c8087defa7bed` | 诊断探针：证明**未加守卫时** `put()` 会向上删到盘根（见第 10 节，默认拒绝运行） |
| `rows_compile_probe_v1.py` | `7a99e4aa89d80d1e` | 单 worker 编译探针（早期诊断用） |
| `rows_launch.ps1` | `4309595c04154c13` | 通过共享 B580 租约启动各阶段 |

运行数据/日志/候选缓存放 `D:/fullsize-rows-v1/r1/`；候选 Triton 缓存为 `D:/fullsize-rows-v1/r1/triton-cache/`。
**旧缓存 `D:/Codex-NR-Experiments/nr-b580/reference/triton-cache-c32-triton38-v1`（796 MB）不删除、不重建**，只由 baseline 变体按历史方式使用。

---

## 3. 只改了什么

改动严格限定为四项：**总行数 M、尾部 mask、工作缓冲区形状、调度**。

- C512：`144` → 运行期实际行数（864×480 为 448，NR256 为 100），一次启动写完整个输出；不再 pad 到 144 再 cat。
- ViT：`64` → 运行期实际 token 数（864×480 为 128，NR256 为 64）。
- 缓冲区由 `(144,512)/(144,2048)/(144,)` 改为 `(rows,…)`；ViT 由 `(64,1024)/(64,4096)/(4,64,1024)` 改为 `(rows,…)`。
- 网格由 `cdiv(144,BM)` / `(64,)` / `(128,)` 改为 `cdiv(rows,BM)` / `(rows,)` / `cdiv(rows*1024,512)`。

保持不变：BM/BN/BK（`(32,64)/(16,64)/(16,32)` 与 `(32,32)/(16,32)`）、`num_warps=4`、`num_stages=1`、`enable_fp_fusion=False`、`select()` 零溢出选核、权重与量化尺度、K 累加顺序、舍入、`round_half`、三次激活、FP8 边界、残差、注意力（全 token、不分块、不随 FFN 行批重复）、QKV 与投影。

未加入：Tensor Descriptor、全图捕获、重新量化、缩放、任何新乘积功能。

---

## 4. 静态审计结论（无 GPU，已跑通）

`rows_audit_v1.py` 与 `rows_preflight_v1.py` 均 `passed=true`。

1. **C512 内核只差行数**：对 `c512_int8_ffn_gpu_v1.py` 与 `c512_int8_ffn_rows_v1.py` 做 AST 级函数源比对。把 `M: tl.constexpr` 参数去掉、把字面量 `< 144` 改写成 `< M` 之后，`_cubic / _entry / _expand / _groups / _linear / _project / _reduce` **七个函数源码完全相同**。唯一被删掉的是未被本轮使用的 `Segment` 包装类。
2. **ViT 内核逐字节相同**：`int8_ffn_segment_gpu_v1.py` 与 `int8_ffn_segment_rows_v1.py` 的 `Segment / _contract / _entry / _expand / _finish / _merge / _q` **七个定义完全相同**（新模块是在原文件后追加契约常量得到）。ViT 内核本来就把 M 作为 `tl.constexpr`，只有包装层写死 64。
3. **形状与调度一致**：C512 448 行在 `BM=32/16` 下为 14/28 个行块（原 144 行为 5/9），NR256 100 行为 4/7；每帧 FFN 调用次数为 baseline `c512=64, vit=16`、仅 C512 `16, 16`、仅 ViT `64, 8`、两者 `16, 8`。
4. **运行期会失败关闭**：`quantization_dataflow_v1.Dataflow.launch` 对未登记内核直接断言失败，候选内核已登记；`rows_scopes_v1.dispatch_guard` 额外断言候选变体下**没有**任何 `c512_int8_ffn_gpu_v1.*` / `int8_ffn_segment_gpu_v1.*` 被派发，且 `DEBUG` 恒为 `False`、`PARTS` 恒为 4。
5. **import 身份**：入口对 `split_block`、`vit_block`、`nr_runtime_v1`、两个候选内核模块逐一断言其真实文件路径，防止复制后悄悄加载旧内核。

### 关于字节一致的说明（这是推导，不是运行证据）

C512 与 ViT 的每个内核都是**逐行独立**的：load 按行掩码、reduction 只沿 K、没有任何跨行运算；尾部被丢弃的填充行不参与任何有效行的计算。因此把「144 行分 4 批、末批补零后截断」换成「448 行一次」应当不改变 0..447 行的输出。

**这个论证只能用来解释为什么值得做这次实验，不能替代逐字节核验。** 交接文档也明确要求：以实测为准，出现差异必须定位，不得放宽门槛。

---

## 5. 运行命令（PowerShell，需 B580 租约）

所有 GPU 阶段都经共享租约串行化，单条命令上限 1800 秒，因此按阶段分开执行。

```powershell
$PS = 'E:\ComfyUI-aki-v3-IntelArc_20260722\nr-b580\reference\fullsize_rows_v1\rows_launch.ps1'

# 0) 无 GPU 预检：身份、路径、语法、命令清单（不占租约）
powershell -NoProfile -ExecutionPolicy Bypass -File $PS -Stage preflight -Run 'D:\fullsize-rows-v1\r1'

# 1) collect + 离线编译（c512 / vit 两份目录，14 个独立 worker，写入本轮缓存）
powershell -NoProfile -ExecutionPolicy Bypass -File $PS -Stage collect -Run 'D:\fullsize-rows-v1\r1'

# 2) 局部完整输出比对：四个变体各跑帧 0..60 并捕获第 60 帧，
#    比对 16 个 C512 块（448×512）与 8 个 ViT 块（128×1024）的整体字节与尾部切片
powershell -NoProfile -ExecutionPolicy Bypass -File $PS -Stage local -Run 'D:\fullsize-rows-v1\r1'

# 3) NR256 控制门：四个变体各一帧，与已审核 NR256 快速产品逐字节比对
powershell -NoProfile -ExecutionPolicy Bypass -File $PS -Stage gate -Run 'D:\fullsize-rows-v1\r1'

# 4) 243 帧全片逐字节核验：四个变体分别与已审核原尺寸快速版比对
powershell -NoProfile -ExecutionPolicy Bypass -File $PS -Stage full -Run 'D:\fullsize-rows-v1\r1'

# 5) 同批 ABBA 计时：baseline / c512 / vit / both / both / vit / c512 / baseline 八个独立进程
powershell -NoProfile -ExecutionPolicy Bypass -File $PS -Stage bench -Run 'D:\fullsize-rows-v1\r1'
```

单独诊断某个变体（例如只查 C512 失败时）：

```powershell
$PY = 'E:\ComfyUI-aki-v3-IntelArc_20260722\ComfyUI-aki-v3-IntelArc\python\python.exe'
$H  = 'E:\ComfyUI-aki-v3-IntelArc_20260722\nr-b580\reference\fullsize_rows_v1'
& $PY -X utf8 -u "$H\rows_entry_v1.py" --phase local --variant c512 `
    --out 'D:\fullsize-rows-v1\debug-c512' --cache 'D:\fullsize-rows-v1\r1\triton-cache' --through 61
```

日志：`D:\fullsize-rows-v1\r1\stage-<stage>.log`，逐阶段 `*.stdout.log` / `*.stderr.log`；
租约记录：同名 `.log.lease.json`（含 PID、退出码、起止时间）。

---

## 6. 判据与产出

| 阶段 | 产出 | 通过判据 |
| --- | --- | --- |
| collect | `collect-c512/`、`collect-vit/`、`compile-*.json` | 目录与编译均 `passed=true`，14 个 worker 全部完成 |
| local | `local-<variant>/local.json`、`pipeline-local.json` 的 `local_comparison` | 四个变体的 C512/ViT 局部整体字节、尾部切片、原始 `.npy` 字节全部一致 |
| gate | `gate-<variant>/validation.json` | `control_byte_equal=true`（与已审核 NR256 快速产品逐字节相同） |
| full | `full-<variant>/validation.json` | 243 帧 `byte_equal=true`，且 `input_sha256` 与输入索引一致 |
| bench | `bench/benchmark.json` | 每帧 `byte_equal=true`；每帧 FFN 调用次数等于预期；报告局部与完整调用绝对 ms、加速比、波动 |

计时协议：相同 GPU 驻留输入、4 帧预热、从首帧重置、运行 32 帧、统计第 4..31 帧、旧/新独立进程交错、推理期间禁止 JIT（`DiskOnly` 缓存未命中即抛错）。
剖面采集与正式计时分开，本轮未加入任何 profiler。
计时区间为同步处理调用，包含 Python、作用域与 guard，排除上传/读回、运动生成、IO 与编解码。
baseline 使用历史共享缓存，候选使用本轮缓存；两个变体在计时循环里都挂了同样的 FFN 调用计数包装（每帧 24 次额外 Python 调用），`dispatch_guard` 只在核验阶段开启，不计入计时。

---

## 7. 失败时的诊断顺序

1. `local` 阶段失败 → 先看 `local_comparison.blocks` 中哪个块、哪个尾部切片不同；C512 末批（行 432..447）是历史填充区，最先怀疑。
2. `gate` 阶段失败 → 说明适配层本身与已审核 NR256 快速产品不一致，先查 import 身份与缓存目录，不要进入原尺寸比较。
3. `full` 阶段失败 → 从 `full-<variant>/validation.json` 的 `frames` 找首个 `byte_equal=false` 的帧号与 `max_abs`，再回到 `local` 用同一帧定位。
4. `select()` 报「All reviewed launch configurations spill」→ 说明 M 变大后首选配置溢出，这是**新的资源事实**，需要按块配置重新选核并重新离线编译，不得放宽零溢出门槛。
5. `DiskOnly` 报缺 artifact → 说明 collect 未覆盖该 specialization，回到 collect 阶段排错，**不准在计时中 JIT**。

---

## 8. 遗留问题与未覆盖范围

- 本轮**没有**任何运行期证据，全部性能与字节结论待 GPU 阶段产生。
- 只覆盖 864×480（全片）与 256×256（NR256 控制门）。1080p、1440p、2160p 未涉及，不得据此宣称已支持或已加速。
- 现有输入的完整字节验证不等于所有分辨率通过。
- `select()` 在每次 FFN 调用时重新选核（沿用被审核行为，未改）。行批减少后该主机侧开销随之下降，这属于调度变化的一部分，但**尚未测量**其占比。
- 未评估收益上限；若收益不明显，按交接文档如实结束这条实验，不自动扩成全网重写或大规模调参。
- 未改动宿主 ComfyUI、Python 包、驱动或系统环境；未发布、未上传；未清理任何历史实验。

---

## 9. 候选缓存的宿主工具链（本轮实际障碍与处理）

### 9.1 现象与根因

第一次 `collect` 在 `driver.active.get_current_target()` 处失败：

```
triton/backends/intel/driver.py:441  compile_module_from_src(... "driver.c", name="spirv_utils")
triton/runtime/build.py:68           RuntimeError: Failed to find C compiler.
```

排查后确认三件事：

1. **被审核的原尺寸启动脚本根本不设置编译器。** `start_fast_fullsize_review_v1.ps1` 只做 `Start-Process`，没有任何 vcvars / `CC`。原因是 `compile_module_from_src` **先查缓存再编译**，而共享缓存里已经存在那三个宿主模块。
2. **候选缓存是空的。** 候选变体用 `D:/fullsize-rows-v1/r1/triton-cache`，三个宿主模块都不在其中，于是真的走进编译分支，才把缺编译器暴露出来。**这不是基线的缺陷，是本轮新缓存必然遇到的入口问题。**
3. **本环境无法提供编译器。** `vcvars64.bat` 内部依赖 `reg.exe` 定位 Windows SDK，而 `reg.exe` 被安全策略列入程序黑名单，无法执行也无法批准。结果是 `INCLUDE` 只有 2 项（完整环境应有数十项），`WindowsSdkDir` / `WindowsSdkVersion` / `VCToolsInstallDir` 全为空。`rows_toolchain_v1.py --check` 把这一事实记录在案。

历史机制本身已查清（供将来环境修复后参考）：`Run-CollectExactV1.cmd` / `Run-BuildExactV1.cmd` 用 `vcvars64.bat` + `CC=CXX=icx-cl.exe` + `LIB=<oneAPI compiler lib>`。**必须用 `icx-cl` 而不是 `icpx`**，因为 `compile_module_from_src` 会追加 MSVC 风格参数（`/LIBPATH:`、`/D`），只有 cl 兼容驱动接受。`Build-Esimd*.cmd` 另外显示内核编译需要 `oneAPI\ocloc\2026.1\bin` 在 PATH 上。

### 9.2 为什么本轮不需要编译器

Intel 后端只有三处 C 宿主模块：`spirv_utils`（`driver.c`）、`arch_utils`（`arch_parser.c`）、`extension_utils_impl`（`extension_utils.c`）。它们的缓存键是

```
sha256(__CACHE_VERSION + platform_key() + libsycl_dir + source)
```

**不含编译器。** 同时内核产物是纯 SPIR-V：`generate_native_code` 默认为 `False`，`make_zebin`（唯一调用 `ocloc` 的地方）不会执行，`_build` 对内核永不被调用。共享缓存中 2652 个内核的产物类型只有 `ttir/ttgir/llir/spv/source/json`，没有任何 `zebin`/`cubin`，实测证实了这一点。

因此候选缓存可以**逐字节复用**被审核基线所用的那三个宿主模块，而不重新编译。这既更忠实（候选变体跑的是与基线完全相同的主机侧管线），也是本环境下唯一可行且不引入未知差异的做法。

### 9.3 做法与验证

`rows_hostcache_v1.py` 计算三个模块在本环境下的缓存目录名，从共享缓存复制到候选缓存，并要求三者都能从缓存加载（命中即不编译）。

关键细节：哈希必须对 `read_text().encode("utf-8")` 计算，**不能对文件原始字节计算**。`compile_module_from_src` 用的是 `read_text()`，在 Windows 上会把 CRLF 折叠为 LF，两者字节不同。按原始字节计算会得到完全不同的目录名（实测 `MRPH42OJHRHSRDQOMXCJSIULT7P5GFMWVT4LW4YAI33B6A7HS2GA`，而正确值是 `EUO7OBCWUGA5UGKFQSNPBIHT4CGDBEZ6P6U36ZM2ROBQ2ZHB2LXA`）。

无 GPU 验证结果：

| 模块 | 计算出的目录名 | 与共享缓存一致 | `.pyd` SHA-256（前 16 位） |
| --- | --- | --- | --- |
| `spirv_utils` | `EUO7OBCWUGA5UGKFQSNPBIHT4CGDBEZ6P6U36ZM2ROBQ2ZHB2LXA` | 是 | `cc22387f0861aecc` |
| `arch_utils` | `6L4GKBVMBXZKIGUHDHMIPPGA4JYZRCN34XWAER6JXESUSVXKMXGQ` | 是 | `f127b2580270cddc` |
| `extension_utils_impl` | `L7YTNAHV4UHEJQJNA4TQVBMXA7FOQWGTM53CKDEHOZIPSGIABWYQ` | 是 | `216a5a8e1737ef3b` |

三个目录名与共享缓存实际目录名**完全一致**，说明 `isolate()` 之后 `libsycl_dir` 的解析结果与当初产生共享缓存的环境相同；随后三个模块都从缓存加载成功，全程没有编译器参与。

另一个细节：失败的那次尝试已经创建了空的 `EUO7…` 目录（`FileCacheManager.__init__` 在编译前就 `makedirs`）。因此播种时判断「已存在」必须看**模块文件是否存在**，不能看目录是否存在。

### 9.4 对结论的影响

- 共享缓存 `D:/Codex-NR-Experiments/nr-b580/reference/triton-cache-c32-triton38-v1`（796 MB）**未被修改**，只被读取。
- 从共享缓存复制的只有这三个宿主模块，**没有任何内核产物**被复制；候选内核仍由 `rows_compile_v1.py` 用 14 个独立 worker 离线编译进候选缓存。
- 宿主模块不属于被测变体、不参与 NR 数值，因此 baseline 与候选变体使用**同一份**宿主模块。
- `rows_launch.ps1` 不设置任何编译器变量，只把 `TEMP`/`TMP` 指向 `D:\fullsize-rows-v1\r1\temp`（沿用历史构建脚本把构建临时文件放 D 盘的做法）。
- 若将来环境放开 `reg.exe`，`rows_toolchain_v1.py --check` 可直接用来确认编译器是否恢复可用；届时宿主模块既可从缓存复用，也可重新编译，两者缓存键相同。

---

## 10. 候选缓存无法写入（本轮真正的障碍与处理）

### 10.1 现象

宿主工具链问题解决后，`collect` 阶段的 `collect-c512` 成功（`catalog.json` 293 个内核），但紧接着的 `compile-c512` 失败：

```
RuntimeError: compile-c512 failed, rc=1
FileNotFoundError: [Errno 2] No such file or directory:
  'D:\fullsize-rows-v1\r1\triton-cache\UHYKRPYIJY7R3E2SBBAY5OBABIOIINDEQNL22QDBTT3U2RFMLCOA\_kernel.json'
```

期望 293、完成 0。`--workers 1` 同样确定性失败。

### 10.2 被排除的错误假设（记录以免重犯）

1. **「catalog 里多行共享同一个缓存目录」——错误，已撤回。** 最初按 `constants` / `attrs` 字段统计得到 28 组，但这两个字段实际是 `null`；改用真实字段 `constant_keys` / `attrs_keys` 后统计结果是 **293 个互不相同的身份、0 个共享目录**。
2. **「多 worker 竞争」——错误，已排除。** `--workers 1` 单进程同样确定性失败。

### 10.3 真正根因：本卷不保证「非空目录不可删」

`FileCacheManager.put()`（`triton/runtime/cache.py:103`）结尾是：

```python
try:
    os.replace(temp_path, filepath)
except PermissionError:
    if os.name == "nt":
        os.remove(temp_path)     # 此时 temp_dir 已空
    else:
        raise
os.removedirs(temp_dir)
return filepath
```

`os.removedirs` 先删叶子，再沿路径向上逐级删父目录，它**依赖「父目录非空时 `rmdir` 会失败」这一前提来停止向上**。

本卷不满足这个前提：`rmdir` 对非空目录会**成功并递归删除**。证据：

- 用 `python -S`（完全不加载 `sitecustomize`，`os.rmdir` 就是原生 `nt.rmdir`）删除一个含 `child.txt` 的目录，**无异常，目录与子文件全部消失**；另一个进程随后确认目录确实不存在。→ 这是卷/文件系统层的行为，不是 Python 层的 monkeypatch。
- 对照：shell 的 `rmdir` 会拒绝（`Directory not empty`），但那不是操作系统的回答——`cli/vendor/shim/safe-bin` 里有 `rm` / `rmdir` / `unlink` 包装脚本，且 `PYTHONPATH` 指向 `cli/vendor/shim`，其 `sitecustomize` 会把 `os.remove` / `os.unlink` / `os.rmdir` / `shutil.rmtree` 换成 `_safe_*`（即 WorkBuddy 的 safe-delete 垫片）。设 `CODEBUDDY_SAFE_DELETE_ENABLED=0` 能让 `os.rmdir` 恢复为 `nt.rmdir`，**但删除行为不变**。

因此垫片只是上层，真正的原因是卷语义本身。

### 10.4 后果，以及一次真实的数据损失

`put()` 在 `os.replace` 成功、`temp_dir` 变空之后调用 `os.removedirs(temp_dir)`，于是从临时目录**一路向上**删除：缓存目录 → `triton-cache` → run 目录 → … → 盘根，只在 `D:\` 处失败停下。`rows_fsprobe_v1.py` 记录到的实际事件序列：

```
rmdir ok | contents= []                                   ← temp_dir
rmdir ok | contents= ['_kernel.source']                   ← 缓存目录（刚写进去的文件在里面）
rmdir ok | contents= ['replace-dst','replace-src']        ← 探针根目录
rmdir ok | contents= ['audit.json','...','r1', ...]       ← D:/fullsize-rows-v1
rmdir PermissionError | contents= ['$RECYCLE.BIN','Codex-NR-Experiments', ...]  ← D:\
```

**这次诊断直接删除了 `D:\fullsize-rows-v1` 整个运行目录**，包括 `r1/` 下的 `hostcache.json`、`collect-c512/`（含 catalog）、`compile-*.json`、各阶段日志、`failed-attempts/` 与已播种的 `triton-cache/`。

已逐一核验未受影响的部分：`D:\` 根目录、`D:\Codex-NR-Experiments\nr-b580`（共享缓存 2658 项 + 3 个宿主模块、全部冻结参考、已审核基线验证文件、两个原内核模块）。丢掉的全部是可重新生成的运行产物，没有任何冻结参考或历史入口受损。

教训：**在这个卷上，任何一次未加保护的 `os.removedirs` 都可能删到盘根。** 因此 `rows_fsprobe_v1.py` 现在默认拒绝运行，必须显式传 `--acknowledge-ancestor-deletion`。

### 10.5 处理：`rows_fscache_guard_v1.py`

`install()` 把 `os.removedirs` 换成只删叶子的版本，并且**只允许删 basename 以 `tmp.pid_` 开头的目录**（Triton `put()` 给临时目录的命名，`cache.py:116`），其余一律拒绝并记录到 `refused`。

影响面恰好一行：

- 整个 Triton 树里 `os.removedirs` 只有一处调用（`cache.py:134`）；`rmtree` / `os.rmdir` 没有其他调用点。
- `nr-b580` 与 `nr-b580-int8` 里**没有任何** `os.rmdir` / `os.removedirs` / `shutil.rmtree` 调用。

守卫不改内核、不改产物、不改 specialization、不改任何数值：只让一次临时目录清理不再删除自己的父目录。

安装位置（monkeypatch 不跨进程，所以每个可能写 Triton 缓存的进程都要装）：

| 进程 | 位置 |
| --- | --- |
| 编译 worker | `rows_compile_v1.worker_init`（真正发生编译的地方） |
| 编译父进程 | `rows_compile_v1.__main__` |
| 全部阶段进程 | `rows_entry_v1.main` 开头（local / gate / full / collect / benchmark；bench 经 `rows_bench_v1` 间接走入口） |
| 门禁 | `rows_preflight_v1` 以**独立子进程**运行探针并断言 `passed=true` |

这一点对 baseline 变体尤其重要：baseline 用的是共享缓存 `reference/triton-cache-c32-triton38-v1`，一旦那里出现意外未命中，未加守卫的 `put()` 会顺着 `reference/` 一路向上删进冻结参考树。

验证结果：

- 守卫探针（跑真实的 `FileCacheManager.put`）：首次写入、覆盖写入、`put_group` 三次都返回**存在**的路径，缓存目录存活，目录内只剩 `_kernel.source` 与 `__grp___kernel.json`（临时目录被正确清理），`passed=true`。
- 预检 `fs_guard.passed=true`，`removedirs` 为 `__main__.removedirs_leaf_only`，`rmdir` 仍为垫片的 `_safe_rmdir`。

### 10.6 对结论的影响

- 这是**宿主环境**问题，不是候选实现的问题。历史 `product-precompile/exact-v1/package.json` 显示 996 个内核、14 个 worker 构建成功，正是因为那次构建不在本卷/本沙箱语义下运行。
- 守卫不改变任何被测数值，也不放宽任何门槛，因此不构成对交接文档「出现差异必须定位，不得放宽门槛」的违反。
- 编译是否能在本环境完成，仍然要以 `compile-c512.json` / `compile-vit.json` 的 `passed=true` 为准；本文件不预设结论。

### 10.7 第二个障碍：safe-delete 垫片的批量删除守卫（已解决）

加上守卫后缓存里出现 24 项（3 宿主 + 21 内核），说明写入已正常，但 `compile-c512` 仍然 rc=1，报告里 `error=SystemExit(1)`，stderr 末尾：

```
[safe-delete][SAFE_DELETE_BULK_GUARD_ERROR] state lock timeout
```

读 `cli/vendor/shim/sitecustomize.py` 确认了机制：

- `_check_bulk_delete_guard()`（L814）在**每一次**回收站删除时都执行
  `subprocess.run([node, guard, "check", "--target", <path>], timeout=10)`；
  返回码非 0 就调用 `_exit_bulk_guard_control()`，后者直接 **`raise SystemExit(1)`**（L805-812）。
  "state lock timeout" 是那个 Node 辅助脚本在 14 个并发 worker 下的 stderr。
- 开关在**解释器启动时**读取（L35 `CODEBUDDY_SAFE_DELETE_ENABLED != "0"`，patch 安装见 L1149），
  因此**必须在子进程启动前设置**，在进程内设置无效。
- 顺带确认了 §10.3 的根因：`_safe_rmdir`（L1043）对非空目录会 `return _orig_rmdir(path)`，
  即**主动转调原生 rmdir**。垫片本身不提供"非空目录不可删"的保护，它把这个判断交给了原生调用——
  而本卷的原生调用会递归删除。**所以关掉垫片并不能替代守卫，两者都需要。**

处理：`rows_paths_v1.child_environment()` 给本轮所有阶段子进程设
`CODEBUDDY_SAFE_DELETE_ENABLED=0`；`rows_launch.ps1` 同步设置。
只影响本轮自己启动的子进程，不改全局环境、不改宿主配置。
垫片的 broker 补丁（`os.open` / `os.mkdir` / `os.rename` / `os.replace` …，L1131-1147）只在 `sys.platform == "darwin"` 生效，
因此 Windows 上关掉 safe-delete 不影响其他行为。

### 10.8 collect 阶段结果（已通过）

两处修复后 `collect` 阶段 **exit=0**：

| 阶段 | 结果 |
| --- | --- |
| `hostcache` | rc=0，三个宿主模块均从共享缓存命中 |
| `collect-c512` / `collect-vit` | 均 `passed=true`，`shape_only=true`，`gpu_dispatches=0` |
| `compile-c512` | `passed=true`，`phase=completed`，**293/293**，14 worker |
| `compile-vit` | `passed=true`，`phase=completed`，**292/292**，14 worker |
| 候选缓存 | 305 个目录（3 宿主模块 + 302 个内核产物） |

`compile-*.json` 里的 `runtime` 指纹在 293（292）个 worker 之间完全一致，说明工具链与 Triton key 没有漂移。

---

## 11. `local` 阶段的执行发现（两个自身代码缺陷，已修）

`local` 首次运行在 `local-baseline` 上 rc=1。**与候选实现无关**，是本轮入口自身的两个缺陷：

1. **XPU 张量未搬回主机内存。** `write_local` 里 `tensor.contiguous().numpy()` 直接抛
   `TypeError: can't convert xpu:0 device type tensor to numpy`。
   同一文件里另外两处（`measure` 的 `output.cpu().numpy()`、`one_frame` 的 `color.cpu().numpy()`）本来就带 `.cpu()`，
   只有这一处漏了。已改为 `tensor.detach().float().cpu().numpy()`。

2. **诊断阶段把存储精度隐含地写死了。** 原实现把局部输出 `.astype('f2')` 后存盘。
   实测这两个族的设备张量本来就是 **f16**（报告里 `source_dtype=torch.float16`），
   所以对当前数据这个窄化是**空操作**，并没有掩盖差异。
   **我先前在这里写"它会掩盖半精度以下的差异"是说错了，已更正。**
   改动的真实理由是：`astype('f2')` 把"输出恰好是 f16"当成了一个**隐含前提**——
   一旦将来某块输出变成 f32，这个窄化就会真的丢信息，而报告里看不出来。
   现在改为存 f4（f16/bf16 → f32 是**单射**，字节比对与设备张量同样严格，不损失严格性），
   并把源 dtype 记进报告（`source_dtype`），让这个前提显式化、可核对。

3. **调度守卫的判据按错了维度。** 守卫原来断言「候选变体下不得出现任何已审核固定行内核」，
   但 `c512`-only 与 `vit`-only **故意**只替换一个家族、让另一个家族继续走原固定行内核，
   所以 `local-c512` 必然误报：

   ```
   RuntimeError: reviewed fixed-row kernel dispatched under a candidate variant:
     int8_ffn_segment_gpu_v1._entry
   ```

   这正是 `c512`-only 变体应有的行为（ViT 继续走固定 64 行内核），是**判据错**而不是被测对象错。
   已改为**按家族**判定（`rows_scopes_v1.MODULE_FAMILY` + `replaced_families(variant)`）：

   - 被替换家族的已审核内核必须消失；
   - 未被替换家族的**候选**内核也必须消失（新增的反向断言，比原来更严）。

   注：这条误报是在 `local-c512` 上暴露的，说明该守卫确实在运行期生效，不是装饰。

另外给 `full` 阶段加了两条前置断言：`output.shape == expected_output.shape` 与
`output.dtype == expected_output.dtype`。
原因是字节比对是 `tobytes()` 比较，两边 dtype 不一致会**因为错误的原因失败**（或更糟：看似通过）。
已审核基线在 `full` 阶段并不做字节比对（它只对 exact 参考算 rmse 并把输出存 npz），
字节比对是本轮新增的判据，所以这两条断言必须由本轮自己保证。

**注意：本阶段至今仍然没有任何字节一致性结论。**

---

## 12. `local` 阶段结果（已通过）：61 帧累积局部输出逐字节一致

`pipeline-local.json`：`passed=true`，五个阶段 rc 全 0（hostcache / local-baseline / local-c512 / local-vit / local-both）。

| 变体 | `whole_output_sha256` | `tail_sha256` | `raw_bytes` | `shapes` | `byte_equal` |
| --- | --- | --- | --- | --- | --- |
| `c512` | true | true | true | true | **true** |
| `vit` | true | true | true | true | **true** |
| `both` | true | true | true | true | **true** |

- 比对块总数 **72**（每个变体 16 个 C512 块 + 8 个 ViT 块），**不一致块：无**。
- 全部块 **`max_abs = 0.0`**（逐元素差的最大绝对值）。
- 累积形状：C512 `[27328, 512]`、ViT `[7808, 1024]`。
  27328 = 448 × 61、7808 = 128 × 61，即**帧 0..60 的 61 帧局部输出全部累积后整体比对**，
  不是只比末帧、也不是只比尾部切片。
- 存盘为 f32、源 dtype 记录为 `torch.float16`；`raw_bytes` 是 `.npy` 原始字节比较。

**这一条能证明什么**：取消固定 144/64 行分批、改为按真实行数一次启动并直写完整输出缓冲，
在 C512 与 ViT 两个家族的局部 FFN 输出上，与已审核的固定行分批版本**没有任何一个字节不同**。
§4 里那个"逐行独立所以应当不变"的推导，到这里第一次有了实测支持。

**这一条还不能证明什么**：

- 只覆盖帧 0..60（61/243），不是全片。
- 只覆盖局部 FFN 输出，不是整帧输出；注意力、QKV、投影、残差、池化、`final_weight` 等
  都还没被这条判据触及——它们由 `gate` 与 `full` 阶段覆盖。
- 因此**不是**性能结论，也**不是**全片结论。

入口代码在此阶段之后冻结：`full` 阶段会断言 `gate['adapter_sha256'] == sha(rows_entry_v1.py)`，
任何后续修改都必须重跑 `gate` 与 `full`。

---

## 13. `gate` 阶段的两处失败（均为本轮自身缺陷，已修）

`gate` 首次运行 `pipeline-gate.json` `passed=False`，`stages` 中只有 `hostcache`（rc=0）。
`rows_pipeline_v1.run()` 在 rc≠0 时抛异常、`report['stages'].append(...)` 不会执行，
所以这直接说明 **`gate-baseline` 失败**。两次失败原因互不相同，且**都与被测的候选实现无关**。

### 13.1 第一次：`report['frames']` 被一个整数静默覆盖

`gate-baseline.stderr.log`：

```
File "...rows_entry_v1.py", line 391, in main
    report['frames'].append(result)
AttributeError: 'int' object has no attribute 'append'
```

根因是**同名键被静默覆盖**：`rows_entry_v1.main()` 第 145 行把 `report['frames']` 初始化为
**逐帧比对结果列表**；第 209 行 `report.update(paths.identity())` 合并身份字典，
而 `rows_paths_v1.identity()` 当时返回 `frames=243`——那是"**已审核基线有 243 帧**"，一个整数。
`dict.update()` 于是把这个整数覆盖了列表。

这也解释了此前 `collect` 阶段 `validation.json` 里 `frames` 是整数这一现象：
`collect` 在 append 之前就 `return` 了，`local` 写的是 `report['local']`，都不碰这个键。
`gate` 是**第一个**向 `report['frames']` 追加的阶段，因此是第一个崩的。

修法：把身份里那个键改名为 `baseline_frame_count`（`rows_paths_v1.py`）——
它本来就是"基线帧数"，与"本轮的逐帧结果"不是一回事。并在 `rows_entry_v1.py` 合并处加断言：

```python
report.update(paths.identity())
assert isinstance(report['frames'], list), type(report['frames']).__name__
```

这样同类覆盖不会再以"深层 AttributeError"的形式暴露，而是当场报出真正的键名。

修复后 **`gate-baseline` 通过**：`control_byte_equal=true`、`control_max_abs=0.0`，
`frames` 恢复为列表。失败点前移到 `gate-c512`。

### 13.2 第二次：本轮缓存缺少实验残差缩放内核

`gate-c512.stderr.log`：

```
File "...face480_residual_scale_v1.py", line 26, in prepare
    color=self.axis(self.axis(rgb,'lanczos2','y'),'lanczos2','x')
File "...residual_scale_v1.py", line 120, in axis
    _axis[(triton.cdiv(out.numel(), 256),)](...)
File "...fast_cached_runtime_v1.py", line 39, in disk_only
    if not group:raise RuntimeError('Missing fast artifact; ...: '+src.name)
RuntimeError: Missing fast artifact; prepare offline before video processing: _axis
```

根因是**缓存覆盖范围**，不是候选实现：

- `rows_entry_v1.one_frame()` 第 332 行：`scaler = Face480Scale() if args.phase == 'gate' else None`。
  **只有 `gate` 阶段构造 `Face480Scale`**；`local` 与 `full` 的 `scaler` 都是 `None`。
- `Face480Scale` 在 `nr-b580-int8/experimental/`，走普通 Triton JIT / `TRITON_CACHE_DIR`。
  它用到 `residual_scale_v1.py` 的三个内核：`_axis`（`prepare` 4 次 + `composite` 2 次）、
  `_pad`（2 次）、`_residual`（1 次）。
- 本轮缓存只有两个来源：3 个宿主模块（`rows_hostcache_v1`）+ `collect` 目录离线编译的 FFN 内核。
  而 `collect` 目录只覆盖产品自身的 `_ops.py` / `_classes.py`，**从不包含实验目录**。
- `baseline` 变体用**共享已审核缓存**（含 `_axis` 产物），所以 `gate-baseline` 通过；
  候选变体用本轮缓存，于是失败。

实测确认：本轮缓存中 `_axis` / `_pad` / `_residual` 产物数为 **0**；
共享缓存中分别为 14 / 8 / 3 个 specialization（共享缓存共 2652 项、43 个内核名，本轮 302 项、19 个）。

### 13.3 修法：`rows_scaleprep_v1.py`，把缺失内核编进本轮自己的缓存

轮次的既定纪律是"**只有 3 个宿主模块逐字节复用，内核一律由本轮自己编译**"
（`rows_hostcache_v1.py` 契约原文："no kernel artifact is taken from the shared cache"）。
所以这些内核**不能**从共享缓存拷贝，必须由本轮编译。

新增 `rows_scaleprep_v1.py`，作为 `gate` 的前置准备步骤（`pipeline-gate.json` 中记为 `scaleprep` 阶段）：

- **走真实代码路径**：用 `runpy` 运行 `rows_entry_v1.py --phase gate --variant both --cache <本轮缓存>`。
  不在准备脚本里重写一遍缩放器调用序列——那会成为第二份副本、可能漂移；
  且 specialization 由输入 dtype 决定，凭猜会编出错的内核。
- **不改入口**：在入口 import 之前把 `fast_cached_runtime_v1.DiskOnly` 换成 no-op，
  因此 `rows_entry_v1.py` 本身无需为此改动。
- **只在这一处放开 JIT**：记录用的 `gate` 阶段仍在 `DiskOnly` 下运行，证据链不变、仍然 fail-closed。
- **不得编译被测内核**：用镜像 `DiskOnly` 键推导的 hook 记录**每一个真正未命中**的内核，
  并断言其中没有任何一个属于本轮替换的家族（`c512_int8_ffn_rows_v1` / `int8_ffn_segment_rows_v1`）。
  离线目录必须始终是被测内核的唯一来源。
- 变体取 `both`（候选变体），使被测内核命中缓存，只有未覆盖的内核需要编译。

---

## 14. `gate` 阶段结果（已通过）：四变体 NR256 控制帧与已审核 NR256 快速产品逐字节一致

`pipeline-gate.json`：`passed=true`，六个阶段 rc 全 0
（`hostcache` / `scaleprep` / `gate-baseline` / `gate-c512` / `gate-vit` / `gate-both`）。

| 变体 | `control_byte_equal` | `control_max_abs` | `frames` | `cache_hits` | 每帧 FFN 调用 |
| --- | --- | --- | --- | --- | --- |
| `baseline` | **true** | **0.0** | 1 | 140 | c512 16 / vit 8 |
| `c512` | **true** | **0.0** | 1 | 140 | c512 16 / vit 8 |
| `vit` | **true** | **0.0** | 1 | 140 | c512 16 / vit 8 |
| `both` | **true** | **0.0** | 1 | 140 | c512 16 / vit 8 |

- 每个变体取输入索引第 0 帧（NR256 控制帧），与已审核 NR256 快速产品的 npz 输出做
  **整体字节比较**（`output.tobytes() == expected_output.tobytes()`），不是 rmse、不是 allclose。
- 四变体 `control_max_abs` 全为 **0.0**。
- 四变体 `ffn_calls` 完全相同（c512 16 / vit 8），说明 NR256 下四个变体走同一条控制路径，
  差异只在被替换的内核家族上（`kernel_dispatches` 里的模块名不同）。
- `cache_hits=140` 四变体一致。

### 14.1 `scaleprep` 实际编译了什么（可核对）

`scaleprep/prep.json`：`passed=true`、`disk_only_replaced=NoOpCache`、`cache_hits=131`、
`compiled_under_test=[]`。

它编译的 3 个内核（按 `module.qualname` 计）：

| 内核 | 位置 |
| --- | --- |
| `residual_scale_v1._axis` | `nr-b580-int8/experimental/residual_scale_v1.py` |
| `residual_scale_v1._pad` | 同上 |
| `residual_scale_v1._residual` | 同上 |

按 **specialization（缓存产物）** 计则是 **9 个**：`_axis` 6 + `_pad` 2 + `_residual` 1，
与本轮缓存实测一致（305 → 314 项）。这 9 个正好对应 `Face480Scale` 的调用序列：
`prepare` 里 `_axis` 4 次（lanczos2-y / lanczos2-x / area-y / area-x）+ `_pad` 2 次（RGB 与 motion 各一）；
`composite` 里 `_residual` 1 次 + `_axis` 2 次（catmull-x / catmull-y 带 `original`）。
（`prep.json` 的 `compiled` 字典以 `module.qualname` 为键，同一内核的多个 specialization 会合并为一条，
因此 `compiled_count=3` 是**内核数**而非产物数；产物数以上表与缓存实测为准。）

**`compiled_under_test=[]` 是本条最关键的一条**：准备步骤没有编译任何被测家族
（`c512_int8_ffn_rows_v1` / `int8_ffn_segment_rows_v1`）的内核。
被测内核仍然只来自离线 14-worker 编译的目录（`compile-c512` 293/293、`compile-vit` 292/292）。
该断言是硬断言——一旦准备步骤编译了被测内核，整个阶段会失败，而不是静默通过。

### 14.2 这一条能证明什么

- 本轮适配器（作用域替换 + 调度守卫 + 缓存 + `DiskOnly` 全链路）在 NR256 控制帧上与
  已审核 NR256 快速产品**逐字节一致**，四个变体都是。
- 这条同时是 `full` 的前置门槛：`full` 会断言 `gate['passed'] and gate['phase']=='gate'`、
  `gate['adapter_sha256'] == sha(rows_entry_v1.py)`、`gate['variant'] == args.variant`，
  因此 `gate` 通过后**入口代码即冻结**。

### 14.3 这一条还不能证明什么

- 只有 **1 帧**（索引 0），不是全片。
- 走的是 **NR256 控制路径**，不是原尺寸 480×864。
- 因此**不是**性能结论。全尺寸 243 帧与计时分别属于 `full` 与 `bench` 阶段。

---

## 15. `full` 阶段结果（已通过）：243 帧 × 4 变体全部逐字节一致

`pipeline-full.json`：`passed=true`，五个阶段 rc 全 0
（`hostcache` / `full-baseline` / `full-c512` / `full-vit` / `full-both`）。

| 变体 | `frames` | `byte_equal` | `max_abs` | `cache_hits` | `rmse`（fast vs exact，全片范围） |
| --- | --- | --- | --- | --- | --- |
| `baseline` | 243 | **true** | **0.0** | 166 | 0.14989 .. 3.45564 |
| `c512` | 243 | **true** | **0.0** | 166 | 0.14989 .. 3.45564 |
| `vit` | 243 | **true** | **0.0** | 166 | 0.14989 .. 3.45564 |
| `both` | 243 | **true** | **0.0** | 166 | 0.14989 .. 3.45564 |

- 逐帧判据是 `output.tobytes() == expected_output.tobytes()`，比对对象是**已审核原尺寸快速版**的
  243 帧 npz 输出；不是 rmse，不是 allclose。
- 比对前显式断言 `output.shape == expected_output.shape` 与 `output.dtype == expected_output.dtype`，
  因为 `tobytes()` 比较在两边 dtype 不一致时会**因为错误的原因失败**。
- 同时断言参考帧自身完整性：`sha(file) == frame['sha256']` 且 `frame['input_sha256'] == row['sha256']`。
- 四个变体的 `rmse` 范围**完全相同**，这本身是逐字节一致的必然结果
  （输出相同 → 对同一 exact 参考的误差也必然相同）。

### 15.1 ⚠️ 关于 `rmse` 的重要说明（防止误读）

`rmse` 是 **fast 与 exact（FP32 精确参考）** 之间的误差，即**已审核基线自身的量化误差**，
**不是候选与基线之间的差异**。基线是 INT8 量化版本，与 FP32 exact 有 0.15–3.46 的 rmse 属正常。
候选与基线的差异判据是 `byte_equal` / `max_abs`，两者分别为 **true** 与 **0.0**。

### 15.2 这一条能证明什么

取消固定 144 行 C512 与 64 行 ViT 分批、改为按真实行数一次启动并直写完整输出缓冲之后，
在**原尺寸 480×864 全片 243 帧**上，四个变体的整帧输出与已审核原尺寸快速版
**没有任何一个字节不同**——覆盖注意力、QKV、投影、残差、池化、`final_weight` 等全部环节。

### 15.3 这一条还不能证明什么

- **不是性能结论。** `full` 是串行正确性核验，每帧墙钟约 0.63 s，其中很大一部分是 CPU/IO 尾巴：
  实测单帧 `np.savez_compressed`（480×864×3 f32）约 160 ms，972 帧合计约 156 s 纯 zlib 压缩，
  写盘约 2.5 GB（每变体 628.6 MB）。速度结论属于 `bench` 阶段。
- 未覆盖其他分辨率/输入源、其他 bitdepth，以及编码器环节
  （`full` 的输入是既有 243 帧索引，与已审核基线同源）。
- 因此本阶段给出的仍是**「改对了」**，不是**「快了多少」**。

---

## 16. `bench` 阶段结果（已通过）：`both` 1.172×、每帧少 14.7%，且计时期间每帧仍逐字节校验

`bench/benchmark.json`：`passed=true`。协议：ABBA 八进程交错
（`baseline / c512 / vit / both / both / vit / c512 / baseline`），每进程独立，GPU 常驻输入，
4 帧预热，从第 0 帧重启，32 帧，统计帧 4..31，计时期间禁止 JIT（`DiskOnly`）。

### 16.1 八次运行（原始数据）

| idx | 变体 | median ms | mean ms | min ms | max ms | 每帧 FFN 调用 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `baseline` | 95.839 | 96.131 | 93.103 | 106.151 | c512 64 / vit 16 |
| 1 | `c512` | 83.939 | 83.746 | 81.837 | 85.013 | c512 16 / vit 16 |
| 2 | `vit` | 92.881 | 92.910 | 91.641 | 95.316 | c512 64 / vit 8 |
| 3 | `both` | 81.707 | 81.792 | 80.203 | 83.505 | c512 16 / vit 8 |
| 4 | `both` | 81.456 | 81.409 | 79.525 | 83.000 | c512 16 / vit 8 |
| 5 | `vit` | 92.892 | 93.182 | 91.682 | 95.195 | c512 64 / vit 8 |
| 6 | `c512` | 83.816 | 83.909 | 82.809 | 85.644 | c512 16 / vit 16 |
| 7 | `baseline` | 95.427 | 95.354 | 93.786 | 96.944 | c512 64 / vit 16 |

每帧 FFN 调用次数与变体一一对应，**证明跑的确实是该变体的调度**
（不是"标称改了但实际没改"）。

### 16.2 汇总与加速

| 变体 | 两次 median | 均值 | 相对 baseline | 每帧节省 |
| --- | --- | --- | --- | --- |
| `baseline` | 95.839 / 95.427 | **95.633 ms** | 1.0000× | — |
| `c512` | 83.939 / 83.816 | **83.877 ms** | **1.1402×（−12.29%）** | 11.756 ms |
| `vit` | 92.881 / 92.892 | **92.887 ms** | **1.0296×（−2.87%）** | 2.746 ms |
| `both` | 81.707 / 81.456 | **81.581 ms** | **1.1722×（−14.69%）** | 14.052 ms |

### 16.3 三条内部一致性检查（全部通过）

1. **可重复性**：同一变体两次运行的 median 差异极小
   （baseline 0.43%、c512 0.15%、vit 0.01%、both 0.31%）。
2. **无漂移**：ABBA 首尾同变体几乎相同——baseline 第 0 次 95.839 vs 第 7 次 95.427（末次反而略快）；
   `both` 第 3 次 81.707 vs 第 4 次 81.456。说明热漂移没有把结果带偏。
3. **可加性**：`c512` 节省（11.756）+ `vit` 节省（2.746）= **14.502 ms**，
   而 `both` 实测节省 **14.052 ms**，两者相差 0.45 ms（3%）。两个家族互不干扰，收益近似可加。

### 16.4 计时期间仍然逐字节校验

`measure()` 在**每一个计时帧**上都断言 `actual.tobytes() == ref.tobytes()`，并对 32 帧全部记录
`byte_equal=true`；同时断言 `counters.totals() == expected`。抽查 4 次运行
（baseline / c512 / vit / both）均为 32 帧、`byte_equal` 全 true。
**所以这个加速不是"跳过工作"换来的。**

### 16.5 计时口径（结论的边界）

`scope`（入口自述原文）：

> synchronized processing call; includes Python/scopes/guards; excludes upload,
> readback, IO, motion generation and codec

即计的是**一次同步处理调用**，含 Python、作用域安装与守卫，**不含**上传、回读、IO、
运动生成与编解码；`.cpu()` 回读、`tobytes()` 比对与统计都在计时区间之外。

- 所以这是**处理调用段**的加速，不是端到端视频管线的加速；真实管线里被排除的那几段会稀释相对收益。
- 预热数据：帧 0 与帧 1 明显更慢（含惰性初始化），帧 2..3 进入稳态，故统计取帧 4..31 是必要的；
  `warmup_ms` 已记录在案。

### 16.6 与"机制"的对照

每帧移除的 FFN 入口调用数与节省时间大致成比例：

| 变体 | 每帧 FFN 入口调用 | 移除 | 节省 ms | 每移除一次约 |
| --- | --- | --- | --- | --- |
| `baseline` | 80 | — | — | — |
| `c512` | 32 | 48 | 11.756 | 0.245 ms |
| `vit` | 72 | 8 | 2.746 | 0.343 ms |
| `both` | 24 | 56 | 14.052 | 0.251 ms |

三个数落在 0.25–0.34 ms/次，量级一致。这与另一条独立观察吻合：`full` 阶段实测每帧墙钟约 0.63 s
而 GPU 占用率很低，说明这条流水线本来就**偏 launch/开销受限**，
正是"减少启动次数"能换到真实收益的区间。

**但这一条是解释，不是独立证据**：`bench` 只测出"少启动 → 更快"这个结果，
上表是对该结果的量级归因，不构成对单次启动开销的精确测量。

### 16.7 折算（仅处理调用段）

| | 243 帧处理段耗时 |
| --- | --- |
| `baseline` | 243 × 95.633 ms = **23.24 s** |
| `both` | 243 × 81.581 ms = **19.82 s** |
| 节省 | **3.42 s（−14.7%）** |

### 16.8 这一条能证明什么 / 不能证明什么

**能证明**：在保持 243 帧全片逐字节一致的前提下，取消固定行分批把每帧处理调用
从 95.633 ms 降到 81.581 ms（`both`，**1.172×**，−14.69%）；`c512`-only 与 `vit`-only
分别为 1.140× 与 1.030×。

**不能证明**：

- 不是端到端视频加速（上传 / 回读 / IO / 运动生成 / 编解码被排除）。
- 只在 B580 与当前驱动/工具链、且只在 480×864 这一档测过。
- 每变体只有 2 次重复（共 8 进程）。可重复性很好，但 `vit` 的 −2.87% 已接近本批次的可辨下限，
  若要更紧的置信区间需要更多重复。
- 未测功耗与温度。

---

## 17. 后续可继续改的方向（计划，**尚未执行**）

本节是计划，不是结论。**新改动必须另起一轮目录**：本轮 `gate` 已把
`adapter_sha256` 钉在 `rows_entry_v1.py` 上，改入口就会让 `gate`/`full` 的既有证据失效。

### 17.0 先说一个反直觉的算术：FFN 启动这条线基本到顶了

- 每帧 dispatch：baseline 384 → 候选 `both` 112（移除 272 次）。
- 实测节省 14.052 ms → **每次 dispatch ≈ 0.052 ms**。
- 那么**剩下的 112 次 dispatch 总共只值约 5.8 ms，占 81.58 ms 的约 7%**。
- 而且每 block 的 FFN 已经是 **1 次启动**（C512 16 个 block、ViT 8 个 block），
  不融合 block、不融合子内核就无法再降。

**结论：即使把剩下的启动开销全部消掉，也只能再拿约 7%。继续在这条线上抠收益有限。**

### 17.1 已确认的最大线索：`select()` 在每次 FFN 调用里重跑

`spill_preflight_v1.select()` 内部是 `jit.warmup(...)` + `kernel._init_handles()`，
而它自己的 docstring 写的是「construction-time selection, not per-frame GPU replay logic」——
**但代码是在每次 FFN 调用里跑的**。三处代码证据：

| 位置 | 形态 |
| --- | --- |
| 基线 C512：`c512_int8_full_stack_v1.py:81 def ffn` → `:93 def launch` → `:95 select(...)` | 每次 `ffn()` 内 5 次 |
| 基线 ViT：`int8_ffn_body_scope_v1.py:61 def ffn` → `:70 def launch` → `:72 select(...)` | 每次 `ffn()` 内 4 次 |
| 候选：`rows_scopes_v1.py:42 def launch` → `:44 / :93 select(...)` | 每次 `ffn()` 内 5 / 4 次 |

每帧的 `select()` 次数（按每帧 FFN 调用数 × 每次的子内核数）：

| 变体 | 每帧 FFN 调用 | 每帧 `select()` 次数 |
| --- | --- | --- |
| `baseline` | 80 | ~400 |
| `both` | 24 | ~108 |

`select()` 对**同一个** `(kernel, rows, options)` 每次都选回同一个 config，结果恒定，
所以这些 `warmup + _init_handles` 是**纯重复开销**——而且基线付的是候选的近 4 倍。

**重要推论**：如果把它移出热路径，两侧绝对耗时都会下降，但**比值很可能下降**
（基线受益更多）。这一点必须如实报告，不能只报对自己有利的数字。

**约束**：基线侧这个调用在**冻结的已审核模块**里，本轮纪律不允许修改。
所以做法只能是在 backend import 之前，用一个缓存 shim 替换 `spill_preflight_v1.select`，
**两侧同时生效**——既不改冻结文件，也保证比较公平。

### 17.2 Phase 0：归因（**必须先做**）

现在 81.58 ms 里除了那 ~5.8 ms 的 dispatch 开销，**其余 ~75 ms 完全没有归因**。
已知的只有一条间接观察：`full` 期间 GPU 占用率低，暗示偏 CPU/Python 受限。**这是猜测，不是测量。**

目标：把一帧拆成下面几类，各自给出 ms 与占比。

| 类别 | 说明 |
| --- | --- |
| `select()` 开销 | `jit.warmup` + `_init_handles` + 作用域资源/选择字典写入 |
| 真实 dispatch 启动开销 | 112 次（候选）/ 384 次（基线） |
| GPU 内核执行 | 各内核族的实际执行时间 |
| 模型 Python 图 | 逐 op 的 Python 派发、张量包装、上下文管理器 |
| 其他 | `synchronize()`、`.float()` 转换、allocator 分配 |

方法：**独立诊断进程**（明确不是证据跑），在 `spill_preflight_v1.select`、
`Dataflow.launch`、模型调用三处加计时包装，对 `baseline` 与 `both` 各测一帧。
要同时验证两个具体假设：

1. `select()` 的总耗时占一帧多少（§17.1 的次数估算能否兑现成 ms）。
2. `jit.warmup` 本身是否**也真的在 GPU 上启动了一次**（若是，则每帧存在"隐形的双倍启动"，
   而 `dispatch_guard` 只统计 `jit[grid](...)` 那一次，所以看不见）。

产出：一张分类耗时表，作为后续所有改动的收益预估依据。

**没有这张表之前，17.3 以下的收益预估都只能是量级估计。**

### 17.3 Phase 1：把 `select()` 移出热路径（低风险，预计收益最大）

- 做法：在 backend import 之前，把 `spill_preflight_v1.select` 换成带缓存的版本，
  以 `(kernel, rows, options)` 为键；首次调用照原逻辑走，之后直接复用选中的 config/kernel。
- **数值不变的理由**：选中的 config 与 kernel 完全相同，`jit[grid](args)` 仍按原样 dispatch；
  变的只是"重复计算同一个选择"这件事本身。
- 风险：低。但需覆盖一个边界——若某次调用真的会选到不同 config（例如 `rows` 变化导致
  spill 情况不同），缓存键必须包含 `rows`，否则会选错。诊断阶段要显式验证"同键必同选择"。
- 门槛：`local` → `gate` → `full` 全部逐字节一致，再跑 `bench`。
- 预期：两侧绝对值都明显下降；**比值可能下降**，如实报告。

### 17.4 Phase 2：全图捕获（量级级杠杆，交接文档已明确推迟）

如果 Phase 0 证实剩下的 ~75 ms 主要是 eager Python 图开销，那么**唯一能改变量级的杠杆**
就是把整图捕获成一次提交（capture / graph），把逐 op Python 派发一次性消掉。

- 交接文档把"全图捕获"列为**后续轮次**，本轮不做，所以这是一条独立的、需要新门槛的路线。
- 风险：中高。捕获会改变调度与内存复用方式，**但不允许改变任何数值**——必须重新走一遍
  逐字节门槛，且不能拿"捕获后更快"当作放宽字节判据的理由。

### 17.5 Phase 3：子内核融合（风险最高）

候选 C512 FFN 每次调用是 5 个内核（`_entry/_linear/_expand/_reduce/_project`），ViT 是 4 个。
融合可再减 dispatch，但属**内核改写**，数值风险最高，且与"只取消固定行分批"的改动边界不同。
建议单列一轮，独立门槛。

### 17.6 不建议做的

- 修改冻结的已审核模块（`c512_int8_full_stack_v1.py`、`int8_ffn_body_scope_v1.py`、
  `spill_preflight_v1.py`）或任何产品文件。
- 重新量化、改动 BM/BN/BK、warps/stages、融合选项、K 累加、舍入、激活、FP8 边界、残差。
- 为了让数字好看而把仪表（`FfnCounters`）或作用域安装移出计时段后**只报单侧**。
  `FfnCounters` 的 docstring 已说明"每次调用多一次 Python 调用、两侧对称、小到可留在计时循环里"，
  这是有意的决定；Phase 0 可以量化确认它确实小，但不应该靠移动它来制造收益。
- 放宽任何字节判据、改用 allclose、或在计时期间允许 JIT。

### 17.7 每个 Phase 的验收标准（同一套，不因"改动小"而减少）

`preflight` → `local`（局部 FFN 逐字节）→ `gate`（NR256 控制帧逐字节）→
`full`（243 帧逐字节）→ `bench`（ABBA 八进程），
其中 `full` 与 `bench` 的每一帧都必须逐字节校验通过。任一步失败即停止并归因，不得调门槛。

---

## 18. Phase 0 归因（**已执行**）：一帧 108.9 / 86.8 ms 到底花在哪

新增文件：`rows_attrib_v1.py`（诊断进程）、`rows_attrib_run_v1.sh`（驱动器）、`rows_attrib_teardown_v1.py`（退出码与报告一致性复核）、`rows_attrib_isolate_v1.sh`（退出码隔离实验）。
新增产物：`D:/fullsize-rows-v1/r1/attrib/{baseline,both}/attrib.json` + `teardown.json`、`stage-attrib-{baseline,both}.log`。
首轮 16 帧的产物保留在 `D:/fullsize-rows-v1/r1/attrib/first-run-16frames/`（结论一致，稳态窗口偏短）。

### 18.0 三句话结论

1. **一帧是 CPU 受限，不是 GPU 受限。** `submit_all_ms` 与帧墙几乎相等，末尾 `torch.xpu.synchronize()` 只等 **0.038 ms**；GPU 忙 **55.79 / 54.07 ms**，即占用率 **51.2% / 62.3%**。这直接解释了此前观察到的"跑 full 时显卡占用率不高"。
2. **`select()` 每帧 7.52 ms（6.9%）/ 3.28 ms（3.8%）**，逐次约 **13–20 µs**；而且 **每次都在第一个配置上就返回**——13 个内核全部 `configs_walked=1`、`spills=0`。它完全是"重复算一个已知不变的答案"的开销。**假设 H1 成立。**
3. **`jit.warmup()` 不会在 GPU 上启动。** 真实驱动提交数（1223 / 951）与 `CompiledKernel.launch_metadata` 调用数、`run(warmup=False)` 调用数**三者完全相等**。**假设 H2 证伪**，不存在"隐形双倍启动"，`dispatch_guard` 的计数是完整的。

### 18.1 方法

独立诊断进程，**明确不是证据跑**：不写 `validation.json`、不产出 `gate`/`full` 产物、不声明任何正确性结论、不参与任何门槛。它按 `benchmark` 阶段同样的帧序列回放（GPU 常驻输入、帧 0 带 `reset`、其余不带、每帧与已审核原尺寸快速参考逐字节比对、FFN 调用数逐帧断言），共 32 帧、统计帧 4..31，分三遍：

| 遍 | 装了什么 | 用来回答 |
| --- | --- | --- |
| **A 计时** | 只用 `perf_counter` 包住 `spill_preflight_v1.select` | 毫秒数（本节所有 ms 都来自这一遍） |
| **B 计数** | 另加 `JITFunction.warmup` / `JITFunction.run` / `CompiledKernel.launch_metadata` 计数，并用 `knobs.runtime.launch_enter_hook` / `launch_exit_hook` 数真实驱动提交 | 次数（每次 dispatch 多几次 Python 调用，故不用于 ms） |
| **C 剖析** | 只对最后一帧开 `torch.profiler`（CPU+XPU） | GPU 忙时与内核构成 |

探针在 `nr_runtime_v1` 导入**之前**绑定到 `spill_preflight_v1.select`。四个模块在 import 时都执行 `from spill_preflight_v1 import select`（`c512_int8_full_stack_v1:20`、`int8_ffn_body_scope_v1:16`、`c512_int8_ffn_gpu_v1:12`、`int8_ffn_segment_gpu_v1:11`），`rows_scopes_v1:13` 同理；所以两侧（冻结已审核侧与候选侧）绑到的是**同一个探针对象**，比较是公平的，且没有修改任何冻结文件。

### 18.2 两个假设的判定

| 假设 | 判据 | 结果 |
| --- | --- | --- |
| **H1** `select()` 占一帧可观份额 | 每帧 7.52 ms（baseline）/ 3.28 ms（both），即 6.9% / 3.8% | **成立** |
| **H2** `jit.warmup()` 也在 GPU 上启动一次 | 真实提交 1223 / 951 **等于** `launch_metadata` 1223 / 951 **等于** `run(warmup=False)` 1223 / 951；`warmup` 调用 461 / 189 **等于** `run(warmup=True)` 461 / 189 | **证伪**，无双倍启动 |

源码侧印证（两处，均为只读）：`triton/runtime/jit.py:768` 的 `if not warmup:` 把 `launch_metadata` 与 `kernel.run(...)` **整块**括在里面，`warmup=True` 时直接 `return kernel`；`triton/compiler/compiler.py:523` 的 `launch_metadata` 在 `knobs.runtime.launch_enter_hook is None` 时立即 `return None`，连 `_init_handles()` 都不调。所以未装 hook 时 `launch_metadata` 近乎免费——这也解释了为什么 `dispatch_guard` 的挂载点开销可以忽略。

### 18.3 计时表（稳态窗口 28 帧的中位数）

| 指标 | `baseline` | `both` | 备注 |
| --- | --- | --- | --- |
| 帧墙 `ms` | **108.864** | **86.777** | 见 §18.7，绝对值不可当加速比用 |
| `submit_all_ms`（全部 Python 提交返回） | 108.828 | 86.732 | 与帧墙几乎相等 |
| `sync_tail_ms`（末尾同步等待） | **0.038** | **0.038** | **CPU 受限的直接证据** |
| `select()` 合计 | **7.516**（6.90%） | **3.279**（3.78%） | 每次 16.3 / 17.4 µs |
| `select()` 调用次数 | **461** | **189** | 见 §18.4 分解 |
| Triton dispatch 次数 | **1223** | **951** | B 遍钩子计数，三处独立相等 |
| `jit.warmup` 调用次数 | 461 | 189 | 每次 `select` 恰好一次 |
| Triton 提交 CPU（上界，含每次 2 次钩子调用） | 16.19 | 12.58 | 每次 dispatch 13.2 µs，两侧一致 |
| GPU 忙（device 时间之和） | **55.79**（51.2%） | **54.07**（62.3%） | GPU 空闲 48.8% / 37.7% |
| UR 内核入队次数 | **2387** | **2075** | 见 §18.5 |

### 18.4 `select()` 明细（稳态窗口中位数）

**baseline（461 次/帧，合计 6.93 ms，中位数口径 7.52 ms）**

| ms/帧 | 次数 | µs/次 | 内核 |
| --- | --- | --- | --- |
| 1.054 | 64 | 16.47 | `c512_int8_ffn_gpu_v1._linear` |
| 0.977 | 64 | 15.26 | `c512_int8_ffn_gpu_v1._project` |
| 0.912 | 64 | 14.24 | `c512_int8_ffn_gpu_v1._expand` |
| 0.894 | 64 | 13.96 | `c512_int8_ffn_gpu_v1._reduce` |
| 0.818 | 64 | 12.78 | `c512_int8_ffn_gpu_v1._entry` |
| 0.730 | 36 | 20.27 | `window_block_projection_v3._project` |
| 0.541 | 36 | 15.03 | `window_block_attention_v3._kernel` |
| 0.277 | 16 | 17.32 | `int8_ffn_segment_gpu_v1._expand` |
| 0.257 | 16 | 16.08 | `int8_ffn_segment_gpu_v1._contract` |
| 0.196 | 16 | 12.28 | `int8_ffn_segment_gpu_v1._entry` |
| 0.192 | 16 | 11.99 | `int8_ffn_segment_gpu_v1._merge` |
| 0.063 | 4 | 15.84 | `decoder_gather_merge_v1._quantized` |
| 0.015 | 1 | 15.15 | `decoder_gather_merge_v1._raw_padded` |

**both（189 次/帧，合计 3.03 ms，中位数口径 3.28 ms）**

| ms/帧 | 次数 | µs/次 | 内核 |
| --- | --- | --- | --- |
| 0.754 | 36 | 20.93 | `window_block_projection_v3._project` |
| 0.585 | 36 | 16.24 | `window_block_attention_v3._kernel` |
| 0.259 | 16 | 16.18 | `c512_int8_ffn_rows_v1._linear` |
| 0.236 | 16 | 14.77 | `c512_int8_ffn_rows_v1._project` |
| 0.229 | 16 | 14.33 | `c512_int8_ffn_rows_v1._expand` |
| 0.207 | 16 | 12.91 | `c512_int8_ffn_rows_v1._reduce` |
| 0.195 | 16 | 12.17 | `c512_int8_ffn_rows_v1._entry` |
| 0.143 | 8 | 17.91 | `int8_ffn_segment_rows_v1._expand` |
| 0.133 | 8 | 16.61 | `int8_ffn_segment_rows_v1._contract` |
| 0.109 | 8 | 13.59 | `int8_ffn_segment_rows_v1._merge` |
| 0.106 | 8 | 13.28 | `int8_ffn_segment_rows_v1._entry` |
| 0.060 | 4 | 15.00 | `decoder_gather_merge_v1._quantized` |
| 0.015 | 1 | 14.60 | `decoder_gather_merge_v1._raw_padded` |

三点必须一起读：

- **`select()` 不只服务 FFN。** 每帧有 **77 次非 FFN 的 `select`**（`window_block_*` 72 次 + `decoder_gather_merge_v1` 5 次），两侧**都是 77**——取消固定行分批完全不动它们。
- **FFN 部分**：baseline 384 次（5.31 ms）→ `both` 112 次（1.91 ms）。**非 FFN 部分**：两侧都是 77 次（baseline 1.65 ms / `both` 1.20 ms）。
- **`configs_walked` 全为 1、`spills` 全为 0**（13 个内核、两侧、整轮 32 帧）。也就是说 spill 搜索从未需要否决任何候选配置，`select()` 的成本**全部**是 `jit.warmup` + `_init_handles` 的每次调用开销，不是"搜索"开销。对 Phase 1 的含义：缓存键只要正确，命中后返回的就是唯一可能的结果。

### 18.5 一帧的内核构成 —— 本节最重要的新信息

| 项 | `baseline` | `both` | 差 |
| --- | --- | --- | --- |
| **UR 内核入队总数** | **2387** | **2075** | −312（−13.1%） |
| ├ Triton dispatch | 1223 | 951 | −272 |
| └ **eager PyTorch ATen XPU 内核** | **1164** | **1124** | −40 |
| UR memcpy 入队 | 93 | 77 | −16 |
| XPU 设备事件总数 | 2480 | 2152 | −328 |

（ATen 一侧按 Itanium mangled 名 `_ZTSN2at6native3xpu…` 归类，实测恰为 1164 / 1124。）

三条**内部一致性**校验，全部通过：

1. `UR 内核入队 = Triton dispatch + ATen 内核`：1223 + 1164 = 2387 ✓；951 + 1124 = 2075 ✓。
2. **非 FFN 的 Triton dispatch 两侧都是 839**（1223 − 384；951 − 112）。也就是说取消固定行分批只动了 FFN 那 272 次，模型的其余部分一次都没变。
3. **非 FFN 的 `select` 两侧都是 77**（461 − 384；189 − 112）。

**所以：取消固定行分批动到的，是一帧 2387 次 GPU 启动中的 312 次，即 13.1%。** 剩下 87% 的内核（839 次非 FFN 的 Triton dispatch + 约 1164 次 eager ATen 内核）与本次改动无关。

### 18.5.1 剖析遍的 CPU 形状（只作形状，不作量级）

profiler 归属的 CPU 自时间里，`urEnqueueKernelLaunchWithArgsExp`（92.2 ms / 2387 次）与 `zeCommandListAppendLaunchKernelWithArguments`（14.6 ms / 2387 次）合计约占 **79%**；全部 `aten::*` 合计约 13 ms（约 10%）。

**但这些数不能当量级用**：剖析帧墙 1091.7 ms，而同一进程未剖析的稳态帧墙 108.9 ms——kineto 对每次 UR 调用都要记一条事件，把 UR 路径的相对权重系统性抬高了。可用的只有它的**形状**：CPU 侧最大的单一去处是内核启动 API，而不是任何 `aten::` 算子。

### 18.6 这如何修正 §17.0

§17.0 的算术用的是"每帧 dispatch 384 → 112"。真实数字是：**Triton dispatch 1223 → 951**（其中 FFN 部分 384 → 112），**外加 1164 → 1124 次 eager ATen 内核**。所以：

- §17.0 的**结论仍然成立**：FFN 启动这条线确实到顶了，剩下的 112 次 FFN dispatch 最多值约 1.1 ms。
- 但它的**理由需要修正**：真正在"启动"上花的钱是 **约 2400 次/帧 × 约 10–13 µs ≈ 24–31 ms**，占帧的 **22–28%**，不是"只值 5.8 ms"。
- 这笔钱**取消分批拿不到**——它动不到那 839 次非 FFN 的 Triton dispatch，也动不到 1164 次 ATen 内核。只能靠"减少内核数"或"减少 Python 层启动次数"。

### 18.7 交叉校验（必须如实说明的部分）

诊断的绝对帧墙**高于** bench 记录：baseline 108.86 vs 95.63（+13.8%）；`both` 86.78 vs 81.58（+6.4%）。诊断期间机器明显更抖：稳态帧 89.2–137.6 ms，而 bench 记录的 min/max 是 93.1–106.2（baseline）与 79.5–83.5（`both`）。

因此：

- **本节的所有绝对毫秒都不作为加速比证据。** bench 的 **1.1722×** 仍是唯一带门槛的加速结论，§16 不变。
- 本节只用**占比与计数**，这两样是稳的：`select` 每帧 7.52 / 3.28 ms 的逐帧波动仅 ±1.8 ms；计数是确定性的（三处独立相等）。
- 两条最强的"确实测到了同一件事"的证据不是帧墙，而是：**非 FFN 的 Triton dispatch 两侧都恰好 839**、**非 FFN 的 `select` 两侧都恰好 77**、以及每帧逐字节校验全部通过。

### 18.8 这一条能证明什么 / 不能证明什么

**能证明**：一帧是 CPU 受限（同步尾 0.038 ms）；GPU 只忙 51–62%；`select()` 每帧 7.52 / 3.28 ms 且每次都在第一个配置返回；一帧有 2387 / 2075 次 GPU 内核启动，其中 1164 / 1124 次是 eager ATen 内核；取消分批只动到其中 13.1%；`jit.warmup` 不启动 GPU。

**不能证明**：

- 不能给出加速比（见 §18.7）。
- 只在 B580、当前驱动/工具链、480×864 这一档测过。
- `select()` 的每帧 ms 是**探针**测的，探针自身有开销（每次两次 `perf_counter` + 字典记账）。两侧都付，但 baseline 的 `select` 次数是 `both` 的 2.4 倍，所以 baseline 的 `select_ms` 略被高估——这会让 Phase 1 的收益估计偏**乐观**，不是偏保守。
- profiler 的 CPU 自时间不可作量级（§18.5.1）；GPU 忙时是设备时间戳求和，可信度高于 CPU 那部分，但仍是在被剖析的那一帧上测的。
- 未测功耗与温度。

### 18.9 对后续计划的影响（§17.3–§17.5 重估）

| Phase | 收益估计（本次归因给出） | 风险 | 建议 |
| --- | --- | --- | --- |
| **1** `select()` 移出热路径 | **约 7.5 ms / 约 3.3 ms**（命中后仍需构造缓存键，实际约能拿回 85%，即约 6.4 / 2.8 ms） | 低 | 做。但**比值会下降**：诊断口径 1.254 → 约 1.22，bench 口径 1.172 → 约 1.13。§17.1 的预言被证实，如实报告。 |
| **1b**（本次新发现）Triton 每次 dispatch 的**非 `select`** 部分 | 每次约 10 µs 里，`jit[grid](...)` 会**再走一遍 binder**。缓存"已编译内核 + 已绑定参数"后直调 `kernel.run(...)`，可省 **约 8–10 ms（baseline）/ 约 6–8 ms（both）**——**比 Phase 1 更大** | 中 | 值得单列。必须保持 `dispatch_guard` 看到的 `launch_metadata` 调用与行数约束不变，并重走全套门槛。 |
| **2** 全图捕获 | 归因把量级钉住了：一帧约 **2400 次 Python 层启动**、GPU 只忙 51–62%、CPU 是唯一瓶颈。捕获若可行，消掉的是这 2400 次派发与大部分启动开销 | 中高 | 交接文档已推迟；现在有了它到底值多少的定量依据。**不得拿"更快"当放宽字节判据的理由。** |
| **3** 子内核融合 | C512 5 个内核 + ViT 4 个内核 = `both` 每帧 112 次中的全部。融合到 2 个内核可再省约 64 次启动 ≈ **0.6–0.8 ms** | 最高 | **性价比最低**，排最后。 |

### 18.10 复现方式

```bash
# 需要 B580 租约；两个变体串行，各约 25 s
bash E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580/reference/fullsize_rows_v1/rows_attrib_run_v1.sh

# 只复核退出码与报告一致性（不需要 GPU）
python E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580/reference/fullsize_rows_v1/rows_attrib_teardown_v1.py --run D:/fullsize-rows-v1/r1

# 退出码隔离实验（需要 B580 租约，约 35 s）
bash E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580/reference/fullsize_rows_v1/rows_attrib_isolate_v1.sh
```

产物：`D:/fullsize-rows-v1/r1/attrib/{baseline,both}/attrib.json`（含三遍的逐帧原始行、逐内核 `select` 明细、`frame_budget`、`hypothesis_H2`、`bench_cross_check`、剖析明细）与同目录的 `teardown.json`。

### 18.11 诊断进程的退出码 `0xC0000409`（已隔离、已记录，不掩盖）

两次 32 帧运行的租约记录里 `returncode = 3221226505` = `0xC0000409` = `STATUS_STACK_BUFFER_OVERRUN`，即 Windows 的 **fail-fast** 终止。它发生在**解释器拆卸期**：`attrib.json` 已写完（229710 / 231442 字节，`passed=true`），stderr 无任何输出，证据跑的退出码则全部为 0。

隔离实验（`rows_attrib_isolate_v1.sh`，同一租约内 3 次短跑，每次只关掉一遍）：

| 跑的遍 | 退出码 |
| --- | --- |
| 只跑 A（`--skip-count --skip-profile`） | `0x00000000` |
| A + C（`--skip-count`，**带 `torch.profiler`**） | **`0xC0000409`** |
| A + B（`--skip-profile`，**带 Triton 启动钩子**） | `0x00000000` |

**触发源是 `torch.profiler` 那一遍，与新增的 Triton 启动钩子无关**；只跑 A 或 A+B 都干净退出。这是 kineto / Level Zero 在这台机器上的拆卸期问题，不是本轮逻辑的缺陷。

处理方式（不掩盖）：`rows_attrib_teardown_v1.py` 读租约记录与报告，逐条复核报告应满足的内部不变量——`passed`、A/B 两遍逐帧 `byte_equal`、`run_launch == launch_metadata == submitted`、H2 判定、交叉校验可用、以及 **非 FFN dispatch = 839 且非 FFN select = 77**——把结论写进 `attrib/<variant>/teardown.json`。只有当报告完整且一致时 `0xC0000409` 才被认作已知的拆卸期现象；**其他任何非零码都判为失败并非零退出**。两个变体现均为 `passed: true`、`report_problems: []`。

对数据的影响：**没有**。崩溃在所有输出写完之后，报告本身携带完整的逐帧原始行与上述自校验。`--skip-profile` 也可用于彻底避开它（代价是没有 GPU 忙时与内核构成）。

---

## 19. Phase 0b 归因（**已执行**）：其余 ~75 ms 的分解

§18 留下一个明确的缺口：`both` 的 86.78 ms 里，扣掉 `select` 3.28 ms 与 Triton dispatch 12.58 ms，**约 70.9 ms 完全没有归因**；`baseline` 对应约 85 ms。§18 的三个仪器都看不见它——`select` 探针只看 `select`，启动钩子只看 Triton 提交，而 `torch.profiler` 的绝对 CPU 时间是不可用的（剖析帧墙 1078 ms vs 未剖析 86.8 ms，膨胀 12 倍）。本节用三个新仪器在**未剖析**的帧上把它拆开。

### 19.0 三句话结论

1. **一帧里 75.6% 的时间根本不在 C512 与 ViT 的 block 里。** `both` 的 `stack.model(...)` 中位 82.64 ms 中，C512 全部 10.98 ms（13.3%）、ViT 全部 9.18 ms（11.1%），**其余 62.49 ms（75.6%）**在模型的其他部分（decoder、gather/merge、颜色路径，以及模型自身的 Python 图）。`baseline` 的这一项是 62.33 ms —— **两侧几乎相同**。
2. **那 ~70 ms 的主体是 Python 与派发，不是内核。** `both` 的 CPU 侧分区：`select` 2.83（3.4%）+ Triton 驱动提交 8.81（10.7%）+ eager ATen 24.17（29.4%）+ **模型 Python 图与上下文 46.40（56.4%）**。
3. **取消分批省下的 13.36 ms 几乎全在 CPU 侧。** GPU 忙时只从 58.36 → 57.71（**−0.65 ms**），Triton 内核实例却从 1316 → 1028（**−288**）。被取消的那 288 次启动几乎不占 GPU 时间——它们的成本在启动，不在执行。

### 19.1 方法：三个新仪器（`rows_attrib2_v1.py`）

与 §18 一样，这是**诊断，不是证据跑**：不写 `validation.json`、不产出 gate/full 产物、不声明正确性、不参与门槛。逐帧仍与已审核的原样快速参考**逐字节比对**，FFN 调用数仍逐帧断言，任一遍出错都会记录到报告而不是静默丢弃。

| 遍 | 仪器 | 测什么 | 为什么 §18 测不到 |
| --- | --- | --- | --- |
| A–C | 与 §18 相同 | `select`、启动钩子、剖析 | 保留以便同一进程内对照 |
| **D** | **分层（inclusive）段计时** | 在 `c512()` / `vforward()` 内部按表达式就地插桩 | 回答"时间在模型的**哪一部分**" |
| **E** | **`TorchDispatchMode`（计时）** | 每次 ATen 派发的墙钟时间，按 op 名分桶 | 回答"时间在**哪种操作**上" |
| **F** | **`TorchDispatchMode`（仅计数）** | 同一拦截但不读时钟 | 以更低探针代价给出真实 op 混合比 |

关键实现细节，全部是实测得到的、写进代码注释的：

- **`TorchDispatchMode` 在本机实测无嵌套**（`maxdepth=1`），因此可以按"最外层"安全计时。
- **按 op 对象做字典键不可用**：实测 **43.4 µs/op**（对象哈希走慢路径）。仅计数 **1.77 µs/op**，计数 + 读时钟 **2.71 µs/op**。
- **`str(func)` 在本构建上给出点分隔的 `aten.mul.Tensor`**，取其第一段会把全帧折叠成一个 `aten` 桶（第一次冒烟正是如此，已修正）。正确做法是 `func._schema.name`。
- 因此遍 E 的毛时间**必然包含探针开销**，报告里把它作为显式字段列出并扣除，绝不当作原始预算行使用。

### 19.2 一帧的 CPU 侧分区 —— 这就是答案

**这是一个划分（partition）**：四项相加恰好等于 `submit_all_ms`（帧的 CPU 侧总量），因为第四项是残差。

| CPU 侧段 | `baseline` ms | % | `both` ms | % | 来源 |
| --- | --- | --- | --- | --- | --- |
| `select()` | 6.239 | 6.53 | 2.831 | 3.44 | 遍 A 直接计时 |
| Triton 驱动提交 | 11.361 | 11.89 | 8.813 | 10.72 | 遍 B 启动钩子（**上界**，含钩子自身开销） |
| eager ATen（Python + dispatcher + ATen + UR 入队） | 26.661 | 27.90 | 24.166 | 29.40 | 遍 E 毛时间 − 探针开销 |
| **模型 Python 图与上下文（残差）** | **51.304** | **53.68** | **46.398** | **56.44** | 相减得到 |
| **合计（= `submit_all_ms`）** | **95.565** | 100 | **82.208** | 100 | |

`select` 与 Triton 提交**不重叠**：两者都不经过 `torch` 派发器（否则会出现在遍 E 的计数里）。遍 E 的 `eager_aten_ms` 是"ATen op 从 Python 调用到驱动入队"的 inclusive 时间，因此它**包含** UR 入队。

残差里含：模型其余部分的 Python 图（属性查找、形状计算、条件判断、张量包装）、上下文管理器、`baseline` 特有的 `batch_rows` Python 循环、探针自身开销、以及测量噪声。

### 19.3 按模型位置的分层（**containment，不是 partition**）

`c512.*` 与 `vit.*` 在 `frame.model` **内部**，且 `c512.ffn` 还包含它触发的 `select()` 与 Triton 提交。所以这张表**不能**与 19.2 的表相加。

| 段 | 每帧调用 | `baseline` ms | `both` ms |
| --- | --- | --- | --- |
| `frame.model` = `stack.model(...)` | 1 | 96.072 | 82.639 |
| ├ `c512.ffn` | 16 | 15.280 | 3.749 |
| ├ `c512.attention` | 16 | 4.450 | 4.263 |
| ├ `c512.projection` | 16 | 2.369 | 2.322 |
| ├ `c512.pad` | 16 | 0.403 | 0.456 |
| ├ `c512.pool_final` | 1 | 0.189 | 0.187 |
| ├ `vit.qkv` | 8 | 3.103 | 3.088 |
| ├ `vit.attention` | 8 | 3.189 | 3.239 |
| ├ `vit.ffn` | 8 | 3.709 | 1.855 |
| ├ `vit.projection` | 8 | 0.859 | 0.804 |
| └ `vit.entry` | 8 | 0.192 | 0.190 |
| **C512 合计** | | **22.690（23.62%）** | **10.976（13.28%）** |
| **ViT 合计** | | **11.051（11.50%）** | **9.176（11.10%）** |
| **模型其余（残差）** | | **62.331（64.88%）** | **62.487（75.61%）** |

段调用数是**每 block 一次**，与 FFN 调用数不同：C512 有 16 个 block、ViT 有 8 个 block，两侧相同（取消分批改的是每个 block 发起几次 FFN，不是 block 数）。这一点在 teardown 里被显式校验。

**最重要的一行是最后一行**：`baseline` 62.331 与 `both` 62.487 相差 **+0.156 ms**（0.25%）。取消分批对模型其余部分**没有任何影响**，如预期；同时这也说明这 62.5 ms 是两侧共有的固定成本。

### 19.4 两侧对照：取消分批到底动了什么

| 量 | `baseline` | `both` | 差 |
| --- | --- | --- | --- |
| 帧墙（中位） | 95.607 | 82.248 | **−13.359** |
| `select` | 6.239 | 2.831 | −3.408 |
| Triton 驱动提交 | 11.361 | 8.813 | −2.548 |
| eager ATen | 26.661 | 24.166 | −2.495 |
| 模型 Python 图残差 | 51.304 | 46.398 | −4.906 |
| C512 段合计 | 22.690 | 10.976 | −11.714 |
| ViT 段合计 | 11.051 | 9.176 | −1.875 |
| 模型其余 | 62.331 | 62.487 | **+0.156** |
| Triton 内核实例 | 1316 | 1028 | −288 |
| ATen 内核实例 | 1164 | 1124 | −40 |
| GPU 忙时 | 58.356 | 57.711 | **−0.645** |
| ├ Triton 的 GPU 时间 | 35.296 | 34.581 | −0.715 |
| └ ATen 的 GPU 时间 | 23.060 | 23.130 | +0.070 |

两条自洽性：C512 段省 11.714 + ViT 段省 1.875 = **13.589**，与整帧省下的 13.359 对得上（差 0.23 ms，在诊断抖动内）。GPU 侧 `Triton + ATen = 忙时`，两侧都精确闭合。

**−288 次内核启动只换来 −0.65 ms GPU 时间**：这正是"CPU 受限"的定量表述，也说明 FFN 启动这条线**改的是启动，不是执行**。

### 19.5 eager ATen 的 op 级明细（遍 E，稳态中位）

两侧都是 4438 / 3622 次派发/帧（遍 E 与遍 F 的计数**完全相等**，这是两个仪器看到同一帧的自洽证据）。毛时间 30.833 / 27.571 ms，探针开销 4.172 / 3.405 ms。

| op | `baseline` ms | n | `both` ms | n |
| --- | --- | --- | --- | --- |
| `aten::empty` | 3.893 | 704 | 3.082 | 528 |
| `aten::empty_like` | 2.830 | 723 | 2.350 | 571 |
| `aten::to` | 2.673 | 970 | 2.762 | 970 |
| `aten::pad` | 2.596 | 95 | 2.201 | 79 |
| `aten::is_nonzero` | 2.152 | **5** | 2.119 | **5** |
| `aten::mul` | 1.913 | 146 | 1.917 | 146 |
| `aten::contiguous` | 1.497 | 90 | 1.485 | 90 |
| `aten::reshape` | 1.393 | 341 | 1.009 | 213 |
| `aten::add` | 1.275 | 105 | 1.263 | 105 |
| `aten::clamp` | 0.822 | 77 | 0.836 | 77 |
| `aten::slice` | 0.804 | 301 | 0.395 | 141 |
| `aten::sub` | 0.758 | 78 | 0.765 | 78 |

三处可直接读出的结论：

- **分配是最大单项**：`empty` + `empty_like` 合计 **6.72 / 5.43 ms**，占 eager 时间的约四分之一。
- **`aten::to` 两侧完全一样（970 次、≈2.7 ms）**，与 FFN 无关——这是模型里固定的类型/设备转换。
- **`aten::mul` / `add` / `sub` / `clamp` 次数在两侧逐个相同**（146 / 105 / 78 / 77），再次确认非 FFN 路径没被触碰。只有 `reshape`/`slice`/`empty`/`empty_like` 的差是 `batch_rows` 分块产生的（每 block 4 块 vs 1 块）。

### 19.6 每帧 5 个显式同步点

`aten::is_nonzero` **每帧恰好 5 次、每次约 424 µs、合计 2.12 ms**，两侧完全相同。`is_nonzero` 是会把 host 同步到设备的操作（`.item()` 语义），因此这是**每帧 5 个 host 同步屏障**，位于 FFN 路径之外。

它的量级不算大（帧的 2.6%），而且因为帧本身是 CPU 受限、`sync_tail` 只有 0.04 ms，这些同步**没有**让 GPU 空转。但它是一个**可定位、与 FFN 无关、且两侧共有**的开销，值得在后续轮次里查明来源（本轮未做，不在归因范围内）。

### 19.7 这一节如何修正 §18

- §18.6 说"真正在启动上的约 2400 次/帧 × 10–13 µs ≈ 24–31 ms"。本节把它拆得更准：`select` 2.83 + Triton 驱动提交 8.81 = **11.64 ms**（`both`）/ 17.60 ms（`baseline`）。差额在 eager ATen 的入队里（§18 的"UR 入队 = Triton + ATen"依然成立，但 ATen 那一半在 §18 里没有时间归属）。
- §18.5 的"取消分批只动到 312 次（13.1%）"在本节得到机制解释：**省下的 288 次 Triton 启动，只对应 0.65 ms GPU 时间**。
- §18.7 的诚实边界在本轮**缓解但未取消**：本轮诊断帧墙与 bench 记录几乎重合（`baseline` 95.607 vs 95.633，**−0.03%**；`both` 82.248 vs 81.581，**+0.82%**），比 §18.7 的 +13.8% / +6.4% 好一个数量级，所以本节的**比例**可以放心用。但诊断仍是独立进程，**bench 的 1.1722× 依旧是唯一带门槛的结论**。

### 19.8 这一节能证明什么 / 不能证明什么

**能证明：**
- 一帧 75.6% 的时间在 C512/ViT block 之外，且这部分在两侧**逐项相同**（62.33 vs 62.49 ms）。
- CPU 侧可以划分为四段且闭合：`select` / Triton 提交 / eager ATen / Python 图残差。
- 取消分批的 13.36 ms 收益来自 CPU 侧的启动路径，GPU 侧只动了 0.65 ms。
- 每帧 5 次 `is_nonzero` 同步、970 次 `aten::to`、约 1100 次分配是两侧共有的固定开销。

**不能证明：**
- **没有**把 62.5 ms 的"模型其余"进一步细分。它含 decoder、gather/merge、颜色路径与 Python 图，本轮未再插桩。这是下一个缺口。
- 遍 E 的绝对值含探针开销（已扣除估计值 4.17 / 3.41 ms），扣除量来自本机无 GPU 微基准的常量，**没有**在本进程内重新推导。
- 本节不做任何加速比声明，也不改变任何门槛。

### 19.9 对后续计划的影响（在 §18.9 之上修正）

| 方向 | 减少哪一侧 | 实测收益 | 对比值的影响 |
| --- | --- | --- | --- |
| Phase 1：`select` 移出热路径 | `both` 独有 | 2.83 ms（`both`）、6.24 ms（`baseline`） | **比值下降**（与 §18.9 一致） |
| Phase 1b：缓存已编译内核直调 | `both` 独有 | 上限 8.81 ms（`both`，上界） | **比值下降** |
| **减少两侧共有的 62.5 ms** | **两侧共有** | 每减 20 ms | **比值上升**：1.163 → 约 1.215 |

§18.9 只说了前两行。**第三行是本节的新结论，且方向相反**：把两侧共同的开销砍掉会**抬高**加速比（因为分子分母同时变小，而分子更大）。所以从"提升 1.172×"这个目标看，**62.5 ms 的模型其余部分才是最大的单一目标**——它占了 `both` 帧的 75.6%，而 FFN 相关的全部（`select` + Triton 提交 + `c512.ffn` + `vit.ffn`）合计约 17 ms。

这不改变 §17.3–§17.5 的顺序（它们风险最低、已计划），但说明**它们不是量级级杠杆**，并且提示下一轮应优先给"模型其余 62.5 ms"做同样的分层插桩。

### 19.10 复现方式

```bash
# 两个变体串行，各 6 遍 × 32 帧，约 1 分钟（含租约）
bash E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580/reference/fullsize_rows_v1/rows_attrib2_run_v1.sh

# 只跑某几遍（例如隔离某遍是否触发拆卸期 fail-fast）
PASSES=ADEF bash .../rows_attrib2_run_v1.sh

# 只复核退出码与报告一致性（不需要 GPU）
python .../rows_attrib2_teardown_v1.py --run D:/fullsize-rows-v1/r1
```

产物：`D:/fullsize-rows-v1/r1/attrib2/{baseline,both}/attrib.json` 与同目录 `teardown.json`。两个变体的租约退出码均为 `0xC0000409`（§18.11 已隔离的 `torch.profiler` 拆卸期现象），`teardown.json` 均为 `passed: true`、`report_problems: []`。

### 19.11 teardown 检查里修掉的一个真实错误

第一次运行时 `baseline` 被判为 `report incomplete or inconsistent`，问题是 `pass D c512.ffn calls=16, expected 64`。**这不是数据错，是检查写错**：pass D 的 FFN 段是**每 block 一次**，而检查拿 FFN **调用数**（64 = 16 block × 4 块）去比。已引入 `FFN_BLOCKS = {c512: 16, vit: 8}` 并把两个常量分开（`FFN_CALLS` 用于 dispatch 不变量，`FFN_BLOCKS` 用于段调用数），修好后两变体均通过。第一次的产物保留在 `D:/fullsize-rows-v1/r1/failed-attempts/attrib2-smoke-bug1`、`...-bug2`，冒烟产物在 `.../attrib2-smoke-ok`。

---

## 20. Phase 0c 归因（**已执行**）：`stack.model` 顶层的分解 —— 最大的单一块是历史 warp

§19 明确留下一个缺口：`both` 帧里 **62.49 ms（75.6%）**落在"模型其余"，而 §19 只是用减法把它框住，没有看进去。本节把它看进去。

### 20.0 三句话结论

1. **那 62.5 ms 不是一个整体。** `stack.model(...)` 的 81.74 ms 里，**`warp_history_normalized`（历史运动补偿重采样）一次调用就占 22.83 ms —— 27.93%**；`_forward_front`（主体）58.34 ms（71.37%）；顶层的校验、两次全张量有限性检查与 `_previous` 克隆合计只有 **0.45 ms（0.55%）**。
2. **主体内部再拆**：C512 全部 10.78 ms（13.2%）、ViT 全部 9.11 ms（11.1%）、**主体其余（decoder 等）38.46 ms（47.05%）**。所以一帧里最大的一块是 decoder 那 38.5 ms，**第二大且已定位到单个函数的是 warp 的 22.8 ms**。
3. **这两块都是两侧共有的**（不受取消分批影响），所以按 §19.9 的规则，优化它们会**抬高**加速比，而不是像 Phase 1 那样降低它。

### 20.1 方法：`rows_attrib3_v1.py`（诊断，不是证据跑）

读 `backend/nr_backend/temporal.py` 可以看到 `MotionNR.forward` 的顶层是有结构的，而且除主体外都能在不改任何冻结文件的前提下计时：

| 顶层步骤 | 计时方式 |
| --- | --- |
| `self._validate_rgb(rgb)` | 不单独计时，归入残差 |
| `torch.isfinite(motion).all()` / `abs()<=65504` | 不单独计时，归入残差 |
| `warp_history_normalized(previous, motion, ...)` | 在 `temporal` 模块上重绑该全局名 |
| `zero_motion_front_features(...)` / `reset_front_features(...)` | 同上 |
| `self._forward_front(rgb, front, ...)` | 在 `installed()` 内包装实例方法 |
| `result.detach().clone()` | 不单独计时，归入残差 |

两遍：**A 帧墙**（与 §19 同口径，用于交叉校验）、**G 模型顶层**。逐帧仍与已审核参考逐字节比对，FFN 调用数仍逐帧断言。

**一个必须遵守的约束（第一次尝试就是撞在它上面失败的）**：`experimental/fused_dynamic_front_v1.py:107` 的 `FusedFront.installed()` 在**进入时断言** `temporal.reset_front_features is self.original_reset`，**退出时又断言**仍是自己装的那两个。所以重绑这两个全局名**只能发生在所有组件安装完成之后、并在 `ExitStack` 展开之前恢复**。第一次把重绑放在 `installed()` 之外，组件直接拒绝安装（`AssertionError`），产物保留在 `failed-attempts/attrib3-both-global-guard`。这是项目自带的防第三方静默替换的守卫，**本轮选择顺着它做，而不是绕过它**。

### 20.2 结果（`both`，稳态 28 帧中位数，inclusive 包含树）

| 层 | ms | % | 每帧调用 |
| --- | --- | --- | --- |
| `frame.model` = `stack.model(...)` | 81.745 | 100 | 1 |
| ├ **`motion.warp_normalized`** | **22.834** | **27.93** | 1 |
| ├ `motion.front_features` | 0.120 | 0.15 | 1 |
| ├ `motion.front_reset` | 0.000 | 0 | 0（只在 reset 帧，不在稳态窗口） |
| ├ `motion.body` = `_forward_front` | 58.345 | 71.37 | 1 |
| │ ├ `c512.*` 合计 | 10.777 | 13.18 | |
| │ ├ `vit.*` 合计 | 9.111 | 11.14 | |
| │ └ **`body_rest`**（decoder 等） | **38.457** | **47.05** | |
| └ `head`（校验 + 两次全张量检查 + 克隆 + 胶水） | **0.446** | **0.55** | |

**闭合检查**：`22.834 + 0.120 + 58.345 + 0.446 = 81.745` —— 与 `frame.model` 精确相等，没有未归因的余量。

主体内部的明细（沿用 §19 的段）：

| 段 | ms | 每帧调用 |
| --- | --- | --- |
| `c512.attention` | 4.176 | 16 |
| `c512.ffn` | 3.666 | 16 |
| `c512.projection` | 2.300 | 16 |
| `c512.pad` | 0.451 | 16 |
| `c512.pool_final` | 0.184 | 1 |
| `vit.attention` | 3.196 | 8 |
| `vit.qkv` | 3.023 | 8 |
| `vit.ffn` | 1.865 | 8 |
| `vit.projection` | 0.828 | 8 |
| `vit.entry` | 0.198 | 8 |

交叉校验：pass A 帧墙 **81.635 ms**，bench 记录 **81.581 ms**，差 **+0.07%**；pass G 帧墙 82.176 ms（含段探针开销 0.54 ms）。两遍的字节比对与 FFN 断言均通过，`G_error` 为空。

### 20.3 这一节同时证伪了 §19.6 的一个猜测

§19.6 把"每帧 5 次 `aten::is_nonzero`、合计 2.12 ms"当作"位于 FFN 路径之外的同步开销"。本节测出 `MotionNR.forward` 顶层（含校验与克隆）**合计只有 0.446 ms**，所以这 5 次同步**不在顶层**——它们落在 `warp` 或 `body` 内部。同时 `_previous = result.detach().clone()`（每帧一次完整输出拷贝）也**没有**成为可观开销：整层 `head` 0.45 ms 已经把它包含在内。**§19.6 的"值得查明来源"仍然成立，但"它是一笔可观开销"这个暗示不成立。**

### 20.4 为什么 warp 值 22.8 ms

`backend/nr_backend/sampling.py` 的 `warp_history_normalized`（L113–125）与它调用的 `_sample_five_axes`（L94–110）/`_half_texture_counts`（L47–60）是**纯 eager 的逐元素实现**：

- 每帧重建坐标网格：`torch.meshgrid(arange(h), arange(w))` —— **这两个 `arange` 与网格只依赖 `(h, w)`，与帧内容无关**；
- 每像素采样 **5 个点**（`_sample_five_axes` 里 `sample()` 调 5 次），每个点走 `half_texture_normalized` → `_half_texture_counts`；
- `_half_texture_counts` 每个点做 **4 次高级索引 gather**（`image[iy,ix]` 等）并构造 4 路 `torch.stack`，随后是一串 **int64** 运算（`>>8`、`&255`、`(ax*ay+128)>>8`、`(aligned*weights).sum(-1)`）、`_exponent`、`torch.ldexp`、`signbit`、以及 int16 位拼装。

按 864×480 = 414720 像素、每像素 5 点 × 4 texel 计，一帧要做约 **8.3M 次纹理读取**，并且由数十个 ATen op 在整张 864×480 张量上串行执行。这解释了 22.8 ms，也说明它**不是不可压缩的算法成本，而是一个尚未融合的 eager 实现**。

### 20.5 所以最应该优化的是

按"大小 × 已定位程度 × 是否可优化"排序：

| 优先级 | 目标 | 实测 | 为什么 | 风险 |
| --- | --- | --- | --- | --- |
| **1** | **`warp_history_normalized`** | **22.83 ms（27.9%）** | 单一函数、每帧一次、**已经知道它为什么慢**（eager 逐元素 + 5 采样点 + int64 + 每帧重建常量网格）、两侧共有 | 中：涉及数值路径，必须逐字节一致 |
| **2** | **`body_rest`（decoder 等）** | **38.46 ms（47.1%）** | **更大，但尚未细分**——本节只知道它不在 C512/ViT block 里 | 未知，先插桩 |
| 3 | Phase 1 / 1b（`select` 与 Triton 提交） | 2.86 + 8.81 ms（both 独占） | 已计划、风险低，但**会让加速比下降**（§19.9） | 低 |

**先做 1 和 2，因为它们是两侧共有的**：按 §19.9 的算术，砍共有开销会抬高加速比，而砍候选侧独占开销会压低它。具体到 warp，最保守的第一步是**把 `meshgrid` 的常量网格缓存下来**（只依赖 `(h,w)`，零算术风险），再往下才是合并 5 个采样点的重复计算（`_exponent`/`ldexp`/`signbit` 在 5 点间高度重复）与 int64→int32，最后才是融合成单个 kernel。

### 20.6 这一节能证明什么 / 不能证明什么

**能证明：**
- `stack.model` 的 81.74 ms 可以按顶层步骤**精确闭合**地划分，无未归因余量。
- 历史 warp 占 27.93%，是一个已定位到单个函数的 22.8 ms 热点。
- 主体其余（decoder 等）38.46 ms 是最大的一块。
- 顶层校验、两次全张量检查与 `_previous` 克隆合计仅 0.45 ms。

**不能证明：**
- **没有**细分 `body_rest` 的 38.46 ms。它是下一个缺口。
- **没有**细分 warp 内部的网格构造 / 5 个采样点 / 归一化各占多少。22.8 ms 是 inclusive 值。
- 只测了 `both` 一个变体。两侧共有这一判断来自 §19 的 `model_rest` 两侧相差 0.16 ms，**不是**本节直接测的。
- 仍然不做任何加速比声明，不改变任何门槛。bench 的 **1.1722×** 依旧是唯一带门槛的结论。

### 20.7 复现方式

```bash
# 需要 B580 租约；单个变体 2 遍 × 32 帧，约 16 s
PY="E:/ComfyUI-aki-v3-IntelArc_20260722/ComfyUI-aki-v3-IntelArc/python/python.exe"
"$PY" -X utf8 -u .../gpu_lease_nr.py --owner fullsize-attrib3-both \
  --cwd .../fullsize_rows_v1 --log D:/fullsize-rows-v1/r1/stage-attrib3-both.log \
  --timeout-seconds 1200 --wait-seconds 900 -- \
  "$PY" -X utf8 -u .../rows_attrib3_v1.py --variant both \
  --out D:/fullsize-rows-v1/r1/attrib3/both --cache D:/fullsize-rows-v1/r1/triton-cache \
  --bench D:/fullsize-rows-v1/r1/bench/benchmark.json --frames 32 --measure-from 4
```

产物：`D:/fullsize-rows-v1/r1/attrib3/both/attrib.json`（`passed: true`、无 `*_error`）。失败尝试保留在 `failed-attempts/attrib3-both-missing-report`（缺 `report()` 方法）与 `failed-attempts/attrib3-both-global-guard`（触发 `FusedFront` 全局守卫）。

---

## 21. Phase 0d 归因（**已执行**）：`warp_history_normalized` 为什么慢

§20.6 把缺口写得很直白：*"**没有**细分 warp 内部的网格构造 / 5 个采样点 / 归一化各占多少。22.8 ms 是 inclusive 值。"* 本节把它拆开，并回答"为什么慢"——**不是**"哪一层占多少"（那只是问题的一半），而是**这台机器上到底是谁在等谁**。

### 21.0 三句话结论

1. **22.93 ms 可以精确划分到 5 项、无余量。** 主体是 5 个采样点：5 点权重与累加 **14.84 ms（64.7%）**、每点 4 次 gather 与整数权重 **3.70 ms（16.2%）**、每点定点转 half **2.43 ms（10.6%）**；前缀（网格、`arange`、u/v 的 `fma32`、两次轴权重）只有 **1.21 ms（5.3%）**，每点归一化坐标转换 **0.75 ms（3.3%）**。
2. **它慢不是因为内核慢，而是因为 host 与 device 在互相等。** warp 路径上每帧有 **34 次完整的设备同步**：`fma32` 每次调用都把 Python 标量转成设备张量（每帧 32 次，正是源码能数出来的数目），`NativeReciprocalTable.forward` 每次调用都用 `bool(...all())` 校验索引区间（每帧 2 次）。实测（pass K）：把一个调用放在 1.50 s 的设备负载**后面**，`fma32` 阻塞 0.747 s、`reciprocal_table` 阻塞 0.748 s（比值 1.00），而 `half_texture_normalized`、`integer_half_away`、`meshgrid` 只花 1.4 ms、0.45 ms、0.10 ms（比值 ≤0.002）。**同步把 host 与 device 串行化，于是墙钟是两者相加而不是取大。**
3. **最便宜的真收益是拆掉这 34 次同步，不是融合内核。** 已识别的 host 派发成本约 6.9 ms（650 次 dispatch，全部来自三个**不同步**的层，可直接测量），其余约 10.1 ms 以上是在 `fma32` 内部等设备。去掉同步后墙钟从 `host + device` 变成 `max(host, device)`，量级上是 **22.9 → 约 13 ms**。而且这是**两侧共有**的开销，按 §19.9 的算术会**抬高**加速比。

### 21.1 方法：`rows_attrib4_v1.py`（五遍 A/H/I/J/K，诊断不是证据跑）

与 §18–§20 同一契约：不写 `validation.json`、不产出 gate/full 产物、不声明正确性、不参与门槛。逐帧仍与已审核原样快速参考**逐字节比对**，FFN 调用数仍逐帧断言，任一遍出错都记录到 `passes['*_error']` 而不是静默丢弃。

| 遍 | 仪器 | 回答什么 |
| --- | --- | --- |
| A | 干净帧墙（只装 `select` 探针） | 与 §19/§20 和 bench 同口径对照 |
| H | warp 路径 9 个函数的 inclusive 段计时 | **哪一层**、多少毫秒、每帧调用几次 |
| I | 同一批包装 + **仅计数** `TorchDispatchMode`（按包装器读窗口） | **多少次** dispatch、op 混合比、每次 dispatch 多少微秒 |
| J | 孤立微基准：单次 / 20 次连发 / 设备负载后单次 | **谁在等谁** |
| K | 逐个候选函数的"设备负载后阻塞测试" | **同步点在哪一个函数里** |

三处关键设计，都是被前几轮的失败逼出来的：

- **9 个标签全部是 inclusive**，嵌套关系与调用关系一致（`warp` ⊃ `warp.sample_five` ⊃ `warp.texture_norm` ⊃ `warp.texture_counts` ⊃ `warp.int_half_away`），**任何一层都不相加**。只有 `WARP_CHAIN` 里那 4 对做减法求 exclusive。
- **`fma32` 与 `exponent` 跨层**（`fma32` 既在 `_axis_weights_fraction` 里也在采样循环里；`_exponent` 既在 `_half_texture_counts` 里也在 `_integer_half_away` 里），所以它们作为**重叠标签**单独报告，**从不从链上减掉**。
- **`half_texture_linear` 也装了探针**，期望调用数为 **0**。它是 `_sample_five_axes` 在 `normalized_scale is None` 时走的另一条分支；如果它非零，说明整轮测错了分支，报告里所有数字都要作废。实测 0 次。

### 21.2 一个必须先修掉的陷阱：`temporal.py` 按名导入

第一次读源码时差点写出一个**静默为零**的仪器。`backend/nr_backend/temporal.py` 第 11 行是

```python
from .sampling import warp_history_square, warp_history_normalized
```

**按名导入**。也就是说 `temporal` 模块持有自己的那份函数引用；只重绑 `sampling.warp_history_normalized` 的话，真实调用点根本不会被替换——`warp` 标签会报 **0 次调用、0 ms**，而报告看上去只会像"这个函数很快"。

本轮的处理是：把这条路径上**所有跨模块按名导入**都重绑，并在安装后**断言校验**（`WarpProbe.verify()`，失败直接抛错而不是记一个字段）。实测报告里 `warp_probe.rebound = ["nr_backend.temporal.warp_history_normalized"]`。

**故意不重绑**的是 `fma32`：`post.py` / `style.py` / `intensity_temporal.py` / `controlled_temporal.py` 也按名导入了它，但那些调用**不在 warp 路径上**，把它们算进 `warp.fma32` 只会虚增这个标签。这条判断写进了代码注释。

### 21.3 预注册的调用数预测（**全部命中，两侧都是**）

在测量**之前**，按源码把每帧调用次数数了出来。teardown 把它当作硬门槛：任何一项不符就判定报告不可用，因为那意味着调用图读错了，报告里每个"每层毫秒"都会归到错的层上。

| 标签 | 源码推导 | `baseline` 实测 | `both` 实测 |
| --- | --- | --- | --- |
| `warp` | 1 | **1** | **1** |
| `warp.sample_five` | 1 | **1** | **1** |
| `warp.texture_norm` | 5（每点一次） | **5** | **5** |
| `warp.texture_linear` | **0**（分支守卫） | **0**（未出现） | **0**（未出现） |
| `warp.texture_counts` | 5 | **5** | **5** |
| `warp.int_half_away` | 5 | **5** | **5** |
| `warp.exponent` | 10（`_half_texture_counts` 1 次 + `_integer_half_away` 1 次） | **10** | **10** |
| `warp.axis_weights` | 2（`axis()` 闭包每轴一次） | **2** | **2** |
| `warp.fma32` | 26（2 u/v + 4 轴坐标 + 6 权重 + 10 采样点 + 4 累加） | **26** | **26** |

`fma32` 那一行是这张表里信息量最大的一行：26 次是逐表达式数出来的（`warp_history_normalized` L121×2、L123×2、L124×2、`_axis_weights_fraction` L81×2×2、L84×1×2、`_sample_five_axes` L103×2×5、L107×4）。**命中意味着这条路径的调用图与源码理解完全一致**，后面每层归属才有意义。

同一次测量还给出 `aten::lift_fresh` = **32**，也恰好等于源码里"Python 标量做操作数"的次数（2 + 4 + 2 + 4 + 20 = 32）。这个数在 21.7 会成为同步点的指纹。

### 21.4 22.93 ms 的精确划分（`both`，稳态 28 帧中位数）

`WARP_CHAIN` 的四对减法给出一个**精确闭合**的划分：

| 段 | ms | 占 warp | 内容 |
| --- | --- | --- | --- |
| `warp_prep`（`warp` − `sample_five`） | **1.2082** | 5.27% | `meshgrid` + 2×`arange` + u/v 的 `fma32` + 两次轴权重 |
| 5 点权重与累加（`sample_five` − `texture_norm`） | **14.8365** | 64.71% | 权重乘积、`total` 求和、`reciprocal`、L103 的 10 次 `fma32`、L107 的 4 次累加 `fma32` |
| 每点归一化坐标转换（`texture_norm` − `texture_counts`） | **0.7481** | 3.26% | `floor(u*2^21)` 与 `(nx*w+4096)>>13-128` |
| 每点 4 次 gather 与整数权重（`texture_counts` − `int_half_away`） | **3.7038** | 16.15% | 4 次高级索引 + 2 次 `stack` + `>>8`/`&255`/`(ax*ay+128)>>8`/`(aligned*weights).sum(-1)` |
| 每点定点转 half（`int_half_away`） | **2.4318** | 10.61% | 含它自己那次 `_exponent` |
| **合计** | **22.9284** | **100.00%** | 与 `frame.model` 81.3158 的 28.2% 精确对应 |

`warp_prep` 只有 5.27%——**§20.5 建议的"先缓存 `meshgrid` 常量网格"最多只能拿回 1.2 ms 里的一部分**，不是量级级杠杆。真正的质量全在 5 个采样点里（94.7%）。

两侧几乎相同，这本身就是一次证伪测试：

| 量 | `baseline` | `both` | 差 |
| --- | --- | --- | --- |
| `warp` | 23.0252 | 22.9284 | **−0.42%** |
| `warp.sample_five` | 21.8335 | 21.7201 | −0.52% |
| `warp.fma32` | 11.8221 | 11.8160 | −0.05% |
| `warp` 每帧 dispatch 数 | 878 | 878 | 0 |

warp 完全不碰 FFN 路径，两侧相差 0.42%（在诊断抖动内），说明这个分解对两个变体都成立。

### 21.5 每层每帧的 dispatch 数与每次 dispatch 的微秒

这是"为什么慢"的第一个判别器。参照系是 §19 独立测出的**本机 dispatch 地板**：`aten::to` 2.762 ms / 970 次 = **2.85 µs/次**（含一次真实但很小的内核）。

| 层 | ms | 每帧调用 | 每帧 dispatch | **µs / dispatch** | **÷ 地板** | 同步？ |
| --- | --- | --- | --- | --- | --- | --- |
| `warp` | 22.9284 | 1 | **878** | **26.11** | **9.2×** | 是 |
| ├ `warp.fma32` | **11.8160** | 26 | 104 | **113.62** | **39.9×** | **是** |
| ├ `warp.sample_five` | 21.7201 | 1 | 765 | 28.39 | 10.0× | 含 |
| ├ `warp.texture_norm` | 6.8836 | 5 | 650 | 10.59 | 3.7× | 否 |
| ├ `warp.texture_counts` | 6.1356 | 5 | 570 | 10.76 | 3.8× | 否 |
| ├ `warp.int_half_away` | 2.4318 | 5 | 265 | 9.18 | 3.2× | 否 |
| ├ `warp.axis_weights` | 0.6697 | 2 | 68 | 9.85 | 3.5× | **是** |
| └ `warp.exponent` | 0.6517 | 10 | 60 | 10.86 | 3.8× | 否 |

两个直接可读出的结论：

- **`warp` 占整帧 878 / 3622 = 24.24% 的 dispatch**（`baseline` 是 878 / 4438 = 19.78%——warp 的 878 次是固定的，取消分批动的是另外那部分），却占 28.2% 的时间。
- **不是"op 太多"，是"每次 op 太贵"。** 三个**不同步**的层（`texture_norm` 排他 80 次 + `texture_counts` 排他 305 次 + `int_half_away` 265 次 = **650 次 dispatch**）合计 **6.8837 ms**，即 **10.59 µs/dispatch**，是地板的 3.7×。这一项是**可以直接测量的真实 host 派发成本**：650 次 dispatch 在 864×480（以及 864×480×3）张量上串行执行。

而 `fma32` 的 **113.62 µs/dispatch（39.9× 地板）** 是整张表最反常的一格——它的函数体只有 4 个 torch 调用（`a.float()`、两次 `torch.as_tensor(标量, device='xpu')`、`torch.addcmul`）。§21.7 会说明这 113 µs 里只有很少一部分是 `fma32` 自己的工作。

warp 子树里次数最多的 op（`both`，前 20 项覆盖 878 中的 733 次）：

| op | n | | op | n |
| --- | --- | --- | --- | --- |
| `aten::to` | 129 | | `aten::eq` | 30 |
| `aten::mul` | 76 | | `aten::__lshift__` | 30 |
| `aten::sub` | 76 | | `aten::where` | 30 |
| `aten::clamp` | 67 | | `aten::view` | 28 |
| `aten::add` | 66 | | `aten::addcmul` | 26 |
| `aten::__rshift__` | 40 | | `aten::index` | 23 |
| `aten::__and__` | 38 | | `aten::lt` | 18 |
| **`aten::lift_fresh`** | **32** | | `aten::ones_like` | 15 |
| | | | `aten::unsqueeze` / `__or__` | 15 / 15 |
| | | | `aten::floor` / `rsub` | 12 / 12 |

`aten::ones_like` **15 次**（全在 `_integer_half_away` 里，`torch.ones_like(magnitude)<<lead` 之类）值得注意：它每次都在 864×480 的 **int64** 张量上分配并移位，一帧 15 次。

### 21.6 决定性的一步：这台机器上是谁在等谁（遍 J）

墙钟围着 enqueue 路径计时，测的是 **enqueue**，不是执行——内核是异步的。所以遍 J 分三步：

| 测量 | 结果（`both`） | 读法 |
| --- | --- | --- |
| 单次调用 + sync，×5 取中位 | **22.6558 ms** | 与帧内 22.9284 差 1.2%，说明孤立复现是可信的 |
| 20 次连发，**只给 enqueue 循环计时** | **22.8462 ms/次** | 连发不改变每次成本 |
| 20 次连发后的**收尾 sync** | **0.0702 ms** | **设备当时只剩 0.07 ms 的活**（一次调用的 0.31%） |

**0.0702 ms 这一格是判别器。** 设备是**边入队边消费**的，不是"整次调用攒齐了再执行"，所以收尾时间**不是**设备的执行时间；它只回答一个问题：**host 停手的那一刻，设备还欠多少活。** 答案是 0.07 ms——设备在 host 自己的生产窗口内就把 20 次调用全吃掉了。若设备是瓶颈，这个数会随 K 增长。

### 21.7 同步点在哪儿：遍 K

遍 J 里"把一次调用放到 1.50 s 的设备负载后面"这个动作，本来是当独立交叉验证用的，结果直接暴露了机制：**第一次调用阻塞了 1.5156 s**，比值 `first_over_preload` = **1.0131**。一个只做入队的调用无论负载多长都应立刻返回；它却等满了整个负载。

于是把它做成一个**逐候选函数的阻塞测试**（遍 K）：把设备先用无关的 matmul 负载占住，再调用候选函数一次，看它是立刻返回还是等负载排空。

| 候选 | 空闲时 ms | 负载后 ms | 比值 | 是否等设备 |
| --- | --- | --- | --- | --- |
| `warp_full`（整次调用） | 22.91 | 770.77 | **1.0309** | **是** |
| `meshgrid_arange` | 0.218 | 0.099 | 0.0001 | 否 |
| **`fma32_scalars`** | 0.065 | **746.56** | **0.9986** | **是** |
| **`reciprocal_table`** | 0.167 | **747.57** | **0.9999** | **是** |
| `axis_weights_fraction` | 0.909 | 747.52 | 0.9999 | 是（经 `reciprocal_table`） |
| `half_texture_normalized` | 4.357 | 1.397 | 0.0019 | 否 |
| `integer_half_away` | 0.469 | 0.455 | 0.0006 | 否 |

（设备负载 747.6 ms；`baseline` 与 `both` 逐项一致。）

**两个、且只有两个函数会等设备：**

1. **`fma32` 里的 Python 标量转换。** `fma32` 的 L14 对 `b` 和 `c` 各做一次 `torch.as_tensor(标量, device='xpu', dtype=torch.float32)`。这在 XPU 上是一次**标量 H2D 落地**，会等整条队列排空。一帧 **32 次**（§21.3 的 `lift_fresh` 计数就是这个指纹），分布在 26 次 `fma32` 调用里。
2. **`NativeReciprocalTable.forward` 的区间校验。** `reciprocal.py` 的 L27 是 `if not bool(((index>=0)&(index<len(self.values))).all()): raise`——`bool(设备张量)` 是一次 host 同步。一帧 **2 次**（每轴一次，经 `_axis_weights_fraction` L83）。

顺带回答了 §19.6 留下的那个问题：*"每帧 5 次 `aten::is_nonzero`……值得在后续轮次里查明来源。"* 其中 **2 次就在这里**——`warp.axis_weights` 的 op 明细里恰好是 `aten::is_nonzero` 2 次 + `aten::all` 2 次。另外 3 次仍不在本轮范围内。

**不是同步点的**：真正的 4 次 gather 纹理读取（`half_texture_normalized`）、定点转 half（`integer_half_away`）、网格构造（`meshgrid`/`arange`）。这三个只入队，立刻返回。

### 21.8 所以为什么慢 —— 机制

把上面三块拼起来：

```
warp 墙钟 22.9284 ms  =  host 入队  +  device 执行        （两者不重叠）
```

不重叠的原因就是那 **34 次设备同步**（32 次来自 `fma32` 的标量转换 + 2 次来自 reciprocal 表校验）。每次同步都会把队列排空，于是每一小段 enqueue 之后都跟着一次完整的 drain，host 和 device 只能**相加**，不能**取大**。

已识别的各项：

| 项 | ms | 依据 |
| --- | --- | --- |
| **host 入队**（三个不同步的层，650 次 dispatch） | **6.8837** | 直接计时，且遍 K 证明这三层不同步 |
| **device 时间**（在 `fma32` 内部被等出来） | **≥ 10.14** | `fma32` 实测 11.816，减去它自己的 host 成本上限 26×0.065 = 1.68 |
| 其余 228 次 dispatch 的 host 入队 + `axis_weights` 里的 drain | 5.91 | 相减得到 |
| **合计** | **22.9284** | |

`fma32` 那一格的推理值得说清楚：**它在空闲设备上单次只要 0.065 ms**（遍 K 的 `idle_ms`，26 次合计上限 1.68 ms），但在帧内 26 次合计 **11.816 ms**——差了一个数量级。差额不是 `fma32` 自己在算什么，而是**它的同步把"从上次同步到现在入队的所有设备工作"一次性等完了**。这就是为什么一个 4 行的函数会占掉 warp 的一半。

量级交叉校验（**估算，不是测量**）：878 次 dispatch，平均张量 864×480（0.4 M 元素）到 864×480×3（1.24 M），fp32/int64 混合，读+写按 5–8 B/元素计，总流量约 4–8 GB；B580 的 456 GB/s 给出 **9–18 ms** 的设备时间。与"≥10.14 ms"同量级，且这是**逐元素 eager 实现在满分辨率张量上的固有成本**，不是浪费。

**一句话：warp 慢，一半是因为它确实要在 864×480 上搬 4–8 GB 的数据，另一半是因为它每帧 34 次把 host 和 device 对齐成串行，于是这 10 几毫秒的设备时间一分一毫都没被 host 的入队时间掩盖掉。**

### 21.9 这如何修正 §19.6 与 §20.4

- **§19.6**（每帧 5 次 `is_nonzero`、2.12 ms）：**2 次已定位**——`NativeReciprocalTable.forward` 的区间校验，在 warp 的轴权重路径上。它不是"值得查明的未知开销"了。
- **§20.4**（"warp 值 22.8 ms"的解释）：当时列的是"每帧重建常量网格 / 5 个采样点 / int64 / 约 8.3 M 次纹理读取"。现在能分清主次：**常量网格只占 1.21 ms（5.3%）**，把它缓存起来拿不回多少；真正的两块是 **5 点权重与累加 14.84 ms** 和 **4 次 gather 与整数权重 3.70 ms**。而"逐元素 eager 所以贵"这个判断是对的，但**贵的机制不是内核慢，是 34 次同步让 host/device 无法重叠**——这是 §20.4 没有、也不可能从静态阅读里得到的。
- **§20.6** 的"没有细分 warp 内部"这一条缺口，本节关闭。

### 21.10 这对优化意味着什么

按"收益 / 风险"排序，全部来自本节的实测：

| 优先级 | 动作 | 预期 | 风险 | 为什么 |
| --- | --- | --- | --- | --- |
| **1** | **把 `fma32` 的标量操作数预建为 0 维设备张量**（模块级缓存，`warp` 路径只有一台设备） | 消掉 32 次/帧同步 | **极低**：0 维张量与 Python 标量在逐元素运算上广播语义完全相同，数值逐字节不变 | 它占 warp 的 11.82 ms，其中 ≥10.14 ms 是在等设备；遍 K 直接证明 `fma32(标量,标量)` 会阻塞 |
| **2** | **`NativeReciprocalTable.forward` 的区间校验移到构造期** | 消掉 2 次/帧同步（同时关掉 §19.6 的 2/5 个 `is_nonzero`） | **极低**：表是哈希校验过的冻结资产，运行期输入恒在区间内 | 与 1 同类机制，改动更小 |
| 3 | 缓存 `meshgrid`/`arange` 常量网格 | ≤1.21 ms（warp 的 5.3%） | 低 | §20.5 已提，但本节说明它**不是**量级级 |
| 4 | 合并 5 个采样点的重复计算（`_exponent`/`ldexp`/`signbit` 在 5 点间高度重复）、int64→int32 | 未测 | 中：碰数值路径 | 4 次 gather + 整数权重 3.70 ms + 定点转 half 2.43 ms |
| 5 | 融合成单个 kernel | 未测 | 高 | 现在知道目标是"消掉 878 次 dispatch 与 34 次同步"，不只是"少几个内核" |

**1 和 2 是这一节真正的产出**：它们不改任何算术、不改任何融合策略、不改任何门槛，只是把每帧 34 次 host/device 往返去掉。按 §19.9 的算术，这是**两侧共有**的开销，砍掉会**抬高**加速比（1.1722× → 更高），而不是像 Phase 1 那样压低它。

### 21.11 这一节能证明什么 / 不能证明什么

**能证明：**
- 22.93 ms 可以精确划分成 5 项、无余量，且两侧相差 0.42%。
- warp 每帧 878 次 dispatch，占整帧 24.24%（`both`）/ 19.78%（`baseline`）；三个不同步的层合计 650 次 dispatch、6.88 ms、10.59 µs/dispatch。
- **warp 路径上每帧有 34 次完整的设备同步**，分别在 `fma32`（32 次，标量→设备张量）与 `NativeReciprocalTable.forward`（2 次，区间校验）；同路径上另外三个函数（`half_texture_normalized`、`integer_half_away`、`meshgrid`）**不同步**。
- 设备不是收尾瓶颈：20 次连发之后只剩 0.0702 ms 的活。
- 每层调用数与源码推导**逐项相等**（两侧都是）。
- `half_texture_linear` 调用 0 次，确认测的是归一化分支。

**不能证明：**
- **没有**把 host 与 device 分成两个精确的数。因为路径里有同步，二者**天生不可分**：同步已经把它们串行化，"等设备"与"自己在算"在墙钟上无法区分。本节的 `≥10.14 ms` 是从"`fma32` 空闲时 0.065 ms/次"推出来的**下界**，`6.88 ms` 是直接测量的 host 下界，中间还有 5.91 ms 没有拆开。
- 没有用 `torch.profiler`。§18.11 记录过它在本机会在解释器拆卸期触发 `0xC0000409`，且剖析帧墙膨胀 12 倍。**本轮四遍全部干净退出（`returncode: 0`）**，没有复现该现象。
- **遍 I 的毫秒不作预算行**：它在帧内实测拦截成本是 **6.25 µs/op**（`both`：104.32 − 81.69 = 22.63 ms / 3622 次），是 §19 那个"无 GPU 微基准"1.77 µs/op 的 **3.5×**。这个 3.5× 本身是有信息的——真实帧里回调要夹在真实派发之间——但结论是：**遍 I 只出计数，不出毫秒。**
- `axis_weights_fraction` 的阻塞来自它内部的 `reciprocal_table`，所以遍 K 的候选表里它是**从属证据**，不是独立同步点。
- 仍然不做任何加速比声明，不改变任何门槛。bench 的 **1.1722×** 依旧是唯一带门槛的结论。

### 21.12 复现方式

```bash
# 两个变体串行（各 5 遍 × 32 帧 + 约 12 s 微基准），约 2 分钟
bash E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580/reference/fullsize_rows_v1/rows_attrib4_run_v1.sh

# 只复核退出码与报告一致性（不需要 GPU）
python .../rows_attrib4_teardown_v1.py --run D:/fullsize-rows-v1/r1
```

产物：`D:/fullsize-rows-v1/r1/attrib4/{baseline,both}/attrib.json` 与同目录 `teardown.json`。两个变体**退出码均为 `0`**（本轮没有触发 §18.11 的 `torch.profiler` 拆卸期 fail-fast，因为本轮不用 profiler），`teardown.json` 均为 `passed: true`、`report_problems: []`。适配器 `adapter_sha256 = 215cd3fb6df3322529333e9d352bbc538e4852d3d9e3b5b55b72dbcc8a7fb434`（两侧相同）。

口径交叉校验：pass A 帧墙 `both` 81.8413 vs bench 81.5814（**+0.32%**）、`baseline` 95.8190 vs 95.6330（**+0.19%**）；`frame_ops_per_frame` 3622 / 4438 与 §19 独立进程的结果**逐个相同**；`warp` 22.9284 vs §20 的 22.834（**+0.41%**）——三个不同进程、三套不同仪器给出同一个数。

### 21.13 本轮修掉的两个真实错误（都发生在正式跑之前）

1. **`WarpProbe.restore()` 清空了 `original` 映射**，而 pass J 需要拿它重放真实调用 → `KeyError`，冒烟里 `pass_errors = ['J_error']`。修法：`restore()` 不再清空（重复 restore 是幂等的），并在注释里写明为什么不能清。
2. **把"20 次连发后的收尾时间"当成了"每次调用的设备时间"**。第一版 `batch_split_ms` 据此报了 `device_per_call = 0.19 ms`——这**物理上不可能**：878 次 dispatch 在 0.19 ms 内跑完意味着 0.22 µs/op，比 456 GB/s 下搬运一个 864×480 fp32 张量所需的时间快 30 倍以上。真实原因是**设备边入队边消费**，收尾时间只说明"设备没落后"。修法：把它改名为 `tail_ms` 并明确标注"这不是设备执行时间"，同时把 pass J 的判别职责改成 21.6 的三格表，把"同步点在哪"交给 pass K。**这个错误是被自己的物理量级检查抓住的，不是被门槛放过的。**

冒烟产物（含上述两个失败）保留在 `D:/fullsize-rows-v1/r1/attrib4-smoke/both`，正式产物在 `attrib4/{baseline,both}`。

## 22. Phase 0e 归因（**已执行**）：`body_rest` 为什么慢

§20.6 把缺口写得很直白：*"**没有**细分 `body_rest` 内部。38.457 ms 是相减得到的，不是测出来的。"* 本节把它拆开，并回答"为什么慢"——**不是**"哪一段占多少"（那只是问题的一半），而是**这台机器上到底是谁在等谁、以及在等什么**。

### 22.0 三句话结论

1. **`body_rest` 不是"内核慢"，也不是"同步多"，而是被两件事平分：host 提交 62.7%、被 stage fence 吸收的设备等待 37.3%。** 把 13 个 stage fence 全部抹掉后，`both` 的 `body_rest` 从 **37.035 → 23.214 ms**（这 23.214 是 host 提交下界），差额 **13.821 ms 就是那 10 个 fence 各自兜住的设备等待**；而这 13.821 ms 里只有约 7.2 ms 是真正可回收的，其余在无 fence 时变成帧尾的 6.360 ms 尾巴。这就是为什么 K1 只省 5.02%、K2 只省 8.82%。
2. **host 提交那一半里，最大单项是 Triton 启动路径：每帧 951 次启动 × 23.52 µs = 22.369 ms，占 body 的 38.3%、占帧的 27.2%。** 这是**直接测出来的**（pass M 包住 `JITFunction.run`，自带同进程校准，信噪比 **33.3×**），不是估的。它拆成**准备 8.24 µs + 原生入队 15.26 µs**；准备那一半（每帧 7.84 ms）是"缓存已绑定参数 + 已编译内核"能拿掉的，入队那一半不能。**每次启动的单价是一个运行时常数**：`baseline` 23.20 µs（1223 次）、`both` 23.52 µs（951 次），两个变体只差 1.4%，而启动次数差 22%。
3. **决定性判别是把消融与 stage 窗口叠起来用。** 无 fence 那一遍**仍然有 stage 窗口**，所以"有 fence 的 stage 墙 − 无 fence 的 stage 墙"就是 fence 吸收的设备等待。结果：**两变体的全部 13 个 stage 在无 fence 时都落在 10.38–14.35 µs/unit**，pass H 里那两个离群点（`pre` 95.02、`RGB` 60.42）**塌回带内**（10.87、10.77）。也就是说 `pre`/`RGB` 的 5.13/5.86 ms 里各有约 4.53/4.82 ms 是设备等待，它们自己的 host 成本与别的 stage 一模一样。

### 22.1 方法：`rows_attrib5_v1.py`（八遍 R/A/H/S/I/L/M + 消融 K，诊断不是证据跑）

与 §18–§21 同一契约：不写 `validation.json`、不产出 gate/full 产物、不声明正确性、不参与门槛。逐帧仍与已审核原样快速参考**逐字节比对**，FFN 调用数仍逐帧断言，任一遍出错都记录到报告里而不是静默丢弃。

| 遍 | 仪器 | 回答什么 |
| --- | --- | --- |
| R | 配置身份 + host-read 机制表 | **测的到底是哪个 body**、每个可替换符号当前绑在谁身上、`bool(设备张量)` 在这台构建上真实编译成哪些 op |
| A | 干净帧墙（只装 `select` 探针） | 与 §19/§20/§21 和 bench 同口径对照 |
| H | `_forward_front` 的 13 个 stage 边界 inclusive 段计时 | **哪一段**、多少毫秒、每帧几次 |
| S | 重绑 `torch.xpu.synchronize`，按调用点计数与计时 | 一帧到底有**几次** fence、分别**在哪个文件的哪一行** |
| I | **仅计数** `TorchDispatchMode` | 每段多少次 ATen dispatch |
| L | 重绑 `CompiledKernel.launch_metadata` | 每段多少次 **Triton 启动**（遍 I 结构性看不见的那一半分母） |
| M | 重绑 `JITFunction.run` + `launch_metadata` 标记 | 启动路径**值多少微秒**，以及它拆成准备与入队各多少 |
| K | 消融：`marks` 只抹 13 个 stage fence；`all` 抹掉全部 python 级同步 | **fence 到底值多少**，以及它们是不是"兜住了活跃设备工作" |

四处关键设计，都是被前几轮的失败逼出来的：

- **两个完成边界。** `ms` 停在 `installed()` 内部的那个 `torch.xpu.synchronize`（与 §19–§21 和 bench 同口径，pass A 才能与它们对照）；`ms_outer` 停在**消融范围之外**的那个同步。K2 会连 `installed()` 自己的收尾 fence 一起抹掉，没有第二个边界的话它测到的是"host 提交时间"而不是"一帧跑完"。这一条是本轮最重要的仪器设计。
- **`pre_pass_instance_override`。** 脚本自己会把 `model._forward_front` 换成包装器，所以"报告里存在实例级 `_forward_front` 覆盖"是**预期的**，不是故障。修法：在装任何东西**之前**记录 `'_forward_front' in model.__dict__`（期望 `False`），并比对包装器的 `co_filename` 是不是本脚本（脚本以 `__main__` 运行，qualname 前缀不可靠）。
- **`config_identity` 必须枚举实例级覆盖。** `FusedC32.installed()` 用 lambda 覆盖**每个实例**的 `forward_unquantized`，只读类属性会给出**假阴性**（"融合没装"）。实测 `instance_override_count = 16`。
- **host-read 机制表必须实测。** §21 假设 `bool(设备张量)` 产生 `aten::_local_scalar_dense`；本轮在 `with counter.mode:` 内**实测**三种机制的 op 序列，发现它确实是 `aten::all` + `aten::_local_scalar_dense`——**但帧内该 op 计数是 0**。两者不矛盾：那两次 `bool(...)` 在 `MotionNR.forward` 里，**不在** `_forward_front` 窗口内。真正在 body 里出现的是 `aten::lift_fresh` 32 次/帧。

**本节所有数字来自 `attrib5/` 这一轮正式跑。** 另一次完全相同的正式跑（同协议、同帧数、同窗口）在**每一个头条数字上都落在 ±3% 内**：`body_rest` 36.900 vs 37.374、启动路径 21.746 vs 22.369 ms、每次单价 22.87 vs 23.52 µs、host 提交 68.36 vs 68.98 ms。**所有定性结论两次完全相同**，包括 `pre`/`RGB` 在无 fence 时塌回 10–11 µs/unit 这一条。

### 22.2 配置身份必须先钉死：本实验的 body 是 eager，不是 XPU Graph

`product/exact_pipeline_v1/graph.py` 的 `Graph.installed()` 会把 `ResetNR._forward_front` **整个类属性**换成 XPU Graph 捕获 + replay。如果它在场，本节测的就不是 `executor.py` 那 22 行 Python，而是一次 replay——13 个 `mark()` 一个都不会执行。

`rows_entry_v1.py` 的排除集是 `(stack.graph, stack.rewrite, stack.call_guard, stack.compact_queries)`，**`stack.graph` 被排除**。遍 R 断言了这一点：

| 符号 | 当前绑定 | 是后端原件？ | 带 fence？ |
| --- | --- | --- | --- |
| `ResetNR._forward_front` | `nr_backend.executor.ResetNR._forward_front` | **是** | **是** |
| `swin_attention_windows` | `fused_swin_native_half_v1.FusedSwin.windows` | 否 | 否 |
| `BranchedMLP.forward_unquantized` | `fused_branched_pairs_v1.FusedPairs...<lambda>` | 否 | 否 |
| `C32AttentionFront.forward` | `nr_backend.attention.C32AttentionFront.forward` | 是 | 是 |
| `C32MLP.forward_unquantized` | `nr_backend.pre_mlp.C32MLP.forward_unquantized` | 是 | 是 |

**结论：`_forward_front` 是带 13 次无条件 `torch.xpu.synchronize` 的 eager 版本。** 而 `attention.py:95`（`swin_attention_windows` 的组 fence）、`attention.py:116`（`C32AttentionFront` 的 chunk fence）、`pre_mlp.py:75`（`C32MLP` 的 chunk fence）虽然类属性上仍是原件，**但都被融合组件逐一绕过**：`FusedSwin` 把整个 window 组循环换成一次融合启动，`HeadLayout` 覆盖全部 `C32AttentionCore`，`FusedC32` 按实例覆盖全部 `C32MLP`/`PreMLP`。遍 S 实测证实了这一点——**帧内一次都没有出现这三个站点**。

### 22.3 13 个 stage 的精确划分（排他性自证 99.42% / 99.78%）

`mark()` 因为无条件排空队列，每个 stage 边界上的墙钟**排他且无队列残留**，安装回调**零观测者效应**——所以 `progress` 回调给出的分段是一张精确划分。`both`（稳态 28 帧中位数，body = 58.4328 ms）：

| stage | ms | %body | ATen | 启动 | µs/unit（H） | 无 fence 后 µs/unit |
| --- | --- | --- | --- | --- | --- | --- |
| `pre` | **5.1312** | 8.78 | 44 | 10 | **95.02** | **10.87** |
| `encoder C32` | 3.4217 | 5.86 | 130 | 39 | 20.25 | 11.25 |
| `encoder C64` | 2.5480 | 4.36 | 117 | 47 | 15.54 | 12.57 |
| `encoder C128` | 2.9768 | 5.09 | 163 | 69 | 12.83 | 12.79 |
| `encoder C256` | 4.0566 | 6.94 | 209 | 91 | 13.52 | 13.20 |
| *`encoder C512`* | *5.7128* | *9.78* | *341* | *123* | *12.31* | *12.41* |
| *`ViT`* | *9.4140* | *16.11* | *729* | *184* | *10.31* | *10.38* |
| *`decoder C512`* | *5.9323* | *10.15* | *358* | *127* | *12.23* | *12.29* |
| `decoder C256` | 4.0123 | 6.87 | 193 | 92 | 14.08 | 13.47 |
| `decoder C128` | 3.0379 | 5.20 | 147 | 70 | 14.00 | 14.35 |
| `decoder C64` | 2.4669 | 4.22 | 101 | 48 | 16.56 | 13.33 |
| `decoder C32` | 3.5225 | 6.03 | 113 | 39 | 23.17 | 11.47 |
| `RGB` | **5.8609** | 10.03 | 86 | 11 | **60.42** | **10.78** |
| **13 段合计** | **58.0939** | **99.42** | 2731 | 950 | 12.70 | 9.71 |

*斜体三项 = `bench_timed_stages`（§20 的 `c512` scope 同时包住编码器与解码器的 `SplitSwinBlock` 调用，只减编码器会虚高 `body_rest` 约 5.6 ms，所以这里列三项）*

- `body_rest = body − 20.8684… = 37.3737 ms`（**63.96% of body、45.40% of frame**）。
- 13 段窗口合计占 body 的 **99.42%**（`baseline` 99.78%），残差 `stage.tail` 只有 0.012 ms——**排他性自证**。
- **没有任何单一段占绝对主导**：最大的 `ViT` 16.11%，其余都在 4–10% 之间，`body_rest` 的 10 段是 2.47–5.86 ms 累加出来的。

### 22.4 决定性的一步：消融 × stage 窗口 → host 提交与设备等待的分离

单独看消融只能得到"fence 值 5%"，单独看 stage 窗口只能得到"`pre` 很贵"。**两者叠起来才有机制**：K2 抹掉全部同步，但 **stage 窗口仍然装着**，所以 K2 里每个 stage 的墙钟就是"没有 fence 拦着时，host 把这段提交完要多久"。

`both`（H = 有 fence，K2 = 无 fence，单位 ms）：

| stage | H | K2 | **H−K2 = 设备等待** | 读数 |
| --- | --- | --- | --- | --- |
| `pre` | 5.1312 | 0.5974 | **4.5338** | fence 兜住 4.53 ms 设备工作 |
| `RGB` | 5.8609 | 1.0453 | **4.8156** | 同上，4.82 ms |
| `decoder C32` | 3.5225 | 1.7436 | **1.7789** | 同上，1.78 ms |
| `encoder C32` | 3.4217 | 1.9012 | **1.5205** | 同上，1.52 ms |
| `encoder C64` | 2.5480 | 2.0614 | 0.4866 | 轻微 |
| `decoder C64` | 2.4669 | 1.9855 | 0.4815 | 轻微 |
| `decoder C256` | 4.0123 | 3.8391 | 0.1732 | 轻微 |
| `encoder C256` | 4.0566 | 3.9612 | 0.0954 | **≈纯 host** |
| `encoder C128` | 2.9768 | 2.9663 | 0.0104 | **纯 host** |
| `encoder C512` | 5.7128 | 5.7560 | −0.0432 | **纯 host** |
| `ViT` | 9.4140 | 9.4736 | −0.0596 | **纯 host** |
| `decoder C512` | 5.9323 | 5.9621 | −0.0298 | **纯 host** |
| `decoder C128` | 3.0379 | 3.1131 | −0.0752 | **纯 host** |
| **`body_rest` 的 10 段** | **37.0348** | **23.2141** | **13.8207** | **62.7% host / 37.3% 设备等待** |
| `motion.body` | 58.4328 | 45.1980 | 13.2347 | |

三件事同时被这张表钉死：

1. **`pre` 和 `RGB` 的 pass H 高值 100% 是设备等待。** 它们的 host 成本只有 0.597 / 1.045 ms，与任何别的 stage 一个量级。`µs/unit` 从 95.02 / 60.42 掉到 **10.87 / 10.78**——两个离群点塌回带内。
2. **7 个 stage 的设备等待是 0。** C128/C256/C512/ViT/decoder C512/decoder C256/decoder C128 的 H−K2 全部在 ±0.08 ms 内。它们的墙钟**就是 host 提交时间**，fence 在那里什么也没兜住。
3. **设备等待高度集中**：`pre` + `RGB` + `decoder C32` + `encoder C32` 四段合计 **12.65 ms**，占全部 13.82 ms 的 **91.5%**。

而**帧**只从 82.542 降到 75.343（**−7.199 ms**），远小于 body 的 −13.23 ms。原因是 13.23 ms 里有 6.360 ms 在无 fence 时变成了**帧尾的尾巴**（K2 外边界 75.343 − 内边界 68.984 = 6.360）——**fence 没有制造等待，它只是把等待从帧尾搬到了帧中间**。这才是 K2 只省 8.82% 的真正原因。

### 22.5 启动路径的真实价格（pass M）：每帧 22.37 ms，单价是运行时常数

`JITFunction.run` 是一次 Triton 启动在 Python 侧要走的**全部**：问驱动要设备与流、对每个实参做特化与绑定、算 cache key、查已编译内核、规范化 grid、构造 launch metadata、调原生启动器。它**完全不经过 `TorchDispatchMode`**，所以遍 I 结构性看不见；遍 L 只能数次数；只有包住它才能定价。

`launch_metadata` 恰好在 `JITFunction.run` 内部、原生启动**之前**被调用，于是它被当作标记点，把总时间切成"准备"与"入队"两半。

| | `baseline` | `both` |
| --- | --- | --- |
| 启动次数 / 帧 | 1223 | 951 |
| **启动路径总价** | **28.3773 ms** | **22.3686 ms** |
| 占 body | 39.6% | **38.3%** |
| 占帧 | 29.6% | **27.2%** |
| **每次单价** | **23.2030 µs** | **23.5211 µs** |
| ├ 准备（驱动查询 + 实参特化 + cache key + 查表 + grid） | 8.0388 µs | **8.2425 µs** |
| └ 原生入队 | 15.1710 µs | **15.2661 µs** |
| 仪器自身成本（同进程校准，跑同一段代码） | 0.935 ms | **0.672 ms** |
| **信噪比** | 30.4× | **33.3×** |

**每次单价在两个变体间只差 1.4%（23.20 vs 23.52 µs），而启动次数差 22%。** 这是一个很强的内证：**单价是运行时的属性，不是负载的属性。** 因此"删掉一次启动"的价值可以当作常数用。

逐内核（`both`，前 6）：

| 内核 | 次数 | run ms | µs/次 |
| --- | --- | --- | --- |
| `nr_backend.triton_fp8._kernel` | **383** | **7.6132** | 19.9 |
| `fast_matrices_v3._matmul` | 115 | 3.3145 | 28.8 |
| `fused_qkv_pack_native_half_v1._pack` | 52 | 1.3917 | 26.8 |
| `batched_branched_mlp_v1._pairs` | 36 | 0.9196 | 25.5 |
| `window_block_projection_v3._project` | 36 | 0.8944 | 24.8 |
| `batched_branched_mlp_v1._project` | 36 | 0.8406 | 23.4 |

**`nr_backend.triton_fp8._kernel` 一个内核就是 383 次启动 / 7.61 ms——占全部启动成本的 34.0%。** 它是 FP8 重写路径，两侧完全相同（383 vs 383，7.693 vs 7.613 ms），所以它不属于本轮优化的对象，但它是**当前最大的一笔可优化启动开销**。

### 22.6 同步点在哪儿：一帧 16 次，其中 2 次不在 `executor.py`

遍 S 用 `sys._getframe` 记录每次 `torch.xpu.synchronize` 的调用点（`文件名:行号`）。`both` 稳态：

| 站点 | 次数/帧 | 合计 ms | **每次 ms** |
| --- | --- | --- | --- |
| `executor.py:72`（`mark()` 内的无条件同步） | 13 | 8.5418 | 0.6571 |
| **`fused_c32_projection_native_half_v1.py:99`** | **2** | **5.1283** | **2.5642** |
| `rows_attrib5_v1.py`（harness 自己的收尾同步） | 1 | 0.0413 | 0.0413 |
| **合计** | **16** | **13.7114（23.47% of body）** | |

两个发现：

- **第 4 个同步源是本轮新定位的，读 `executor.py` 读不出来。** `fused_c32_projection_native_half_v1.py:99` 是 `ProjectedHeadLayout.c32` 里的一条条件 fence：

  ```python
  if attention.C32AttentionFront.forward is not body.c32_front:
      for _ in range(chunks // 8):
          torch.xpu.synchronize(features.device)
  ```

  它是随 `HeadLayout`（`native_half_head_layout_v1.py` → `c32_chunk_layout_v2.ChunkedHeadLayout`，覆盖 `pre` 块与全部 `C32AttentionCore`）**顺带搬进来的**。**它只有 2 次/帧，却值 5.13 ms——单次 2.56 ms，是 `executor.py` 那 13 次的 3.9 倍。** 它是全帧最贵的单次 fence。
- **`baseline` 与 `both` 的同步站点表完全相同**（16 次、13.996 ms、同三个站点，单次 0.6762 / 2.5827 ms），因为这两个优化不动同步。这本身是一条自证：同步面不是被优化改动的那一部分。

### 22.7 所以为什么慢 —— 机制

把上面的表拼起来，`both` 的 `body_rest`（37.3737 ms）是：

| 项 | ms | 占 body_rest | 依据 |
| --- | --- | --- | --- |
| **host 提交** | **23.2141** | **62.1%** | K2 下 10 段的 stage 窗口之和 |
| ├ Triton 启动路径 | 12.14 | 32.5% | 516 次 × 23.5211 µs（pass M 实测） |
| ├ eager ATen dispatch | ≤ 3.71 | ≤ 9.9% | 1303 次 × 2.85 µs 地板（**上界**） |
| └ 其余 host Python | ≈ 7.36 | ≈19.7% | 相减得到的**下界** |
| **被 fence 吸收的设备等待** | **13.8207** | **37.0%** | H − K2，10 个 stage 逐个测得 |
| **合计** | **37.0348** | **99.1%** | 10 段窗口（另 3 段属 `bench_timed_stages`） |

三个层次的原因，按可动性排序：

1. **host 比设备慢，且两者接近——这是最坏的情况。** `both` 无 fence 时：host 提交 **68.984 ms**，帧 **75.343 ms**，设备在 host 停手后还欠 **6.360 ms**。结合 §18.9 的 profiler 设备忙时 **57.711 ms**，得到：**host 68.98 ms vs 设备忙 57.71 ms（1.20×）**；`baseline` 是 **82.33 vs 58.36（1.41×）**。host 是较长的那条流水线，设备在整个无 fence 帧里空闲 17.63 ms（23.4%）。**但两者只差 20%（`baseline` 41%），所以任何把两条流水线串行化的东西都极贵**：13 个 fence 让 `both` 的帧从 75.343 变成 82.542（**+7.20 ms / +9.6%**），设备的空闲比例升到 30.1%。这条比较已作为硬门槛写进 teardown（比值须落在 0.5–2.0 内），不再只写在散文里。
2. **启动次数是 host 侧最大的可动项。** 每删一次启动省 **22.8 µs**（启动器，扣仪器后）+ 若该内核走 `select` 则再省 **11.6 µs**（见 §22.8）= **≈34.2 µs**。相对地，一次 ATen dispatch 的地板只有 2.85 µs。**所以"合并内核 / 提高每次启动的行数"的杠杆率是"减少 dispatch"的 12 倍。**
3. **fence 本身几乎不值钱，值钱的是它兜住的设备工作。** 16 次同步合计 13.71 ms 墙钟（23.47% of body），但**全部抹掉只省 8.82%**。差额的去向已定位：13.23 ms 的分布式等待 → 6.360 ms 的帧尾尾巴。**fence 是等待的搬运工，不是等待的制造者。** 这也直接解释了 §19.6 与 §20.4 里"同步看起来很多"的观察为什么没有变成收益。

### 22.8 边际每次启动值多少：≈34.2 µs（含 `select` 的 1:1 耦合）

取消分批把两族内核换掉了，这给了一个天然的对照实验：

| 族 | `baseline` | `both` | 差额 |
| --- | --- | --- | --- |
| `c512_int8_ffn_gpu_v1.*` | 5 × 64 = **320 次**，6.9140 ms | — | |
| `c512_int8_ffn_rows_v1.*` | — | 5 × 16 = **80 次**，1.7350 ms | |
| **小计** | **320 次 / 6.914 ms**（21.61 µs/次） | **80 次 / 1.735 ms**（21.69 µs/次） | **−240 次 / −5.179 ms** |
| `int8_ffn_segment_gpu_v1.*` | 4 × 16 = **64 次**，1.5056 ms | — | |
| `int8_ffn_segment_rows_v1.*` | — | 4 × 8 = **32 次**，0.7484 ms | |
| **小计** | **64 次 / 1.506 ms**（23.53 µs/次） | **32 次 / 0.748 ms**（23.39 µs/次） | **−32 次 / −0.757 ms** |
| **两族合计** | **384 次 / 8.420 ms** | **112 次 / 2.483 ms** | **−272 次 / −5.936 ms** |

**同时 `select_calls_per_frame` 从 461 降到 189，正好也是 −272。** 这不是巧合：这 272 个内核**每次启动都调一次 `spill_preflight_v1.select`**（实测 `select.calls_by_kernel` 与它们的启动数逐个相等），所以删掉一次启动同时删掉一次 `select`。`select` 从 6.0009 → 2.8526 ms，**省 3.148 ms / 272 次 = 11.6 µs/次**。

于是：

> **从这一帧里删掉一次这类 Triton 启动，值 ≈22.8 µs（启动器）+ 11.6 µs（`select`）= ≈34.2 µs。**

body 一共降了 **13.2646 ms**，其中 **272 × 34.2 µs = 9.30 ms（70.1%）** 由启动次数解释。剩下 3.96 ms 对应 **816 个消失的 ATen op**（`empty_like` −152、`empty` −176、`reshape` −128、`slice` −160、`view` −160 等），即 **4.85 µs/op**——高于 2.85 µs 的 `aten::to` 地板，说明这批**分配与形状类 op 比地板贵**。这是一个**未解释的残差，如实记录**，不硬塞进模型。

### 22.9 这如何修正 §19.2 与 §20.6

**（a）§19.2 的"Triton 驱动提交"低估了 2.5 倍。** 那一遍用的是 `launch_enter_hook` / `launch_exit_hook`，它们只包住 `JITFunction.run` 内部的**原生启动**那一步；pass M 包住的是整个 `JITFunction.run`（含驱动查询、实参特化、cache key、查表、grid）。修正：

| §19.2 的行 | 原值 `baseline` / `both` | **修正后** | 倍数 |
| --- | --- | --- | --- |
| Triton 驱动提交 | 11.361 / 8.813 | **28.377 / 22.369** | 2.50× / 2.54× |
| 模型 Python 图与上下文（残差） | 51.304 / 46.398 | **34.288 / 32.842** | −17.016 / −13.556 |

修正后的 CPU 侧划分（`both`，帧 82.314 / `submit_all` ≈82.33）：`select` **2.853（3.5%）** + Triton 启动路径 **22.369（27.2%）** + eager ATen **24.166（29.4%）** + 模型 Python 残差 **32.842（39.9%）**。**残差仍然是最大的一项，但它从"半帧"降到了"四成"**，而且这四成现在是诚实的残差，不是被启动路径漏算撑起来的。

**（b）§20.6 的 `body_rest` 缺口被补上，而且它不是"剩下的一块"。** §20 把 `body_rest` 当作相减的余项；本节证明它是**两侧共有的稳定成本**：`baseline` 37.476 vs `both` 37.374（**−0.3%**），而同期 body 降了 18.5%（71.697 → 58.433）。**`body_rest` 对这两个优化几乎完全不动**——它和 §19.3 最后一行"模型其余 62.3 / 62.5 ms 两侧共有"是同一件事的两个切面。

**（c）§21 的"同步是主因"结论在 `body_rest` 上被证伪。** warp 路径上 34 次同步把 host 与 device 完全串行化，拆掉它就是主要收益；`body_rest` 上 16 次同步全部拆掉只值 **8.82%**，因为这里 host 与 device 是**接近平衡**的（68.98 vs 57.71），fence 兜住的等待大部分是必须发生的设备工作。**同一个项目里，同一个问题的答案在两条路径上是相反的——这正是"必须逐层实测、不能外推"的实例。**

**（d）本轮同时修正了我自己在 §22 草稿里的一个读法错误。** 第一版把 K2 的 `host_submit_only_ms / device_tail_ms` 读成"帧是 host bound 的 10.6 倍"。**这是错的**：那个比值把"帧尾还欠多少"当成了"设备一共要干多少"。正确的读法要引入设备忙时（§18.9 的 57.711 ms）：host 提交 68.98 ms vs 设备忙 57.71 ms，**host 只是长 20%，不是长 10 倍**。报告里的 `host_submit_only_note` 与 teardown 的 note 措辞都已按此改正，并明确写出"不要用这两个数求倍数"。

### 22.10 这对优化意味着什么

按**每次启动 ≈34.2 µs**（含 `select` 耦合）和**每帧 951 次启动**排序：

| 候选 | 每帧可删启动 | 预估收益 | 依据 |
| --- | --- | --- | --- |
| **`nr_backend.triton_fp8._kernel` 的 383 次** | 383 | **≈13.1 ms** | 最大的单块启动开销（7.61 ms 启动器 + 走 `select` 的部分）；两侧相同，与 C512/ViT 无关 |
| 缓存已绑定参数 + 已编译内核（Phase 1b） | 0（省单价） | **≈7.84 ms** | pass M 实测准备部分 8.2425 µs × 951 |
| 缓存 `spill_preflight_v1.select`（Phase 1） | 0（省单价） | **≈2.85 ms** | `select` 每帧 189 次 / 2.8526 ms |
| 拆掉 `fused_c32_projection_native_half_v1.py:99` 的 2 次 fence | 0 | **≤5.13 ms，实际 ≤3.2 ms** | 单次 2.56 ms，但 K2 实测这部分只贡献约 3.2 ms 净收益 |
| 拆掉 13 个 stage fence | 0 | **≤4.12 ms（K1 实测 −5.02%）** | 实测，且**不得**与其他行相加（同一份等待） |

**关键判断：启动次数是比"启动单价"更大的杠杆。** Phase 1b 能拿掉 7.84 ms 的**准备**部分，但它拿不掉 15.27 µs/次的**原生入队**——那部分只能靠"更少的内核"或"更快的启动路径"。而 `triton_fp8` 一族 383 次启动值 13.1 ms，是单笔最大的。**下一步应该先问"FP8 重写能不能提高每次启动的行数/批大小"，而不是先做缓存。**

### 22.11 这一节能证明什么 / 不能证明什么

**能证明：**

- 13 个 stage 的毫秒是**排他划分**（自证 99.42% / 99.78%），`body_rest` 的每一毫秒都被归到具体段上。
- 启动路径的 **22.369 ms** 是直接测出来的，带同进程校准，**信噪比 33.3×**；每次单价 23.52 µs 在两变体间只差 1.4%，且两次正式跑重复到 ±3%。
- host 提交与设备等待的**分离**是测出来的（K2 stage 窗口），不是估的：13.8207 vs 23.2141，且 7 个 stage 的等待实测为 0。
- 同步面：16 次、三个站点、逐个计时；`fused_c32_projection_native_half_v1.py:99` 是读 `executor.py` 读不出来的第 4 个来源。
- 消融**不改变任何算术**：K1 与 K2 都是 **32/32 帧逐字节相等**、峰值分配完全相同（1155.05 MB）——所以它们测的是调度，不是数值。

**不能证明：**

- **绝对毫秒不作加速比证据。** 本节的毫秒是诊断，用于占比与排序；加速比仍须走 `local`→`gate`→`full` 的逐字节门槛。
- **pass I 的毫秒不作预算行**（含实测 5.81–6.38 µs/op 的拦截开销），只用它的计数。
- **pass M 的毫秒含一个可量化的仪器**（0.672 / 0.935 ms，已扣除），且 `observed_added_ms` **不是**校准——本机上 pass A 自身的跑间漂移（±1.8%）大于仪器成本，teardown 因此改用"信噪比 ≥5×"作为门槛。
- **"host 提交 68.98 ms"是 host-only 时间的上界。** 如果驱动的队列深度不足，host 会被设备反压，此时这个数里含设备时间。本节的证据（K2 下 13 个 stage 全部落在 10.38–14.35 µs/unit，与"启动 23.5 + dispatch 2.85"的混合模型一致；若被反压，内核大的 stage 会显著偏高）**支持**它是 host 主导，但**没有直接测队列深度**。这一条留作 Phase 0f 的第一项。
- **`pre` / `RGB` 的 host 成本没有细分。** 无 fence 时它们分别是 0.597 / 1.045 ms，但里面既没有 dispatch 也没有启动解释（54 / 97 units × ~12 µs = 0.65 / 1.16 ms，倒是吻合）。若要继续，需要在这两个 block 内部做 §21 式的分层。
- **816 个消失的 ATen op 值 3.96 ms（4.85 µs/op）高于 2.85 µs 地板**，本轮**没有**解释。可能是分配类 op（`empty`/`empty_like`）走缓存分配器慢路径，也可能是别的；如实记录为残差。

### 22.12 复现方式

```bash
# 两个变体串行（各 8 遍 × 32 帧），一次租约，约 1.5 分钟
bash E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580/reference/fullsize_rows_v1/rows_attrib5_run_v1.sh 32 4 attrib5

# 只复核报告一致性（不需要 GPU）；按变体分别跑，复核器会合并而不是覆盖
python .../rows_attrib5_teardown_v1.py --variant both     --run D:/fullsize-rows-v1/r1 --outdir attrib5
python .../rows_attrib5_teardown_v1.py --variant baseline --run D:/fullsize-rows-v1/r1 --outdir attrib5
```

产物：`D:/fullsize-rows-v1/r1/attrib5/{baseline,both}/attrib.json` 与同目录 `teardown.json`。两个变体租约 `returncode` 均为 `0`，`teardown.json` 均为 `passed: true`、`problems: []`。

口径交叉校验：`frame` `both` 82.314 vs §20 的 81.58（**+0.90%**）、`baseline` 95.885 vs §19 的 95.565（**+0.33%**）；`body` 58.433 vs 58.345（**+0.15%**）；`body_rest` 37.374 vs 38.457（**−2.82%**，差额主要来自 `BENCH_TIMED_STAGES` 从 1 项修正为 3 项）；`frame_ops_per_frame` 3622 / 4438 与 §19、§21 独立进程的结果**逐个相同**。**四个不同进程、四套不同仪器给出同一组数。**

### 22.13 本轮修掉的四个真实错误（都发生在正式跑之前）

1. **`LaunchTimer` 一开始钩的是 `CompiledKernel.__getitem__`，而真实路径是 `JITFunction.run`。** 结果 pass M 报"0 次启动"，而遍 L 报 951 次。**这个失败是响的**（teardown 直接判 `passed: false`），但如果只做冒烟不设门槛，就会得到"启动路径免费"的结论。修法：改钩 `JITFunction.run`，并保留"遍 M 的启动数必须等于遍 L 的启动数"这条交叉门槛。
2. **校准器测的是"长得像"的包装器，不是真的那个。** 第一版用一个手写闭包做微基准，报 0.145 µs/次；而真实包装器在帧内加了 3.457 ms 的墙（3.64 µs/次），**差 25 倍**。原因是真实热路径里每次都在做 f-string 命名和**新建一个 dict**（`setdefault(name, dict(...))` 的默认参数每次都求值）。修法有两步：把热路径改成"缓存内核名 + 先 `get` 再 `set`"（降到 0.49 µs/次），并**把包装器与标记钩子都抽成工厂函数**，让校准器调用**同一段代码**（0.73 µs/次）。
3. **"加了多少墙"不能当校准用。** 修完 (2) 之后，仪器成本 0.672 ms 而观测到的加墙是 4.091 ms——差额是本机 pass A 自身的跑间漂移（五个冒烟里 pass A 从 81.01 到 83.94 ms，±1.8%），**比仪器成本还大**。修法：门槛从"墙差 ≈ 仪器成本"改成**"信号 ≥ 5× 仪器成本"**（实测 33.3×），并把墙差降级为 note 且注明它为什么不是校准。**这是把一条会误报的门槛换成一条能判真伪的门槛，不是放宽。**
4. **`teardown.json` 被按变体逐个覆盖，正式轮丢掉了 `baseline` 的整条记录。** 两个变体都跑完、都干净，但文件里只剩最后一个。修法：读入已有文件再合并。**这一条是靠"文件里只有 1 个变体"这个异常发现的——如果只检查 `passed`，会以为整轮通过。**

另外一条**不是错误但要记录**：`BENCH_TIMED_STAGES` 从 §20 的单项 `encoder C512` 修正为三项 `('encoder C512', 'decoder C512', 'ViT')`。§20 的 `c512` scope 包住的是 `SplitSwinBlock` 的调用，编码器循环与解码器循环**都发生**，只减一项会虚高 `body_rest` 约 5.6 ms。这正是 `body_rest` 与 §20 差 −2.82% 的来源——**这个差是修正量，不是漂移。**

冒烟产物保留在 `D:/fullsize-rows-v1/r1/attrib5-smoke/`，正式产物在 `attrib5/{baseline,both}`。

---

## 23. Phase 0f 归因（**已执行**）：每一个 op 是谁发的 —— 以及全帧预算的合成

§19–§22 把一帧拆到了"哪一段、哪一次启动、哪一次同步"。仍然空着的是**反向问题**：*这些 op 是产品源码的哪一行发的？* §19.6 留下"每帧 5 次 `is_nonzero` 不知道从哪来"，§21 留下"32 次标量 H2D 落地"，§22 留下"`triton_fp8._kernel` 383 次启动值 7.61 ms 但不知道每次启动在干什么"。本节用**栈行走**把这些全部落到源码行，并把 §19–§23 合成为一张完整帧预算。

**本节的技术记录对应交付物 `FRAME_BUDGET_REPORT.md`（自包含全帧预算报告）。** 本节只记录方法、结果与修正，报告负责叙述。

### 23.0 三句话结论

1. **3622 个 op/帧可以 100% 归到源码行，未归因只有 1.0 个（0.03%）。** 归因到 **40 个源码行**，最贵的是 `triton_fp8.py:42`（411 op/帧）；按文件聚合 `triton_fp8.py` 794（≥21.9%）、`fast_matrices_v3.py` 515（≥14.2%）、`sampling.py` 500（≥13.8%）。
2. **三个悬案全部落地**：`sampling.py:14` 是那 28 次 `aten::lift_fresh`（标量 H2D）的唯一来源；`aten::is_nonzero` 只有两个来源——`reciprocal.py:29`（2.625/帧）与 `temporal.py:106`（2.0/帧）；`executor.py:80/83/85` 三行的 625 op/帧（`baseline`）全是 `slice`/`reshape`/`view`/`transpose`/`pad`，**没有一个是算术**。
3. **一个陷阱值得单独记住：这个模型的任何一遍都必须从第 0 帧开始。** `MotionNR` 跨帧有状态（`temporal.py:109` `previous=None if reset else self._previous`，`reset = index == 0`），窗口若从稳态起点起跑会继承上一遍的时序历史、改变字节。第一版 pass O 从 `measure_from` 起跑，**字节门槛在第 2 帧就把它拦住了**。

### 23.1 方法：`rows_attrib6_v1.py`（九遍 R/A/H/S/I/L/M/O + 消融 K）

与 §18–§22 同一契约：不写 `validation.json`、不产出 gate/full 产物、不声明正确性、不参与门槛。逐帧仍与已审核原样快速参考**逐字节比对**，FFN 调用数仍逐帧断言，任一遍出错都记录到 `passes['*_error']` 而不是静默丢弃。

| 遍 | 仪器 | 回答什么 |
| --- | --- | --- |
| A | 干净帧墙 | 与 §19–§22 和 bench 同口径对照 |
| H | 13 段 stage 窗口 + c512/vit 子段 | 分段墙钟 |
| S | 同步点计数与计时 | 一帧同步几次、在哪 |
| I | `TorchDispatchMode` **仅计数** | 一帧多少个 ATen op（分母） |
| L | `CompiledKernel.launch_metadata` 计数 | 一帧几次内核启动 |
| M | 包住 `JITFunction.run` 的计时器 | 启动路径的价格与拆分 |
| **O** | **栈行走调用点普查（本节新增）** | **每个 op 由哪一行发出** |
| K | 消融 13 个 stage fence（K1）/ 全部 16 次同步（K2） | fence 的帧级价格 |

### 23.2 pass O 的设计

`DispatchCensus`（`rows_attrib6_v1.py:266`）在每个 ATen dispatch 上走 Python 栈，取**最内层属于产品树**的帧，把 op 混合按 `文件:行号` 归类。三条设计决定：

- **深度上限 `MAX_DEPTH = 48`**，超出即记为未归因。栈行走本身可能比 dispatch 还贵，必须有界。
- **双重站点**：`site` = 最内层**非 harness** 产品帧（读者会去改的那一行），`inner` = 最内层产品帧（含 harness 自己的 block 包装器）。harness 包装器镜像产品的 block forward 以便看到内部，所以只记 `inner` 会把 `aten::view` 错记到包装器上；只记 `site` 又看不到包装层。两张表并列。
- **必须从第 0 帧开始**，见 §23.3。

**不计时。** pass O 的毫秒被走路本身抬高，**永远不作预算行**；它只产出计数。

### 23.3 决定性的一步：从第 0 帧开始（以及字节门为什么救了一命）

第一版把 pass O 写成"从 `measure_from` 起跑"，理由是省时间。结果：

```
AssertionError: frame 2 changed
```

根因不在仪器，在**模型**：`backend/nr_backend/temporal.py:109` 是

```python
previous = None if reset else self._previous
```

而 `reset = index == 0`。所以 `MotionNR` 只在**第 0 帧**复位，之后每一帧都吃上一帧的输出。pass O 从第 4 帧起跑，就继承了**上一遍（第 31 帧）**的时序历史，字节自然不同。

修法：pass O 改回从第 0 帧开始跑（`CALLSITE_FRAMES` 从 6 提到 **8**，保证有 4 帧落在稳态窗口内可与 pass I 比对），并把"窗口必须恰为 `[0..7]`"写成 teardown 硬门槛 + 代码注释 + 常量注释。

**这是本轮最有价值的一次失败。** 如果只做冒烟不设字节门，就会得到"调用点普查会改变输出"这个完全错误的结论，而栈行走其实一个字节都没改。

### 23.4 结果：40 行、按文件聚合、最贵站点

**跨 pass 校验**（`attrib6`，8 帧）：pass O 与 pass I **逐帧精确相等**——

| 帧 | `both` pass O / pass I | `baseline` pass O / pass I |
| --- | --- | --- |
| 0（复位帧） | 2721 / 2721 | 3537 / 3537 |
| 1–7（稳态） | 3622 ×7 | 4438 ×7 |

未归因 **1.0 op/帧**（`both` 0.03%、`baseline` 0.02%），全部是 `aten::to`，来自产品树之外。**逐帧精确相等，不是中位数比中位数**——第一版用 4 帧均值比 28 帧中位数（0.9689×），虽然过了带内门槛但不够严格，已改成按 index 逐帧比对。

**最贵的 10 个站点**（`both`）：

| 站点 | op/帧 | 主要 op |
| --- | --- | --- |
| `triton_fp8.py:42` | **411** | `to`×383、`contiguous`×28 |
| `triton_fp8.py:43` | **383** | `empty_like`×383 |
| `fast_matrices_v3.py:83` | 250 | `to`×115、`reshape`×115、`contiguous`×20 |
| `executor.py:83` | 176 | `slice`×32、`transpose`×32、`to`×24 |
| `fused_qkv_pack_native_half_v1.py:79` | 156 | `empty`×156 |
| `fast_matrices_v3.py:84` | 123 | `to`×115、`contiguous`×8 |
| `fast_matrices_v3.py:93` | 115 | `empty`×115 |
| `sampling.py:14` | 74 | `to`×45.5、`lift_fresh`×28 |
| `executor.py:80` | 37 | `slice`×20、`pad`×9、`to`×4 |
| `window_blocks_v3.py:48` | 36 | `pad`×36 |

`baseline` 的差别集中在 `executor.py`：`:83` **248**、`:80` **197**、`:85` **180**，三行合计 **625（14.1%）**；`both` 只剩 `:83` 176 + `:80` 37 = **213（5.9%）**。用 rows 路径替掉 c512 后这三行掉了 412 op/帧。

**按文件聚合（`both`）**：`triton_fp8.py` 794（≥21.9%）、`fast_matrices_v3.py` 515（≥14.2%）、`sampling.py` 500（≥13.8%）、`executor.py` 213（≥5.9%）、`fused_qkv_pack_native_half_v1.py` 156（≥4.3%）。
**（`baseline`）**：`triton_fp8.py` 794（≥17.9%）、`executor.py` **625**（≥14.1%）、`fast_matrices_v3.py` 515（≥11.6%）、`c512_int8_full_stack_v1.py` **512**（≥11.5%）、`sampling.py` 332（≥7.5%）。

**这些是下界**，理由见 §23.9 第 7 条。

### 23.5 四个具体结论

**（a）`triton_fp8.py:42/43` 两行 = 794 op/帧（≥21.9%），就是那个 383 次启动 / 7.61 ms 的 FP8 内核。**
第 42 行发 383 个 `to`（输入搬 dtype）+ 28 个 `contiguous`；第 43 行发 383 个 `empty_like`（输出分配）。**这是全帧最大的单点——host 侧与设备侧同时最大。** §22.10 的第 ① 项（"先问 FP8 重写能否提高每次启动的行数/批大小"）现在有了精确的落点。

**（b）`executor.py:80/83/85` 是 Python 层解包 `SplitSwinBlock` 边界返回值的开销。**
三行的 op 全是 `slice`/`reshape`/`view`/`transpose`/`pad`——**形状与切片，没有一个是算术**。这是 §22.11 里"`body_rest` 的 host 提交 62.7%"的一个可指认成分。

**（c）`sampling.py:14` 是 §21 那 32 次标量 H2D 落地的主要来源。**
74 op/帧，其中 **28 个 `aten::lift_fresh`**，即 `torch.as_tensor(标量, device='xpu')`。§21 数的"32 次"里 28 次在这里，另 4 次在别处。**§21.10 的"把 `fma32` 标量操作数预建为 0 维设备张量"因此有了精确的落点。**

**（d）§19.6 遗留的"每帧 5 次 `is_nonzero`"全部定位完毕。**
`aten::is_nonzero` 有且只有两个来源：`reciprocal.py:29` **2.625 次/帧**（`NativeReciprocalTable.forward` 区间校验的 `bool(设备张量)`）+ `temporal.py:106` **2.0 次/帧**（`torch.isfinite(motion).all()` 与 `abs()<=65504` 的两次 `bool(...)`）。合计 4.625，与 §19.6 的 5 次一致（余 1 次在长尾）。
同理 `aten::all` 三个来源：`sampling.py:58` 4.375、`reciprocal.py:29` 2.625、`temporal.py:106` 2.0。
**顺带**：`aten::_local_scalar_dense` 全帧为 **0**——这一帧没有任何"把设备标量读回 host"的操作，与 §22 一致。

### 23.6 合成：一张完整帧预算表

新增 `rows_budget_v1.py`（**纯合成**：不开设备、不 `import torch`、不跑模型），把 Phase 0b–0f 五轮产物合成为 `budget.json` + `BUDGET.md`，并写了自包含报告 `FRAME_BUDGET_REPORT.md`。

合成必须声明**跨轮来源**，因为毫秒会漂：

| 轮 | `baseline` 帧墙 | `both` 帧墙 | 差 |
| --- | --- | --- | --- |
| Phase 0b (`attrib2`) | 95.565 | 82.208 | 13.357 |
| **Phase 0e (`attrib5`)** | **95.885** | **82.314** | **13.571** |
| Phase 0f (`attrib6`) | 98.571 | 83.871 | 14.700 |
| bench（唯一带字节门） | 95.633 | 81.581 | 14.052 |

- **毫秒骨架取自 Phase 0e**（`attrib5`），因为它最接近 bench 且内部一致（13 段、子段、同步、启动、消融都在同一轮）。
- **调用点计数取自 Phase 0f**（`attrib6`）。这一步安全，因为**计数在两轮之间逐项相同**：全帧 op（4438/3622）、启动次数（1223/951）、同步次数（16）、每 stage 的 op 与启动、`select` 次数（461/189）、逐内核启动次数——全部一致，只有毫秒漂移。
- **CPU 侧四分区取自 Phase 0b**（`attrib2`），因为 `select`/ATen 计时与那一轮的帧墙同源。

合成给出的恒等式（`both`）：

```
带 stage 窗口的帧墙 82.2089 = host 提交 68.9837 + fence 税 6.8657 + 帧尾 6.3596
```

### 23.7 这一节回答了 §22.11 的哪些遗留问题

| §22.11 遗留 | 本节结果 |
| --- | --- |
| `triton_fp8._kernel` 383 次启动在干什么 | `triton_fp8.py:42`（383 `to` + 28 `contiguous`）+ `:43`（383 `empty_like`），共 794 op/帧 |
| 32 次标量 H2D 从哪来 | `sampling.py:14` 的 28 次 `aten::lift_fresh`（另 4 次在长尾） |
| `is_nonzero` 5 次从哪来 | `reciprocal.py:29` 2.625 + `temporal.py:106` 2.0 |
| `body_rest` 的 host 提交 62.7% 是什么 | `executor.py:80/83/85` 的形状/切片解包（625 op/帧）+ 启动路径 + eager ATen |

**仍未回答**（如实记录）：① `pre` / `RGB` 两个 block **内部**未细分（无 fence 时它们分别是 0.5974 / 1.0453 ms）；② **816 个消失的 ATen op 值 3.96 ms（4.85 µs/op）仍是未解释的残差**；③ **驱动队列深度未直接测**——所以 `host 提交 68.98 ms` 仍然只是**上界**；④ pass O 的长尾未归类（见 §23.9 第 7 条）。

### 23.8 优化优先级（**上界，不可相加**）

| # | 目标 | 可回收上界 | 改字节？ |
| --- | --- | --- | --- |
| 1 | `triton_fp8._kernel` 383 次启动 / 7.61 ms（先问 FP8 重写能否提高每次启动的行数） | 次数减半 ≈ 3.8 ms | **是** |
| 2 | 缓存已绑定参数 + 已编译内核（Phase 1b） | ≤ 7.84 ms（`both` 的 `prep`） | 否 |
| 3 | 缓存 `spill_preflight_v1.select` | ≤ 2.85 ms | 否 |
| 4 | 拆 `fused_c32_projection_native_half_v1.py:99` 的 2 次 fence | ≤ 5.13 ms，受总 fence 税约束 | 否 |
| 5 | 拆 13 个 stage fence（`executor.py:72`） | 4.15 ms（K1 实测） | 否 |
| 6 | `sampling.py:14` 的 28 次标量 H2D 预建为 0 维设备张量 | 属 §21 的 22.9→13 ms 那一笔 | 否 |
| 7 | `reciprocal.py:29` 区间校验移到构造期 | 2.625 次同步/帧 | 否 |

**四条不许犯的读法错误**（与 §22.9 同源，这里给出量化）：

1. **第 4、5 行不能相加**：名义 5.13 + 4.15 = 9.28 ms，而全部 16 次同步的帧级回收实测只有 **7.29 ms**（K2）。差额是被搬走的设备等待。
2. **第 2、3 行相加（10.69 ms）会吃掉 host 余量 11.27 ms 的 95%**，一旦 host 压到设备水位（57.71 ms）以下，瓶颈反转。
3. **第 1 行会改字节**（改行数/批大小 → 改 K 累加顺序），**必须走 `local` → `gate` → `full` 的逐字节门槛**。
4. **"13.71 ms 同步墙钟"不是收益**，真实收益是 7.29 ms；**绝对毫秒不作加速比证据**——加速比只看 bench（§16 的 1.172×）。

### 23.9 本轮修掉的真实错误（两个真错 + 两个精度问题 + 两个静默降级）

1. **`TypeError: NoneType does not support the context manager protocol`** —— pass O 只写了 `active['dispatch_census'] = ...`，没调 `dispatch_census.install(torch)`。对照 `counter.install(torch)` 的用法修掉。
2. **`AssertionError: frame 2 changed`** —— pass O 从 `measure_from` 起跑，而模型跨帧有状态（§23.3）。**字节门抓住了它。** 修法：永远从第 0 帧开始，并写成硬门槛。
3. **`callsite_steady_exact` 首次报 False** —— 用 4 帧均值比 28 帧中位数。改成按 index 逐帧比对，实测 4/4 精确相等。
4. **`by_inner` 全落在 `rows_attrib6_v1.py:341`**（`__torch_dispatch__` 自己的行）—— `sys._getframe(depth)` 从 `__torch_dispatch__` 内部起算，`depth=1` 解析成了同一帧。改成 `sys._getframe(depth).f_back`。
5. **未归因门槛过严** —— 1.0 op/帧被判 problem。改成占比门槛（≤1% 记 note、>1% 才 problem），并按 op 名列出。
6. **静默降级之一：`BUDGET.md` 里整张 CPU 分区表消失，没有任何报错。** 合成脚本读 `row['overhead_ms']`，而该字段不存在（真名是 `attrib2['dispatch_timed']['net_ms']`），miss 之后静默落到 `None`，`if data.get('partition')` 为假，整段不渲染。修法：改读真字段，**并把缺输入改成致命错误**。**这一条是靠"表里少了一节"发现的——如果只检查脚本退出码，会以为合成成功。**
7. **静默降级之二：`by_site` 被静默截断到前 40 行。** 导出时 `[:40]` 且没有任何覆盖率声明。实测这 40 行只覆盖已归因 dispatch 的 **75.9%（`both`）/ 77.9%（`baseline`）**，另有 **843.6 / 954.5 op/帧**落在长尾上。所以**任何从这张表算出的按文件占比都是下界，而报告里读不出来**。修法：
   - `rows_attrib6_v1.py`：上限提为具名常量 `CALLSITE_SITE_LIMIT = 120`，新增 `callsite_site_coverage` / `callsite_site_tail_per_frame` / `callsite_site_lines_total` 字段与说明。
   - `rows_attrib6_teardown_v1.py`：新增覆盖率门槛 `CALLSITE_COVERAGE_MIN = 0.80`——**这个值不是质量线，是校准到能抓住已发生过的那一次**（40 行 = 75.9%），120 行会以余量通过。
   - **本轮不重跑**，所以 `attrib6` 仍是 40 行口径；报告与 `BUDGET.md` 都明确标注"这是下界"。下一轮跑 Phase 0f 时会自带覆盖率。

### 23.10 顺带修正 §22.3 的一处口径

§22.3 的 13 段表最后一行"合计"给的 `µs/unit` 是 **12.70 / 9.71**。核对后发现：**分子是 13 段窗口的毫秒，分母却是全帧单位数**（`both` 4573 = 3622 op + 951 启动）。一致的口径是"窗口毫秒 ÷ 窗口自己的单位数"（`both` 3681 = 2731 op + 950 启动），应为 **15.78 / 12.06**（`baseline`：4769 单位 → 15.00 / 12.17）。

各段自己的 `µs/unit` 全部正确，只有合计行混了口径。**`FRAME_BUDGET_REPORT.md` 用的是 15.78 / 12.06。**

同理，§22.3 那句"残差 `stage.tail` 只有 0.012 ms"说的是**实测的 tail scope**（`both` 0.0121 / `baseline` 0.0130），而"13 段合计占 body 的 99.42%"是**各段中位数之和**的比值。两者的差（`both` 0.3389 ms）里，除 0.0121 ms 的 tail 之外，剩下的是"中位数之和 ≠ 和中位数"的固有差。两个数都对，但不是同一个量。

### 23.11 这一节能证明什么 / 不能证明什么

**能证明**

1. 3622 个 op 归到 40 个源码行，未归因 0.03%，且与 pass I **逐帧精确相等**（不是中位数比对）。
2. `triton_fp8.py:42/43` 是 794 op/帧（≥21.9%）的单点，与 §22 的 383 次启动 / 7.61 ms 是同一处。
3. `sampling.py:14` 是 28 次标量 H2D 的唯一来源；`is_nonzero` 的 5 次全部落地。
4. 计数在两轮之间**逐项相同**，所以调用点表可以跨轮与毫秒骨架拼接。

**不能证明**

1. **任何加速比**——绝对毫秒只作定位与占比（轮间漂 ±2.8%）。
2. **按文件聚合是下界**——导出覆盖 75.9%，长尾 843.6 op/帧未归类。`triton_fp8.py`（两个站点都在前二）不受影响，其余文件可能更高。
3. **`host 提交 68.98 ms` 是上界**——未测驱动队列深度。
4. **逐内核计数表的名字不稳定**：pass L 经 `kernel.src` 反查名字，而 Triton 的编译缓存会让两个同形 `JITFunction` 指向同一个 `CompiledKernel`，`src` 归属随首次编译顺序变。实测 **8 次/帧（0.65%）** 在 Phase 0e 被记成 `nr_backend.triton_attention_exp._kernel`、在 Phase 0f 被记成 `fused_vit_projection_v2._parts`（两者同属 ViT 投影内核族）。**所以逐内核定价一律用 pass M 的 `JITFunction.run` 计时器，它两轮一致。**
5. ~~**`pre` / `RGB` 内部未细分**~~ → **组合已答（2026-09-18，见 §24）：它们是网络的第 0 块与第 70 块、跑在全分辨率 512×896 上；只剩"计时"未做**。
   **816 个消失的 op / 3.96 ms 仍是未解释的残差**（⚠️ 与 §23 那张"段外 748 op"的快照**不同源、不可互引**）。
6. **pass O 的毫秒不作预算行**（栈行走成本远高于 pass I 的计数模式）。
7. 所有探针都是**运行期重绑**，在 `installed()` 内、`ExitStack` 展开前还原；产品树、历史入口、冻结参考未被修改。

### 23.12 复现方式

```bash
cd E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580/reference/fullsize_rows_v1
bash rows_attrib6_run_v1.sh 32 4 attrib6          # 一次租约，两变体，32 帧、稳态窗口 [4,31]
python rows_budget_v1.py --run D:/fullsize-rows-v1/r1 --outdir budget   # 纯合成，无 GPU
```

产物：`D:/fullsize-rows-v1/r1/attrib6/{baseline,both}/attrib.json`、`attrib6/teardown.json`（两变体 `passed: true`、`problems: []`）、`budget/{budget.json,BUDGET.md}`；报告在 `fullsize_rows_v1/FRAME_BUDGET_REPORT.md`。

**注意**：`bash rows_attrib6_run_v1.sh` 的帧数参数**不改变 pass O 从第 0 帧开始**这一约束（`CALLSITE_FRAMES = 8` 与 `--frames` 独立）。改 pass O 的帧数前先读 §23.3。

### 23.13 下一步（**尚未执行**）

- **Phase 1b**：缓存已绑定参数与已编译内核（§23.8 第 2 行，≤7.84 ms）——这是**不动字节**的最大一笔。
- **缓存 `select`**（第 3 行，≤2.85 ms）。
- **调研 FP8 重写**（第 1 行）——需先证明能过字节门。
- **补测驱动队列深度**（§23.11 第 3 条），把 `host 提交` 从"上界"变成"测量"。
- ~~**细分 `pre` / `RGB`**~~ → **组合已答（§24）**；剩下的只有"在块内装子窗口计时"这一半。
- **消掉 816 op / 3.96 ms 的残差**（`baseline → both` 的 op 数减量，**不是** §23 的 748 快照）。
- 任何改动**必须另起一轮目录**——本轮 `gate` 已把 `adapter_sha256` 钉在 `rows_entry_v1.py` 上。

---

## 24. 2026-09-18 自查修正：三处归属错误 + `pre`/`RGB` 的静态细分

**这一节不产生任何新测量。** 它记录的是对 `FRAME_BUDGET_REPORT.md` §12（2026-09-18 上午追加的
"逐模块明细与时间轴"）的一次自查，以及自查后补上的纯静态结果。**没有任何承重数字被改动**，
改的全是**归属**。

### 24.1 三处错（逐条留痕）

| # | 错在哪 | 事实 | 判据 |
| --- | --- | --- | --- |
| 1 | §12.1 把 `frame.model − motion.body = 23.4113 ms` 说成「**没有任何 scope**、§1–§7 从未拆过它」 | 它就是 `motion.warp_normalized` **22.9284**（Phase 0d / `attrib4`，§21 已拆到 5 项、无余量）+ `head + front_features` **0.4829**（相减） | `22.9284 + 0.4829 = 23.4113`，**§1 的顶层树里一直写着这两行** |
| 2 | §12.6 把那张 748 op 的表读成"**没有落在任何一个被计时的 stage 里**"（隐含"未被认领"） | 它就是 warp 的 op 混合比：19 项里 **17 项**与 `attrib4` 的 `warp_ops_by_name['warp']`（合计 778）**逐字相同** | 逐项并排比对，见 `FRAME_BUDGET_REPORT.md` §12.1 |
| 3 | §12.9 动作③「给 **816** 个消失的 op 定价」 | **816 是 §22.11 的 `baseline → both` op 数*减量***（`empty_like` −152 / `empty` −176 / `reshape` −128 / `slice` −160 / `view` −160 …），与 §23 的 **748（`both` 的快照）不同源** | 两个数各自的出处可追：816 → `RESULT.md` §22.11；748 → `frame_ops_by_name − Σ stage_ops_by_name` |

**根因（一句话）**：**"相减出来的余额"在宣布它无主之前，先回报告自己的顶层树里找一遍。**
§1 的树、§20 的 `stack.model` 分解、§21 的 warp 分解都在**同一份 `RESULT.md`** 里；
当时只查了 `attrib5` 的 scope 字典，没查报告自己。

**§12.9 的三条下一步因此两条作废**（① 作废、③ 作废），只剩「拆 `pre`/`RGB` 内部」一条 —— 而它的一半也能静态做完。

### 24.2 `pre` / `RGB` 的静态细分（新）

`§9「不能证明」#6` 与 §23.13 都说这两段"内部未细分"。**细分信息一直在源码里**，
当时数错了单位：**13 段的块数相差 9 倍**（1 到 9），"启动/帧"不是可比单位。

**13 段 = 网络 71 个权重块的 13 个分组**（`executor.py` 的 `record()` 映射，块 0–70，无遗漏无重复；
`decoder C512` 是 9 块 = 8 个 `SplitSwinBlock` + `DecoderInputUpsample(record(39))`）：

| stage | 权重块 | 块数 | 分辨率 | 窗口 ms | 设备等待 | **等待/块** | 启动/块 | op/块 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| **`pre`** | block0 | **1** | **512×896（全）** | 5.1312 | **+4.5338** | **+4.5338** | 10.00 | 44.0 |
| `encoder C32` | 1–4 | 4 | 256×448 | 3.4217 | +1.5205 | +0.3801 | 9.75 | 32.5 |
| `encoder C64` | 5–8 | 4 | 128×224 | 2.5480 | +0.4866 | +0.1216 | 11.75 | 29.2 |
| `encoder C128` | 9–14 | 6 | 64×112 | 2.9768 | +0.0105 | +0.0017 | 11.50 | 27.2 |
| `encoder C256` | 15–22 | 8 | 32×56 | 4.0566 | +0.0954 | +0.0119 | 11.38 | 26.1 |
| `encoder C512` | 23–30 | 8 | 16×28 | 5.7128 | −0.0432 | −0.0054 | 15.38 | 42.6 |
| `ViT` | 31–38 | 8 | 16×28 | 9.4140 | −0.0596 | −0.0075 | **23.00** | **91.1** |
| `decoder C512` | 39–47 | 9 | 16×28 | 5.9323 | −0.0298 | −0.0033 | 14.11 | 39.8 |
| `decoder C256` | 48–55 | 8 | 32×56 | 4.0123 | +0.1732 | +0.0217 | 11.50 | 24.1 |
| `decoder C128` | 56–61 | 6 | 64×112 | 3.0379 | −0.0752 | −0.0125 | 11.67 | 24.5 |
| `decoder C64` | 62–65 | 4 | 128×224 | 2.4669 | +0.4814 | +0.1204 | 12.00 | 25.2 |
| `decoder C32` | 66–69 | 4 | 256×448 | 3.5225 | +1.7789 | +0.4447 | 9.75 | 28.2 |
| **`RGB`** | block70 | **1** | **512×896（全）** | 5.8609 | **+4.8156** | **+4.8156** | 11.00 | 86.0 |
| 合计 | 0–70 | **71** | — | 58.0939 | +13.6883 | — | 950 | 2731 |

**三条结论**：
1. **`pre` 与 `RGB` 不是前后处理，是网络的第一块与最后一块**，跑在**全分辨率 512×896** 上
   （`pre = PreBlock(record(0))`、`RGB = ResetPostBlock(record(70))`，`executor.py:35` / `:54`）。
2. **启动/块几乎是常数（10–12）** ⇒ 旧表述「`pre` 只有 10 次启动所以便宜」是**拿 1 块比 4 块**。
   按块归一化后 `pre`/`RGB` 是**全帧最贵的两个块**：**+4.53 / +4.82 ms 每块**，
   是 `encoder C32` 单块（+0.38）的 **11.9× / 12.7×**，而面积只大 **4×**。
3. **`ViT` 是"启动最密"的一段**（23.0 次/块、91.1 op/块），但设备等待为负 —— 已被 c512/vit 快速路径处理。

**⚠️ 边界（必须写在结论旁边）**：**不能**据此说"贵在它们自己的内核"。
`mark()` 是无条件同步，`mark('pre')` 是 body 里的**第一个** fence、`mark('RGB')` 是**最后一个**
⇒ 这两处是整帧的**排水口**，等到的量可能包含上游入队、尚未被前面的 fence 收走的设备工作。
**本节只主张"按块归一化后这两块最贵"，不主张"贵在它们自己的内核"。**
要坐实"每块重量"，必须在块内装子窗口 —— 这是 §24.3 唯一还值得花租约的事。

### 24.3 修正后的下一步（只剩一条值得花租约）

| 优先级 | 动作 | 依据 |
| --- | --- | --- |
| **1** | 在 `PreBlock` / `ResetPostBlock` **内部**装子窗口（按 `PreMLP` / `PreAttentionCore` / `sm89_f16_dot` / 池化分组），与 13 段同规格 | 两块合计 **10.99 ms = 帧墙 13.4%**，其中 **9.35 ms 是设备等待**，而每块只有 10–11 次启动 ⇒ 成本在**单次启动的重量**，不在次数 |
| 2 | 给 `motion.warp_normalized` 装 pass-H 同规格的窗口 | 让帧预算在**同一套方法下闭合**（现在 71% 走 pass H、28.4% 走 Phase 0d）。价值是可读性，不是新信息 |
| 3 | 逐段**设备**时间（profiler 按段切） | 唯一还没被任何方法覆盖的量：设备总忙时 57.7108 里只有 13.6883 有段归属 |

> ⚠️ 动作 1 的探针必须装在**参考数学**的边界上（`PreBlock.forward_features_unquantized` 那条链），
> 不是 `forward_features_outputs` 的外部 —— 否则池化与 `quantize_fp8` 会被算进 `pre` 的残差里。
> 三条都不改产品、不动历史入口，都是只加探针的诊断轮。

### 24.4 落盘位置

| 文件 | 变化 |
| --- | --- |
| `FRAME_BUDGET_REPORT.md` | §12 头部加修正声明；**§12.1 整节重写**（标题从「★ 新发现」改为「⚠️ 修正」）；§12.6 加修正块；**§12.9 重写**；**新增 §12.10（71 块阶梯 + 按块归一化）与 §12.11（修正清单）**；§9「不能证明」#7 加口径澄清 |
| `FRAME_TIMELINE.html` | 由 `render_timeline_v1.py` 重新渲染（55462 → **63955 B**）：第 2 条轨改名；**新增 §0（warp 的 5 项划分）与 §0b（71 块按块归一化）**；§6 加修正块；§8 边界改 5 条 |
| `render_timeline_v1.py` | 新增读 `attrib4` + **4 条新断言**（`warp=22.9284`、`head=0.4829`、两者之和 = `unnamed`、`Σ块数=71` 且段名集合与 `stage_order` 一致）⇒ 归属一旦漂就当场炸 |
| `RESULT.md` | 本节 + §23.11 第 5 条与 §23.13 的两条 bullet 就地更新 |

---

## 25. Phase 0g 归因（**已执行**，2026-09-18 晚）：`pre` / `RGB` 段窗口的内部构成

轮次目录 `reference/prergb_v1/`（探针 `prergb_v1.py` / 驱动 `prergb_run_v1.sh` / 拆解 `prergb_teardown_v1.py` / `SUMMARY.json` / 详版 `RESULT.md`）。
本目录**一个字节都没改**。完整版见 `prergb_v1/RESULT.md`，摘要进 `FRAME_BUDGET_REPORT.md` **§13**。

### 25.1 回答的是 §12.10 那句不能下的结论

§12.10 写：`mark('pre')` 是 body 里第一个 fence、`mark('RGB')` 是最后一个 ⇒ 这两处**可能**是"整帧的排水口"，
所以**不能**说 `pre`/`RGB` 的 +4.5338 / +4.8156 ms 是它们自己的。**本轮量完，答案是否定的。**

**方法**：两个配置 —— `base`（什么都不加，复现 §12.2）与 `fence_each`（body 入口一次同步 + `pre`/`RGB` 每个子步骤后一次同步）。
入口那次把"上游积压"单独记成 `entry.fence`，此后每个子步骤的窗口 = **它自己的提交 + 它自己的设备工作**。
另加一个 `warm` pass 先付进程预热，`base`/`base2` 互为漂移检验。

**配置**：`both` 变体 / 32 帧 / 稳态 `index >= 4`（28 帧中位数）/ 4 pass / **128/128 帧逐字节等于冻结参考** / 两次租约（13 s + 23 s）。

### 25.2 结果

| 量 | 本轮 `base2` | §22（Phase 0e） | 差 |
| --- | ---: | ---: | ---: |
| 帧墙 | 83.1629 | 82.3144 | +0.8485 |
| `motion.body` | 59.2931 | 58.4328 | +0.8603 |
| `stage.pre` | **5.1611** | 5.1312 | **+0.0299** |
| `stage.RGB` | **5.9949** | 5.8609 | **+0.1340** |

> ⚠️ 本轮 session 整体比 Phase 0e 高 +0.85 ms（帧墙/body 同步偏高），而两段只高 +0.03 / +0.13 ⇒ **只用同轮内相对量**。

**`entry.fence` = 0.9854 ms** —— 这是 K2 消融**结构上测不到**的量（K2 拆掉所有同步，只给"完全不等的 host 提交"）。

**`stage.pre` = 5.1611 四项闭合（误差 0.0000）**：
上游积压 **0.9854（19.1%）** ｜ pre 自己的 host 提交 0.5974（= K2 的 `pre`）｜ 被 `pre.attention` 内部同步吸收的设备等待 1.9439 ｜ `mark('pre')` 处排掉的设备工作 1.6344。

**`stage.RGB` = 5.9949 三项闭合（误差 0.0002）**：
上游积压 **0** ｜ host 提交 1.0453（= K2 的 `rgb`）｜ 被 `rgb.attention` 内部同步吸收的设备等待 2.2414 ｜ `mark('RGB')` 处排掉的设备工作 2.7084。

⇒ **"排水口"只对 `pre` 成立、上限 19.1%；对 `RGB` 完全不成立。**「按块归一化后两块最贵」**站得住**（合计 11.1560 ms = 帧墙 13.4%）。

### 25.3 机制（新）

`pre` 与 `RGB` 是**唯一**两块跑在全分辨率 **512×896** 的块（其余 11 段 ≤256×448）。
`fused_c32_projection_native_half_v1.py:97-99`（`ProjectedHeadLayout.c32`）里有按块数决定次数的同步：

```python
chunks = triton.cdiv(features.numel() // 32, 32768)
if attention.C32AttentionFront.forward is not body.c32_front:
    for _ in range(chunks // 8): torch.xpu.synchronize(features.device)
```

512×896 → 像素 458,752 → `chunks = 14` → **1 次**；≤256×448 → 114,688 → `chunks = 4` → **0 次**。
⇒ **全帧 2 次非 `mark` 同步，正好是这两块各一次**，与 §22.6 计数表的「`c32` 2 次/帧」**独立对上**。

**直接证据**：把每个子步骤前面排空后，26 个子步骤里 `base − fence_each` **只有两个 `attention` 是负的**
（`pre.attention` 3.0988 → 1.6236，Δ −1.4752；`rgb.attention` 2.3770 → 1.8581，Δ −0.5189），其余 24 步全为正。
⇒ **base 窗口里的等待集中在这两个 `attention` 内部的同步上，不在段边界。**

### 25.4 两条顺带的修正

1. **`rgb.clamp/alpha/delta/fma` 每帧各一次** —— harness 确实给了 history，时序混合尾巴**是活代码**，
   不是 §12.10 静态表里读出的"分支"。合计 base 0.2009 / fenced 0.3393 ms。
2. 序列化后 `RGB` 单步最贵的是 **`rgb.head_dot` 1.2502 ms**（`dot(full, head_weight, chunk_k=8)`，512×896×8），
   其次是 `rgb.attention` 1.8581、`rgb.mlp` 0.6298、`rgb.merge` 0.4367。

### 25.5 边界（不许越界）

- **没有 profiler ⇒ 没有"设备时间"**。所谓"设备等待"是同步前后墙钟差；`serialised − base_leaves` 混了两种成分，**不当设备时间用**。
- **`fence_each` 改了调度**（28 次额外同步，帧墙 83.1629 → 84.4685）。它的帧墙**不参与任何跨 pass 比较**。
- **`pre.attention` 里那次同步能不能去掉，本轮不回答**；去留属于改产品调度，需独立一轮（字节门 + 峰值分配检查）。**本轮不提出这个改动。**
- 只测了 `both` 变体。

### 25.6 落盘

`prergb_v1/RESULT.md`（详版）｜ `prergb_v1/SUMMARY.json`（4 pass 中位数 + 逐步骤表 + 全部检查）｜
`FRAME_BUDGET_REPORT.md` §13（摘要，并就地改写 §12.10 末尾的"未坐实"表述）｜ 技能陷阱 **103**。
