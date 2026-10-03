# 复现能力与缺口（截至2026-10-03）

**可以重建历史证据索引、检查源码与原报告hash；不能宣称全部历史GPU实验或当前游戏链在独立机器上已完整复现。** main已经完成实际源码提取／发布准备，本worker没有GPU、模型导入、G盘读取或干净安装实测。缺失weights、frozen素材／原始trace、私有运动实现和旧环境是不同类型的缺口，不能用一个“源码公开”笼统消除。

## 能用本次标准库脚本重复的部分

从`sol-history`运行相应Python脚本；Windows推荐隔离模式`python -I -B`。脚本不导入项目／Torch／Triton，不调用GPU、网络、删除或系统安装。默认库存只输出stdout；`--snapshot`或build脚本只在本archive写索引，不改被读的项目。

| 脚本 | 操作和输入 | 重建内容／限制 |
| --- | --- | --- |
| `rebuild_inventory.py` | 七个明确project/experiment根；`--snapshot`可选 | `SOURCE_INVENTORY.jsonl`、`SOURCE_INDEX.json`；每层lstat/scandir不跟junction；跳过环境、cache、当前Luna attempts |
| `rebuild_inventory.py --check-index` | 本地`EXPERIMENT_INDEX.json`的absolute path/sha字段 | 逐文件存在／hash复查；缺失、目录、越界或reparse明确输出；不重跑实验 |
| `collect_evidence.py` | 已列根报告＋reference第一层RESULT/PLAN/README＋固定Oct3进展 | `EVIDENCE_CATALOG.json`原文摘录／line／hash；state不是指令，未将大state当完整历史 |
| `build_history_indexes.py` | curated41路线、catalog、库存、fixed current receipts | 原结果／参数摘录、source selection、public aliases、stage against main manifest的hash；不复制／执行源 |
| `build_cleanup_recommendations.py` | 明确旧Sep29 spool／Sep24解包副本 | 全零／ZIP byte proof的建议；没有删除命令，不允许未知全树清理 |
| `extend_cleanup_whitelist.py` | 明确旧解包／安装mirror，对应保留ZIP/runtime/candidate | 逐文件SHA相同的binary/编译中间物白名单；保留JSON、源码、报告、licenses、模型/media；不因一项不符拒绝其余已证副本 |

库存是focused observation。D/nr-b580触及180000 entries上限，该根不完整；所有cache-named目录、junction和当前Oct3 Luna attempts明确跳过。小binary通常只计入总数，≥128MiB才逐条列出；text/source≤2MiB默认hash。mtime只表示观察时文件时间，不是实验开始／采用日期。指定旧worktree仅允许读取已引用参数driver，不递归库存其他siblings。重跑库存可能因main正在整理而改变条目，不能假称快照锁住源。

## 分层复现矩阵

| 层级 | 可取得部分 | 尚缺内容 | 当前可主张 |
| --- | --- | --- | --- |
| 阅读／审计历史 | 阶段报告、失败、参数driver／kernel SHA、可找到原JSON | 未列瞬时arms、遗失Git对象／旧输出、部分盘外实验原数据未读 | 可重读主要路线，非所有历史对象完整归档 |
| CPU数学／协议 | 原标量／tile穷举脚本、CPU owner/fence/fixture测试源码及报告 | 具体旧脚本pin可能被后续修正；vendor头／compiler/toolchain版本 | 有原结果证据；本worker未重跑这些项目测试 |
| 局部GPU kernel | 原参数、部分owned activation/manifest、固定Triton／oneAPI报告 | 对应输入array、warmup/cycle/cache、driver/frequency、compile provenance | 部分路线具备参数入口；不保证独立原样重跑 |
| 完整离线NR | original driver、source/geometry/controls/reset/history、部分asset hash | 专有weights/profile arrays、视频、逐帧motion、原模型及source closure | 需用户合法资产／完整environment后可尝试，未称已复现 |
| 原4060精确门 | DLL/model/input/output/trace hash和受测byte结果 | 原库合法来源、616.56+等原环境、Session1、独立frozen原件分发限制 | 本地原件保护；公开不默认提供资产或逐字节重现 |
| 现役720游戏 | main实际installed source提取、accepted controls/constructor、桥合入/验收回执 | 游戏及vendor binaries、私有GPUBlock、模型、实际settings/cache、完整build/download/install门 | 用户场景55–60ms接受；源码身份与binary build identity分别记录 |

## 具体资产、版本和语义缺口

1. **weights与profile不是源码。** 固定WEIGHTS_HT SHA `836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4`、147695410B只是身份。profile／量化尺度／calibration数组须合法本地提取或提供，保留对应版本和hash；只有某个JSON中写了hash不等于数组已外供。不能给出“无需原资产的全部历史重跑”声明。
2. **冻结输入与原输出。** 原4060输入/depth/motion、arena／trace、所有独特原RGB和manifest保护。480／1080／Apex素材需要原decode版、FFmpeg版本、帧率/timebase/audio门和逐帧motion。当前公开工程不默认含用户媒体，索引必须将“报告所记hash”与“worker重新hash文件”分开。旧42.9709 G输出已丢失，不能再做原输出pixel对齐。
3. **私有运动和shader实现。** GPUBlock/DIS私人代码不进入public tree；公开可保接口、单位、shape、manifest和允许的诊断driver。替换motion provider会改变性能／质量／history，不可称原链相同。历史产品用户已取消DIS，不能未经当前授权重新默认为DIS。
4. **环境与硬件。** B580 XPU／Triton版本、oneAPI/compiler、UR/Level Zero/D3D12 adapter、驱动、firmware/power和每轮语言编码固定。隔离Triton3.8 ABBA不能视为用户Comfy环境已升级。原Session0 NGX失败需要Session1重现条件；icpx PATH、ONEAPI selector、runtime preload和中文路径门需单列。
5. **图、冷cache与history。** 720模型720×1280、内部768×1280；540与480padding不同。reset/temporal entries、seed、controls、noise/history、真实motion单位、return ownership和consumer retirement必须相同。captured Python hook执行次数不能当replay逐帧执行次数。warm runtime cache metadata可读不等于冷新进程DiskOnly零miss/compile/write。
6. **INT8范围和误差。** 48个相关calibration frames覆盖有限；CPU finiteE4M3／base128穷举、人工100% eligibility、random720 input与真实game activation不能混为一谈。bare DPAS计时未含rowquant/pad/layout/fusion时只能作local结果。前后quant boundary以及floor16、query trimming、C64 repeat等完整参数必须保存。
7. **桥源、构建和部署。** 当前math以main实际runtime extraction为准，r18 audit_impl只候选。accepted trial ASI12fe…与canonical build368d…是两个pin，CPU ABI／export相同不证明二进制相同。主代理后来对G逐文件核验解决了源码提取缺口；干净build和第三方二进制来源／许可仍由release-review闭合。本worker对stage manifest扩展文件数重新hash，不把authoring新增项伪称都来自G。
8. **历史缺损与采样范围。** 旧交接记录目录/Git事故、部分恢复；一些原数据位于七根以外的独立数据根，未在本轮重新检查。补充报告date未知时保留null，不能用mtime补日期。原报告出现错误“产品默认”／“所有exact收益0”／跨轮速度／host上界时保留原文与更正，不伪造一段干净成功历史。
9. **性能与画面复现。** timer必须声明local/core/rawgraph/body/process/bridge/web/RTSS。不同轮次的收益不能相加，parent inclusive不能叠child。至少同设置重复BCCB/ABBA、冷／hot／退出门、对应数据SHA和用户游戏画面，才讨论新采用。静态FP8/math、CPU mock、两帧HDR与短静止flash片段均有各自覆盖上限。

## 最小保留与公开索引

`HISTORY_SOURCE_SELECTION.json`本地表给出原文件实际路径、原SHA、阶段id和拟发布relative path；`HISTORY_SOURCE_SELECTION.public.json`去绝对路径、使用aliases。最小选择主要是原报告＋可定位driver/kernel，最终项数见该JSON，并非每条路线transitive dependency closure；未定位脚本和hash不齐在实验索引明确列出。684项experimental顶层py/md可作为可选小型历史bundle，仍须逐文件私有实现／许可／个人路径内容审查。本worker不复制原目录，不认为选中即允许公开。

报告脱敏后public byte SHA会改变，原证据SHA与公开文件SHA必须分别保存，不能仍声明public copy=original byte identity。原27MB库存和local receipt中的安装路径无需公开；公开用精简`PUBLIC_HISTORY_INDEX.json`，保留指向原hash和参数／计时边界。所有metadataJSON／report／manifest作为研究证据保留；清理表只授权main后来逐白名单再验证对象，不包含任何删除动作。
