# 公开阶段成果已发布（2026-09-09）

用户明确要求新建公开仓库 gggz114514-oss/nr-b580、发布 v0.1.0-pre，重点记录如何以4060原生参考恢复精确后端，同时固定实际项目成果。不是只发表叙述，也不是宣称完整迁移已完成。

- 仓库：https://github.com/gggz114514-oss/nr-b580
- 预发行：https://github.com/gggz114514-oss/nr-b580/releases/tag/v0.1.0-pre
- 公开主提交：695ae22a32830c6d266c0636d9fd8fb3ffb46a3a
- 注释标签对象：8f754faacf6b0ab6fa5f8589db04ad1da2e4240b
- 独立发布工作区：E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580-public

141文件约1.63MB，包含六篇研究/复现文档、20份历史/当前证据摘要、35文件精确后端、13份CUDA重放程序、早期捕获器及布局工具、50个快速内核依赖模块和5个公开工具。100份原样源码全部与本地源文件逐字节对齐。精确源来自7355848，快速源来自f62ecd34。未上传整个研究Git历史、权重/DLL/CUBIN、模型标量表、图片视频、原始张量或SSH/账号凭据。

发布前24,007组公开half FMA CPU对照零差异；标准库中点反例通过。发布结构核查91个Python语法、29个本地文档链接、20个摘要；源码身份与当前55+13精确回归分开说明。新CLI只做帮助/参数入口检查，未冒充已做完整模型回归；C++源码没有在发布工作区重新编译。公开包未附统一开源许可证，未给第三方资产重新授权。

认证通过用户明确指出的现有Git Credential Manager，账号核对为gggz114514-oss；凭据仅在内存用于api.github.com标准API，没有写入文件、URL或命令行。API建库、git push main与注释tag、API创建prerelease均成功，随后API核对public/prerelease/tag指向提交。

从GitHub重新clone该标签到D盘download-verification，141个发布文件（清单记录其余140个）全部哈希/语法/链接检查通过，独立FMA反例通过。两边HEAD同为695ae22，发布工作区与下载验证副本均干净。

此前内置浏览器未登录；Chrome浏览器工具未连接；Windows Chrome自动控制因无法可靠识别当前URL被工具停止，已停止UI操作。通过现有GCM完成全部授权发布，无需新增登录或用户手动建库。此前网页阻碍已解决，不要继续要求网页登录。

本地收据：D:/Codex-NR-Experiments/nr-b580/reference/publication-v0.1.0-pre，包含github-inspect/create/release/verify-v1.json、source-export-v1.json、publication-finalization-v1.json及远端回读副本。构建公开摘要初版对D盘直存的stage路径和list结构处理有遗漏，在发布前修复并完成验证；未改动原实验文件。

完整迁移目标仍ACTIVE；本次发布阶段任务完成。下一性能里程碑仍是同256输入和同计时范围追平4060，按用户要求优先模型本体。最新快速路径22.143602ms对4060完整host3.566807ms，尚有6.208倍差距；当前没有GPU作业或用户审核等待。
