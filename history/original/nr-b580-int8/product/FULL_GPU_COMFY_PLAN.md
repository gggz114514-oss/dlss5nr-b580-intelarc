# ComfyUI NR + XeSS：全 GPU 接入约束

2026-09-11 用户纠正并确认。本文替代“NR 成片后再交 XeSS”的节点接入计划。
当前为源码核对后的设计，不代表互操作或四个节点已实现、已通过 GPU 验收。

## 产品接口

- 四个独立 VIDEO→VIDEO 节点：NR、NR+SR、NR+FG、NR+SR+FG。
- 每个节点提供精确/快速 NR 选择；精确使用独立 nr/exact，快速使用已接受画质的路线。
- VIDEO 是入口及成片输出合同。内部接现有 GPU worker，不通过中间视频文件、CPU IMAGE
  批次或逐帧像素管道连接 NR/SR/FG。源文件读取、最终编码/封装和音轨处理属于两端。
- 复用 GPU Block/GPU DIS，二者是可选运动后端，不要求两个算法同时运行。
- 代码 E 盘，新增构建、缓存、试验输出 D 盘。本轮不发布 GitHub。

## 已从相邻源码确认

源码基点：xess-tools/projects/xess-offline-dlss5-experiment/src/gpu/。

`gpu_frame_contract.h` 的 FrameLease/MotionPacket 已定义：源帧编号、原始 PTS、颜色
元数据、adapter LUID、reset、资源 owner、生产 fence 和消费者完成 fence。
运动主方向为 current→previous，单位为源有效区域像素；还有可选 reverse/confidence。
运动缩放由消费者适配层执行，不在 SR 输出上重新估计。

`vpl_gpu_full_fg.cpp` 已在原始 source slot 上分析运动，SR/FG 消费共享结果。
`record_gpu_consumers` 当前从 slot.color 读 SR 颜色，再将 slot.sr_output 交给 FG。
GPU Block 还含原生内部路径；GPU DIS 走 MotionProvider。不能假定两者已经通过完全
相同的 provider 入口，接入 NR 时要统一消费其有效输出，保留各自已验证语义。

## 目标资源流

GPU 解码 → 原始颜色 → 一次源帧对运动分析 → NR → 可选 SR → 可选 FG → GPU 编码。
共享运动场分别供 NR/SR/FG 使用，GPU 上适配目标尺寸、像素中心、位移单位及格式。
一次源帧对分析可能包含双方向和修正 dispatch，不能把它报告为只有一次 kernel。
原始颜色仍须保留给下一对源帧分析，NR 输出使用独立资源；不得原位覆盖 source slot。
生成帧只在末端产生，不回灌 NR 历史或运动分析。统一 reset 原因，独立维护各模型历史。
共享 AI 深度只交给已有需要它的消费者；不能直接改变当前精确 NR 的零深度/default
controls 验证范围。精确版 NR 保留原尺寸，快速版复用现有 NR256+残差重建。

## 优先实现顺序

1. 独立验证同一 B580 上 D3D12 共享资源 ↔ PyTorch XPU/SYCL 的导入、GPU 转换和
   完成同步。先用固定模式验证往返位模式、slot 复用与取消销毁，暂不调用 NR 模型。
   现有本机头文件含 external-memory 和 external-semaphore 接口；驱动实际支持尚未
   验证，不能凭声明声称可用。必要的 texture↔linear buffer 转换/拷贝保持在 GPU。
2. 将 NR 实现为共享 worker 中的 GPU 消费者，新增增强颜色输出及完成 fence；GPU
   队列等待生产者完成后消费，所有消费者完成后才复用 slot。CPU 允许调度和诊断，
   图像及运动场不落 CPU。现有 Session 的 host completion 仍有成本，不能直接删掉。
3. 精确/快速分别接入同一个 NR 接口，独立验证同输入颜色/运动/reset 下的 NR 原始
   输出。更换 CPU DIS 为 GPU 运动场后，不能要求与旧的不同运动输入参考成片一致；
   应分开验收“算术迁移一致”与“新增运动链路画质”。
4. 接四个 ComfyUI 包装节点。复用现有 GPU SR/FG 路线、时间戳、音轨、取消与成片
   校验，验证单 NR 路线不误启 SR/FG，组合路线实际只执行用户选中的阶段。
5. 记录 NR/SR/FG 消费数、源帧对分析数、CPU 像素上传/回读字节数、GPU 拷贝及同步，
   再做连续视频、重置、显存回收和最终用户肉眼验收。禁止用本体计时代替整链耗时。

## 当前测试分工

既有 nr-product-full-v1 任务保持冻结，由原 Luna/max 监控和固定审计；主助手不轮询。
主助手编写/修复/冻结并实际启动后续 GPU 测试，Luna 只监控与执行固定审计，异常
立即回报，不自行修复或重试。当前文档没有启动新 GPU 测试，也未修改冻结算术源文件。
