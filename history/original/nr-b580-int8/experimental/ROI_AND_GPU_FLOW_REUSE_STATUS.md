# 局部 NR 与已有 GPU 光流接入（2026-09-09）

用户明确：额外缩放器指只处理画面中的指定区域；只是听说，没有具体链接。隔壁项目已有稳定快速的 GPU Block/GPU DIS，授权复用。停止把生成参考输入所用的 CPU DIS 当作产品瓶颈，停止开发替代光流，继续以 NR 执行速度为主。

## 找到符合描述的公开实现

`ClarkCheekyKent/CheekyFoveatedDLSS`，此次固定检查提交 `97f8e4e300cda835a0f73ed848134da530f9b4ce`。不能断言它就是用户听说的项目。README 将DLSS-NR支持标为实验性，区域的宽、高、形状和过渡可以独立设置或链接SR配置。圆角/椭圆形状作用于合成，实际网络工作范围仍为外接矩形，不能用圆形可见面积推算省算力。

- `src/dlss_nr.cpp:1339` 计算NR裁剪区域，再用区域宽高和 `nr_working_scale` 决定工作尺寸；因此同时支持区域裁剪及区域内部降分辨率。
- `src/dlss_nr.cpp:509` 将真实NR feature的Width/Height和多组输入输出尺寸设置为工作尺寸；`1533`附近设置小纹理及对应subrect。不是仅在整图运算后遮罩效果。
- `src/dlss_nr.cpp:1090`、`src/dlss_nr_contract.cpp:8` 把裁剪范围对齐到8像素；`scaled_extent` 最少32并按8对齐。这是该插件调用原生DLL的策略，不证明B580后端已经支持这些尺寸。
- `src/dlss_nr_contract.cpp:16` 比较前后区域大小、工作尺寸与运动比例，移动区域时补偿原点差；不兼容几何或无重叠则重置。`src/dlss_nr.cpp:1441`附近把补偿录入私有运动纹理，失败则重置，不覆写游戏原纹理。
- `src/dlss_nr.cpp:918`附近将处理结果按形状权重和feather混回区域原图，其余位置保留原画。NR色彩合成并非本项目的简单有符号残差公式，不能声称像素等效。

主来源：

- https://github.com/ClarkCheekyKent/CheekyFoveatedDLSS
- https://github.com/ClarkCheekyKent/CheekyFoveatedDLSS/blob/97f8e4e300cda835a0f73ed848134da530f9b4ce/src/dlss_nr.cpp
- https://github.com/ClarkCheekyKent/CheekyFoveatedDLSS/blob/97f8e4e300cda835a0f73ed848134da530f9b4ce/src/dlss_nr_contract.cpp

11份公开源码/文档快照在D盘 `reference/community/cheeky-foveated-review-v1/`，manifest记录逐文件URL/SHA256；仅阅读，无编译、安装或源码移植。上游GPL-3.0-only；这里没有把它的源码拷进本地实现。

## 与当前B580速度工作的关系

当前最快256残差路径，把整幅1080p缩为256×144有效区域，置于256×256模型输入内，再由原网络填充到320×320。裁剪后若仍装入相同模型尺寸，网络计算量仍大致相同，只会把相同计算预算集中到较小画面区域。不能据此声称把35ms直接再打折。

局部处理可以与内部缩放叠加，但必须实际减少网络张量尺寸才有对应计算收益。下一步需衡量受现有填充/ViT token约束影响的固定成本，继续融合C512/ViT等热点，并独立验证更小/矩形几何。当前精确算术实现只允许已捕获的尺寸，最小已验证输入256×256；不得把放宽尺寸检查当作完成原生几何迁移。

裁剪也会改变跨区域上下文；本模型含全局ViT注意力，不能保证裁剪区域内与整图推理字节相同。保护边和羽化只能改善边缘衔接，不能恢复被删除的全局上下文。局部模式只能作为快速分支选项，保留完整NR模型和整图精确分支；画质变化需完整连续帧及人工审核。

## 复用本地GPU链路的接口

已只读检查：

`E:/ComfyUI-aki-v3-IntelArc_20260722/xess-tools/projects/xess-offline-toolbox/`

- `src/gpu/gpu_frame_contract.h`：`MotionPacket.current_to_previous` 已是当前→前帧、源有效区域像素位移；记录前后源帧ID、有效区域、纹理状态、LUID、生成fence。`FrameLease`保持源表面直到所有消费者完成。
- 同文件的 `MotionProvider.record` 向调用方命令列表录制，契约明确不提交、不等待调用方队列；`RecordContext.completion`由调用方提交后signal。
- `src/gpu/native_strict_dis_provider.h/.cpp`：现有Fast/Medium factory，NV12 Y/R8输入；输出源尺寸原始运动，可关反向流；按槽完成fence保护复用。严格DIS语义和既有产品快速GPU DIS不是自动等同，接入时选择已验收provider实例，不能只凭名字替换。
- `src/core/include/xve_gpu_block_core.h`：共享已审核Full H2实现的帧对和设备/队列契约，Lite V4为显式实验选择。`xve_motion.h`已有GPU Block/实验GPU DIS等provider标识。
- 离线验收文档记录GPU Block/GPU DIS/AMD完整流程已运行；本轮没有重新跑这些算法，也没有拿离线成片FPS当单帧光流延迟。

可以直接复用算法和纹理，尚须完成NR的D3D12纹理到SYCL/XPU输入互操作与完成通知。取provider原始源网格运动，避免重复消费已经为XeSS输出分辨率缩放过的速度；缩放只做一次。区域局部坐标需包含原点差：固定等尺寸情况下 `v_local = v_source + current_origin - previous_origin`，然后按模型工作尺寸/区域尺寸缩放XY。需一并验证有效区域、源帧ID、reset、浮点范围、槽生命周期及NR完成回执。

绝不关闭/重置/提交调用方未提交的命令列表，也不等待一个必须靠该未提交列表才能产生的输入fence。接入应顺着既有host提交顺序安排GPU依赖。当前本轮只确认接口，没有宣称已完成零拷贝或实时整链。未修改隔壁项目、默认光流路线或发布源码。
