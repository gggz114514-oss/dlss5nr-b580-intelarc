# B580 单帧 RGB 执行状态

2026-09-08：完整原始模型已在B580独立执行。已验证范围为256×256、SDR、reset帧、默认控制参数，固定SF-v2参考运行库。用户已审核人物、赛车、Apex三组输出，明确回复“这版画面正常，继续”。完整迁移仍未完成。

## 执行与证据

`backend/nr_backend/executor.py` 的 `ResetNR256` 加载原始权重和192MiB原生标量查找表，直接计算RGB→RGB。表覆盖全部24位均匀随机标量输入；与图片或种子无关，不是缓存参考特征。实际推理不调用CUDA/NVIDIA运行库，不导入第三方模型Python。

- `reference/results/reset-executor-v1/validation.json`：gradient、checker、gray、Apex(seed17)，最终FP32 RGB逐字节零差异。
- `reference/results/reset-executor-v1/visual-validation.json`：新视频裁剪portrait(seed0x12345678)、car(seed0xffffffff)，最终FP32 RGB零差异；1544份原生流水线输入输出的哈希与来源核对通过。
- `reference/results/post-holdouts-v1/encoder-prefix-validation.json`：四份素材、CPU/B580、每份176个完整缓冲区，1408次比较全部零差异。这项从相同原生front出发；上面的六份RGB验证另行覆盖了独立front。
- `reference/results/noise-lut-v2/front-validation.json`：全16通道front、四份素材、CPU/B580均零差异。
- `reference/reset-visual-user-acceptance.json`：本轮用户视觉验收。

原生参考使用社区SF-v2 RTX40兼容运行库，不能外推为与官方RTX50版本等价。

## 调用

```python
from nr_backend import ResetNR256
model = ResetNR256.from_assets(weights_path, noise_directory).to('xpu').eval()
result = model(rgb_hwc_float.to('xpu'), seed=0)
```

也可使用 `backend/run_reset.py INPUT --output NEW_DIRECTORY --device xpu --seed 0`。输入必须是256×256，CLI输出PNG、FP32 RGB NPY和运行清单，不自动缩放图片。

## 当前限制

正确性路径使用整数张量模拟原生矩阵算子的舍入，B580单帧实测约17–20秒，不适合实时游戏。图像尺寸、视频历史状态、非默认控制参数尚未完成；后续XMX/INT8优化需以这些参考结果验证误差与画面，OptiScaler和现有工具尚未接入。INT8峰值算力不能直接当作本模型的实测性能。

后续更大尺寸和视频输出仍需代理自检与用户肉眼审核。当前接受记录只覆盖本轮256×256单帧原型。
