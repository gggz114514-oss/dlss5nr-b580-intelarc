"""Instance-scoped FP16 QKV front fusion on top of the batched-head layout."""
import nr_backend.multihead_block as multihead
from nr_backend.execution import current_arithmetic_backend, record_arithmetic_dispatch
from fused_head_layout_v1 import HeadLayout as Base
from fused_qkv_pack_v1 import forward as prepare
from fused_swin_heads_v1 import forward
from window_layout_v1 import unpack


class QKVHeadLayout(Base):
    def __init__(self, model, provider, *, rows=16):
        super().__init__(model, provider)
        self.rows = rows

    def multi(self, model, features):
        if id(model) not in self.modules or self.provider.mode != 'fp16_xmx' or current_arithmetic_backend() != 'triton' or features.device.type != 'xpu':
            return super().multi(model, features)
        height, width, channels = features.shape
        if channels != model.channels or height % 8 or width % 8:
            raise ValueError('Expected whole multihead windows')
        z = multihead.sm89_f16_dot(multihead.quantize_fp8(features), model.qkv, chunk_k=16).reshape(height, width, model.heads, 3, 32)
        (q, k, v), _ = prepare(z, model.scale, model.pixel_order, rows=self.rows)
        for _ in range(2):
            record_arithmetic_dispatch('attention_normalize_c32')
        for _ in range(3):
            record_arithmetic_dispatch('fp8')
        result, _ = forward(q, k, v, model.bias)
        for _ in range(model.heads * ((height // 8 * (width // 8) + 1023) // 1024)):
            record_arithmetic_dispatch('batched'); record_arithmetic_dispatch('attention_exp_swin')
            record_arithmetic_dispatch('attention_weights'); record_arithmetic_dispatch('batched')
        self.calls['multi'] = self.calls.get('multi', 0) + 1
        self.calls['batched_heads'] = self.calls.get('batched_heads', 0) + model.heads
        self.calls['qkv_pack'] = self.calls.get('qkv_pack', 0) + 1
        return unpack(result, model.pixel_inverse).reshape(height, width, channels)
