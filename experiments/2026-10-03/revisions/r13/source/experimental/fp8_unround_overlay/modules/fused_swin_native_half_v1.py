# Explicit native-half attention version; see native_half_attention_fma_v1.py.
"""Serialized FP16 Swin core fusion using the measured BM32/warps4/stages1."""
from swin_scheduling_v1 import SwinScheduling as Base
from fused_swin_core_native_half_v1 import forward
from nr_backend.execution import current_arithmetic_backend, record_arithmetic_dispatch
from nr_backend.unround_policy import ENABLED, current_attention_owner


class FusedSwin(Base):
    def __init__(self, provider):
        super().__init__(enabled=True)
        self.provider = provider

    def windows(self, query, key, value, bias):
        if (self.provider.mode != 'fp16_xmx' or current_arithmetic_backend() != 'triton'
                or query.device.type != 'xpu'
                or not all(t.is_contiguous() for t in (query, key, value, bias))):
            return super().windows(query, key, value, bias)
        family = current_attention_owner()
        result, _ = forward(query, key, value, bias, bm=32, warps=4, stages=1,
                            round_weights=family not in ENABLED)
        groups = (query.numel() // 2048 + 1023) // 1024
        # Logical old operations for graph-completeness comparisons; one physical
        # fused launch replaces these groups plus their result slice copies.
        for _ in range(groups):
            record_arithmetic_dispatch('batched')
            record_arithmetic_dispatch('attention_exp_swin')
            record_arithmetic_dispatch('attention_weights')
            record_arithmetic_dispatch('batched')
        self.counts['fused_calls'] = self.counts.get('fused_calls', 0) + 1
        self.counts['logical_groups'] = self.counts.get('logical_groups', 0) + groups
        return result
