"""Measured per-channel streaming pair configurations; tiny rows stay original.

1024 is an experimental conservative cutoff, not a measured optimum. Whole-model
benchmarks determine whether these choices help at the actual input sizes.
"""
from fused_branched_pairs_v1 import FusedPairs as Previous, forward
from nr_backend.execution import current_arithmetic_backend


class FusedPairs(Previous):
    def apply(self, module, features):
        if (id(module) not in self.modules or current_arithmetic_backend() != 'triton'
                or features.device.type != 'xpu' or self.provider.mode != 'fp16_xmx'
                or features.numel() // module.channels < 1024):
            return self.original(module, features)
        c = module.channels
        value, _ = forward(features, module.expand, module.reduce, module.project,
                           module.skip_scale, bm=16, warps=4, stages=2 if c == 256 else 1)
        self.calls[str(c)] = self.calls.get(str(c), 0) + 1
        return value
