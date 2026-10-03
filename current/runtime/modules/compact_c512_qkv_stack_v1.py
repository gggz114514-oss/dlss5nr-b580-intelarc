"""C512 QKV projection on valid pixels with full-window attention preserved."""
from contextlib import contextmanager
import numpy as np
import torch
import triton
import nr_backend.split_block as blocks
from nr_backend.execution import record_arithmetic_dispatch
from fast_matrices_v3 import _matmul
from spill_preflight_v1 import select
from c512_window_layout_scope_v1 import C512WindowLayout
from c512_window_projection_v1 import forward as project
from window_block_attention_v3 import forward as attend
from compact_c512_qkv_pack_v1 import _pack
from quantization_dataflow_v1 import CONTRACTS
from int8_ffn_calibrated_stack_v1 import Stack as RepairedStack

EXTRA = {'compact_c512_qkv_pack_v1._pack': (('Q', 'K', 'V'), ('Q', 'K', 'V'))}


class CompactLayout(C512WindowLayout):
    def __init__(self, stack, bm):
        super().__init__(stack)
        assert bm in (32, 64)
        self.bm = bm
        self.compact_resources, self.compact_selections = {}, {}
        self.compact_calls = 0
        self.qkv_probe = None
        # Padded zero rows must multiply finite fixed weights. Do not shortcut
        # the normalization/scale of zero QKV; signed zeros follow the old pack.
        for group in (self.model.encoder512, self.model.decoder512):
            for block in group:
                w = block.attention.qkv
                assert tuple(w.shape) == (512, 1536) and w.dtype == torch.float16
                assert np.isfinite(w.cpu().numpy()).all()

    def launch(self, label, jit, configs, args_for, grid_for, **options):
        cfg, kernel, decision = select(jit, configs, args_for, grid_for, **options)
        self.compact_resources[label + ':' + kernel.hash] = dict(spills=kernel.n_spills,
            registers=kernel.n_regs, shared_bytes=kernel.metadata.shared)
        self.compact_selections[label] = decision
        assert jit[grid_for(cfg)](*args_for(cfg), **options) is kernel
        return kernel

    def windows(self, module, features, shift):
        assert tuple(features.shape) == (12, 12, 512)
        assert features.is_contiguous() and features.dtype == torch.float16 and features.device.type == 'xpu'
        assert tuple(module.qkv.shape) == (512, 1536) and module.qkv.is_contiguous()
        assert module.qkv.device == features.device and module.scale.shape == (16,)
        assert module.pixel_order.shape == (64,) and module.pixel_order.dtype == torch.int64
        sy, sx = shift
        assert sy in (0, 4) and sx in (0, 4)
        x = blocks.q(features)
        z = torch.empty((12, 12, 16, 3, 32), dtype=x.dtype, device=x.device)
        configs = [(64, 64), (32, 64), (16, 64)] if self.bm == 64 else [(32, 64), (16, 64)]
        # Reuse the actual selected FP16 arithmetic kernel, same BK32 and options.
        self.launch('dense', _matmul, configs,
            lambda c: (x, module.qkv, x, x, x, z, 144, 1536, 512, False, False, False, *c, 32),
            lambda c: (triton.cdiv(144, c[0]), triton.cdiv(1536, c[1]), 1),
            num_warps=4, enable_fp_fusion=False)
        self.provider.record('fp16_dense')
        record_arithmetic_dispatch('dense')
        shape = (16, 2, 2, 64, 32)
        q, k, v = [torch.empty(shape, dtype=x.dtype, device=x.device) for _ in range(3)]
        br = self.stack.window_blocks.layout.rows
        assert br in (8, 16, 32, 64)
        self.launch('pack_' + str(shift), _pack, [(br,)] if br == 8 else [(br,), (8,)],
            lambda c: (z, module.scale, module.pixel_order, q, k, v, sy, sx, c[0]),
            lambda c: (triton.cdiv(4096, c[0]), 3),
            num_warps=4, enable_fp_fusion=False)
        for _ in range(2):
            record_arithmetic_dispatch('attention_normalize_c32')
        for _ in range(3):
            record_arithmetic_dispatch('fp8')
        if self.qkv_probe is not None:
            self.qkv_probe(module, features, shift, (q, k, v))
        result, self.stack.window_blocks.last_attention_kernel, self.stack.window_blocks.last_attention_selection = attend(q, k, v, module.bias)
        for _ in range(module.heads):
            for key in ('batched', 'attention_exp_swin', 'attention_weights', 'batched'):
                record_arithmetic_dispatch(key)
        for key, n in (('multi', 1), ('batched_heads', module.heads), ('qkv_pack', 1)):
            self.stack.window_blocks.layout.calls[key] = self.stack.window_blocks.layout.calls.get(key, 0) + n
        self.compact_calls += 1
        return result

    def split(self, module, features):
        if self.probe is not None:
            return super().split(module, features)
        assert id(module) in self.modules and tuple(features.shape) == (12, 12, 512)
        sy, sx = module.window_shift
        ffwd = module.ffwd(features)
        mlp = module.ffwd_projection(ffwd, features)
        packed = self.windows(module.attention, mlp, (sy, sx))
        full, kernel, selection = project(packed, module.projection.weight, blocks.q(mlp),
            module.projection.skip_scale, module.attention.pixel_inverse, shift=(sy, sx))
        self.resources[kernel.hash] = dict(spills=kernel.n_spills, registers=kernel.n_regs,
                                          shared_bytes=kernel.metadata.shared)
        name = self.modules[id(module)]
        self.selections[name] = selection
        self.provider.record('fp16_dense')
        record_arithmetic_dispatch('dense')
        for _ in range(2):
            record_arithmetic_dispatch('fp8')
        output = blocks.q(full)
        final = None
        if module.final_weight is not None:
            top = (full[0::2, 0::2] + full[0::2, 1::2]).half()
            bottom = (full[1::2, 0::2] + full[1::2, 1::2]).half()
            pool = blocks.q(((top + bottom).half() * .25).half())
            pool = torch.nn.functional.pad(pool, (0, 0, 0, (-pool.shape[1]) % 4, 0, (-pool.shape[0]) % 4))
            final = blocks.q(blocks.dot(pool, module.final_weight, chunk_k=16))
        self.calls += 1
        self.blocks[name] = self.blocks.get(name, 0) + 1
        return output, output if final is None else final

    @contextmanager
    def installed(self):
        assert not any(k in CONTRACTS for k in EXTRA)
        CONTRACTS.update(EXTRA)
        try:
            with super().installed():
                yield self
        finally:
            for k, v in EXTRA.items():
                assert CONTRACTS.pop(k) == v

    def verify_restored(self):
        super().verify_restored()
        assert not any(k in CONTRACTS for k in EXTRA)


class Stack(RepairedStack):
    def __init__(self, exact_root, *, hidden_scales, share_with=None, bm=64):
        super().__init__(exact_root, hidden_scales=hidden_scales, share_with=share_with)
        assert not self.graph.entries and self.rewrite.layout.calls == 0
        self.rewrite.layout = CompactLayout(self, bm)

    def metadata(self):
        layout = self.rewrite.layout
        return dict(super().metadata(), range_calibration_unchanged=True,
            compact_qkv=dict(requested_bm=layout.bm, resources=layout.compact_resources,
                selections=layout.compact_selections, calls=layout.compact_calls,
                dense_rows=144, original_dense_rows=256, full_window_shape=[16, 2, 2, 64, 32],
                finite_qkv_weights_verified=True, full_keys_and_values_preserved=True,
                pooling_unquantized=True, new_weight_buffers=False))
