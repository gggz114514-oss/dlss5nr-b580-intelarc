"""Owned W8A8 QKV on the frozen floor16/native-query NR256 stack.

The three arithmetic kernels and 16x64 tile match the isolated idle screen.
Register weight/scale buffers before any graph capture; preserve both CPU guards.
"""
from contextlib import contextmanager
import torch
import triton
import nr_backend.split_block as blocks
from nr_backend.execution import record_arithmetic_dispatch
import compact_c512_qkv_stack_v1 as compact
from c512_quad_query_stack_v1 import Stack as QuadStack
from c512_int8_full_stack_v1 import CallGuard as BaseGuard
from graph_front_v6 import GraphFront
from quantization_dataflow_v1 import CONTRACTS
import c512_qkv_int8_jobs_v1 as frozen

EXTRA = {'fast_matrices_v3._quantize_rows': (('Q', 'SCALE'), ())}
NAMES = ('qw', 'sw')
SCREEN_HASHES = {
    'int8:quantize_rows': '93ad46fab67572f2f0e77db1eb0362c267cfa3854395c35e2d79f7897a8de52e',
    'int8:dense(16, 64)': 'dbbace663f5d27035656af90ee50a04d57a4fe277db87d66af888ac58f1f803b',
    'qkv:pack_(0, 0)': '361c1d360996593612d295698f527a69ab180d6968081032e90f409e2b6b9661',
    'qkv:pack_(0, 4)': '27c54e18c78146c1f0b753db031030eb7ef0aee1b489955228e1eb87768943b1',
    'qkv:pack_(4, 0)': '575ec49e35e81f98d8d7cdb914be853901485a59b7c3ea84d9c9a1c54c8f07e3',
    'qkv:pack_(4, 4)': '7ce50afb3792dc4514bed8d70b8427c7ee89d6fdf67f69f9efe1c8c815bf4945',
}


class QKV:
    def __init__(self, stack):
        self.stack, self.layout = stack, stack.c512_int8
        assert not torch.is_inference_mode_enabled()
        self.original = self.layout.windows
        self.modules = {id(m.attention): (name, m) for name, m in stack.model.named_modules()
                        if id(m) in self.layout.modules}
        assert len(self.modules) == 16
        self.packed = {name: frozen.prepare_weight(dict(module=m)) for name, m in self.modules.values()}
        self.constant_signature = self.signature()
        self.calls, self.block_calls, self.resources = 0, {}, {}
        self.probe = None

    def signature(self):
        return tuple((id(t), t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device, t._version)
                     for pack in self.packed.values() for t in (pack['qw'], pack['sw']))

    def validate_constants(self):
        assert self.signature() == self.constant_signature, 'QKV INT8 constants changed'
        registry = self.stack.model._c512_qkv_int8_buffers
        assert len(registry) == len(self.packed) == 16
        for row, pack in zip(registry, self.packed.values()):
            assert all(getattr(row, key) is pack[key] for key in NAMES), 'QKV registered buffer replaced'

    def launch(self, label, jit, args, grid):
        options = dict(num_warps=4, enable_fp_fusion=False)
        kernel = jit.warmup(*args, grid=grid, **options)
        kernel._init_handles()
        assert kernel.n_spills == 0, (label, kernel.n_spills)
        assert kernel.hash == SCREEN_HASHES[label], (label, 'screened binary changed')
        meta = dict(hash=kernel.hash, spills=kernel.n_spills,
                    registers=kernel.n_regs, shared_bytes=kernel.metadata.shared)
        if label in self.resources:
            assert self.resources[label] == meta
        self.resources[label] = meta
        assert jit[grid](*args, **options) is kernel

    def windows(self, attention, features, shift):
        assert id(attention) in self.modules
        name, module = self.modules[id(attention)]
        assert shift == module.window_shift and tuple(features.shape) == (12, 12, 512)
        assert features.dtype == torch.float16 and features.device.type == 'xpu' and features.is_contiguous()
        pack = self.packed[name]
        x = blocks.q(features).view(144, 512)
        qx = torch.empty_like(x, dtype=torch.int8)
        sx = torch.empty(144, dtype=torch.float32, device=x.device)
        z = torch.empty((12, 12, 16, 3, 32), dtype=x.dtype, device=x.device)
        out = tuple(torch.empty((16, 2, 2, 64, 32), dtype=x.dtype, device=x.device) for _ in range(3))
        self.launch('int8:quantize_rows', frozen._quantize_rows,
                    (x, qx, sx, 144, 512, 512, 1, 512), (144,))
        self.launch('int8:dense(16, 64)', frozen._matmul,
                    (qx, pack['qw'], sx, pack['sw'], qx, z, 144, 1536, 512,
                     False, False, True, 16, 64, 32), (triton.cdiv(144, 16), 24, 1))
        self.launch('qkv:pack_'+str(shift), frozen._pack,
                    (z, attention.scale, attention.pixel_order, *out, *shift, 16), (256, 3))
        self.stack.provider.record('int8_dense')
        record_arithmetic_dispatch('dense')
        record_arithmetic_dispatch('c512_int8_qkv')
        for _ in range(2):
            record_arithmetic_dispatch('attention_normalize_c32')
        for _ in range(3):
            record_arithmetic_dispatch('fp8')
        if self.probe is not None:
            self.probe(name, module, x, pack, qx, sx, z, out)
        if self.layout.qkv_probe is not None:
            self.layout.qkv_probe(attention, features, shift, out)
        # Use the module lookup: CompactQueries installs its native-query attend
        # callable here for this active block; importing the old callable would bypass it.
        result, self.stack.window_blocks.last_attention_kernel, self.stack.window_blocks.last_attention_selection = compact.attend(*out, attention.bias)
        for _ in range(attention.heads):
            for key in ('batched', 'attention_exp_swin', 'attention_weights', 'batched'):
                record_arithmetic_dispatch(key)
        for key, n in (('multi', 1), ('batched_heads', attention.heads), ('qkv_pack', 1)):
            calls = self.stack.window_blocks.layout.calls
            calls[key] = calls.get(key, 0)+n
        self.layout.compact_calls += 1
        self.calls += 1
        self.block_calls[name] = self.block_calls.get(name, 0)+1
        return result

    @contextmanager
    def installed(self):
        self.validate_constants()
        assert 'windows' not in self.layout.__dict__ and self.layout.windows == self.original
        assert not any(k in CONTRACTS for k in EXTRA)
        replacement = self.windows
        self.layout.windows = replacement
        CONTRACTS.update(EXTRA)
        try:
            yield self
        finally:
            valid = self.layout.windows is replacement
            del self.layout.windows
            for key, value in EXTRA.items():
                assert CONTRACTS.pop(key) == value
            assert valid, 'QKV scope interference'
            self.validate_constants()

    def verify_restored(self):
        assert 'windows' not in self.layout.__dict__ and self.layout.windows == self.original
        assert not any(k in CONTRACTS for k in EXTRA)
        self.validate_constants()


class CallGuard(BaseGuard):
    def validate(self):
        super().validate()
        self.stack.qkv_int8.validate_constants()


class Stack(QuadStack):
    def __init__(self, exact_root, **kwargs):
        super().__init__(exact_root, **kwargs)
        old_graph, old_guard = self.graph, self.call_guard
        assert not old_graph.entries and old_graph.replays == 0 and old_graph._graph_cache is None
        self.qkv_int8 = QKV(self)
        registry = torch.nn.ModuleList()
        for pack in self.qkv_int8.packed.values():
            row = torch.nn.Module()
            for key in NAMES:
                row.register_buffer(key, pack[key], persistent=False)
            registry.append(row)
        assert not hasattr(self.model, '_c512_qkv_int8_buffers')
        self.model.add_module('_c512_qkv_int8_buffers', registry)
        old_graph.close()
        self.graph = GraphFront(self.model, arithmetic=self.provider)
        self.components[self.components.index(old_graph)] = self.graph
        self.call_guard = CallGuard(self)
        self.components[self.components.index(old_guard)] = self.call_guard
        self.components.append(self.qkv_int8)
        assert len(self.graph.constants) == 1043
        assert len([n for n, *_ in self.graph.constants if n.startswith('_c512_qkv_int8_buffers.')]) == 32

    def metadata(self):
        return dict(super().metadata(), qkv_int8=dict(config=[16, 64], registered_constants=32,
            calls=self.qkv_int8.calls, blocks=self.qkv_int8.block_calls, resources=self.qkv_int8.resources,
            input_quantization_per_replay=True, weight_packing_at_construction=True,
            fp8_boundaries_preserved=True, per_invocation_scratch=True))

    def close(self):
        self.qkv_int8.verify_restored()
        super().close()
