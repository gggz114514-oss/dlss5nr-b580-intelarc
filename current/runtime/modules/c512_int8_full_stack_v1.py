"""Experimental complete NR256 with frozen floor16 C512 FFNs and repaired ViT.

Only construction reconstructs packed weights from authenticated static scales.
Each eager/captured call allocates its own scratch; no microbenchmark input pointer
or intermediate is retained by the module. The five arithmetic kernels are frozen.
"""
from contextlib import contextmanager
import hashlib
import numpy as np
import torch
import triton
from triton.compiler.compiler import CompiledKernel
import nr_backend.split_block as blocks
from nr_backend.execution import record_arithmetic_dispatch
from compact_c512_qkv_stack_v1 import Stack as CompactStack, CompactLayout
from c512_window_projection_v1 import forward as project
from int8_ffn_nr_stack_v1 import CallGuard as VitGuard
from graph_front_v6 import GraphFront
from quantization_dataflow_v1 import CONTRACTS
from spill_preflight_v1 import select
import c512_int8_ffn_gpu_v1 as kernels
import c512_int8_ffn_oracle_v1 as oracle
import compressed_arrays_v1 as arrays

NAMES = ('w0', 's0', 'sz', 'we', 'se', 'sh', 'wr', 'sr', 'sg', 'wp', 'sp', 'skip')
EXTRA = {
    'c512_int8_ffn_gpu_v1._entry': (('QX', 'SX'), ()),
    'c512_int8_ffn_gpu_v1._linear': (('QZ',), ()),
    'c512_int8_ffn_gpu_v1._expand': (('QH',), ()),
    'c512_int8_ffn_gpu_v1._reduce': (('QG',), ()),
    'c512_int8_ffn_gpu_v1._project': (('OUT',), ('OUT',)),
}


def metadata(a):
    return dict(shape=list(a.shape), dtype=a.dtype.str, bytes=a.nbytes,
                raw_sha256=hashlib.sha256(a.tobytes()).hexdigest())


def reconstruct(module, frozen):
    weights = [t.cpu().numpy().copy() for t in (module.ffwd.linear,
        module.ffwd.expand, module.ffwd.reduce, module.ffwd_projection.weight,
        module.ffwd_projection.skip_scale)]
    oracle.checked_weights(weights)
    w0, we, wr, wp, skip = weights
    p = {k: arrays.load(frozen['scales'][k]) for k in ('sz', 'sh', 'sg')}
    p['skip'] = skip.copy()
    p['w0'], p['s0'] = oracle.quantize_axis(w0, 0)
    p['we'], p['se'] = oracle.quantize_axis(we.astype('f4') * p['sz'].reshape(8, 64, 1), 1)
    p['wr'], p['sr'] = oracle.quantize_axis(wr.astype('f4') * p['sh'].reshape(8, 256, 1), 1)
    p['wp'], p['sp'] = oracle.quantize_axis(wp.astype('f4') * p['sg'].reshape(512, 1), 0)
    assert {k: metadata(v) for k, v in p.items()} == frozen['packed_metadata']
    return p


class Layout(CompactLayout):
    def __init__(self, stack, calibration):
        super().__init__(stack, 32)
        assert not torch.is_inference_mode_enabled()
        self.packed, self.packed_metadata = {}, {}
        self.ffn_resources, self.ffn_selections = {}, {}
        self.ffn_calls, self.ffn_blocks, self.ffn_probe = 0, {}, None
        expected = [f'{side}512.{i}.ffn' for side in ('encoder', 'decoder') for i in range(8)]
        assert [r['name'] for r in calibration] == expected
        modules = dict(self.model.named_modules())
        for row in calibration:
            name = row['name'].removesuffix('.ffn')
            p = reconstruct(modules[name], row['variants']['floor16'])
            values = tuple(torch.from_numpy(v).to('xpu') for v in oracle.gpu_constants(p))
            self.packed[name] = values
            self.packed_metadata[name] = {k: metadata(v) for k, v in p.items()}
        self.constant_signature = self.signature()

    def signature(self):
        return tuple((id(t), t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device, t._version)
                     for values in self.packed.values() for t in values)

    def validate_constants(self):
        assert self.signature() == self.constant_signature, 'C512 constants changed'

    def ffn(self, name, features):
        assert features.shape == (12, 12, 512) and features.dtype == torch.float16
        assert features.is_contiguous() and features.device.type == 'xpu'
        x = features.view(144, 512)
        w0, s0, sz, we, se, sh, wr, sr, sg, wp, sp, skip = self.packed[name]
        qx = torch.empty((144, 512), dtype=torch.int8, device=x.device)
        sx = torch.empty(144, dtype=torch.float32, device=x.device)
        qz = torch.empty_like(qx)
        qh = torch.empty((144, 2048), dtype=torch.int8, device=x.device)
        qg = torch.empty_like(qx)
        out = torch.empty_like(x)

        def launch(label, jit, configs, args_for, grid_for):
            options = dict(num_warps=4, num_stages=1, enable_fp_fusion=False)
            cfg, kernel, choice = select(jit, configs, args_for, grid_for, **options)
            self.ffn_resources[label] = dict(hash=kernel.hash, spills=kernel.n_spills,
                registers=kernel.n_regs, shared_bytes=kernel.metadata.shared)
            self.ffn_selections[label] = choice
            assert jit[grid_for(cfg)](*args_for(cfg), **options) is kernel

        dense = [(32, 64), (16, 64), (16, 32)]
        grid = lambda c: (triton.cdiv(144, c[0]), 512 // c[1])
        # DEBUG=false pointers alias the read-only input, not any output. The
        # installed metadata guard rejects debug stores before dispatch.
        launch('entry', kernels._entry, [(1,)], lambda c: (x, qx, sx), lambda c: (144,))
        launch('linear_debugFalse', kernels._linear, dense,
            lambda c: (qx, sx, w0, s0, sz, qz, x, *c, False), grid)
        launch('expand_debugFalse', kernels._expand, dense,
            lambda c: (qz, we, se, sh, qh, x, *c, False),
            lambda c: (triton.cdiv(144, c[0]), 8, 256 // c[1]))
        launch('reduce_debugFalse', kernels._reduce, dense,
            lambda c: (qh, wr, sr, sg, qg, x, *c, False),
            lambda c: (triton.cdiv(144, c[0]), 8, 64 // c[1]))
        launch('project_debugFalse', kernels._project, dense,
            lambda c: (qg, wp, sp, x, skip, out, x, *c, False), grid)
        self.ffn_calls += 1
        self.ffn_blocks[name] = self.ffn_blocks.get(name, 0) + 1
        record_arithmetic_dispatch('c512_int8_ffn_segment')
        if self.ffn_probe is not None:
            self.ffn_probe(name, x, dict(qx=qx, sx=sx, qz=qz, qh=qh, qg=qg, out=out))
        return out.view(12, 12, 512)

    def split(self, module, features):
        assert self.probe is None and id(module) in self.modules
        name = self.modules[id(module)]
        sy, sx = module.window_shift
        mlp = self.ffn(name, features)
        packed = self.windows(module.attention, mlp, (sy, sx))
        full, kernel, selection = project(packed, module.projection.weight, blocks.q(mlp),
            module.projection.skip_scale, module.attention.pixel_inverse, shift=(sy, sx))
        self.resources[kernel.hash] = dict(spills=kernel.n_spills, registers=kernel.n_regs,
                                          shared_bytes=kernel.metadata.shared)
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
        self.validate_constants()
        assert not any(k in CONTRACTS for k in EXTRA)
        original = CompiledKernel.launch_metadata
        def checked(kernel, grid, stream, *args):
            jit = kernel.src.fn
            name = f'{jit.fn.__module__}.{jit.fn.__name__}'
            if name in EXTRA:
                bound = dict(zip(jit.arg_names, args))
                if 'DEBUG' in bound:
                    assert bound['DEBUG'] is False
            return original(kernel, grid, stream, *args)
        CONTRACTS.update(EXTRA)
        CompiledKernel.launch_metadata = checked
        try:
            with super().installed():
                yield self
        finally:
            valid = CompiledKernel.launch_metadata is checked
            CompiledKernel.launch_metadata = original
            for k, v in EXTRA.items():
                assert CONTRACTS.pop(k) == v
            assert valid, 'C512 launch guard interference'
            self.validate_constants()

    def verify_restored(self):
        super().verify_restored()
        assert not any(k in CONTRACTS for k in EXTRA)
        self.validate_constants()


class CallGuard(VitGuard):
    def validate(self):
        super().validate()
        scope = self.stack.c512_int8
        assert self.stack.rewrite.layout is scope and scope.enabled
        scope.validate_constants()
        for row, values in zip(self.model._c512_int8_buffers, scope.packed.values()):
            assert all(getattr(row, key) is t for key, t in zip(NAMES, values))


class Stack(CompactStack):
    def __init__(self, exact_root, *, hidden_scales, calibration, share_with=None):
        super().__init__(exact_root, hidden_scales=hidden_scales, share_with=share_with, bm=32)
        old_graph, old_guard = self.graph, self.call_guard
        assert not old_graph.entries and old_graph.replays == 0 and old_graph._graph_cache is None
        scope = Layout(self, calibration)
        registry = torch.nn.ModuleList()
        for values in scope.packed.values():
            row = torch.nn.Module()
            for key, tensor in zip(NAMES, values):
                row.register_buffer(key, tensor, persistent=False)
            registry.append(row)
        assert not hasattr(self.model, '_c512_int8_buffers')
        self.model.add_module('_c512_int8_buffers', registry)
        old_graph.close()
        self.graph = GraphFront(self.model, arithmetic=self.provider)
        self.components[self.components.index(old_graph)] = self.graph
        self.c512_int8 = scope
        self.rewrite.layout = scope
        self.call_guard = CallGuard(self)
        self.components[self.components.index(old_guard)] = self.call_guard
        assert len([n for n, *_ in self.graph.constants if n.startswith('_c512_int8_buffers.')]) == 192
        assert len(self.graph.constants) == 1011

    def metadata(self):
        scope = self.c512_int8
        return dict(super().metadata(), c512_int8=dict(policy='frozen_floor16_split',
            registered_constants=192, blocks=scope.ffn_blocks, calls=scope.ffn_calls,
            resources=scope.ffn_resources, selections=scope.ffn_selections,
            reconstructed_pack_metadata=scope.packed_metadata, runtime_calibration=False,
            per_invocation_scratch=True, arithmetic_kernels_unchanged=True))
