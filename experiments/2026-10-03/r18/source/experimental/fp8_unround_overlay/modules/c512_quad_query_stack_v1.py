"""Scoped frozen native-quadrant attention/projection on accepted floor16.

Keep the existing Layout and both CPU guards. Reuse its original split/windows
Python bodies, including unquantized pooling; replace two producer/consumer
callables only for the owned active C512 block. No public boundary API changes.
"""
from contextlib import contextmanager
import nr_backend.split_block as blocks
import compact_c512_qkv_stack_v1 as compact
import c512_int8_full_stack_v1 as full
import c512_quad_dispatch_v1 as kernels
import hashlib
import torch
from quantization_dataflow_v1 import CONTRACTS

EXTRA = {'c512_quad_queries_v2._attend': (('OUT',), ('OUT',)),
         'c512_quad_queries_v2._project': (('OUT',), ())}


class CompactQueries:
    def __init__(self, stack):
        self.stack, self.layout = stack, stack.c512_int8
        # No new model buffers. Existing graph constants already protect these
        # exact tensors on every frame; the value check is construction-only.
        order = [base+y*8+x for base in (0, 4, 32, 36) for y in range(4) for x in range(4)]
        inverse = [order.index(i) for i in range(64)]
        self.permutations = {}
        for name, module in stack.model.named_modules():
            if id(module) not in self.layout.modules:
                continue
            values = {}
            for key, expected in (('pixel_order', order), ('pixel_inverse', inverse)):
                t = getattr(module.attention, key)
                assert t.shape == (64,) and t.dtype == torch.int64 and t.is_contiguous()
                assert t.device.type == 'xpu' and t.cpu().tolist() == expected, (name, key)
                values[key] = hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()
                assert any(n == name+'.attention.'+key and identity == id(t)
                           for n, identity, *_ in stack.graph.constants)
            self.permutations[name] = values
        assert set(self.permutations) == set(self.layout.modules.values()) and len(self.permutations) == 16
        self.original_split = self.layout.split
        self.original_attend, self.original_project = compact.attend, full.project
        self.active, self.probe = None, None
        self.calls, self.blocks, self.resources, self.selections = 0, {}, {}, {}

    def record(self, label, kernel, decision):
        self.resources[label+':'+kernel.hash] = dict(spills=kernel.n_spills,
            registers=kernel.n_regs, shared_bytes=kernel.metadata.shared)
        self.selections[label] = decision

    def split(self, module, features):
        assert self.active is None and id(module) in self.layout.modules
        assert tuple(features.shape) == (12, 12, 512)
        self.active = module
        try:
            result = self.original_split(module, features)
            name = self.layout.modules[id(module)]
            self.calls += 1
            self.blocks[name] = self.blocks.get(name, 0)+1
            return result
        finally:
            self.active = None

    def attend(self, q, k, v, bias):
        module = self.active
        assert module is not None and bias is module.attention.bias
        result, kernel, choice = kernels.attend(q, k, v, bias,
            module.attention.pixel_inverse, shift=module.window_shift)
        self.record('attention_'+str(module.window_shift), kernel, choice)
        if self.probe is not None:
            self.probe('attention', module, (q, k, v, bias), result)
        return result, kernel, choice

    def project(self, packed, weight, residual, scale, inverse, *, shift):
        module = self.active
        assert module is not None and weight is module.projection.weight
        assert scale is module.projection.skip_scale and inverse is module.attention.pixel_inverse
        assert shift == module.window_shift
        result, kernel, choice = kernels.project(packed, weight, residual, scale)
        self.record('projection', kernel, choice)
        if self.probe is not None:
            self.probe('projection', module, (packed, weight, residual, scale, inverse), result)
        return result, kernel, choice

    @contextmanager
    def installed(self):
        assert 'split' not in self.layout.__dict__ and self.layout.split == self.original_split
        assert compact.attend is self.original_attend and full.project is self.original_project
        assert self.active is None and not any(k in CONTRACTS for k in EXTRA)
        methods = blocks.SplitSwinBlock.forward, blocks.SplitSwinBlock.forward_boundaries
        split, attend, project = self.split, self.attend, self.project
        self.layout.split, compact.attend, full.project = split, attend, project
        CONTRACTS.update(EXTRA)
        try:
            yield self
        finally:
            valid = (self.layout.split is split and compact.attend is attend and full.project is project
                and self.active is None and methods == (blocks.SplitSwinBlock.forward, blocks.SplitSwinBlock.forward_boundaries))
            del self.layout.split
            compact.attend, full.project = self.original_attend, self.original_project
            for key, contract in EXTRA.items():
                assert CONTRACTS.pop(key) == contract
            assert valid, 'Compact-query scope interference'

    def verify_restored(self):
        assert self.active is None and 'split' not in self.layout.__dict__
        assert compact.attend is self.original_attend and full.project is self.original_project
        assert not any(k in CONTRACTS for k in EXTRA)


class Stack(full.Stack):
    def __init__(self, exact_root, **kwargs):
        super().__init__(exact_root, **kwargs)
        self.compact_queries = CompactQueries(self)
        self.components.append(self.compact_queries)

    def close(self):
        self.compact_queries.verify_restored()
        super().close()
