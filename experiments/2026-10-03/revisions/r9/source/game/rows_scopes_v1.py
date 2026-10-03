"""Full-row replacements for the two reviewed fixed-row INT8 FFN calls.

Only four things change relative to the reviewed path: the total row count, the
tail mask, the scratch buffer shapes and the launch schedule. BM/BN/BK, warps,
stages, fusion options, packed weights, scales, K accumulation, rounding, the
cubic activation, FP8 boundaries and residuals are taken from the reviewed scope
objects unchanged. No attention, QKV, graph, rewrite or capture change is made.
"""
from contextlib import contextmanager
import torch
import triton
from nr_backend.execution import record_arithmetic_dispatch
from spill_preflight_v1 import select
from quantization_dataflow_v1 import CONTRACTS
import c512_int8_ffn_rows_v1 as c512_kernels
import int8_ffn_segment_rows_v1 as vit_kernels
from nr_backend.unround_policy import ENABLED
from rows_paths_v1 import OLD_KERNEL_MODULES, ROWS_KERNEL_MODULES

ROWS_EXTRA = {**c512_kernels.EXTRA, **vit_kernels.EXTRA}


def c512_ffn(stack, name, features, sink=None):
    """One INT8 C512 FFN over every row, replacing 144-row padded batches.

    Mirrors c512_int8_full_stack_v1.Layout.ffn step for step; the only difference
    is that rows is the real row count instead of the literal 144, so there is no
    pad/cat batching and the kernels write the complete output directly.
    """
    scope = stack.c512_int8
    assert features.dtype == torch.float16 and features.is_contiguous()
    assert features.device.type == 'xpu' and features.ndim == 3 and features.shape[2] == 512
    rows = int(features.shape[0]) * int(features.shape[1])
    x = features.view(rows, 512)
    w0, s0, sz, we, se, sh, wr, sr, sg, wp, sp, skip = scope.packed[name]
    qx = torch.empty((rows, 512), dtype=torch.int8, device=x.device)
    sx = torch.empty(rows, dtype=torch.float32, device=x.device)
    qz = torch.empty_like(qx)
    qh = torch.empty((rows, 2048), dtype=torch.int8, device=x.device)
    qg = torch.empty_like(qx)
    out = torch.empty_like(x)

    def launch(label, jit, configs, args_for, grid_for):
        options = dict(num_warps=4, num_stages=1, enable_fp_fusion=False)
        config, kernel, choice = select(jit, configs, args_for, grid_for, **options)
        scope.ffn_resources[label] = dict(hash=kernel.hash, spills=kernel.n_spills,
            registers=kernel.n_regs, shared_bytes=kernel.metadata.shared)
        scope.ffn_selections[label] = choice
        assert jit[grid_for(config)](*args_for(config), **options) is kernel

    dense = list(c512_kernels.DENSE_CONFIGS)
    grid = lambda c: (triton.cdiv(rows, c[0]), 512 // c[1])
    launch('entry', c512_kernels._entry, [(1,)],
        lambda c: (x, qx, sx, rows, 'c512' not in ENABLED), lambda c: (rows,))
    launch('linear_debugFalse', c512_kernels._linear, dense,
        lambda c: (qx, sx, w0, s0, sz, qz, x, rows, *c, False), grid)
    launch('expand_debugFalse', c512_kernels._expand, dense,
        lambda c: (qz, we, se, sh, qh, x, rows, *c, False),
        lambda c: (triton.cdiv(rows, c[0]), 8, 256 // c[1]))
    launch('reduce_debugFalse', c512_kernels._reduce, dense,
        lambda c: (qh, wr, sr, sg, qg, x, rows, *c, False),
        lambda c: (triton.cdiv(rows, c[0]), 8, 64 // c[1]))
    launch('project_debugFalse', c512_kernels._project, dense,
        lambda c: (qg, wp, sp, x, skip, out, x, rows, *c, False, 'c512' not in ENABLED), grid)
    scope.ffn_calls += 1
    scope.ffn_blocks[name] = scope.ffn_blocks.get(name, 0) + 1
    record_arithmetic_dispatch('c512_int8_ffn_segment')
    if sink is not None:
        sink(name, out)
    if scope.ffn_probe is not None:
        scope.ffn_probe(name, x, dict(qx=qx, sx=sx, qz=qz, qh=qh, qg=qg, out=out))
    return out.view(features.shape)


def vit_ffn(stack, index, x, sink=None):
    """One INT8 ViT FFN over every token, replacing 64-row padded batches.

    Mirrors int8_ffn_body_scope_v1.Int8VitLayout.ffn. That scope already passes M
    as a constexpr to every kernel; only its wrapper fixed the row count at 64.
    """
    scope = stack.int8_vit
    assert x.dtype == torch.float16 and x.is_contiguous() and x.device.type == 'xpu'
    assert x.ndim == 2 and x.shape[1] == 1024
    rows = int(x.shape[0])
    we, se, sh, wc, sc, skip = scope.packed[index]
    qx = torch.empty((rows, 1024), dtype=torch.int8, device=x.device)
    sx = torch.empty(rows, dtype=torch.float32, device=x.device)
    qh = torch.empty((rows, 4096), dtype=torch.int8, device=x.device)
    partial = torch.empty((4, rows, 1024), dtype=torch.int32, device=x.device)
    out = torch.empty_like(x)

    def launch(label, jit, configs, args_for, grid_for):
        options = dict(num_warps=4, num_stages=1, enable_fp_fusion=False)
        config, kernel, choice = select(jit, configs, args_for, grid_for, **options)
        scope.ffn_resources[label + ':' + kernel.hash] = dict(spills=kernel.n_spills,
            registers=kernel.n_regs, shared_bytes=kernel.metadata.shared)
        scope.selections[label] = choice
        assert jit[grid_for(config)](*args_for(config), **options) is kernel

    launch('entry', vit_kernels._entry, [(1,)],
        lambda c: (x, qx, sx, rows, 1024), lambda c: (rows,))
    launch('expand', vit_kernels._expand, list(vit_kernels.EXPAND_CONFIGS),
        lambda c: (qx, sx, we, se, sh, qh, x, rows, 1024, 4096, *c, False),
        lambda c: (triton.cdiv(rows, c[0]), triton.cdiv(4096, c[1])))
    launch('contract_p4', vit_kernels._contract, list(vit_kernels.CONTRACT_CONFIGS),
        lambda c: (qh, wc, sc, x, skip, partial, out, x, rows, 4096, 1024, 4, *c, False,
                   'vit' not in ENABLED),
        lambda c: (triton.cdiv(rows, c[0]), triton.cdiv(1024, c[1]), 4))
    launch('merge', vit_kernels._merge, list(vit_kernels.MERGE_CONFIGS),
        lambda c: (partial, sc, x, skip, out, x, rows, 1024, False, 'vit' not in ENABLED),
        lambda c: (triton.cdiv(rows * 1024, 512),))
    if sink is not None:
        sink(index, out)
    return out, qh


def install(stack, variant, c512_sink=None, vit_sink=None):
    """Point the session's two FFN callables at the full-row implementations.

    baseline keeps both reviewed callables untouched.
    """
    if variant in ('c512', 'both'):
        stack.c512_int8.ffn = lambda name, features: c512_ffn(stack, name, features, c512_sink)
    if variant in ('vit', 'both'):
        stack.int8_vit.ffn = lambda index, x: vit_ffn(stack, index, x, vit_sink)


class FfnCounters:
    """Count FFN invocations at the callable boundary.

    One extra Python call per FFN invocation, so the cost is identical for the
    reviewed and the candidate schedule and small enough to leave in the timed
    loop. This is what proves which schedule actually ran while timing.
    """

    def __init__(self, stack):
        self.stack = stack
        self.c512 = 0
        self.vit = 0

    def attach(self):
        scope = self.stack.c512_int8
        inner_c512 = scope.ffn

        def counted_c512(name, features):
            self.c512 += 1
            return inner_c512(name, features)

        scope.ffn = counted_c512
        scope_vit = self.stack.int8_vit
        inner_vit = scope_vit.ffn

        def counted_vit(index, x):
            self.vit += 1
            return inner_vit(index, x)

        scope_vit.ffn = counted_vit
        return self

    def reset(self):
        self.c512 = 0
        self.vit = 0

    def totals(self):
        return dict(c512=self.c512, vit=self.vit)


# Which FFN family each kernel module belongs to. Needed because the c512-only
# and vit-only variants deliberately leave the *other* family on its reviewed
# fixed-row kernel, so "no reviewed kernel anywhere" is the wrong invariant.
MODULE_FAMILY = {'c512_int8_ffn_gpu_v1': 'c512', 'c512_int8_ffn_rows_v1': 'c512',
                 'int8_ffn_segment_gpu_v1': 'vit', 'int8_ffn_segment_rows_v1': 'vit'}


def replaced_families(variant):
    """Which reviewed fixed-row FFN callables this variant replaces."""
    return dict(c512=variant in ('c512', 'both'), vit=variant in ('vit', 'both'))


def expected_ffn_calls(variant):
    """Per-frame FFN invocation counts at 864x480 (16 C512 blocks, 8 ViT blocks)."""
    return dict(c512=16 if variant in ('c512', 'both') else 64,
                vit=8 if variant in ('vit', 'both') else 16)


@contextmanager
def dispatch_guard(variant):
    """Register candidate contracts and observe which kernels were dispatched.

    Dataflow.launch already fails closed on unregistered kernels. This guard adds
    what that layer cannot check, per FFN family:

      * the reviewed fixed-row kernel of a family this variant replaces must not
        be dispatched at all;
      * the candidate kernel of a family this variant keeps on the reviewed
        schedule must not be dispatched either (the reverse mistake);
      * DEBUG stores were never requested, and the real constexpr row counts.

    It takes the variant name rather than a boolean because c512-only and
    vit-only each replace exactly one family and must keep the other one.
    Yields ``{'kernels': {name: count}, 'rows': {module: {rows: count}}}``.
    """
    from triton.compiler.compiler import CompiledKernel
    replaced = replaced_families(variant)
    candidate = any(replaced.values())
    if candidate:
        assert not any(key in CONTRACTS for key in ROWS_EXTRA)
    original = CompiledKernel.launch_metadata
    seen = dict(kernels={}, rows={})

    def checked(kernel, grid, stream, *args):
        jit = kernel.src.fn
        name = f'{jit.fn.__module__}.{jit.fn.__name__}'
        module = name.split('.')[0]
        family = MODULE_FAMILY.get(module)
        if family is not None:
            reviewed = module in OLD_KERNEL_MODULES
            if reviewed and replaced[family]:
                raise RuntimeError(
                    f'reviewed fixed-row {family} kernel dispatched under variant '
                    f'{variant}, which replaces it: {name}')
            if not reviewed and not replaced[family]:
                raise RuntimeError(
                    f'candidate {family} kernel dispatched under variant {variant}, '
                    f'which keeps the reviewed fixed-row {family} schedule: {name}')
            seen['kernels'][name] = seen['kernels'].get(name, 0) + 1
        if candidate and name in ROWS_EXTRA:
            bound = dict(zip(jit.arg_names, args))
            if 'DEBUG' in bound:
                assert bound['DEBUG'] is False, (name, 'debug store requested')
            if 'PARTS' in bound:
                assert bound['PARTS'] == 4, (name, bound['PARTS'])
            if 'M' in bound:
                rows = int(bound['M'])
                bucket = seen['rows'].setdefault(module, {})
                bucket[rows] = bucket.get(rows, 0) + 1
        return original(kernel, grid, stream, *args)

    if candidate:
        CONTRACTS.update(ROWS_EXTRA)
    CompiledKernel.launch_metadata = checked
    try:
        yield seen
    finally:
        valid = CompiledKernel.launch_metadata is checked
        CompiledKernel.launch_metadata = original
        if candidate:
            for key, value in ROWS_EXTRA.items():
                assert CONTRACTS.pop(key) == value
        assert valid, 'rows dispatch guard interference'
