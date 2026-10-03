"""Describe the current C512 FFN/QKV kernels using actual selected configurations.

No new GPU arithmetic. Owned scratch matches the current full-stack bodies.
The caller compiles and checks every binary against the observed model dispatch.
"""
import torch
import triton
import c512_int8_ffn_gpu_v1 as ffn
from fast_matrices_v3 import _matmul
from compact_c512_qkv_pack_v1 import _pack


def plan(label, jit, args, grid, *, ffn_options=False):
    options = dict(num_warps=4, enable_fp_fusion=False)
    if ffn_options:
        options['num_stages'] = 1
    return dict(label=label, jit=jit, args=args, grid=grid, options=options)


def ffn_jobs(stack, name, x):
    assert x.shape == (144, 512) and x.is_contiguous() and x.dtype == torch.float16
    scope = stack.c512_int8
    w0, s0, sz, we, se, sh, wr, sr, sg, wp, sp, skip = scope.packed[name]
    qx = torch.empty_like(x, dtype=torch.int8)
    sx = torch.empty(144, device=x.device, dtype=torch.float32)
    qz = torch.empty_like(qx)
    qh = torch.empty((144, 2048), device=x.device, dtype=torch.int8)
    qg = torch.empty_like(qx)
    out = torch.empty_like(x)
    jobs = [plan('ffn:entry', ffn._entry, (x, qx, sx), (144,), ffn_options=True)]
    def config(key):
        c = tuple(scope.ffn_selections[key]['selected'])
        assert c in ((32, 64), (16, 64), (16, 32))
        return c
    c = config('linear_debugFalse')
    jobs.append(plan('ffn:linear_debugFalse', ffn._linear,
        (qx, sx, w0, s0, sz, qz, x, *c, False), (triton.cdiv(144, c[0]), 512//c[1]), ffn_options=True))
    c = config('expand_debugFalse')
    jobs.append(plan('ffn:expand_debugFalse', ffn._expand,
        (qz, we, se, sh, qh, x, *c, False), (triton.cdiv(144, c[0]), 8, 256//c[1]), ffn_options=True))
    c = config('reduce_debugFalse')
    jobs.append(plan('ffn:reduce_debugFalse', ffn._reduce,
        (qh, wr, sr, sg, qg, x, *c, False), (triton.cdiv(144, c[0]), 8, 64//c[1]), ffn_options=True))
    c = config('project_debugFalse')
    jobs.append(plan('ffn:project_debugFalse', ffn._project,
        (qg, wp, sp, x, skip, out, x, *c, False), (triton.cdiv(144, c[0]), 512//c[1]), ffn_options=True))
    return jobs, out


def qkv_jobs(stack, fixture, x):
    assert x.shape == (144, 512) and x.is_contiguous() and x.dtype == torch.float16
    module, shift = fixture['module'].attention, fixture['module'].window_shift
    # x is the already-FP8 complete FFN output. No extra blocks.q call/copy is
    # inserted into this diagnostic; the full graph proves that elision.
    z = torch.empty((12, 12, 16, 3, 32), device=x.device, dtype=x.dtype)
    q, k, v = [torch.empty((16, 2, 2, 64, 32), device=x.device, dtype=x.dtype) for _ in range(3)]
    c = tuple(stack.c512_int8.compact_selections['dense']['selected'])
    assert c in ((32, 64), (16, 64))
    jobs = [plan('qkv:dense', _matmul,
        (x, module.qkv, x, x, x, z, 144, 1536, 512, False, False, False, *c, 32),
        (triton.cdiv(144, c[0]), triton.cdiv(1536, c[1]), 1))]
    label = 'pack_'+str(shift)
    c = tuple(stack.c512_int8.compact_selections[label]['selected'])
    assert len(c) == 1 and c[0] in (8, 16, 32, 64)
    jobs.append(plan('qkv:'+label, _pack,
        (z, module.scale, module.pixel_order, q, k, v, *shift, c[0]), (triton.cdiv(4096, c[0]), 3)))
    return jobs, (q, k, v)


def build(stack, fixtures, stage):
    assert stage in ('ffn', 'qkv', 'chain')
    work = dict(stage=stage, jobs=[], outputs=[], expected=[])
    for name, fixture in fixtures.items():
        if stage in ('ffn', 'chain'):
            jobs, x = ffn_jobs(stack, name, fixture['x'])
            work['jobs'].extend(jobs)
        else:
            x = fixture['mlp']
        if stage in ('qkv', 'chain'):
            jobs, out = qkv_jobs(stack, fixture, x)
            work['jobs'].extend(jobs)
            work['outputs'].extend(out)
            work['expected'].extend(fixture['hashes'][k] for k in ('q', 'k', 'v'))
        else:
            work['outputs'].append(x)
            work['expected'].append(fixture['hashes']['mlp'])
    assert len(work['jobs']) == {'ffn': 80, 'qkv': 32, 'chain': 112}[stage]
    return work
