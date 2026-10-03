"""C512 complete INT8 FFN screen: 5-kernel split and 4-kernel fused groups.

M144 only. Static scales/packed weights are construction inputs. The fused path
keeps quantized cubic hidden tiles in registers; the split control stores QH.
No inner FP8 round trips. Debug stores are separate untimed specializations.
"""
import torch
import triton
import triton.language as tl
from short_fp8_v2 import round_half
from int8_ffn_segment_gpu_v1 import _q
from spill_preflight_v1 import select


@triton.jit
def _cubic(x):
    t = tl.minimum(tl.maximum(x.to(tl.float32), -4.), 4.).to(tl.float16)
    p = tl.fma(-tl.abs(t), tl.full((), .055908203125, tl.float16), tl.full((), .447265625, tl.float16)).to(tl.float16)
    v = tl.fma(t, p, tl.full((), .89453125, tl.float16)).to(tl.float16)
    return (x.to(tl.float32) * v.to(tl.float32)).to(tl.float16)


@triton.jit
def _entry(X, QX, SX):
    row = tl.program_id(0)
    col = tl.arange(0, 512)
    x = round_half(tl.load(X + row*512 + col)).to(tl.float32)
    maximum = tl.max(tl.abs(x), 0)
    scale = tl.where(maximum > 0, tl.div_rn(maximum, 127.), 1.)
    tl.store(QX + row*512 + col, _q(x, scale))
    tl.store(SX + row, scale)


@triton.jit
def _linear(QX, SX, W, SW, SZ, QZ, RAW, BM: tl.constexpr, BN: tl.constexpr, DEBUG: tl.constexpr):
    row = tl.program_id(0)*BM + tl.arange(0, BM)
    col = tl.program_id(1)*BN + tl.arange(0, BN)
    lane = tl.arange(0, 32)
    total = tl.full((BM, BN), 0, tl.int32)
    for start in range(16):
        k = start*32 + lane
        a = tl.load(QX + row[:, None]*512 + k[None, :], row[:, None] < 144, other=0)
        w = tl.load(W + col[None, :]*512 + k[:, None])
        total = tl.dot(a, w, total, out_dtype=tl.int32)
    sx = tl.load(SX + row, row < 144, other=1.)
    sw, sz = tl.load(SW + col), tl.load(SZ + col)
    z = ((total.to(tl.float32)*sx[:, None])*sw[None, :]).to(tl.float16)
    offset = row[:, None]*512 + col[None, :]
    tl.store(QZ + offset, _q(z.to(tl.float32), sz[None, :]), row[:, None] < 144)
    if DEBUG:
        tl.store(RAW + offset, z, row[:, None] < 144)


@triton.jit
def _expand(QZ, W, SW, SH, QH, RAW, BM: tl.constexpr, BN: tl.constexpr, DEBUG: tl.constexpr):
    row = tl.program_id(0)*BM + tl.arange(0, BM)
    group = tl.program_id(1)
    col = tl.program_id(2)*BN + tl.arange(0, BN)
    lane = tl.arange(0, 32)
    total = tl.full((BM, BN), 0, tl.int32)
    for start in range(2):
        k = start*32 + lane
        a = tl.load(QZ + row[:, None]*512 + group*64 + k[None, :], row[:, None] < 144, other=0)
        w = tl.load(W + group*256*64 + col[None, :]*64 + k[:, None])
        total = tl.dot(a, w, total, out_dtype=tl.int32)
    sw, sh = tl.load(SW + group*256 + col), tl.load(SH + group*256 + col)
    h = _cubic((total.to(tl.float32)*sw[None, :]).to(tl.float16))
    offset = row[:, None]*2048 + group*256 + col[None, :]
    tl.store(QH + offset, _q(h.to(tl.float32), sh[None, :]), row[:, None] < 144)
    if DEBUG:
        tl.store(RAW + offset, h, row[:, None] < 144)


@triton.jit
def _reduce(QH, W, SW, SG, QG, RAW, BM: tl.constexpr, BN: tl.constexpr, DEBUG: tl.constexpr):
    row = tl.program_id(0)*BM + tl.arange(0, BM)
    group = tl.program_id(1)
    col = tl.program_id(2)*BN + tl.arange(0, BN)
    lane = tl.arange(0, 32)
    total = tl.full((BM, BN), 0, tl.int32)
    for start in range(8):
        k = start*32 + lane
        a = tl.load(QH + row[:, None]*2048 + group*256 + k[None, :], row[:, None] < 144, other=0)
        w = tl.load(W + group*64*256 + col[None, :]*256 + k[:, None])
        total = tl.dot(a, w, total, out_dtype=tl.int32)
    sw, sg = tl.load(SW + group*64 + col), tl.load(SG + group*64 + col)
    g = (total.to(tl.float32)*sw[None, :]).to(tl.float16)
    offset = row[:, None]*512 + group*64 + col[None, :]
    tl.store(QG + offset, _q(g.to(tl.float32), sg[None, :]), row[:, None] < 144)
    if DEBUG:
        tl.store(RAW + offset, g, row[:, None] < 144)


@triton.jit
def _groups(QZ, WE, SE, SH, WR, SR, SG, QG, RAW, BM: tl.constexpr, BN: tl.constexpr, DEBUG: tl.constexpr):
    row = tl.program_id(0)*BM + tl.arange(0, BM)
    group = tl.program_id(1)
    col = tl.program_id(2)*BN + tl.arange(0, BN)
    lane = tl.arange(0, 32)
    total = tl.full((BM, BN), 0, tl.int32)
    for part in range(8):
        hcol = part*32 + lane
        expanded = tl.full((BM, 32), 0, tl.int32)
        for start in range(2):
            k = start*32 + lane
            a = tl.load(QZ + row[:, None]*512 + group*64 + k[None, :], row[:, None] < 144, other=0)
            we = tl.load(WE + group*256*64 + hcol[None, :]*64 + k[:, None])
            expanded = tl.dot(a, we, expanded, out_dtype=tl.int32)
        se, sh = tl.load(SE + group*256 + hcol), tl.load(SH + group*256 + hcol)
        hidden = _cubic((expanded.to(tl.float32)*se[None, :]).to(tl.float16))
        qh = _q(hidden.to(tl.float32), sh[None, :])
        wr = tl.load(WR + group*64*256 + col[None, :]*256 + hcol[:, None])
        total = tl.dot(qh, wr, total, out_dtype=tl.int32)
    sr, sg = tl.load(SR + group*64 + col), tl.load(SG + group*64 + col)
    g = (total.to(tl.float32)*sr[None, :]).to(tl.float16)
    offset = row[:, None]*512 + group*64 + col[None, :]
    tl.store(QG + offset, _q(g.to(tl.float32), sg[None, :]), row[:, None] < 144)
    if DEBUG:
        tl.store(RAW + offset, g, row[:, None] < 144)


@triton.jit
def _project(QG, W, SW, X, SKIP, OUT, RAW, BM: tl.constexpr, BN: tl.constexpr, DEBUG: tl.constexpr):
    row = tl.program_id(0)*BM + tl.arange(0, BM)
    col = tl.program_id(1)*BN + tl.arange(0, BN)
    lane = tl.arange(0, 32)
    total = tl.full((BM, BN), 0, tl.int32)
    for start in range(16):
        k = start*32 + lane
        a = tl.load(QG + row[:, None]*512 + k[None, :], row[:, None] < 144, other=0)
        w = tl.load(W + col[None, :]*512 + k[:, None])
        total = tl.dot(a, w, total, out_dtype=tl.int32)
    offset = row[:, None]*512 + col[None, :]
    x = round_half(tl.load(X + offset, row[:, None] < 144, other=0))
    sw, skip = tl.load(SW + col), tl.load(SKIP + col)
    initial = (x.to(tl.float32)*skip[None, :].to(tl.float32)).to(tl.float16)
    raw = (total.to(tl.float32)*sw[None, :] + initial.to(tl.float32)).to(tl.float16)
    tl.store(OUT + offset, round_half(raw), row[:, None] < 144)
    if DEBUG:
        tl.store(RAW + offset, raw, row[:, None] < 144)


class Segment:
    def __init__(self, x, w0, s0, sz, we, se, sh, wr, sr, sg, wp, sp, skip):
        constants = (w0, s0, sz, we, se, sh, wr, sr, sg, wp, sp, skip)
        shapes = ((512, 512), (512,), (512,), (8, 256, 64), (8, 256), (2048,),
                  (8, 64, 256), (8, 64), (512,), (512, 512), (512,), (512,))
        dtypes = (torch.int8, torch.float32, torch.float32, torch.int8, torch.float32, torch.float32,
                  torch.int8, torch.float32, torch.float32, torch.int8, torch.float32, torch.float16)
        for t, shape, dtype in [(x, (144, 512), torch.float16), *zip(constants, shapes, dtypes)]:
            assert tuple(t.shape) == shape and t.dtype == dtype and t.is_contiguous()
            assert t.device == x.device and t.device.type == 'xpu'
        self.buffers = {}
        for name, shape, dtype in (
            ('qx', (144, 512), torch.int8), ('sx', (144,), torch.float32),
            ('qz', (144, 512), torch.int8), ('qh', (144, 2048), torch.int8),
            ('qg', (144, 512), torch.int8), ('zraw', (144, 512), torch.float16),
            ('hraw', (144, 2048), torch.float16), ('graw', (144, 512), torch.float16),
            ('raw', (144, 512), torch.float16), ('out', (144, 512), torch.float16)):
            self.buffers[name] = torch.empty(shape, dtype=dtype, device=x.device)
        b = self.buffers
        self.resources, self.selections, self.plans = {}, {}, {}

        def bind(label, jit, configs, args_for, grid_for):
            options = dict(num_warps=4, num_stages=1, enable_fp_fusion=False)
            cfg, kernel, choice = select(jit, configs, args_for, grid_for, **options)
            self.resources[label] = dict(hash=kernel.hash, spills=kernel.n_spills,
                registers=kernel.n_regs, shared_bytes=kernel.metadata.shared)
            self.selections[label] = choice
            args, grid = args_for(cfg), grid_for(cfg)
            def run():
                assert jit[grid](*args, **options) is kernel
            return run

        entry = bind('entry', _entry, [(1,)], lambda _: (x, b['qx'], b['sx']), lambda _: (144,))
        dense_configs = [(32, 64), (16, 64), (16, 32)]
        group_configs = [(16, 64), (16, 32)]
        dense_grid = lambda c: (triton.cdiv(144, c[0]), 512//c[1])
        for debug in (False, True):
            linear = bind(f'linear_debug{debug}', _linear, dense_configs,
                lambda c: (b['qx'], b['sx'], w0, s0, sz, b['qz'], b['zraw'], *c, debug), dense_grid)
            expand = bind(f'expand_debug{debug}', _expand, dense_configs,
                lambda c: (b['qz'], we, se, sh, b['qh'], b['hraw'], *c, debug),
                lambda c: (triton.cdiv(144, c[0]), 8, 256//c[1]))
            reduce = bind(f'reduce_debug{debug}', _reduce, dense_configs,
                lambda c: (b['qh'], wr, sr, sg, b['qg'], b['graw'], *c, debug),
                lambda c: (triton.cdiv(144, c[0]), 8, 64//c[1]))
            groups = bind(f'groups_debug{debug}', _groups, group_configs,
                lambda c: (b['qz'], we, se, sh, wr, sr, sg, b['qg'], b['graw'], *c, debug),
                lambda c: (triton.cdiv(144, c[0]), 8, 64//c[1]))
            project = bind(f'project_debug{debug}', _project, dense_configs,
                lambda c: (b['qg'], wp, sp, x, skip, b['out'], b['raw'], *c, debug), dense_grid)
            self.plans['split', debug] = (entry, linear, expand, reduce, project)
            self.plans['fused', debug] = (entry, linear, groups, project)

    def run(self, variant, debug=False):
        for step in self.plans[variant, debug]:
            step()
        return self.buffers['out']
