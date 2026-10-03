"""Checked launch adapters for the frozen, locally validated V2 tile kernels.

Fixed native quadrant order is authenticated by the owning stack at construction.
Only captured/eager construction uses these Python functions; frames replay graphs.
Every candidate binary must match the completed V2 screen before dispatch.
"""
import torch
import c512_quad_queries_v2 as kernels
from c512_quad_queries_v2 import tile_geometry
from spill_preflight_v1 import select

SCREEN_HASHES = {
    'attention_(0, 0)': '33043824cf0351f2c9f2dd865bc90c3cd1d3fe6e662ffb5d180fe941f9d96155',
    'attention_(0, 4)': 'ea186fc3c689f3a34ddbc332df1751dc9a5b3b44227a4fa36b905ffa3cf690ee',
    'attention_(4, 0)': '2980bc2cfe1baefca912701d0a9ffeb3653eda825cdbbf1934f4df2686c2ec59',
    'attention_(4, 4)': 'de82a4672cff0b3b3444308a57607f8a97ba26e04678040aad077ae1955d48ec',
    'projection': 'efcd2f9bb9d5514ba86b2af2d8affeb77154d8e7aefec89d721dab752786b328',
}
ATTENTION_OPTIONS = dict(num_warps=4, num_stages=1, enable_fp_fusion=False)
PROJECT_OPTIONS = dict(num_warps=4, enable_fp_fusion=False)
GRID = (9, 16)


def operands(values):
    first = values[0]
    if any(t.device.type != 'xpu' or t.device != first.device or
           t.dtype != torch.float16 or not t.is_contiguous() for t in values):
        raise ValueError('Expected contiguous half tensors on one XPU')


def checked(label, jit, args, config, options):
    _, compiled, decision = select(jit, [config], lambda _: args, lambda _: GRID, **options)
    assert compiled.hash == SCREEN_HASHES[label], ('Unscreened compiled kernel', label, compiled.hash)
    return compiled, decision


def attention_args(query, key, value, bias, inverse, shift):
    if any(tuple(t.shape) != (16, 2, 2, 64, 32) for t in (query, key, value)) or bias.shape != (16, 64, 64):
        raise ValueError('Expected complete C512 QKV and head-specific bias')
    operands((query, key, value, bias))
    if inverse.shape != (64,) or inverse.dtype != torch.int64 or inverse.device != query.device or not inverse.is_contiguous():
        raise ValueError('Expected authenticated native inverse order')
    tile_geometry(shift)
    out = torch.empty((16, 9, 16, 32), device=query.device, dtype=query.dtype)
    return (query, key, value, bias, out, *shift)


def projection_args(features, weight, residual, scale):
    if features.shape != (16, 9, 16, 32) or weight.shape != (512, 512) or residual.shape != (12, 12, 512) or scale.shape != (512,):
        raise ValueError('Expected native compact-tile projection geometry')
    operands((features, weight, residual, scale))
    return (features, weight, residual, scale, torch.empty_like(residual))


def attend(query, key, value, bias, inverse, *, shift):
    args = attention_args(query, key, value, bias, inverse, shift)
    compiled, decision = checked('attention_'+str(shift), kernels._attend, args, (16,), ATTENTION_OPTIONS)
    assert kernels._attend[GRID](*args, **ATTENTION_OPTIONS) is compiled
    return args[4], compiled, decision


def project(features, weight, residual, scale):
    args = projection_args(features, weight, residual, scale)
    compiled, decision = checked('projection', kernels._project, args, (16, 32), PROJECT_OPTIONS)
    assert kernels._project[GRID](*args, **PROJECT_OPTIONS) is compiled
    return args[4], compiled, decision


def preflight(query, key, value, bias, inverse, weight, residual, scale):
    """Compile/load all five candidate binaries; do not dispatch or read outputs."""
    result = []
    for shift in ((0, 0), (0, 4), (4, 0), (4, 4)):
        args = attention_args(query, key, value, bias, inverse, shift)
        label = 'attention_'+str(shift)
        k, decision = checked(label, kernels._attend, args, (16,), ATTENTION_OPTIONS)
        result.append(dict(label=label, hash=k.hash, spills=k.n_spills, selection=decision))
    args = projection_args(args[4], weight, residual, scale)
    k, decision = checked('projection', kernels._project, args, (16, 32), PROJECT_OPTIONS)
    result.append(dict(label='projection', hash=k.hash, spills=k.n_spills, selection=decision))
    return result
