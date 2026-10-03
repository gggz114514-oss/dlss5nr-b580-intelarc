"""Approximate INT8 QKV products inside the existing FP8 preparation boundaries.

Frozen GPU kernels; immutable per-column weight packing is construction-only.
Dynamic per-row input quantization remains in every timed graph replay.
"""
import hashlib
import numpy as np
import torch
import triton
from fast_matrices_v3 import _quantize_rows, _matmul
from compact_c512_qkv_pack_v1 import _pack
import int8_ffn_segment_oracle_v1 as oracle

CONFIGS = ((32, 64), (16, 64), (16, 32))
ROWS = (0, 47, 95, 143)
COLS = (0, 31, 32, 63, 64, 95, 511, 512, 1023, 1024, 1504, 1535)


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def prepare_weight(fixture):
    w = fixture['module'].attention.qkv.cpu().numpy().copy()
    assert w.shape == (512, 1536) and w.dtype == np.dtype('f2') and np.isfinite(w).all()
    qw, sw = oracle.quantize_axis(w, 0)
    packed = np.ascontiguousarray(qw.T)
    scale = np.ascontiguousarray(sw.reshape(1536))
    return dict(qw=torch.from_numpy(packed).to('xpu'), sw=torch.from_numpy(scale).to('xpu'),
                host_qw=packed, host_sw=scale, source_sha256=digest(w),
                qw_sha256=digest(packed), sw_sha256=digest(scale))


def build(fixtures, config):
    assert config in CONFIGS
    bm, bn = config
    name = f'int8_{bm}x{bn}'
    work = dict(stage=name, jobs=[], outputs=[], blocks=[])
    options = dict(num_warps=4, enable_fp_fusion=False)
    for block, f in fixtures.items():
        x, module, shift, packed = f['mlp'], f['module'].attention, f['module'].window_shift, f['int8_pack']
        assert x.shape == (144, 512) and x.is_contiguous() and x.dtype == torch.float16
        qx = torch.empty_like(x, dtype=torch.int8)
        sx = torch.empty(144, device=x.device, dtype=torch.float32)
        z = torch.empty((12, 12, 16, 3, 32), device=x.device, dtype=x.dtype)
        out = tuple(torch.empty_like(f[k]) for k in ('q', 'k', 'v'))
        work['jobs'].extend([
            dict(label='int8:quantize_rows', jit=_quantize_rows,
                 args=(x, qx, sx, 144, 512, 512, 1, 512), grid=(144,), options=options),
            dict(label='int8:dense'+str(config), jit=_matmul,
                 args=(qx, packed['qw'], sx, packed['sw'], qx, z, 144, 1536, 512,
                       False, False, True, bm, bn, 32),
                 grid=(triton.cdiv(144,bm), triton.cdiv(1536,bn), 1), options=options),
            dict(label='qkv:pack_'+str(shift), jit=_pack,
                 args=(z, module.scale, module.pixel_order, *out, *shift, 16),
                 grid=(256, 3), options=options),
        ])
        work['outputs'].extend(out)
        work['blocks'].append(dict(name=block, fixture=f, qx=qx, sx=sx, z=z, outputs=out))
    assert len(work['jobs']) == 48 and len(work['outputs']) == 48
    return work


def verify_integer_samples(work):
    """Full entry-quantization check plus 48 independent dot samples per block."""
    records = []
    for b in work['blocks']:
        f, pack = b['fixture'], b['fixture']['int8_pack']
        x = f['mlp'].cpu().numpy()
        assert x.tobytes() == oracle.fp8_boundary(x).tobytes()
        qx, sx = oracle.quantize_axis(x, 1)
        got_qx, got_sx = b['qx'].cpu().numpy(), b['sx'].cpu().numpy()
        assert got_qx.tobytes() == qx.tobytes() and got_sx.tobytes() == sx.reshape(-1).tobytes()
        # All bounded INT8 products/sums fit below 2**23, so INT32->FP32 is exact.
        sums = qx[list(ROWS)].astype('i8') @ pack['host_qw'][list(COLS)].T.astype('i8')
        assert np.abs(sums).max() <= 512*127*127 < 2**23
        want = ((sums.astype('f4') * sx[list(ROWS)]) * pack['host_sw'][list(COLS)][None, :]).astype('f2')
        z = b['z'].cpu().numpy().reshape(144, 1536)
        assert np.isfinite(z).all()
        got = np.ascontiguousarray(z[np.ix_(ROWS, COLS)])
        assert got.tobytes() == want.tobytes(), (work['stage'], b['name'])
        records.append(dict(name=b['name'], input_quantization_byte_equal=True,
            dense_samples_byte_equal=True, samples=48, qx_sha256=digest(qx),
            sx_sha256=digest(sx), sample_sha256=digest(got)))
    return records


def error_metrics(actual, expected, fixture):
    a, b = actual.cpu().numpy(), expected.cpu().numpy()
    assert a.shape == b.shape == (16, 2, 2, 64, 32)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert a.tobytes() == oracle.fp8_boundary(a).tobytes()
    inverse = np.argsort(fixture['module'].attention.pixel_order.cpu().numpy())
    sy, sx = fixture['module'].window_shift
    valid = np.zeros((2, 2, 64), dtype=bool)
    for row in range(144):
        y, x = row//12+sy, row%12+sx
        valid[y//8, x//8, inverse[(y%8)*8+x%8]] = True
    assert valid.sum() == 144
    assert a[:, ~valid, :].tobytes() == b[:, ~valid, :].tobytes()
    a, b = a[:, valid, :], b[:, valid, :]
    d = a.astype('f8')-b.astype('f8')
    return dict(count=a.size, padding_values=16*112*32, padding_byte_equal=True,
                max_abs=float(np.abs(d).max()), mean_abs=float(np.abs(d).mean()),
                rms=float(np.sqrt(np.mean(d*d))), mean_signed=float(d.mean()),
                byte_equal_fraction=float(np.mean(a.view('u2')==b.view('u2'))),
                finite=True, output_fp8_boundary=True)
