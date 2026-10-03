"""Approximate C512 FFN contract, including both 512 projections and residual.

Static channel scales are fitted in order from the candidate's own activations.
Weights absorb each preceding static scale. INT32 sums replace the old FP16/
FP8 internal boundaries; only entry and final output retain an FP8 boundary.
This is a new numerical algorithm, not an exact migration or a quality claim.
"""
import numpy as np
from int8_ffn_segment_oracle_v1 import (
    quantize, quantize_axis, integer_dot, cubic_half, fp8_boundary, hidden_scale,
    error_metrics,
)


def checked_weights(weights):
    shapes = ((512, 512), (8, 64, 256), (8, 256, 64), (512, 512), (512,))
    assert len(weights) == len(shapes)
    for value, shape in zip(weights, shapes):
        assert value.shape == shape and value.dtype == np.dtype('f2')
        assert np.isfinite(value).all()


def entry(x, packed):
    x = np.asarray(x, dtype=np.float16).reshape(144, 512)
    qx, sx = quantize_axis(fp8_boundary(x), 1)
    z = ((integer_dot(qx, packed['w0']).astype('f4') * sx) * packed['s0']).astype('f2')
    assert np.isfinite(z).all()
    return z, qx, sx


def expand(qz, packed):
    values = []
    for g in range(8):
        a = integer_dot(qz[:, g*64:(g+1)*64], packed['we'][g]).astype('f4')
        values.append(cubic_half((a * packed['se'][g]).astype('f2')))
    return np.concatenate(values, axis=1)


def reduce(qh, packed):
    values = []
    for g in range(8):
        a = integer_dot(qh[:, g*256:(g+1)*256], packed['wr'][g]).astype('f4')
        values.append((a * packed['sr'][g]).astype('f2'))
    out = np.concatenate(values, axis=1)
    assert np.isfinite(out).all()
    return out


def calibrate(inputs, weights):
    """Six fixed samples only; held-out inputs/errors never influence scales."""
    checked_weights(weights)
    assert len(inputs) == 6
    w0, we, wr, wp, skip = weights
    packed = dict(skip=skip.copy())
    packed['w0'], packed['s0'] = quantize_axis(w0, 0)
    zs = [entry(x, packed)[0] for x in inputs]
    packed['sz'] = hidden_scale(np.concatenate(zs), 1.25)
    packed['we'], packed['se'] = quantize_axis(
        we.astype('f4') * packed['sz'].reshape(8, 64, 1), 1)
    hs = [expand(quantize(z, packed['sz']), packed) for z in zs]
    packed['sh'] = hidden_scale(np.concatenate(hs), 1.25)
    packed['wr'], packed['sr'] = quantize_axis(
        wr.astype('f4') * packed['sh'].reshape(8, 256, 1), 1)
    gs = [reduce(quantize(h, packed['sh']), packed) for h in hs]
    packed['sg'] = hidden_scale(np.concatenate(gs), 1.25)
    packed['wp'], packed['sp'] = quantize_axis(
        wp.astype('f4') * packed['sg'].reshape(512, 1), 0)
    for values, name in ((zs, 'sz'), (hs, 'sh'), (gs, 'sg')):
        assert all(np.max(np.abs(v.astype('f4') / packed[name])) <= 127 for v in values)
    return packed


def run(x, packed):
    x = np.asarray(x, dtype=np.float16).reshape(144, 512)
    z, qx, sx = entry(x, packed)
    qz = quantize(z, packed['sz'])
    h = expand(qz, packed)
    qh = quantize(h, packed['sh'])
    g = reduce(qh, packed)
    qg = quantize(g, packed['sg'])
    initial = (fp8_boundary(x).astype('f4') * packed['skip'].astype('f4')).astype('f2')
    raw = (integer_dot(qg, packed['wp']).astype('f4') * packed['sp'] + initial.astype('f4')).astype('f2')
    out = fp8_boundary(raw)
    result = dict(qx=qx, sx=sx.reshape(-1), zraw=z, qz=qz,
                  hraw=h, qh=qh, graw=g, qg=qg, raw=raw, out=out)
    assert all(np.isfinite(v).all() for v in result.values())
    return result


def gpu_constants(packed):
    return [np.ascontiguousarray(v) for v in (
        packed['w0'].T, packed['s0'].reshape(-1), packed['sz'].reshape(-1),
        packed['we'].transpose(0, 2, 1), packed['se'].reshape(8, 256), packed['sh'].reshape(-1),
        packed['wr'].transpose(0, 2, 1), packed['sr'].reshape(8, 64), packed['sg'].reshape(-1),
        packed['wp'].T, packed['sp'].reshape(-1), packed['skip'],
    )]


def clipping(detail, packed):
    return {stage: dict(fraction=float((np.abs(detail[value].astype('f4') / packed[scale]) > 127).mean()),
                        maximum_range_multiple=float(np.max(np.abs(detail[value].astype('f4') / packed[scale])) / 127))
            for stage, value, scale in (('linear', 'zraw', 'sz'), ('cubic', 'hraw', 'sh'), ('group_reduce', 'graw', 'sg'))}
