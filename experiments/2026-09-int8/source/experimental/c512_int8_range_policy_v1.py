"""Fixed training-only hidden range floors and diagnostic attribution.

No runtime quantizer/kernel changes. Keep the original calibration and widen
only weak hidden ranges to 1/64 or 1/16 of their own group's training maximum
(with the existing 1.25 margin). Refit downstream scales on training inputs.
Probe/counterfactual outputs never influence packing. Counterfactuals are CPU
diagnostics, not proposed deployable arithmetic or full-image quality metrics.
"""
import numpy as np
import c512_int8_ffn_oracle_v1 as oracle

POLICIES = ('original', 'floor64', 'floor16')


def prepare(inputs, weights):
    assert len(inputs) == 6
    original = oracle.calibrate(inputs, weights)
    hidden = [oracle.expand(oracle.quantize(oracle.entry(x, original)[0], original['sz']), original)
              for x in inputs]
    maximum = np.max(np.abs(np.concatenate(hidden).astype('f4')), axis=0, keepdims=True)
    assert oracle.hidden_scale(np.concatenate(hidden), 1.25).tobytes() == original['sh'].tobytes()
    group_maximum = maximum.reshape(8, 256).max(axis=1, keepdims=True)
    variants = {'original': original}
    for divisor in (64, 16):
        p = dict(original)
        floor = np.repeat(group_maximum, 256, axis=1).reshape(1, 2048)
        floor = (floor*np.float32(1.25)/np.float32(divisor))/np.float32(127)
        p['sh'] = np.maximum(original['sh'], floor).astype('f4')
        p['wr'], p['sr'] = oracle.quantize_axis(weights[2].astype('f4')*p['sh'].reshape(8, 256, 1), 1)
        groups = [oracle.reduce(oracle.quantize(h, p['sh']), p) for h in hidden]
        # Do not narrow any previously covered group-output channel.
        p['sg'] = np.maximum(original['sg'], oracle.hidden_scale(np.concatenate(groups), 1.25))
        p['wp'], p['sp'] = oracle.quantize_axis(weights[3].astype('f4')*p['sg'].reshape(512, 1), 0)
        assert all(np.max(np.abs(h.astype('f4')/p['sh'])) <= 127 for h in hidden)
        assert all(np.max(np.abs(g.astype('f4')/p['sg'])) <= 127 for g in groups)
        assert all(p[key] is original[key] for key in ('w0', 's0', 'sz', 'we', 'se', 'skip'))
        variants['floor' + str(divisor)] = p
    assert np.all(variants['floor16']['sh'] >= variants['floor64']['sh'])
    return variants, maximum


def folding_summary(packed, weights, training_maximum):
    reference = weights[2].reshape(2048, 64)
    quantized = packed['wr'].reshape(2048, 64)
    reference_nonzero = np.count_nonzero(reference, axis=1)
    quantized_nonzero = np.count_nonzero(quantized, axis=1)
    lost = (reference_nonzero > 0) & (quantized_nonzero == 0)
    inactive = training_maximum.reshape(2048) == 0
    return dict(original_nonzero_rows=int((reference_nonzero > 0).sum()),
                packed_nonzero_rows=int((quantized_nonzero > 0).sum()),
                lost_entire_rows=int(lost.sum()), lost_training_inactive_rows=int((lost & inactive).sum()),
                training_inactive_channels=int(inactive.sum()),
                hidden_channels_at_min_scale=int((packed['sh'] == packed['sh'].min()).sum()))


def hidden_channels(detail, packed, weights, training_maximum, original_scale):
    hidden = detail['hraw'].astype('f4')
    capacity = packed['sh'].reshape(2048)*np.float32(127)
    magnitude = np.abs(hidden)
    clipped = magnitude/packed['sh'] > 127
    counts = clipped.sum(axis=0)
    maximum = magnitude.max(axis=0)
    original = weights[2].reshape(2048, 64)
    quantized = packed['wr'].reshape(2048, 64)
    original_nz = np.count_nonzero(original, axis=1)
    packed_nz = np.count_nonzero(quantized, axis=1)
    floor_limited = original_scale.reshape(2048) == original_scale.min()
    inactive = training_maximum.reshape(2048) == 0
    rows = []
    for channel in np.flatnonzero(counts):
        rows.append(dict(channel=int(channel), group=int(channel//256), count=int(counts[channel]),
            calibration_absmax=float(training_maximum[0, channel]), sample_absmax=float(maximum[channel]),
            capacity=float(capacity[channel]), maximum_overshoot=float(maximum[channel]-capacity[channel]),
            range_multiple=float(maximum[channel]/capacity[channel]),
            original_floor_limited=bool(floor_limited[channel]), training_inactive=bool(inactive[channel]),
            original_reduce_nonzero=int(original_nz[channel]), packed_reduce_nonzero=int(packed_nz[channel]),
            lost_reduce_row=bool(original_nz[channel] > 0 and packed_nz[channel] == 0)))
    assert sum(row['count'] for row in rows) == int(clipped.sum())
    return dict(total_values=144*2048, clipped_values=int(clipped.sum()), clipped_channels=len(rows),
        clipped_values_in_original_floor_channels=int(counts[floor_limited].sum()),
        clipped_values_in_training_inactive_channels=int(counts[inactive].sum()),
        clipped_values_in_lost_reduce_rows=int(counts[(original_nz > 0) & (packed_nz == 0)].sum()), channels=rows)


def _finish(x, groups, packed):
    qg = oracle.quantize(groups, packed['sg'])
    initial = (oracle.fp8_boundary(x).astype('f4')*packed['skip'].astype('f4')).astype('f2')
    raw = (oracle.integer_dot(qg, packed['wp']).astype('f4')*packed['sp'] + initial.astype('f4')).astype('f2')
    return oracle.fp8_boundary(raw)


def counterfactuals(x, detail, packed, weights):
    """Separate hidden clamp from folded-weight loss at a fixed baseline input.

The first keeps rounded integer hidden units but removes their +/-127 clamp,
using FP64 for the now-wide integer dot. The second keeps clamped QH but uses
original FP16 reduction weights in a diagnostic FP64 dot. Both retain original
SG and output projection. They do not model a new full NR inference chain.
"""
    unit = detail['hraw'].astype('f4')/packed['sh']
    rounded = np.floor(np.abs(unit)+np.float32(.5))*np.where(unit < 0, np.float32(-1), np.float32(1))
    assert np.isfinite(rounded).all()
    assert rounded.dtype == np.float32 and np.max(np.abs(rounded))*127*256 < 2**53
    groups_wide, groups_half_weights = [], []
    for group in range(8):
        columns = slice(group*256, (group+1)*256)
        wide = rounded[:, columns].astype('f8') @ packed['wr'][group].astype('f8')
        assert np.isfinite(wide).all() and np.array_equal(wide, np.rint(wide))
        groups_wide.append((wide.astype('f4')*packed['sr'][group]).astype('f2'))
        physical_hidden = detail['qh'][:, columns].astype('f8')*packed['sh'][:, columns].astype('f8')
        groups_half_weights.append((physical_hidden @ weights[2][group].astype('f8')).astype('f2'))
    wide_out = _finish(x, np.concatenate(groups_wide, axis=1), packed)
    weight_out = _finish(x, np.concatenate(groups_half_weights, axis=1), packed)
    no_clip = not np.any(np.abs(unit) > 127)
    if no_clip:
        assert wide_out.tobytes() == detail['out'].tobytes()
    return dict(no_hidden_clamp=wide_out, original_reduce_weights_fp64=weight_out)
