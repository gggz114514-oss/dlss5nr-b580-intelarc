"""CPU streaming NPY comparisons. No NumPy/Torch/device dependency."""
from __future__ import annotations
import array
import ast
import hashlib
import math
import struct
import sys
import runner_common as c

def header(stream):
    c.require(stream.read(6) == b'\x93NUMPY', 'Not NPY')
    version = tuple(stream.read(2))
    c.require(version in ((1, 0), (2, 0), (3, 0)), 'Unsupported NPY version')
    count = struct.unpack('<H' if version == (1, 0) else '<I', stream.read(2 if version == (1, 0) else 4))[0]
    c.require(count <= 4096, 'Oversized NPY header')
    row = ast.literal_eval(stream.read(count).decode('utf-8' if version == (3, 0) else 'latin1'))
    c.require(set(row) == {'descr', 'fortran_order', 'shape'} and row['fortran_order'] is False
              and row['descr'] in ('<f2', '<f4', '<f8'), 'Only contiguous little-endian float output accepted')
    return row

def values(block, descr):
    if descr == '<f2':
        return [v[0] for v in struct.iter_unpack('<e', block)]
    result = array.array('f' if descr == '<f4' else 'd')
    result.frombytes(block)
    if sys.byteorder != 'little':
        result.byteswap()
    return result

def compare_pair(reference, candidate, *, shape=c.FRAME_SHAPE):
    ref_path, path = c.checked(reference), c.checked(candidate)
    sums = squares = maximum = 0.0
    changed_values = changed_pixels = count = 0
    hashes = (hashlib.sha256(), hashlib.sha256())
    with ref_path.open('rb') as left, path.open('rb') as right:
        ha, hb = header(left), header(right)
        c.require(ha['shape'] == hb['shape'] == tuple(shape), 'Comparison shape mismatch')
        c.require(ha['descr'] == reference['dtype'] and hb['descr'] == candidate['dtype'], 'Saved dtype receipt differs')
        n = math.prod(shape)
        channels = shape[-1]
        remaining = n
        while remaining:
            size = min(32766 - 32766 % channels, remaining)
            ba, bb = left.read(size * int(ha['descr'][-1])), right.read(size * int(hb['descr'][-1]))
            c.require(len(ba) == size * int(ha['descr'][-1]) and len(bb) == size * int(hb['descr'][-1]), 'Truncated NPY')
            hashes[0].update(ba); hashes[1].update(bb)
            va, vb = values(ba, ha['descr']), values(bb, hb['descr'])
            pixel_changed = False
            for index, (a, b) in enumerate(zip(va, vb)):
                c.require(math.isfinite(a) and math.isfinite(b), 'Nonfinite saved output/private history')
                delta = abs(b - a)
                sums += delta; squares += delta * delta; maximum = max(maximum, delta)
                changed_values += a != b; pixel_changed |= a != b
                if (index + 1) % channels == 0:
                    changed_pixels += pixel_changed; pixel_changed = False
            count += size; remaining -= size
        c.require(not left.read(1) and not right.read(1), 'Unexpected trailing NPY bytes')
    for row, digest in zip((reference, candidate), hashes):
        c.require(row['raw_sha256'] == digest.hexdigest(), 'Raw SHA receipt differs from saved bytes')
    mse = squares / count
    identical = ha['descr'] == hb['descr'] and hashes[0].digest() == hashes[1].digest()
    return dict(finite=True, nan_count=0, inf_count=0, byte_identical=identical,
                mae=sums/count, max_abs_error=maximum, mse=mse, rmse=math.sqrt(mse),
                psnr_peak_1_db=None if mse == 0 else -10*math.log10(mse),
                psnr_zero_error=mse == 0, changed_values=changed_values, changed_pixels=changed_pixels,
                scalar_count=count, pixel_count=count//shape[-1],
                reference_raw_sha256=reference['raw_sha256'], candidate_raw_sha256=candidate['raw_sha256'])

def compare_frames(reference, candidate, *, shape=c.FRAME_SHAPE):
    c.require(len(reference) == len(candidate) and reference, 'Unequal input sequence lengths')
    rows = []
    for ref, cand in zip(reference, candidate):
        keys = ('frame_id', 'reset', 'original_rgb_raw_sha256', 'original_motion_raw_sha256', 'seed_after')
        c.require(all(ref[k] == cand[k] for k in keys), 'Comparison uses different source/reset/control/history sequences')
        rows.append(dict(frame_id=ref['frame_id'], output=compare_pair(ref['output'], cand['output'], shape=shape),
                         history=compare_pair(ref['history'], cand['history'], shape=shape)))
    aggregate = {}
    for role in ('output', 'history'):
        reports = [row[role] for row in rows]
        scalar_count = sum(r['scalar_count'] for r in reports)
        mse = sum(r['mse'] * r['scalar_count'] for r in reports) / scalar_count
        aggregate[role] = dict(finite=True, all_byte_identical=all(r['byte_identical'] for r in reports),
            mae=sum(r['mae'] * r['scalar_count'] for r in reports)/scalar_count,
            max_abs_error=max(r['max_abs_error'] for r in reports), mse=mse,
            psnr_peak_1_db=None if mse == 0 else -10*math.log10(mse), psnr_zero_error=mse == 0,
            changed_pixels=sum(r['changed_pixels'] for r in reports),
            changed_values=sum(r['changed_values'] for r in reports))
    return dict(frame_count=len(rows), aggregate=aggregate, per_frame=rows)
