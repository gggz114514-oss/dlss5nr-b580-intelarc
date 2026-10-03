"""Exact base-128 digit dots with an explicit, still per-product correction.

CPU work-count study only. Does not claim that the correction is accelerated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from probe_int8_remainder_cpu_v1 import fixtures, exponent, half_bits


def pack(x, axis):
    scaled = x.astype('f8')*512
    units = scaled.astype('i8')
    assert np.array_equal(units, scaled)
    ored = np.bitwise_or.reduce(np.abs(units), axis=axis)
    shift = np.log2(np.maximum(ored & -ored, 1)).astype('i8')
    q = units >> np.expand_dims(shift, axis)
    assert np.max(np.abs(q)) < 128**3
    digits = [np.sign(q)*((np.abs(q) >> (7*i)) & 127) for i in range(3)]
    assert np.array_equal(sum(d << (7*i) for i, d in enumerate(digits)), q)
    return q, shift, digits


def check_tile(a, w, initial):
    acc = initial.astype('f4')
    k = a.shape[1]
    dots = []
    correction_groups = negative_groups = nonzero_terms = remainder_terms = 0
    low_pair_counts = []
    naive_wrong = 0
    for start in range(0, k, 16):
        aa = a[:, start:start+16].astype('f4')
        ww = w[start:start+16].astype('f4')
        qa, ashift, ad = pack(aa, 1)
        qw, wshift, wd = pack(ww, 0)
        ea = np.where(aa == 0, -1000, np.maximum(exponent(aa), -6))
        ew = np.where(ww == 0, -1000, np.maximum(exponent(ww), -6))
        ex = np.clip(np.maximum(np.max(ea[:, :, None]+ew[None, :, :], 1), exponent(acc)), -50, 50)
        reference = np.trunc(np.ldexp(aa[:, :, None]*ww[None, :, :], 13-ex[:, None, :])).astype('i8').sum(1)
        shift = ashift[:, None]+wshift[None, :]-5-ex
        rs = np.maximum(-shift, 0)
        assert rs.max() < 63
        divisor = np.left_shift(np.int64(1), rs)
        summed = np.zeros(acc.shape, dtype='i8')
        count = 0
        # Low-digit pairs indicate explicit correction work, NOT extra dot calls.
        low_pairs = np.zeros(acc.shape, dtype='i8')
        for i, ai in enumerate(ad):
            for j, wi in enumerate(wd):
                if not np.any(np.any(ai != 0, 0) & np.any(wi != 0, 1)):
                    continue
                summed += (ai @ wi) << (7*(i+j))
                count += 1
                connected = (ai != 0).astype('i8') @ (wi != 0).astype('i8') > 0
                low_pairs += connected & (7*(i+j) < rs)
        product = qa[:, :, None]*qw[None, :, :]
        assert np.array_equal(summed, product.sum(1))
        remainder = np.sign(product)*(np.abs(product) % divisor[:, None, :])
        remsum = remainder.sum(1)
        assert np.all((summed-remsum) % divisor == 0)
        corrected = np.left_shift((summed-remsum)//divisor, np.maximum(shift, 0))
        assert np.array_equal(corrected, reference)
        naive = np.left_shift(np.sign(summed)*(np.abs(summed)//divisor), np.maximum(shift, 0))
        naive_wrong += int(np.count_nonzero(naive != reference))
        correction_groups += int(np.count_nonzero(np.any(remainder != 0, 1)))
        negative_groups += int(np.count_nonzero((rs > 0) & np.any(product != 0, 1)))
        nonzero_terms += int(np.count_nonzero(product))
        remainder_terms += int(np.count_nonzero(remainder))
        dots.append(count)
        low_pair_counts.extend(low_pairs[rs > 0].tolist())
        aligned = np.trunc(np.ldexp(acc, 13-ex)).astype('i8')
        ref_bits = np.array([half_bits(v, e) for v, e in zip((reference+aligned).flat, (ex-13).flat)], dtype='u2').reshape(acc.shape)
        bits = np.array([half_bits(v, e) for v, e in zip((corrected+aligned).flat, (ex-13).flat)], dtype='u2').reshape(acc.shape)
        assert np.array_equal(bits, ref_bits)
        acc = bits.view('f2').astype('f4')
        assert np.isfinite(acc).all()
    groups = initial.size*(k//16)
    return dict(k16_groups=groups, digit_dots_per_tile_k16=dots,
                mean_digit_dots=float(np.mean(dots)), max_digit_dots=max(dots),
                nonzero_terms=nonzero_terms, remainder_terms=remainder_terms,
                groups_with_nonzero_remainder=correction_groups,
                groups_with_negative_shift_and_nonzero_product=negative_groups,
                groups_wrong_without_correction=naive_wrong,
                mean_low_digit_pairs_per_negative_shift_output=float(np.mean(low_pair_counts)) if low_pair_counts else 0,
                integer_equal=True, half_bytes_equal=True)


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for name, a, w, init in fixtures.cases()[3:]:
        m, k = a.shape
        n = w.shape[1]
        counts = (a != 0).reshape(m//16, 16, k).sum(1, dtype='i8') @ (w != 0).reshape(k, n//32, 32).sum(2, dtype='i8')
        dense = np.argsort(-counts.ravel(), kind='stable')[:2]
        selected = sorted(set(map(int, np.concatenate((dense, np.linspace(0, counts.size-1, 3, dtype=int))))))
        tiles = []
        for flat in selected:
            ti, tj = divmod(flat, n//32)
            row, col = slice(ti*16, (ti+1)*16), slice(tj*32, (tj+1)*32)
            result = check_tile(a[row], w[:, col], init[row, col])
            assert result['nonzero_terms'] == int(counts[ti, tj])
            tiles.append(dict(tile_row=ti, tile_col=tj, selected_dense=flat in dense, **result))
        results.append(dict(name=name, tiles=tiles))
        print(json.dumps(dict(case=name, tiles=len(tiles))), flush=True)
    # Dense synthetic E4M3 operands avoid dependence on sparse archived samples.
    rng = np.random.default_rng(20260914)
    values = np.unique(np.concatenate((np.arange(1, 8), *[np.arange(8, 16)*(2**e) for e in range(14)], np.arange(8, 15)*(2**14)))).astype('f4')/512
    values = np.concatenate((-values, [0], values)).astype('f2')
    a = rng.choice(values, (16, 32))
    w = rng.choice(values, (32, 32))
    # Restrict magnitude for finite accumulator, retain exponent diversity.
    a = np.clip(a, -4, 4).astype('f2')
    w = np.clip(w, -4, 4).astype('f2')
    controls = [check_tile(a, w, np.full((16, 32), v, dtype='f2')) for v in (0, 64, -64)]
    report = dict(passed=True, device='CPU', cases=results, controls=controls,
                  gpu_tested=False, performance_measured=False,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  oracle_sha256=hashlib.sha256(Path(__file__).with_name('probe_int8_remainder_cpu_v1.py').read_bytes()).hexdigest(),
                  limitations=['At most nine base-128 signed digit dots per K16 tile; K padding and conversion costs not timed.',
                               'CPU oracle still computes every product for remainder; no evidence of cheap correction.',
                               'Row/column shared factors allow separable operands; divisor remains output/accumulator-dependent.',
                               'Low digit products must retain per-product carries before remainder reduction; separate dot remainders are not a replacement.',
                               'Shared half conversion, archived operands and synthetic controls; no full model or new 4060 comparison.'])
    (out/'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PASS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args().out)
