"""CPU feasibility probe of separable exponent masks on complete 16x32 tiles.

The correction is deliberately explicit. This measures work counts, not speed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from probe_int8_remainder_cpu_v1 import fixtures, decode, exponent, half_bits


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for name, a, w, init in fixtures.cases()[3:]:
        m, k = a.shape
        n = w.shape[1]
        assert m % 16 == n % 32 == k % 16 == 0
        # Exact census of nonzero products for every full output tile.
        ac = (a != 0).reshape(m//16, 16, k).sum(1, dtype=np.int64)
        wc = (w != 0).reshape(k, n//32, 32).sum(2, dtype=np.int64)
        counts = ac @ wc
        dense = np.argsort(-counts.ravel(), kind='stable')[:2]
        spread = np.linspace(0, counts.size-1, 3, dtype=int)
        selected = sorted(set(map(int, np.concatenate((dense, spread)))))
        tiles = []
        for flat in selected:
            ti, tj = divmod(flat, n//32)
            av = a[ti*16:(ti+1)*16].astype('f4')
            wv = w[:, tj*32:(tj+1)*32].astype('f4')
            ma, sa = decode(av)
            mw, sw = decode(wv)
            acc = init[ti*16:(ti+1)*16, tj*32:(tj+1)*32].astype('f4')
            active_pair_counts = []
            nonzero = remainders = negative_shift_products = bad_naive = 0
            correction_pairs = 0
            for start in range(0, k, 16):
                aa, ww = av[:, start:start+16], wv[start:start+16]
                am, wm = ma[:, start:start+16], mw[start:start+16]
                ae, we = sa[:, start:start+16], sw[start:start+16]
                ea = np.where(aa == 0, -1000, np.maximum(exponent(aa), -6))
                ew = np.where(ww == 0, -1000, np.maximum(exponent(ww), -6))
                ex = np.clip(np.maximum(np.max(ea[:, :, None]+ew[None, :, :], 1), exponent(acc)), -50, 50)
                reference = np.trunc(np.ldexp(aa[:, :, None]*ww[None, :, :], 13-ex[:, None, :])).astype('i8').sum(1)
                aligned = np.trunc(np.ldexp(acc, 13-ex)).astype('i8')
                result = np.zeros((16, 32), dtype='i8')
                naive = np.zeros_like(result)
                pairs = 0
                for e_a in np.unique(ae[am != 0]):
                    qa = np.where(ae == e_a, am, 0)
                    for e_w in np.unique(we[wm != 0]):
                        qw = np.where(we == e_w, wm, 0)
                        # Only these separable operand masks can be fed to a dot.
                        if not np.any(np.any(qa != 0, 0) & np.any(qw != 0, 1)):
                            continue
                        assert np.abs(qa).max() <= 15 and np.abs(qw).max() <= 15
                        dot = qa @ qw
                        pairs += 1
                        shifts = e_a+e_w+13-ex
                        rs = np.maximum(-shifts, 0)
                        assert rs.max() < 63
                        denominator = np.left_shift(np.int64(1), rs)
                        # Explicit per-product correction oracle, NOT cheap GPU code.
                        product = qa[:, :, None]*qw[None, :, :]
                        remainder = np.sign(product)*(np.abs(product) % denominator[:, None, :])
                        remsum = remainder.sum(1)
                        assert np.all((dot-remsum) % denominator == 0)
                        result += np.left_shift((dot-remsum)//denominator, np.maximum(shifts, 0))
                        naive += np.left_shift(np.sign(dot)*(np.abs(dot)//denominator), np.maximum(shifts, 0))
                        nonzero += int(np.count_nonzero(product))
                        remainders += int(np.count_nonzero(remainder))
                        negative_shift_products += int(np.count_nonzero((product != 0) & (shifts[:, None, :] < 0)))
                        correction_pairs += int(np.any(remainder))
                assert np.array_equal(result, reference), (name, flat, start)
                bad_naive += int(np.count_nonzero(naive != reference))
                active_pair_counts.append(pairs)
                ref_bits = np.array([half_bits(v, e) for v, e in zip((reference+aligned).flat, (ex-13).flat)], dtype='u2').reshape(16, 32)
                got_bits = np.array([half_bits(v, e) for v, e in zip((result+aligned).flat, (ex-13).flat)], dtype='u2').reshape(16, 32)
                assert np.array_equal(ref_bits, got_bits)
                acc = got_bits.view('f2').astype('f4')
                assert np.isfinite(acc).all()
            assert nonzero == int(counts[ti, tj])
            tiles.append(dict(tile_row=ti, tile_col=tj, selected_dense=flat in dense,
                              total_product_slots=16*32*k, nonzero_products=nonzero,
                              nonzero_remainders=remainders,
                              negative_shift_nonzero_products=negative_shift_products,
                              remainder_fraction_nonzero=remainders/nonzero if nonzero else None,
                              active_exponent_pairs_per_k16=active_pair_counts,
                              mean_masked_dots_per_k16=float(np.mean(active_pair_counts)),
                              pairs_with_nonzero_correction=correction_pairs,
                              integer_groups_wrong_without_correction=bad_naive,
                              checked_output_k16_groups=16*32*(k//16),
                              every_k16_integer_sum_equal=True, every_k16_half_bytes_equal=True))
        results.append(dict(name=name, all_tile_count=int(counts.size),
                            full_matrix_nonzero_product_fraction=float(counts.sum())/(m*n*k),
                            dense_tile_nonzero_counts=[int(counts.ravel()[i]) for i in dense], tiles=tiles))
        print(json.dumps(dict(case=name, completed_tiles=len(tiles))), flush=True)
    report = dict(passed=True, device='CPU', gpu_tested=False, performance_measured=False,
                  cases=results, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  oracle_source_sha256=hashlib.sha256(Path(__file__).with_name('probe_int8_remainder_cpu_v1.py').read_bytes()).hexdigest(),
                  limitations=['Archived operands, not fresh exact model activations.',
                               'Counts describe this separable exponent-pair scheme, not a lower bound for all methods.',
                               'Weight masks can be static; A masks and accumulator-dependent corrections remain dynamic.',
                               'Explicit correction still forms per-product intermediates; no GPU speed claim.',
                               'Shared half converter, no new independent 4060 validation.'])
    (out/'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
