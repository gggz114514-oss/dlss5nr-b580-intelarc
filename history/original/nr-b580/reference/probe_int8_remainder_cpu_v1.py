"""Bounded CPU proof of signed-remainder correction; no GPU speed claim."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('fixtures', ROOT/'nr-b580-int8/product/int8_exact_probe_v1/fixtures.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def exponent(x):
    x = np.asarray(x, dtype=np.float32)
    return np.where(x == 0, -1000, ((x.view(np.int32) >> 23) & 255)-127)


def half_bits(value, scale):
    # Integer ties-to-even conversion, matching the documented K16 oracle.
    value, scale = int(value), int(scale)
    mag = abs(value)
    if not mag:
        return 0
    e = mag.bit_length()-1+scale
    shift = max(e-10, -24)-scale
    if shift > 0:
        q, r = divmod(mag, 1 << shift)
        q += r > (1 << (shift-1)) or (r == (1 << (shift-1)) and q & 1)
    else:
        q = mag << -shift
    return min(max(e+14, 0)*1024+q, 0x7c00) | ((value < 0) << 15)


def decode(x):
    # E4M3 subnormals use the fixed 2^-9 unit; normals have four-bit mantissas.
    e = np.maximum(exponent(x), -6)-3
    m = np.ldexp(x.astype(np.float64), -e).astype(np.int64)
    if not np.array_equal(np.ldexp(m.astype(np.float64), e), x):
        raise ValueError('Operand is outside exact E4M3 mantissa representation')
    if np.max(np.abs(m)) > 15:
        raise ValueError('Mantissa exceeds four bits')
    return m, e


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    # Exhaust every possible signed four-bit mantissa pair and tested shift.
    m = np.arange(-15, 16, dtype=np.int64)
    p = m[:, None]*m[None, :]
    checked = 0
    for shift in range(-24, 17):
        if shift < 0:
            d = 1 << -shift
            corrected = (p-np.sign(p)*(np.abs(p) % d))//d
        else:
            corrected = p << shift
        assert np.array_equal(corrected, np.trunc(np.ldexp(p.astype('f8'), shift)).astype('i8'))
        checked += p.size
    results = []
    for name, full_a, full_w, full_init in fixtures.cases()[3:]:
        rows = np.unique(np.linspace(0, full_a.shape[0]-1, 4, dtype=int))
        cols = np.unique(np.linspace(0, full_w.shape[1]-1, 8, dtype=int))
        a = full_a[rows].astype('f4')
        w = full_w[:, cols].astype('f4')
        ma, sa = decode(a)
        mw, sw = decode(w)
        acc = full_init[np.ix_(rows, cols)].astype('f4')
        groups = products = remainder_nonzero = buckets = corrected_buckets = 0
        uncorrected_different = 0
        for start in range(0, a.shape[1], 16):
            av, wv = a[:, start:start+16], w[start:start+16]
            ea = np.where(av == 0, -1000, np.maximum(exponent(av), -6))
            ew = np.where(wv == 0, -1000, np.maximum(exponent(wv), -6))
            ex = np.clip(np.maximum(np.max(ea[:, :, None]+ew[None, :, :], axis=1), exponent(acc)), -50, 50)
            original = np.trunc(np.ldexp(av[:, :, None]*wv[None, :, :], 13-ex[:, None, :])).astype('i8').sum(axis=1)
            aligned = np.trunc(np.ldexp(acc, 13-ex)).astype('i8')
            next_bits = np.empty(acc.shape, dtype='u2')
            for i in range(len(rows)):
                for j in range(len(cols)):
                    prod = ma[i, start:start+16]*mw[start:start+16, j]
                    shifts = sa[i, start:start+16]+sw[start:start+16, j]+13-int(ex[i, j])
                    corrected_sum = naive_sum = 0
                    for shift in np.unique(shifts[prod != 0]):
                        part = prod[(shifts == shift) & (prod != 0)]
                        dot = int(part.sum())  # prospective masked INT8 dot
                        buckets += 1
                        if shift < 0:
                            d = 1 << int(-shift)
                            rem = np.sign(part)*(np.abs(part) % d)
                            numerator = dot-int(rem.sum())
                            assert numerator % d == 0
                            corrected_sum += numerator//d
                            naive_sum += (abs(dot)//d)*(1 if dot >= 0 else -1)
                            remainder_nonzero += int(np.count_nonzero(rem))
                            corrected_buckets += int(np.any(rem))
                        else:
                            corrected_sum += dot << int(shift)
                            naive_sum += dot << int(shift)
                    assert corrected_sum == int(original[i, j]), (name, start, i, j)
                    ref = half_bits(int(original[i, j]+aligned[i, j]), int(ex[i, j])-13)
                    candidate = half_bits(corrected_sum+int(aligned[i, j]), int(ex[i, j])-13)
                    assert ref == candidate
                    next_bits[i, j] = candidate
                    groups += 1
                    products += 16
                    uncorrected_different += naive_sum != corrected_sum
            acc = next_bits.view('f2').astype('f4')
            assert np.isfinite(acc).all()
        results.append(dict(name=name, sampled_rows=rows.tolist(), sampled_cols=cols.tolist(),
                            k=a.shape[1], k16_output_groups=groups, products=products,
                            nonzero_signed_remainders=remainder_nonzero,
                            exponent_buckets=buckets, mean_buckets_per_output_k16=buckets/groups,
                            buckets_needing_correction=corrected_buckets,
                            groups_wrong_without_correction=uncorrected_different,
                            every_k16_integer_sum_equal=True, every_k16_half_bytes_equal=True))
    report = dict(passed=True, device='CPU', exhaustive_product_shift_cases=checked,
                  cases=results, gpu_tested=False, performance_measured=False,
                  limitation='Scalar/output-dependent buckets; counts are not GPU dot dispatch counts. Both paths share the half converter. No new 4060 validation.',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
