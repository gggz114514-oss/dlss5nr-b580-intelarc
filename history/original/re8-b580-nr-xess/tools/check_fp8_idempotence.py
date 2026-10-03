"""CPU bit-model check of triton_fp8._round_fp8_half on every FP16 pattern.

This only proves the bit formula is idempotent; it does not prove that skipping
the second model-layer q() preserves layout, graph capture or end-to-end output.
"""
import numpy as np


def round_bits(input_bits: np.ndarray) -> np.ndarray:
    bits = input_bits.astype(np.int32)
    sign = bits & 0x8000
    raw = bits & 0x7fff
    mag = np.minimum(raw, 0x5f00)
    exponent = mag >> 10
    sig = (mag & 1023) + np.where(exponent != 0, 1024, 0)
    shift = np.maximum(16 - np.maximum(exponent, 1), 1)
    quotient = sig >> shift
    remainder = sig - (quotient << shift)
    midpoint = 1 << (shift - 1)
    rounded = quotient + np.logical_or(
        remainder > midpoint,
        np.logical_and(remainder == midpoint, (quotient & 1) != 0),
    ).astype(np.int32)
    sub = (rounded.astype(np.float32) * np.float32(0.001953125))
    sub_bits = sub.astype(np.float16).view(np.uint16).astype(np.int32)
    normal = (mag + 63 + ((mag >> 7) & 1)) & 0x7f80
    result = np.where(mag < 0x2400, sub_bits, normal) | sign
    result = np.where(raw > 0x7c00, 0x7f80 | sign, result)
    return result.astype(np.uint16)


if __name__ == "__main__":
    all_half = np.arange(65536, dtype=np.uint16)
    once = round_bits(all_half)
    twice = round_bits(once)
    mismatched = np.flatnonzero(once != twice)
    print(f"FP16 patterns: {len(all_half)}; non-idempotent: {len(mismatched)}")
    if len(mismatched):
        print("first mismatches:", mismatched[:16].tolist())
        raise SystemExit(1)
