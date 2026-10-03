"""Numeric helpers only; the owned post forward keeps its original dataflow.

CPU development status: AST only, no import/compile/launch. Native RTZ uses
the Intel Triton downcast API and must pass the caller's binary rounding gate.
There is no bit correction, RGB/history tail fusion or reference replacement.
"""
import triton
import triton.language as tl


@triton.jit
def native_rtz_cast_probe(VALUE, OUT, N: tl.constexpr, B: tl.constexpr):
    """Independent diagnostic only; no clamp, reference correction or post code."""
    p = tl.program_id(0) * B + tl.arange(0, B)
    valid = p < N
    value = tl.load(VALUE + p, valid, other=0).to(tl.float32)
    result = value.to(tl.float16, fp_downcast_rounding="rtz")
    tl.store(OUT + p, result, valid)


@triton.jit
def sigmoid_half_to_float(LOGIT, OUT, H: tl.constexpr, W: tl.constexpr,
                          SY: tl.constexpr, SX: tl.constexpr, B: tl.constexpr):
    p = tl.program_id(0) * B + tl.arange(0, B)
    valid = p < H * W
    x = tl.load(LOGIT + (p // W) * SY + (p % W) * SX,
                valid, other=0).to(tl.float16).to(tl.float32)
    z = tl.exp(-tl.abs(x))
    denominator = 1.0 + z
    value = tl.where(x >= 0.0, 1.0 / denominator, z / denominator)
    tl.store(OUT + p, value, valid)


@triton.jit
def store_float_to_half(VALUE, OUT, N: tl.constexpr, UNIT: tl.constexpr,
                        ROUNDING: tl.constexpr, B: tl.constexpr):
    p = tl.program_id(0) * B + tl.arange(0, B)
    valid = p < N
    value = tl.load(VALUE + p, valid, other=0).to(tl.float32)
    if UNIT and ROUNDING == "rtz":
        # Separate stores avoid clamp-select lowering that lost NaN and -0
        # in the tested reset specialization. Masks cover every valid lane.
        low = value < 0.0
        high = value > 1.0
        keep = ~(low | high)
        raw_half = value.to(tl.float16, fp_downcast_rounding="rtz")
        tl.store(OUT + p, raw_half, valid & keep)
        tl.store(OUT + p, tl.full((B,), 0.0, tl.float16), valid & low)
        tl.store(OUT + p, tl.full((B,), 1.0, tl.float16), valid & high)
    else:
        if UNIT:
            value = tl.where(value < 0.0, 0.0, value)
            value = tl.where(value > 1.0, 1.0, value)
        if ROUNDING == "rtz":
            result = value.to(tl.float16, fp_downcast_rounding="rtz")
        else:
            tl.static_assert(ROUNDING == "rtne", "Unknown half store rounding")
            result = value.to(tl.float16, fp_downcast_rounding="rtne")
        tl.store(OUT + p, result, valid)
