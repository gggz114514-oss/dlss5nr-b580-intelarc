"""Bitwise full-half-domain comparison in the new compilation context."""
import triton
import triton.language as tl
from nr_backend.triton_fp8 import _round_fp8_half as original
from .fp8 import _round_fp8_half as candidate


@triton.jit
def compare(X, OLD, NEW, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i)
    tl.store(OLD + i, original(x))
    tl.store(NEW + i, candidate(x))


def check():
    import numpy as np
    import torch
    bits = np.arange(65536, dtype=np.uint16)
    x = torch.from_numpy(bits.view(np.float16)).to('xpu')
    old, new = torch.empty_like(x), torch.empty_like(x)
    compare[(65536 // 256,)](x, old, new, 256, enable_fp_fusion=False)
    a = old.cpu().numpy().view(np.uint16)
    b = new.cpu().numpy().view(np.uint16)
    unequal = int(np.count_nonzero(a != b))
    assert unequal == 0, unequal
    return dict(passed=True, half_patterns=65536, mismatches=unequal)
