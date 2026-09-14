"""Full half-domain bit comparison of the complete cubic+FP8 operation."""
def check():
    import numpy as np
    import torch
    from nr_backend.triton_cubic_fp8 import direct_cubic_fp8 as original
    from .cubic import direct_cubic_fp8 as candidate
    bits = np.arange(65536, dtype=np.uint16)
    x = torch.from_numpy(bits.view(np.float16)).to('xpu')
    a = original(x).cpu().numpy().view(np.uint16)
    b = candidate(x).cpu().numpy().view(np.uint16)
    unequal = int(np.count_nonzero(a != b))
    assert unequal == 0, unequal
    return dict(passed=True, half_patterns=65536, mismatches=unequal)
