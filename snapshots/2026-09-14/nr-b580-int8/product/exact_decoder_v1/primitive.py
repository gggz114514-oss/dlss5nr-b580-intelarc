"""Odd extents, noncontiguous operands and raw positive-zero padding."""
def check():
    import torch
    from nr_backend.decoder import q, half_fma
    from nr_backend.execution import use_arithmetic_backend
    from .kernels import forward
    rows = []
    with torch.inference_mode(), use_arithmetic_backend('triton'):
        for channels in (512, 256, 128, 64, 32):
            p = ((torch.arange(4*10*channels, device='xpu') % 37 - 18).float() / 16).half().reshape(4, 10, channels)[:, ::2]
            s = ((torch.arange(7*18*channels, device='xpu') % 29 - 14).float() / 16).half().reshape(7, 18, channels)[:, ::2]
            scale = torch.full((channels * 2,), .75, device='xpu', dtype=torch.float16)[::2]
            skip = q(s)
            expanded = p.repeat_interleave(2, 0).repeat_interleave(2, 1)[:7, :9]
            merged = half_fma(skip, scale, expanded)
            shift = (4, 4) if channels == 32 else None
            expected = q(merged) if shift is None else torch.nn.functional.pad(merged, (0, 0, 4, 3, 4, 5))
            actual, resource = forward(p, skip, scale, shift=shift)
            assert actual.cpu().numpy().tobytes() == expected.cpu().numpy().tobytes(), channels
            rows.append(dict(channels=channels, byte_equal=True, resource=resource))
    return dict(passed=True, cases=rows)
