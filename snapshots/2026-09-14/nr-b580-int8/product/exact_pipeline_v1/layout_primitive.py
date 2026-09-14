"""Direct projection against materialized original layout and exact arithmetic."""
def check():
    import torch
    from nr_backend.pre_mlp import quantize_fp8 as q
    from nr_backend.tensor_math import sm89_f16_dot
    from nr_backend.vit_block import split_k_projection
    from nr_backend.execution import use_arithmetic_backend
    from .layout_projection import project
    rows = []
    with torch.inference_mode(), use_arithmetic_backend('triton'):
        def values(shape):
            count = 1
            for n in shape:
                count *= n
            return ((torch.arange(count, device='xpu') % 31 - 15).float()/32).half().reshape(shape)
        for tokens in (64, 96):
            packed = q(values((32, tokens, 32)))
            weight = q(values((1024, 1024)))
            initial = values((tokens, 1024))
            expected = split_k_projection(packed.transpose(0, 1).reshape(tokens, 1024), weight, initial)
            actual = project(packed, weight, initial, parts=4)
            assert actual.cpu().numpy().tobytes() == expected.cpu().numpy().tobytes()
            rows.append(dict(kind='vit', tokens=tokens, byte_equal=True))
        order = torch.tensor([base+g%4+8*(g//4)+16*word for base in (0,4,32,36) for word in range(2) for g in range(8)], device='xpu')
        inverse = torch.argsort(order)
        for sy, sx in ((0, 0), (4, 4)):
            packed = values((16, 2, 2, 64, 32))
            unpacked = packed[..., inverse, :].reshape(16, 2, 2, 8, 8, 32).permute(1, 3, 2, 4, 0, 5).reshape(16, 16, 512)[sy:sy+12, sx:sx+12]
            weight = q(values((512, 512)))
            initial = values((12, 12, 512))
            expected = sm89_f16_dot(q(unpacked), weight, chunk_k=16, initial=initial)
            actual = project(packed, weight, initial, geometry=(12, 12, sy, sx), inverse=inverse)
            assert actual.cpu().numpy().tobytes() == expected.cpu().numpy().tobytes()
            rows.append(dict(kind='c512', shift=[sy, sx], byte_equal=True))
    return dict(passed=True, cases=rows)
