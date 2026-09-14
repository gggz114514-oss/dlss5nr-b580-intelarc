"""Byte checks for fused stores, compact addresses and alternate exact tiles."""
from types import SimpleNamespace


def check(*, shape_only=False):
    import torch
    from nr_backend.execution import use_arithmetic_backend
    from nr_backend.pre_mlp import quantize_fp8 as q
    from nr_backend.tensor_math import sm89_f16_dot
    from exact_pipeline_v1.compact import CompactLayout as OldCompact
    from exact_pipeline_v1.layout_projection import project as old_project
    from .compact import CompactLayout
    from .math import dot_q
    from .projection import project
    rows = []
    def equal(actual, expected, label):
        if shape_only:
            from torch._subclasses.fake_tensor import FakeTensor
            assert isinstance(actual, FakeTensor) and isinstance(expected, FakeTensor)
            assert actual.shape == expected.shape
            return
        assert actual.cpu().contiguous().numpy().tobytes() == expected.cpu().contiguous().numpy().tobytes(), label
    def values(shape):
        count = 1
        for n in shape:
            count *= n
        return ((torch.arange(count,device='xpu')%47-23).float()/64).half().reshape(shape)
    with torch.inference_mode(), use_arithmetic_backend('triton'):
        for m,k,n in ((7,16,33),(12,64,256),(16,256,64),(9,512,512)):
            a,w = q(values((m,k))), q(values((k,n)))
            initial = values((m,n))
            expected = q(sm89_f16_dot(a,w,chunk_k=16,initial=initial))
            actual = dot_q(a,w,initial=initial)
            equal(actual, expected, ('store',m,k,n))
            rows.append(dict(kind='fp8-store',shape=[m,k,n],byte_equal=None if shape_only else True))
        order = torch.tensor([base+g%4+8*(g//4)+16*word for base in (0,4,32,36) for word in range(2) for g in range(8)],device='xpu')
        inverse = torch.argsort(order)
        module = SimpleNamespace(heads=16, qkv=q(values((512,1536))),
                                 scale=values((16,)), bias=values((16,64,64)),pixel_order=order)
        for h,w,sy,sx in ((12,12,0,0),(12,12,4,4),(8,16,4,0)):
            ph,pw = ((h+sy+7)//8)*8,((w+sx+7)//8)*8
            input = torch.nn.functional.pad(q(values((h,w,512))),(0,0,sx,pw-w-sx,sy,ph-h-sy))
            old,new = OldCompact(None),CompactLayout(None)
            old.geometry = new.geometry = (h,w,sy,sx)
            new.dataflow = True
            packed = old.windows(module,input)
            actual = new.windows(module,input)
            baseline = packed[...,inverse,:].reshape(16,ph//8,pw//8,8,8,32).permute(0,1,3,2,4,5).reshape(16,ph,pw,32)[:,sy:sy+h,sx:sx+w].reshape(16,h*w,32)
            equal(actual[0][actual[1]], baseline, ('compact',h,w,sy,sx))
            weight,initial = q(values((512,512))),values((h,w,512))
            expected = old_project(packed,weight,initial,geometry=(h,w,sy,sx),inverse=inverse)
            for bm in (4,8):
                result = project(actual,weight,initial,geometry=(h,w,sy,sx),inverse=inverse,bm=bm)
                equal(result, expected, ('projection',h,w,sy,sx,bm))
            rows.append(dict(kind='compact-and-tiles',geometry=[h,w,sy,sx],byte_equal=None if shape_only else True))
    return dict(passed=None if shape_only else True,shape_only=shape_only,numerical_validation=not shape_only,cases=rows)
