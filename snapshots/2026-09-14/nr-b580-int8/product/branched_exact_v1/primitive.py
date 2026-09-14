"""Original branch loop versus batched/fused branch arithmetic, raw byte check."""
from types import SimpleNamespace


def check(*,shape_only=False):
    import torch
    from nr_backend.execution import use_arithmetic_backend
    from nr_backend.multihead_block import BranchedMLP,quantize_fp8
    from .kernels import forward
    rows=[]
    def values(shape):
        count=1
        for n in shape:count*=n
        return ((torch.arange(count,device='xpu')%53-26).float()/64).half().reshape(shape)
    with torch.inference_mode(),use_arithmetic_backend('triton'):
        for h,w,c in ((5,7,128),(9,9,256)):
            b=c//32
            model=SimpleNamespace(channels=c,expand=quantize_fp8(values((b,c,128))),
                reduce=quantize_fp8(values((b,128,32))),project=quantize_fp8(values((b,32,c))),
                skip_scale=values((c,)))
            x=values((h,w,c))
            expected=BranchedMLP.forward_unquantized(model,x)
            result,resources=forward(model,quantize_fp8(x),diagnostics=not shape_only)
            if shape_only:
                from torch._subclasses.fake_tensor import FakeTensor
                assert isinstance(result,FakeTensor) and result.shape==expected.shape
            else:
                assert result.cpu().numpy().tobytes()==expected.cpu().numpy().tobytes(),(h,w,c)
            rows.append(dict(shape=[h,w,c],byte_equal=None if shape_only else True,resources=resources))
    return dict(passed=None if shape_only else True,numerical_validation=not shape_only,cases=rows)
