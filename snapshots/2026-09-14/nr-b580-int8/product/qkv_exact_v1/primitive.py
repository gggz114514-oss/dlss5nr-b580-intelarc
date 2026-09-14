"""Direct pack bytes versus original norm/scale/q/window including half encodings."""


def check(*,shape_only=False):
    import torch
    from nr_backend.attention import normalize_c32
    from nr_backend.pre_mlp import quantize_fp8 as q
    from nr_backend.execution import use_arithmetic_backend
    from .pack import pack
    rows=[]
    with torch.inference_mode(),use_arithmetic_backend('triton'):
        order=torch.tensor([base+g%4+8*(g//4)+16*word for base in (0,4,32,36) for word in range(2) for g in range(8)],device='xpu')
        inverse=torch.argsort(order)
        for h,w,heads,bits in ((8,8,4,False),(16,24,8,False),(16,24,8,True)):
            sequence=torch.arange(h*w*heads*96,device='xpu')
            data=(sequence%65536).to(torch.int16).view(torch.float16) if bits else ((sequence%53-26).float()/32).half()
            z=data.reshape(h,w,heads,3,32)
            scale=(torch.arange(heads,device='xpu').float()/8+.5).half()
            query=q((normalize_c32(z[:,:,:,0])*scale[None,None,:,None]).half())
            key=q(normalize_c32(z[:,:,:,1]))
            value=q(z[:,:,:,2])
            def window(t):
                return t.reshape(h//8,8,w//8,8,heads,32).permute(4,0,2,1,3,5).reshape(heads,h//8,w//8,64,32)[...,order,:]
            expected=tuple(window(t) for t in (query,key,value))
            actual,resource=pack(z,scale,inverse,diagnostics=not shape_only)
            for a,b in zip(actual,expected):
                if shape_only:
                    from torch._subclasses.fake_tensor import FakeTensor
                    assert isinstance(a,FakeTensor) and a.shape==b.shape
                else:
                    assert a.cpu().numpy().tobytes()==b.cpu().numpy().tobytes(),(h,w,heads,bits)
            rows.append(dict(shape=[h,w,heads],all_half_encodings=bits,byte_equal=None if shape_only else True,resource=resource))
    return dict(passed=None if shape_only else True,numerical_validation=not shape_only,cases=rows)
