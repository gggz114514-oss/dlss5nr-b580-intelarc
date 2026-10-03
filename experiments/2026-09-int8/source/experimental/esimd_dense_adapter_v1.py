"""Isolated C ABI probe; no changes to Torch installation or NR dispatch.

Caller retains tensors and this object until its queue completes. Kernel resource
checks occur before any submission. Only aligned, finite FP16 dense test inputs
are covered; input packing is a separate GPU operation with a counted cost.
"""
import ctypes as c
import os
from pathlib import Path
import sys
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _pack_a(A, P, M:tl.constexpr, K:tl.constexpr):
    tile=tl.program_id(0)
    rb=tile//(K//16);kb=tile%(K//16)
    i=tl.arange(0,128)
    row=rb*8+i//16;col=kb*16+i%16
    value=tl.load(A+row*K+col,row<M,other=0)
    tl.store(P+tile*128+i,value)


class Resources(c.Structure):
    _fields_=[(name,c.c_uint32) for name in ('spill_bytes','private_bytes','local_bytes','max_group','registers')]


class Dense:
    def __init__(self,dll):
        self.dll_directory=os.add_dll_directory(str(Path(sys.executable).parent/'Library/bin'))
        self.library=c.CDLL(str(dll))
        self.library.nr_esimd_error.restype=c.c_char_p
        self.library.nr_esimd_resources.argtypes=[c.c_void_p,c.c_int,c.POINTER(Resources)]
        self.library.nr_esimd_resources.restype=c.c_int
        self.library.nr_esimd_dense.argtypes=[c.c_void_p]*5+[c.c_int]*4
        self.library.nr_esimd_dense.restype=c.c_int
        self.device=torch.device('xpu',torch.xpu.current_device())
        self.resources={}
        for packed in (False,True):
            record=Resources()
            self.check(self.library.nr_esimd_resources(torch.xpu.current_stream(self.device).sycl_queue,int(packed),c.byref(record)))
            self.resources[packed]={name:getattr(record,name) for name,_ in Resources._fields_}

    def check(self,status):
        if status:raise RuntimeError(self.library.nr_esimd_error().decode('utf-8','replace'))

    def require(self,packed):
        if self.resources[packed]['spill_bytes']!=0 or self.resources[packed]['max_group']<16:
            raise ValueError('Kernel rejected by resource preflight')

    def pack_weight(self,weight):
        if weight.dtype!=np.float16 or weight.ndim!=2:raise ValueError('FP16 K,N array required')
        k,n=weight.shape
        if k%16 or n%16:raise ValueError('K,N must be multiples of16')
        # [Kblock,Kpair,2,Nblock,Nlane] -> [Nblock,Kblock,Kpair,Nlane,2].
        packed=weight.reshape(k//16,8,2,n//16,16).transpose(3,0,1,4,2).copy()
        return torch.from_numpy(packed).to(self.device)

    def pack_activation(self,a):
        if a.ndim!=2 or a.dtype!=torch.float16 or a.device!=self.device or not a.is_contiguous():
            raise ValueError('Contiguous M,K half XPU input required')
        m,k=a.shape
        if k%16:raise ValueError('K alignment')
        out=torch.empty(((m+7)//8,k//16,8,16),dtype=a.dtype,device=a.device)
        _pack_a[((m+7)//8*(k//16),)](a,out,m,k,num_warps=4)
        return out

    def into(self,a,b,initial,out,*,m,k,n,packed):
        self.require(packed)
        for tensor in (a,b,initial,out):
            if tensor is not None and (tensor.device!=self.device or tensor.dtype!=torch.float16 or not tensor.is_contiguous()):
                raise ValueError('All buffers must be contiguous half on this XPU')
        if not (0<m<=65536 and 16<=k<=4096 and 16<=n<=4096 and k%16==n%16==0):
            raise ValueError('Invalid dense shape')
        if a.numel()!=(((m+7)//8*8*k) if packed else m*k) or b.numel()!=k*n or out.numel()!=m*n:
            raise ValueError('Buffer extent mismatch')
        if initial is not None and initial.numel()!=m*n:raise ValueError('Initial extent mismatch')
        if out.data_ptr() in [t.data_ptr() for t in (a,b,initial) if t is not None]:raise ValueError('Output aliases input')
        self.check(self.library.nr_esimd_dense(torch.xpu.current_stream(self.device).sycl_queue,
            a.data_ptr(),b.data_ptr(),None if initial is None else initial.data_ptr(),out.data_ptr(),m,k,n,int(packed)))
        return out
