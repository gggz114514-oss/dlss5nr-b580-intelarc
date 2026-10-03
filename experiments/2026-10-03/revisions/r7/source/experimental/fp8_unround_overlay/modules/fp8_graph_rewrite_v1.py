"""Eliminate proven idempotent FP8 conversions inside the owned NR256 body.

The value analysis runs only when constructing the static GPU graph. Every
elision borrows a known FP8 storage and rejects subsequent writes through any
alias. Unknown kernel contracts fail closed. External model/state/output
ownership remains with the existing serialized GraphFront adapter.
"""
from contextlib import contextmanager
import torch
import capture_body_v1 as body
import nr_backend.triton_fp8 as fp8
from nr_backend.execution import current_arithmetic_backend
from quantization_dataflow_v1 import Dataflow,FP8,tensors


class ProofDataflow(Dataflow):
    def __init__(self):
        super().__init__();self.proofs=[]

    def launch(self,kernel,grid,stream,args):
        super().launch(kernel,grid,stream,args)
        event=self.events[-1]
        if event.get('redundant_fp8_proof'):
            bound=dict(zip(kernel.src.fn.arg_names,args))
            self.proofs.append((bound['X'],bound['Y']))

    def verify_proofs(self):
        for source,output in self.proofs:
            assert source.cpu().numpy().tobytes()==output.cpu().numpy().tobytes()
        return len(self.proofs)


class RewritingDataflow(Dataflow):
    def __init__(self):
        super().__init__();self.protected=set();self.elisions=[];self.quantization_calls=0

    def write(self,t,event,domain=None):
        if self.state(t)['generation'] in self.protected:
            raise RuntimeError('Write through a borrowed FP8 graph value')
        return super().write(t,event,domain)

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}
        for i,spec in enumerate(func._schema.arguments):
            if spec.alias_info is not None and spec.alias_info.is_write:
                value=args[i] if i<len(args) else kwargs.get(spec.name)
                for t in tensors(value):
                    if self.state(t)['generation'] in self.protected:
                        raise RuntimeError('Write through a borrowed FP8 graph value')
        return super().__torch_dispatch__(func,types,args,kwargs)

    @contextmanager
    def installed(self):
        original=fp8.quantize_fp8
        def quantize(x):
            self.quantization_calls+=1
            if x.device.type=='xpu' and x.dtype==torch.float16:
                ref=self.ref(x)
                if ref['domain']==FP8:
                    self.protected.add(ref['storage'])
                    self.elisions.append(dict(input=ref,producer=self.events[ref['producer']]['name']))
                    return x.contiguous()
            return original(x)
        fp8.quantize_fp8=quantize
        try:
            with super().installed():yield self
        finally:
            assert fp8.quantize_fp8 is quantize
            fp8.quantize_fp8=original

    def rewrite_summary(self):
        return dict(**self.summary(),quantization_calls=self.quantization_calls,
                    elided_fp8=len(self.elisions),elisions=self.elisions)


class FP8GraphRewrite:
    def __init__(self,model,provider):
        self.model=model;self.provider=provider;self.original=body.forward_front;self.builds=[]

    def apply(self,module,rgb,front,**options):
        if (module is not self.model or self.provider.mode!='fp16_xmx' or current_arithmetic_backend()!='triton'
                or rgb.device.type!='xpu' or tuple(rgb.shape)!=(256,256,3)
                or tuple(front.shape)!=(320,320,16) or options.get('return_float32',False)):
            return self.original(module,rgb,front,**options)
        analysis=RewritingDataflow()
        with analysis.installed():result=self.original(module,rgb,front,**options)
        self.builds.append(analysis.rewrite_summary())
        return result

    @contextmanager
    def installed(self):
        assert body.forward_front is self.original
        replacement=self.apply;body.forward_front=replacement
        try:yield self
        finally:
            assert body.forward_front is replacement
            body.forward_front=self.original
