"""Selected NR body with a native-aware FP8 analysis and local packed ViT chain.

The source body order is unchanged outside its eight ViT blocks. Native writes
are explicitly recorded, never inferred from an allocation or an untracked
device pointer. All contracts and method overrides are scoped to construction.
"""
from contextlib import contextmanager
import torch
from fp8_graph_rewrite_v1 import RewritingDataflow,FP8GraphRewrite
from quantization_dataflow_v1 import CONTRACTS,FP8
from nr_backend.execution import current_arithmetic_backend
from esimd_vit_chain_v1 import VitChain,CONTRACTS as CHAIN_CONTRACTS


class NativeDataflow(RewritingDataflow):
    def native_dense(self,a,b,out):
        assert a.is_contiguous() and b.is_contiguous() and out.is_contiguous()
        assert all(t.dtype==torch.float16 for t in (a,b,out))
        assert a.shape==(8,64,8,16) and b.numel()==1024*4096 and out.shape==(64,4096)
        assert self.ref(a)['domain']==FP8
        assert self.state(out)['generation'] not in {self.state(a)['generation'],self.state(b)['generation']}
        index=len(self.events)
        event=dict(id=index,kind='native',name='esimd_vit_expand_no_initial',reads={'A':self.ref(a),'B':self.ref(b)},writes={})
        self.events.append(event)
        event['writes']['OUT']=self.write(out,index,None)

    def rewrite_summary(self):
        return dict(**super().rewrite_summary(),native_dense_calls=sum(e['kind']=='native' for e in self.events))


def forward_front(model,rgb,front,chain,analysis,*,previous=None,sigmoid=None,blend_scale=None,history_reciprocal=None,return_float32=False):
    pre_skip,x=model.pre.forward_features_outputs(front);del front;skips=[]
    for c,group in zip((32,64,128,256),model.encoder):
        for block in group:
            skip,down=block.forward_outputs(x);x=down if down is not None else skip
        skips.append(skip)
    for i,block in enumerate(model.encoder512):
        values=block.forward_boundaries(x);x=values[-1]
        if i==7:skip512=values[3]
    vit_shape=x.shape;x=chain(x.reshape(-1,1024),analysis)
    x=model.decoder_input(x.reshape(vit_shape),skip512)
    for block in model.decoder512:x=block(x)
    for c,group,skip in zip((256,128,64,32),model.decoder,reversed(skips)):
        x=group[0](x,skip)
        for block in group[1:]:x=block(x)
    skips.clear();del skip,skip512,values
    return model.post(x,pre_skip,rgb,previous=previous,sigmoid=sigmoid,blend_scale=blend_scale,history_reciprocal=history_reciprocal,return_float32=return_float32)


class EsimdVitGraph(FP8GraphRewrite):
    def __init__(self,model,provider):
        super().__init__(model,provider);self.chain=VitChain(model,provider)

    def apply(self,module,rgb,front,**options):
        if (module is not self.model or self.provider.mode!='fp16_xmx' or current_arithmetic_backend()!='triton'
                or rgb.device.type!='xpu' or tuple(rgb.shape)!=(256,256,3)
                or tuple(front.shape)!=(320,320,16) or options.get('return_float32',False)):
            return super().apply(module,rgb,front,**options)
        analysis=NativeDataflow()
        with analysis.installed():result=forward_front(module,rgb,front,self.chain,analysis,**options)
        summary=analysis.rewrite_summary();assert summary['native_dense_calls']==8
        self.builds.append(summary)
        return result

    @contextmanager
    def installed(self):
        assert not any(k in CONTRACTS for k in CHAIN_CONTRACTS)
        CONTRACTS.update(CHAIN_CONTRACTS)
        try:
            with super().installed():yield self
        finally:
            for k,v in CHAIN_CONTRACTS.items():assert CONTRACTS.pop(k)==v

    def close(self):self.chain.close()
