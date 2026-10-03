"""Capture-only, NR256 ViT scope with the screened P4 INT8 FFN kernels.

Static-body experiment only: external packed constants are explicitly owned and
checked by this scope. They are not yet part of GraphFront's production signature.
Each logical invocation gets fresh intermediates; captured graphs own their pool
storage. No debug/P1 buffers or half hidden tensor are allocated in this route.
"""
from contextlib import contextmanager
import numpy as np
import torch
import triton
import nr_backend.vit_block as vit
from nr_backend.execution import current_arithmetic_backend,record_arithmetic_dispatch
from vit_head_layout_scope_v1 import VitHeadLayout,prepare,attend,project
from quantization_dataflow_v1 import CONTRACTS,Dataflow
from spill_preflight_v1 import select
import int8_ffn_segment_gpu_v1 as kernels
import int8_ffn_segment_oracle_v1 as oracle

EXTRA={
    'int8_ffn_segment_gpu_v1._entry':(('Q','S'),()),
    'int8_ffn_segment_gpu_v1._expand':(('QH',),()),
    'int8_ffn_segment_gpu_v1._contract':(('PARTIAL',),()),
    'int8_ffn_segment_gpu_v1._merge':(('OUT',),('OUT',)),
}


class Int8VitLayout(VitHeadLayout):
    def __init__(self,model,provider,hidden_scales):
        super().__init__(model,provider)
        assert len(hidden_scales)==len(model.vit)==8
        self.launch_original=Dataflow.launch
        self.packed=[];self.ffn_resources={};self.selections={};self.int8_probe=None
        # Construct before capture and outside inference_mode: version guards need
        # ordinary tensor counters. Pack once, without saving another weight set.
        assert not torch.is_inference_mode_enabled()
        for module,sh in zip(model.vit,hidden_scales):
            assert sh.shape in ((1,4096),(4096,)) and sh.dtype==np.float32
            assert np.isfinite(sh).all() and (sh>0).all()
            assert module.expand.device.type=='xpu'
            e=module.expand.cpu().numpy();c=module.contract.cpu().numpy()
            assert e.shape==(1024,4096) and c.shape==(4096,1024)
            assert e.dtype==c.dtype==np.float16
            assert module.ffn_skip.shape==(1024,) and module.ffn_skip.dtype==torch.float16
            assert module.ffn_skip.is_contiguous()
            qe,se=oracle.quantize_axis(e,0)
            qc,sc=oracle.quantize_axis(c.astype(np.float32)*sh.reshape(-1,1),0)
            values=[torch.from_numpy(np.ascontiguousarray(a)).to(module.expand.device)
                    for a in (qe.T,se.reshape(-1),sh.reshape(-1),qc.T,sc.reshape(-1))]
            values.append(module.ffn_skip)
            self.packed.append(tuple(values))
        self.constant_signature=self.signature()

    def signature(self):
        return tuple((id(t),t.data_ptr(),tuple(t.shape),tuple(t.stride()),t.dtype,t.device,t._version)
                     for values in self.packed for t in values)

    def validate_constants(self):
        assert self.signature()==self.constant_signature,'INT8 packed constants changed'

    def ffn(self,index,x):
        assert x.shape==(64,1024) and x.dtype==torch.float16 and x.is_contiguous()
        we,se,sh,wc,sc,skip=self.packed[index]
        qx=torch.empty((64,1024),dtype=torch.int8,device=x.device)
        sx=torch.empty(64,dtype=torch.float32,device=x.device)
        qh=torch.empty((64,4096),dtype=torch.int8,device=x.device)
        partial=torch.empty((4,64,1024),dtype=torch.int32,device=x.device)
        out=torch.empty_like(x)

        def launch(label,jit,configs,args_for,grid_for):
            options=dict(num_warps=4,num_stages=1,enable_fp_fusion=False)
            config,kernel,selection=select(jit,configs,args_for,grid_for,**options)
            self.ffn_resources[label+':'+kernel.hash]=dict(spills=kernel.n_spills,
                registers=kernel.n_regs,shared_bytes=kernel.metadata.shared)
            self.selections[label]=selection
            assert jit[grid_for(config)](*args_for(config),**options) is kernel
        launch('entry',kernels._entry,[(1,)],lambda c:(x,qx,sx,64,1024),lambda c:(64,))
        # Unused debug pointers alias input, never outputs: DEBUG=false is guarded
        # before dispatch in installed(), and contracts describe only actual writes.
        launch('expand',kernels._expand,[(32,32),(16,32)],
            lambda c:(qx,sx,we,se,sh,qh,x,64,1024,4096,*c,False),
            lambda c:(triton.cdiv(64,c[0]),triton.cdiv(4096,c[1])))
        launch('contract_p4',kernels._contract,[(32,32),(16,32)],
            lambda c:(qh,wc,sc,x,skip,partial,out,x,64,4096,1024,4,*c,False),
            lambda c:(triton.cdiv(64,c[0]),triton.cdiv(1024,c[1]),4))
        launch('merge',kernels._merge,[(512,)],
            lambda c:(partial,sc,x,skip,out,x,64,1024,False),lambda c:(128,))
        return out,qh

    def apply(self,module,features):
        if (not self.enabled or id(module) not in self.modules or self.provider.mode!='fp16_xmx'
                or current_arithmetic_backend()!='triton' or features.device.type!='xpu'
                or tuple(features.shape)!=(64,1024)):
            return self.original(module,features)
        index=self.modules[id(module)]
        x=vit.q(features)
        mlp,qh=self.ffn(index,x)
        mlp=vit.q(mlp)
        record_arithmetic_dispatch('int8_ffn_segment')
        query,key,value=prepare(mlp,module.qkv_weight,module.query_scale,bm=32,bn=64,rows=16)[0]
        for _ in range(2):record_arithmetic_dispatch('dense')
        for _ in range(2):record_arithmetic_dispatch('attention_normalize_c32')
        for _ in range(3):record_arithmetic_dispatch('fp8')
        heads=attend(query.transpose(0,1),key.transpose(0,1),value.transpose(0,1),bm=32,warps=4,stages=1)[0]
        for kind in ('batched','attention_exp_vit','fp8','batched','attention_row_sum64','fp8'):
            record_arithmetic_dispatch(kind)
        projected,compiled=project(heads,module.projection,(mlp*module.attn_skip).half())
        kernel=compiled[0]
        self.resources[kernel.hash]=dict(spills=kernel.n_spills,registers=kernel.n_regs,shared_bytes=kernel.metadata.shared)
        for _ in range(4):record_arithmetic_dispatch('dense')
        output=vit.q(projected)
        self.calls+=1;self.blocks[index]=self.blocks.get(index,0)+1
        if self.int8_probe is not None:self.int8_probe(index,x,mlp,output,qh)
        return output

    @contextmanager
    def installed(self):
        self.validate_constants()
        assert Dataflow.launch is self.launch_original and not any(k in CONTRACTS for k in EXTRA)
        original=self.launch_original
        def checked(analysis,kernel,grid,stream,args):
            jit=kernel.src.fn;name=f'{jit.fn.__module__}.{jit.fn.__name__}'
            if name in EXTRA:
                bound=dict(zip(jit.arg_names,args))
                assert bound['M']==64
                if 'DEBUG' in bound:assert bound['DEBUG'] is False
                if 'PARTS' in bound:assert bound['PARTS']==4
                if 'H' in bound:assert bound['H']==4096
                if 'K' in bound:assert bound['K']==1024
                if 'N' in bound:assert bound['N']==1024
            return original(analysis,kernel,grid,stream,args)
        CONTRACTS.update(EXTRA);Dataflow.launch=checked
        try:
            with super().installed():yield self
        finally:
            valid=Dataflow.launch is checked
            Dataflow.launch=original
            for k,v in EXTRA.items():assert CONTRACTS.pop(k)==v
            assert valid,'INT8 dataflow scope interference'
            self.validate_constants()

    def verify_restored(self):
        super().verify_restored()
        assert Dataflow.launch is self.launch_original and not any(k in CONTRACTS for k in EXTRA)
        self.validate_constants()
