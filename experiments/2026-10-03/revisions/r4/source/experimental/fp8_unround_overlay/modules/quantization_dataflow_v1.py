"""Scoped non-timing value provenance for the selected functional NR body.

StorageWeakRef tracks StorageImpl lifetime without retaining GPU allocations.
Storage generation and write revision, never allocator data_ptr, identify values.
Unknown kernels fail closed. ATen schema writes invalidate the whole storage;
partial kernel stores do not establish a whole-storage FP8 proof.
"""
from collections import Counter
from contextlib import contextmanager
import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode
from triton.compiler.compiler import CompiledKernel

FP8='satfinite_e4m3_decoded_half'
# (output parameters, outputs proven FP8 by their complete stores).
# These contracts apply to the authenticated selected source versions only.
CONTRACTS={
 'nr_backend.triton_fp8._kernel':(('Y',),('Y',)),
 'fast_matrices_v3._matmul':(('OUT',),()),
 'window_layout_v1._unpack':(('OUT',),()),
 'fused_qkv_pack_native_half_v1._pack':(('Q','K','V'),('Q','K','V')),
 'fused_swin_heads_native_half_v1._kernel':(('OUT',),()),
 'batched_branched_mlp_v1._pairs':(('LATENT',),('LATENT',)),
 'batched_branched_mlp_v1._project':(('OUT',),()),
 'native_cubic_batched_v1._pairs':(('LATENT',),('LATENT',)),
 'native_cubic_batched_v1._project':(('OUT',),()),
 'native_cubic_split_v1._kernel':(('OUT',),('OUT',)),
 'fused_vit_projection_v2._parts':(('OUT',),()),
 'fused_vit_projection_v2._merge':(('OUT',),()),
 'native_cubic_c32_v1._kernel':(('OUT',),()),
 'fused_c32_projection_native_half_v1._kernel':(('Q','K','V'),('Q','K','V')),
 'fused_swin_core_native_half_v1._kernel':(('OUT',),()),
 'nr_backend.triton_cubic_fp8._direct':(('Y',),('Y',)),
 'fused_vit_qkv_v1._parts':(('OUT',),()),
 'fused_vit_qkv_v1._prepare':(('Q','K','V'),('Q','K','V')),
 'fused_vit_attention64_v2._kernel':(('OUT',),('OUT',)),
 'nr_backend.triton_half_fma._half_fma':(('OUT',),()),
 'nr_backend.triton_math._tiled':(('OUT',),()),
}
COPY_OPS={'aten.reshape.default','aten.contiguous.default','aten.clone.default',
          'aten.repeat_interleave.self_int','aten.to.dtype','aten.to.device'}

def tensors(value):
    if isinstance(value,torch.Tensor):
        if value.device.type=='xpu':yield value
    elif isinstance(value,(tuple,list)):
        for v in value:yield from tensors(v)
    elif isinstance(value,dict):
        for v in value.values():yield from tensors(v)


class Dataflow(TorchDispatchMode):
    def __init__(self):
        super().__init__();self.states={};self.events=[];self.generation=0

    def state(self,t):
        storage=t.untyped_storage();key=storage._cdata
        state=self.states.get(key)
        if state is None or state['weak'].expired():
            self.generation+=1
            state=dict(weak=StorageWeakRef(storage),generation=self.generation,revision=0,
                       producer=None,domain=None,nbytes=storage.nbytes())
            self.states[key]=state
        return state

    def ref(self,t):
        s=self.state(t)
        return dict(storage=s['generation'],revision=s['revision'],producer=s['producer'],
            domain=s['domain'] if t.dtype==torch.float16 else None,
            shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),offset=t.storage_offset(),
            logical_bytes=t.numel()*t.element_size(),storage_bytes=s['nbytes'])

    def write(self,t,event,domain=None):
        s=self.state(t);s['revision']+=1;s['producer']=event
        complete=t.is_contiguous() and t.storage_offset()==0 and t.numel()*t.element_size()==s['nbytes']
        s['domain']=domain if complete and t.dtype==torch.float16 else None
        return self.ref(t)

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {};inputs=list(tensors((args,kwargs)))
        before=[self.ref(t) for t in inputs]
        result=func(*args,**kwargs);outputs=list(tensors(result))
        if not inputs and not outputs:return result
        index=len(self.events);name=str(func)
        event=dict(id=index,kind='aten',name=name,reads=before,writes=[],outputs=[])
        self.events.append(event)
        mutated=[]
        for i,spec in enumerate(func._schema.arguments):
            if spec.alias_info is not None and spec.alias_info.is_write:
                value=args[i] if i<len(args) else kwargs.get(spec.name)
                mutated.extend(tensors(value))
        touched=set()
        for t in mutated:
            s=self.state(t)
            if s['generation'] not in touched:
                event['writes'].append(self.write(t,index));touched.add(s['generation'])
        known=(len(before)==1 and before[0]['domain']==FP8)
        preserving=name in COPY_OPS
        if name=='aten.pad.default':
            mode=args[2] if len(args)>2 else kwargs.get('mode','constant')
            value=args[3] if len(args)>3 else kwargs.get('value')
            preserving=mode=='constant' and value in (None,0)
        input_storages={v['storage'] for v in before}
        for t in outputs:
            s=self.state(t)
            if s['generation'] not in input_storages and s['generation'] not in touched:
                event['writes'].append(self.write(t,index,FP8 if known and preserving else None));touched.add(s['generation'])
            event['outputs'].append(self.ref(t))
        return result

    def launch(self,kernel,grid,stream,args):
        jit=kernel.src.fn;fn=jit.fn;name=f'{fn.__module__}.{fn.__name__}'
        assert name in CONTRACTS,('Unreviewed kernel',name)
        assert len(args)==len(jit.arg_names),(name,len(args),len(jit.arg_names))
        bound=dict(zip(jit.arg_names,args));outs,quantized=CONTRACTS[name]
        if name=='fused_vit_attention64_v2._kernel':assert bound['DEBUG'] is False
        index=len(self.events)
        reads={n:self.ref(t) for n,t in bound.items() if isinstance(t,torch.Tensor) and t.device.type=='xpu' and n not in outs}
        event=dict(id=index,kind='triton',name=name,grid=list(grid),reads=reads,writes={})
        if name=='nr_backend.triton_fp8._kernel':
            event['redundant_fp8_proof']=reads['X']['domain']==FP8
        self.events.append(event)
        for n in outs:
            domain=FP8 if n in quantized else None
            if name=='window_layout_v1._unpack' and reads['X']['domain']==FP8:domain=FP8
            event['writes'][n]=self.write(bound[n],index,domain)

    @contextmanager
    def installed(self):
        original=CompiledKernel.launch_metadata
        def observe(kernel,grid,stream,*args):
            result=original(kernel,grid,stream,*args)
            self.launch(kernel,grid,stream,args)
            return result
        CompiledKernel.launch_metadata=observe
        try:
            with self:yield self
        finally:
            assert CompiledKernel.launch_metadata is observe
            CompiledKernel.launch_metadata=original

    def summary(self):
        fp8=[e for e in self.events if e['name']=='nr_backend.triton_fp8._kernel']
        producers=Counter('external' if e['reads']['X']['producer'] is None else self.events[e['reads']['X']['producer']]['name'] for e in fp8)
        return dict(triton_calls=sum(e['kind']=='triton' for e in self.events),aten_calls=sum(e['kind']=='aten' for e in self.events),
            standalone_fp8=len(fp8),proven_redundant_fp8=sum(e['redundant_fp8_proof'] for e in fp8),
            fp8_input_producers=dict(producers.most_common()),storage_generations=self.generation)
