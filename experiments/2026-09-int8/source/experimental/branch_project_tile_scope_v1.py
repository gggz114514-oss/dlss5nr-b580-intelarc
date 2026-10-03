"""Serialized capture-only experiment: C256/M576 projection BN64 versus BN32.

Preserve pair kernel, all branch half rounding and logical accounting. Only the
native adapter's already selected geometry changes. No installed source edits.
"""
from contextlib import contextmanager
from triton.compiler.compiler import CompiledKernel
import native_cubic_adapters_v1 as adapters


class ProjectTile:
    def __init__(self, bn):
        if bn not in (32,64):
            raise ValueError('Only baseline64 and candidate32 are supported')
        self.bn=bn
        self.original=adapters.batched_forward
        self.resources={}
        self.calls=0
        self.launches=0

    @contextmanager
    def installed(self):
        assert adapters.batched_forward is self.original
        original_metadata=CompiledKernel.launch_metadata
        def replacement(features,*args,**kwargs):
            assert features.shape[-1]==256 and features.numel()//256==576
            assert kwargs==dict(pair_bm=32,pair_stages=1,project_bm=16,project_bn=64)
            self.calls+=1
            kwargs['project_bn']=self.bn
            return self.original(features,*args,**kwargs)
        def checked(kernel,grid,stream,*args):
            fn=kernel.src.fn.fn
            if (fn.__module__,fn.__name__)==('native_cubic_batched_v1','_project'):
                bound=dict(zip(kernel.src.fn.arg_names,args))
                assert (bound['M'],bound['C'],bound['BM'],bound['BN'])==(576,256,16,self.bn)
                kernel._init_handles()
                record=dict(spills=kernel.n_spills,registers=kernel.n_regs,
                    shared_bytes=kernel.metadata.shared,M=576,C=256,BM=16,BN=self.bn,grid=list(grid))
                self.resources[kernel.hash]=record
                if self.bn==32 and (not isinstance(kernel.n_spills,int) or kernel.n_spills!=0):
                    raise RuntimeError('Candidate projection spill rejected before dispatch: '+str(record))
                self.launches+=1
            return original_metadata(kernel,grid,stream,*args)
        adapters.batched_forward=replacement
        CompiledKernel.launch_metadata=checked
        try:
            yield self
        finally:
            restored=(adapters.batched_forward is replacement and CompiledKernel.launch_metadata is checked)
            adapters.batched_forward=self.original
            CompiledKernel.launch_metadata=original_metadata
            assert restored,'Unexpected interference in projection capture scope'

    def verify_restored(self):
        assert adapters.batched_forward is self.original
