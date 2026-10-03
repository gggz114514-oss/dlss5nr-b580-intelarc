"""Owned NR256 post dependency crop; preserve complete 8x8 attention windows.

RGB pixels [0,256) correspond to padded queries [4,260). Their complete windows
are [0,264), so merged pixels [0,260) and coarse pixels [0,130) suffice. The
right/bottom context is real input, not new zeros. Other APIs retain full heads.
"""
from contextlib import contextmanager
import torch
import nr_backend.post as post
import native_cubic_adapters_v1 as cubic_adapters
from nr_backend.execution import current_arithmetic_backend


class PostRegion:
    def __init__(self,model,provider,enabled=True):
        self.model=model;self.provider=provider;self.module=model.post
        self.original=self.module.forward;self.enabled=enabled;self.calls=0
        self.geometries=[]

    def head(self,features,skip):
        module=self.module
        assert tuple(features.shape)==(160,160,32) and tuple(skip.shape)==(320,320,32)
        assert module.body.window_shift==(4,4)
        # Every operation before attention acts independently per pixel/channel.
        expanded=(post.q(features[:130,:130]).repeat_interleave(2,0).repeat_interleave(2,1)*module.input_weight).half()
        merged=post.half_fma(post.q(skip[:260,:260]),module.input_skip_scale,expanded)
        padded=torch.nn.functional.pad(merged,(0,0,4,0,4,0))
        assert tuple(padded.shape)==(264,264,32)
        old_rows=cubic_adapters.C32_ROWS
        cubic_adapters.C32_ROWS=old_rows|{264*264}
        try:
            mlp=module.body.mlp.forward_unquantized(padded)
        finally:
            cubic_adapters.C32_ROWS=old_rows
        attended=module.body.attention(post.q(mlp))
        # Projection and K8 head do not mix spatial positions; omit unused rows.
        attended_roi=attended[4:260,4:260]
        initial=(mlp[4:260,4:260]*module.body.skip_scale).half()
        full=post.dot(post.q(attended_roi),module.body.output_weight,chunk_k=16,initial=initial)
        result=post.dot(full,module.head_weight,chunk_k=8)
        assert tuple(result.shape)==(256,256,8)
        self.calls+=1
        self.geometries.append(dict(coarse=[130,130],merged=[260,260],padded=[264,264],
            attention_windows=33*33,projection=[256,256],head=[256,256,8]))
        return result

    @contextmanager
    def installed(self):
        module=self.module
        assert 'forward' not in module.__dict__ and 'forward_head' not in module.__dict__
        def replacement(features,skip,rgb,**options):
            eligible=(self.enabled and self.provider.mode=='fp16_xmx' and current_arithmetic_backend()=='triton'
                and rgb.device.type=='xpu' and tuple(rgb.shape)==(256,256,3)
                and tuple(features.shape)==(160,160,32) and tuple(skip.shape)==(320,320,32)
                and features.dtype==skip.dtype==torch.float16)
            if not eligible:
                return self.original(features,skip,rgb,**options)
            assert 'forward_head' not in module.__dict__
            head=self.head;module.forward_head=head
            try:
                return self.original(features,skip,rgb,**options)
            finally:
                valid=module.forward_head is head
                del module.forward_head
                assert valid,'Post head override interference'
        module.forward=replacement
        try:
            yield self
        finally:
            valid=module.forward is replacement
            del module.forward
            assert valid,'Post forward override interference'

    def verify_restored(self):
        assert 'forward' not in self.module.__dict__ and 'forward_head' not in self.module.__dict__
