"""Owned complete body with direct C512 attention-window consumers.

Only the rewrite's inner body callable is replaced, so ShortFP8/provenance stay
active. Public SplitSwinBlock forward and all-boundaries methods stay intact.
Probe-only unpack exposes every original boundary plus unquantized full; it is
disabled before graph capture and timing. ViT layout and post crop are off.
"""
from contextlib import contextmanager
import torch
import nr_backend.split_block as blocks
from nr_backend.execution import current_arithmetic_backend,record_arithmetic_dispatch
from c512_window_projection_v1 import forward as project
from window_layout_v1 import unpack
from quantization_dataflow_v1 import CONTRACTS

EXTRA={'c512_window_projection_v1._project':(('OUT',),())}


class C512WindowLayout:
    def __init__(self,stack,enabled=True):
        self.stack=stack;self.model=stack.model;self.provider=stack.provider;self.enabled=enabled
        self.original=stack.rewrite.original
        self.modules={id(b):f'{group}.{i}' for group in ('encoder512','decoder512')
                      for i,b in enumerate(getattr(self.model,group))}
        assert len(self.modules)==16
        self.calls=0;self.blocks={};self.resources={};self.probe=None;self.selections={}

    def split(self,module,features):
        assert id(module) in self.modules and tuple(features.shape)==(12,12,512)
        h,w=features.shape[:2];sy,sx=module.window_shift
        ffwd=module.ffwd(features)
        mlp=module.ffwd_projection(ffwd,features)
        padded=torch.nn.functional.pad(mlp,(0,0,sx,(-w-sx)%8,sy,(-h-sy)%8))
        # Existing producer preserves half attention then rounds FP8 in its store.
        packed=self.stack.window_blocks.windows(module.attention,padded)
        full,kernel,selection=project(packed,module.projection.weight,blocks.q(mlp),
                                     module.projection.skip_scale,module.attention.pixel_inverse,shift=(sy,sx))
        self.resources[kernel.hash]=dict(spills=kernel.n_spills,registers=kernel.n_regs,shared_bytes=kernel.metadata.shared)
        name=self.modules[id(module)];self.selections[name]=selection
        self.provider.record('fp16_dense')
        record_arithmetic_dispatch('dense')
        # Logical attention rounding and repeated projection input rounding are
        # fused/proven in the producer; these receipts are not GPU launch counts.
        for _ in range(2):record_arithmetic_dispatch('fp8')
        output=blocks.q(full)
        pool=final=None
        if module.final_weight is not None:
            top=(full[0::2,0::2]+full[0::2,1::2]).half()
            bottom=(full[1::2,0::2]+full[1::2,1::2]).half()
            pool=blocks.q(((top+bottom).half()*.25).half())
            pool=torch.nn.functional.pad(pool,(0,0,0,(-pool.shape[1])%4,0,(-pool.shape[0])%4))
            final=blocks.q(blocks.dot(pool,module.final_weight,chunk_k=16))
        self.calls+=1;self.blocks[name]=self.blocks.get(name,0)+1
        if self.probe is not None:
            hp,wp=padded.shape[:2]
            attended=unpack(packed,module.attention.pixel_inverse).reshape(hp,wp,512)[sy:sy+h,sx:sx+w].contiguous()
            outputs=(ffwd,mlp,attended,output) if final is None else (ffwd,mlp,attended,output,pool,final)
            self.probe(name,module,features,outputs,full,packed)
        return output,output if final is None else final

    def forward_front(self,module,rgb,front,**options):
        if (not self.enabled or module is not self.model or self.provider.mode!='fp16_xmx'
                or current_arithmetic_backend()!='triton' or rgb.device.type!='xpu'
                or tuple(rgb.shape)!=(256,256,3) or tuple(front.shape)!=(320,320,16)
                or options.get('return_float32',False)):
            return self.original(module,rgb,front,**options)
        pre_skip,x=module.pre.forward_features_outputs(front);del front;skips=[]
        for c,group in zip((32,64,128,256),module.encoder):
            for block in group:
                skip,down=block.forward_outputs(x);x=down if down is not None else skip
            skips.append(skip)
        for i,block in enumerate(module.encoder512):
            output,x=self.split(block,x)
            if i==7:skip512=output
        vit_shape=x.shape;x=x.reshape(-1,1024)
        for block in module.vit:x=block(x)
        x=module.decoder_input(x.reshape(vit_shape),skip512)
        for block in module.decoder512:output,x=self.split(block,x)
        for c,group,skip in zip((256,128,64,32),module.decoder,reversed(skips)):
            x=group[0](x,skip)
            for block in group[1:]:x=block(x)
        skips.clear();del skip,skip512,output
        return module.post(x,pre_skip,rgb,**options)

    @contextmanager
    def installed(self):
        assert self.stack.rewrite.original is self.original and not any(k in CONTRACTS for k in EXTRA)
        methods=(blocks.SplitSwinBlock.forward,blocks.SplitSwinBlock.forward_boundaries)
        assert self.stack.window_blocks.layout.native_normalize and self.stack.window_blocks.layout.native_swin
        replacement=self.forward_front
        CONTRACTS.update(EXTRA);self.stack.rewrite.original=replacement
        try:
            yield self
        finally:
            valid=self.stack.rewrite.original is replacement and methods==(blocks.SplitSwinBlock.forward,blocks.SplitSwinBlock.forward_boundaries)
            self.stack.rewrite.original=self.original
            for k,v in EXTRA.items():assert CONTRACTS.pop(k)==v
            assert valid,'C512 body scope interference'

    def verify_restored(self):
        assert self.stack.rewrite.original is self.original and not any(k in CONTRACTS for k in EXTRA)
