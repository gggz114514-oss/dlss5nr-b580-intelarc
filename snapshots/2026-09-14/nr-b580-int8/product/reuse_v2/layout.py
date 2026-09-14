"""Local FP8 producer proofs and compact projection; no pointer-based elision."""
import torch
import nr_backend.split_block as split
import nr_backend.vit_block as vit
from nr_backend.attention import normalize_c32
from .compact import CompactLayout
from .projection import project
from .math import dot_q


class ReuseLayout(CompactLayout):
    dataflow = True
    compact_output = True
    tile = False

    def split(self, module, features):
        previous = self.geometry
        h, w = features.shape[:2]
        sy, sx = module.window_shift
        self.geometry = (h,w,sy,sx)
        try:
            if self.dataflow:
                # Entry q remains: public input has no assumed FP8 proof.
                x = split.q(features)
                z = dot_q(x, module.ffwd.linear)
                chunks = []
                for i in range(8):
                    hidden = split.cubic_quantize(split.dot(z[...,i*64:(i+1)*64], module.ffwd.expand[i], chunk_k=16))
                    chunks.append(dot_q(hidden, module.ffwd.reduce[i]))
                ffwd = torch.cat(chunks, -1)  # concatenation preserves the FP8 domain
                mlp = dot_q(ffwd, module.ffwd_projection.weight,
                            initial=(x*module.ffwd_projection.skip_scale).half())
            else:
                ffwd = module.ffwd(features)
                mlp = module.ffwd_projection(ffwd, features)
            padded = torch.nn.functional.pad(mlp, (0,0,sx,(-w-sx)%8,sy,(-h-sy)%8))
            packed = self.windows(module.attention, padded)  # q output + zero padding is FP8
            residual = mlp if self.dataflow else split.q(mlp)
            full = project(packed, module.projection.weight, (residual*module.projection.skip_scale).half(),
                           geometry=(h,w,sy,sx), inverse=module.attention.pixel_inverse,
                           bm=8 if self.tile else 4)
            output = split.q(full)
            final = output
            if module.final_weight is not None:
                # Pool uses the unquantized full result, not output.
                top = (full[0::2,0::2]+full[0::2,1::2]).half()
                bottom = (full[1::2,0::2]+full[1::2,1::2]).half()
                pool = split.q(((top+bottom).half()*.25).half())
                pool = torch.nn.functional.pad(pool,(0,0,0,(-pool.shape[1])%4,0,(-pool.shape[0])%4))
                final = dot_q(pool,module.final_weight) if self.dataflow else split.q(split.dot(pool,module.final_weight,chunk_k=16))
            self.calls += 1
            return output, final
        finally:
            self.geometry = previous

    def vit(self, module, features):
        # Keep entry quantization and all four ordered half split-K merges.
        tokens = features.shape[0]
        x = vit.q(features)
        hidden = vit.cubic_quantize(vit.dot(x,module.expand,chunk_k=16))
        mlp = vit.q(vit.split_k_projection(hidden,module.contract,(x*module.ffn_skip).half()))
        z = (vit.dot(mlp[:,:512],module.qkv_weight[:512],chunk_k=16)+vit.dot(mlp[:,512:],module.qkv_weight[512:],chunk_k=16)).half().reshape(tokens,32,3,32)
        query = vit.q((normalize_c32(z[:,:,0])*5.65625).half()*module.query_scale[None,:,None])
        key = vit.q(normalize_c32(z[:,:,1]))
        value = vit.q(z[:,:,2])
        packed = vit.vit_attention(query.transpose(0,1),key.transpose(0,1),value.transpose(0,1))
        result = project(packed.contiguous(),module.projection,(mlp*module.attn_skip).half(),parts=4,
                         bm=8 if self.tile else 4)
        self.calls += 1
        return vit.q(result)
