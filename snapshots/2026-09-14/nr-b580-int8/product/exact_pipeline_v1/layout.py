"""C512/ViT exact producers feed layout-aware projection without unpacking."""
from contextlib import contextmanager
from types import MethodType
import torch
import nr_backend.multihead_block as multi
import nr_backend.split_block as split
import nr_backend.vit_block as vit
from nr_backend.attention import normalize_c32
from .layout_projection import project


class Layout:
    def __init__(self, model):
        self.model = model
        self.calls = 0

    def windows(self, module, features):
        h, w, channels = features.shape
        rows, cols = h//8, w//8
        z = multi.sm89_f16_dot(multi.quantize_fp8(features), module.qkv, chunk_k=16).reshape(h, w, module.heads, 3, 32)
        q = multi.quantize_fp8((normalize_c32(z[:, :, :, 0])*module.scale[None, None, :, None]).half())
        k = multi.quantize_fp8(normalize_c32(z[:, :, :, 1]))
        v = multi.quantize_fp8(z[:, :, :, 2])
        def window(t):
            return t.reshape(rows, 8, cols, 8, module.heads, 32).permute(4, 0, 2, 1, 3, 5).reshape(module.heads, rows, cols, 64, 32)[..., module.pixel_order, :]
        q, k, v = window(q), window(k), window(v)
        return torch.stack([multi.swin_attention_windows(q[i], k[i], v[i], module.bias[i]) for i in range(module.heads)])

    def split(self, module, features):
        h, w = features.shape[:2]
        sy, sx = module.window_shift
        ffwd = module.ffwd(features)
        mlp = module.ffwd_projection(ffwd, features)
        padded = torch.nn.functional.pad(mlp, (0, 0, sx, (-w-sx)%8, sy, (-h-sy)%8))
        packed = self.windows(module.attention, padded)
        full = project(packed, module.projection.weight,
                       (split.q(mlp)*module.projection.skip_scale).half(),
                       geometry=(h, w, sy, sx), inverse=module.attention.pixel_inverse)
        output = split.q(full)
        final = output
        if module.final_weight is not None:
            top = (full[0::2, 0::2]+full[0::2, 1::2]).half()
            bottom = (full[1::2, 0::2]+full[1::2, 1::2]).half()
            pool = split.q(((top+bottom).half()*.25).half())
            pool = torch.nn.functional.pad(pool, (0, 0, 0, (-pool.shape[1])%4, 0, (-pool.shape[0])%4))
            final = split.q(split.dot(pool, module.final_weight, chunk_k=16))
        self.calls += 1
        return output, final

    def vit(self, module, features):
        tokens = features.shape[0]
        x = vit.q(features)
        hidden = vit.cubic_quantize(vit.dot(x, module.expand, chunk_k=16))
        mlp = vit.q(vit.split_k_projection(hidden, module.contract, (x*module.ffn_skip).half()))
        z = (vit.dot(mlp[:, :512], module.qkv_weight[:512], chunk_k=16)+vit.dot(mlp[:, 512:], module.qkv_weight[512:], chunk_k=16)).half().reshape(tokens, 32, 3, 32)
        query = vit.q((normalize_c32(z[:, :, 0])*5.65625).half()*module.query_scale[None, :, None])
        key = vit.q(normalize_c32(z[:, :, 1]))
        value = vit.q(z[:, :, 2])
        packed = vit.vit_attention(query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1))
        result = project(packed.contiguous(), module.projection, (mlp*module.attn_skip).half(), parts=4)
        self.calls += 1
        return vit.q(result)

    def forward_body(self, model, rgb, front, **options):
        pre_skip, x = model.pre.forward_features_outputs(front)
        skips = []
        for group in model.encoder:
            for block in group:
                skip, down = block.forward_outputs(x)
                x = down if down is not None else skip
            skips.append(skip)
        for i, block in enumerate(model.encoder512):
            output, x = self.split(block, x)
            if i == 7:
                skip512 = output
        shape = x.shape
        x = x.reshape(-1, 1024)
        for block in model.vit:
            x = self.vit(block, x)
        x = model.decoder_input(x.reshape(shape), skip512)
        for block in model.decoder512:
            _, x = self.split(block, x)
        for group, skip in zip(model.decoder, reversed(skips)):
            x = group[0](x, skip)
            for block in group[1:]:
                x = block(x)
        return model.post(x, pre_skip, rgb, **options)

    @contextmanager
    def installed(self):
        from nr_backend.executor import ResetNR
        original = ResetNR._forward_front
        def forward(model, rgb, front, *, progress=None, **options):
            if model is not self.model or progress is not None:
                return original(model, rgb, front, progress=progress, **options)
            result = self.forward_body(model, rgb, front, **options)
            torch.xpu.synchronize(rgb.device)
            return result
        ResetNR._forward_front = forward
        try:
            yield
        finally:
            ResetNR._forward_front = original
