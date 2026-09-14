"""Only requested C512 queries; full 64-key/value context remains intact."""
import torch
from nr_backend.attention import normalize_c32, normalize_attention_weights, score_exponential
from nr_backend.tensor_math import sm89_f16_batched_dot
import nr_backend.multihead_block as multi
from .layout import Layout


class CompactLayout(Layout):
    def __init__(self, model):
        super().__init__(model)
        self._indices = {}
        self.geometry = None

    def split(self, module, features):
        previous = self.geometry
        self.geometry = (*features.shape[:2], *module.window_shift)
        try:
            return super().split(module, features)
        finally:
            self.geometry = previous

    def windows(self, module, features):
        h, w, channels = features.shape
        rows, cols = h//8, w//8
        z = multi.sm89_f16_dot(multi.quantize_fp8(features), module.qkv, chunk_k=16).reshape(h, w, module.heads, 3, 32)
        q = multi.quantize_fp8((normalize_c32(z[:, :, :, 0])*module.scale[None, None, :, None]).half())
        k = multi.quantize_fp8(normalize_c32(z[:, :, :, 1]))
        v = multi.quantize_fp8(z[:, :, :, 2])
        def window(t):
            return t.reshape(rows, 8, cols, 8, module.heads, 32).permute(4, 0, 2, 1, 3, 5).reshape(module.heads, rows, cols, 64, 32)[..., module.pixel_order, :].reshape(-1, 64, 32)
        q, k, v = window(q), window(k), window(v)
        key = (h, w, module.heads, self.geometry, features.device)
        groups = self._indices.get(key)
        if groups is None:
            height, width, sy, sx = self.geometry
            order = [base+g%4+8*(g//4)+16*word for base in (0,4,32,36) for word in range(2) for g in range(8)]
            grouped = {}
            for head in range(module.heads):
                for wy in range(rows):
                    for wx in range(cols):
                        valid = tuple(i for i, pixel in enumerate(order)
                                      if sy <= wy*8+pixel//8 < sy+height and sx <= wx*8+pixel%8 < sx+width)
                        if valid:
                            grouped.setdefault(valid, []).append((head*rows*cols+wy*cols+wx, head))
            groups = [(torch.tensor([p[0] for p in pairs], device=features.device),
                       torch.tensor([p[1] for p in pairs], device=features.device),
                       torch.tensor(indices, device=features.device)) for indices, pairs in grouped.items()]
            self._indices[key] = groups
        output = torch.zeros_like(q)
        for batches, heads, query_ids in groups:
            queries = q.index_select(0, batches).index_select(1, query_ids)
            keys, values = k.index_select(0, batches), v.index_select(0, batches)
            bias = module.bias.index_select(0, heads).index_select(1, query_ids)
            scores = sm89_f16_batched_dot(queries, keys.transpose(-1, -2), initial=bias)
            weights = normalize_attention_weights(score_exponential(scores))
            output[batches[:, None], query_ids[None, :]] = sm89_f16_batched_dot(weights, values)
        return output.reshape(module.heads, rows, cols, 64, 32)
