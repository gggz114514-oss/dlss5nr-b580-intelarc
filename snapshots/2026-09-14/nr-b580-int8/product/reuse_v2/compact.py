"""Only requested C512 queries; full 64-key/value context remains intact."""
import torch
from nr_backend.attention import normalize_c32, normalize_attention_weights, score_exponential
from nr_backend.tensor_math import sm89_f16_batched_dot
import nr_backend.multihead_block as multi
from exact_pipeline_v1.layout import Layout
from nr_backend.triton_math import _tiled
import triton


class CompactLayout(Layout):
    dataflow = False
    compact_output = True
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
        z = multi.sm89_f16_dot(features if self.dataflow else multi.quantize_fp8(features), module.qkv, chunk_k=16).reshape(h, w, module.heads, 3, 32)
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
            # Map each logical head/query to its contiguous group result. Only
            # geometry is cached: activations are recomputed every frame.
            mapping = [[-1]*(height*width) for _ in range(module.heads)]
            offset = 0
            for indices, pairs in grouped.items():
                for batch_index, (batch, head) in enumerate(pairs):
                    window_index = batch % (rows*cols)
                    wy, wx = divmod(window_index, cols)
                    for query_index, pixel_index in enumerate(indices):
                        pixel = order[pixel_index]
                        y, x = wy*8+pixel//8-sy, wx*8+pixel%8-sx
                        if mapping[head][y*width+x] != -1:
                            raise RuntimeError('Duplicate compact query')
                        mapping[head][y*width+x] = offset+batch_index*len(indices)+query_index
                offset += len(pairs)*len(indices)
            if any(v < 0 for row in mapping for v in row) or offset != module.heads*height*width:
                raise RuntimeError('Incomplete compact query coverage')
            groups = (groups, torch.tensor(mapping, device=features.device), offset)
            self._indices[key] = groups
        groups, mapping, count = groups
        output = (torch.empty((count,32), device=q.device, dtype=q.dtype)
                  if self.compact_output else torch.zeros_like(q))
        offset = 0
        for batches, heads, query_ids in groups:
            queries = q.index_select(0, batches).index_select(1, query_ids)
            keys, values = k.index_select(0, batches), v.index_select(0, batches)
            bias = module.bias.index_select(0, heads).index_select(1, query_ids)
            scores = sm89_f16_batched_dot(queries, keys.transpose(-1, -2), initial=bias)
            weights = normalize_attention_weights(score_exponential(scores))
            if self.compact_output:
                b, m, inner = weights.shape
                target = output[offset:offset+b*m].view(b,m,32)
                # Original exact tiled kernel, writing directly into a disjoint
                # contiguous result slice. Same K16 math, no scatter/full padding.
                _tiled[(triton.cdiv(m,4),1,b)](weights.contiguous(), values.contiguous(), weights,
                    target, m, 32, inner, 16, False, True, 4, 32,
                    num_warps=1, enable_fp_fusion=False)
                offset += b*m
            else:
                output[batches[:, None], query_ids[None, :]] = sm89_f16_batched_dot(weights, values)
        return (output, mapping) if self.compact_output else output.reshape(module.heads, rows, cols, 64, 32)
