"""Opt-in real-shape tile policy on the already validated fast arithmetic path.

Only explicitly measured dense geometries change; attention, K8, quantization,
cache invalidation, model boundaries and unlisted geometries keep their paths.
"""
import json
from pathlib import Path
from strided_batched_v2 import StridedMatrices
from fused_cached_matrices_v2 import geometry
from fused_activation_int8_v1 import dot as fused
from dense_tiles_v1 import dot


class TiledMatrices(StridedMatrices):
    def __init__(self):
        super().__init__()
        self.policy = json.loads(Path(__file__).with_name('dense_tile_policy_v1.json').read_text(encoding='utf-8'))

    def dense(self, a, w, *, chunk_k, initial=None, **kwargs):
        if chunk_k != 16 or (self.mode != 'fp16_xmx' and not self.expanded):
            return super().dense(a, w, chunk_k=chunk_k, initial=initial, **kwargs)
        k, n = w.shape
        m = a.numel() // k
        mode = 'fp16_xmx' if self.mode == 'fp16_xmx' else 'int8_fused_v2'
        key = f'{m}x{k}x{n}:initial={initial is not None}'
        option = self.policy['modes'][mode].get(key)
        if option is None:
            return super().dense(a, w, chunk_k=chunk_k, initial=initial, **kwargs)
        if k % 16:
            raise ValueError('Unsupported original reduction size')
        packed = None
        int8 = mode == 'int8_fused_v2'
        if int8:
            packed = self.static_cache.lookup(w)
            if packed is None:
                return super().dense(a, w, chunk_k=chunk_k, initial=initial, **kwargs)
            self.record('packed_weight_hit')
        is_fused = int8 and geometry(m, k, n) is not None
        if is_fused != option['fused']:
            raise ValueError('Tile policy arithmetic path mismatch')
        tile = tuple(option['tile'])
        if is_fused:
            out, kernel = fused(a, w, initial=initial, packed=packed, bm=tile[0], bn=tile[1], warps=tile[3])
        else:
            out, kernel = dot(a, w, initial=initial, packed=packed, int8=int8, tile=tile)
        kind = ('int8_fused' if is_fused else 'int8_separate') if int8 else 'fp16_dense'
        self.record(kind)
        if int8:
            self.record('int8_dense')
        self.record('tiled_dense')
        if kind not in self.compiled:
            self.compiled[kind] = kernel
        return out
