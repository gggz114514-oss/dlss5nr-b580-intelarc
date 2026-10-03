"""Keep the measured long-K value-product regression on its original copy path."""
from fused_cached_matrices_v2 import FusedCachedMatrices
from strided_batched_v1 import StridedMatrices as Base


class StridedMatrices(Base):
    def batched(self, a, w, *, initial=None, **kwargs):
        # ViT [32,640,640] @ strided [32,640,32] regressed in primitive v1.
        # Only short K (window products and QK) is enabled pending other evidence.
        if a.shape[-1] > 64 or all(t is None or t.is_contiguous() for t in (a, w, initial)):
            return FusedCachedMatrices.batched(self, a, w, initial=initial, **kwargs)
        return super().batched(a, w, initial=initial, **kwargs)
