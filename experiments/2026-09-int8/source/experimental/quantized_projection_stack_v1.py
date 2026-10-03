"""Experimental projection-store fusion on the accepted static-range INT8 route."""
from int8_ffn_calibrated_stack_v1 import Stack as RepairedStack
from quantized_projection_store_v1 import ProjectionStores,C512QuantizedLayout


class Stack(RepairedStack):
    def __init__(self,exact_root,*,hidden_scales,share_with=None,c512=False):
        super().__init__(exact_root,hidden_scales=hidden_scales,share_with=share_with)
        self.projection_stores=ProjectionStores(self)
        self.components.append(self.projection_stores)
        self.quantized_c512=bool(c512)
        if c512:self.rewrite.layout=C512QuantizedLayout(self,self.projection_stores)

    def metadata(self):
        return dict(super().metadata(),projection_stores=self.projection_stores.metadata(),
                    quantized_c512=self.quantized_c512,range_calibration_unchanged=True)

    def close(self):
        self.projection_stores.verify_restored()
        super().close()
