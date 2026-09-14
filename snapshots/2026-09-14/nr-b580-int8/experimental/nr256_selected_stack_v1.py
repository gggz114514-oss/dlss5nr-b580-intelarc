"""Reusable construction of the validated c5ac61b serial NR256 FP16 stack.

The caller selects and authenticates the isolated toolchain before importing.
This factory does not choose asset paths, precision, or execution ownership for
production. Sharing constants must happen before graph-session construction.
"""
from contextlib import ExitStack,contextmanager
from nr_backend.temporal import MotionNR
from current_dense_tiled_provider_v1 import CurrentDenseTiledMatrices
from native_cubic_adapters_v1 import FusedBatched,FusedSplit,FusedC32
from cubic_lut_constant_v1 import register
from fused_vit_projection_v3 import FusedVitProjection
from fused_swin_native_half_v1 import FusedSwin
from native_half_head_layout_v1 import HeadLayout
from fused_dynamic_front_v1 import FusedFront
from graph_front_v6 import GraphFront
from fused_graph_history_warp_v2 import FusedHistoryWarp
from fused_vit_attention64_adapter_v2 import FusedVitQKV
from fp8_graph_rewrite_v1 import FP8GraphRewrite
from shared_model_buffers_v1 import share_identical_buffers


class Stack:
    def __init__(self,exact_root,*,rewrite_type=FP8GraphRewrite,share_with=None):
        self.model=MotionNR.from_assets(exact_root/'model-assets/sf-v2/WEIGHTS_HT.bin',
            exact_root/'model-assets/noise-sm89-v2',exact_root/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
        register(self.model)
        self.shared=None if share_with is None else share_identical_buffers(self.model,share_with.model)
        self.provider=CurrentDenseTiledMatrices(self.model);self.provider.select('fp16_xmx')
        self.graph=GraphFront(self.model,arithmetic=self.provider)
        self.warp=FusedHistoryWarp(self.model)
        self.rewrite=rewrite_type(self.model,self.provider)
        self.components=[self.provider,FusedSwin(self.provider),HeadLayout(self.model,self.provider),self.graph,
            FusedBatched(self.model,self.provider,workload='small'),FusedSplit(self.model,self.provider),
            FusedC32(self.model,self.provider,bm=32,warps=4,stages=1),FusedVitProjection(self.model,self.provider),
            FusedFront(self.model),self.warp,self.rewrite,FusedVitQKV(self.model,self.provider)]

    @contextmanager
    def installed(self):
        with ExitStack() as stack:
            for item in self.components:stack.enter_context(item.installed())
            yield self

    def close(self):
        self.warp.close();self.graph.close()
