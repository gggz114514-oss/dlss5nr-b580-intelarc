"""Candidate only: independently screened layouts plus post dependency crop.

Install the three scopes only during ShortFP8 body construction, never during
steady graph replay or the public progress fallback. The selected default and
the exact branch are untouched. Complete-call validation is still required.
"""
from contextlib import ExitStack
from nr_backend.execution import current_arithmetic_backend
from nr256_selected_stack_v3 import Stack as SelectedStack
from short_fp8_graph_v1 import ShortFP8Graph
from c512_window_layout_scope_v1 import C512WindowLayout
from vit_head_layout_scope_v1 import VitHeadLayout
from post_region_scope_v1 import PostRegion


class LayoutCropGraph(ShortFP8Graph):
    def attach(self,stack):
        assert not hasattr(self,'layout') and stack.rewrite is self
        self.layout=C512WindowLayout(stack)
        self.vit_layout=VitHeadLayout(stack.model,stack.provider)
        self.post_region=PostRegion(stack.model,stack.provider)

    def apply(self,module,rgb,front,**options):
        eligible=(module is self.model and self.provider.mode=='fp16_xmx'
            and current_arithmetic_backend()=='triton' and rgb.device.type=='xpu'
            and tuple(rgb.shape)==(256,256,3) and tuple(front.shape)==(320,320,16)
            and not options.get('return_float32',False))
        if not eligible:
            return super().apply(module,rgb,front,**options)
        with ExitStack() as scopes:
            for scope in (self.layout,self.vit_layout,self.post_region):
                scopes.enter_context(scope.installed())
            return super().apply(module,rgb,front,**options)

    def metadata(self):
        return dict(super().metadata(),layout_crop=dict(
            c512_calls=self.layout.calls,c512_blocks=self.layout.blocks,
            c512_resources=self.layout.resources,c512_selections=self.layout.selections,
            vit_calls=self.vit_layout.calls,vit_blocks=self.vit_layout.blocks,
            vit_resources=self.vit_layout.resources,post_calls=self.post_region.calls,
            post_geometries=self.post_region.geometries))

    def verify_restored(self):
        for scope in (self.layout,self.vit_layout,self.post_region):
            scope.verify_restored()
        self.fork.verify_restored()

    def close(self):
        self.verify_restored()
        super().close()


class Stack(SelectedStack):
    def __init__(self,exact_root,*,share_with=None):
        super().__init__(exact_root,rewrite_type=LayoutCropGraph,share_with=share_with)
        self.rewrite.attach(self)
