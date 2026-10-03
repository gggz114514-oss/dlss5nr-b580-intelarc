"""Capture-only, serialized NR256 experiment for the equivalent FP8 rounder.

Default selected runtime is unchanged. Explicit preload makes the dependency
fork independent of a prior baseline run. Borrowed FP8 domain analysis remains
enabled. Only genuinely changed compiled kernels are subject to the spill gate.
"""
from contextlib import contextmanager
import importlib
from pathlib import Path
from triton.compiler.compiler import CompiledKernel
from fp8_graph_rewrite_v1 import FP8GraphRewrite
from fork_fp8_jit_v2 import Fork
from nr_backend.execution import current_arithmetic_backend

PRELOAD = (
    'nr_backend.triton_fp8', 'nr_backend.triton_cubic_fp8',
    'fused_branched_mlp_v1', 'fused_c32_mlp_v1', 'fused_c32_mlp_lut_v1',
    'fused_mlp_pair_v1', 'batched_branched_mlp_v1', 'fused_split_ffwd_v1',
    'native_half_cubic_v1', 'native_cubic_c32_v1', 'native_cubic_batched_v1',
    'native_cubic_split_v1', 'nr_backend.triton_attention_weights',
    'fused_swin_core_native_half_v1', 'fused_swin_core_v2',
    'fused_swin_heads_v1', 'fused_qkv_pack_v1', 'fused_c32_qkv_scatter_v2',
    'fused_c32_projection_native_half_v1', 'fused_c32_projection_pack_v1',
    'fused_qkv_pack_native_half_v1', 'fused_swin_heads_native_half_v1',
    'fused_vit_qkv_v1', 'fused_vit_attention64_v2', 'window_block_attention_v3',
)


class ShortFP8Graph(FP8GraphRewrite):
    def __init__(self, model, provider):
        super().__init__(model, provider)
        for name in PRELOAD:
            importlib.import_module(name)
        self.fork = Fork(Path(__file__).resolve().parent.parent)
        self.changed = {new for old, new in self.fork.memo.items() if new is not old}
        self.resources = {}
        self.scopes = 0

    @contextmanager
    def resource_gate(self):
        original = CompiledKernel.launch_metadata
        def checked(kernel, grid, stream, *args):
            if kernel.src.fn in self.changed:
                kernel._init_handles()
                item = dict(name=f'{kernel.src.fn.fn.__module__}.{kernel.src.fn.fn.__name__}',
                    spills=kernel.n_spills, registers=kernel.n_regs, shared_bytes=kernel.metadata.shared)
                self.resources[kernel.hash] = item
                if not isinstance(kernel.n_spills, int) or kernel.n_spills != 0:
                    raise RuntimeError('Short FP8 candidate spill rejected before dispatch: ' + str(item))
            return original(kernel, grid, stream, *args)
        CompiledKernel.launch_metadata = checked
        try:
            yield
        finally:
            assert CompiledKernel.launch_metadata is checked
            CompiledKernel.launch_metadata = original

    def apply(self, module, rgb, front, **options):
        eligible = (module is self.model and self.provider.mode == 'fp16_xmx'
            and current_arithmetic_backend() == 'triton' and rgb.device.type == 'xpu'
            and tuple(rgb.shape) == (256, 256, 3) and tuple(front.shape) == (320, 320, 16)
            and not options.get('return_float32', False))
        if not eligible:
            return super().apply(module, rgb, front, **options)
        with self.fork.installed(), self.resource_gate():
            self.scopes += 1
            return super().apply(module, rgb, front, **options)

    def close(self):
        self.fork.verify_restored()

    def metadata(self):
        return dict(forks=self.fork.forks, bindings=len(self.fork.bindings),
            sources=self.fork.sources, resources=self.resources, capture_scopes=self.scopes)
