"""Candidate fixed-network-size fast NR modes for a variable game render.

The network runs on the selected active image, never on the old 256-square
canvas. 540p needs two edge-replicated rows on each side for the model's 8x8
windows; 480p uses the observed 480x864 model shape with five edge columns
on each side of an 854x480 active image. 360p uses 640x360 directly.

The 720p, 540p and 360p internal padding contracts are derived, not NVIDIA-observed.
Only expose a mode in game after its offline compile and DiskOnly check pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import ExitStack, nullcontext, contextmanager
import gc
import os
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_MODULE = ROOT / "game"


@dataclass(frozen=True)
class Mode:
    height: int
    active: tuple[int, int]
    model: tuple[int, int]
    internal: tuple[int, int]
    inset: tuple[int, int]


MODES = {
    720: Mode(720, (720, 1280), (720, 1280), (768, 1280), (0, 0)),
    540: Mode(540, (540, 960), (544, 960), (640, 1024), (2, 0)),
    480: Mode(480, (480, 854), (480, 864), (512, 896), (0, 5)),
    360: Mode(360, (360, 640), (360, 640), (384, 768), (0, 0)),
}


def mode_for(height: int) -> Mode:
    if type(height) is not int or height not in MODES:
        raise ValueError("Original-size NR input height must be 720, 540, 480 or 360")
    return MODES[height]


class FrameGeometry:
    """GPU resampling and edge context around an original-size NR model."""

    def __init__(self, height: int, *, source=(540, 960), device="xpu"):
        from residual_scale_v1 import _axis, filter_table

        self.mode = mode_for(height)
        self.source = source
        self._axis_kernel = _axis
        self.tables = {}
        ah, aw = self.mode.active
        for kind, axis, source, destination in (
            ("lanczos2", "y", source[0], ah), ("lanczos2", "x", source[1], aw),
            ("area", "y", source[0], ah), ("area", "x", source[1], aw),
            ("catmull", "x", aw, source[1]), ("catmull", "y", ah, source[0]),
        ):
            if source != destination:
                indices, weights = filter_table(source, destination, kind)
                self.tables[(kind, axis)] = (torch.from_numpy(indices).to(device),
                                             torch.from_numpy(weights).to(device))
        top, left = self.mode.inset
        mh, mw = self.mode.model
        self.rows = torch.arange(mh, device=device).sub(top).clamp(0, ah - 1)
        self.columns = torch.arange(mw, device=device).sub(left).clamp(0, aw - 1)

    def _axis(self, tensor, kind, axis, *, dtype=torch.float32, original=None):
        import triton

        if (kind, axis) not in self.tables:
            if original is not None:
                if tensor.shape != original.shape or tensor.device != original.device:
                    raise ValueError("Residual and source geometry must match for direct composition")
                return (tensor.float() + original.float()).clamp(0, 1).to(dtype)
            return tensor
        indices, weights = self.tables[(kind, axis)]
        h, w, channels = tensor.shape
        out_h, out_w = (indices.shape[0], w) if axis == "y" else (h, indices.shape[0])
        out = torch.empty((out_h, out_w, channels), dtype=dtype, device=tensor.device)
        self._axis_kernel[(triton.cdiv(out.numel(), 256),)](
            tensor, indices, weights, tensor if original is None else original,
            out, h, w, out_h, out_w, channels, axis == "y", indices.shape[1],
            original is not None, 256, enable_fp_fusion=False)
        return out

    def prepare(self, color, motion):
        if (tuple(color.shape) != (*self.source, 3) or
                tuple(motion.shape) != (*self.source, 2)
                or color.device != motion.device or color.device.type != "xpu"):
            raise ValueError("Expected matching source-resolution XPU RGB and HWC2 pixel motion")
        ah, aw = self.mode.active
        if self.source == (ah, aw):
            active_color, active_motion = color, motion
        else:
            active_color = self._axis(self._axis(color, "lanczos2", "y"),
                                      "lanczos2", "x")
            active_motion = self._axis(self._axis(motion, "area", "y"), "area", "x")
            active_motion = active_motion * torch.tensor(
                [aw / self.source[1], ah / self.source[0]],
                dtype=torch.float32, device=color.device)
        top, left = self.mode.inset
        canvas, flow = active_color, active_motion
        if top:
            canvas, flow = canvas.index_select(0, self.rows), flow.index_select(0, self.rows)
        if left:
            canvas, flow = canvas.index_select(1, self.columns), flow.index_select(1, self.columns)
        canvas, flow = canvas.float(), flow.float()
        if top or left:
            valid_rows = ((torch.arange(self.mode.model[0], device=color.device) >= top) &
                          (torch.arange(self.mode.model[0], device=color.device) < top + ah))
            valid_columns = ((torch.arange(self.mode.model[1], device=color.device) >= left) &
                             (torch.arange(self.mode.model[1], device=color.device) < left + aw))
            flow = flow * (valid_rows[:, None] & valid_columns[None, :])[..., None]
        return canvas.contiguous(), flow.contiguous()

    def composite(self, original, canvas, result):
        if result.shape != (*self.mode.model, 3) or canvas.shape != (*self.mode.model, 3):
            raise ValueError("Original-size NR result geometry mismatch")
        ah, aw = self.mode.active
        top, left = self.mode.inset
        processed = result[top:top + ah, left:left + aw]
        if self.source == (ah, aw):
            return processed.float().clamp(0, 1)
        source = canvas[top:top + ah, left:left + aw]
        residual = (processed.float() - source.float()).half().contiguous()
        horizontal = self._axis(residual, "catmull", "x", dtype=torch.float16)
        return self._axis(horizontal, "catmull", "y", original=original)


class FullsizeGameModes:
    """One active original-size fast model; mode changes use prepared disk cache."""

    def __init__(self, exact_root, profile_path, profile_sha256, *, controlled=False,
                 graph_capture_policy="all", combo_modes=(),
                 c128_pairwise_720=False, c128_dual_qkv_720=False,
                 c64_attention_project_720=False,
                 c128_attention_project_720=False,
                 c512_qkv_library_720=False,
                 c512_encoder_window_blocks_720=(),
                 native_k8_720=False,
                 c32_window_chain_720=False,
                 post_rgb_tail_720=False,
                 decoder_gather_unround_720=False,
                 c32_hidden_native_720=False,
                 c512_probability_unround_720=False,
                 decoder_merge_unround_720=False,
                 decoder_merge_native_fma_720=False,
                 history_compact_720=False,
                 post_native_fma_720=None,
                 vit_head_720=False,
                 numeric_cleanup_720=None, native_rtz_receipt=None):
        numeric_options = None
        if numeric_cleanup_720 is not None:
            from numeric_cleanup_options_720_v1 import NumericCleanupOptions
            numeric_options = NumericCleanupOptions.parse(numeric_cleanup_720)
        numeric_active = numeric_options is not None and numeric_options.active
        if native_rtz_receipt is not None:
            if not numeric_active or numeric_options.post_store != "native_rtz":
                raise ValueError("A native RTZ receipt requires the explicit native_rtz numerical arm")
            from os import fspath
            native_rtz_receipt = fspath(native_rtz_receipt)
            if not isinstance(native_rtz_receipt, str) or not native_rtz_receipt:
                raise TypeError("native_rtz_receipt must be a nonempty path string")
        if graph_capture_policy not in ("all", "initial-360"):
            raise ValueError("Unknown graph capture policy")
        selected_combo = frozenset(combo_modes)
        if not selected_combo.issubset({360, 480, 540, 720}) or (selected_combo and not controlled):
            raise ValueError("Structure combo is only supported by controlled fast modes")
        if ((c128_pairwise_720 or c128_dual_qkv_720 or c64_attention_project_720 or
             c128_attention_project_720) and
                (not controlled or 720 not in selected_combo)):
            raise ValueError("720p structure extensions require the controlled 720p combo")
        if c128_attention_project_720 and not c64_attention_project_720:
            raise ValueError("C128 attention/project fusion requires the C64 delegate")
        encoder_window_blocks = tuple(c512_encoder_window_blocks_720)
        if (len(set(encoder_window_blocks)) != len(encoder_window_blocks) or
                any(type(index) is not int or index not in range(8)
                    for index in encoder_window_blocks)):
            raise ValueError("Expected unique C512 encoder block indices 0..7")
        if (c512_qkv_library_720 or encoder_window_blocks or native_k8_720 or
                c32_window_chain_720 or post_rgb_tail_720 or
                decoder_gather_unround_720 or c32_hidden_native_720 or
                c512_probability_unround_720 or decoder_merge_unround_720 or
                decoder_merge_native_fma_720 or history_compact_720 or
                post_native_fma_720 or vit_head_720) and (not controlled or 720 not in selected_combo):
            raise ValueError("Experimental candidates require the controlled 720p combo")
        if c32_hidden_native_720 and (not c512_qkv_library_720 or not native_k8_720 or
                                      c32_window_chain_720):
            raise ValueError("Native C32 hidden requires the C512+K8 baseline without window-layout experiments")
        if c512_probability_unround_720 and (not c512_qkv_library_720 or not native_k8_720 or
                                            encoder_window_blocks):
            raise ValueError("C512 probability candidate requires C512+K8 without encoder-window experiments")
        if (decoder_merge_unround_720 or decoder_merge_native_fma_720) and (
                not c512_qkv_library_720 or not native_k8_720):
            raise ValueError("Decoder merge candidate requires the C512+K8 baseline")
        if post_native_fma_720 not in (None, "fp16_fma", "fp32_fma"):
            raise ValueError("Unknown post native FMA mode")
        if numeric_active and (not controlled or 720 not in selected_combo or
                               not c512_qkv_library_720 or not native_k8_720):
            raise ValueError("Numerical cleanup requires the controlled 720p C512+K8 baseline")
        if numeric_active and (c32_window_chain_720 or post_rgb_tail_720 or
                               history_compact_720 or vit_head_720 or encoder_window_blocks):
            raise ValueError("Numerical cleanup cannot mix with dormant layout experiments")
        self.exact_root = exact_root
        self.profile_path = profile_path
        self.profile_sha256 = profile_sha256
        self.controlled = controlled
        self.graph_capture_policy = graph_capture_policy
        self.combo_modes = selected_combo
        self.c128_pairwise_720 = bool(c128_pairwise_720)
        self.c128_dual_qkv_720 = bool(c128_dual_qkv_720)
        self.c64_attention_project_720 = bool(c64_attention_project_720)
        self.c128_attention_project_720 = bool(c128_attention_project_720)
        self.c512_qkv_library_720 = bool(c512_qkv_library_720)
        self.c512_encoder_window_blocks_720 = encoder_window_blocks
        self.native_k8_720 = bool(native_k8_720)
        self.c32_window_chain_720 = bool(c32_window_chain_720)
        self.post_rgb_tail_720 = bool(post_rgb_tail_720)
        self.decoder_gather_unround_720 = bool(decoder_gather_unround_720)
        self.c32_hidden_native_720 = bool(c32_hidden_native_720)
        self.c512_probability_unround_720 = bool(c512_probability_unround_720)
        self.decoder_merge_unround_720 = bool(decoder_merge_unround_720)
        self.decoder_merge_native_fma_720 = bool(decoder_merge_native_fma_720)
        self.history_compact_720 = bool(history_compact_720)
        self.post_native_fma_720 = post_native_fma_720
        self.vit_head_720 = bool(vit_head_720)
        self.numeric_cleanup_720 = numeric_options
        self.native_rtz_receipt = native_rtz_receipt
        self.numeric_cleanup_calls = None
        self._numeric_cleanup_scope = None
        self._c128_scope = None
        self._c512_library_scope = None
        self.c512_library_calls = None
        self._c512_encoder_scope = None
        self.c512_encoder_calls = None
        self._native_k8_scope = None
        self.native_k8_calls = None
        self._c32_window_scope = None
        self.c32_window_calls = None
        self._c32_hidden_scope = None
        self.c32_hidden_calls = None
        self._c512_probability_scope = None
        self.c512_probability_calls = None
        self._decoder_merge_scope = None
        self.decoder_merge_calls = None
        self._post_rgb_tail_scope = None
        self.post_rgb_tail_calls = None
        self._vit_scope = None
        self.vit_head_calls = None
        self.history_compact_calls = None
        self.c128_pairwise_calls = None
        self.c128_dual_calls = None
        self.c64_attention_project_calls = None
        self.c128_attention_project_calls = None
        self._combo_scope = None
        self._post_scope = None
        self.post_calls = None
        self.combo_calls = None
        self.last_combo_capture_gate = None
        self.last_combo_frame_route = None
        self._ever_selected = False
        self._allow_capture_for_session = False
        self._session_frames = 0
        self.last_graph_used = False
        self.height = None
        self.source = None
        self.variant = None
        self.geometry = None
        self.session = None

    def _close_combo(self):
        numerical_scope = self._numeric_cleanup_scope
        self._numeric_cleanup_scope = None
        self.numeric_cleanup_calls = None
        try:
            if numerical_scope is not None:
                numerical_scope.__exit__(None, None, None)
        finally:
            self._close_combo_core()

    def _close_combo_core(self):
        decoder_merge_scope = self._decoder_merge_scope
        c512_probability_scope = self._c512_probability_scope
        c32_hidden_scope = self._c32_hidden_scope
        c32_window_scope = self._c32_window_scope
        post_rgb_tail_scope = self._post_rgb_tail_scope
        encoder_scope = self._c512_encoder_scope
        vit_scope = self._vit_scope
        c512_library_scope = self._c512_library_scope
        native_k8_scope = self._native_k8_scope
        c128_scope = self._c128_scope
        post_scope = self._post_scope
        scope = self._combo_scope
        self._c128_scope = None
        self._c512_library_scope = None
        self.c512_library_calls = None
        self._c512_encoder_scope = None
        self.c512_encoder_calls = None
        self._native_k8_scope = None
        self.native_k8_calls = None
        self._c32_window_scope = None
        self.c32_window_calls = None
        self._c32_hidden_scope = None
        self.c32_hidden_calls = None
        self._c512_probability_scope = None
        self.c512_probability_calls = None
        self._decoder_merge_scope = None
        self.decoder_merge_calls = None
        self._post_rgb_tail_scope = None
        self.post_rgb_tail_calls = None
        self._vit_scope = None
        self.vit_head_calls = None
        self.history_compact_calls = None
        self.c128_pairwise_calls = None
        self.c128_dual_calls = None
        self.c64_attention_project_calls = None
        self.c128_attention_project_calls = None
        self._post_scope = None
        self.post_calls = None
        self._combo_scope = None
        self.combo_calls = None
        self.last_combo_capture_gate = None
        self.last_combo_frame_route = None
        try:
            try:
                try:
                    try:
                        if decoder_merge_scope is not None:
                            decoder_merge_scope.__exit__(None, None, None)
                    finally:
                        if c512_probability_scope is not None:
                            c512_probability_scope.__exit__(None, None, None)
                finally:
                    if c32_hidden_scope is not None:
                        c32_hidden_scope.__exit__(None, None, None)
            finally:
                if encoder_scope is not None:
                    encoder_scope.__exit__(None, None, None)
        finally:
            self._close_combo_remaining(
                vit_scope, c512_library_scope, native_k8_scope,
                c32_window_scope, c128_scope, post_rgb_tail_scope, post_scope, scope)

    @staticmethod
    def _close_combo_remaining(vit_scope, c512_library_scope, native_k8_scope,
                               c32_window_scope, c128_scope, post_rgb_tail_scope,
                               post_scope, scope):
        try:
            if vit_scope is not None:
                vit_scope.__exit__(None, None, None)
        finally:
            try:
                if c512_library_scope is not None:
                    c512_library_scope.__exit__(None, None, None)
            finally:
                try:
                    if native_k8_scope is not None:
                        native_k8_scope.__exit__(None, None, None)
                finally:
                    try:
                        if c32_window_scope is not None:
                            c32_window_scope.__exit__(None, None, None)
                    finally:
                        try:
                            if c128_scope is not None:
                                c128_scope.close()
                        finally:
                            try:
                                if post_rgb_tail_scope is not None:
                                    post_rgb_tail_scope.__exit__(None, None, None)
                            finally:
                                try:
                                    if post_scope is not None:
                                        post_scope.__exit__(None, None, None)
                                finally:
                                    if scope is not None:
                                        scope.__exit__(None, None, None)

    @staticmethod
    def _variant_scope(variant):
        if variant is None:
            return nullcontext()
        from nr_backend.unround_policy import selected_game_variant
        return selected_game_variant(variant)

    @staticmethod
    @contextmanager
    def _decoder_scope(variant, height, unround_720=False):
        if variant is None:
            yield
            return
        old = os.environ.get("NR_FAST_DECODER_UNROUND")
        os.environ["NR_FAST_DECODER_UNROUND"] = (
            "1" if variant == "unrounded" and
            (height in (480, 540) or (height == 720 and unround_720)) else "0")
        try:
            yield
        finally:
            if old is None:
                os.environ.pop("NR_FAST_DECODER_UNROUND", None)
            else:
                os.environ["NR_FAST_DECODER_UNROUND"] = old

    def select(self, height: int, source=(540, 960), *, variant=None):
        with self._variant_scope(variant):
            self._select(height, source, variant=variant)

    def _select(self, height, source, *, variant):
        mode = mode_for(height)
        if variant == "unrounded" and height in (480, 540) and height not in self.combo_modes:
            raise RuntimeError("Unrounded 480/540p requires the installed structure combo")
        if self.height == height and self.source == source and self.variant == variant:
            return
        if self.session is not None:
            old_session = self.session
            try:
                self._close_combo()
            finally:
                try:
                    old_session.close()
                finally:
                    self.session = None
                    self.geometry = None
                    self.height = None
                    self.source = None
                    self.variant = None
            if self.controlled:
                # A closed graph/provider can keep cyclic Python references
                # alive until the next GC. Release them before allocating the
                # next geometry; the shared style LUT remains live.
                gc.collect()
                torch.xpu.empty_cache()
        if str(GEOMETRY_MODULE) not in sys.path:
            sys.path.insert(0, str(GEOMETRY_MODULE))
        from re4_session_v1 import open_session

        with self._decoder_scope(variant, height, self.decoder_gather_unround_720):
            if self.controlled:
                import nr256_selected_stack_v1 as selected_stack
                from nr_game_controlled_model import GameLiveControlledNR
                original_factory = selected_stack.MotionNR
                selected_stack.MotionNR = GameLiveControlledNR
                try:
                    self.session = open_session(self.exact_root, self.profile_path,
                                                self.profile_sha256, geometry=mode.model,
                                                padding=mode.internal)
                finally:
                    selected_stack.MotionNR = original_factory
            else:
                self.session = open_session(self.exact_root, self.profile_path,
                                            self.profile_sha256, geometry=mode.model,
                                            padding=mode.internal)
        try:
            self.geometry = FrameGeometry(height, source=source)
            if height in self.combo_modes:
                from three_structure_combo_v1 import installed
                scope = installed(self.session._stack, height=height,
                                  output_size=mode.model, session=self.session)
                calls = scope.__enter__()
                self._combo_scope = scope
                self.combo_calls = calls
                if height == 720 or (variant == "unrounded" and height in (480, 540)):
                    from post_entry_mlp_fusion_v1 import preflight, padded
                    from post_attention_k8_combined_v1 import installed as post_installed
                    self.session._ready()
                    model = self.session._stack.model
                    ih, iw = mode.internal
                    features = torch.empty((ih // 2, iw // 2, 32), device="xpu", dtype=torch.float16)
                    skip = torch.empty((ih, iw, 32), device="xpu", dtype=torch.float16)
                    if height == 720 and variant == "unrounded" and self.post_native_fma_720:
                        from post_entry_native_fma_720_v1 import (
                            preflight as native_fma_preflight,
                            padded as native_fma_padded)
                        native_fma_preflight(model.post, features, skip,
                                             mode=self.post_native_fma_720, bm=32)
                        fma_mode = self.post_native_fma_720
                        entry = lambda module, f, s: module.body.mlp.forward_unquantized(
                            native_fma_padded(module, f, s, mode=fma_mode, bm=32))
                    else:
                        preflight(model.post, features, skip, stage="padded", lut_guard=None, bm=32)
                        entry = lambda module, f, s: module.body.mlp.forward_unquantized(
                            padded(module, f, s, bm=32))
                    post_scope = post_installed(
                        model.post, output_size=mode.model,
                        post_calls=self.combo_calls["post"],
                        post_attention_calls=self.combo_calls["post_attention"],
                        entry_fn=entry)
                    self.post_calls = post_scope.__enter__()
                    self._post_scope = post_scope
                    if height == 720 and variant == "unrounded" and self.post_rgb_tail_720:
                        from post_rgb_history_tail_720_v1 import installed as post_rgb_installed
                        tail_scope = post_rgb_installed(self.session)
                        self.post_rgb_tail_calls = tail_scope.__enter__()
                        self._post_rgb_tail_scope = tail_scope
                if (height == 720 and variant == "unrounded" and
                        (self.c128_pairwise_720 or self.c128_dual_qkv_720 or
                         self.c64_attention_project_720 or
                         self.c128_attention_project_720)):
                    self.session._ready()
                    stack = self.session._stack
                    scopes = ExitStack()
                    try:
                        if self.c128_pairwise_720:
                            from branched_mlp_pairwise_family_720_v1 import installed as pairwise_installed
                            self.c128_pairwise_calls = scopes.enter_context(
                                pairwise_installed(stack, families=(128,)))
                            self.c128_pairwise_calls.preflight()
                        if self.c128_dual_qkv_720:
                            from qkv_dual_segment_c128_family_720_v1 import (
                                compile_preflight, installed as dual_installed)
                            compile_preflight(stack, variant=variant)
                            self.c128_dual_calls = scopes.enter_context(
                                dual_installed(stack, combo_calls=self.combo_calls,
                                               variant=variant))
                        if self.c64_attention_project_720:
                            from c64_attention_project_family_720_v1 import (
                                compile_preflight as c64_preflight,
                                installed as c64_installed)
                            c64_preflight(stack, variant=variant)
                            self.c64_attention_project_calls = scopes.enter_context(
                                c64_installed(stack, combo_calls=self.combo_calls,
                                              variant=variant))
                        if self.c128_attention_project_720:
                            from c128_attention_project_family_720_v1 import (
                                compile_preflight as c128_attention_preflight,
                                installed as c128_attention_installed)
                            c128_attention_preflight(stack, variant=variant)
                            self.c128_attention_project_calls = scopes.enter_context(
                                c128_attention_installed(stack, combo_calls=self.combo_calls,
                                                         variant=variant))
                    except BaseException:
                        scopes.close()
                        self.c128_pairwise_calls = None
                        self.c128_dual_calls = None
                        self.c64_attention_project_calls = None
                        self.c128_attention_project_calls = None
                        raise
                    self._c128_scope = scopes
                if height == 720 and variant == "unrounded":
                    if self.c32_window_chain_720:
                        from c32_window_mlp_family_720_v1 import installed as c32_window_installed
                        c32_window_scope = c32_window_installed(
                            self.session, sites=("encoder32_1", "encoder32_2",
                                                 "decoder32_2", "decoder32_3"))
                        self.c32_window_calls = c32_window_scope.__enter__()
                        self._c32_window_scope = c32_window_scope
                    if self.c512_qkv_library_720 and self.native_k8_720:
                        from c512_k8_joint_scope_720_v1 import installed as joint_installed
                        library_scope = joint_installed(self.session)
                        (self.c512_library_calls,
                         self.native_k8_calls) = library_scope.__enter__()
                        self._c512_library_scope = library_scope
                    elif self.c512_qkv_library_720:
                        from c512_qkv_library_16_v1 import installed as c512_library_installed
                        library_scope = c512_library_installed(self.session)
                        self.c512_library_calls = library_scope.__enter__()
                        self._c512_library_scope = library_scope
                    elif self.native_k8_720:
                        from native_k8_active_720_v1 import installed as k8_installed
                        k8_scope = k8_installed(self.session, "both")
                        self.native_k8_calls = k8_scope.__enter__()
                        self._native_k8_scope = k8_scope
                    if self.c32_hidden_native_720:
                        from c32_hidden_native_720_v1 import installed as c32_hidden_installed
                        hidden_scope = c32_hidden_installed(self.session)
                        self.c32_hidden_calls = hidden_scope.__enter__()
                        self._c32_hidden_scope = hidden_scope
                        self.c32_hidden_calls.preflight()
                    if self.c512_probability_unround_720:
                        from c512_probability_unround_scope_v1 import installed_720 as probability_installed
                        probability_scope = probability_installed(self.session)
                        self.c512_probability_calls = probability_scope.__enter__()
                        self._c512_probability_scope = probability_scope
                        self.c512_probability_calls.preflight()
                    if self.decoder_merge_unround_720 or self.decoder_merge_native_fma_720:
                        if self.decoder_merge_native_fma_720:
                            from decoder_merge_native_fma_720_v1 import installed as merge_installed
                            merge_scope = merge_installed(
                                self.session, unround_output=self.decoder_merge_unround_720)
                        else:
                            from decoder_merge_unround_720_v1 import installed as merge_installed
                            merge_scope = merge_installed(self.session)
                        self.decoder_merge_calls = merge_scope.__enter__()
                        self._decoder_merge_scope = merge_scope
                        self.decoder_merge_calls.preflight()
                    if self.vit_head_720:
                        from vit_head_dataflow_720_v1 import installed as vit_installed
                        vit_scope = vit_installed(self.session)
                        self.vit_head_calls = vit_scope.__enter__()
                        self._vit_scope = vit_scope
                    if self.c512_encoder_window_blocks_720:
                        from c512_encoder_window_chain_720_v1 import installed as encoder_installed
                        encoder_scope = encoder_installed(
                            self.session,
                            block_indices=self.c512_encoder_window_blocks_720)
                        self.c512_encoder_calls = encoder_scope.__enter__()
                        self._c512_encoder_scope = encoder_scope
        except BaseException:
            try:
                self._close_combo()
            finally:
                self.session.close()
                self.session = None
                self.geometry = None
            raise
        self.height = height
        self.source = source
        self.variant = variant
        self._allow_capture_for_session = (self.graph_capture_policy == "all" or
                                            not self._ever_selected and height == 360)
        self._ever_selected = True
        self._session_frames = 0
        if (height == 720 and variant == "unrounded" and
                self.numeric_cleanup_720 is not None and self.numeric_cleanup_720.active):
            try:
                from numeric_cleanup_suite_720_v1 import installed as numerical_installed
                numerical_scope = numerical_installed(
                    self, self.numeric_cleanup_720, native_rtz_receipt=self.native_rtz_receipt)
                self.numeric_cleanup_calls = numerical_scope.__enter__()
                self._numeric_cleanup_scope = numerical_scope
            except BaseException:
                failed_session = self.session
                try:
                    self._close_combo()
                finally:
                    try:
                        if failed_session is not None:
                            failed_session.close()
                    finally:
                        self.session = None
                        self.geometry = None
                        self.height = None
                        self.source = None
                        self.variant = None
                raise

    def process(self, color, motion, *, height: int, reset: bool,
                history_warp: str = "reference", graph_replay: bool = False,
                controls=None, variant=None):
        with self._variant_scope(variant):
            try:
                return self._process(color, motion, height=height, reset=reset,
                                     history_warp=history_warp, graph_replay=graph_replay,
                                     controls=controls, variant=variant)
            except BaseException:
                if self.session is not None and (self.c32_hidden_native_720 or
                                                  self.c512_probability_unround_720 or
                                                  self.decoder_merge_unround_720 or
                                                  self.decoder_merge_native_fma_720 or
                                                  self.numeric_cleanup_calls is not None):
                    self.session._failed = True
                raise

    def _process(self, color, motion, *, height, reset, history_warp,
                 graph_replay, controls, variant):
        source = tuple(color.shape[:2])
        changed = self.height != height or self.source != source or self.variant != variant
        self._select(height, source, variant=variant)
        numerical_owner = self.numeric_cleanup_calls or getattr(self, "_numeric_cleanup_owner", None)
        if numerical_owner is not None:
            if history_warp != "fused" or graph_replay is not True:
                self.session._failed = True
                raise RuntimeError("Active numerical candidates require fused history and graph replay")
            numerical_owner.validate_frame_context()
        canvas, flow = self.geometry.prepare(color, motion)
        def execute():
            if self.controlled:
                self.session._ready()
                model = self.session._stack.model
                graph = self.session._stack.graph
                before_replays = graph.replays
                before_entries = len(graph.entries)
                if self.combo_calls is not None and graph_replay:
                    from three_structure_combo_v1 import snapshot_calls
                    before_combo = snapshot_calls(self.combo_calls)
                else:
                    before_combo = None
                before_pairwise = (dict(self.c128_pairwise_calls.capture_calls)
                                   if self.c128_pairwise_calls is not None and graph_replay else None)
                before_c64_attention = (
                    dict(self.c64_attention_project_calls.capture_calls)
                    if self.c64_attention_project_calls is not None and graph_replay else None)
                before_c128_attention = (
                    dict(self.c128_attention_project_calls.capture_calls)
                    if self.c128_attention_project_calls is not None and graph_replay else None)
                if self.c128_dual_calls is not None and graph_replay:
                    from qkv_dual_segment_c128_family_720_v1 import snapshot_calls as snapshot_c128_dual
                    before_dual = snapshot_c128_dual(self.c128_dual_calls)
                else:
                    before_dual = None
                before_post_entry = (self.post_calls["entry"]
                                     if self.post_calls is not None and graph_replay else None)
                before_post_rgb_tail = (self.post_rgb_tail_calls["tail"]
                                        if self.post_rgb_tail_calls is not None and graph_replay else None)
                before_c512_library = (dict(self.c512_library_calls)
                                       if self.c512_library_calls is not None and graph_replay else None)
                before_c512_encoder = (
                    {index: self.c512_encoder_calls[f"encoder512_{index}"]
                     for index in self.c512_encoder_window_blocks_720}
                    if self.c512_encoder_calls is not None and graph_replay else None)
                before_native_k8 = (dict(self.native_k8_calls)
                                    if self.native_k8_calls is not None and graph_replay else None)
                before_c32_window = (
                    {name: self.c32_window_calls[name]
                     for name in ("encoder32_1", "encoder32_2", "decoder32_2", "decoder32_3")}
                    if self.c32_window_calls is not None and graph_replay else None)
                before_c32_hidden = (
                    dict(self.c32_hidden_calls.capture_calls)
                    if self.c32_hidden_calls is not None and graph_replay else None)
                before_c512_probability = (
                    dict(self.c512_probability_calls.capture_calls)
                    if self.c512_probability_calls is not None and graph_replay else None)
                before_decoder_merge = (
                    dict(self.decoder_merge_calls.capture_calls)
                    if self.decoder_merge_calls is not None and graph_replay else None)
                before_vit_head = (dict(self.vit_head_calls)
                                   if self.vit_head_calls is not None and graph_replay else None)
                with self.session._installed(), torch.inference_mode(), \
                        self.session._use_arithmetic_backend("triton"), \
                        (graph.installed() if graph_replay else nullcontext()), \
                        (model.graph_controls(
                            graph, allow_capture=self._allow_capture_for_session and
                            (self.graph_capture_policy == "all" or
                             self._session_frames < 16)) if graph_replay else nullcontext()):
                    result = model(canvas, flow, controls=controls,
                                   reset=bool(reset or changed)).float()
                    torch.xpu.synchronize()
                self.last_graph_used = graph.replays > before_replays
                captured = len(graph.entries) > before_entries
                self.last_combo_frame_route = (
                    "capture" if captured else
                    "replay" if self.last_graph_used else
                    "eager" if graph_replay else "eager-requested")
                self.last_combo_capture_gate = None
                if before_combo is not None and captured:
                    from three_structure_combo_v1 import capture_delta_gate
                    self.last_combo_capture_gate = capture_delta_gate(
                        before_combo, self.combo_calls)
                    if before_post_entry is not None:
                        self.last_combo_capture_gate["post_entry"] = (
                            self.post_calls["entry"] > before_post_entry)
                    if before_post_rgb_tail is not None:
                        self.last_combo_capture_gate["post_rgb_history_tail"] = (
                            self.post_rgb_tail_calls["tail"] > before_post_rgb_tail)
                    if before_c512_library is not None:
                        self.last_combo_capture_gate["c512_library_all_sixteen"] = (
                            len(before_c512_library) == 16 and
                            all(self.c512_library_calls[name] > value
                                for name, value in before_c512_library.items()))
                    if before_c512_encoder is not None:
                        self.last_combo_capture_gate["c512_encoder_windows"] = (
                            all(self.c512_encoder_calls[f"encoder512_{index}"] > value
                                for index, value in before_c512_encoder.items()))
                    if before_native_k8 is not None:
                        self.last_combo_capture_gate["native_k8_pre_post"] = (
                            all(self.native_k8_calls[name] > before_native_k8[name]
                                for name in ("pre", "post")))
                    if before_c32_window is not None:
                        self.last_combo_capture_gate["c32_window_four_sites"] = (
                            all(self.c32_window_calls[name] > value
                                for name, value in before_c32_window.items()))
                    if before_c32_hidden is not None:
                        self.last_combo_capture_gate["c32_hidden_native_ten"] = (
                            len(before_c32_hidden) == 10 and
                            all(self.c32_hidden_calls.capture_calls[name] == value + 1
                                for name, value in before_c32_hidden.items()))
                    if before_c512_probability is not None:
                        self.last_combo_capture_gate["c512_probability_all_sixteen"] = (
                            len(before_c512_probability) == 16 and
                            all(self.c512_probability_calls.capture_calls[name] == value + 1
                                for name, value in before_c512_probability.items()))
                    if before_decoder_merge is not None:
                        expected_merges = 5 if self.decoder_merge_native_fma_720 else 4
                        gate_name = ("decoder_merge_native_fma_five" if self.decoder_merge_native_fma_720
                                     else "decoder_merge_unround_four")
                        self.last_combo_capture_gate[gate_name] = (
                            len(before_decoder_merge) == expected_merges and
                            all(self.decoder_merge_calls.capture_calls[name] == value + 1
                                for name, value in before_decoder_merge.items()))
                    if before_vit_head is not None:
                        self.last_combo_capture_gate["vit_head_all_eight"] = (
                            len(before_vit_head) == 8 and
                            all(self.vit_head_calls[index] > value
                                for index, value in before_vit_head.items()))
                    if before_pairwise is not None:
                        after_pairwise = self.c128_pairwise_calls.capture_calls
                        self.last_combo_capture_gate["c128_pairwise_twelve"] = (
                            len(before_pairwise) == 12 and
                            all(after_pairwise[name] == value + 1
                                for name, value in before_pairwise.items()))
                    if before_c64_attention is not None:
                        after_c64_attention = self.c64_attention_project_calls.capture_calls
                        self.last_combo_capture_gate["c64_attention_project_eight"] = (
                            len(before_c64_attention) == 8 and
                            all(after_c64_attention[name] == value + 1
                                for name, value in before_c64_attention.items()))
                    if before_c128_attention is not None:
                        after_c128_attention = self.c128_attention_project_calls.capture_calls
                        self.last_combo_capture_gate["c128_attention_project_twelve"] = (
                            len(before_c128_attention) == 12 and
                            all(after_c128_attention[name] == value + 1
                                for name, value in before_c128_attention.items()))
                    if before_dual is not None:
                        after_dual = self.c128_dual_calls["capture_by_block"]
                        self.last_combo_capture_gate["c128_dual_twelve"] = (
                            len(before_dual["by_block"]) == 12 and
                            all(after_dual[name] == value + 1
                                for name, value in before_dual["capture_by_block"].items()))
                    if not all(self.last_combo_capture_gate.values()):
                        raise RuntimeError("Structure combo missed a captured NR path")
                return result
            if not graph_replay:
                return self.session.process(canvas, flow, reset=reset or changed).color
            # A replay intentionally skips per-operator Python callbacks, so
            # FullsizeSession.process's eager FFN call counter is inapplicable.
            # The graph frontend captures one model shape per selected session.
            # Each game geometry must pass its own eager/replay output gate.
            self.session._ready()
            with self.session._installed(), torch.inference_mode(), \
                    self.session._use_arithmetic_backend("triton"), \
                    self.session._stack.graph.installed():
                result = self.session._stack.model(
                    canvas, flow, reset=bool(reset or changed)).float()
                torch.xpu.synchronize()
            return result

        if history_warp == "reference":
            processed = execute()
        elif history_warp == "fused":
            # MotionNR resolves this imported function at call time. Restrict
            # the fast-only override to one synchronous game frame and restore
            # it even when the frame fails; shared exact source stays untouched.
            import nr_backend.temporal as temporal
            from nr_game_history_fused import warp_history_fused

            original = temporal.warp_history_normalized
            if self.history_compact_720 and height == 720 and variant == "unrounded":
                from history_front_compact_720_v1 import installed as compact_installed
                with compact_installed() as compact:
                    from nr_game_history_fused import warp_history_fused
                    temporal.warp_history_normalized = warp_history_fused
                    try:
                        processed = execute()
                    finally:
                        temporal.warp_history_normalized = original
                    # Captured graphs bypass Python on replay. Keep the
                    # capture-time route evidence instead of reporting an
                    # inactive candidate on every successful replay frame.
                    earlier = self.history_compact_calls or {}
                    self.history_compact_calls = {
                        name: earlier.get(name, 0) + count
                        for name, count in compact.calls.items()}
            else:
                temporal.warp_history_normalized = warp_history_fused
                try:
                    processed = execute()
                finally:
                    temporal.warp_history_normalized = original
        else:
            raise ValueError("Unknown game history sampler")
        self._session_frames += 1
        return self.geometry.composite(color, canvas, processed)

    def close(self):
        session = self.session
        try:
            self._close_combo()
        finally:
            self.session = None
            try:
                if session is not None:
                    session.close()
            finally:
                self.geometry = None
                self.height = None
                self.source = None
                self.variant = None
