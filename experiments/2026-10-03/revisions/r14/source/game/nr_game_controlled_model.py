"""Game-only controlled model with style assets outside the frozen body graph.

The fast stack pins the neural body's registered-buffer inventory. The style
postprocessor runs after that body and is owned by this model, but its large
lookup tables must not enter the body's graph constant list.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch

from nr_backend.controlled_temporal import ControlledMotionNR, NRControls
from nr_backend.live_temporal import LiveControlledMotionNR
from nr_backend.post import store_half_rz
from nr_backend.style import NativeStylePost
from nr_backend.temporal import MotionNR
from graph_front_v1 import descriptor
from audit_history_input_720_v1 import input_summary_frame
from audit_control_graph_720_v1 import cold_raw_capture_allowed, control_lanes


_STYLE_POST = None
_STYLE_DIRECTORY = None


class GameLiveControlledNR(LiveControlledMotionNR):
    @torch.inference_mode()
    def forward(self, rgb, motion, *, controls=None, reset=False, progress=None):
        requested = self._controls if controls is None else controls
        if not isinstance(requested, NRControls):
            raise TypeError("Expected NRControls")
        with input_summary_frame(self, rgb, motion, check_color=requested != NRControls()) as summary:
            if requested != NRControls():
                # The resize filter can overshoot SDR at high-contrast edges.
                # Controlled postprocessing has a strict SDR input contract.
                if summary is None and not bool(torch.isfinite(rgb).all()):
                    raise ValueError("Game NR color must be finite")
                clamped = rgb.clamp(0, 1)
                if summary is not None:
                    summary.clamped_rgb = clamped
                return super().forward(clamped, motion, controls=requested,
                                       reset=reset, progress=progress)

            # The stock fast route historically accepts filter overshoot. Preserve
            # its exact output at default settings, even on such frames.
            previous = self._controls
            changed_front = (previous.style, previous.local_tone, previous.local_structure) != (0, 1.0, 1.0)
            changed_aux = previous.auto_mask or previous.skin_structure is not None
            self._controls = requested
            try:
                return MotionNR.forward(self, rgb, motion,
                                        reset=reset or changed_front or changed_aux,
                                        progress=progress)
            except Exception:
                self._controls = previous
                raise

    @classmethod
    def from_assets(cls, *args, **kwargs):
        global _STYLE_POST, _STYLE_DIRECTORY
        controls = kwargs.get("controls")
        if controls is not None and controls != NRControls():
            raise ValueError("Game model must start with default controls")
        if kwargs.get("enable_ui", False):
            raise ValueError("Game model has no UI input textures")
        sigmoid_directory = args[2] if len(args) > 2 else kwargs["sigmoid_directory"]
        directory = Path(kwargs.get("style_directory") or
                         Path(sigmoid_directory).parent / "style-sm89-v1").resolve()
        if _STYLE_POST is None:
            _STYLE_POST = NativeStylePost.from_directory(directory).to("xpu").eval()
            _STYLE_DIRECTORY = directory
        elif directory != _STYLE_DIRECTORY:
            raise ValueError("Game style asset directory changed within a process")
        model = ControlledMotionNR.from_assets.__func__(cls, *args, **kwargs)
        model.__dict__["style_post"] = _STYLE_POST
        return model

    def _native_controls_present_720(self, front, controls):
        """Consume one real producer receipt on eager and captured routes."""
        lanes = control_lanes(controls)
        published = self.__dict__.pop("_audit_front_lanes_720", None)
        history_proof = (published is not None and published[0] is front
                         and published[1] == self._next_seed
                         and published[2] == lanes and published[3] is self)
        front_proof = False
        if self.__dict__.get("_audit_fdp_front_producer") is not None:
            from audit_front_decoder_post_720_v1 import front_controls_present
            front_proof = front_controls_present(self, front, controls)
        if history_proof and front_proof:
            raise RuntimeError("Two independent producers wrote the same front controls")
        return history_proof or front_proof

    @contextmanager
    def graph_controls(self, graph, *, allow_capture=False):
        """Apply live control lanes before the fast graph consumes the front.

        GraphFront installs an instance-level _forward_front, bypassing the
        ControlledMotionNR override. Keep that override's front semantics and
        raw-private intensity path while leaving the frozen graph body intact.
        """
        original = self.__dict__.get("_forward_front")
        if getattr(original, "__self__", None) is not graph:
            raise RuntimeError("Controlled graph wrapper requires installed graph")

        def controlled_front(rgb, front, **kwargs):
            controls = self._controls
            raw_private = controls.style == 0 and controls.intensity < 1
            key = tuple(descriptor(value) for value in
                        (rgb, front, kwargs.get("previous"),
                         kwargs.get("history_reciprocal"))) + (raw_private,)
            cold_raw = raw_private and cold_raw_capture_allowed(self, graph)
            if (raw_private and key not in graph.entries and not cold_raw
                    and self.__dict__.get('_audit_history_host_modes_720') is not None):
                # Fail before private history/seed commits. Integration must
                # bootstrap reset/history FP32 entries in two real cold frames;
                # never substitute a half-output graph for the low-strength mix.
                raise RuntimeError('Fixed720 low intensity requires cold raw-private graph warmup')
            # The authenticated cold token also permits the extra raw entries
            # after the two default bootstrap frames; steady gameplay cannot
            # acquire this permission from the old allow_capture boolean.
            if key not in graph.entries and not ((allow_capture and not raw_private) or cold_raw):
                # Capturing a new variant during gameplay has caused XPU device
                # loss. The original eager executor supports these controls.
                if not self._native_controls_present_720(front, controls):
                    for lane, value in zip(range(10, 15), control_lanes(controls)):
                        front[..., lane] = value
                if raw_private:
                    kwargs["return_float32"] = True
                    self._raw_private = graph.original(rgb, front, **kwargs)
                    return store_half_rz(self._raw_private)
                return graph.original(rgb, front, **kwargs)
            if not self._native_controls_present_720(front, controls):
                for lane, value in zip(range(10, 15), control_lanes(controls)):
                    front[..., lane] = value
            if raw_private:
                kwargs["return_float32"] = True
                self._raw_private = original(rgb, front, **kwargs)
                return store_half_rz(self._raw_private)
            return original(rgb, front, **kwargs)

        self._forward_front = controlled_front
        try:
            yield
        finally:
            assert self._forward_front is controlled_front
            self._forward_front = original
