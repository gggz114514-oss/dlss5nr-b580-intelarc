"""Per-frame NR controls with transactional private temporal history."""
from __future__ import annotations
from pathlib import Path
import torch
from .controlled_temporal import ControlledMotionNR,NRControls
from .style import NativeStylePost


class LiveControlledMotionNR(ControlledMotionNR):
    """Observed-dimension correctness session with an NRControls value per frame.

    A successful call retains its controls for subsequent calls that omit them.
    Style/tone/structure/auto-mask/skin changes reset history; intensity alone
    preserves it. Skin changes reset even while automatic masking is disabled.
    Assets are loaded before inference, including styles when starting at style0.
    Explicit mask binding changes preserve history. No concurrent calls on a
    session. Explicit masks follow the parent's per-frame contract;
    UI resources are per-frame. Validation coverage varies by dimension and
    control set; wider UI/depth contracts remain unfinished.
    """

    @classmethod
    def from_assets(cls,weights,noise_directory,sigmoid_directory,reciprocal_directory=None,dimension_reciprocal_directory=None,*,controls=None,style_directory=None,enable_ui:bool=False):
        model=super().from_assets(weights,noise_directory,sigmoid_directory,reciprocal_directory,dimension_reciprocal_directory,controls=controls,style_directory=style_directory,enable_ui=enable_ui)
        if model.style_post is None:
            directory=Path(sigmoid_directory).parent/'style-sm89-v1'if style_directory is None else Path(style_directory)
            model.style_post=NativeStylePost.from_directory(directory)
        return model

    @torch.inference_mode()
    def forward(self,rgb,motion,*,controls:NRControls|None=None,control_mask=None,ui=None,reset:bool=False,progress=None):
        requested=self._controls if controls is None else controls
        if not isinstance(requested,NRControls):raise TypeError('Expected NRControls')
        if requested.style and self.style_post is None:raise ValueError('Style changes require loaded native style assets')
        previous_controls=self._controls
        auxiliary_change=(requested.auto_mask,requested.skin_structure)!=(previous_controls.auto_mask,previous_controls.skin_structure)
        changed_front=(requested.style,requested.local_tone,requested.local_structure)!=(previous_controls.style,previous_controls.local_tone,previous_controls.local_structure)
        self._controls=requested
        try:
            return super().forward(rgb,motion,control_mask=control_mask,ui=ui,reset=reset or changed_front or auxiliary_change,progress=progress)
        except Exception:
            self._controls=previous_controls
            raise
