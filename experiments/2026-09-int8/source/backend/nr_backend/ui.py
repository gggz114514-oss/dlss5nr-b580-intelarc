"""Optional UI textures for the fixed native peripheral contract."""
from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass(frozen=True)
class NRUIInputs:
    """Current UI resources on the original RGB device.

    Backbuffer is HWC3, sampled as native HALF RGB. Alpha is HW, sampled as
    native FLOAT R. If Alpha is absent, UI must be HWC4; its HALF A is sampled.
    Explicit Alpha takes precedence over UI.A. Without Backbuffer the additive
    base is original HALF RGB. A Backbuffer-only object instead selects that
    texture as primary NR scene input, with no UI peripheral. Supply ui=None
    to model.forward for the ordinary original-scene path.
    Original RGB is supplied independently of backbuffer. Same-resolution SDR
    inputs only; subrects and wider flag/control combinations remain unverified.
    """
    backbuffer: Tensor|None = None
    alpha: Tensor|None = None
    ui: Tensor|None = None

    @property
    def has_alpha_source(self) -> bool:
        return self.alpha is not None or self.ui is not None

    def validate(self,original:Tensor) -> None:
        if self.backbuffer is None and not self.has_alpha_source:
            raise ValueError('Supply at least one UI resource, or pass ui=None to the model')
        for label,value,shape in(('backbuffer',self.backbuffer,original.shape),('alpha',self.alpha,original.shape[:2]),('texture',self.ui,(*original.shape[:2],4))):
            if value is None:continue
            if not isinstance(value,Tensor)or value.shape!=shape or value.device!=original.device or not value.is_floating_point():
                raise ValueError(f'UI {label} must be same-device floating point with shape {tuple(shape)}')
            if not bool((torch.isfinite(value)&(value>=0)&(value<=1)).all()):
                raise ValueError(f'UI {label} must be finite and in [0,1]')
    def prepare(self,original:Tensor) -> tuple[Tensor,Tensor]:
        self.validate(original)
        if not self.has_alpha_source:
            raise ValueError('Backbuffer-only selects the NR scene input; it has no UI peripheral')
        base=original if self.backbuffer is None else self.backbuffer
        alpha=self.alpha.float()if self.alpha is not None else self.ui.half()[...,3].float()
        return base.half().float(),alpha
