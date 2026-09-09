"""Creation-time local tone or structure, at the observed front channels."""
from __future__ import annotations
from .temporal import MotionNR


def _checked_local(tone,structure):
    if any(type(v)not in(int,float)or v not in(0,1,2)for v in(tone,structure)):raise ValueError('Only measured local controls 0, 1 and 2 are supported')
    if tone!=1 and structure!=1:raise ValueError('Combined local tone/structure changes require separate validation')
    return float(tone),float(structure)


class LocalControlMotionNR(MotionNR):
    """256x256 fixed local tone OR structure, default style and intensity.

    Model output remains its private history. Parameters are fixed per session;
    combinations, continuous values and larger dimensions need independent tests.
    """
    PADDED_SIZES={(256,256):(320,320)}

    def __init__(self,*args,local_tone=1,local_structure=1,**kwargs):
        super().__init__(*args,**kwargs);self._local_tone,self._local_structure=_checked_local(local_tone,local_structure)

    @property
    def local_tone(self):return self._local_tone

    @property
    def local_structure(self):return self._local_structure

    @classmethod
    def from_assets(cls,*args,local_tone=1,local_structure=1,**kwargs):
        tone,structure=_checked_local(local_tone,local_structure);model=super().from_assets(*args,**kwargs);model._local_tone,model._local_structure=tone,structure;return model

    def _forward_front(self,rgb,front,**kwargs):
        front[...,11]=self._local_tone;front[...,12]=self._local_structure
        return super()._forward_front(rgb,front,**kwargs)
