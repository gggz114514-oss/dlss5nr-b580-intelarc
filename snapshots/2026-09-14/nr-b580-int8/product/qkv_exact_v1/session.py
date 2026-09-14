from branched_exact_v1.session import Session as Previous
from nr_exact_runtime_v1 import _SERIAL
from .scope import QKV


class Session(Previous):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.qkv=QKV(self._model)

    def process(self,rgb,motion,*,reset=False):
        with _SERIAL:
            self._ready()
            self.qkv.diagnostics=self.diagnostics
            with self.qkv.installed():
                result=super().process(rgb,motion,reset=reset)
            if self.diagnostics:self.resources.update(self.qkv.resources)
            return result

    def close(self):
        with _SERIAL:
            self.qkv.modules.clear()
            super().close()
