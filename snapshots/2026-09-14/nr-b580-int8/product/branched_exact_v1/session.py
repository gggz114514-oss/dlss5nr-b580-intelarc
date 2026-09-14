"""Independent candidate atop validated all-v1, not reuse_v2."""
from nr_exact_all_candidate_v1 import Session as Previous
from nr_exact_runtime_v1 import _SERIAL
from .scope import Branched


class Session(Previous):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.branched=Branched(self._model)

    def process(self,rgb,motion,*,reset=False):
        with _SERIAL:
            self._ready()
            self.branched.diagnostics=self.diagnostics
            with self.branched.installed():
                result=super().process(rgb,motion,reset=reset)
            if self.diagnostics:
                for resource in self.branched.resources:
                    self.resources[resource['hash']]=resource
                self.branched.resources.clear()
            return result

    def close(self):
        with _SERIAL:
            self.branched.modules.clear()
            super().close()
