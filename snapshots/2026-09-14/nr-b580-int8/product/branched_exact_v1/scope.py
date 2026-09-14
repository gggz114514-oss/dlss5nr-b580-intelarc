"""Owned C128/C256 forward override, restores every method on exit."""
from contextlib import contextmanager
from .kernels import forward


class Branched:
    def __init__(self,model):
        from nr_backend.multihead_block import BranchedMLP
        self.modules=[m for m in model.modules() if isinstance(m,BranchedMLP) and m.channels in (128,256)]
        self.diagnostics=False
        self.resources=[]
        self.calls=0

    @contextmanager
    def installed(self):
        import nr_backend.multihead_block as multi
        restored=[]
        for module in self.modules:
            had='forward_unquantized' in module.__dict__
            old=module.__dict__.get('forward_unquantized')
            def run(features,owner=module):
                result,resources=forward(owner,multi.quantize_fp8(features),diagnostics=self.diagnostics)
                if self.diagnostics:
                    self.resources.extend(resources)
                    self.calls+=1
                return result
            module.forward_unquantized=run
            restored.append((module,had,old))
        try:yield
        finally:
            for module,had,old in reversed(restored):
                if had:module.forward_unquantized=old
                else:del module.forward_unquantized
