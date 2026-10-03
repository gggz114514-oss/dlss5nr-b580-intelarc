"""Native-size Session adapter around the untouched exact backend.

Load in an isolated worker whose nr_backend imports resolve to nr/exact. This
module never installs the fast stack, residual scaling, new quantization, or
shared depth controls. GPU integration acceptance is separate from body parity.
"""
from dataclasses import dataclass
from pathlib import Path
import threading

_SERIAL = threading.RLock()


@dataclass(frozen=True)
class FrameResult:
    color: object
    low_color: object
    sequence: int
    reset_applied: bool


class Session:
    @classmethod
    def create(cls, exact_root):
        import sys
        import torch
        import nr_backend.temporal as temporal
        import nr_backend.execution as execution
        backend = (Path(exact_root).resolve()/'backend').resolve()
        # A fast module already imported under the same package name must not
        # silently turn the exact selector into a different arithmetic route.
        for name,module in tuple(sys.modules.items()):
            if name!='nr_backend' and not name.startswith('nr_backend.'):
                continue
            path = getattr(module,'__file__',None)
            if path is None or not Path(path).resolve().is_relative_to(backend):
                raise RuntimeError('Exact Session requires a separate worker using nr/exact backend')
        with _SERIAL:
            root = Path(exact_root).resolve()/'model-assets'
            model = temporal.MotionNR.from_assets(root/'sf-v2/WEIGHTS_HT.bin',
                root/'noise-sm89-v2', root/'sigmoid-sm89-v1').to('xpu').eval()
            return cls(model, torch, execution.use_arithmetic_backend)

    def __init__(self, model, torch_module, arithmetic_scope):
        self._model = model
        self._torch = torch_module
        self._arithmetic_scope = arithmetic_scope
        self._source_size = None
        self._closed = False
        self._failed = False

    def _ready(self):
        if self._closed:
            raise RuntimeError('Exact NR session is closed')
        if self._failed:
            raise RuntimeError('Exact NR execution failed; stop this isolated worker')

    def process(self, rgb, motion, *, reset=False):
        with _SERIAL:
            self._ready()
            t = self._torch
            if not isinstance(rgb,t.Tensor) or not isinstance(motion,t.Tensor):
                raise TypeError('Expected GPU tensors')
            size = tuple(rgb.shape[:2])
            if rgb.ndim!=3 or rgb.shape[-1]!=3 or size not in self._model.PADDED_SIZES:
                raise ValueError('Unsupported exact-backend source geometry')
            if rgb.device.type!='xpu' or rgb.dtype!=t.float32:
                raise ValueError('Color must be float32 SDR RGB on XPU')
            if motion.device!=rgb.device or motion.dtype!=t.float32 or motion.shape!=(*size,2):
                raise ValueError('Motion must be same-device float32 HWC2 pixel displacement')
            if self._source_size is not None and size!=self._source_size and not reset:
                raise ValueError('Source size change requires reset=True')
            applied_reset = bool(reset or self._source_size is None)
            try:
                with t.inference_mode(), self._arithmetic_scope('triton'):
                    low = self._model(rgb,motion,reset=applied_reset)
                    color = low.float()
                    t.xpu.synchronize()
                self._source_size = size
                return FrameResult(color,low,self._model.next_seed,applied_reset)
            except BaseException:
                self._failed = True
                raise

    def reset(self):
        with _SERIAL:
            self._ready()
            self._model.reset()
            self._source_size = None

    def close(self):
        with _SERIAL:
            if not self._closed:
                self._model = None
                self._closed = True
