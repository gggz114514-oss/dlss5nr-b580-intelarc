"""Single public, serialized GPU frame API for the consolidated local NR product.

One Session per source stream. Frames and motion stay on XPU, and outputs are
caller-owned. The caller supplies current-to-previous pixel motion, not generated
frames. D3D12 texture import/export is a separate integration boundary.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import sys
import threading
import torch
from nr256_product_stack_v1 import Stack
from residual_scale_v1 import ResidualScale
from face480_residual_scale_v1 import Face480Scale
from nr_backend.execution import use_arithmetic_backend
import compressed_arrays_v1 as arrays

_SERIAL = threading.RLock()
PROFILE_ID = 'nr256-reviewed-v1'
SOURCE_SIZES = ((1080, 1920), (480, 864), (256, 256))

# The host side of a frame is dominated by Triton launch preparation, which is
# memoizable because a frame repeats the same call sites. It is installed here,
# at the public runtime boundary, so every consumer gets it. `NR_LAUNCH_FASTPATH=off`
# turns it off; an unimportable or uninstallable fast path degrades to the
# original path instead of failing the product.
try:
    from launch_fastpath_v1 import install_product_fastpath as _install_launch_fastpath
except ImportError as _fastpath_import_error:  # pragma: no cover - layout guard
    _install_launch_fastpath = None
    print('[launch-fastpath] module not importable (%s); original path in use'
          % _fastpath_import_error, file=sys.stderr, flush=True)

_LAUNCH_FASTPATH = _install_launch_fastpath() if _install_launch_fastpath is not None else None

# Backend sidecars: int8-tree-only replacements for code that both trees share.
# The shared files have to stay byte-identical (rows_paths_v1.duplicate_source_
# agreement) and the exact line's precompile package pins the exact-tree sources
# by sha256, so a change written into the shared file costs the exact line its
# runnable entries. A sidecar keeps the shared file untouched and binds its own
# copy here, at the public runtime boundary. `NR_BACKEND_SIDECARS=off` installs
# nothing; an unimportable or uninstallable sidecar degrades to the original path.
try:
    from sidecar_loader_v1 import install as _install_sidecars
except ImportError as _sidecar_import_error:  # pragma: no cover - layout guard
    _install_sidecars = None
    print('[sidecar] loader not importable (%s); original path in use'
          % _sidecar_import_error, file=sys.stderr, flush=True)

_SIDECARS = _install_sidecars() if _install_sidecars is not None else None


def sidecar_report():
    """Installed backend sidecars and their counters for the frame report."""
    import sidecar_loader_v1
    return sidecar_loader_v1.report()


def launch_fastpath_report():
    """Host-side fast-path counters for the frame report."""
    import launch_fastpath_v1
    import spill_preflight_v1
    from nr_backend import sampling
    return dict(launch=launch_fastpath_v1.report(),
                select=spill_preflight_v1.select_stats(),
                scalar=sampling.scalar_cache_stats())



def load_profile(path, expected_sha256):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('NR profile checksum mismatch')
    profile = json.loads(data)
    if profile['schema'] != 1 or profile['profile'] != PROFILE_ID:
        raise ValueError('Unsupported NR profile')
    scales = [arrays.load(row) for row in profile['vit_hidden_scales']]
    if len(scales) != 8 or len(profile['c512']) != 16:
        raise ValueError('Incomplete NR profile')
    return scales, profile['c512']


@dataclass(frozen=True)
class FrameResult:
    color: torch.Tensor
    low_color: torch.Tensor
    sequence: int
    reset_applied: bool


class Session:
    def __init__(self, stack):
        """Advanced ownership constructor: Session owns and closes this stack."""
        if not isinstance(stack, Stack):
            raise TypeError('Expected the consolidated product stack')
        self._stack = stack
        self._scalers = {}
        self._source_size = None
        self._closed = False
        self._failed = False

    @classmethod
    def create(cls, exact_root, profile_path, profile_sha256, *, share_with=None):
        with _SERIAL:
            scales, calibration = load_profile(profile_path, profile_sha256)
            return cls(Stack(Path(exact_root), hidden_scales=scales, calibration=calibration,
                             share_with=share_with))

    def _ready(self):
        if self._closed:
            raise RuntimeError('NR session is closed')
        if self._failed:
            raise RuntimeError('NR execution failed; close and recreate the session')

    @torch.inference_mode()
    def process(self, rgb, motion, *, reset=False):
        with _SERIAL:
            self._ready()
            if not isinstance(rgb, torch.Tensor) or not isinstance(motion, torch.Tensor):
                raise TypeError('Expected GPU tensors')
            size = tuple(rgb.shape[:2])
            if rgb.ndim != 3 or rgb.shape[-1] != 3 or size not in SOURCE_SIZES:
                raise ValueError('Supported HWC RGB sizes: 1920x1080, 864x480, 256x256')
            if rgb.device.type != 'xpu' or rgb.dtype != torch.float32:
                raise ValueError('Color must be float32 SDR RGB on XPU')
            if motion.device != rgb.device or motion.dtype != torch.float32 or motion.shape != (*size, 2):
                raise ValueError('Motion must be same-device float32 HWC2 pixel displacement')
            if self._source_size is not None and size != self._source_size and not reset:
                raise ValueError('Source size change requires reset=True')
            applied_reset = bool(reset or self._source_size is None)
            try:
                if size != (256, 256) and size not in self._scalers:
                    self._scalers[size] = ResidualScale(256) if size == (1080, 1920) else Face480Scale()
                scaler = self._scalers.get(size)
                with self._stack.installed(), use_arithmetic_backend('triton'):
                    canvas, flow = (rgb, motion) if scaler is None else scaler.prepare(rgb, motion)
                    low = self._stack.model(canvas, flow, reset=applied_reset)
                    color = low.float() if scaler is None else scaler.composite(rgb, canvas, low)
                    # Establish completion before changing a process-global scope
                    # or allowing another serialized session to execute.
                    torch.xpu.synchronize()
                self._source_size = size
                return FrameResult(color, low, self._stack.model.next_seed, applied_reset)
            except BaseException:
                self._failed = True
                raise

    def reset(self):
        with _SERIAL:
            self._ready()
            self._stack.model.reset()
            self._source_size = None

    def close(self):
        with _SERIAL:
            if not self._closed:
                self._stack.close()
                self._scalers.clear()
                self._stack = None
                self._closed = True

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self, *_):
        self.close()
