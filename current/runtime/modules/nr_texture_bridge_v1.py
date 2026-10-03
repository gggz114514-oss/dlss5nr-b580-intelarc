"""In-process GPU texture boundary; no CPU pixel transport or motion estimator.

The native worker must finish resource transitions to NON_PIXEL_SHADER_RESOURCE
and provide a producer fence. A downstream aggregate fence retires each borrowed
output before reuse. Calls are serialized on one host thread and one XPU stream.
"""
import ctypes as C
from dataclasses import dataclass
from pathlib import Path
import threading


@dataclass(frozen=True)
class SourceFrame:
    color: int
    motion: int
    width: int
    height: int
    frame_id: int
    previous_id: int
    reset: bool
    producer_fence: int = 0
    producer_value: int = 0
    motion_direction: str = 'current_to_previous'
    motion_units: str = 'source_pixels'
    color_space: str = 'sdr_rgb'
    generated: bool = False


@dataclass(frozen=True)
class TextureOutput:
    resource: int
    fence: int
    value: int
    frame_id: int
    width: int
    height: int
    format: str = 'RGBA32F'


class TextureBridge:
    def __init__(self, dll_path, width, height, *, native_device=0, native_queue=0):
        import torch
        self.torch = torch
        self.width, self.height = width, height
        self.thread = threading.get_ident()
        self.stream = torch.xpu.current_stream()
        self.device = torch.device('xpu', torch.xpu.current_device())
        self.dll = C.CDLL(str(Path(dll_path).resolve()))
        self._bind()
        self.handle = None
        self.frame_id = None
        error = C.create_string_buffer(8192)
        self.handle = self.dll.nr_texture_create(self.stream.sycl_queue, native_device,
            native_queue, width, height, error, len(error))
        if not self.handle:
            raise RuntimeError(error.value.decode('utf-8', errors='replace'))

    def _bind(self):
        p, u, q = C.c_void_p, C.c_uint32, C.c_uint64
        pp, qp = C.POINTER(p), C.POINTER(q)
        specs = {
            'create': ([p,p,p,u,u], p), 'prepare': ([p,p,p,p,q,q,q,u,p,p], C.c_int),
            'export': ([p,q,p], C.c_int), 'output': ([p,pp,pp,qp], C.c_int),
            'retire': ([p,p,q], C.c_int), 'close': ([p], C.c_int),
            'fixture': ([p,u,u,pp,pp], C.c_int), 'audit_output': ([p,p], C.c_int),
            'audit_inputs': ([p,p,p], C.c_int), 'test_context': ([p,pp,pp], C.c_int),
        }
        for name, (args, result) in specs.items():
            fn = getattr(self.dll, 'nr_texture_'+name)
            fn.argtypes = args + [C.c_char_p, u]
            fn.restype = result

    def _ready(self):
        if not self.handle:
            raise RuntimeError('Texture bridge is closed')
        if threading.get_ident() != self.thread:
            raise RuntimeError('Texture bridge requires its owning host thread')
        if self.torch.xpu.current_stream().sycl_queue != self.stream.sycl_queue:
            raise RuntimeError('Texture bridge requires its owning XPU stream')

    def _call(self, name, *args):
        self._ready()
        error = C.create_string_buffer(8192)
        rc = getattr(self.dll, 'nr_texture_'+name)(self.handle, *args, error, len(error))
        if rc:
            raise RuntimeError(error.value.decode('utf-8', errors='replace'))

    def _tensor(self, value, channels):
        t = self.torch
        if (not isinstance(value, t.Tensor) or value.dtype != t.float32 or
                value.device != self.device or not value.is_contiguous() or
                tuple(value.shape) != (self.height, self.width, channels)):
            raise ValueError('Expected owned-stream device float32 contiguous HWC tensor')

    def empty(self, channels):
        self._ready()
        return self.torch.empty((self.height, self.width, channels),
                               dtype=self.torch.float32, device=self.device)

    def prepare(self, frame):
        self._ready()
        if not isinstance(frame, SourceFrame):
            raise TypeError('Expected SourceFrame')
        if (frame.width, frame.height) != (self.width, self.height):
            raise ValueError('Source/bridge geometry mismatch')
        if (frame.generated or frame.motion_direction != 'current_to_previous' or
                frame.motion_units != 'source_pixels' or frame.color_space != 'sdr_rgb'):
            raise ValueError('Expected original SDR source and current-to-previous source-pixel motion')
        for v in (frame.frame_id, frame.previous_id, frame.producer_value):
            if not isinstance(v, int) or not 0 <= v < 2**64:
                raise ValueError('Invalid unsigned frame/fence value')
        rgb, motion = self.empty(3), self.empty(2)
        self._call('prepare', frame.color, frame.motion, frame.producer_fence,
            frame.producer_value, frame.frame_id, frame.previous_id, int(frame.reset),
            rgb.data_ptr(), motion.data_ptr())
        self.frame_id = frame.frame_id
        return rgb, motion

    def export(self, color, *, frame_id):
        self._tensor(color, 3)
        self._call('export', frame_id, color.data_ptr())

    def output(self):
        resource, fence, value = C.c_void_p(), C.c_void_p(), C.c_uint64()
        self._call('output', C.byref(resource), C.byref(fence), C.byref(value))
        return TextureOutput(resource.value, fence.value, value.value,
                             self.frame_id, self.width, self.height)

    def retire(self, fence, value):
        self._call('retire', fence, value)

    def close(self):
        if self.handle:
            self._call('close')
            self.handle = None

    # Diagnostic endpoints are deliberately separate from the product path.
    def fixture(self, seed, motion_bits):
        color, motion = C.c_void_p(), C.c_void_p()
        self._call('fixture', seed, motion_bits, C.byref(color), C.byref(motion))
        return color.value, motion.value

    def audit_output(self):
        target = self.empty(3)
        self._call('audit_output', target.data_ptr())
        return target

    def audit_inputs(self):
        color, motion = self.empty(3), self.empty(2)
        self._call('audit_inputs', color.data_ptr(), motion.data_ptr())
        return color, motion

    def test_context(self):
        device, queue = C.c_void_p(), C.c_void_p()
        self._call('test_context', C.byref(device), C.byref(queue))
        return device.value, queue.value


class TextureNR:
    """Connect an independently configured NR Session to a native frame stream.

Exact and fast sessions must be loaded in separately configured worker processes.
The supplied session owns NR history; this adapter never estimates/rescales flow,
replaces depth controls, transforms color, or routes generated FG frames to NR.
"""
    def __init__(self, bridge, session):
        self.bridge, self.session = bridge, session
        self.failed = False

    def process(self, frame):
        if self.failed:
            raise RuntimeError('NR texture execution failed; stop this worker')
        color, motion = self.bridge.prepare(frame)
        try:
            result = self.session.process(color, motion, reset=frame.reset)
            self.bridge.export(result.color.contiguous(), frame_id=frame.frame_id)
            return self.bridge.output()
        except BaseException:
            self.failed = True
            raise
