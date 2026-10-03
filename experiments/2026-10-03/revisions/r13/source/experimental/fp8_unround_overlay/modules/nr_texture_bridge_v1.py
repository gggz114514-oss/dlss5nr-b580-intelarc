"""In-process GPU texture boundary; no CPU pixel transport or motion estimator.

The native worker must finish resource transitions to NON_PIXEL_SHADER_RESOURCE
and provide a producer fence. A downstream aggregate fence retires each borrowed
output before reuse. Calls are serialized on one host thread and one XPU stream.
"""
import ctypes as C
from dataclasses import dataclass
from pathlib import Path
import threading
import sys
import time


# One optional asynchronous owner per process. Failed owners remain reachable
# with DLL/stream/tensors until isolated worker exit; failure cannot accumulate
# new per-frame imports by constructing replacement asynchronous bridges.
_gpu_handoff_owner = None


class HandoffStats(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in (
        'abi_size', 'version', 'enabled', 'active', 'poisoned', 'phase', 'records', 'forward_live')]
    _fields_ += [(name, C.c_uint64) for name in (
        'frames_started', 'frames_retired', 'forward_imports', 'forward_releases',
        'pack_submits', 'unpack_submits', 'xpu_waits', 'xpu_signals',
        'consumer_registrations', 'busy_rejections', 'pack_value', 'output_value',
        'backward_value', 'retained_bytes', 'borrowed_queue', 'native_context', 'native_device')]


class HandoffConfig(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in
                ('abi_size', 'version', 'enabled', 'producer_fence_required')]
    _fields_ += [(name, C.c_uint64) for name in ('borrowed_sycl_queue', 'borrowed_d3d12_queue')]


class HandoffInfo(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in (
        'abi_size', 'version', 'enabled', 'healthy', 'reuse_safe_idle', 'active', 'poisoned',
        'queue_context_equal', 'in_order', 'native_luid_matched', 'max_active_frames',
        'private_records', 'forward_import_per_wait', 'producer_cpu_wait_bypass_supported',
        'producer_cpu_wait_bypass_configured', 'producer_fence_required')]
    _fields_ += [(name, C.c_uint64) for name in (
        'borrowed_sycl_queue', 'native_context', 'native_device', 'd3d12_device',
        'd3d12_queue', 'active_frame', 'retained_producer_fence', 'retained_producer_value')]


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
    def __init__(self, dll_path, width, height, *, native_device=0, native_queue=0,
                 gpu_handoff=False):
        global _gpu_handoff_owner
        if type(gpu_handoff) is not bool:
            raise TypeError('gpu_handoff must be a bool')
        if gpu_handoff and _gpu_handoff_owner is not None:
            raise RuntimeError('An asynchronous bridge already owns this worker; close it first')
        self.gpu_handoff = False
        self._gpu_refs = []
        self._gpu_failed = False
        self._gpu_exported = False
        self._producer_fence_required = False
        self._ever_gpu_handoff = False
        import torch
        self.torch = torch
        self.width, self.height = width, height
        self.thread = threading.get_ident()
        self.native_thread = threading.get_native_id()
        self._native_queue = int(native_queue)
        self.stream = torch.xpu.current_stream()
        self.device = torch.device('xpu', torch.xpu.current_device())
        self.dll = C.CDLL(str(Path(dll_path).resolve()))
        self._bind()
        self.handle = None
        self.frame_id = None
        if gpu_handoff and not self._handoff_supported:
            raise RuntimeError('This DLL has no optional GPU handoff ABI')
        if gpu_handoff:
            _gpu_handoff_owner = self
        error = C.create_string_buffer(8192)
        try:
            self.handle = self.dll.nr_texture_create(self.stream.sycl_queue, native_device,
                native_queue, width, height, error, len(error))
        except BaseException:
            if _gpu_handoff_owner is self:
                _gpu_handoff_owner = None
            raise
        if not self.handle:
            if _gpu_handoff_owner is self:
                _gpu_handoff_owner = None
            raise RuntimeError(error.value.decode('utf-8', errors='replace'))
        if self._owner_transfer_supported and not self._native_queue:
            _, self._native_queue = self.test_context()
        if gpu_handoff:
            self.set_gpu_handoff(True)

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
        # Default construction remains compatible with the installed old DLL.
        optional = {'set_gpu_handoff': [p,u], 'poll_idle': [p,C.POINTER(u)],
                    'gpu_handoff_stats': [p,C.POINTER(HandoffStats)],
                    'configure_gpu_handoff': [p,C.POINTER(HandoffConfig)],
                    'gpu_handoff_info': [p,C.POINTER(HandoffInfo)]}
        self._handoff_supported = all(hasattr(self.dll, 'nr_texture_'+name) for name in optional)
        if self._handoff_supported:
            for name, args in optional.items():
                fn = getattr(self.dll, 'nr_texture_'+name)
                fn.argtypes = args + [C.c_char_p,u]
                fn.restype = C.c_int
        self._owner_transfer_supported = hasattr(self.dll, 'nr_texture_transfer_owner')
        if self._owner_transfer_supported:
            fn = self.dll.nr_texture_transfer_owner
            fn.argtypes = [p, C.c_uint64, C.c_uint64, C.c_uint64, C.POINTER(u), C.c_char_p, u]
            fn.restype = C.c_int

    def transfer_serial_thread(self, *, game_adapter, serial_guard, previous_thread, timeout=10.0):
        """Cold CPU-worker change under the actual frame/retire serial owner.

        No stream, queue, graph, input mapping, history or fence is substituted.
        Python ownership changes only after the native idle proof succeeds.
        """
        if (sys.modules.get('cyberpunk_nr_adapter') is not game_adapter or
                sys._getframe(1).f_code is not game_adapter.process.__code__ or
                serial_guard is not getattr(game_adapter, '_process_serial_lock', None) or
                not callable(getattr(serial_guard, '_is_owned', None)) or
                not serial_guard._is_owned() or previous_thread != self.thread or
                not self.handle or self._gpu_failed):
            raise RuntimeError('Texture owner transfer requires the bound serial adapter process')
        if self.torch.xpu.current_stream().sycl_queue != self.stream.sycl_queue:
            raise RuntimeError('Texture owner transfer changed the model XPU queue')
        current = threading.get_ident()
        if current == self.thread:
            return
        if not self._owner_transfer_supported:
            if self._handoff_supported or self._ever_gpu_handoff:
                raise RuntimeError('GPU handoff helper lacks explicit owner transfer ABI')
            # Original helpers do not implement the optional thread-owned GPU
            # path; retain the existing serialized legacy-only adapter behavior.
            self.thread, self.native_thread = current, threading.get_native_id()
            return
        if not isinstance(timeout, (int, float)) or not 0 < timeout <= 10:
            raise ValueError('Owner transfer timeout must be between zero and 10 seconds')
        deadline = time.monotonic() + timeout
        adopted, error = C.c_uint32(), C.create_string_buffer(8192)
        while True:
            rc = self.dll.nr_texture_transfer_owner(
                self.handle, self.native_thread, self.stream.sycl_queue, self._native_queue,
                C.byref(adopted), error, len(error))
            if rc:
                raise RuntimeError(error.value.decode('utf-8', errors='replace'))
            if adopted.value == 1:
                break
            if adopted.value != 0 or time.monotonic() >= deadline:
                raise RuntimeError('Texture owner transfer did not prove actual retired idle')
            time.sleep(.001)
        self.native_thread, self.thread = threading.get_native_id(), current
        self._gpu_refs.clear()
        self._gpu_exported = False

    def set_gpu_handoff(self, enabled):
        global _gpu_handoff_owner
        self._ready()
        if type(enabled) is not bool:
            raise TypeError('gpu_handoff must be a bool')
        if not self._handoff_supported:
            raise RuntimeError('This DLL has no optional GPU handoff ABI')
        if enabled:
            if _gpu_handoff_owner is not None and _gpu_handoff_owner is not self:
                raise RuntimeError('An asynchronous bridge already owns this worker; close it first')
            # Reserve before the native import can throw. On failure this keeps
            # the native handle and all queue/context/DLL owners reachable.
            _gpu_handoff_owner = self
            self._ever_gpu_handoff = True
        self._call('set_gpu_handoff', int(enabled))
        self.gpu_handoff = enabled
        self._gpu_refs.clear()  # native setter proved actual retired idle
        self._gpu_exported = False

    def poll_idle(self):
        if not self._handoff_supported:
            raise RuntimeError('This DLL has no completion observation ABI')
        idle = C.c_uint32()
        self._call('poll_idle', C.byref(idle))
        if idle.value:
            self._gpu_refs.clear()
            self._gpu_exported = False
        return bool(idle.value)

    def handoff_stats(self):
        if not self._handoff_supported:
            raise RuntimeError('This DLL has no GPU handoff stats ABI')
        stats = HandoffStats()
        stats.abi_size = C.sizeof(stats)
        self._call('gpu_handoff_stats', C.byref(stats))
        return {name: getattr(stats, name) for name, _ in stats._fields_}

    def handoff_info(self):
        if not self._handoff_supported:
            raise RuntimeError('This DLL has no GPU handoff info ABI')
        info = HandoffInfo()
        info.abi_size = C.sizeof(info)
        self._call('gpu_handoff_info', C.byref(info))
        return {name: getattr(info, name) for name, _ in info._fields_}

    def gpu_handoff_info(self):
        return self.handoff_info()

    def configure_gpu_handoff(self, enabled, *, producer_fence_required):
        global _gpu_handoff_owner
        self._ready()
        if (type(enabled) is not bool or type(producer_fence_required) not in (bool, int) or
                producer_fence_required not in (0, 1)):
            raise TypeError('GPU handoff enabled must be bool and producer_fence_required bool or 0/1')
        if not self._handoff_supported:
            raise RuntimeError('This DLL has no GPU handoff config ABI')
        if enabled:
            if _gpu_handoff_owner is not None and _gpu_handoff_owner is not self:
                raise RuntimeError('An asynchronous bridge already owns this worker')
            _gpu_handoff_owner = self
            self._ever_gpu_handoff = True
        _, queue = self.test_context()
        config = HandoffConfig(C.sizeof(HandoffConfig), 1, int(enabled),
                               int(producer_fence_required), self.stream.sycl_queue, queue)
        self._call('configure_gpu_handoff', C.byref(config))
        self.gpu_handoff = enabled
        self._producer_fence_required = bool(producer_fence_required)
        self._gpu_refs.clear()
        self._gpu_exported = False

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
        if self._ever_gpu_handoff and self._gpu_failed:
            raise RuntimeError('Bridge with handoff ownership failed; retain owners and stop this worker')
        if self.gpu_handoff:
            if self._producer_fence_required and not (frame.producer_fence and frame.producer_value):
                raise ValueError('Configured GPU handoff requires the exact SourceFrame producer fence/value')
            if self._gpu_failed:
                raise RuntimeError('Asynchronous bridge failed; retain owners and stop this worker')
            if not self.poll_idle():
                raise RuntimeError('Previous asynchronous frame has not actually retired')
        rgb, motion = self.empty(3), self.empty(2)
        if self._ever_gpu_handoff:
            self._gpu_refs = [rgb, motion]
        try:
            self._call('prepare', frame.color, frame.motion, frame.producer_fence,
                frame.producer_value, frame.frame_id, frame.previous_id, int(frame.reset),
                rgb.data_ptr(), motion.data_ptr())
        except BaseException:
            if self._ever_gpu_handoff:
                self._gpu_failed = True
            raise
        self.frame_id = frame.frame_id
        return rgb, motion

    def export(self, color, *, frame_id):
        self._tensor(color, 3)
        if self._ever_gpu_handoff and self._gpu_failed:
            raise RuntimeError('Bridge with handoff ownership failed; retain owners and stop this worker')
        if self.gpu_handoff:
            if self._gpu_failed or self._gpu_exported:
                raise RuntimeError('Asynchronous output already exported or worker failed')
            if frame_id != self.frame_id:
                raise ValueError('NR output frame ID mismatch')
        if self._ever_gpu_handoff:
            self._gpu_refs.append(color)
        try:
            self._call('export', frame_id, color.data_ptr())
        except BaseException:
            if self._ever_gpu_handoff:
                self._gpu_failed = True
            raise
        if self.gpu_handoff:
            self._gpu_exported = True
        elif self._ever_gpu_handoff:
            self._gpu_refs.clear()  # original OFF export completed both queues synchronously

    def output(self):
        resource, fence, value = C.c_void_p(), C.c_void_p(), C.c_uint64()
        self._call('output', C.byref(resource), C.byref(fence), C.byref(value))
        return TextureOutput(resource.value, fence.value, value.value,
                             self.frame_id, self.width, self.height)

    def retire(self, fence, value):
        self._call('retire', fence, value)

    def close(self):
        global _gpu_handoff_owner
        if self.handle:
            self._call('close')
            self.handle = None
            self._gpu_refs.clear()
            if _gpu_handoff_owner is self:
                _gpu_handoff_owner = None

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
