"""One summary read before temporal commit; exact route keeps its original guards."""
from contextlib import contextmanager
from threading import get_ident
import sys


class InputSummary720:
    def __init__(self, child):
        self.child = child
        self.motion = self.rgb = self.clamped_rgb = None
        self.flags = None
        self.thread = None
        self.readbacks = 0

    def require(self, model):
        c = self.child
        if (c.model is not model or c.modes.session is not c.session
                or c.session.__dict__.get('_history_numeric_suite_720') is not c
                or not c.active or c.retired or not c.in_frame
                or c.thread != get_ident() or c.torch.xpu.is_current_stream_capturing()
                or sys.modules[type(c).__module__]._IN_FLIGHT is not c):
            raise RuntimeError('Input summary requires the actual serial outside-graph history frame')

    def begin(self, model, rgb, motion, check_color):
        self.require(model)
        if self.motion is not None:
            raise RuntimeError('Nested input summary frame')
        c = self.child
        if (tuple(rgb.shape) != (720, 1280, 3) or rgb.device != c.device or rgb.dtype != c.torch.float32
                or tuple(motion.shape) != (720, 1280, 2) or motion.device != c.device
                or motion.dtype not in (c.torch.float16, c.torch.float32)
                or not rgb.is_contiguous() or not motion.is_contiguous()):
            raise ValueError('Summary expects prepared contiguous 720p FP32 RGB / FP16-or-FP32 RG pixel motion')
        partial = c._prepare_buffers['input_partial']
        flags = c._prepare_buffers['input_flags']
        block, count = 1024, 720 * 1280 * (3 if check_color else 2)
        chunks = c.triton.cdiv(count, block)
        label = ('input_fp16' if motion.dtype == c.torch.float16 else 'input_fp32') + ('_color' if check_color else '_motion')
        c._launch(label, c.jits['_input_summary'],
                  (motion, rgb, partial, 720 * 1280 * 2, 720 * 1280 * 3, check_color, block), count, block)
        c._launch('input_flags_color' if check_color else 'input_flags_motion', c.jits['_summary_flags'], (partial, flags, chunks, 4096), 4096, 4096)
        bits = int(flags.item())
        self.readbacks += 1
        if check_color and bits & 8:
            raise ValueError('Game NR color must be finite')
        if bits & 3:
            raise ValueError('Motion must be finite and within the FP16 texture range +/-65504 pixels')
        self.motion, self.rgb, self.flags, self.thread = motion, rgb, bits, get_ident()

    def end(self):
        self.motion = self.rgb = self.clamped_rgb = self.flags = self.thread = None

    def current(self, model, motion=None):
        self.require(model)
        return (self.flags is not None and self.thread == get_ident()
                and (motion is None or motion is self.motion))


def _summary(model):
    modes = model.__dict__.get('_audit_history_host_modes_720')
    if modes is None:
        return None
    child = modes.session.__dict__.get('_history_numeric_suite_720') if modes.session is not None else None
    result = None if child is None else getattr(child, 'input_summary', None)
    if result is not None and child.model is not model:
        raise RuntimeError('Input summary belongs to a different model')
    return result


@contextmanager
def input_summary_frame(model, rgb, motion, *, check_color):
    summary = _summary(model)
    if summary is None:
        yield None
        return
    try:
        summary.begin(model, rgb, motion, check_color)
        yield summary
    finally:
        summary.end()


def validated_motion(model, motion):
    summary = _summary(model)
    return summary is not None and summary.current(model, motion)


def validated_sdr_color(model, rgb):
    summary = _summary(model)
    return summary is not None and summary.current(model) and summary.clamped_rgb is rgb
