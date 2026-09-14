"""Owned scopes for exact history, scheduling and output dependency cropping."""
from contextlib import contextmanager
import torch
from . import body, history_kernel


class History:
    def __init__(self, model):
        self.model = model
        self.calls = 0

    @contextmanager
    def installed(self):
        import nr_backend.temporal as temporal
        original = temporal.warp_history_square
        def replacement(image, motion, *, return_components=False, reciprocal_source=None):
            if (reciprocal_source is not self.model.reciprocal or not return_components
                    or image.device.type != 'xpu' or not image.is_contiguous()
                    or not motion.is_contiguous() or image.dtype != torch.float16
                    or tuple(image.shape[:2]) not in ((256, 256), (512, 512))):
                return original(image, motion, return_components=return_components,
                                reciprocal_source=reciprocal_source)
            (numerator, reciprocal, flags), _ = history_kernel.forward(
                image, motion, reciprocal_source.values)
            if not bool(flags.all()):
                raise ValueError('History input outside native reciprocal interval')
            self.calls += 1
            return numerator, reciprocal
        temporal.warp_history_square = replacement
        try:
            yield
        finally:
            temporal.warp_history_square = original


class Schedule:
    def __init__(self, model):
        self.model = model
        self.calls = 0

    @contextmanager
    def installed(self):
        from nr_backend.executor import ResetNR
        original = ResetNR._forward_front
        def replacement(model, rgb, front, *, progress=None, **options):
            if model is not self.model or progress is not None:
                return original(model, rgb, front, progress=progress, **options)
            with body.installed():
                result = body.forward_front(model, rgb, front, **options)
            # Keep completion before MotionNR commits private history and seed.
            torch.xpu.synchronize(rgb.device)
            self.calls += 1
            return result
        ResetNR._forward_front = replacement
        try:
            yield
        finally:
            ResetNR._forward_front = original


class PostRegion:
    def __init__(self, model):
        self.module = model.post
        self.calls = 0
        self.geometries = []

    def head(self, features, skip, height, width):
        import nr_backend.post as post
        ph, pw = skip.shape[:2]
        if tuple(features.shape) != (ph // 2, pw // 2, 32) or self.module.body.window_shift != (4, 4):
            raise ValueError('Unexpected post geometry')
        eh, ew = ((height + 11) // 8) * 8, ((width + 11) // 8) * 8
        mh, mw = min(ph, eh - 4), min(pw, ew - 4)
        expanded = post.q(features[:(mh + 1)//2, :(mw + 1)//2]).repeat_interleave(2, 0).repeat_interleave(2, 1)[:mh, :mw]
        expanded = (expanded * self.module.input_weight).half()
        merged = post.half_fma(post.q(skip[:mh, :mw]), self.module.input_skip_scale, expanded)
        padded = torch.nn.functional.pad(merged, (0, 0, 4, ew-mw-4, 4, eh-mh-4))
        mlp = self.module.body.mlp.forward_unquantized(padded)
        attended = self.module.body.attention(post.q(mlp))
        crop = (slice(4, height+4), slice(4, width+4))
        full = post.dot(post.q(attended[crop]), self.module.body.output_weight,
                        chunk_k=16, initial=(mlp[crop] * self.module.body.skip_scale).half())
        result = post.dot(full, self.module.head_weight, chunk_k=8)
        self.calls += 1
        return result

    @contextmanager
    def installed(self):
        module = self.module
        if 'forward' in module.__dict__ or 'forward_head' in module.__dict__:
            raise RuntimeError('Post override already installed')
        original = module.forward
        def forward(features, skip, rgb, **options):
            if rgb.device.type != 'xpu':
                return original(features, skip, rgb, **options)
            module.forward_head = lambda f, s: self.head(f, s, *rgb.shape[:2])
            try:
                return original(features, skip, rgb, **options)
            finally:
                del module.forward_head
        module.forward = forward
        try:
            yield
        finally:
            del module.forward
