"""Owned decoder transitions only. Original projection and body computations stay.

Five boundaries: ViT -> C512, then C256/C128/C64/C32. The C32 unquantized
residual includes its original padded context; all other stores cross the same
existing FP8 boundary. This scope adds no buffers or changed weights.
"""
from contextlib import contextmanager
from types import MethodType
import torch
import nr_backend.decoder as decoder
from . import kernels
class DecoderGather:
    def __init__(self, model):
        self.model = model
        self.diagnostics = True
        self.modules = [('decoder_input', model.decoder_input)] + [
            (f'decoder.{i}.0', group[0]) for i, group in enumerate(model.decoder)]
        assert len(self.modules) == 5
        assert isinstance(self.modules[0][1], decoder.DecoderInputUpsample)
        assert [m.channels for _, m in self.modules[1:]] == [256, 128, 64, 32]
        self.original = {name: m.forward for name, m in self.modules}
        assert all('forward' not in m.__dict__ for _, m in self.modules)
        self.calls, self.resources, self.probe = {}, {}, None

    def merge(self, name, projected, skip, scale, shift=None):
        quantized_skip = decoder.q(skip)
        out, resource = kernels.forward(projected, quantized_skip, scale, shift=shift, diagnostics=self.diagnostics)
        if self.diagnostics:
            key = name + ':' + resource['hash']
            self.resources[key] = resource
            self.calls[name] = self.calls.get(name, 0) + 1
        if self.probe is not None:
            self.probe(name, projected, quantized_skip, scale, shift, out)
        return out

    def input(self, module, features, skip):
        if features.ndim != 3 or features.shape[-1] != 1024 or skip.ndim != 3 or skip.shape[-1] != 512:
            raise ValueError('Expected HWC1024 input and HWC512 skip')
        h, w = skip.shape[:2]
        if min(h, w) == 0 or h > 2 * features.shape[0] or w > 2 * features.shape[1]:
            raise ValueError('Skip must fit the expanded projection')
        initial = torch.zeros((*features.shape[:2], 512), device=features.device, dtype=torch.float16)
        projected = decoder.split_k_projection(decoder.q(features), module.weight, initial)
        return self.merge('decoder_input', projected, skip, module.skip_scale)

    def upsample(self, name, module, features, skip):
        if features.ndim != 3 or features.shape[-1] != 2 * module.channels or skip.ndim != 3 or skip.shape[-1] != module.channels:
            raise ValueError('Unexpected upsample tensor shapes')
        h, w = skip.shape[:2]
        if min(h, w) == 0 or h > 2 * features.shape[0] or w > 2 * features.shape[1]:
            raise ValueError('Skip must fit the expanded projection')
        projected = decoder.dot(decoder.q(features), module.weight, chunk_k=16)
        if module.channels != 32:
            return module.body(self.merge(name, projected, skip, module.input_skip_scale))
        sy, sx = module.body.window_shift
        padded = self.merge(name, projected, skip, module.input_skip_scale, (sy, sx))
        mlp = module.body.mlp.forward_unquantized(padded)
        attended = module.body.attention(decoder.q(mlp))
        value = decoder.dot(decoder.q(attended), module.body.output_weight, chunk_k=16,
                            initial=(mlp * module.body.skip_scale).half())
        return decoder.q(value[sy:sy+h, sx:sx+w])

    @contextmanager
    def installed(self):
        self.verify_restored()
        replacements = []
        for name, module in self.modules:
            def forward(m, features, skip, name=name):
                if name == 'decoder_input':
                    return self.input(m, features, skip)
                return self.upsample(name, m, features, skip)
            bound = MethodType(forward, module)
            replacements.append((module, bound))
        for module, bound in replacements:
            module.forward = bound
        try:
            yield self
        finally:
            valid = all(m.forward is f for m, f in replacements)
            for module, _ in replacements:
                del module.forward
            assert valid, 'Decoder scope interference'
            self.verify_restored()

    def verify_restored(self):
        assert all('forward' not in m.__dict__ and m.forward == self.original[n] for n, m in self.modules)
