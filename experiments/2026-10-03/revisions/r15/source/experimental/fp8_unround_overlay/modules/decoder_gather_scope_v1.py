"""Owned decoder transitions only. Original projection and body computations stay.

Five boundaries: ViT -> C512, then C256/C128/C64/C32. The C32 unquantized
residual includes its original padded context; all other stores cross the same
existing FP8 boundary. This scope adds no buffers or changed weights.
"""
from contextlib import contextmanager
from types import MethodType
import os
import torch
import nr_backend.decoder as decoder
from nr_backend.unround_policy import ENABLED, FAMILIES, round_activation
import decoder_gather_merge_v1 as kernels
from quantization_dataflow_v1 import CONTRACTS

EXTRA = {
    'decoder_gather_merge_v1._quantized': (('OUT',), ('OUT',)),
    'decoder_gather_merge_v1._raw_padded': (('OUT',), ()),
}

# A cold-owned callback for the C32 consumer that bypasses body.forward.
# None preserves the original projection/merge/attention route exactly.
_AUDIT_LOCAL = None


class DecoderGather:
    def __init__(self, model):
        self.model = model
        self.modules = [('decoder_input', model.decoder_input)] + [
            (f'decoder.{i}.0', group[0]) for i, group in enumerate(model.decoder)]
        assert len(self.modules) == 5
        assert isinstance(self.modules[0][1], decoder.DecoderInputUpsample)
        assert [m.channels for _, m in self.modules[1:]] == [256, 128, 64, 32]
        self.original = {name: m.forward for name, m in self.modules}
        assert all('forward' not in m.__dict__ for _, m in self.modules)
        selection = os.environ.get('NR_FAST_DECODER_UNROUND', '0')
        if selection not in ('0', '1'):
            raise ValueError('NR_FAST_DECODER_UNROUND must be 0 or 1')
        self.unround_activations = selection == '1'
        if self.unround_activations and ENABLED != FAMILIES:
            raise ValueError('Decoder unround candidate requires NR_FAST_UNROUND=all')
        merge_selection = os.environ.get('NR_FAST_DECODER_MERGE_UNROUND', '0')
        if merge_selection not in ('0', '1'):
            raise ValueError('NR_FAST_DECODER_MERGE_UNROUND must be 0 or 1')
        self.unround_merge = merge_selection == '1'
        if self.unround_merge and ENABLED != FAMILIES:
            raise ValueError('Decoder merge unround requires NR_FAST_UNROUND=all')
        self.calls, self.resources, self.probe = {}, {}, None

    def boundary(self, family, value):
        if self.unround_activations:
            return round_activation(family, value)
        return decoder.q(value)

    def merge(self, name, projected, skip, scale, shift=None):
        family = f'c{skip.shape[-1]}'
        quantized_skip = self.boundary(family, skip)
        out, resource = kernels.forward(projected, quantized_skip, scale, shift=shift,
                                        round_output=not self.unround_merge)
        key = name + ':' + resource['hash']
        if key in self.resources:
            assert self.resources[key] == resource
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
        projected = decoder.split_k_projection(self.boundary('vit', features), module.weight, initial)
        return self.merge('decoder_input', projected, skip, module.skip_scale)

    def upsample(self, name, module, features, skip):
        if features.ndim != 3 or features.shape[-1] != 2 * module.channels or skip.ndim != 3 or skip.shape[-1] != module.channels:
            raise ValueError('Unexpected upsample tensor shapes')
        h, w = skip.shape[:2]
        if min(h, w) == 0 or h > 2 * features.shape[0] or w > 2 * features.shape[1]:
            raise ValueError('Skip must fit the expanded projection')
        family = f'c{module.channels}'
        projected = decoder.dot(self.boundary(family, features), module.weight, chunk_k=16)
        if module.channels != 32:
            return module.body(self.merge(name, projected, skip, module.input_skip_scale))
        sy, sx = module.body.window_shift
        padded = self.merge(name, projected, skip, module.input_skip_scale, (sy, sx))
        if _AUDIT_LOCAL is not None:
            result = _AUDIT_LOCAL.decoder_tail_override(self, name, module, padded, h, w)
            if result is not None:
                return self.boundary('c32', result)
        mlp = module.body.mlp.forward_unquantized(padded)
        attended = module.body.attention(self.boundary('c32', mlp))
        value = decoder.dot(self.boundary('c32', attended), module.body.output_weight, chunk_k=16,
                            initial=(mlp * module.body.skip_scale).half())
        return self.boundary('c32', value[sy:sy+h, sx:sx+w])

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
        CONTRACTS.update(EXTRA)
        for module, bound in replacements:
            module.forward = bound
        try:
            yield self
        finally:
            valid = all(m.forward is f for m, f in replacements)
            for module, _ in replacements:
                del module.forward
            for key, contract in EXTRA.items():
                assert CONTRACTS.pop(key) == contract
            assert valid, 'Decoder scope interference'
            self.verify_restored()

    def verify_restored(self):
        assert not any(k in CONTRACTS for k in EXTRA)
        assert all('forward' not in m.__dict__ and m.forward == self.original[n] for n, m in self.modules)
