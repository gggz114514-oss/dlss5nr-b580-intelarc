"""Owned fast-only removal of the four decoder merge output FP8 rounds.

Only ROUND_OUTPUT changes. Preserve skip input policy, compensated half FMA,
nearest sampling and the raw padded C32 merge. Still four quantized-entry
kernel launches, now storing FP16 without E4M3 rounding.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch
import triton

import decoder_gather_merge_v1 as kernels
from quantization_dataflow_v1 import CONTRACTS
from nr_backend.unround_policy import ENABLED, FAMILIES


CONTRACT_KEY = "decoder_gather_merge_v1._quantized"
RAW_CONTRACT = (("OUT",), ())


class MergeCounter:
    def __init__(self, targets):
        self.targets = targets
        self.calls = dict.fromkeys(targets, 0)
        self.capture_calls = dict.fromkeys(targets, 0)
        self.resources = {}
        self._compiled = {}
        self.reference_kernel = kernels._quantized

    def preflight(self):
        from spill_preflight_v1 import select

        for label, (module, shape) in self.targets.items():
            h, w, c = shape
            device = module.weight.device
            p = torch.empty((h // 2, w // 2, c), dtype=torch.float16, device=device)
            s, out = torch.empty(shape, dtype=torch.float16, device=device), torch.empty(
                shape, dtype=torch.float16, device=device)
            scale = module.skip_scale if label == "decoder_input" else module.input_skip_scale
            params = (p, s, scale, out, h, w, c, p.stride(), s.stride(),
                      scale.stride(0), 256, False)
            _, kernel, resource = select(
                kernels._quantized, [(256,)], lambda _: params,
                lambda _: (triton.cdiv(out.numel(), 256),),
                num_warps=4, enable_fp_fusion=False)
            self.resources[label] = {"kernel_hash": kernel.hash, "round_output": False,
                                     "shape": list(shape), "resource_gate": resource}
            self._compiled[label] = kernel
        return self.resources


@contextmanager
def installed(session):
    stack = session._stack
    decoder = stack.decoder_gather
    if (stack.provider.mode != "fp16_xmx" or ENABLED != FAMILIES or
            stack.graph.entries or decoder.unround_merge):
        raise RuntimeError("Decoder merge candidate requires a fresh unrounded baseline")
    modules = dict(decoder.modules)
    expected = {"decoder_input": (24, 40, 512), "decoder.0.0": (48, 80, 256),
                "decoder.1.0": (96, 160, 128), "decoder.2.0": (192, 320, 64)}
    if set(modules) != set(expected) | {"decoder.3.0"}:
        raise RuntimeError("Unexpected owned decoder transition set")
    targets = {label: (modules[label], shape) for label, shape in expected.items()}
    if "merge" in decoder.__dict__ or "installed" in decoder.__dict__:
        raise RuntimeError("Decoder merge owner already has an instance override")
    counter = MergeCounter(targets)
    original_merge, original_installed = decoder.merge, decoder.installed

    def merge(self, name, projected, skip, scale, shift=None):
        try:
            if name not in targets:
                # Raw padded C32 keeps exactly its existing implementation.
                return original_merge(name, projected, skip, scale, shift)
            module, shape = targets[name]
            expected_scale = module.skip_scale if name == "decoder_input" else module.input_skip_scale
            expected_projection = (shape[0] // 2, shape[1] // 2, shape[2])
            if name not in counter._compiled or kernels._quantized is not counter.reference_kernel:
                raise RuntimeError(f"Decoder merge requires its owned preflight before launch: {name}")
            if (tuple(skip.shape) != shape or tuple(projected.shape) != expected_projection or
                    scale is not expected_scale or shift is not None or
                    not projected.is_contiguous() or not scale.is_contiguous() or
                    any(value.dtype != torch.float16 or value.device != scale.device
                        for value in (projected, skip))):
                raise RuntimeError(f"Decoder merge boundary changed: {name}")
            quantized_skip = self.boundary(f"c{shape[2]}", skip)
            if not quantized_skip.is_contiguous():
                raise RuntimeError(f"Decoder merge skip layout changed: {name}")
            value = torch.empty(shape, dtype=torch.float16, device=skip.device)
            params = (projected, quantized_skip, scale, value, *shape,
                      projected.stride(), quantized_skip.stride(), scale.stride(0), 256, False)
            grid, options = (triton.cdiv(value.numel(), 256),), dict(num_warps=4, enable_fp_fusion=False)
            expected = counter._compiled[name]
            ready = kernels._quantized.warmup(*params, grid=grid, **options)
            if ready is not expected or ready.hash != counter.resources[name]["kernel_hash"]:
                raise RuntimeError(f"Decoder merge missed its screened binary before launch: {name}")
            launched = kernels._quantized[grid](*params, **options)
            if launched is not expected:
                raise RuntimeError(f"Decoder merge dispatch changed its screened binary: {name}")
            resource = dict(hash=launched.hash, spills=launched.n_spills,
                            registers=launched.n_regs, shared_bytes=launched.metadata.shared,
                            selection=counter.resources[name]["resource_gate"])
            key = name + ":" + resource["hash"]
            if key in self.resources and self.resources[key] != resource:
                raise RuntimeError(f"Decoder merge resource receipt changed: {name}")
            self.resources[key] = resource
            self.calls[name] = self.calls.get(name, 0) + 1
            if self.probe is not None:
                self.probe(name, projected, quantized_skip, scale, shift, value)
            counter.calls[name] += 1
            if torch.xpu.is_current_stream_capturing():
                counter.capture_calls[name] += 1
            return value
        except BaseException:
            session._failed = True
            raise

    @contextmanager
    def declared_installed(self):
        # The frozen G owner registers the old FP8-output receipt. Adjust it
        # only during this owned session's installation and restore before
        # its existing cleanup checks execute.
        with original_installed() as active:
            previous = CONTRACTS[CONTRACT_KEY]
            CONTRACTS[CONTRACT_KEY] = RAW_CONTRACT
            try:
                yield active
            finally:
                valid = CONTRACTS.get(CONTRACT_KEY) == RAW_CONTRACT
                CONTRACTS[CONTRACT_KEY] = previous
                if not valid:
                    session._failed = True
                    raise RuntimeError("Decoder merge output contract changed during capture")

    bound_merge = MethodType(merge, decoder)
    bound_installed = MethodType(declared_installed, decoder)
    decoder.merge, decoder.installed = bound_merge, bound_installed
    decoder.unround_merge = True
    try:
        yield counter
    finally:
        valid = (decoder.__dict__.get("merge") is bound_merge and
                 decoder.__dict__.get("installed") is bound_installed and decoder.unround_merge)
        del decoder.merge, decoder.installed
        decoder.unround_merge = False
        if stack.graph.entries and not getattr(session, "_closed", False):
            session._failed = True
        if not valid:
            raise RuntimeError("Decoder merge owner changed while its scope was active")
