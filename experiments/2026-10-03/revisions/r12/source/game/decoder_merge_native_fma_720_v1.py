"""Owned 720p decoder half-FMA candidate; output unround is independent.

Five merges use native half FMA. Skip quantization, projection, nearest
sampling, C32 padding and all other decoder/body operations remain intact.
The four output FP8 boundaries may separately remain enabled or be disabled.
Do not nest with the output-only merge scope: select one owner at construction.
"""
from contextlib import contextmanager
from types import MethodType

import torch

import decoder_merge_native_fma_kernel_720_v1 as kernels
from nr_backend.unround_policy import ENABLED, FAMILIES
from quantization_dataflow_v1 import CONTRACTS


SHAPES = {"decoder_input": (24, 40, 512), "decoder.0.0": (48, 80, 256),
          "decoder.1.0": (96, 160, 128), "decoder.2.0": (192, 320, 64),
          "decoder.3.0": (384, 640, 32)}


def _scale(name, module):
    return module.skip_scale if name == "decoder_input" else module.input_skip_scale


class NativeMergeCounter:
    def __init__(self, targets, unround_output):
        self.targets, self.unround_output = targets, unround_output
        self.calls, self.capture_calls = dict.fromkeys(targets, 0), dict.fromkeys(targets, 0)
        self.resources, self._compiled = {}, {}
        self.kernels_by_site = {}
        self.reference_functions = (kernels.parameters, kernels._quantized, kernels._raw_padded)

    def preflight(self):
        from spill_preflight_v1 import select

        for name, (module, shape, shift) in self.targets.items():
            device, c = module.weight.device, shape[2]
            projected = torch.empty((shape[0] // 2, shape[1] // 2, c), device=device,
                                     dtype=torch.float16)
            skip = torch.empty(shape, device=device, dtype=torch.float16)
            jit, args, out, grid = kernels.parameters(
                projected, skip, _scale(name, module), shift=shift,
                round_output=not self.unround_output)
            _, compiled, resource = select(jit, [(256,)], lambda _: args, lambda _: grid,
                                           num_warps=4, enable_fp_fusion=False)
            self._compiled[name] = compiled
            self.resources[name] = dict(kernel_hash=compiled.hash, shape=list(shape),
                                        output_shape=list(out.shape), native_half_fma=True,
                                        round_output=shift is None and not self.unround_output,
                                        source_kernel=f"{jit.fn.__module__}.{jit.fn.__name__}",
                                        resource_gate=resource)
        return self.resources


@contextmanager
def installed(session, *, unround_output=False):
    if type(unround_output) is not bool:
        raise ValueError("Decoder output unround must be a fixed bool")
    stack, decoder = session._stack, session._stack.decoder_gather
    if (stack.provider.mode != "fp16_xmx" or ENABLED != FAMILIES or stack.graph.entries or
            decoder.unround_merge or "merge" in decoder.__dict__ or
            "installed" in decoder.__dict__):
        raise RuntimeError("Native decoder FMA requires a fresh owned unrounded session")
    modules = dict(decoder.modules)
    if set(modules) != set(SHAPES):
        raise RuntimeError("Native decoder FMA lost the five decoder transition owners")
    targets = {name: (modules[name], shape, tuple(modules[name].body.window_shift)
                      if name == "decoder.3.0" else None) for name, shape in SHAPES.items()}
    weights = {name: (module.weight, _scale(name, module))
               for name, (module, _, _) in targets.items()}
    counter = NativeMergeCounter(targets, unround_output)
    original_installed = decoder.installed
    contracts = {f"{kernels.__name__}._quantized": (("OUT",), () if unround_output else ("OUT",)),
                 f"{kernels.__name__}._raw_padded": (("OUT",), ())}
    if any(key in CONTRACTS for key in contracts):
        raise RuntimeError("Native decoder merge contracts already installed")

    def merge(self, name, projected, skip, scale, shift=None):
        try:
            if (name not in targets or self is not decoder or
                    self.unround_merge is not unround_output or
                    (kernels.parameters, kernels._quantized, kernels._raw_padded) != counter.reference_functions):
                raise RuntimeError("Native decoder FMA owner or implementation changed")
            if name not in counter._compiled:
                raise RuntimeError("Native decoder FMA requires preflight before dispatch")
            module, shape, expected_shift = targets[name]
            if (module.weight is not weights[name][0] or _scale(name, module) is not weights[name][1] or
                    scale is not weights[name][1] or tuple(skip.shape) != shape or shift != expected_shift or
                    tuple(projected.shape) != (shape[0] // 2, shape[1] // 2, shape[2]) or
                    not projected.is_contiguous() or not scale.is_contiguous()):
                raise RuntimeError(f"Native decoder FMA boundary changed: {name}")
            quantized_skip = self.boundary(f"c{shape[2]}", skip)
            if not quantized_skip.is_contiguous():
                raise RuntimeError(f"Native decoder skip layout changed: {name}")
            jit, args, out, grid = kernels.parameters(projected, quantized_skip, scale,
                                                       shift=shift, round_output=not unround_output)
            options = dict(num_warps=4, enable_fp_fusion=False)
            expected = counter._compiled[name]
            ready = jit.warmup(*args, grid=grid, **options)
            if ready is not expected or ready.hash != counter.resources[name]["kernel_hash"]:
                raise RuntimeError(f"Native decoder FMA binary changed before launch: {name}")
            launched = jit[grid](*args, **options)
            if launched is not expected:
                raise RuntimeError(f"Native decoder FMA dispatch missed the screened binary: {name}")
            resource = dict(hash=launched.hash, spills=launched.n_spills, registers=launched.n_regs,
                            shared_bytes=launched.metadata.shared,
                            selection=counter.resources[name]["resource_gate"])
            key = name + ":" + resource["hash"]
            if key in self.resources and self.resources[key] != resource:
                raise RuntimeError("Native decoder resource receipt changed")
            self.resources[key] = resource
            self.calls[name] = self.calls.get(name, 0) + 1
            if self.probe is not None:
                self.probe(name, projected, quantized_skip, scale, shift, out)
            counter.calls[name] += 1
            counter.kernels_by_site[name] = launched.hash
            if torch.xpu.is_current_stream_capturing():
                counter.capture_calls[name] += 1
            return out
        except BaseException:
            session._failed = True
            raise

    @contextmanager
    def declared_installed(self):
        with original_installed() as active:
            if any(key in CONTRACTS for key in contracts):
                raise RuntimeError("Native decoder contracts collided during installation")
            CONTRACTS.update(contracts)
            try:
                yield active
            finally:
                valid = all(CONTRACTS.get(key) == value for key, value in contracts.items())
                for key in contracts:
                    CONTRACTS.pop(key, None)
                if not valid:
                    session._failed = True
                    raise RuntimeError("Native decoder output contracts changed")

    bound_merge, bound_installed = MethodType(merge, decoder), MethodType(declared_installed, decoder)
    decoder.merge, decoder.installed, decoder.unround_merge = bound_merge, bound_installed, unround_output
    try:
        yield counter
    except BaseException:
        session._failed = True
        raise
    finally:
        valid = (decoder.__dict__.get("merge") is bound_merge and
                 decoder.__dict__.get("installed") is bound_installed and
                 decoder.unround_merge is unround_output)
        del decoder.merge, decoder.installed
        decoder.unround_merge = False
        if (stack.graph.entries or any(counter.calls.values())) and not getattr(session, "_closed", False):
            session._failed = True
        if not valid:
            session._failed = True
            raise RuntimeError("Native decoder FMA scope ownership changed")
