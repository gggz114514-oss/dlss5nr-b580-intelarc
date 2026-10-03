"""Fast C512+K8-only removal of the ten C32 hidden cubic/FP8 lookups.

Reuse the existing native half cubic kernel, with the same four K32 matrix
stages, residual and half stores. Install before capture and retain for the
whole session. The caller must dispose of its captured graphs on scope exit.
The immutable LUT remains registered, but this kernel never reads it.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch
import triton

import native_cubic_adapters_v1 as adapters
import native_cubic_c32_v1 as native
from nr_backend.execution import current_arithmetic_backend, record_arithmetic_dispatch
from nr_backend.unround_policy import ENABLED, FAMILIES


def _targets(model):
    if len(model.encoder[0]) != 4 or len(model.decoder[-1]) != 4:
        raise RuntimeError("Expected four encoder/decoder C32 blocks")
    targets = {"pre": (model.pre.mlp, (768, 1280, 32), "pre")}
    for side, blocks in (("encoder32", model.encoder[0]),
                         ("decoder32", model.decoder[-1])):
        for index, block in enumerate(blocks):
            if side == "decoder32" and index == 0:
                block = block.body
            sy, sx = block.window_shift
            if sy not in (0, 4) or sx not in (0, 4):
                raise RuntimeError("Unexpected C32 window shift")
            shape = (384 + (8 if sy else 0), 640 + (8 if sx else 0), 32)
            targets[f"{side}.{index}"] = (block.mlp, shape, "c32")
    targets["post"] = (model.post.body.mlp, (776, 1288, 32), "post")
    if len(targets) != 10 or len({id(m) for m, _, _ in targets.values()}) != 10:
        raise RuntimeError("Expected ten distinct owned C32 MLP modules")
    return targets


class NativeCubicCounter:
    def __init__(self, targets, owner):
        self.targets = targets
        self.owner = owner
        self.calls = dict.fromkeys(targets, 0)
        self.capture_calls = dict.fromkeys(targets, 0)
        self.resources = {}
        self.kernels_by_site = {}
        self._compiled = {}

    def preflight(self):
        """Load each real M specialization before capture; never launch here."""
        from spill_preflight_v1 import select

        options = self.owner.options
        lut = self.owner.constant.require()
        for label, (module, shape, _) in self.targets.items():
            rows = shape[0] * shape[1]
            if rows not in self._compiled:
                features = torch.empty(shape, dtype=torch.float16, device=module.expansion.device)
                output = torch.empty_like(features)
                params = (features, module.expansion, module.contraction, module.skip_scale,
                          lut, output, rows, options["bm"], False)
                _, kernel, report = select(
                    native._kernel, [(options["bm"],)], lambda _: params,
                    lambda _: (triton.cdiv(rows, options["bm"]),),
                    num_warps=options["warps"], num_stages=options["stages"],
                    enable_fp_fusion=False)
                self._compiled[rows] = kernel
                self.resources[str(rows)] = {
                    "kernel_hash": kernel.hash,
                    "round_activation": False,
                    "source_kernel": "native_cubic_c32_v1._kernel",
                    "shape": list(shape), "resource_gate": report,
                }
            self.kernels_by_site[label] = self._compiled[rows].hash
        return self.resources


@contextmanager
def installed(session):
    """Wrap the existing owner, including pre and the first decoder32 body."""
    stack = session._stack
    if (stack.provider.mode != "fp16_xmx" or ENABLED != FAMILIES or
            stack.graph.entries):
        raise RuntimeError("Native C32 hidden scope requires a fresh unrounded FP16 graph")
    owners = [item for item in stack.components if type(item) is adapters.FusedC32]
    if len(owners) != 1:
        raise RuntimeError("Expected the one installed native-cubic C32 adapter")
    owner = owners[0]
    targets = _targets(stack.model)
    if {id(m) for m in owner.modules} != {id(m) for m, _, _ in targets.values()}:
        raise RuntimeError("C32 owner differs from the ten selected modules")
    if owner.options != {"bm": 32, "warps": 4, "stages": 1}:
        raise RuntimeError("Native hidden candidate must preserve the current C32 launch")
    if "apply" in owner.__dict__:
        raise RuntimeError("C32 apply already has an instance override")
    original = owner.apply
    by_id = {id(module): (label, module, shape, family)
             for label, (module, shape, family) in targets.items()}
    counter = NativeCubicCounter(targets, owner)
    weights = {label: (module.expansion, module.contraction, module.skip_scale)
               for label, (module, _, _) in targets.items()}

    def run_selected(self, module, old_forward, features):
        target = by_id.get(id(module))
        if target is None:
            return original(module, old_forward, features)
        label, expected, shape, family = target
        if any(current is not saved for current, saved in zip(
                (module.expansion, module.contraction, module.skip_scale), weights[label])):
            raise RuntimeError(f"Native C32 hidden weights changed: {label}")
        if (module is not expected or tuple(features.shape) != shape or
                features.dtype != torch.float16 or features.device.type != "xpu" or
                not features.is_contiguous() or self.provider.mode != "fp16_xmx" or
                current_arithmetic_backend() != "triton" or
                getattr(module, "rounding_family", "c32") != family):
            raise RuntimeError(f"Native C32 hidden boundary changed: {label}")
        rows = shape[0] * shape[1]
        if rows not in counter._compiled:
            raise RuntimeError("Native C32 hidden candidate requires preflight before capture")
        result, kernel = native.forward(
            features, module.expansion, module.contraction, module.skip_scale,
            self.constant.require(), round_activation=False, **self.options)
        if kernel is not counter._compiled[rows]:
            raise RuntimeError(f"Native C32 launch missed its screened binary: {label}")
        # These are the existing logical-work receipts, not physical FP8 ops.
        for _ in range(triton.cdiv(rows, 32768)):
            for kind in ("fp8", "cubic_fp8", "dense", "dense"):
                record_arithmetic_dispatch(kind)
        key = str(rows)
        self.calls[key] = self.calls.get(key, 0) + 1
        counter.calls[label] += 1
        if torch.xpu.is_current_stream_capturing():
            counter.capture_calls[label] += 1
        return result

    def selected_apply(self, module, old_forward, features):
        try:
            return run_selected(self, module, old_forward, features)
        except BaseException:
            session._failed = True
            raise

    replacement = MethodType(selected_apply, owner)
    owner.apply = replacement
    try:
        yield counter
    finally:
        valid = owner.__dict__.get("apply") is replacement
        del owner.apply
        if stack.graph.entries and not getattr(session, "_closed", False):
            session._failed = True
        if not valid:
            raise RuntimeError("Native C32 owner changed while its scope was active")
