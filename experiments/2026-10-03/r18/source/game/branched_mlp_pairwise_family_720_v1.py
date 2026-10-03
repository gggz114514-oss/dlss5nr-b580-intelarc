"""Opt-in pairwise MLP scope for a whole C128 and/or C256 block family.

Install before the first 720p graph capture. The baseline projection, tensor
layout, model weights, and every other component remain unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch

import branched_mlp_pairwise_720_v1 as pairwise
from nr_backend.execution import current_arithmetic_backend


EXPECTED = {128: 12, 256: 16}


class FamilyCounter:
    def __init__(self, targets, fused):
        self.targets = targets
        self.fused = fused
        self.calls = {label: 0 for label in targets}
        self.capture_calls = {label: 0 for label in targets}
        self._capturing = False

    def preflight(self):
        """Compile each distinct real block shape under the caller's policy."""
        resources = {}
        for label, (module, shape) in self.targets.items():
            if shape not in resources:
                sample = torch.empty(shape, dtype=torch.float16, device="xpu")
                resources[shape] = pairwise.preflight(sample, module, self.fused)
        return {"x".join(map(str, shape)): value for shape, value in resources.items()}

    @contextmanager
    def capture(self):
        """Require each selected MLP to execute exactly once in one capture."""
        if self._capturing:
            raise RuntimeError("Nested pairwise family capture check")
        before = dict(self.capture_calls)
        self._capturing = True
        try:
            yield self
        finally:
            self._capturing = False
        missed = {label: self.capture_calls[label] - before[label]
                  for label in self.targets
                  if self.capture_calls[label] - before[label] != 1}
        if missed:
            raise RuntimeError(f"Pairwise family capture count mismatch: {missed}")


def _targets(stack, families):
    families = tuple(sorted(set(families)))
    if not families or not set(families).issubset(EXPECTED):
        raise ValueError("Choose C128 and/or C256")
    targets = {}
    owner = None
    for family in families:
        found = 0
        for side, group in (("encoder", {128: 2, 256: 3}[family]),
                            ("decoder", {128: 1, 256: 0}[family])):
            for index in range(len(getattr(stack.model, side)[group])):
                block, fused = pairwise._select(stack, family, side, index)
                if owner is None:
                    owner = fused
                elif owner is not fused:
                    raise RuntimeError("Selected MLP blocks have different fused owners")
                sy, sx = block.window_shift
                h, w = pairwise._BASE[family]
                shape = (h + (8 if sy else 0), w + (8 if sx else 0), family)
                label = f"{side}-{family}-{index}"
                if any(existing is block.mlp for existing, _ in targets.values()):
                    raise RuntimeError("Duplicate selected MLP module")
                targets[label] = (block.mlp, shape)
                found += 1
        if found != EXPECTED[family]:
            raise RuntimeError(f"Expected {EXPECTED[family]} C{family} blocks, got {found}")
    return targets, owner


@contextmanager
def installed(stack, *, families=(128,)):
    """Replace only the selected family MLP calls in the active 720p session."""
    targets, fused = _targets(stack, families)
    if "apply" in fused.__dict__:
        raise RuntimeError("Active fused MLP already has an instance override")
    by_id = {id(module): (label, module, shape)
             for label, (module, shape) in targets.items()}
    counter = FamilyCounter(targets, fused)
    original = fused.apply

    def selected_apply(self, module, features):
        selected = by_id.get(id(module))
        if selected is None:
            return original(module, features)
        label, expected, shape = selected
        if (module is not expected or tuple(features.shape) != shape or
                current_arithmetic_backend() != "triton" or
                self.provider.mode != "fp16_xmx"):
            raise RuntimeError(f"Selected pairwise MLP boundary changed: {label}")
        value, _ = pairwise.forward(features, module, self)
        self.calls[str(module.channels)] = self.calls.get(str(module.channels), 0) + 1
        counter.calls[label] += 1
        if torch.xpu.is_current_stream_capturing():
            counter.capture_calls[label] += 1
        return value

    bound = MethodType(selected_apply, fused)
    fused.apply = bound
    try:
        yield counter
    finally:
        if fused.__dict__.get("apply") is not bound:
            raise RuntimeError("Pairwise family override changed during scope")
        del fused.apply
