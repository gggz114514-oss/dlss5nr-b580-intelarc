"""Scoped C512 attention-probability ablation for the isolated fastline.

The native-head adapter dispatches C512 as 16 heads. Other channel families
continue through their existing kernels. No class or global replacement survives
this context; capture a fresh model/graph for each arm.
"""
from contextlib import contextmanager

import torch

import native_half_head_layout_v1 as layout
from quantization_dataflow_v1 import CONTRACTS
import c512_probability_unround_kernel_v1 as kernels


KERNEL = "c512_probability_unround_kernel_v1._kernel"
CONTRACT = (("OUT",), ())


@contextmanager
def installed(*, unround: bool):
    if type(unround) is not bool:
        raise ValueError("unround must be fixed before graph capture")
    if KERNEL in CONTRACTS:
        raise RuntimeError("C512 probability scope already active")
    original = layout.native_heads
    calls = {"c512": 0, "other": 0, "candidate_hashes": set()}

    def forward(query, key, value, bias):
        if query.shape[0] != 16:
            calls["other"] += 1
            return original(query, key, value, bias)
        calls["c512"] += 1
        if not unround:
            return original(query, key, value, bias)
        out, kernel = kernels.forward(query, key, value, bias,
                                      round_weights=False)
        calls["candidate_hashes"].add(kernel.hash)
        return out, kernel

    layout.native_heads = forward
    CONTRACTS[KERNEL] = CONTRACT
    try:
        yield calls
    finally:
        if layout.native_heads is not forward:
            raise RuntimeError("C512 probability scope was replaced")
        layout.native_heads = original
        if CONTRACTS.pop(KERNEL) != CONTRACT:
            raise RuntimeError("C512 probability contract changed")


class Probability720Counter:
    def __init__(self, targets):
        self.targets = targets
        self.calls = dict.fromkeys(targets, 0)
        self.capture_calls = dict.fromkeys(targets, 0)
        self.resources = {}
        self.kernels_by_site = {}
        self._compiled = {}

    def preflight(self):
        """Compile the four real window grids before any model capture."""
        from spill_preflight_v1 import select

        for label, (attention, shape, _) in self.targets.items():
            wph = shape[1] * shape[2]
            if wph not in self._compiled:
                q = torch.empty(shape, dtype=torch.float16, device=attention.bias.device)
                k, v, out = torch.empty_like(q), torch.empty_like(q), torch.empty_like(q)
                params = (q, k, v, attention.bias, out, wph, 32, False)
                _, kernel, report = select(
                    kernels._kernel, [(32,)], lambda _: params,
                    lambda _: (2, q.numel() // 2048),
                    num_warps=4, num_stages=1, enable_fp_fusion=False)
                self._compiled[wph] = kernel
                self.resources[str(wph)] = {
                    "kernel_hash": kernel.hash, "round_weights": False,
                    "shape": list(shape), "resource_gate": report,
                }
            self.kernels_by_site[label] = self._compiled[wph].hash
        return self.resources


@contextmanager
def installed_720(session):
    """Own all sixteen C512 biases and both active native-head call sites.

    The legacy installed() remains for historical experiments. The decoder
    owns a separately imported native_heads alias; patching only layout's
    alias leaves half the active blocks on the old probability rounding.
    """
    import c512_window_projection_game_v1 as decoder_layout
    from nr_backend.execution import current_arithmetic_backend
    from nr_backend.unround_policy import ENABLED, FAMILIES

    stack = session._stack
    if (stack.provider.mode != "fp16_xmx" or ENABLED != FAMILIES or
            stack.graph.entries or KERNEL in CONTRACTS):
        raise RuntimeError("C512 probability candidate requires a fresh unrounded graph")
    targets = {}
    for side, blocks in (("encoder", stack.model.encoder512),
                         ("decoder", stack.model.decoder512)):
        if len(blocks) != 8:
            raise RuntimeError("Expected eight encoder and decoder C512 blocks")
        for index, block in enumerate(blocks):
            sy, sx = block.window_shift
            if sy not in (0, 4) or sx not in (0, 4):
                raise RuntimeError("Unexpected C512 window shift")
            shape = (16, (24 + (8 if sy else 0)) // 8,
                     (40 + (8 if sx else 0)) // 8, 64, 32)
            targets[f"{side}512.{index}"] = (block.attention, shape, side)
    if len({id(module.bias) for module, _, _ in targets.values()}) != 16:
        raise RuntimeError("Expected sixteen distinct owned C512 biases")
    by_bias = {id(module.bias): (name, module, shape, side)
               for name, (module, shape, side) in targets.items()}
    counter = Probability720Counter(targets)
    original_encoder = layout.native_heads
    original_decoder = decoder_layout.native_heads

    def run_owned(side, original, query, key, value, bias):
        target = by_bias.get(id(bias))
        if target is None:
            return original(query, key, value, bias)
        name, module, shape, expected_side = target
        if (bias is not module.bias or side != expected_side or
                tuple(query.shape) != shape or current_arithmetic_backend() != "triton"):
            raise RuntimeError(f"Owned C512 probability boundary changed: {name}")
        wph = shape[1] * shape[2]
        if wph not in counter._compiled:
            raise RuntimeError("C512 probability candidate requires preflight before capture")
        out, kernel = kernels.forward(query, key, value, bias, round_weights=False)
        if kernel is not counter._compiled[wph]:
            raise RuntimeError(f"C512 probability launch missed the screened kernel: {name}")
        counter.calls[name] += 1
        if torch.xpu.is_current_stream_capturing():
            counter.capture_calls[name] += 1
        return out, kernel

    def forward(side, original, query, key, value, bias):
        try:
            return run_owned(side, original, query, key, value, bias)
        except BaseException:
            session._failed = True
            raise

    def encoder(query, key, value, bias):
        return forward("encoder", original_encoder, query, key, value, bias)

    def decoder(query, key, value, bias):
        return forward("decoder", original_decoder, query, key, value, bias)

    layout.native_heads, decoder_layout.native_heads = encoder, decoder
    CONTRACTS[KERNEL] = CONTRACT
    try:
        yield counter
    finally:
        valid = (layout.native_heads is encoder and decoder_layout.native_heads is decoder)
        layout.native_heads, decoder_layout.native_heads = original_encoder, original_decoder
        if stack.graph.entries and not getattr(session, "_closed", False):
            session._failed = True
        if CONTRACTS.pop(KERNEL) != CONTRACT or not valid:
            raise RuntimeError("C512 probability aliases or contract changed during capture")
