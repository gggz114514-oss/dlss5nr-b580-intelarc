"""Owned 720p ViT numeric candidates; CPU-only source handoff, 2026-09-30.

API matches numeric_cleanup_options_720_v1.selected_scopes():
    with installed(modes, vit_qkv_full_k=False, vit_projection_full_k=False,
                   vit_exp_zero_constant=False, vit_denominator="reference",
                   vit_norm_fma=False, vit_exp_fma=False) as counter:
        counter.preflight()     # Luna: compile/load; optional scalar setup launch
        counter.validate()      # suite calls before process/replay
        # actual720 / fused history / graph replay, reset and temporal frames
        receipt = counter.snapshot()

Call only after select() on a fresh controlled unrounded C512-library/native-K8
session, source/model 720x1280, internal canvas 768x1280. This child layers
session._installed and local owned methods only. Main owns graph signature,
options binding and modes selection/process hooks. identity is immutable and
must be included by main; no child changes graph.constants or model buffers.

All-false/reference clones authenticated active vforward/attention bytecode
only to observe the unchanged reference callees. QKV full-K uses FP16 operands,
FP32 accumulation, one half store and the original reshape. Projection reuses
the existing single-accumulator _project JIT, only eight K1024 output weights.
INT8 FFN, query scale, q boundaries, padding, numerator matrices, controls and
history commit remain with their original owners; no head/layout experiment.

Ordered-fused copies four reference 64-key half trees and ordered half totals.
FP32-reduction explicitly sums padded256 special exponent values in FP32,
rounds the total to half, then keeps correction=float(exp0)*16 -> half,
subtraction -> half, clamp and reciprocal -> half. Native norm/exp reuse the
reviewed JITs, never the old scopes. exp(0) setup uses the selected reference or
native algorithm and verifies bits 0x2d60; one immutable scalar is held by this
counter outside inference_mode. Runtime constant mode creates no scalar zero
and launches no zero exp. Constants stay strongly referenced through graph
retirement, and are never registered as model buffers.

Theoretical per body: QKV 16->8, projection 16->8, scalar exp 8->0,
row sums 32->8, hence at most 48 fewer launches in combination; native FMA
removes none. ATen merge/copy/cast changes require a real graph measurement.
Python sites separate eager/warmup from actual capture and skip replay; each
frame records fused sampler and capture/replay route with graph evidence.

Only CPU AST verification is authorized here. No candidate has been imported,
compiled or run by this task. Luna still owes reference-vs-G bytes, real-shape
binary/hash/zero-spill gates, single-axis 13-frame reset/history/finite/private
state tests, paired 50-frame timings and complete user video review. Static
source agreement is not a runtime hit, DPAS/ISA proof or performance result.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import import_module
from pathlib import Path
import sys
from threading import RLock, get_ident
from types import FunctionType, MethodType

import torch
import triton

import vit_numeric_suite_720_v1_kernels as kernels
from numeric_model_forward_720_v1 import register_forward, unregister_forward, require_forward
from nr_backend.execution import current_arithmetic_backend, record_arithmetic_dispatch
from nr_backend.unround_policy import ENABLED, FAMILIES
from quantization_dataflow_v1 import CONTRACTS


_SERIAL_GUARD_TYPE = type(RLock())
TOKENS, WIDTH, HEADS, CHANNELS, PADDED_KEYS = 240, 1024, 32, 32, 256
NORM_SHAPE, SCORE_SHAPE = (240, 32, 32), (32, 240, 256)
WEIGHT_SHAPES = {
    "qkv_weight": (1024, 3072), "query_scale": (32,),
    "projection": (1024, 1024), "attn_skip": (1024,),
    "expand": (1024, 4096), "contract": (4096, 1024), "ffn_skip": (1024,),
}
REFERENCE_SHA256 = {
    "fullsize_session_v1.py": "ff475dba5fe6f55073600906d81eddbe3df31e1d75764997fc6ad19d6ffb5cfb",
    "vit_block.py": "b7f8a992d04f3c5eceb4daf007b3153178f818ccb376fbb219aea0364469d4d5",
    "attention.py": "db67828fecec5c48e120a564163b72503b73a4fcd7efb674cadbaf671b1d8841",
    "tensor_math.py": "75835d42a26f462952f1ae3e911d54e300b163f9ce5b4392e35033b2e0667f04",
    "triton_attention_normalize.py": "9a7a91f13d485392b8b3b4045600cd15633903532266c058cda7e50c1ab439cc",
    "triton_attention_exp.py": "5ec5eb86a3d26ffb48899307c3dac2b1f8ee1603ad99946f3945f4c3db18d7dd",
    "triton_attention_weights.py": "0e75817665d55a8cafd342b5f6d81c20b6a3ac5dc0cd2592327fd1467273fa97",
    "vit_native_fma_720_v1.py": "bba9060775904ce6b5de40478054abf755f1009d6e5956946fbb8f712812fde1",
    "vit_projection_single_accum_720_v1.py": "6378b81031a254d0625a448610ad7fe171ca880e10a22b646e4266727f76ab9f",
    "fused_vit_projection_v1.py": "344e8b77ff8e0cbc3c5fe77e62e736340c96d908764afe8ebfa91bfde3bdc276",
    "fused_vit_projection_v2.py": "f617c6c8ba328fe4f511445b4456e9cc717e302594ef5b7cecf1b8c93e196392",
    "fused_vit_projection_v3.py": "e236e8c849c5eb3234865c221a0b065d25f284fbf330d739c1ca9754236ac479",
    "c512_k8_joint_scope_720_v1.py": "4621cda4df3426cd6898ace624432b148ea17f77e5374b2cd5be9b22812c2ca5",
    "graph_front_v5.py": "8887facb3d98f91f6a068206698283b2865d895cbe7cdae366b0455939f3a6ec",
    "graph_front_v6.py": "3cf46091d2a44457975a95670aa317d6205e0ec9db4c5bc5acf4e70c9af7a052",
    "spill_preflight_v1.py": "74ce04f648c09c851d7f169332bc35110586d2c6632aa904d4e351289027798d",
    "nr_game_history_fused.py": "6e4f73ce30a29ffe3366abe558dbb1ef204726a10a1facab2487f1ff7d48e1f0",
    "unround_policy.py": "87bf4ba9b9f60e8c10173dc8498d54e23e3387dee0a4485fae08db21dedda827",
}


def _closure(function):
    return dict(zip(function.__code__.co_freevars,
                    (cell.cell_contents for cell in function.__closure__ or ())))


def _source(path, *, known=False):
    path = Path(path).resolve(strict=True)
    digest = sha256(path.read_bytes()).hexdigest()
    if known and REFERENCE_SHA256.get(path.name) != digest:
        raise RuntimeError(f"Unreviewed ViT source: {path}")
    return {"path": str(path), "sha256": digest}


def _installed_chain(session, *, authenticate_source=False):
    """Follow explicit parent links, never mistake a candidate neighbour for G."""
    current, chain, seen = session._installed, [], set()
    for _ in range(64):
        if not isinstance(current, FunctionType) or id(current) in seen:
            raise RuntimeError("Unknown/cyclic installed scope; cannot authenticate vforward")
        chain.append(current)
        seen.add(id(current))
        parent = getattr(current, "__nr_numeric_original_installed__", None)
        if parent is not None:
            current = parent
            continue
        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is not None:
            current = wrapped
            continue
        cells = _closure(current)
        forward = cells.get("vforward")
        if forward is not None:
            if not isinstance(forward, FunctionType):
                raise RuntimeError("Invalid fullsize vforward")
            owned = _closure(forward)
            if (forward.__code__.co_name != "vforward" or
                    current.__code__.co_name != "installed" or
                    owned.get("stack") is not session._stack or
                    owned.get("counts") is not session._counts or
                    owned.get("vit") is not session._vit or
                    forward.__code__.co_filename != current.__code__.co_filename):
                raise RuntimeError("Active vforward is not owned by this fullsize session")
            if authenticate_source:
                _source(forward.__code__.co_filename, known=True)
            return forward, chain
        parents = {id(cells[name]): cells[name] for name in
                   ("original_scope", "original_installed", "previous_scope", "previous_installed")
                   if name in cells and isinstance(cells[name], FunctionType)}
        if len(parents) != 1:
            raise RuntimeError("Install wrapper has no unambiguous reviewed parent link")
        current = next(iter(parents.values()))
    raise RuntimeError("ViT installed chain is too deep")


def _half_xpu(value, shape, device, *, contiguous=False, dtype=torch.float16):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape) or
            value.dtype != dtype or value.device.type != "xpu" or value.device != device or
            (contiguous and not value.is_contiguous())):
        raise RuntimeError(f"ViT boundary changed: expected XPU {shape}/{dtype}")


def _stamp(value):
    return (value.data_ptr(), value._version, tuple(value.shape), tuple(value.stride()),
            value.dtype, value.device)


def _tensor_receipt(value):
    return {"shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "device": str(value.device),
            "data_ptr": value.data_ptr(), "version": value._version,
            "object_identity_checked": True}


def _clone(function, *, globals_=None, replacements=None):
    replacements = replacements or {}
    def cell(value):
        return (lambda: value).__closure__[0]
    closure = tuple(cell(replacements[name]) if name in replacements else original
                    for name, original in zip(function.__code__.co_freevars,
                                               function.__closure__ or ()))
    result = FunctionType(function.__code__, globals_ or function.__globals__,
                          function.__name__, function.__defaults__, closure or None)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


@dataclass(frozen=True)
class Mode:
    vit_qkv_full_k: bool = False
    vit_projection_full_k: bool = False
    vit_exp_zero_constant: bool = False
    vit_denominator: str = "reference"
    vit_norm_fma: bool = False
    vit_exp_fma: bool = False

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name != "vit_denominator" and type(value) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        if self.vit_denominator not in ("reference", "ordered_fused", "fp32_reduction"):
            raise ValueError("Unknown vit_denominator; no fallback is supported")

    @property
    def identity(self):
        return (__name__, tuple(asdict(self).items()))

    @property
    def sites(self):
        qkv = ("qkv_full_k",) if self.vit_qkv_full_k else ("qkv_part0", "qkv_part1")
        rows = (("denominator_" + self.vit_denominator,) if self.vit_denominator != "reference"
                else tuple(f"row_sum64_{i}" for i in range(4)))
        return qkv + ("query_norm", "key_norm", "score_exp") + rows + (
            "zero_constant" if self.vit_exp_zero_constant else "zero_exp",
            "projection_full_k" if self.vit_projection_full_k else "projection_reference")

    @property
    def launches_removed(self):
        return (8 * (self.vit_qkv_full_k + self.vit_projection_full_k +
                     self.vit_exp_zero_constant) +
                (24 if self.vit_denominator != "reference" else 0))


class _VitAliases:
    def __init__(self, vit, counter, attention):
        self._vit = vit
        self.dot, self.normalize_c32 = counter._reference_dot, counter._norm
        self.vit_attention, self.split_k_projection = attention, counter._projection

    def __getattr__(self, name):
        return getattr(self._vit, name)


class NumericSuite720Counter:
    """Strongly owned constants/binaries; receipts contain JSON values only."""

    def __init__(self, modes, session, mode):
        self.modes, self.session, self._mode = modes, session, mode
        self._frozen_identity = mode.identity
        self.stack, self.graph, self.vit = session._stack, session._stack.graph, session._vit
        self.model = self.stack.model
        self._model_forward = self.model.forward
        if (not isinstance(self._model_forward, MethodType) or
                self._model_forward.__self__ is not self.model or
                self._model_forward.__func__ is not type(self.model).forward):
            raise RuntimeError("ViT requires the actual owned model forward before frame hooks")
        self._model_forward_code = self._model_forward.__func__.__code__
        self.blocks = tuple(self.stack.model.vit)
        if (len(self.blocks) != 8 or len({id(b) for b in self.blocks}) != 8 or
                self.stack.int8_vit.modules != {id(b): i for i, b in enumerate(self.blocks)} or
                any(type(b) is not self.vit.VitBlock or "forward" in b.__dict__ for b in self.blocks)):
            raise RuntimeError("Expected eight distinct active owned ViT blocks, without another ViT scope")
        self.weights = tuple({name: getattr(block, name) for name in WEIGHT_SHAPES}
                             for block in self.blocks)
        self.weight_stamps = tuple({name: _stamp(t) for name, t in row.items()} for row in self.weights)
        device = self.weights[0]["qkv_weight"].device
        for row in self.weights:
            for name, tensor in row.items():
                _half_xpu(tensor, WEIGHT_SHAPES[name], device, contiguous=True)
        self.ffn = self.stack.int8_vit.ffn
        self.reference_vforward, _ = _installed_chain(session, authenticate_source=True)
        self.sources = {"vforward": _source(self.reference_vforward.__code__.co_filename, known=True)}
        self.modules = {name: import_module(module) for name, module in {
            "attention": "nr_backend.attention", "math": "nr_backend.tensor_math",
            "norm": "nr_backend.triton_attention_normalize", "exp": "nr_backend.triton_attention_exp",
            "rows": "nr_backend.triton_attention_weights", "native": "vit_native_fma_720_v1",
            "policy": "nr_backend.unround_policy",
            "project": "vit_projection_single_accum_720_v1", "joint": "c512_k8_joint_scope_720_v1",
            "history": "nr_game_history_fused", "screen": "spill_preflight_v1",
            "projection_owner_v1": "fused_vit_projection_v1",
            "projection_owner_v2": "fused_vit_projection_v2",
            "projection_owner_v3": "fused_vit_projection_v3",
        }.items()}
        for role, module in dict(self.modules, vit=self.vit).items():
            self.sources[role] = _source(module.__file__, known=True)
        # Known fast-source hashes and real callable identity permit E/G/D
        # snapshot mixing without depending on a candidate neighbour's path.
        self.sources["graph_capture"] = _source(self.graph._capture.__func__.__code__.co_filename, known=True)
        if (self.graph._capture.__func__.__module__ != "graph_front_v5" or
                type(self.graph).__module__ not in ("graph_front_v5", "graph_front_v6")):
            raise RuntimeError("Unknown graph frontend; warmup/capture contract is unreviewed")
        self.sources["graph_frontend"] = _source(import_module(type(self.graph).__module__).__file__, known=True)
        self.candidate_sources = {"scope": _source(__file__), "kernels": _source(kernels.__file__)}
        self._forward_module = import_module("numeric_model_forward_720_v1")
        if (Path(self._forward_module.__file__).resolve(strict=True) !=
                Path(__file__).with_name("numeric_model_forward_720_v1.py").resolve(strict=True)):
            raise RuntimeError("ViT model-forward helper must come from this candidate snapshot")
        self._forward_file = self._forward_module.__file__
        self._forward_bindings = {name: (function, function.__code__) for name, function in (
            ("register_forward", register_forward), ("unregister_forward", unregister_forward),
            ("require_forward", require_forward))}
        self.candidate_sources["forward_owner"] = _source(self._forward_file)
        self.reference_callees = {name: getattr(self.vit, name) for name in
                                  ("dot", "normalize_c32", "vit_attention",
                                   "vit_exponential", "attention_row_sum64", "sm89_f16_batched_dot", "q")}
        attention, math = self.modules["attention"], self.modules["math"]
        if (self.vit.dot is not math.sm89_f16_dot or
                self.vit.sm89_f16_batched_dot is not math.sm89_f16_batched_dot or
                self.vit.normalize_c32 is not attention.normalize_c32 or
                self.vit.attention_row_sum64 is not attention.attention_row_sum64 or
                self.vit.q is not self.modules["policy"].round_vit or
                self.vit.vit_attention.__globals__["vit_exponential"] is not self.vit.vit_exponential):
            raise RuntimeError("ViT reference aliases were replaced by an unrecognised callee")
        owner_type = self.modules["projection_owner_v3"].FusedVitProjection
        owners = [c for c in self.stack.components if type(c) is owner_type]
        if len(owners) != 1 or owners[0].provider is not self.stack.provider:
            raise RuntimeError("Expected the unique owned reference ViT projection component")
        self.projection_owner = owners[0]
        self._projection_method = self.projection_owner.apply.__func__
        self._projection_code = self._projection_method.__code__
        if (any(len({id(row[name]) for row in self.weights}) != 8
                for name in ("qkv_weight", "projection", "contract")) or
                {id(row["projection"]) for row in self.weights} &
                {id(row["contract"]) for row in self.weights}):
            raise RuntimeError("Expected eight distinct QKV/output weights and separate FFN weights")
        for row in self.weights:
            for name in ("projection", "contract"):
                if self.projection_owner.weights.get(id(row[name])) is not row[name]:
                    raise RuntimeError("ViT output/FFN projection weight ownership changed")
        self.baseline_dense, self.baseline_post = self.modules["native"]._owned_baseline(modes, session, self.stack)
        self.calls = {f"vit.{i}.{site}": 0 for i in range(8) for site in mode.sites}
        self.capture_calls, self.eager_calls = dict.fromkeys(self.calls, 0), dict.fromkeys(self.calls, 0)
        self.owner_calls, self.capture_owner_calls = dict.fromkeys(range(8), 0), dict.fromkeys(range(8), 0)
        self.resources, self.capture_gates, self.frame_routes = {}, [], []
        self.failures = []
        self.actualkernelhash, self.capture_binary_sha256 = {}, {}
        self.setup_launches = {}
        self.history_route_calls, self.frontend_route_calls = {}, {}
        self._compiled, self._specs, self._kernel_bindings = {}, {}, {}
        self._binary_payloads = {}
        self._constant, self._constant_stamp = None, None
        self._constant_owner = None
        self._constant_bits = None
        self._history_binding, self._history_source = None, None
        self._history_counter, self._history_dispatch_binding = None, None
        self._history_sources, self._history_suite = {}, None
        self._active, self._frame = None, None
        self._live, self._in_scope, self._preflight_complete = True, False, False
        self._wrapper = None
        self._contracts = {}
        self._contract_registry, self._contract_entries = CONTRACTS, {}
        self.retired_graphs, self.validation_calls = [], 0
        self.denominator_calls = {f"vit.{i}.denominator": 0 for i in range(8)}
        self.capture_denominator_calls = dict.fromkeys(self.denominator_calls, 0)
        self._callable_codes = tuple((f, f.__code__) for f in
                                     (*self.reference_callees.values(), self.reference_vforward,
                                      self.baseline_dense.__func__, self.baseline_post))
        for index, (function, _) in enumerate(self._callable_codes):
            self.sources[f"actual_callee.{index}"] = dict(
                _source(function.__code__.co_filename, known=True),
                callable=f"{function.__module__}.{function.__qualname__}")
        self.sources["actual_projection_owner"] = dict(
            _source(self._projection_method.__code__.co_filename, known=True),
            callable=f"{self._projection_method.__module__}.{self._projection_method.__qualname__}")
        self._method_bindings = {name: (getattr(self, name).__func__, getattr(self, name).__func__.__code__)
                                 for name in ("_full_qkv_forward", "_attention", "_projection",
                                              "_norm", "_exp", "_row_sum", "_reference_dot")}

    @property
    def mode(self):
        return self._mode

    @property
    def identity(self):
        return self.mode.identity

    def _guard(self):
        from replay_lifecycle_audit_rest_720_v1 import live_guard
        if live_guard(self, 'vit'):
            return
        self.session._ready()
        if (not self._live or self.identity != self._frozen_identity or
                self.modes.session is not self.session or self.session._stack is not self.stack or
                self.modes.height != 720 or tuple(self.modes.source) != (720, 1280) or
                self.modes.variant != "unrounded" or not self.modes.controlled or
                not self.modes.c512_qkv_library_720 or not self.modes.native_k8_720 or
                getattr(self.modes, "vit_head_720", False) or
                getattr(self.modes, "history_compact_720", False) or
                getattr(self.modes, "post_rgb_tail_720", False) or
                tuple(self.session.fullsize_geometry) != (720, 1280) or
                tuple(self.session.fullsize_padding["padding"]) != (768, 1280) or
                self.stack.provider.mode != "fp16_xmx" or self.graph.closed or
                self.stack.int8_vit.ffn is not self.ffn or
                tuple(self.stack.model.vit) != self.blocks or
                self.stack.int8_vit.modules != {id(b): i for i, b in enumerate(self.blocks)}):
            raise RuntimeError("ViT scope lost its frozen actual720 C512+K8 session/owners")
        if (self.stack.model is not self.model or
                type(self.model).forward is not self._model_forward.__func__ or
                self._model_forward.__func__.__code__ is not self._model_forward_code):
            raise RuntimeError("ViT model/base forward identity changed")
        if (sys.modules.get("numeric_model_forward_720_v1") is not self._forward_module or
                self._forward_module.__file__ != self._forward_file or any(
                    getattr(self._forward_module, name) is not function or function.__code__ is not code
                    for name, (function, code) in self._forward_bindings.items())):
            raise RuntimeError("Canonical ViT model-forward helper identity changed")
        require_forward(self.model.forward, self._model_forward, self.model, self.session)
        if self._wrapper is not None and self.session.__dict__.get("_vit_numeric_suite_720") is not self:
            raise RuntimeError("ViT numeric session marker changed")
        if (self.stack.provider.__dict__.get("dense") is not self.baseline_dense or
                self.modules["joint"].active_post._contiguous_cropped_head is not self.baseline_post or
                self.projection_owner.apply.__func__ is not self._projection_method or
                self._projection_method.__code__ is not self._projection_code or
                any(getattr(self.vit, name) is not value for name, value in self.reference_callees.items()) or
                any(function.__code__ is not code for function, code in self._callable_codes)):
            raise RuntimeError("Actual ViT/C512/K8 callee changed")
        for name, (function, code) in self._method_bindings.items():
            if getattr(self, name).__func__ is not function or function.__code__ is not code:
                raise RuntimeError(f"Owned numeric callee changed: {name}")
        for block, row, stamps in zip(self.blocks, self.weights, self.weight_stamps):
            for name, tensor in row.items():
                if getattr(block, name) is not tensor or _stamp(tensor) != stamps[name]:
                    raise RuntimeError(f"Owned ViT {name} identity/version/pointer changed")
                if name in ("projection", "contract") and (
                        self.projection_owner.weights.get(id(tensor)) is not tensor):
                    raise RuntimeError("ViT projection component lost an owned weight")
        for name, (jit, function, code) in self._kernel_bindings.items():
            if jit.fn is not function or function.__code__ is not code:
                raise RuntimeError(f"Candidate JIT callee changed: {name}")
        if self._constant is not None and (
                self._constant is not self._constant_owner or _stamp(self._constant) != self._constant_stamp):
            raise RuntimeError("Immutable ViT exp(0) identity/version/pointer changed")
        if any(CONTRACTS.get(name) != contract for name, contract in self._contracts.items()):
            raise RuntimeError("ViT numeric output contracts changed")

    def _validate_replay(self):
        return self.validate()

    def validate(self):
        """No mandatory argument; main invokes this before process/replay."""
        try:
            self._guard()
            reference, chain = _installed_chain(self.session)
            if reference is not self.reference_vforward or self._wrapper not in chain:
                raise RuntimeError("Layered ViT scope lost its actual fullsize parent")
            # Source manifest was authenticated at initialization/preflight.
            # Per-frame validation is metadata-only, without disk IO or readback.
            if self._history_binding is not None:
                self._require_history_binding()
            from replay_lifecycle_audit_rest_720_v1 import hot_active
            if not hot_active(self):
                for name, kernel in self._compiled.items():
                    self._require_binary(name, kernel)
            if self._in_scope:
                self._active_callees()
            self.validation_calls += 1
            return {"passed": True, "identity": self.identity,
                    "preflight_complete": self._preflight_complete}
        except BaseException as error:
            self.failures.append({"phase": "validate", "type": type(error).__name__, "message": str(error)})
            self.session._failed = True
            raise

    def _transfer_serial_thread(self, owner, *, serial_guard, previous_thread):
        """Suite-only idle handoff check; ViT has no CPU thread pin to move.

        The suite owns bridge rebinding and the whole-process RLock. Keep the
        ordinary validation, permanent contracts and installed chain intact.
        """
        try:
            module = sys.modules.get("numeric_cleanup_suite_720_v1")
            suite_type = getattr(module, "NumericCleanupCounter", None)
            transaction = getattr(suite_type, "transfer_serial_thread", None)
            caller = sys._getframe(1)
            if (module is None or type(owner) is not suite_type or
                    not isinstance(transaction, FunctionType) or
                    caller.f_code is not transaction.__code__ or
                    caller.f_globals is not module.__dict__ or
                    caller.f_locals.get("self") is not owner or
                    caller.f_locals.get("serial_guard") is not serial_guard or
                    caller.f_locals.get("previous_thread") != previous_thread):
                raise RuntimeError("ViT serial handoff requires the actual owned suite transaction")
            if type(previous_thread) is not int:
                raise RuntimeError("ViT serial handoff requires an explicit previous CPU thread ident")
            adapter = sys.modules.get("cyberpunk_nr_adapter")
            host = sys.modules.get("nr_game_pre_xess_host")
            if (adapter is None or host is None or getattr(adapter, "host", None) is not host or
                    type(serial_guard) is not _SERIAL_GUARD_TYPE or
                    serial_guard is not getattr(adapter, "_process_serial_lock", None) or
                    serial_guard is getattr(adapter, "_settings_lock", None) or
                    not serial_guard._is_owned()):
                raise RuntimeError("ViT serial handoff requires the actual process RLock held by this thread")
            if (owner.closed or not owner.options.active or not self._live or
                    getattr(self.modes, "_numeric_cleanup_owner", None) is not owner or
                    owner.children.get("vit") is not self or owner.modes is not self.modes or
                    owner.session is not self.session or owner.stack is not self.stack or
                    owner.graph is not self.graph or self.stack.graph is not self.graph or
                    self.session.__dict__.get("_vit_numeric_suite_720") is not self or
                    getattr(host, "_modes", None) is not self.modes or getattr(host, "_failed", False) or
                    (owner._game_serial_adapter is not None and
                     (owner._game_serial_adapter is not adapter or owner._game_serial_guard is not serial_guard))):
                raise RuntimeError("ViT serial handoff lost its active suite/session/graph owner")
            owner._validate_suite_owner()
            if self._frame is not None or self._active is not None or self._in_scope:
                raise RuntimeError("ViT serial handoff refused while a frame/owner/scope is active")
            if (self.session.__dict__.get("_numeric_model_forward_registry_720") or
                    any("forward" in block.__dict__ for block in self.blocks)):
                raise RuntimeError("ViT serial handoff refused while temporary model/ViT forward hooks are active")
            if torch.xpu.is_current_stream_capturing():
                raise RuntimeError("ViT serial handoff refused during XPU stream capture")
            current_thread = get_ident()
            bridge = getattr(host, "_bridge", None)
            if (bridge is None or getattr(host, "_thread", None) != current_thread or
                    bridge.thread != current_thread or bridge.torch is not torch or
                    bridge.device != self.weights[0]["qkv_weight"].device or
                    torch.xpu.current_stream().sycl_queue != bridge.stream.sycl_queue):
                raise RuntimeError("ViT serial handoff requires its device and the already-bound host/bridge stream")
            dataflow = sys.modules.get("quantization_dataflow_v1")
            if (getattr(dataflow, "CONTRACTS", None) is not self._contract_registry or
                    CONTRACTS is not self._contract_registry or
                    self._contract_entries.keys() != self._contracts.keys() or
                    any(CONTRACTS.get(name) is not entry for name, entry in self._contract_entries.items())):
                raise RuntimeError("ViT serial handoff permanent contract registry/entry identity changed")
            self._validate_replay()
            return current_thread
        except BaseException as error:
            self.failures.append({"phase": "serial_thread_handoff", "type": type(error).__name__,
                                  "message": str(error)})
            self.session._failed = True
            raise

    def _active_callees(self, *, require_arithmetic=False):
        if (self.vit.VitBlock.forward is not self.reference_vforward or
                getattr(self.vit.split_k_projection, "__self__", None) is not self.projection_owner or
                self.vit.split_k_projection != self.projection_owner.apply):
            raise RuntimeError("Active fullsize ViT/FP16/unrounded callee changed")
        if require_arithmetic and (current_arithmetic_backend() != "triton" or ENABLED != FAMILIES):
            raise RuntimeError("Active ViT arithmetic must remain Triton/unrounded")

    def _bind_history(self):
        """Bind the actual history child after all suite children are installed."""
        fused = self.modules["history"].warp_history_fused
        if not isinstance(fused, FunctionType):
            raise RuntimeError("Unknown actual fused sampler at preflight")
        self._history_binding = (fused, fused.__code__)
        counter = getattr(self.session, "_history_numeric_suite_720", None)
        function = fused
        if counter is not None:
            module = import_module("history_numeric_suite_720_v1")
            suite = getattr(self.modes, "_numeric_cleanup_owner", None)
            if (type(counter) is not module.HistoryNumericCounter720 or
                    counter.session is not self.session or counter.model is not self.model or
                    counter.modes is not self.modes or counter.reference_game is not fused or
                    not counter.active or counter.retired or suite is None or
                    suite.session is not self.session or
                    not any(child is counter for child in suite.children.values()) or
                    not any(child is self for child in suite.children.values())):
                raise RuntimeError("History dispatcher is not this suite's owned session/model child")
            function = type(counter).__call__
            if (not isinstance(function, FunctionType) or
                    counter.sources.get("candidate") is not module or
                    Path(function.__code__.co_filename).resolve(strict=True) !=
                    Path(module.__file__).resolve(strict=True)):
                raise RuntimeError("Actual history dispatcher is outside its imported source")
            for role, source_module in counter.sources.items():
                source = _source(source_module.__file__)
                if (sys.modules.get(source_module.__name__) is not source_module or
                        counter.source_files.get(role) != source_module.__file__ or
                        counter.source_manifest.get(role) != source):
                    raise RuntimeError(f"History child source changed before binding: {role}")
                self._history_sources[role] = (source_module, source_module.__name__,
                                               source_module.__file__, source)
                self.sources[f"history_child.{role}"] = dict(source)
            self._history_counter, self._history_suite = counter, suite
            self._history_dispatch_binding = (type(counter), function, function.__code__)
        self._history_source = dict(_source(function.__code__.co_filename),
                                    callable=f"{function.__module__}.{function.__qualname__}")
        self.sources["actual_history_dispatch"] = self._history_source
        self._require_history_binding()

    def _require_history_binding(self):
        from replay_lifecycle_audit_rest_720_v1 import history_binding
        if history_binding(self):
            return
        """Metadata only; the temporary temporal alias is checked at consumption."""
        if self._history_binding is None:
            raise RuntimeError("Actual history dispatcher needs preflight")
        fused, fused_code = self._history_binding
        counter = self._history_counter
        if (self.modules["history"].warp_history_fused is not fused or
                fused.__code__ is not fused_code or
                getattr(self.session, "_history_numeric_suite_720", None) is not counter):
            raise RuntimeError("Frozen fused/history child binding changed")
        if counter is None:
            return
        counter_type, function, code = self._history_dispatch_binding
        suite = self._history_suite
        if (type(counter) is not counter_type or counter_type.__call__ is not function or
                function.__code__ is not code or counter.session is not self.session or
                counter.model is not self.model or counter.modes is not self.modes or
                counter.reference_game is not fused or not counter.active or counter.retired or
                getattr(self.modes, "_numeric_cleanup_owner", None) is not suite or
                suite.session is not self.session or
                not any(child is counter for child in suite.children.values()) or
                not any(child is self for child in suite.children.values())):
            raise RuntimeError("Owned history dispatcher/session/model/code changed")
        for role, (module, name, filename, source) in self._history_sources.items():
            if (sys.modules.get(name) is not module or module.__file__ != filename or
                    counter.sources.get(role) is not module or
                    counter.source_files.get(role) != filename or
                    counter.source_manifest.get(role) != source):
                raise RuntimeError(f"Bound history child source identity changed: {role}")
        candidate = self._history_sources["candidate"][0]
        if candidate.HistoryNumericCounter720 is not counter_type:
            raise RuntimeError("Imported history dispatcher type changed")

    def _history_callee(self):
        import nr_backend.temporal as temporal
        self._require_history_binding()
        current = self._history_counter if self._history_counter is not None else self._history_binding[0]
        if temporal.warp_history_normalized is not current:
            raise RuntimeError("ViT frame sampler differs from its actual preflighted history binding")
        return self._history_source

    @staticmethod
    def _binary(kernel):
        """Require loaded machine-binary bytes; textual IR is not a binary gate."""
        if (type(getattr(kernel, "n_spills", None)) is not int or kernel.n_spills != 0 or
                not isinstance(getattr(kernel, "hash", None), str) or not kernel.hash):
            raise RuntimeError("Missing actual kernel hash or nonzero/unknown spills")
        value = getattr(kernel, "kernel", None)
        if type(value) is not bytes or not value:
            raise RuntimeError("Actual loaded binary bytes unavailable; no IR/hash fallback")
        assembly = getattr(kernel, "asm", {})
        name = getattr(kernel.metadata, "binary_ext", None)
        if name is None:
            formats = [format_name for format_name in ("spv", "zebin")
                       if assembly.get(format_name) is value]
            if len(formats) != 1:
                raise RuntimeError("Actual executable SPIR-V/zebin format is missing/ambiguous")
            name = formats[0]
        if name not in ("spv", "zebin") or (name in assembly and assembly[name] is not value):
            raise RuntimeError("Unreviewed/mismatched actual Intel binary format")
        return {"actualkernelhash": kernel.hash, "binary_format": name,
                "actualbinary_sha256": sha256(value).hexdigest(),
                "actualbinary_bytes": len(value), "spills": kernel.n_spills,
                "registers": kernel.n_regs, "shared_bytes": kernel.metadata.shared}

    def _screen(self, name, jit, args, grid, options):
        function = jit.fn
        source = _source(function.__code__.co_filename,
                         known=Path(function.__code__.co_filename).name in REFERENCE_SHA256)
        if (source["sha256"] not in {item["sha256"] for item in self.sources.values()} and
                source != self.candidate_sources["kernels"]):
            raise RuntimeError(f"Unrecognised actual candidate JIT source: {name}")
        _, kernel, gate = self.modules["screen"].select(
            jit, [()], lambda _: args, lambda _: grid, **options)
        resource = self._binary(kernel)
        self._kernel_bindings[name] = (jit, function, function.__code__)
        self._compiled[name], self._specs[name] = kernel, (jit, grid, options)
        self._binary_payloads[name] = kernel.kernel
        self.resources[name] = dict(
            resource, source_kernel=f"{function.__module__}.{function.__name__}",
            source=source,
            tensor_arguments=[_tensor_receipt(t) for t in args if isinstance(t, torch.Tensor)],
            constexpr=[value for value in args if not isinstance(value, torch.Tensor)],
            grid=list(grid), launch_options=dict(options), resource_gate=gate,
            setup_only=(name == "zero_setup"), launched_during_preflight=False)

    def _require_binary(self, name, kernel):
        if (kernel is not self._compiled[name] or
                kernel.kernel is not self._binary_payloads[name] or
                kernel.hash != self.resources[name]["actualkernelhash"] or
                type(kernel.n_spills) is not int or kernel.n_spills != 0):
            raise RuntimeError(f"Dispatch differs from the screened actual zero-spill binary: {name}")

    def _warm_kernel(self, name, args):
        if name not in self._compiled:
            raise RuntimeError(f"Real-shape ViT preflight is required: {name}")
        jit, grid, options = self._specs[name]
        kernel = jit.warmup(*args, grid=grid, **options)
        kernel._init_handles()
        self._require_binary(name, kernel)
        return kernel

    def _launch(self, name, args):
        self._warm_kernel(name, args)
        jit, grid, options = self._specs[name]
        kernel = jit[grid](*args, **options)
        self._require_binary(name, kernel)
        return kernel

    def preflight(self):
        """Luna only: real-shape compile/load; optional one-time exp(0) launch.

        No model forward. Creates external constants outside inference mode.
        Default/reference has no new kernels and does not initialize a GPU
        constant. Unknown spill/binary metadata is an error, never fallback.
        """
        try:
            self.validate()
            if (self.graph.entries or self.graph.replays or self._in_scope or
                    any(self.owner_calls.values()) or torch.xpu.is_current_stream_capturing()):
                raise RuntimeError("Preflight must precede all body warmup/capture")
            if self._preflight_complete:
                return self.snapshot()["resources"]
            for source in (*self.sources.values(), *self.candidate_sources.values()):
                if _source(source["path"]) != {"path": source["path"], "sha256": source["sha256"]}:
                    raise RuntimeError(f"ViT source changed before preflight: {source['path']}")
            self._bind_history()
            device = self.weights[0]["qkv_weight"].device
            matrix_options = dict(num_warps=4, num_stages=1, enable_fp_fusion=False)
            with torch.inference_mode(False):
                if self.mode.vit_qkv_full_k:
                    x = torch.empty((240, 1024), dtype=torch.float16, device=device)
                    out = torch.empty((240, 3072), dtype=torch.float16, device=device)
                    self._screen("qkv", kernels._qkv_full_k,
                                 (x, self.weights[0]["qkv_weight"], out, 16, 32),
                                 (15, 96), matrix_options)
                if self.mode.vit_projection_full_k:
                    x = torch.empty((240, 1024), dtype=torch.float16, device=device)
                    initial, out = torch.empty_like(x), torch.empty_like(x)
                    self._screen("projection", self.modules["project"]._project,
                                 (x, self.weights[0]["projection"], initial, out, 16, 32),
                                 (15, 32), matrix_options)
                if self.mode.vit_norm_fma:
                    x = torch.empty(NORM_SHAPE, dtype=torch.float16, device=device)
                    self._screen("norm", self.modules["native"]._normalize,
                                 (x, torch.empty_like(x), 7680, 16), (480,),
                                 dict(num_warps=4, enable_fp_fusion=False))
                if self.mode.vit_exp_fma:
                    x = torch.empty(SCORE_SHAPE, dtype=torch.float16, device=device)
                    self._screen("score_exp", self.modules["native"]._exp_vit,
                                 (x, torch.empty_like(x), 1966080, 512), (3840,),
                                 dict(enable_fp_fusion=False))
                if self.mode.vit_denominator != "reference":
                    x = torch.empty(SCORE_SHAPE, dtype=torch.float16, device=device)
                    out = torch.empty((32, 240, 1), dtype=torch.float16, device=device)
                    self._screen("denominator", kernels._denominator,
                                 (x, out, 7680, 16, self.mode.vit_denominator == "ordered_fused"),
                                 (480,), dict(num_warps=4, enable_fp_fusion=False))
                if self.mode.vit_exp_zero_constant or self.mode.vit_exp_fma:
                    x = torch.empty((), dtype=torch.float16, device=device)
                    out = torch.empty_like(x)
                    native = self.mode.vit_exp_fma
                    args = (x, out, 1, 512) if native else (x, out, 1, True, 512)
                    name = "zero_setup" if self.mode.vit_exp_zero_constant else "zero_exp"
                    self._screen(name, self.modules["native"]._exp_vit if native else self.modules["exp"]._kernel,
                                 args, (1,), dict(enable_fp_fusion=False))
                    if self.mode.vit_exp_zero_constant:
                        source = torch.zeros((), dtype=torch.float32, device=device).half().contiguous()
                        constant = torch.empty((), dtype=torch.float16, device=device)
                        params = ((source, constant, 1, 512) if native else
                                  (source, constant, 1, True, 512))
                        kernel = self._launch(name, params)
                        # _launch already checked this returned, loaded binary.
                        # This is one setup launch, not eight body consumption sites.
                        receipt = dict(
                            self._binary(kernel), resource=name, stage="setup", setup_only=True,
                            launched_during_preflight=True, constant_bits_verified=False,
                            source_kernel=self.resources[name]["source_kernel"],
                            tensor_arguments=[_tensor_receipt(source), _tensor_receipt(constant)],
                            constexpr=list(params[2:]), grid=list(self._specs[name][1]),
                            launch_options=dict(self._specs[name][2]))
                        self.setup_launches["setup.zero_setup"] = receipt
                        self.actualkernelhash["setup.zero_setup"] = kernel.hash
                        self.resources[name]["launched_during_preflight"] = True
                        bits = int(constant.view(torch.int16).item()) & 0xffff
                        receipt["actual_output_bits"] = bits
                        self.resources[name]["actual_output_bits"] = bits
                        if bits != 0x2d60:
                            raise RuntimeError(f"Selected exp(0) algorithm produced unreviewed bits: {bits:#06x}")
                        self._constant = self._constant_owner = constant
                        self._constant_stamp, self._constant_bits = _stamp(constant), bits
                        receipt["constant_bits_verified"] = True
            self._preflight_complete = True
            return self.snapshot()["resources"]
        except BaseException as error:
            self.failures.append({"phase": "preflight", "type": type(error).__name__, "message": str(error)})
            self.session._failed = True
            raise

    def _hit(self, site, kernel_name=None):
        if self._active is None:
            raise RuntimeError("ViT site outside its owned forward")
        index, progress, capturing = self._active
        if len(progress) >= len(self.mode.sites) or self.mode.sites[len(progress)] != site:
            raise RuntimeError(f"Owned ViT {index} site/order changed: {site}, after {progress}")
        if bool(torch.xpu.is_current_stream_capturing()) != capturing:
            raise RuntimeError("ViT capture state changed within the owner")
        label = f"vit.{index}.{site}"
        self.calls[label] += 1
        (self.capture_calls if capturing else self.eager_calls)[label] += 1
        if kernel_name is not None:
            resource = self.resources[kernel_name]
            self.actualkernelhash[label] = resource["actualkernelhash"]
            if capturing:
                self.capture_binary_sha256[label] = resource["actualbinary_sha256"]
        progress.append(site)

    def _logical_denominator(self):
        index, _, capturing = self._active
        label = f"vit.{index}.denominator"
        self.denominator_calls[label] += 1
        if capturing:
            self.capture_denominator_calls[label] += 1

    def _reference_dot(self, features, weight, *, chunk_k, **kwargs):
        """Reference bytecode passes the original strided K512 views unchanged."""
        index, progress, _ = self._active
        part = len(progress)
        if part not in (0, 1) or chunk_k != 16 or kwargs:
            raise RuntimeError("Unknown QKV reference dot")
        base = self.weights[index]["qkv_weight"]
        _half_xpu(features, (240, 512), base.device)
        _half_xpu(weight, (512, 3072), base.device, contiguous=True)
        if (tuple(features.stride()) != (1024, 1) or
                weight.data_ptr() != base.data_ptr() + part * 512 * 3072 * base.element_size()):
            raise RuntimeError("QKV reference no longer uses the owned K512 views")
        output = self.reference_callees["dot"](features, weight, chunk_k=16)
        _half_xpu(output, (240, 3072), base.device, contiguous=True)
        self._hit(f"qkv_part{part}")
        return output

    def _norm(self, value):
        index, progress, _ = self._active
        site = self.mode.sites[len(progress)]
        if site not in ("query_norm", "key_norm"):
            raise RuntimeError("Unexpected ViT norm site")
        _half_xpu(value, NORM_SHAPE, self.weights[index]["qkv_weight"].device)
        if tuple(value.stride()) != (3072, 96, 1):
            raise RuntimeError("Original Q/K stride changed")
        if self.mode.vit_norm_fma:
            source = value.half().contiguous()
            output = torch.empty_like(source)
            self._launch("norm", (source, output, 7680, 16))
            record_arithmetic_dispatch("attention_normalize_c32")
        else:
            output = self.reference_callees["normalize_c32"](value)
        _half_xpu(output, NORM_SHAPE, value.device, contiguous=True)
        self._hit(site, "norm" if self.mode.vit_norm_fma else None)
        return output

    def _exp(self, value):
        index, progress, _ = self._active
        site = self.mode.sites[len(progress)]
        if site not in ("score_exp", "zero_exp"):
            raise RuntimeError("Unexpected ViT special-exp site")
        shape = SCORE_SHAPE if site == "score_exp" else ()
        _half_xpu(value, shape, self.weights[index]["qkv_weight"].device,
                  dtype=torch.float16 if site == "score_exp" else torch.float32)
        if self.mode.vit_exp_fma:
            source = value.half().contiguous()
            output = torch.empty_like(source)
            count = 1966080 if site == "score_exp" else 1
            self._launch(site, (source, output, count, 512))
            record_arithmetic_dispatch("attention_exp_vit")
        else:
            output = self.reference_callees["vit_exponential"](value)
        _half_xpu(output, shape, value.device, contiguous=True)
        self._hit(site, site if self.mode.vit_exp_fma else None)
        return output

    def _row_sum(self, value):
        index, progress, _ = self._active
        site = self.mode.sites[len(progress)]
        if not site.startswith("row_sum64_"):
            raise RuntimeError("Unexpected ViT reference row_sum64")
        _half_xpu(value, (32, 240, 64), self.weights[index]["qkv_weight"].device)
        if tuple(value.stride()) != (61440, 256, 1):
            raise RuntimeError("Original 64-key row slice changed")
        output = self.reference_callees["attention_row_sum64"](value)
        _half_xpu(output, (32, 240, 1), value.device, contiguous=True)
        self._hit(site)
        if site == "row_sum64_3":
            self._logical_denominator()
        return output

    def _attention(self, query, key, value):
        """Use reference bytecode unless scalar caching/reduction needs a branch."""
        index = self._active[0]
        device = self.weights[index]["qkv_weight"].device
        for tensor in (query, key, value):
            _half_xpu(tensor, (32, 240, 32), device)
        if self.mode.vit_denominator == "reference" and not self.mode.vit_exp_zero_constant:
            return self._reference_attention(query, key, value)
        padding = 16
        key = torch.nn.functional.pad(key, (0, 0, 0, padding))
        value = torch.nn.functional.pad(value, (0, 0, 0, padding))
        e = self._exp(self.vit.sm89_f16_batched_dot(query, key.transpose(-1, -2)))
        numerator = self.vit.sm89_f16_batched_dot(self.vit.q(e), value)
        if self.mode.vit_denominator == "reference":
            total = self._row_sum(e[..., :64])
            for start in range(64, 256, 64):
                total = (total + self._row_sum(e[..., start:start + 64])).half()
        else:
            _half_xpu(e, SCORE_SHAPE, device, contiguous=True)
            total = torch.empty((32, 240, 1), dtype=torch.float16, device=device)
            self._launch("denominator", (e, total, 7680, 16,
                                        self.mode.vit_denominator == "ordered_fused"))
            # Keep the four logical row operations distinct from one physical kernel.
            for _ in range(4):
                record_arithmetic_dispatch("attention_row_sum64")
            self._hit("denominator_" + self.mode.vit_denominator, "denominator")
            self._logical_denominator()
        if self.mode.vit_exp_zero_constant:
            if self._constant is None or _stamp(self._constant) != self._constant_stamp:
                raise RuntimeError("Selected exp(0) constant is not initialized/immutable")
            zero = self._constant
            self._hit("zero_constant")
            record_arithmetic_dispatch("vit_exp_zero_constant")
        else:
            zero = self._exp(torch.zeros((), device=e.device))
        correction = (zero.float() * padding).half()
        total = (total - correction).half()
        reciprocal = total.clamp(min=0.00006198883056640625).float().reciprocal().half()
        return self.vit.q((numerator * reciprocal).half())

    def _projection(self, features, weight, initial, parts=4):
        index = self._active[0]
        owned = self.weights[index]
        if weight is not owned["projection"] or parts != 4:
            raise RuntimeError("ViT numeric suite may only intercept its eight output projections")
        for tensor in (features, initial):
            _half_xpu(tensor, (240, 1024), weight.device, contiguous=True)
        if self.mode.vit_projection_full_k:
            output = torch.empty((240, 1024), dtype=torch.float16, device=weight.device)
            # Identical JIT/math/tile/options to the existing project() function.
            self._launch("projection", (features, weight, initial, output, 16, 32))
            for _ in range(4):
                record_arithmetic_dispatch("dense")
            self.projection_owner.calls += 1
        else:
            output = self.vit.split_k_projection(features, weight, initial, parts)
        _half_xpu(output, (240, 1024), weight.device, contiguous=True)
        self._hit("projection_full_k" if self.mode.vit_projection_full_k else "projection_reference",
                  "projection" if self.mode.vit_projection_full_k else None)
        return output

    def _full_qkv_forward(self, module, x):
        """Only the QKV expression differs from the authenticated vforward."""
        index = self.stack.int8_vit.modules[id(module)]
        x = self.vit.q(x)
        mlp = self.vit.q(self.ffn(index, x)[0])
        _half_xpu(mlp, (240, 1024), module.qkv_weight.device, contiguous=True)
        z = torch.empty((240, 3072), dtype=torch.float16, device=mlp.device)
        self._launch("qkv", (mlp, module.qkv_weight, z, 16, 32))
        for _ in range(2):
            record_arithmetic_dispatch("dense")
        self._hit("qkv_full_k", "qkv")
        z = z.reshape(240, 32, 3, 32)
        query = self.vit.q((self._norm(z[:, :, 0]) * 5.65625).half() *
                           module.query_scale[None, :, None])
        key = self.vit.q(self._norm(z[:, :, 1]))
        value = self.vit.q(z[:, :, 2])
        attended = self._attention(query.transpose(0, 1), key.transpose(0, 1),
                                   value.transpose(0, 1)).transpose(0, 1).reshape(240, 1024)
        self.session._counts["vit"] += 1
        return self.vit.q(self._projection(attended, module.projection,
                                          (mlp * module.attn_skip).half()))

    def _owned_forward(self):
        self._active_callees()
        original_attention = self.reference_callees["vit_attention"]
        self._reference_attention = _clone(
            original_attention, globals_=dict(original_attention.__globals__,
                                              vit_exponential=self._exp,
                                              attention_row_sum64=self._row_sum))
        if self.mode.vit_qkv_full_k:
            return self._full_qkv_forward
        aliases = _VitAliases(self.vit, self, self._attention)
        return _clone(self.reference_vforward, replacements={"vit": aliases})

    def run(self, index, module, value, forward):
        self._guard()
        self._active_callees(require_arithmetic=True)
        if (not self._preflight_complete or self._frame is None or
                module is not self.blocks[index] or self._active is not None):
            raise RuntimeError("ViT forward outside its preflighted owned frame")
        _half_xpu(value, (240, 1024), self.weights[index]["qkv_weight"].device, contiguous=True)
        capturing = bool(torch.xpu.is_current_stream_capturing())
        progress = []
        self._active = (index, progress, capturing)
        try:
            output = forward(module, value)
            if tuple(progress) != self.mode.sites:
                raise RuntimeError(f"ViT owner {index} missed its frozen site set: {progress}")
            _half_xpu(output, (240, 1024), value.device, contiguous=True)
            self.owner_calls[index] += 1
            if capturing:
                self.capture_owner_calls[index] += 1
            return output
        finally:
            self._active = None

    def frontend(self, original, rgb, motion, args, kwargs):
        """Observe the unchanged owned model entry, including actual motion input."""
        self._guard()
        self._active_callees(require_arithmetic=True)
        if self._frame is None or self._frame["frontend_calls"] != 0:
            raise RuntimeError("Expected exactly one controlled model entry per frame")
        device = self.weights[0]["qkv_weight"].device
        # Geometry.prepare supplies the 720 model input. MotionNR/front owns
        # the unchanged 768 internal padding; motion/history stay at 720.
        _half_xpu(rgb, (720, 1280, 3), device, dtype=torch.float32)
        _half_xpu(motion, (720, 1280, 2), device, dtype=torch.float32)
        if args or kwargs.get("progress") is not None:
            raise RuntimeError("Unsupported frontend progress/positional route")
        sampler = self._history_callee()
        self._frame.update(
            frontend_calls=1, history_sampler=sampler,
            motion_input={"shape": list(motion.shape), "dtype": str(motion.dtype),
                          "device": str(motion.device), "data_ptr": motion.data_ptr(),
                          "content": "passed unchanged; no zero/static-motion assumption"},
            reset_requested=bool(kwargs.get("reset", False)),
            model_input=[720, 1280], internal_padding=[768, 1280],
            selected={"height": self.modes.height, "source": list(self.modes.source),
                      "variant": self.modes.variant, "c512_qkv_library_720": True,
                      "native_k8_720": True, "unround_families": sorted(ENABLED)},
            effective_options=asdict(self.mode))
        # No clamping, cast, history/seed/control change at this observer.
        return original(rgb, motion, **kwargs)

    def finish_frame(self, before, owners_before, denominator_before, entries_before,
                     replays_before, baseline_before, before_implementation,
                     lifecycle_token, warm_counters):
        added = set(self.graph.entries) - entries_before
        proof = None
        if lifecycle_token is not None:
            from audit_replay_receipt_720_v1 import capture_delta
            new_count, _, proof = capture_delta(self, lifecycle_token)
            if new_count != len(added):
                raise RuntimeError("ViT capture token differs from actual new graph entries")
        if not added and warm_counters != tuple(tuple(counts.items()) for counts in (
                self.calls, self.capture_calls, self.eager_calls, self.owner_calls,
                self.capture_owner_calls, self.denominator_calls, self.capture_denominator_calls,
                self.modes.c512_library_calls, self.modes.native_k8_calls)):
            raise RuntimeError("Warm ViT graph replay changed actual Python/capture/provider counters")
        route = "capture" if added else "replay" if self.graph.replays > replays_before else "eager"
        frame = dict(self._frame or {})
        frame.update(frontend_route=route, new_entries=len(added),
                     graph_replays_before=replays_before, graph_replays_after=self.graph.replays,
                     passed=(frame.get("frontend_calls") == 1 and
                             self.graph.replays == replays_before + 1))
        last = self.graph.last_entry
        if last is not None:
            frame["body_route"] = "reset" if last.inputs.get("previous") is None else "history"
            frame["entry_replays"] = last.replays
        from replay_lifecycle_audit_rest_720_v1 import keep_frame_diagnostics
        if not frame["passed"] or keep_frame_diagnostics(self):
            self.frame_routes.append(frame)
        self.history_route_calls["fused_sampler_bound"] = (
            self.history_route_calls.get("fused_sampler_bound", 0) + frame.get("frontend_calls", 0))
        history_route = "reset_no_warp" if frame.get("body_route") == "reset" else "fused_history"
        self.history_route_calls[history_route] = self.history_route_calls.get(history_route, 0) + 1
        self.frontend_route_calls[route] = self.frontend_route_calls.get(route, 0) + 1
        if not frame["passed"]:
            raise RuntimeError("ViT numeric frame missed fused-history graph replay")
        if added:
            delta = {name: value - before[name] for name, value in self.capture_calls.items()}
            owner_delta = {i: value - owners_before[i] for i, value in self.capture_owner_calls.items()}
            denominator_delta = {name: value - denominator_before[name]
                                 for name, value in self.capture_denominator_calls.items()}
            if proof is not None:
                c512_delta, k8_delta = proof["c512_calls"], proof["k8_calls"]
                provider_proof = proof.get("body_provider_proof")
                if provider_proof is None:
                    raise RuntimeError("ViT actual capture missed its body provider proof")
            else:
                c512_before, k8_before = baseline_before
                c512_delta = {name: value - c512_before[name] for name, value in self.modes.c512_library_calls.items()}
                k8_delta = {name: value - k8_before[name] for name, value in self.modes.native_k8_calls.items()}
                from implementation_capture_roles_720_v1 import body_provider_capture_gate
                provider_proof = body_provider_capture_gate(
                    self.modes, before_implementation, c512_delta, k8_delta, builds=len(added))
            candidate_sites = {}
            for index in range(8):
                if self.mode.vit_qkv_full_k:
                    candidate_sites[f"vit.{index}.qkv_full_k"] = "qkv"
                if self.mode.vit_projection_full_k:
                    candidate_sites[f"vit.{index}.projection_full_k"] = "projection"
                if self.mode.vit_norm_fma:
                    for site in ("query_norm", "key_norm"):
                        candidate_sites[f"vit.{index}.{site}"] = "norm"
                if self.mode.vit_exp_fma:
                    candidate_sites[f"vit.{index}.score_exp"] = "score_exp"
                    if not self.mode.vit_exp_zero_constant:
                        candidate_sites[f"vit.{index}.zero_exp"] = "zero_exp"
                if self.mode.vit_denominator != "reference":
                    candidate_sites[f"vit.{index}.denominator_{self.mode.vit_denominator}"] = "denominator"
            binary_gate = all(self.capture_binary_sha256.get(label) == self.resources[name]["actualbinary_sha256"]
                              for label, name in candidate_sites.items())
            constant_gate = (not self.mode.vit_exp_zero_constant or (
                self._constant is self._constant_owner and self._constant_bits == 0x2d60 and
                _stamp(self._constant) == self._constant_stamp))
            passed = (all(value == len(added) for value in delta.values()) and
                      all(value == len(added) for value in owner_delta.values()) and
                      all(value == len(added) for value in denominator_delta.values()) and
                      len(c512_delta) == 16 and set(k8_delta) == {"pre", "post"} and
                      provider_proof["passed"] and
                      binary_gate and constant_gate)
            gate = {"new_entries": len(added), "capture_calls": delta, "owners": owner_delta,
                    "denominator_logical_calls": denominator_delta, "c512_calls_with_two_warmups": c512_delta,
                    "k8_calls_with_two_warmups": k8_delta, "passed": passed,
                    "body_provider_proof": provider_proof,
                    "actual_binary_gate": binary_gate, "constant_gate": constant_gate,
                    "runtime_scalar_exp_calls": 0 if self.mode.vit_exp_zero_constant else 8 * len(added),
                    "constant_consumptions": 8 * len(added) if self.mode.vit_exp_zero_constant else 0,
                    "capture_actualbinary_sha256": dict(self.capture_binary_sha256)}
            self.capture_gates.append(gate)
            if not passed:
                raise RuntimeError("Captured ViT/C512/K8 graph missed the frozen actual site set")

    def snapshot(self):
        """Detached JSON receipts, including post-exit/failed-session diagnostics."""
        import copy
        selected = getattr(self.modes, "numeric_cleanup_720", None)
        requested = selected.to_dict() if hasattr(selected, "to_dict") else None
        constant = None if self._constant is None else dict(
            _tensor_receipt(self._constant), bits=self._constant_bits,
            exp_mode="native_half_fma" if self.mode.vit_exp_fma else "reference",
            ownership="counter strong reference, not model buffer",
            immutable_stamp_matches=(_stamp(self._constant) == self._constant_stamp))
        requested_matches = (None if requested is None else
                             all(requested.get(name) == value for name, value in asdict(self.mode).items()))
        result = {
            "options": asdict(self.mode), "identity": self.identity,
            "selected_numeric_options": requested, "matches_selected_vit_options": requested_matches,
            "sources": self.sources, "candidate_sources": self.candidate_sources,
            "targets": [f"vit.{i}" for i in range(8)],
            "weights": {f"vit.{i}": {name: _tensor_receipt(value) for name, value in row.items()}
                        for i, row in enumerate(self.weights)},
            "expected_sites": {name: 1 for name in self.calls},
            "calls": self.calls, "capture_calls": self.capture_calls, "eager_calls": self.eager_calls,
            "owner_calls": self.owner_calls, "capture_owner_calls": self.capture_owner_calls,
            "denominator_logical_calls": self.denominator_calls,
            "capture_denominator_logical_calls": self.capture_denominator_calls,
            "resources": self.resources, "preflight_resources": self.resources,
            "preflight_complete": self._preflight_complete, "actualkernelhash": self.actualkernelhash,
            "setup_launches": self.setup_launches,
            "capture_actualbinary_sha256": self.capture_binary_sha256,
            "constant": constant, "capture_gates": self.capture_gates,
            "history_route_calls": self.history_route_calls, "frontend_route_calls": self.frontend_route_calls,
            "frame_route_calls": self.frontend_route_calls, "frame_routes": self.frame_routes,
            "validation_calls": self.validation_calls,
            "failures": self.failures,
            "contracts": self._contracts, "graph_replays": self.graph.replays,
            "retired_graphs": self.retired_graphs, "live": self._live,
            "theoretical_launches_removed": self.mode.launches_removed,
            "denominator_contract": ("reference_ordered_half" if self.mode.vit_denominator == "reference"
                                     else "ordered_half_padded256" if self.mode.vit_denominator == "ordered_fused"
                                     else "fp32_padded256_sum_then_half_before_original_correction"),
            "logical_arithmetic": "reference work receipts; not physical launches",
            "measurement_overhead": {
                "per_frame": "metadata-only validate, owned frontend observer, route/graph checks and Python receipts",
                "eager_or_capture_only": "eight owned forward/site checks and matching warmup-cache identity before candidate launches",
                "initialization_only": "source manifest/file SHA, binary SHA/resource screening and optional scalar exp(0) readback",
                "timing_policy": "include all actual timed-path validation overhead; do not subtract it to report gains",
                "overhead_measured": False,
                "disk_io_in_validate_or_dispatch": False,
                "extra_device_readback_in_validate_or_dispatch": False,
            },
            "replay_evidence": "graph/entry replay counters; Python sites do not advance on replay",
            "history_evidence": "actual sampler binding + unchanged frontend + graph previous-input route",
            "validation": "CPU AST only; actual hit/binary/zero-spill/bytes/quality/timing/video pending Luna",
        }
        return copy.deepcopy(result)

    report = snapshot


def _register_contracts(counter):
    selected = []
    mode, modules = counter.mode, counter.modules
    if mode.vit_qkv_full_k:
        selected.append((kernels._qkv_full_k, "OUT"))
    if mode.vit_projection_full_k:
        selected.append((modules["project"]._project, "OUT"))
    if mode.vit_denominator != "reference":
        selected.append((kernels._denominator, "OUT"))
    if mode.vit_norm_fma:
        selected.append((modules["native"]._normalize, "Y"))
    if mode.vit_exp_fma:
        selected.append((modules["native"]._exp_vit, "Y"))
    if mode.vit_exp_zero_constant and not mode.vit_exp_fma:
        selected.append((modules["exp"]._kernel, "Y"))
    owned = []
    for jit, output in selected:
        name = f"{jit.fn.__module__}.{jit.fn.__name__}"
        contract = ((output,), ())
        if name in CONTRACTS and CONTRACTS[name] != contract:
            raise RuntimeError(f"Conflicting ViT numeric kernel contract: {name}")
        counter._contracts[name] = contract
    # Validate the entire set before a mutation, so partial setup cannot leak.
    for name, contract in counter._contracts.items():
        if name not in CONTRACTS:
            CONTRACTS[name] = contract
            owned.append(name)
    # Equal pre-existing contracts are legal; freeze the actual live entries.
    counter._contract_entries = {name: CONTRACTS[name] for name in counter._contracts}
    return owned


@contextmanager
def installed(modes, *, vit_qkv_full_k=False, vit_projection_full_k=False,
              vit_exp_zero_constant=False, vit_denominator="reference",
              vit_norm_fma=False, vit_exp_fma=False):
    """Layer a single owned eight-block scope; parent suite owns outer hooks."""
    session = getattr(modes, "session", None)
    if session is None:
        raise RuntimeError("Select a fresh actual720 controlled C512+K8 session first")
    try:
        mode = Mode(vit_qkv_full_k, vit_projection_full_k, vit_exp_zero_constant,
                    vit_denominator, vit_norm_fma, vit_exp_fma)
        session._ready()
        graph = session._stack.graph
        if (graph.entries or graph.replays or graph.closed or
                getattr(modes, "_session_frames", None) != 0 or
                "_vit_numeric_suite_720" in session.__dict__ or
                "_vit_native_fma_720" in session.__dict__):
            raise RuntimeError("ViT suite needs a fresh session; do not combine old ViT/FMA/projection scopes")
        counter = NumericSuite720Counter(modes, session, mode)
        previous_scope = session._installed
        # Check configuration without assuming anything about outer modes/graph hooks.
        counter._guard()
    except BaseException:
        session._failed = True
        raise

    @contextmanager
    def wrapped_scope():
        model = counter.stack.model
        try:
            counter._validate_replay()
            if not counter._preflight_complete:
                raise RuntimeError("ViT numeric suite requires preflight before any frame")
            from audit_replay_receipt_720_v1 import capture_token
            lifecycle_token = capture_token(counter)
            before = dict(counter.capture_calls)
            owners_before = dict(counter.capture_owner_calls)
            denominator_before = dict(counter.capture_denominator_calls)
            entries_before, replays_before = set(graph.entries), graph.replays
            baseline_before = ((dict(modes.c512_library_calls), dict(modes.native_k8_calls))
                               if lifecycle_token is None else None)
            before_implementation = None
            if lifecycle_token is None and modes.implementation_calls_720:
                from implementation_capture_roles_720_v1 import capture_before
                before_implementation = capture_before(modes)
            warm_counters = tuple(tuple(counts.items()) for counts in (
                counter.calls, counter.capture_calls, counter.eager_calls, counter.owner_calls,
                counter.capture_owner_calls, counter.denominator_calls, counter.capture_denominator_calls,
                modes.c512_library_calls, modes.native_k8_calls))
            # All class/alias replacement dependencies are installed by the real parent.
            with previous_scope():
                counter._in_scope = True
                counter._frame = {"frontend_calls": 0, "index": len(counter.frame_routes)}
                forward = counter._owned_forward()
                model_had_forward = "forward" in model.__dict__
                model_saved_forward = model.__dict__.get("forward")
                original_model_forward = model.forward
                require_forward(original_model_forward, counter._model_forward, model, session)

                def model_forward(owner, rgb, motion, *args, **kwargs):
                    try:
                        if owner is not model:
                            raise RuntimeError("ViT frontend lost its model owner")
                        return counter.frontend(original_model_forward, rgb, motion, args, kwargs)
                    except BaseException:
                        session._failed = True
                        raise

                model_replacement = MethodType(model_forward, model)
                replacements = {}
                counter._lifecycle_block_hooks = replacements
                registered = model_set = False
                try:
                    register_forward(model_replacement, original_model_forward, counter)
                    registered = True
                    model.forward = model_replacement
                    model_set = True
                    for index, block in enumerate(counter.blocks):
                        if "forward" in block.__dict__:
                            raise RuntimeError("Another scope already owns a ViT instance forward")

                        def owned_forward(owner, value, *, _index=index):
                            try:
                                return counter.run(_index, owner, value, forward)
                            except BaseException:
                                session._failed = True
                                raise

                        replacement = MethodType(owned_forward, block)
                        block.forward = replacement
                        replacements[index] = replacement
                    yield
                    counter.finish_frame(before, owners_before, denominator_before, entries_before,
                                         replays_before, baseline_before, before_implementation,
                                         lifecycle_token, warm_counters)
                finally:
                    valid = ((not model_set or model.__dict__.get("forward") is model_replacement) and
                             all(counter.blocks[i].__dict__.get("forward") is value
                                 for i, value in replacements.items()))
                    try:
                        for index in reversed(replacements):
                            counter.blocks[index].__dict__.pop("forward", None)
                        if model_set:
                            if model_had_forward:
                                model.forward = model_saved_forward
                            else:
                                model.__dict__.pop("forward", None)
                    finally:
                        if registered:
                            unregister_forward(model_replacement, counter)
                    if not valid:
                        raise RuntimeError("ViT owned forwards changed during their frame")
        except BaseException as error:
            counter.failures.append({"phase": "frame_scope", "type": type(error).__name__, "message": str(error)})
            if counter._frame is not None and len(counter.frame_routes) == counter._frame["index"]:
                counter.frame_routes.append(dict(counter._frame, frontend_route="failed", passed=False,
                                                 error=str(error)))
            session._failed = True
            raise
        finally:
            counter._lifecycle_block_hooks = {}
            counter._active, counter._frame, counter._in_scope = None, None, False

    wrapped_scope.__nr_numeric_original_installed__ = previous_scope
    # Keep contextmanager's __wrapped__ (its generator); the explicit link above
    # identifies the actual parent even when this child is inside other children.
    counter._wrapper = wrapped_scope
    owned_contracts = []
    installed_state = False
    try:
        owned_contracts = _register_contracts(counter)
        session._vit_numeric_suite_720 = counter
        session._installed = wrapped_scope
        installed_state = True
        yield counter
    except BaseException:
        session._failed = True
        raise
    finally:
        valid = (not installed_state or (session._installed is wrapped_scope and
                 session.__dict__.get("_vit_numeric_suite_720") is counter and
                 all(CONTRACTS.get(name) == value for name, value in counter._contracts.items())))
        if installed_state:
            session._installed = previous_scope
            session.__dict__.pop("_vit_numeric_suite_720", None)
        # The single-use session is retired even if failure occurred before a graph
        # existed, or only eager work changed private history. Never reuse it.
        session._failed = True
        try:
            counter.retired_graphs = graph.metadata()
            if not graph.closed:
                graph.close()
        finally:
            # Constants/binaries remain strongly owned by counter after graph close.
            counter._live = False
            for name in owned_contracts:
                CONTRACTS.pop(name, None)
        if not valid:
            raise RuntimeError("ViT numeric parent/marker/contracts changed before retirement")
