"""Default-off, owned DecoderGather.input numerical candidate for actual720.

API: installed(modes, decoder_input_full_k=False) -> counters, with no-argument
preflight(), validate(), snapshot(). Enable after select, before frames/capture;
use fused history and graph replay. preflight compiles/loads, never dispatches.
Only CPU AST validation has been performed during implementation. GPU binary,
spill, numerical, performance and continuous-video acceptance remain untested.

Keep the original feature boundary, zero-half initial, self.merge, NN and skip.
Replace four independent K256/half partitions by K1024 FP16 GEMM/FP32 sum and
one final half. No ViT or other upsample is selected. New device constants are
not needed. Children never own graph signatures or game select/process hooks.

The small shared receipt/lifecycle helpers below are also used by the assigned
branch_accum_native_720_v1 module. Project modules are loaded only when enabled.
Actual E/G/D source files are authenticated; candidate hashes use imported files.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
from importlib import import_module
import inspect
import json
from pathlib import Path
import sys
from threading import get_ident
from types import CodeType, MethodType


_KNOWN = {
    "nr_backend.unround_policy": "87bf4ba9b9f60e8c10173dc8498d54e23e3387dee0a4485fae08db21dedda827",
    "nr_backend.execution": "8fbfe06ff6bcd94011da4ec4038a9735ada672dd90df2ff793417344fd045462",
    "nr_backend.temporal": "35c1c1d5c1475ac0e3385dfa76f0e1020109a376a1d233a64e6c2a1036ca1828",
    "graph_front_v5": "8887facb3d98f91f6a068206698283b2865d895cbe7cdae366b0455939f3a6ec",
    "graph_front_v6": "3cf46091d2a44457975a95670aa317d6205e0ec9db4c5bc5acf4e70c9af7a052",
    "nr_game_controlled_model": "12477f3ed5999b714af5dd4a2f7b8c9dcb0b2962f84ed3b270f7c111a850a7ff",
    "nr_game_history_fused": "6e4f73ce30a29ffe3366abe558dbb1ef204726a10a1facab2487f1ff7d48e1f0",
    "decoder_gather_scope_v1": "c0a420b80151a0d253be26d84fe6a211f836613a9beb207441acbb0f02af4521",
    "nr_backend.decoder": "7f0e368e9791c78b2758938dd40ffe92fd357262ff9c296b6f43a2b3f95e4cde",
    "nr_backend.vit_block": "b7f8a992d04f3c5eceb4daf007b3153178f818ccb376fbb219aea0364469d4d5",
    "quantization_dataflow_v1": "c3cc91598f8378592cd8ea950ab32bfbab63f1b7997fcbfc7000518795912bd9",
    "c512_k8_joint_scope_720_v1": "4621cda4df3426cd6898ace624432b148ea17f77e5374b2cd5be9b22812c2ca5",
    "c512_qkv_library_16_v1": "67ed1c4e3865a3435c2f618491d4420f517235e1b5d66a9edb05f8fb393a0d00",
    "native_k8_fp16_v1": "45296adf5e43d9caf857a1a292eb162224c5125c26bad13e0a4c986d2121716e",
    "native_k8_active_720_v1": "e3362da7c8d499e99b7a0ab86c1ea3d95d0127aff2883a826f9e939b7f04acb9",
    "post_attention_k8_combined_v1": "8444a14eda43e1b5ea2409a28c1066ab60b1e8a1684bc522fa0908a5e52aec4c",
}


def _identity(options):
    return hashlib.sha256(json.dumps(options, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _function(value):
    value = getattr(value, "__func__", value)
    value = getattr(value, "fn", value)
    if not inspect.isfunction(value):
        value = type(value).__call__
    return inspect.unwrap(value)


def _same(left, right):
    return (left is right or
            (inspect.ismethod(left) and inspect.ismethod(right) and
             left.__self__ is right.__self__ and left.__func__ is right.__func__))


def _closure(function, name):
    function = _function(function)
    cells = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
    if name not in cells:
        raise RuntimeError(f"Missing reviewed closure {function.__qualname__}:{name}")
    return cells[name].cell_contents


def _nested_code(function, qualname):
    pending = [_function(function).__code__]
    while pending:
        code = pending.pop()
        if code.co_qualname == qualname:
            return code
        pending.extend(value for value in code.co_consts if isinstance(value, CodeType))
    raise RuntimeError(f"Reviewed nested callee code is missing: {qualname}")


class _Sources:
    def __init__(self):
        self.modules, self.receipts, self.symbols, self.stats = {}, {}, [], {}

    def add(self, module, *, sha256=None):
        path = Path(module.__file__).resolve()
        if path.suffix != ".py":
            raise RuntimeError(f"Expected inspectable Python source: {module.__name__}")
        stat = path.stat()
        state = (stat.st_size, stat.st_mtime_ns)
        old = self.receipts.get(module.__name__)
        expected = sha256 if sha256 is not None else _KNOWN.get(module.__name__)
        accepted = (expected,) if isinstance(expected, str) else expected
        if (old is not None and self.modules[module.__name__] is module and
                old["path"] == str(path) and self.stats[module.__name__] == state):
            if accepted is not None and old["sha256"] not in accepted:
                raise RuntimeError(f"Cached source is not approved: {module.__name__}")
            return module
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if accepted is not None and digest not in accepted:
            raise RuntimeError(f"Unreviewed source bytes: {module.__name__} at {path}")
        receipt = {"path": str(path), "sha256": digest}
        if old is not None and old != receipt:
            raise RuntimeError(f"Source changed: {module.__name__}")
        self.modules[module.__name__], self.receipts[module.__name__] = module, receipt
        self.stats[module.__name__] = state
        return module

    def load(self, name, *, sha256=None):
        return self.add(import_module(name), sha256=sha256)

    def function(self, value, *, module=None, qualname=None):
        function = _function(value)
        actual = sys.modules.get(function.__module__)
        if actual is None or (module is not None and actual is not module):
            raise RuntimeError("Actual callee belongs to an unknown imported module")
        if qualname is not None and function.__qualname__ != qualname:
            raise RuntimeError(f"Unrecognized actual callee: {function.__qualname__}")
        self.add(actual)
        row = self.receipts[actual.__name__]
        if Path(function.__code__.co_filename).resolve() != Path(row["path"]):
            raise RuntimeError("Actual callee co_filename differs from its imported source")
        self.watch(function, "__code__")
        if hasattr(value, "fn"):
            self.watch(value, "fn")
        return {"callee": f"{function.__module__}.{function.__qualname__}", **row}

    def watch(self, owner, name):
        if not any(saved_owner is owner and saved_name == name
                   for saved_owner, saved_name, _ in self.symbols):
            self.symbols.append((owner, name, getattr(owner, name)))

    def verify(self):
        for name, module in tuple(self.modules.items()):
            path = Path(module.__file__).resolve()
            if (str(path) != self.receipts[name]["path"] or
                    hashlib.sha256(path.read_bytes()).hexdigest() != self.receipts[name]["sha256"]):
                raise RuntimeError(f"Actual imported source changed: {name}")
        self.verify_symbols()

    def verify_symbols(self):
        """Memory-only callable identity checks for the timed path."""
        if any(not _same(getattr(owner, name), saved) for owner, name, saved in self.symbols):
            raise RuntimeError("An authenticated actual callee changed after binding")


def _stamp(value):
    # No device reads; refusal of inference tensors without _version is deliberate.
    return (value, value._version, value.data_ptr(), tuple(value.shape),
            tuple(value.stride()), value.dtype, value.device)


def _weight_receipt(name, stamp):
    _, version, pointer, shape, stride, dtype, device = stamp
    return {"owner": name, "version": version, "data_ptr": pointer,
            "shape": list(shape), "stride": list(stride),
            "dtype": str(dtype), "device": str(device)}


def _tensor(torch, name, value, shape, device):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or
            value.dtype != torch.float16 or value.device != device or
            device.type != "xpu" or not value.is_contiguous()):
        raise RuntimeError(f"{name}: expected contiguous same-device XPU FP16 {shape}")


def _binary(kernel):
    kernel._init_handles()
    spills = getattr(kernel, "n_spills", None)
    if type(spills) is not int or spills != 0:
        raise RuntimeError(f"Unknown/nonzero spill metadata: {spills!r}")
    key = getattr(kernel, "hash", None)
    if not isinstance(key, str) or not key:
        raise RuntimeError("Compiled kernel hash is unavailable")
    # The inspected Intel compiler can load SPIR-V OR offline zebin. Hash the
    # exact CompiledKernel.kernel passed to the driver, not an earlier IR stage.
    actual = getattr(kernel, "kernel", None)
    binaries = [(name, value) for name, value in kernel.asm.items()
                if name in ("spv", "spirv", "zebin") and isinstance(value, bytes)
                and value and value is actual]
    if len(binaries) != 1:
        raise RuntimeError("Actual executable SPIR-V/zebin bytes are missing/ambiguous")
    kind, binary = binaries[0]
    return {"kernel_hash": key, "binary_kind": kind,
            "binary_sha256": hashlib.sha256(binary).hexdigest(),
            "binary_bytes": len(binary), "spills": spills,
            "registers": getattr(kernel, "n_regs", None),
            "shared_bytes": getattr(kernel.metadata, "shared", None)}


def _screen(jit, args, grid, options):
    kernel = jit.warmup(*args, grid=grid, **options)
    return kernel, {**_binary(kernel), "grid": list(grid), "launch_options": dict(options)}


class _ReferenceCounter:
    def __init__(self, options):
        self._options = deepcopy(options)
        self.calls, self.capture_calls = {}, {}

    def validate(self):
        return True

    def preflight(self):
        return {}

    def snapshot(self):
        return {"options": deepcopy(self._options), "flag_identity": _identity(self._options),
                "enabled": False, "active": False, "reference_noop": True,
                "expected_sites": [], "calls": {}, "capture_calls": {},
                "sources": {}, "callees": {}, "resources": {}, "actualkernelhash": {},
                "capture_gates": [], "frame_routes": [], "frame_route_calls": {},
                "preflight_complete": False, "gpu_validation": "not-run"}


class _SessionCounter:
    """Composable session wrapper; local hooks exist only inside previous()."""
    def __init__(self, modes, options):
        self.modes, self._options = modes, deepcopy(options)
        self._frozen_options = deepcopy(options)
        self.flag_identity = _identity(options)
        self.thread, self.sources = get_ident(), _Sources()
        self.torch = import_module("torch")
        self.policy = self.sources.load("nr_backend.unround_policy")
        self.execution = self.sources.load("nr_backend.execution")
        self.temporal = self.sources.load("nr_backend.temporal")
        self.history = self.sources.load("nr_game_history_fused")
        self.controlled = self.sources.load("nr_game_controlled_model")
        self.graph5 = self.sources.load("graph_front_v5")
        self.graph6 = self.sources.load("graph_front_v6")
        self.dataflow = self.sources.load("quantization_dataflow_v1")
        self.session = getattr(modes, "session", None)
        if self.session is None:
            raise RuntimeError("Select the actual720 session before installing a candidate")
        self.session._ready()
        self.stack, self.geometry = self.session._stack, modes.geometry
        self.model, self.graph, self.provider = self.stack.model, self.stack.graph, self.stack.provider
        self.c512_calls, self.k8_calls = modes.c512_library_calls, modes.native_k8_calls
        self.mode_options = getattr(modes, "numeric_cleanup_720", None)
        self.mode_options_identity = getattr(self.mode_options, "identity", None)
        self.mode_options_value = (self.mode_options.to_dict()
                                   if hasattr(self.mode_options, "to_dict") else None)
        self._mode_fields = (tuple((key, getattr(self.mode_options, key)) for key in self.mode_options_value)
                             if self.mode_options_value is not None else ())
        self._saved_scope = self.session._installed
        self._had_scope = "_installed" in self.session.__dict__
        self._saved_scope_value = self.session.__dict__.get("_installed")
        self.active, self.retired, self._local, self.in_frame = False, False, False, False
        self._idle_slots = [(self.model, name, name in self.model.__dict__,
                             self.model.__dict__.get(name), getattr(self.model, name, None))
                            for name in ("forward", "_forward_front")]
        self._idle_sampler = self.temporal.warp_history_normalized
        self._idle_contract_registry = self.dataflow.CONTRACTS
        self._idle_contracts = {}
        self.calls, self.capture_calls, self.resources, self.actualkernelhash = {}, {}, {}, {}
        from audit_receipt_720_v1 import ReceiptRing
        impl = getattr(modes, "_audit_history_host_options_720", None)
        capacity = 64 if impl is None else impl.receipt_capacity
        self.capture_gates, self.frame_routes = ReceiptRing(capacity), ReceiptRing(capacity)
        self.frame_route_calls = {}
        self.callees, self.weights, self._compiled, self._entries = {}, {}, {}, {}
        self._binary_payloads = {}
        self.actualkernelhash_by_site, self._samplers = {}, []
        self._history_counter = None
        self.preflight_complete = False
        self.wrapper = None
        self.sources.function(self.graph._capture, module=self.graph5, qualname="GraphFront._capture")
        self.sources.function(self.graph.close, module=self.graph6, qualname="GraphFront.close")
        self.sources.watch(type(self.graph), "_capture")
        self.sources.watch(type(self.graph), "close")
        self.sources.watch(self.temporal.MotionNR, "forward")
        self.sources.watch(self.controlled.GameLiveControlledNR, "graph_controls")
        self._require_session(fresh=True)
        self._bind_matrix_owners()
        self._bind_sampler(self.history.warp_history_fused)

    def _bind_matrix_owners(self):
        joint = self.sources.load("c512_k8_joint_scope_720_v1")
        library = self.sources.load("c512_qkv_library_16_v1")
        pre_kernel = self.sources.load("native_k8_fp16_v1")
        post_kernel = self.sources.load("native_k8_active_720_v1")
        self.callees["native_pre_k8"] = self.sources.function(pre_kernel.dot)
        self.callees["native_post_k8"] = self.sources.function(post_kernel.post_head)
        self.sources.watch(pre_kernel, "dot")
        self.sources.watch(post_kernel, "post_head")
        self.sources.watch(joint, "pre_k8")
        self.sources.watch(joint, "post_k8")
        self.active_post = self.sources.load("post_attention_k8_combined_v1")
        self._dense = self.provider.dense
        self._post_head = self.active_post._contiguous_cropped_head
        self.callees["joint_dense"] = self.sources.function(
            self._dense, module=joint, qualname="installed.<locals>.dense")
        self.callees["joint_post_k8"] = self.sources.function(
            self._post_head, module=joint, qualname="installed.<locals>.post_head")
        self._delegate = _closure(self._dense, "delegated_dense")
        self.callees["c512_library_dense"] = self.sources.function(
            self._delegate, module=library, qualname="installed.<locals>.dense")
        if (getattr(self._dense, "__self__", None) is not self.provider or
                getattr(self._delegate, "__self__", None) is not self.provider or
                _closure(self._delegate, "calls") is not self.c512_calls or
                _closure(self._dense, "k8_calls") is not self.k8_calls or
                _closure(self._post_head, "k8_calls") is not self.k8_calls):
            raise RuntimeError("C512/K8 counters differ from the actual owned matrix callees")
        self._library_targets = _closure(self._delegate, "targets")
        self._library_modules = []
        for side in ("encoder512", "decoder512"):
            blocks = getattr(self.model, side)
            if len(blocks) != 8:
                raise RuntimeError("Expected eight encoder and eight decoder C512 owners")
            for index, block in enumerate(blocks):
                owner, label = block.attention, f"{side}_{index}"
                weight = owner.qkv
                _tensor(self.torch, label, weight, (512, 1536), weight.device)
                target = self._library_targets.get(id(weight))
                if target is None or target[0] is not weight or target[1] != label:
                    raise RuntimeError("C512 library target differs from the actual model weight")
                self._library_modules.append((side, index, owner, weight, label))
                self._save_weight(label + ".qkv", owner, "qkv")
        if len(self._library_targets) != 16:
            raise RuntimeError("Expected exactly sixteen distinct C512 library weights")
        self._pre, self._post = self.model.pre, self.model.post
        self._save_weight("model.pre.front_weight", self._pre, "front_weight")
        self._save_weight("model.post.head_weight", self._post, "head_weight")
        self.sources.watch(self.active_post, "_contiguous_cropped_head")

    def _require_matrix_owners(self):
        if (not _same(self.provider.dense, self._dense) or
                self.active_post._contiguous_cropped_head is not self._post_head or
                self.model.pre is not self._pre or self.model.post is not self._post or
                _closure(self._dense, "delegated_dense") is not self._delegate or
                _closure(self._delegate, "targets") is not self._library_targets or
                len(self._library_targets) != 16):
            raise RuntimeError("Actual C512/K8 callee or owner changed before replay")
        for side, index, owner, weight, label in self._library_modules:
            target = self._library_targets.get(id(weight))
            if (getattr(self.model, side)[index].attention is not owner or owner.qkv is not weight or
                    target is None or target[0] is not weight or target[1] != label):
                raise RuntimeError("Actual C512 library module/weight target changed")

    def _require_session(self, *, fresh=False):
        self._require_fixed_session(fresh=fresh)
        self._require_thread()

    def _require_thread(self):
        current_thread = get_ident()
        if current_thread != self.thread:
            raise RuntimeError(
                f"Candidate CPU thread owner changed: expected={self.thread}, current={current_thread}; "
                "explicit owned suite serial handoff is required")

    def _require_fixed_session(self, *, fresh=False):
        """Original session/config/graph gates, independent of a CPU handoff."""
        self._require_live_session()
        modes = self.modes
        expected = {f"{side}_{i}" for side in ("encoder512", "decoder512") for i in range(8)}
        if (not isinstance(self.c512_calls, dict) or set(self.c512_calls) != expected or
                not isinstance(self.k8_calls, dict) or set(self.k8_calls) != {"pre", "post"}):
            raise RuntimeError("Expected the actual sixteen C512 and pre/post K8 receipt objects")
        if fresh and (self.graph.entries or self.graph.replays or modes._session_frames or
                      self.model._previous is not None or self.model._next_seed != 0):
            raise RuntimeError("Candidate requires a fresh uncaptured session with fresh history/seed")

    def _require_live_session(self):
        """Current session/modes/provider gates; no fixed matrix/weight enumeration."""
        self.session._ready()
        modes, mode = self.modes, self.geometry.mode
        if (modes.session is not self.session or
                self.session._stack is not self.stack or self.stack.model is not self.model or
                self.stack.graph is not self.graph or self.stack.provider is not self.provider or
                modes.geometry is not self.geometry or not modes.controlled or
                modes.height != 720 or tuple(modes.source) != (720, 1280) or
                modes.variant != "unrounded" or not modes.c512_qkv_library_720 or
                not modes.native_k8_720 or modes.c512_library_calls is not self.c512_calls or
                modes.native_k8_calls is not self.k8_calls or 720 not in modes.combo_modes or
                modes.graph_capture_policy != "all" or modes.history_compact_720 or
                modes.vit_head_720 or modes.post_rgb_tail_720 or modes.c32_window_chain_720 or
                modes.c512_encoder_window_blocks_720 or
                tuple(mode.active) != (720, 1280) or tuple(mode.model) != (720, 1280) or
                tuple(mode.internal) != (768, 1280) or tuple(mode.inset) != (0, 0) or
                tuple(self.geometry.source) != (720, 1280) or
                tuple(self.session.fullsize_geometry) != (720, 1280) or
                tuple(self.session.fullsize_padding["padding"]) != (768, 1280) or
                tuple(self.model.PADDED_SIZES.get((720, 1280), ())) != (768, 1280) or
                self.graph.closed or self.graph.model is not self.model or
                self.graph.arithmetic is not self.provider or self.provider.mode != "fp16_xmx"):
            raise RuntimeError("Candidate left the owned actual720 C512+K8 session")
        if (getattr(modes, "numeric_cleanup_720", None) is not self.mode_options or
                any(getattr(self.mode_options, key) != value for key, value in self._mode_fields) or
                self._options != self._frozen_options):
            raise RuntimeError("Frozen candidate/mode options changed")

    def _require_dispatch(self):
        from replay_lifecycle_audit_base_720_v1 import require_cold_dispatch
        require_cold_dispatch(self)
        self._require_session()
        if (not self.active or not self._local or not self.preflight_complete or
                self.policy.ENABLED != self.policy.FAMILIES or
                self.execution.current_arithmetic_backend() != "triton"):
            raise RuntimeError("Preflighted candidate must run inside its unrounded installed scope")
        return self._require_front_owner()

    def _require_live_front(self):
        """Actual controlled frontend and arithmetic scope at entry consumption."""
        if (not self.active or not self.in_frame or not self._local or
                self.policy.ENABLED != self.policy.FAMILIES or
                self.execution.current_arithmetic_backend() != "triton"):
            raise RuntimeError("Lifecycle replay requires the actual unrounded frame/arithmetic scope")
        return self._require_front_owner()

    def _require_front_owner(self):
        front = self.model.__dict__.get("_forward_front")
        if (front is None or _function(front).__module__ != self.controlled.__name__ or
                _function(front).__qualname__ != "GameLiveControlledNR.graph_controls.<locals>.controlled_front" or
                getattr(_closure(front, "original"), "__self__", None) is not self.graph):
            raise RuntimeError("Candidate requires the actual owned controlled graph frontend")
        return self._sampler_receipt()

    def _sampler_receipt(self):
        sampler = self.temporal.warp_history_normalized
        for saved, receipt in self._samplers:
            if _same(saved, sampler):
                return receipt
        raise RuntimeError("Actual fused history callee was not authenticated in preflight")

    def _bind_sampler(self, sampler):
        function = _function(sampler)
        if function.__module__ not in ("nr_game_history_fused", "history_numeric_suite_720_v1",
                                        "history_value_native_720_v1"):
            raise RuntimeError("Unknown actual fused history owner")
        if not any(_same(saved, sampler) for saved, _ in self._samplers):
            self._samplers.append((sampler, self.sources.function(sampler)))

    def _preflight_sources(self):
        # All children have been created before suite.preflight(). Bind the
        # optional history child's real dispatcher now, never in timed frames.
        counter = getattr(self.session, "_history_numeric_suite_720", None)
        if counter is not None:
            self._bind_sampler(counter)
            self._history_counter = counter
        self._bind_sampler(self.history.warp_history_fused)
        self.sources.verify()

    def _require_weights(self, names=None):
        for name in self.weights if names is None else names:
            owner, attribute, saved = self.weights[name]
            value = getattr(owner, attribute)
            if value is not saved[0] or _stamp(value) != saved:
                raise RuntimeError(f"Owned constant identity/version/pointer changed: {name}")

    def _save_weight(self, name, owner, attribute):
        self.weights[name] = (owner, attribute, _stamp(getattr(owner, attribute)))

    def _launch(self, role, kernel, args, grid):
        row = self.resources[role]
        if (self._compiled[role] is not kernel or kernel.hash != row["kernel_hash"] or
                kernel.kernel is not self._binary_payloads[role] or
                getattr(kernel, "n_spills", None) != 0):
            raise RuntimeError(f"Dispatch differs from preflighted zero-spill binary: {role}")
        # Dispatch the exact loaded CompiledKernel, bypassing re-specialization.
        kernel[tuple(grid) + (1,) * (3 - len(grid))](*args)
        self.actualkernelhash[role] = row["kernel_hash"]

    def _validate_fixed_owner(self):
        self._require_fixed_session()
        if not self.active or self.retired:
            raise RuntimeError("Candidate scope is inactive/retired")
        from numeric_frame_validation_720_v1 import checked
        return checked(self, "fixed_owner", self._validate_fixed_owner_full,
                       check_thread=False)

    def _validate_fixed_owner_full(self):
        self.sources.verify_symbols()
        self._require_matrix_owners()
        self._require_weights()
        self._validate_callees()
        self._require_installed_owner()
        for role, kernel in self._compiled.items():
            if (kernel.kernel is not self._binary_payloads[role] or
                    kernel.hash != self.resources[role]["kernel_hash"] or kernel.n_spills != 0):
                raise RuntimeError("Preflighted actual binary/hash/spill metadata changed")

    def _require_installed_owner(self):
        """Actual temporary installation chain and selected history owner stay live."""
        current, visited = self.session._installed, set()
        while current is not self.wrapper and id(current) not in visited:
            visited.add(id(current))
            current = getattr(current, "__nr_numeric_original_installed__",
                              getattr(current, "__wrapped__", None))
            if current is None:
                raise RuntimeError("Candidate wrapper is missing from the actual installed chain")
        if current is not self.wrapper:
            raise RuntimeError("Candidate installed wrapper chain contains a cycle")
        if (self._history_counter is not None and
                getattr(self.session, "_history_numeric_suite_720", None) is not self._history_counter):
            raise RuntimeError("Preflighted owned history dispatcher changed")

    def _validate_live_owner(self, *, check_thread=True):
        self._require_live_session()
        if not self.active or self.retired or not self.preflight_complete:
            raise RuntimeError("Lifecycle candidate is inactive/retired/not preflighted")
        self._validate_live_callees()
        self._require_installed_owner()
        if check_thread:
            self._require_thread()

    def _validate_replay(self, *, check_thread=True):
        """Live only under an owned trial frame; legacy checks otherwise."""
        try:
            from replay_lifecycle_audit_base_720_v1 import live_child, invalidate_suite
            if live_child(self, check_thread=check_thread):
                return True
            self._validate_fixed_owner()
            if check_thread:
                self._require_thread()
            return True
        except BaseException:
            self.session._failed = True
            invalidate_suite(getattr(self.modes, "_numeric_cleanup_owner", None), 'Child replay gate failed')
            raise

    def _validate_scope_callees(self):
        from replay_lifecycle_audit_base_720_v1 import live_child
        if not live_child(self):
            self._validate_callees()

    def _require_serial_idle(self, owner):
        if not self.active or self.retired:
            raise RuntimeError("Candidate serial handoff requires an active unretired child")
        if any(getattr(child, name, False) for child in owner.children.values()
               for name in ("in_frame", "_in_frame", "_in_scope", "_local", "_post_active")):
            raise RuntimeError("Candidate serial handoff refused: a suite frame/local scope is active")
        history_module = sys.modules.get("history_numeric_suite_720_v1")
        if history_module is not None and getattr(history_module, "_IN_FLIGHT", None) is not None:
            raise RuntimeError("Candidate serial handoff refused: a global history frame is active")
        if self.torch.xpu.is_current_stream_capturing():
            raise RuntimeError("Candidate serial handoff refused during XPU stream capture")
        if (self.temporal.warp_history_normalized is not self._idle_sampler or
                self.session.__dict__.get("_numeric_model_forward_registry_720") or
                any((name in target.__dict__) != present or
                    target.__dict__.get(name) is not saved or
                    not _same(getattr(target, name, None), resolved)
                    for target, name, present, saved, resolved in self._idle_slots)):
            raise RuntimeError("Candidate serial handoff refused: temporary sampler/forward hooks are active or changed")
        if (self.dataflow.CONTRACTS is not self._idle_contract_registry or
                any((key in self.dataflow.CONTRACTS) != present or
                    self.dataflow.CONTRACTS.get(key) != saved
                    for key, (present, saved) in self._idle_contracts.items())):
            raise RuntimeError("Candidate serial handoff refused: temporary JIT contracts are active or changed")
        history = self.session.__dict__.get("_history_numeric_suite_720")
        if (history is not owner.children.get("history") or
                (self._history_counter is not None and history is not self._history_counter)):
            raise RuntimeError("Candidate serial handoff lost its owned history child reference")
        # History is a separate suite child. Never move or validate its thread
        # here: the parent owns ordering, final validation and overall rollback.

    def _transfer_serial_thread(self, owner, *, serial_guard, previous_thread):
        """Private single-child transaction; the actual suite must call directly."""
        old_thread = self.thread
        try:
            module = sys.modules.get("numeric_cleanup_suite_720_v1")
            caller = sys._getframe(1)
            if (module is None or type(owner) is not module.NumericCleanupCounter or
                    caller.f_code is not module.NumericCleanupCounter.transfer_serial_thread.__code__ or
                    caller.f_globals is not module.__dict__ or caller.f_locals.get("self") is not owner or
                    caller.f_locals.get("serial_guard") is not serial_guard or
                    getattr(self.modes, "_numeric_cleanup_owner", None) is not owner or
                    owner.modes is not self.modes or owner.session is not self.session or
                    owner.children.get(self.serial_child_label) is not self or
                    type(serial_guard) is not module._SERIAL_GUARD_TYPE or not serial_guard._is_owned()):
                raise RuntimeError("Candidate serial handoff requires its owned suite transaction and held process guard")
            if type(previous_thread) is not int or previous_thread != old_thread:
                raise RuntimeError(
                    f"Candidate serial handoff CPU owner mismatch: expected={old_thread}, previous={previous_thread!r}")
            self._require_serial_idle(owner)
            self._validate_replay(check_thread=False)
            self.thread = get_ident()
            self._validate_replay()
            return self.thread
        except BaseException:
            self.thread = old_thread
            self.session._failed = True
            raise

    def validate(self):
        """Explicit uncached diagnostic audit; no compilation or launch."""
        try:
            self._require_fixed_session()
            if not self.active or self.retired:
                raise RuntimeError("Candidate scope is inactive/retired")
            self._validate_fixed_owner_full()
            self._require_thread()
            return True
        except BaseException:
            self.session._failed = True
            raise

    def _frame(self, before, history):
        token = before.get("lifecycle_token")
        if token is not None:
            from audit_replay_receipt_720_v1 import capture_delta
            new_count, replays, proof = capture_delta(self, token)
            current = tuple(self.calls.values()) + tuple(self.capture_calls.values()) + tuple(self.c512_calls.values()) + tuple(self.k8_calls.values())
            if not new_count and current != before["warm_counters"]:
                raise RuntimeError("Warm graph replay changed actual Python/capture/provider counters")
            entry = self.graph.last_entry
            history_frame = entry.inputs.get("previous") is not None
            if new_count:
                self._entries[token.state.consumed[0]] = entry
                self.capture_gates.append({"new_entries": new_count, "passed": proof["passed"],
                                           "actual_capture_proof": proof})
            if not any(entry is saved for saved in self._entries.values()):
                raise RuntimeError("Graph replay used an entry not captured under these fixed flags")
            detail = None
            if self._history_counter is not None:
                from audit_receipt_720_v1 import latest_history_frame
                detail = latest_history_frame(self._history_counter, before["history_frames"])
                history = next(row for saved, row in self._samplers if saved is self._history_counter)
            frontend = "capture-replay" if new_count else "replay"
            route = "history-fused" if history_frame else "reset-no-warp"
            self.frame_routes.append({"frame": self.frame_routes.total, "frontend": frontend,
                                      "history": route, "sampler": history, "history_detail": detail,
                                      "graph_replays": replays, "entry_replays": entry.replays,
                                      "flag_identity": self.flag_identity, "flags_effective": True,
                                      "new_entries": new_count, "passed": True,
                                      "capture_proof_scope": "lifecycle_actual_miss" if proof else "warm_one_replay"})
            for name in (frontend, route):
                self.frame_route_calls[name] = self.frame_route_calls.get(name, 0) + 1
            return
        new = {key: entry for key, entry in self.graph.entries.items() if key not in before["entries"]}
        replays = self.graph.replays - before["replays"]
        stable = all(self.graph.entries.get(key) is entry for key, entry in before["entries"].items())
        captured = {site: self.capture_calls[site] - before["capture"][site] for site in self.calls}
        python = {site: self.calls[site] - before["calls"][site] for site in self.calls}
        c512 = {site: value - before["c512"][site] for site, value in self.c512_calls.items()}
        k8 = {site: value - before["k8"][site] for site, value in self.k8_calls.items()}
        provider_proof = None
        if new:
            from implementation_capture_roles_720_v1 import body_provider_capture_gate
            provider_proof = body_provider_capture_gate(
                self.modes, before["implementation"], c512, k8, builds=len(new))
            providers_passed = provider_proof["passed"]
        else:
            providers_passed = all(count == 0 for count in (*c512.values(), *k8.values()))
        passed = (stable and replays > 0 and
                  all(count == len(new) for count in captured.values()) and
                  all(count == 3 * len(new) for count in python.values()) and
                  providers_passed)
        if new:
            self.capture_gates.append({"new_entries": len(new), "capture_calls": captured,
                                       "python_calls": python, "c512_calls": c512,
                                       "native_k8_calls": k8, "passed": passed,
                                       "body_provider_proof": provider_proof})
        if not passed:
            raise RuntimeError("Graph/candidate/C512/K8 capture or replay route gate failed")
        self._entries.update(new)
        entry = self.graph.last_entry
        if not any(entry is saved for saved in self._entries.values()):
            raise RuntimeError("Graph replay used an entry not captured under these fixed flags")
        history_frame = entry.inputs.get("previous") is not None
        frontend = "capture-replay" if new else "replay"
        history_route = "history-fused" if history_frame else "reset-no-warp"
        history_detail = None
        if self._history_counter is not None:
            from audit_receipt_720_v1 import latest_history_frame
            history_detail = latest_history_frame(self._history_counter, before["history_frames"])
            history = next(row for saved, row in self._samplers if saved is self._history_counter)
        row = {"frame": self.frame_routes.total, "frontend": frontend,
               "history": history_route, "sampler": history, "history_detail": history_detail,
               "graph_replays": replays, "entry_replays": entry.replays,
               "capture_calls": captured, "calls": python, "flag_identity": self.flag_identity,
               "flags_effective": True, "passed": True}
        self.frame_routes.append(row)
        for name in (frontend, history_route):
            self.frame_route_calls[name] = self.frame_route_calls.get(name, 0) + 1

    def snapshot(self):
        # Source I/O is deliberately confined to initialization/preflight/report.
        if self.active:
            self.validate()
        self.sources.verify()
        return deepcopy({"options": self._options, "flag_identity": self.flag_identity,
                         "mode_options_identity": self.mode_options_identity,
                         "mode_options": self.mode_options_value,
                         "enabled": True, "active": self.active, "retired": self.retired,
                         "expected_sites": list(self.calls), "calls": self.calls,
                         "capture_calls": self.capture_calls,
                         "eager_calls": {site: self.calls[site] - self.capture_calls[site]
                                         for site in self.calls},
                         "noncapture_counts_include_graph_warmups": True,
                         "sources": self.sources.receipts, "callees": self.callees,
                         "constants": {name: _weight_receipt(name, saved)
                                       for name, (_, _, saved) in self.weights.items()},
                         "resources": self.resources, "actualkernelhash": self.actualkernelhash,
                         "actualkernelhash_by_site": self.actualkernelhash_by_site,
                         "capture_gates": list(self.capture_gates), "frame_routes": list(self.frame_routes),
                          "frame_sequence": self.frame_routes.total,
                          "receipt_retention": self.frame_routes.snapshot(),
                         "frame_route_calls": self.frame_route_calls,
                         "preflight_complete": self.preflight_complete,
                         "numerical_contract": self.numerical_contract,
                         "theoretical_launches_saved": self.launches_saved,
                         "contracts": {self.contract: [["OUT"], []]},
                         "gpu_validation": "unverified", "python_counts_exclude_replays": True,
                         "timed_validation_overhead": "included; CPU metadata/callee checks, no source I/O/device readback"})

    @contextmanager
    def bound(self):
        previous = self._saved_scope

        @contextmanager
        def wrapped_scope():
            if self.in_frame:
                self.session._failed = True
                raise RuntimeError("Candidate frame scope is already active")
            self.in_frame = True
            try:
                self._validate_replay()
                if not self.preflight_complete:
                    raise RuntimeError("Call counter.preflight() before any frame/capture/replay")
                from audit_replay_receipt_720_v1 import capture_token
                token = capture_token(self)
                if token is not None:
                    before = {"lifecycle_token": token,
                              "warm_counters": tuple(self.calls.values()) + tuple(self.capture_calls.values()) +
                                               tuple(self.c512_calls.values()) + tuple(self.k8_calls.values()),
                              "history_frames": self._history_counter.frame_routes.total if self._history_counter is not None else 0}
                else:
                    before_implementation = None
                    if self.modes.implementation_calls_720:
                        from implementation_capture_roles_720_v1 import capture_before
                        before_implementation = capture_before(self.modes)
                    before = {"entries": dict(self.graph.entries), "replays": self.graph.replays,
                              "capture": dict(self.capture_calls), "calls": dict(self.calls),
                              "c512": dict(self.c512_calls), "k8": dict(self.k8_calls),
                              "implementation": before_implementation,
                              "history_frames": (self._history_counter.frame_routes.total
                                                 if self._history_counter is not None else 0)}
                # The old component/class/alias scope must be active first.
                with previous() as original, self._local_scope():
                    history = self._sampler_receipt()
                    yield original
                    self._frame(before, history)
                self._validate_replay()
            except BaseException:
                self.session._failed = True
                raise
            finally:
                self.in_frame = False

        wrapped_scope.__nr_numeric_original_installed__ = previous
        self.wrapper = wrapped_scope
        self.session._installed, self.active = wrapped_scope, True
        failure = None
        try:
            yield self
        except BaseException as exc:
            failure = exc
            self.session._failed = True
            raise
        finally:
            self.active, self.retired = False, True
            problems = []
            if self.session.__dict__.get("_installed") is wrapped_scope:
                if self._had_scope:
                    self.session._installed = self._saved_scope_value
                else:
                    del self.session._installed
            else:
                problems.append("session installed wrapper ownership changed")
            # Close/reset graphs before dropping screened binary references.
            self.session._failed = True
            graph_retired = False
            try:
                self.graph.close()
                graph_retired = bool(self.graph.closed and not self.graph.entries)
            except BaseException as exc:
                problems.append(f"graph retirement failed: {exc}")
            if graph_retired:
                self._compiled.clear()
                self._binary_payloads.clear()
            if problems:
                message = "; ".join(problems)
                if failure is None:
                    raise RuntimeError(message)
                failure.add_note(message)


class DecoderInputCounter(_SessionCounter):
    serial_child_label = "decoder_input"
    launches_saved = 3
    numerical_contract = ("owned DecoderInput only; K1024 half operands, FP32 full-K dot; "
                          "zero-half initial added once; one final half; unchanged boundary/merge/NN/skip")

    def __init__(self, modes):
        super().__init__(modes, {"decoder_input_full_k": True})
        self.sources.add(sys.modules[__name__])
        self.gather = self.sources.load("decoder_gather_scope_v1")
        self.decoder = self.sources.load("nr_backend.decoder")
        self.vit = self.sources.load("nr_backend.vit_block")
        self.kernels = self.sources.load((__package__ + "." if __package__ else "") +
                                        "decoder_input_full_k_720_kernel_v1")
        self.owner, self.module = self.stack.decoder_gather, self.model.decoder_input
        if (type(self.owner) is not self.gather.DecoderGather or
                type(self.module) is not self.decoder.DecoderInputUpsample or
                self.owner.model is not self.model or self.owner not in self.stack.components or
                len(self.owner.modules) != 5 or self.owner.modules[0] != ("decoder_input", self.module)):
            raise RuntimeError("Expected the actual owned DecoderGather input and five existing transitions")
        self._old_input = self.owner.input
        self.callees["input"] = self.sources.function(
            self._old_input, module=self.gather, qualname="DecoderGather.input")
        self.callees["split_k_projection"] = self.sources.function(
            self.decoder.split_k_projection, module=self.vit, qualname="split_k_projection")
        self.callees["candidate"] = self.sources.function(self.kernels._project_full_k)
        self._old_merge, self._old_boundary = self.owner.merge, self.owner.boundary
        self.callees["merge"] = self.sources.function(self._old_merge)
        self.callees["boundary"] = self.sources.function(self._old_boundary, module=self.gather,
                                                        qualname="DecoderGather.boundary")
        self._gather_flags = (self.owner.unround_activations, self.owner.unround_merge)
        self.sources.watch(self.decoder, "split_k_projection")
        self.sources.watch(self.gather.DecoderGather, "input")
        self.sources.watch(self.kernels, "_project_full_k")
        self.sources.watch(self.kernels._project_full_k, "fn")
        self._forward_code = _nested_code(self.gather.DecoderGather.installed,
                                         "DecoderGather.installed.<locals>.forward")
        self._save_weight("model.decoder_input.weight", self.module, "weight")
        self._save_weight("model.decoder_input.skip_scale", self.module, "skip_scale")
        self.device = self.module.weight.device
        _tensor(self.torch, "DecoderInput weight", self.module.weight, (1024, 512), self.device)
        _tensor(self.torch, "DecoderInput skip scale", self.module.skip_scale, (512,), self.device)
        self.calls, self.capture_calls = {"decoder_input": 0}, {"decoder_input": 0}
        self.tile, self._bound_input = (16, 32), None
        self.contract = self.kernels.__name__ + "._project_full_k"
        self._idle_slots.extend((module, "forward", "forward" in module.__dict__,
                                 module.__dict__.get("forward"), module.forward)
                                for _, module in self.owner.modules)
        self._idle_contracts = {key: (key in self.dataflow.CONTRACTS, self.dataflow.CONTRACTS.get(key))
                                for key in (self.contract, *self.gather.EXTRA)}

    def _validate_callees(self):
        expected = self._bound_input if self._local else self._old_input
        if (self.model.decoder_input is not self.module or self.stack.decoder_gather is not self.owner or
                self.owner.model is not self.model or self.owner.modules[0][1] is not self.module or
                not _same(self.owner.input, expected) or not _same(self.owner.merge, self._old_merge) or
                not _same(self.owner.boundary, self._old_boundary) or
                (self.owner.unround_activations, self.owner.unround_merge) != self._gather_flags):
            raise RuntimeError("DecoderInput owned callee or Gather numeric boundary changed")

    def _validate_live_callees(self):
        self._validate_callees()
        if self._local:
            forward = self.module.forward
            if (getattr(forward, "__self__", None) is not self.module or
                    _function(forward).__code__ is not self._forward_code or
                    _closure(forward, "self") is not self.owner or
                    self.dataflow.CONTRACTS.get(self.contract) != (("OUT",), ())):
                raise RuntimeError("DecoderInput actual temporary Gather/input/contract owner changed")

    def preflight(self):
        """Future Luna gate: real M240/K1024/N512, exact binary, zero spill."""
        try:
            self.validate()
            self._require_session(fresh=True)
            self._preflight_sources()
            if self.torch.xpu.is_current_stream_capturing():
                raise RuntimeError("DecoderInput preflight cannot run during capture")
            if not self.preflight_complete:
                with self.torch.inference_mode(False):
                    sample = self.torch.empty((12, 20, 1024), dtype=self.torch.float16, device=self.device)
                    initial = self.torch.empty((12, 20, 512), dtype=self.torch.float16, device=self.device)
                    output = self.torch.empty_like(initial)
                    args = (sample, self.module.weight, initial, output, *self.tile)
                    kernel, resource = _screen(self.kernels._project_full_k, args, (15, 16),
                                               dict(num_warps=4, num_stages=1, enable_fp_fusion=False))
                self._compiled["decoder_input"], self.resources["decoder_input"] = kernel, resource
                self._binary_payloads["decoder_input"] = kernel.kernel
                resource.update(source_kernel=self.contract, input_shape=[12, 20, 1024],
                                output_shape=[12, 20, 512], tile=list(self.tile),
                                input_stride=[20480, 1024, 1], weight_stride=[512, 1],
                                output_stride=[10240, 512, 1])
                self.preflight_complete = True
            return deepcopy(self.resources)
        except BaseException:
            self.session._failed = True
            raise

    def _input(self, owner, module, features, skip):
        try:
            self._require_dispatch()
            self._require_weights(("model.decoder_input.weight", "model.decoder_input.skip_scale"))
            self._validate_callees()
            if owner is not self.owner or module is not self.module:
                raise RuntimeError("Unrecognized DecoderInput module; no reference fallback")
            _tensor(self.torch, "DecoderInput features", features, (12, 20, 1024), self.device)
            _tensor(self.torch, "DecoderInput skip", skip, (24, 40, 512), self.device)
            x = owner.boundary("vit", features)
            _tensor(self.torch, "DecoderInput boundary", x, (12, 20, 1024), self.device)
            initial = self.torch.zeros((12, 20, 512), device=self.device, dtype=self.torch.float16)
            output = self.torch.empty_like(initial)
            self._launch("decoder_input", self._compiled["decoder_input"],
                         (x, module.weight, initial, output, *self.tile), (15, 16))
            self.actualkernelhash_by_site["decoder_input"] = {
                "projection": self.actualkernelhash["decoder_input"]}
            self.execution.record_arithmetic_dispatch("dense")
            result = owner.merge("decoder_input", output, skip, module.skip_scale)
            self.calls["decoder_input"] += 1
            if self.torch.xpu.is_current_stream_capturing():
                self.capture_calls["decoder_input"] += 1
            return result
        except BaseException:
            self.session._failed = True
            raise

    @contextmanager
    def _local_scope(self):
        self._validate_scope_callees()
        forward = self.module.forward
        if (getattr(forward, "__self__", None) is not self.module or
                _function(forward).__code__ is not self._forward_code or
                _closure(forward, "self") is not self.owner):
            raise RuntimeError("Previous session scope did not install the actual Gather owner")
        had, saved = "input" in self.owner.__dict__, self.owner.__dict__.get("input")
        if self.contract in self.dataflow.CONTRACTS:
            raise RuntimeError("Duplicate DecoderInput top-level kernel contract owner")
        replacement = MethodType(lambda owner, module, features, skip:
                                 self._input(owner, module, features, skip), self.owner)
        self._bound_input = replacement
        self.owner.input, self._local = replacement, True
        self.dataflow.CONTRACTS[self.contract] = (("OUT",), ())
        try:
            yield
        finally:
            valid = self.owner.__dict__.get("input") is replacement
            contract_ok = self.dataflow.CONTRACTS.get(self.contract) == (("OUT",), ())
            if valid:
                if had:
                    self.owner.input = saved
                else:
                    del self.owner.input
            if contract_ok:
                del self.dataflow.CONTRACTS[self.contract]
            self._local, self._bound_input = False, None
            if not valid or not contract_ok:
                self.session._failed = True
                raise RuntimeError("DecoderInput local callee/contract ownership changed")


@contextmanager
def installed(modes, *, decoder_input_full_k=False):
    """Bool default-off API, matching NumericCleanupOptions.selected_scopes()."""
    if type(decoder_input_full_k) is not bool:
        raise TypeError("decoder_input_full_k must be bool")
    if not decoder_input_full_k:
        yield _ReferenceCounter({"decoder_input_full_k": False})
        return
    try:
        counter = DecoderInputCounter(modes)
    except BaseException:
        session = getattr(modes, "session", None)
        if session is not None:
            session._failed = True
        raise
    with counter.bound():
        yield counter
