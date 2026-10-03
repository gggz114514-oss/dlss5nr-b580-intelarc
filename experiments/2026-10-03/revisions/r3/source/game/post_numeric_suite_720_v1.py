"""Opt-in post sigmoid/store experiments, plan 9, actual720 C512+nativeK8.

Development receipt (2026-09-30): CPU AST ONLY. No model import, GPU, Triton
compile, numerical test, video, install or deployment has been performed.
G/E capture_body has a real optional route/lifetime difference. Both exact
sources are independently reviewed for these post/noise hooks; acceptance of
the active G body is not a claim of E/G or GPU numerical equivalence.

After modes.select(720, (720, 1280), variant="unrounded"), before any frame:
    with installed(modes, post_sigmoid="native", post_store="reference") as c:
        c.preflight()  # FUTURE Luna GPU compile/load, never run during this edit
        # modes.process(..., history_warp="fused", graph_replay=True, ...)
        receipt = c.snapshot()
The reference/table default returns a counters object without touching modes.
Native RTZ is unavailable until the actual Intel binary passes the explicit
FP32->FP16 rounding/ISA gate. Unattempted is not backend-unsupported. The
independent native_rtz_probe(count=256) exposes jit/preflight/launch/expected/
snapshot for Luna; it never enables production or needs a model/session.
RNE is independently selectable and lossy.

Only owned post.forward and sigmoid.forward are changed. post.forward is a
FunctionType clone of the authenticated original code with selected globals
replaced; forward_head, layouts, RGB gain, blend_scale, alpha, history
normalization/subtraction, reset clamp, signed overshoot and return_float32
remain the original dataflow. No new RGB/history tail fusion is introduced.
Session wrappers compose with prior wrappers; ONLY the main suite writes the
graph signature. Every candidate scope retires its graphs/session on exit.

LOCAL TESTS FOR LUNA (not executed): all 65536 half logit bit patterns, split
finite/nonfinite, signed zero, subnormals, infinities/NaNs, monotonicity,
[0,1], saturation, first differing bits/ULP; FP32 values adjacent to every
half midpoint and random bit patterns for signed RTZ/RNE; reset clamp and
signed history overshoot. Include blend_scale zero/one/large, return_float32
and raw private history. Then flag-off bytes, reset/history capture/replay,
13 frames, independent 50-frame B-C-C-B arms and full continuous-history
video review. Sigmoid stays one logical helper kernel; exact baseline Torch
launches/store savings require graph evidence, not a source estimate.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
import hashlib
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import struct
import sys
from threading import RLock, get_ident
from types import CodeType, FunctionType, MappingProxyType, MethodType


SIZE, INTERNAL = (720, 1280), (768, 1280)
BLOCK = 256
_SERIAL_GUARD_TYPE = type(RLock())
KNOWN_SHA256 = {
    "post": ("c506e1b32c32d7adb70ed79eee8a1e5920922c86666c960904d7afda05ecee4c",),
    "sigmoid": ("567bed0fc4d908a941db4fea03d3b019b37a5db16279e82baec594cc84282d2c",),
    "front": ("d52cc097c9894345e6177bf17385f9d8ca3b0522d70d00667813e365a9b1ab2b",),
    "fused_front": ("37813197c7326041866b059d6874358914cac297700345f9ed4ab3db5209b552",),
    "temporal": ("35c1c1d5c1475ac0e3385dfa76f0e1020109a376a1d233a64e6c2a1036ca1828",),
    "policy": ("87bf4ba9b9f60e8c10173dc8498d54e23e3387dee0a4485fae08db21dedda827",),
    "execution": ("8fbfe06ff6bcd94011da4ec4038a9735ada672dd90df2ff793417344fd045462",),
    "sampling": ("e95228c1743bdb5013d3c3b68cbb85fae315c6deeb913b6c4a057c76d2ed89f8",),
    "noise": ("1bfdfbdcd883aa2e8db35fae52b158f541d6f094de66282c1ca81175b8500707",),
    "controlled": ("12477f3ed5999b714af5dd4a2f7b8c9dcb0b2962f84ed3b270f7c111a850a7ff",),
    "capture_body": (
        "b2096293aa1b3515aa047453fd98dac2017e6505a964884a26949b9eab46fefe",
        # Independently read active G C512+K8 body: forward_boundaries route.
        # Post clones/reference callables and noise dispatcher need no E-only route.
        "2f7fa28dd66a43cd20d0edaee28d09aaac034fe90b3865080ffc065d7397dfbb"),
    "graph_v1": ("d2b8a5f3cb913b71c836e8c2dc0a19cb34b0dbacf6a83851010d11a5a172f5de",),
    "graph_v2": ("09471129801447263272c596708d4ce0d5b1e059621db35afab7490df6055725",),
    "graph_v3": ("bacf2c8acb00356c79d82b48c73954cf559254fec9d18ab6cca01663f1f05c25",),
    "graph_v4": ("bdd8ed34ee979608f445eb82e2094c8b1b04c3ed288ee5038949c1c52872ca5e",),
    "graph_v5": ("8887facb3d98f91f6a068206698283b2865d895cbe7cdae366b0455939f3a6ec",),
    "graph_v6": ("3cf46091d2a44457975a95670aa317d6205e0ec9db4c5bc5acf4e70c9af7a052",),
    "joint": ("4621cda4df3426cd6898ace624432b148ea17f77e5374b2cd5be9b22812c2ca5",),
    "c512": ("67ed1c4e3865a3435c2f618491d4420f517235e1b5d66a9edb05f8fb393a0d00",),
    "pre_k8": ("45296adf5e43d9caf857a1a292eb162224c5125c26bad13e0a4c986d2121716e",),
    "post_k8": ("e3362da7c8d499e99b7a0ab86c1ea3d95d0127aff2883a826f9e939b7f04acb9",),
    "post_head": ("8444a14eda43e1b5ea2409a28c1066ab60b1e8a1684bc522fa0908a5e52aec4c",),
    "game_history": ("6e4f73ce30a29ffe3366abe558dbb1ef204726a10a1facab2487f1ff7d48e1f0",),
    "spill_screen": ("74ce04f648c09c851d7f169332bc35110586d2c6632aa904d4e351289027798d",),
}
STATIC_RTZ_CAST_AUDIT = {
    "date": "2026-09-30", "cpu_source_only": True,
    "keyword": 'fp_downcast_rounding="rtz"',
    "core": {
        "path": "G:/SteamLibrary/steamapps/common/Resident Evil Village BIOHAZARD VILLAGE/"
                "nr-runtime/toolchain/triton/language/core.py",
        "sha256": "82a2de7d540cd9aab29a71b12b53f32ecd939ea8e271b454c1452a7464556555"},
    "semantic": {
        "path": "G:/SteamLibrary/steamapps/common/Resident Evil Village BIOHAZARD VILLAGE/"
                "nr-runtime/toolchain/triton/language/semantic.py",
        "sha256": "44b53bd2873a5e0b1a4cfda3b9502f7127bd4232e522df67c2acd94e99475e61"},
    "intel": {
        "path": "G:/SteamLibrary/steamapps/common/Resident Evil Village BIOHAZARD VILLAGE/"
                "nr-runtime/toolchain/triton/backends/intel/compiler.py",
        "sha256": "6e94473952ca0fa39f951003a89c319645a7b13c51e38a004598d4995ca543aa"},
    "finding": "tensor.to/cast passes RTZ to create_fp_to_fp; "
               "native generated Intel instruction and bit behavior remain unverified",
}


class UnsupportedNativeRTZ(NotImplementedError):
    """No verified native conversion; never substitute an ordinary half cast."""
    def __init__(self, message, *, category="evidence_unrecognized"):
        super().__init__(message)
        self.category = category


def _loaded(name):
    result = sys.modules.get(name)
    if result is None:
        raise RuntimeError(f"Expected an already-loaded owner module: {name}")
    return result


def _file_row(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _source(role, module, *functions):
    row = _file_row(module.__file__)
    if role in KNOWN_SHA256 and row["sha256"] not in KNOWN_SHA256[role]:
        raise RuntimeError(f"Unreviewed {role} source: {row['path']}")
    for function in functions:
        function = getattr(function, "fn", getattr(function, "__func__", function))
        code = getattr(function, "__code__", None)
        if (code is None or Path(code.co_filename).resolve(strict=True) != Path(row["path"])
                or function.__globals__ is not vars(module)):
            raise RuntimeError(f"{role} callable is outside its actual module/source")
    return row


def _sibling(name):
    """Load our own sibling, also when the suite was imported from a D snapshot."""
    path = Path(__file__).with_name(name + ".py").resolve(strict=True)
    # The runner and scope must compile/dispatch the very same JIT object.
    # Authenticate the full D/E path before reusing a canonical module.
    key = name
    if key in sys.modules:
        result = sys.modules[key]
        if Path(result.__file__).resolve(strict=True) != path:
            raise RuntimeError(f"Candidate import collided with another snapshot: {name}")
        return result
    spec = spec_from_file_location(key, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Missing candidate source: {path}")
    result = module_from_spec(spec)
    sys.modules[key] = result
    try:
        spec.loader.exec_module(result)
    except BaseException:
        sys.modules.pop(key, None)
        raise
    return result


def _closure(function, name):
    function = getattr(function, "__func__", function)
    cells = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
    if name not in cells:
        raise RuntimeError(f"Actual callee lost its owned closure: {name}")
    return cells[name].cell_contents


def _same_method(left, right):
    return (getattr(left, "__self__", None) is getattr(right, "__self__", None)
            and getattr(left, "__func__", left) is getattr(right, "__func__", right))


def _contains_wrapper(current, wanted):
    seen = set()
    while current is not None and id(current) not in seen:
        if _same_method(current, wanted):
            return True
        seen.add(id(current))
        if len(seen) > 64:
            raise RuntimeError("Numeric wrapper chain exceeded its reviewed depth")
        current = getattr(current, "__nr_numeric_original_installed__",
                          getattr(current, "__wrapped__",
                                  getattr(getattr(current, "__func__", None), "__wrapped__", None)))
    return False


def _save_attr(obj, name):
    return name in obj.__dict__, obj.__dict__.get(name)


def _restore_attr(obj, name, saved, expected):
    if obj.__dict__.get(name) is not expected:
        raise RuntimeError(f"Owned attribute changed before restoration: {name}")
    if saved[0]:
        obj.__dict__[name] = saved[1]
    else:
        obj.__dict__.pop(name, None)


def _fingerprint(tensor):
    return (id(tensor), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
            str(tensor.dtype), str(tensor.device), tensor._version)

def _constant_row(fingerprint):
    # The id is explicitly local to this session, never a stable artifact key.
    identity, pointer, shape, stride, dtype, device, version = fingerprint
    return {"session_local_object_id": identity, "data_ptr": pointer, "shape": list(shape),
            "stride": list(stride), "dtype": dtype, "device": device, "version": version}


def _tensor(torch, value, shape, dtype, device, *, contiguous=True):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
            or value.dtype != dtype or value.device != device
            or contiguous and not value.is_contiguous()):
        raise RuntimeError(f"Tensor left its owned 720p contract: expected {shape}/{dtype}")


def _binary_row(kernel):
    binary = getattr(kernel, "kernel", None)
    if not isinstance(binary, (bytes, bytearray, memoryview)) or not binary:
        raise RuntimeError("Actual compiled binary is unavailable")
    if type(getattr(kernel, "n_spills", None)) is not int or kernel.n_spills != 0:
        raise RuntimeError("Actual kernel lacks a verified zero-spill resource gate")
    binary_format = getattr(kernel.metadata, "binary_ext", None)
    if binary_format not in ("spv", "zebin") or kernel.asm.get(binary_format) is not binary:
        matches = [name for name in ("spv", "zebin") if kernel.asm.get(name) is binary]
        if len(matches) != 1:
            raise RuntimeError("Actual loaded conversion payload has no unique binary format")
        binary_format = matches[0]
    return {"actualkernelhash": str(kernel.hash),
            "actualbinary_sha256": hashlib.sha256(bytes(binary)).hexdigest(),
            "binary_format": binary_format,
            "spills": kernel.n_spills, "registers": getattr(kernel, "n_regs", None),
            "shared_bytes": kernel.metadata.shared, "device_isa_verified": False}


def _rtz_binary_gate(kernel):
    """Inspect actual SPIR-V instructions, without a disassembler subprocess.

    This is a direct floating conversion primitive, not integer bit repair.
    SPIR-V evidence does not establish the driver's native machine instruction.
    """
    data = kernel.asm.get("spv")
    if not isinstance(data, (bytes, bytearray)) or len(data) % 4:
        raise UnsupportedNativeRTZ("unsupported native_rtz: no actual SPIR-V rounding evidence")
    words = struct.unpack("<" + "I" * (len(data) // 4), data)
    if len(words) < 5 or words[0] != 0x07230203:
        raise UnsupportedNativeRTZ("unsupported native_rtz: unrecognized actual binary")
    instructions, cursor = [], 5
    while cursor < len(words):
        count, opcode = words[cursor] >> 16, words[cursor] & 0xffff
        if count < 1 or cursor + count > len(words):
            raise UnsupportedNativeRTZ("unsupported native_rtz: malformed actual SPIR-V")
        instructions.append((opcode, words[cursor + 1:cursor + count]))
        cursor += count
    floats, vectors, result_types, rounding = {}, {}, {}, {}
    for opcode, operands in instructions:
        if opcode == 22:  # OpTypeFloat
            floats[operands[0]] = operands[1]
        elif opcode == 23:  # OpTypeVector
            vectors[operands[0]] = operands[1]
        elif opcode == 71 and len(operands) >= 3 and operands[1] == 39:
            rounding[operands[0]] = operands[2]  # FPRoundingMode: RTZ == 1
        if opcode in (12, 61, 79, 80, 81, 82, 83, 115, 124, 129, 131,
                      133, 136, 169, 245) and len(operands) >= 2:
            result_types[operands[1]] = operands[0]

    def width(type_id):
        return floats.get(vectors.get(type_id, type_id))

    conversions, missing_evidence = [], False
    for opcode, operands in instructions:
        if opcode == 124 and width(operands[0]) == 16:
            raise UnsupportedNativeRTZ("unsupported native_rtz: bitcast half reconstruction",
                                       category="contradictory_conversion")
        if opcode == 115 and width(operands[0]) == 16:
            source_width = width(result_types.get(operands[2]))
            if ((source_width is not None and source_width != 32)
                    or (operands[1] in rounding and rounding[operands[1]] != 1)):
                raise UnsupportedNativeRTZ("unsupported native_rtz: FP32->half RTZ was not emitted",
                                           category="contradictory_conversion")
            if source_width is None or operands[1] not in rounding:
                missing_evidence = True
                continue
            conversions.append(operands[1])
    if missing_evidence:
        raise UnsupportedNativeRTZ(
            "native_rtz intermediate source type or rounding decoration is missing")
    if not conversions:
        raise UnsupportedNativeRTZ("unsupported native_rtz: no direct FP32->half RTZ primitive")
    return {"primitive": "OpFConvert FP32->FP16 / FPRoundingMode RTZ",
            "conversion_ids": conversions,
            "spv_sha256": hashlib.sha256(data).hexdigest(),
            "native_device_isa_verified": False, "byte_identity_verified": False}


class ReferenceCounters:
    """No hooks/imports/graph retirement in the default/reference arm."""
    def __init__(self, options, file):
        self.options = dict(options)
        self.file = file
        self.calls, self.capture_calls, self.capture_gates, self.resources = {}, {}, [], {}

    def preflight(self):
        return self.snapshot()

    def validate(self):
        return None

    def snapshot(self):
        return {"options": dict(self.options), "noop": True, "active": False,
                "sources": {"candidate": _file_row(self.file)},
                "calls": {}, "capture_calls": {}, "capture_gates": [],
                "resources": {}, "actualkernelhash": {}, "frame_routes": [],
                "preflight_complete": True, "gpu_numeric_performance": "unverified",
                "reference_byte_identity": "pending Luna"}


class NativeRTZCastProbe:
    """Independent future Luna experiment; production native_rtz stays disabled.

    p = native_rtz_probe(count=65536)
    p.preflight()                 # compile/load only; no input computation
    gpu_half = p.launch(xpu_fp32)  # explicit diagnostic GPU call by Luna
    cpu_reference = p.expected(cpu_fp32)  # signed original reference store
    receipt = p.snapshot()        # JSON only, no tensors/compiled objects

    Access p.jit/p.compiled directly for external ISA inspection. Compare
    finite values by half bit patterns; classify NaN/inf separately, report
    signed zero, subnormal/normal boundaries, overflow and first mismatch.
    The 65536 exact half inputs alone cannot discriminate RTZ from RNE.
    Add FP32 neighbours on BOTH sides of every half midpoint, random uint32
    input bits and signed overshoot. This class generates no test input and
    asserts no ISA support/byte identity. It does not change session hooks.
    """
    def __init__(self, *, device="xpu", count=256):
        if type(count) is not int or count <= 0 or count > 1 << 22:
            raise ValueError("Probe count must be a fixed positive integer <= 2**22")
        self.torch, self.triton = import_module("torch"), import_module("triton")
        self.device = self.torch.device(device)
        if self.device.type != "xpu":
            raise ValueError("Native RTZ probe requires the isolated Intel XPU runtime")
        self.count, self.block = count, BLOCK
        self.kernels = _sibling("post_numeric_suite_720_kernel_v1")
        self.jit = self.kernels.native_rtz_cast_probe
        self.compiled = None
        self.sources = {"probe_host": _file_row(__file__),
                        "probe_kernel": _source("probe_kernel", self.kernels)}
        self.resources, self.launch_hashes, self._constants = {}, {}, ()
        self.status, self.error, self.launches = "not_attempted", None, 0
        self.thread = get_ident()
        self._frozen = (self.device, self.count, self.block, self.jit)
        self.rounding_evidence = {"status": "not_attempted"}
        self.contracts = {f"{self.jit.fn.__module__}.{self.jit.fn.__name__}": (("OUT",), ())}

    def validate(self):
        if (get_ident() != self.thread
                or (self.device, self.count, self.block, self.jit) != self._frozen):
            raise RuntimeError("Tiny RTZ probe owner/configuration changed")
        for tensor, fingerprint in self._constants:
            if _fingerprint(tensor) != fingerprint:
                raise RuntimeError("Tiny RTZ probe preflight buffers changed")

    def preflight(self):
        """Compile/load the real direct-cast probe; NEVER enables production."""
        self.validate()
        if self.torch.xpu.is_current_stream_capturing():
            raise RuntimeError("RTZ probe must be screened before capture")
        if self.compiled is not None:
            return self.snapshot()
        try:
            core = import_module("triton.language.core")
            semantic = import_module("triton.language.semantic")
            intel = import_module("triton.backends.intel.compiler")
            screen = import_module("spill_preflight_v1")
            for role, module in (("triton_cast", core), ("triton_semantic", semantic),
                                 ("triton_intel", intel), ("spill_screen", screen)):
                self.sources[role] = _source(role, module)
            if "fp_downcast_rounding" not in __import__("inspect").signature(core.tensor.to).parameters:
                self.status = "actual_api_missing"
                raise UnsupportedNativeRTZ("Actual loaded Triton cast has no rounding API")
            with self.torch.inference_mode(False):
                self._input = self.torch.empty((self.count,), dtype=self.torch.float32, device=self.device)
                self._output = self.torch.empty((self.count,), dtype=self.torch.float16, device=self.device)
                self._constants = tuple((t, _fingerprint(t)) for t in (self._input, self._output))
                args = (self._input, self._output, self.count, self.block)
                _, compiled, report = screen.select(
                    self.jit, [(self.block,)], lambda _: args,
                    lambda _: (self.triton.cdiv(self.count, self.block),),
                    num_warps=4, num_stages=1, enable_fp_fusion=False)
                self.resources = _binary_row(compiled)
                self.resources["resource_gate"] = report
                self.compiled = compiled  # strongly retained for external ISA inspection
            self.status = "compiled_loaded_diagnostic_only"
            try:
                evidence = _rtz_binary_gate(self.compiled)
                self.rounding_evidence = {"status": "direct_spirv_rtz_verified", **evidence}
            except UnsupportedNativeRTZ as exc:
                # A compiled probe is still available for numerical/ISA audit;
                # lack of recognized SPIR-V proof does NOT imply no backend support.
                self.rounding_evidence = {"status": "unrecognized_or_missing_binary_evidence",
                                          "reason": str(exc)}
            return self.snapshot()
        except BaseException as exc:
            if self.status != "actual_api_missing":
                self.status = "compile_or_resource_gate_failed"
            self.error = f"{type(exc).__name__}: {exc}"
            raise

    def launch(self, value):
        """Diagnostic output tensor only; caller supplies initialized FP32 input."""
        self.validate()
        if self.compiled is None:
            raise RuntimeError("Call the independent RTZ probe preflight before diagnostic launch")
        _tensor(self.torch, value, (self.count,), self.torch.float32, self._input.device)
        output = self.torch.empty_like(value, dtype=self.torch.float16)
        args = (value, output, self.count, self.block)
        grid = (self.triton.cdiv(self.count, self.block),)
        screened = self.jit.warmup(*args, grid=grid, num_warps=4, num_stages=1, enable_fp_fusion=False)
        if screened is not self.compiled or str(screened.hash) != self.resources["actualkernelhash"]:
            raise RuntimeError("Diagnostic RTZ input missed the screened specialization")
        registry = import_module("quantization_dataflow_v1").CONTRACTS
        conflicts = [n for n in self.contracts if n in registry and registry[n] != self.contracts[n]]
        if conflicts:
            raise RuntimeError("RTZ probe contract belongs to a different implementation")
        added = [n for n in self.contracts if n not in registry]
        registry.update(self.contracts)
        try:
            compiled = self.jit[grid](*args, num_warps=4, num_stages=1, enable_fp_fusion=False)
            if compiled is not self.compiled or str(compiled.hash) != self.resources["actualkernelhash"]:
                raise RuntimeError("Diagnostic RTZ launched a different actual binary")
            self.launch_hashes["probe"] = str(compiled.hash)
            self.launches += 1
            return output
        finally:
            for name in added:
                if registry.get(name) != self.contracts[name]:
                    raise RuntimeError("RTZ probe contract changed before restoration")
                registry.pop(name)

    def expected(self, cpu_float32):
        """CPU REFERENCE comparator, exact original signed store_half_rz algorithm.

        Integer correction here is solely the reference oracle. It never runs
        in the native kernel or production candidate and is not claimed native.
        No GPU .cpu() transfer/synchronization is hidden in this method.
        """
        if (not isinstance(cpu_float32, self.torch.Tensor) or cpu_float32.device.type != "cpu"
                or cpu_float32.dtype != self.torch.float32 or tuple(cpu_float32.shape) != (self.count,)):
            raise ValueError("Provide a CPU FP32 vector with this probe's fixed count")
        value = cpu_float32.float()
        rounded = value.half()
        return (rounded.contiguous().view(self.torch.int16) -
                (rounded.float().abs() > value.abs()).to(self.torch.int16)).view(self.torch.float16)

    def snapshot(self):
        return deepcopy({
            "probe": "independent native FP32->FP16 RTZ cast", "status": self.status,
            "production_native_rtz_enabled": False,
            "jit": f"{self.jit.fn.__module__}.{self.jit.fn.__name__}",
            "params": {"shape": [self.count], "input_dtype": "torch.float32",
                       "output_dtype": "torch.float16", "device": str(self.device),
                       "resolved_device": str(self._input.device) if self.compiled is not None else None,
                       "N": self.count, "B": self.block,
                       "grid": [self.triton.cdiv(self.count, self.block)],
                       "num_warps": 4, "num_stages": 1, "enable_fp_fusion": False,
                       "rounding": "rtz", "clamp": False},
            "sources": self.sources, "resources": self.resources,
            "actualkernelhash": self.launch_hashes,
            "rounding_evidence": self.rounding_evidence, "launches": self.launches,
            "error": self.error, "device_isa_verified": False,
            "expected_reference": {
                "api": "probe.expected(cpu_float32)", "origin": "nr_backend.post.store_half_rz",
                "original_source_sha256": KNOWN_SHA256["post"][0],
                "comparison": "finite signed half bits; NaN/Inf classification separately",
                "exact_half_domain_alone_is_insufficient": True,
                "numeric_comparison_completed": False},
            "constants": [_constant_row(fp) for _, fp in self._constants],
            "contracts": self.contracts})


def native_rtz_probe(*, device="xpu", count=256):
    """Return a separate tiny diagnostic; no modes/model/session hook is required."""
    return NativeRTZCastProbe(device=device, count=count)


class _Owned720:
    """Shared lifecycle/receipt implementation used only by these two new files."""
    def __init__(self, modes, options, file, marker):
        self.torch = import_module("torch")
        self.triton = import_module("triton")
        self.modes, self.session = modes, modes.session
        if self.session is None:
            raise RuntimeError("Select a fresh session before installing the 720p scope")
        self.stack = self.session._stack
        self.model, self.provider, self.graph = self.stack.model, self.stack.provider, self.stack.graph
        self.geometry, self.thread = modes.geometry, get_ident()
        self._options = tuple(sorted(options.items()))
        self._frozen_options = self._options
        self.identity = marker + ":" + json.dumps(dict(self._options), sort_keys=True)
        self._frozen_identity = self.identity
        self.marker, self.file = marker, file
        if marker in self.session.__dict__:
            raise RuntimeError("This candidate already owns the session")
        self.sources = {}
        modules = {
            "post": _loaded("nr_backend.post"), "sigmoid": _loaded("nr_backend.sigmoid"),
            "front": _loaded("nr_backend.front"), "temporal": _loaded("nr_backend.temporal"),
            "sampling": _loaded("nr_backend.sampling"), "policy": _loaded("nr_backend.unround_policy"),
            "noise": _loaded("nr_backend.noise"), "execution": _loaded("nr_backend.execution"),
            "fused_front": _loaded("fused_dynamic_front_v1"),
            "joint": _loaded("c512_k8_joint_scope_720_v1"),
            "c512": _loaded("c512_qkv_library_16_v1"),
            "pre_k8": _loaded("native_k8_fp16_v1"), "post_k8": _loaded("native_k8_active_720_v1"),
            "post_head": _loaded("post_attention_k8_combined_v1"),
            "capture_body": _loaded("capture_body_v1"),
            "graph_v1": _loaded("graph_front_v1"), "graph_v5": _loaded("graph_front_v5"),
            "graph_v2": _loaded("graph_front_v2"), "graph_v3": _loaded("graph_front_v3"),
            "graph_v4": _loaded("graph_front_v4"),
            "graph_v6": _loaded("graph_front_v6"),
            "controlled": _loaded(type(self.model).__module__),
            "fullsize": _loaded(type(modes).__module__),
        }
        for role, module in modules.items():
            self.sources[role] = _source(role, module)
        self.modules = modules
        self.sources["common"] = _file_row(__file__)
        self.sources["candidate"] = _file_row(file)
        backend = Path(modules["post"].__file__).resolve().parent
        if (backend.parts[-3:] != ("experimental", "fp8_unround_overlay", "nr_backend")
                or any(Path(modules[n].__file__).resolve().parent != backend for n in
                       ("sigmoid", "front", "noise", "temporal", "policy", "execution", "sampling"))):
            raise RuntimeError("Expected one owned fast overlay backend; exact backend is forbidden")
        if type(self.graph) is not modules["graph_v6"].GraphFront:
            raise RuntimeError("Unrecognized body graph owner")
        self.post, self.sigmoid, self.noise = self.model.post, self.model.sigmoid, self.model.noise
        front_owners = [c for c in self.stack.components
                        if type(c) is modules["fused_front"].FusedFront]
        if len(front_owners) != 1 or front_owners[0].noise is not self.noise:
            raise RuntimeError("Expected the unique owned FusedFront and model noise")
        self.front_owner = front_owners[0]
        self.dense = self.provider.__dict__.get("dense")
        if (getattr(self.dense, "__self__", None) is not self.provider
                or getattr(getattr(self.dense, "__func__", None), "__module__", None)
                != modules["joint"].__name__):
            raise RuntimeError("Expected the actual joint C512/pre-K8 callee")
        _source("joint", modules["joint"], self.dense)
        self.delegate = _closure(self.dense, "delegated_dense")
        if (getattr(self.delegate, "__self__", None) is not self.provider
                or self.delegate.__func__.__module__ != modules["c512"].__name__):
            raise RuntimeError("Expected the sixteen-owner C512 library delegate")
        _source("c512", modules["c512"], self.delegate)
        self.c512_calls, self.k8_calls = modes.c512_library_calls, modes.native_k8_calls
        self.c512_targets = _closure(self.delegate, "targets")
        self.post_k8 = modules["post_head"]._contiguous_cropped_head
        _source("joint", modules["joint"], self.post_k8)
        if (_closure(self.delegate, "calls") is not self.c512_calls
                or _closure(self.dense, "k8_calls") is not self.k8_calls
                or _closure(self.post_k8, "k8_calls") is not self.k8_calls):
            raise RuntimeError("C512/K8 receipts do not belong to the actual dispatches")
        self.weights = []
        for family in ("encoder512", "decoder512"):
            blocks = getattr(self.model, family)
            if len(blocks) != 8:
                raise RuntimeError("Expected eight blocks in each C512 family")
            for index, block in enumerate(blocks):
                weight, label = block.attention.qkv, f"{family}_{index}"
                target = self.c512_targets.get(id(weight))
                if (target is None or target[0] is not weight or target[1] != label
                        or tuple(weight.shape) != (512, 1536) or weight.dtype != self.torch.float16):
                    raise RuntimeError(f"C512 ownership mismatch: {label}")
                self.weights.append((block.attention, weight, label))
        if (len(self.c512_targets) != 16 or set(self.c512_calls) != {n for _, _, n in self.weights}
                or set(self.k8_calls) != {"pre", "post"}):
            raise RuntimeError("C512/K8 target sets changed")
        self.pre_weight, self.head_weight = self.model.pre.front_weight, self.post.head_weight
        self.device = self.head_weight.device
        if self.device.type != "xpu":
            raise RuntimeError("Actual720 owner must use XPU")
        self.weight_fingerprints = {label: _fingerprint(w) for _, w, label in self.weights}
        self.weight_fingerprints["pre"] = _fingerprint(self.pre_weight)
        self.weight_fingerprints["post"] = _fingerprint(self.head_weight)
        self.post_buffers = {name: _fingerprint(t) for name, t in self.post.named_buffers()}
        self.head_method = self.post.forward_head
        self.graph_forward, self.graph_build, self.graph_capture = (
            self.graph.forward, self.graph._build, self.graph._capture)
        self.model_forward = self.model.forward
        self.known_callees = [
            (modules["post"].ResetPostBlock, "forward", modules["post"].ResetPostBlock.forward),
            (modules["sigmoid"].NativeSigmoidTable, "forward", modules["sigmoid"].NativeSigmoidTable.forward),
            (modules["fused_front"].FusedFront, "apply", modules["fused_front"].FusedFront.apply),
            (modules["capture_body"], "forward_front", modules["capture_body"].forward_front),
            (modules["post"], "fma32", modules["sampling"].fma32),
            (modules["graph_v1"].GraphFront, "forward", modules["graph_v1"].GraphFront.forward),
            (modules["graph_v2"].GraphFront, "forward", modules["graph_v2"].GraphFront.forward),
            (modules["graph_v4"].GraphFront, "_build", modules["graph_v4"].GraphFront._build),
            (modules["graph_v5"].GraphFront, "_capture", modules["graph_v5"].GraphFront._capture),
            (modules["graph_v6"].GraphFront, "close", modules["graph_v6"].GraphFront.close),
        ]
        for role, function in (("post", modules["post"].ResetPostBlock.forward),
                               ("sigmoid", modules["sigmoid"].NativeSigmoidTable.forward),
                               ("fused_front", modules["fused_front"].FusedFront.apply),
                               ("capture_body", modules["capture_body"].forward_front)):
            _source(role, modules[role], function)
        self.calls, self.capture_calls, self.resources, self.actualkernelhash = {}, {}, {}, {}
        self.capture_gates, self.frame_routes, self._compiled = [], [], {}
        self._contracts, self._contract_store = {}, None
        self._live, self._in_frame, self._preflight_complete = False, False, False
        self._front_event = None
        self._history_codes = {}
        self._serial_idle_slots = tuple(
            (obj, name, _save_attr(obj, name)) for obj, name in (
                (self.model, "forward"), (self.model, "_forward_front"),
                (self.front_owner, "apply"), (self.post, "forward"),
                (self.sigmoid, "forward")))
        self._serial_reference_warp = modules["temporal"].warp_history_normalized
        self._serial_binding = None
        self._freeze_history_sources()
        self._guard(fresh=True)

    @property
    def options(self):
        return dict(self._options)

    def _guard(self, *, fresh=False, dispatch=False, sources=False):
        self.session._ready()
        if get_ident() != self.thread:
            raise RuntimeError(
                f"Post/front numeric CPU thread owner changed: expected={self.thread}, current={get_ident()}")
        self._guard_fixed(fresh=fresh, dispatch=dispatch, sources=sources)

    def _guard_fixed(self, *, fresh=False, dispatch=False, sources=False):
        from replay_lifecycle_audit_rest_720_v1 import live_guard
        if not fresh and not sources and live_guard(self, self._lifecycle_label(), check_thread=False):
            if dispatch and (not self._live or not self._in_frame or not self._preflight_complete
                    or self.modules['policy'].ENABLED != self.modules['policy'].FAMILIES
                    or self.modules['execution'].current_arithmetic_backend() != 'triton'):
                raise RuntimeError('Candidate requires the owned unrounded Triton frame scope')
            return
        """Original non-thread gates; never skip them during explicit handoff."""
        m, g = self.modes, self.graph
        if (m.session is not self.session
                or self.session._stack is not self.stack or self.stack.model is not self.model
                or self.stack.graph is not g or self.stack.provider is not self.provider
                or m.geometry is not self.geometry or not m.controlled or m.height != 720
                or tuple(m.source) != SIZE or m.variant != "unrounded"
                or not m.c512_qkv_library_720 or not m.native_k8_720
                or tuple(self.geometry.source) != SIZE or self.geometry.mode.model != SIZE
                or self.geometry.mode.active != SIZE or self.geometry.mode.internal != INTERNAL
                or self.geometry.mode.inset != (0, 0)
                or tuple(self.model.PADDED_SIZES.get(SIZE, ())) != INTERNAL
                or g.closed or g.model is not self.model or g.arithmetic is not self.provider
                or self.provider.mode != "fp16_xmx"):
            raise RuntimeError("Candidate left its selected actual720 unrounded graph session")
        if (self.model.post is not self.post or self.model.sigmoid is not self.sigmoid
                or self.model.noise is not self.noise or self.front_owner.noise is not self.noise
                or self.provider.__dict__.get("dense") is not self.dense
                or self.modules["post_head"]._contiguous_cropped_head is not self.post_k8
                or m.c512_library_calls is not self.c512_calls or m.native_k8_calls is not self.k8_calls
                or _closure(self.dense, "delegated_dense") is not self.delegate
                or _closure(self.delegate, "targets") is not self.c512_targets
                or len(self.c512_targets) != 16
                or not _same_method(self.post.forward_head, self.head_method)
                or self.model.pre.front_weight is not self.pre_weight or self.post.head_weight is not self.head_weight):
            raise RuntimeError("Candidate owned module/weight/callee changed")
        from numeric_model_forward_720_v1 import require_forward
        require_forward(self.model.forward, self.model_forward, self.model, self.session)
        if (not _same_method(self.graph.forward, self.graph_forward)
                or not _same_method(self.graph._build, self.graph_build)
                or not _same_method(self.graph._capture, self.graph_capture)):
            raise RuntimeError("Actual body/front model callables changed")
        for owner, name, function in self.known_callees:
            if getattr(owner, name) is not function:
                raise RuntimeError(f"Authenticated reference callee was globally replaced: {name}")
        for owner, weight, label in self.weights:
            target = self.c512_targets.get(id(weight))
            if (owner.qkv is not weight or target is None or target[0] is not weight
                    or target[1] != label or _fingerprint(weight) != self.weight_fingerprints[label]):
                raise RuntimeError(f"Owned C512 weight changed: {label}")
        if (_fingerprint(self.pre_weight) != self.weight_fingerprints["pre"]
                or _fingerprint(self.head_weight) != self.weight_fingerprints["post"]
                or {n: _fingerprint(t) for n, t in self.post.named_buffers()} != self.post_buffers):
            raise RuntimeError("Owned pre/post buffers changed")
        if self.identity != self._frozen_identity or self._options != self._frozen_options:
            raise RuntimeError("Candidate immutable flags identity changed")
        if self._live:
            if (self.session.__dict__.get(self.marker) is not self
                    or not _contains_wrapper(self.session._installed, self._scope_wrapper)
                    or self._scope_wrapper.__nr_numeric_original_installed__ is not self._original_installed):
                raise RuntimeError("Candidate was removed from the layered session")
            if any(self._contract_store.get(name) != value for name, value in self._contracts.items()):
                raise RuntimeError("Candidate top-level JIT output contracts changed")
        if fresh and (g.entries or g.replays or m._session_frames
                      or self.model._previous is not None or self.model._next_seed != 0):
            raise RuntimeError("A fresh uncaptured session is required")
        if dispatch and (not self._live or not self._in_frame
                         or not self._preflight_complete
                         or self.modules["policy"].ENABLED != self.modules["policy"].FAMILIES
                         or self.modules["execution"].current_arithmetic_backend() != "triton"):
            raise RuntimeError("Candidate requires preflight and the owned unrounded Triton frame scope")
        if sources:
            for role, row in self.sources.items():
                if _file_row(row["path"]) != row:
                    raise RuntimeError(f"Actual source changed after candidate construction: {role}")

    def _lifecycle_label(self):
        return 'post' if isinstance(self, PostNumericCounters) else 'front'

    def _validate_replay(self):
        return self.validate()

    def _require_serial_idle(self, owner):
        for child in owner.children.values():
            if any(getattr(child, name, False) for name in
                   ("_in_frame", "in_frame", "_in_scope", "_local", "_post_active")):
                raise RuntimeError("Post/front serial handoff refused: an active child frame/hook is running")
        if self.torch.xpu.is_current_stream_capturing():
            raise RuntimeError("Post/front serial handoff refused during XPU stream capture")
        if (self.front_owner.__dict__.get("__nr_front_route_dispatch_720_v1__") is not None
                or self.session.__dict__.get("_numeric_model_forward_registry_720")
                or self.modules["temporal"].warp_history_normalized is not self._serial_reference_warp
                or any((name in obj.__dict__) != saved[0]
                       or obj.__dict__.get(name) is not saved[1]
                       for obj, name, saved in self._serial_idle_slots)):
            raise RuntimeError("Post/front serial handoff refused: temporary front/model/post/sampler hooks are active or changed")
        for _, name, _ in owner.options.selected_scopes():
            if getattr(_loaded(name), "_IN_FLIGHT", None) is not None:
                raise RuntimeError("Post/front serial handoff refused: a global child frame is active")

    def _transfer_serial_thread(self, owner, *, serial_guard, previous_thread):
        """Actual suite/process-only child transaction; no stream binding or JIT.

        The suite moves every selected child before final validate_context(),
        and must roll back all moved children if a later child/validation fails.
        This transaction restores only this child's thread and retires on error.
        """
        old_thread, old_binding = self.thread, self._serial_binding
        try:
            suite = _loaded("numeric_cleanup_suite_720_v1")
            caller = sys._getframe(1)
            if (type(owner) is not suite.NumericCleanupCounter
                    or caller.f_globals is not vars(suite)
                    or caller.f_code is not suite.NumericCleanupCounter.transfer_serial_thread.__code__
                    or caller.f_locals.get("self") is not owner
                    or caller.f_locals.get("serial_guard") is not serial_guard
                    or caller.f_locals.get("previous_thread") != previous_thread
                    or getattr(self.modes, "_numeric_cleanup_owner", None) is not owner
                    or owner.modes is not self.modes or owner.session is not self.session
                    or owner.graph is not self.graph):
                raise RuntimeError("Post/front serial handoff requires its actual owned suite caller")
            adapter = caller.f_locals.get("game_adapter")
            host = getattr(adapter, "host", None)
            if (adapter is None or adapter is not sys.modules.get("cyberpunk_nr_adapter")
                    or host is None or host is not sys.modules.get("nr_game_pre_xess_host")
                    or getattr(host, "_modes", None) is not self.modes or getattr(host, "_failed", False)
                    or type(serial_guard) is not _SERIAL_GUARD_TYPE
                    or serial_guard is not getattr(adapter, "_process_serial_lock", None)
                    or serial_guard is getattr(adapter, "_settings_lock", None)
                    or not serial_guard._is_owned()):
                raise RuntimeError("Post/front serial handoff requires the actual host and held distinct process RLock")
            if (self._serial_binding is not None
                    and (self._serial_binding[0] is not owner or self._serial_binding[1] is not adapter
                         or self._serial_binding[2] is not serial_guard)):
                raise RuntimeError("Post/front serial handoff suite/adapter/process guard changed after binding")
            if type(previous_thread) is not int or type(old_thread) is not int or previous_thread != old_thread:
                raise RuntimeError(
                    f"Post/front serial handoff CPU owner mismatch: expected={old_thread}, previous={previous_thread!r}")
            owner._validate_suite_owner()
            selected = owner.options.selected_scopes()
            classes = {"decoder_input": "DecoderInputCounter", "branch_accum": "BranchAccumCounter",
                       "vit": "NumericSuite720Counter", "history": "HistoryNumericCounter720",
                       "post": "PostNumericCounters", "front": "FrontNoiseCounters"}
            if (not owner.options.active or set(owner.children) != {label for label, _, _ in selected}
                    or len({id(child) for child in owner.children.values()}) != len(selected)):
                raise RuntimeError("Post/front serial handoff selected child set changed")
            current_thread, own_labels = get_ident(), []
            for label, name, kwargs in selected:
                child, module = owner.children[label], _loaded(name)
                if (type(child) is not getattr(module, classes[label], None)
                        or child.modes is not self.modes or child.session is not self.session
                        or child.graph is not self.graph
                        or not getattr(child, "active", getattr(child, "_live", False))
                        or getattr(child, "retired", False) or getattr(child, "_retired", False)):
                    raise RuntimeError("Post/front serial handoff lost an exact active child/session owner")
                peer_thread = getattr(child, "thread", previous_thread)
                if type(peer_thread) is not int or peer_thread not in (previous_thread, current_thread):
                    raise RuntimeError("Post/front serial handoff peer CPU thread owner mismatch")
                if child is self:
                    own_labels.append(label)
                    if label not in ("post", "front") or self.options != kwargs:
                        raise RuntimeError("Post/front serial handoff selected child options changed")
                if label in ("post", "front"):
                    marker = "_post_numeric_suite_720_v1" if label == "post" else "_front_noise_native_720_v1"
                    if child.marker != marker or self.session.__dict__.get(marker) is not child:
                        raise RuntimeError("Post/front serial handoff active child marker changed")
            if len(own_labels) != 1:
                raise RuntimeError("Post/front serial handoff child is absent from the selected suite")
            bridge = getattr(host, "_bridge", None)
            if (bridge is None or getattr(host, "_thread", None) != current_thread
                    or bridge.thread != current_thread or bridge.torch is not self.torch
                    or bridge.device != self.device
                    or self.torch.xpu.current_stream().sycl_queue != bridge.stream.sycl_queue):
                raise RuntimeError("Post/front serial handoff requires its bridge stream already bound to this thread")
            self._require_serial_idle(owner)
            self._guard_fixed()
            self._validate_local()
            self.thread = current_thread
            self._validate_replay()
            self._serial_binding = (owner, adapter, serial_guard)
            return current_thread
        except BaseException:
            self.thread, self._serial_binding = old_thread, old_binding
            self.session._failed = True
            if owner is getattr(self.modes, "_numeric_cleanup_owner", None):
                self.session._numeric_cleanup_retired_owner = owner
            raise

    def _hit(self, site):
        self.calls[site] = self.calls.get(site, 0) + 1
        if self.torch.xpu.is_current_stream_capturing():
            self.capture_calls[site] = self.capture_calls.get(site, 0) + 1

    def _freeze_history_sources(self):
        """Setup-only source audit; timed frames consult code identity in memory."""
        live = import_module("nr_game_history_fused")
        self.sources["game_history"] = _source("game_history", live)
        modules = [live]
        for module in tuple(sys.modules.values()):
            file = getattr(module, "__file__", None)
            if file and Path(file).name in (
                    "history_numeric_suite_720_v1.py", "history_value_native_720_v1.py"):
                modules.append(module)
        for module in modules:
            row = _file_row(module.__file__)
            role = "history_source." + module.__name__
            self.sources[role] = row

            def remember(code):
                # dataclasses generate methods with this non-file label.
                # They are not source-backed history dispatches; required
                # runtime callees still pass the separate strict _source gate.
                if code.co_filename == "<string>":
                    return
                if Path(code.co_filename).resolve(strict=True) != Path(row["path"]):
                    return
                self._history_codes[code] = row
                for nested in code.co_consts:
                    if isinstance(nested, CodeType):
                        remember(nested)

            for value in vars(module).values():
                if isinstance(value, FunctionType) and value.__globals__ is vars(module):
                    remember(value.__code__)
                elif isinstance(value, type) and value.__module__ == module.__name__:
                    for method in vars(value).values():
                        function = getattr(method, "__func__", method)
                        if isinstance(function, FunctionType) and function.__globals__ is vars(module):
                            remember(function.__code__)

    def _front_route(self, rgb, previous, options):
        self._guard(dispatch=True)
        if self.torch.xpu.is_current_stream_capturing():
            raise RuntimeError("Dynamic front/seed must remain outside the body graph")
        if (self._front_event is not None or tuple(options.get("padded_size", ())) != INTERNAL
                or options.get("noise_source") is not self.noise):
            raise RuntimeError("Expected exactly one owned dynamic front per frame")
        _tensor(self.torch, rgb, (*SIZE, 3), self.torch.float32, self.device)
        seed = options.get("seed")
        expected_seed = 0 if previous is None else self.model._next_seed
        if type(seed) is not int or not 0 <= seed <= 0xffffffff or seed != expected_seed:
            raise RuntimeError("Dynamic uint32 seed/reset contract changed")
        if previous is not None:
            _tensor(self.torch, previous, (*SIZE, 3), self.torch.float32, self.device)
        front = self.model.__dict__.get("_forward_front")
        if not _same_method(front, self.graph_forward):
            if (getattr(front, "__qualname__", "") !=
                    "GameLiveControlledNR.graph_controls.<locals>.controlled_front"
                    or not _same_method(_closure(front, "original"), self.graph_forward)):
                raise RuntimeError("Unrecognized/eager frontend; the owned body graph is required")
        sampler = self.modules["temporal"].warp_history_normalized
        function = getattr(sampler, "__func__", sampler)
        code = getattr(function, "__code__", None)
        if code is None:
            code = getattr(getattr(type(sampler), "__call__", None), "__code__", None)
        if code is None:
            raise RuntimeError("Unrecognized actual history callee")
        row = self._history_codes.get(code)
        if row is None:
            raise RuntimeError("Unknown history callee code; no source lookup or fallback in frames")
        # Other numeric scopes own their history dispatch; do not wrap or replace it.
        # Its path/hash and successful normalized-front consumption are recorded.
        if row["path"] == self.sources["game_history"]["path"]:
            history_mode = "fused_game"
        else:
            holders = [sampler]
            bound_owner = getattr(function, "__self__", None)
            if bound_owner is not None:
                holders.append(bound_owner)
            if getattr(function, "__closure__", None):
                holders += [cell.cell_contents for cell in function.__closure__]
            if not any(getattr(v, "session", None) is self.session or v is self.session for v in holders):
                raise RuntimeError("History numeric callee does not own this session")
            history_mode = "owned_history_numeric"
        role = "actual_history_callee"
        if role in self.sources and self.sources[role] != row:
            raise RuntimeError("History callee source changed during this scope")
        self.sources[role] = row
        self._front_event = {"frontend": "reset" if previous is None else "normalized_history",
                             "seed": seed, "history_mode": history_mode,
                             "history_callee": row,
                             "history_front_consumed": previous is not None,
                             "sampler_calls_directly_instrumented": False,
                             "noise_mode": getattr(self.front_owner.apply,
                                                   "__nr_numeric_front_noise__", "table")}
        self._hit("frontend." + self._front_event["frontend"])

    @contextmanager
    def _observe_front(self):
        from front_route_dispatch_720_v1 import observe
        with observe(self):
            yield

    def _screen(self, label, jit, args, grid, *, block=BLOCK, num_stages=1,
                generate_native_code=None):
        screen = import_module("spill_preflight_v1")
        self.sources["spill_screen"] = _source("spill_screen", screen, screen.select)
        kwargs = {"num_warps": 4, "num_stages": num_stages, "enable_fp_fusion": False}
        if generate_native_code is not None:
            if type(generate_native_code) is not bool:
                raise ValueError("Native codegen must be an explicit boolean")
            kwargs["generate_native_code"] = generate_native_code
        _, kernel, report = screen.select(
            jit, [(block,)], lambda _: args, lambda _: grid,
            **kwargs)
        row = _binary_row(kernel)
        row["resource_gate"] = report
        row["launch_options"] = {"block": block, "num_warps": 4,
                                 "num_stages": num_stages, "enable_fp_fusion": False}
        if generate_native_code is not None:
            parsed_native = getattr(kernel.metadata, "generate_native_code", False)
            if type(parsed_native) is not bool:
                raise RuntimeError("Actual native codegen metadata is not a boolean")
            row["compile_options"] = {"generate_native_code": generate_native_code}
            row["parsed_compile_options"] = {"generate_native_code": parsed_native}
        self._compiled[label], self.resources[label] = kernel, row
        return kernel

    def _launched(self, label, kernel):
        if label not in self._compiled or kernel is not self._compiled[label]:
            raise RuntimeError(f"Actual launch missed its preflight specialization: {label}")
        from replay_lifecycle_audit_rest_720_v1 import launch_owner
        if not launch_owner(self, label, kernel) and (str(kernel.hash) != self.resources[label]["actualkernelhash"]
                or kernel.n_spills != 0):
            raise RuntimeError(f"Actual binary/resources changed: {label}")
        self.actualkernelhash[label] = str(kernel.hash)

    def _finish_frame(self, before):
        self._guard()
        entries, replays, capture, c512, k8, calls = before
        added = set(self.graph.entries) - entries
        if self._front_event is None or self.graph.replays != replays + 1 or len(added) > 1:
            raise RuntimeError("Frame missed its one actual front/body-graph replay")
        new_rows = [self.graph.entries[key] for key in added]
        expected = self._expected_capture(new_rows)
        delta = {n: self.capture_calls.get(n, 0) - capture.get(n, 0)
                 for n in set(self.capture_calls) | set(capture) | set(expected)}
        c512_delta = {n: self.c512_calls[n] - c512[n] for n in c512}
        k8_delta = {n: self.k8_calls[n] - k8[n] for n in k8}
        expected_calls = self._expected_frame_calls(new_rows)
        call_delta = {n: self.calls.get(n, 0) - calls.get(n, 0)
                      for n in set(self.calls) | set(calls) | set(expected_calls)}
        # Authenticated v5 does two warmups plus one capture for each new entry.
        passed = (all(delta.get(n, 0) == expected.get(n, 0) for n in delta)
                  and all(call_delta[n] == expected_calls.get(n, 0) for n in call_delta)
                  and all(v == 3 * len(added) for v in c512_delta.values())
                  and all(v == 3 * len(added) for v in k8_delta.values()))
        from replay_lifecycle_audit_rest_720_v1 import keep_frame_diagnostics
        if not passed or keep_frame_diagnostics(self):
            gate = {"new_entries": len(added), "capture_calls": delta,
                    "expected_capture_calls": expected, "c512_calls": c512_delta,
                    "frame_calls": call_delta, "expected_frame_calls": expected_calls,
                    "k8_calls": k8_delta, "graph_replay_delta": self.graph.replays - replays,
                    "passed": passed}
            self.capture_gates.append(gate)
            self.frame_routes.append({**self._front_event,
                                      "body": "capture_replay" if added else "replay",
                                      "options": self.options, "forced_mode_effective": passed})
        if not passed:
            raise RuntimeError("Candidate or C512/K8 missed its actual capture sites")

    def _expected_frame_calls(self, entries):
        return {"frontend." + self._front_event["frontend"]: 1}

    def snapshot(self):
        return deepcopy({"options": self.options, "identity": self.identity,
                         "noop": False, "active": self._live, "model": list(SIZE),
                         "internal": list(INTERNAL), "sources": self.sources,
                         "calls": self.calls, "capture_calls": self.capture_calls,
                         "resources": self.resources, "actualkernelhash": self.actualkernelhash,
                         "capture_gates": self.capture_gates, "frame_routes": self.frame_routes,
                         "contracts": self._contracts,
                         "owned_weight_constants": {n: _constant_row(v)
                                                    for n, v in self.weight_fingerprints.items()},
                         "owned_post_constants": {n: _constant_row(v)
                                                  for n, v in self.post_buffers.items()},
                         "preflight_complete": self._preflight_complete,
                         "body_graph_replays": self.graph.replays,
                         "c512_calls": dict(self.c512_calls), "k8_calls": dict(self.k8_calls),
                         "python_replay_increments_numeric_calls": False,
                         "timed_validation": {
                             "disk_io": False, "constant_cpu_readback": False,
                             "extra_gpu_synchronize": False,
                             "included_in_timing": True,
                             "checks": "callable/owner/session/shape/pointer/version/flags; "
                                       "capture/replay counters and JIT warmup cache identity"},
                         "gpu_numeric_performance": "unverified",
                         "reference_byte_identity": "pending Luna",
                         "retired": getattr(self, "_retired", False)})

    def validate(self):
        """No launch/sync, usable before replay and outside temporary frame hooks."""
        try:
            self._guard()
            from replay_lifecycle_audit_rest_720_v1 import local_after_guard
            local_after_guard(self)
        except BaseException:
            self.session._failed = True
            raise

    def _validate_local(self):
        pass

    @contextmanager
    def installed(self):
        old_scope = self.session._installed
        scope_saved = _save_attr(self.session, "_installed")

        @contextmanager
        @wraps(old_scope)
        def scope():
            try:
                self._validate_replay()
                if self._in_frame or not self._preflight_complete:
                    raise RuntimeError("Preflight once before frames; recursive frames are forbidden")
                before = (set(self.graph.entries), self.graph.replays, dict(self.capture_calls),
                          dict(self.c512_calls), dict(self.k8_calls), dict(self.calls))
                self._front_event = None
                with old_scope():
                    self._in_frame = True
                    try:
                        with self._methods(), self._observe_front():
                            yield
                            self._finish_frame(before)
                    finally:
                        self._in_frame = False
            except BaseException:
                self.session._failed = True
                raise

        self._scope_wrapper = scope
        # Preserve contextmanager's __wrapped__; this explicit link names the
        # actual previous installed scope rather than this layer's generator.
        scope.__nr_numeric_original_installed__ = old_scope
        self._original_installed = old_scope
        contracts = import_module("quantization_dataflow_v1")
        self.sources["contract_registry"] = _source("contract_registry", contracts)
        self._contract_store = contracts.CONTRACTS
        if any(name in self._contract_store for name in self._contracts):
            raise RuntimeError("Candidate JIT contract already has an owner")
        self._contract_store.update(self._contracts)
        self.session.__dict__[self.marker] = self
        self.session._installed = scope
        self._live = True
        failure = None
        try:
            yield self
        except BaseException as exc:
            failure = exc
            self.session._failed = True
            raise
        finally:
            problems = []
            self._live, self._retired = False, True
            for obj, name, saved, expected in (
                    (self.session, "_installed", scope_saved, scope),):
                try:
                    _restore_attr(obj, name, saved, expected)
                except BaseException as exc:
                    problems.append(str(exc))
            if self.session.__dict__.get(self.marker) is self:
                self.session.__dict__.pop(self.marker)
            else:
                problems.append("Candidate session marker changed")
            self.session._failed = True
            try:
                # Captured commands outlive restored Python aliases.
                self.graph.close()
            except BaseException as exc:
                problems.append(f"Candidate graph retirement failed: {exc}")
            for name, contract in self._contracts.items():
                if self._contract_store.get(name) == contract:
                    self._contract_store.pop(name)
                else:
                    problems.append(f"JIT contract ownership changed: {name}")
            if problems:
                message = "; ".join(problems)
                if failure is not None:
                    failure.add_note(message)
                else:
                    raise RuntimeError(message)


class PostNumericCounters(_Owned720):
    def __init__(self, modes, *, post_sigmoid, post_store, native_rtz_receipt=None):
        super().__init__(modes, {"post_sigmoid": post_sigmoid, "post_store": post_store},
                         __file__, "_post_numeric_suite_720_v1")
        torch = self.torch
        if (type(self.post) is not self.modules["post"].ResetPostBlock
                or type(self.sigmoid) is not self.modules["sigmoid"].NativeSigmoidTable
                or "forward" in self.post.__dict__ or "forward" in self.sigmoid.__dict__):
            raise RuntimeError("The owned original post/sigmoid forwards are required")
        self.kernels = _sibling("post_numeric_suite_720_kernel_v1")
        self.sources["kernels"] = _source("candidate_kernels", self.kernels)
        self.values = self.sigmoid.values
        _tensor(torch, self.values, (65536,), torch.float32, self.device)
        self.table_fingerprint = _fingerprint(self.values)
        self.blend_scale = self.model.blend_scale
        _tensor(torch, self.blend_scale, (), torch.float16, self.device)
        self.blend_fingerprint = _fingerprint(self.blend_scale)
        self.calls = dict.fromkeys(("post.reset", "post.history", "sigmoid.native",
                                   "store.reset", "store.history"), 0)
        self.capture_calls = dict.fromkeys(self.calls, 0)
        self.rtz_status = {"status": "not_requested" if post_store != "native_rtz"
                           else "pending_production_evidence",
                           "byte_identity_verified": False, "device_isa_verified": False}
        self.native_rtz_receipt = native_rtz_receipt
        self._configured_rtz_receipt = native_rtz_receipt
        self._rtz_approval = self._rtz_approval_frozen = None
        self._rtz_compiled = self._rtz_payloads = None
        self._rtz_kwargs = self._rtz_kwargs_frozen = None
        self.rtz_status["static_cast_audit"] = STATIC_RTZ_CAST_AUDIT
        self.rtz_status["independent_probe_api"] = "native_rtz_probe(device='xpu', count=256)"
        self._post_active, self._post_branch, self._return_float32 = False, None, False
        for jit, selected in (
                (self.kernels.sigmoid_half_to_float, post_sigmoid == "native"),
                (self.kernels.store_float_to_half, post_store != "reference")):
            if selected:
                self._contracts[f"{jit.fn.__module__}.{jit.fn.__name__}"] = (("OUT",), ())

    def _post_guard(self):
        self._guard(dispatch=True)
        from replay_lifecycle_audit_rest_720_v1 import local_after_guard
        local_after_guard(self)

    def _validate_local(self):
        from replay_lifecycle_audit_rest_720_v1 import local_guard
        if local_guard(self):
            return
        if (self.sigmoid.values is not self.values or _fingerprint(self.values) != self.table_fingerprint
                or self.model.blend_scale is not self.blend_scale
                or _fingerprint(self.blend_scale) != self.blend_fingerprint):
            raise RuntimeError("Owned sigmoid table/blend scale changed")
        if self.native_rtz_receipt != self._configured_rtz_receipt:
            raise RuntimeError("Native RTZ evidence configuration changed after setup")
        if self.options["post_store"] == "native_rtz" and self._preflight_complete:
            if (self._rtz_approval is None or self._rtz_approval is not self._rtz_approval_frozen
                    or self._rtz_kwargs is None or self._rtz_kwargs is not self._rtz_kwargs_frozen):
                raise RuntimeError("Native RTZ approved evidence owner changed")
            for label in ("store.reset", "store.history"):
                kernel = self._compiled[label]
                if (kernel is not self._rtz_compiled[label]
                        or kernel.kernel is not self._rtz_payloads[label]
                        or str(kernel.hash) != self._rtz_approval.production[label]["actualkernelhash"]
                        or type(getattr(kernel.metadata, "generate_native_code", False)) is not bool
                        or getattr(kernel.metadata, "generate_native_code", False) !=
                           self._rtz_kwargs[label]["generate_native_code"]
                        or kernel.n_spills != 0):
                    raise RuntimeError("Native RTZ production binary changed after approval")
        if self._in_frame:
            if (self.post.__dict__.get("forward") is not self._post_replacement
                    or self.options["post_sigmoid"] == "native"
                    and self.sigmoid.__dict__.get("forward") is not self._sigmoid_replacement):
                raise RuntimeError("Owned post/sigmoid frame hooks changed")
        elif "forward" in self.post.__dict__ or "forward" in self.sigmoid.__dict__:
            raise RuntimeError("Post/sigmoid has an unrecognized persistent forward override")

    def preflight(self):
        """Explicit future GPU compile/load gate; no kernel is launched here."""
        try:
            self._guard(fresh=True, sources=True)
            if not self._live or self.torch.xpu.is_current_stream_capturing():
                raise RuntimeError("Preflight must run inside installed(), before capture")
            if self._preflight_complete:
                return self.snapshot()
            self._freeze_history_sources()
            if self.options["post_store"] == "native_rtz" and self.native_rtz_receipt is None:
                from native_rtz_evidence_720_v1 import PendingRTZEvidence
                self.rtz_status["status"] = "pending_production_evidence"
                raise PendingRTZEvidence("Native RTZ requires the actual production store evidence receipt")
            core = _loaded("triton.language.core")
            semantic = import_module("triton.language.semantic")
            compiler = import_module("triton.backends.intel.compiler")
            self.sources["triton_cast"] = _source("toolchain_cast", core)
            self.sources["triton_semantic"] = _source("toolchain_semantic", semantic)
            self.sources["triton_intel"] = _source("toolchain_intel", compiler)
            if self.options["post_store"] == "native_rtz":
                from native_rtz_evidence_720_v1 import load_evidence, runtime_device_identity
                self._rtz_device = runtime_device_identity(self.torch, self.device)
                try:
                    approval = load_evidence(
                        self.native_rtz_receipt, source_sha256=self.sources["kernels"]["sha256"],
                        toolchain_sha256={"core": self.sources["triton_cast"]["sha256"],
                                          "semantic": self.sources["triton_semantic"]["sha256"],
                                          "intel": self.sources["triton_intel"]["sha256"]},
                        device=self._rtz_device)
                except BaseException as exc:
                    from native_rtz_evidence_720_v1 import PendingRTZEvidence
                    self.rtz_status.update(status="pending_production_evidence" if isinstance(
                        exc, PendingRTZEvidence) else "evidence_mismatch", reason=str(exc))
                    raise
                self._rtz_kwargs = self._rtz_kwargs_frozen = MappingProxyType({
                    label: approval.launch_kwargs(label)
                    for label in ("store.reset", "store.history")})
            if "fp_downcast_rounding" not in __import__("inspect").signature(core.tensor.to).parameters:
                if self.options["post_store"] == "native_rtz":
                    self.rtz_status["status"] = "attempted_api_failure"
                    raise UnsupportedNativeRTZ("unsupported native_rtz: actual cast has no rounding API")
                if self.options["post_store"] == "rne":
                    raise RuntimeError("Actual cast lacks the explicit RNE rounding API")
            with self.torch.inference_mode(False):
                actual = hashlib.sha256(self.values.detach().cpu().numpy().tobytes()).hexdigest()
                if actual != self.modules["sigmoid"].SIGMOID_SHA256:
                    raise RuntimeError("Actual owned sigmoid table hash mismatch")
                self.table_sha256 = actual
                h, w = SIZE
                if self.options["post_sigmoid"] == "native":
                    head = self.torch.empty((h, w, 4), dtype=self.torch.float16, device=self.device)
                    logit = head[..., 3]
                    out = self.torch.empty(SIZE, dtype=self.torch.float32, device=self.device)
                    self._screen("sigmoid", self.kernels.sigmoid_half_to_float,
                                 (logit, out, h, w, w * 4, 4, BLOCK),
                                 (self.triton.cdiv(h * w, BLOCK),))
                if self.options["post_store"] != "reference":
                    value = self.torch.empty((*SIZE, 3), dtype=self.torch.float32, device=self.device)
                    out = self.torch.empty_like(value, dtype=self.torch.float16)
                    rounding = "rtz" if self.options["post_store"] == "native_rtz" else "rtne"
                    for unit, label in ((True, "store.reset"), (False, "store.history")):
                        try:
                            kernel = self._screen(label, self.kernels.store_float_to_half,
                                                  (value, out, h * w * 3, unit, rounding, BLOCK),
                                                  (self.triton.cdiv(h * w * 3, BLOCK),),
                                                  **({"generate_native_code": self._rtz_kwargs[label][
                                                      "generate_native_code"]} if rounding == "rtz" else {}))
                        except BaseException as exc:
                            if rounding != "rtz":
                                raise
                            self.rtz_status.update(status="attempted_compile_failure", reason=str(exc))
                            raise
                        if rounding == "rtz":
                            specification = {"n": h * w * 3, "unit": unit,
                                             "rounding": rounding, "block": BLOCK,
                                             "num_warps": 4, "num_stages": 1,
                                             "enable_fp_fusion": False}
                            try:
                                gate = approval.require_binary(
                                    label, self.resources[label], specialization=specification,
                                    grid=(self.triton.cdiv(h * w * 3, BLOCK),),
                                    input_dtype=str(value.dtype), output_dtype=str(out.dtype),
                                    contiguous=value.is_contiguous() and out.is_contiguous())
                            except BaseException as exc:
                                self.rtz_status.update(status="evidence_mismatch", reason=str(exc))
                                raise
                            # The receipt binds reviewed native ISA to this
                            # exact payload; missing intermediate SPIR-V is not
                            # substituted for a hardware-support verdict.
                            self.resources[label]["rounding_gate"] = gate
                            try:
                                self.resources[label]["spirv_rounding_gate"] = _rtz_binary_gate(kernel)
                            except UnsupportedNativeRTZ as exc:
                                if exc.category == "contradictory_conversion":
                                    self.rtz_status.update(status="evidence_mismatch", reason=str(exc))
                                    raise
                                self.resources[label]["spirv_rounding_gate"] = {
                                    "status": "unrecognized_intermediate_evidence",
                                    "reason": str(exc), "native_isa_receipt_used": True}
                            self.resources[label]["device_isa_verified"] = True
                    if rounding == "rtz":
                        self._rtz_compiled = {n: self._compiled[n] for n in ("store.reset", "store.history")}
                        self._rtz_payloads = {n: self._compiled[n].kernel for n in self._rtz_compiled}
                        self._rtz_approval = self._rtz_approval_frozen = approval
                        self.rtz_status.update(status="production_evidence_verified",
                                               byte_identity_verified=True, device_isa_verified=True,
                                               receipt_path=approval.path, receipt_sha256=approval.sha256)
            self._preflight_complete = True
            return self.snapshot()
        except BaseException:
            self.session._failed = True
            raise

    def _sigmoid(self, logit):
        self._post_guard()
        if not self._post_active or self._post_branch != "history":
            raise RuntimeError("Native sigmoid must be called by the owned history post forward")
        _tensor(self.torch, logit, SIZE, self.torch.float16, self.device, contiguous=False)
        if tuple(logit.stride()) != (SIZE[1] * 4, 4):
            raise RuntimeError("Native sigmoid missed the actual owned four-channel head logit")
        out = self.torch.empty(SIZE, dtype=self.torch.float32, device=self.device)
        args = (logit, out, *SIZE, 1280 * 4, 4, BLOCK)
        grid = (self.triton.cdiv(720 * 1280, BLOCK),)
        screened = self.kernels.sigmoid_half_to_float.warmup(
            *args, grid=grid, num_warps=4, num_stages=1, enable_fp_fusion=False)
        self._launched("sigmoid", screened)
        kernel = self.kernels.sigmoid_half_to_float[grid](
            *args,
            num_warps=4, num_stages=1, enable_fp_fusion=False)
        self._launched("sigmoid", kernel)
        self._hit("sigmoid.native")
        return out

    def _store(self, value, unit):
        self._post_guard()
        if (not self._post_active or self._return_float32
                or unit != (self._post_branch == "reset")):
            raise RuntimeError("Store missed the owned reset/signed-history/float32 post branch")
        _tensor(self.torch, value, (*SIZE, 3), self.torch.float32, self.device)
        label = "store.reset" if unit else "store.history"
        rounding = "rtz" if self.options["post_store"] == "native_rtz" else "rtne"
        out = self.torch.empty_like(value, dtype=self.torch.float16)
        args = (value, out, 720 * 1280 * 3, unit, rounding, BLOCK)
        grid = (self.triton.cdiv(720 * 1280 * 3, BLOCK),)
        kwargs = (self._rtz_kwargs[label] if rounding == "rtz" else
                  {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False})
        screened = self.kernels.store_float_to_half.warmup(
            *args, grid=grid, **kwargs)
        self._launched(label, screened)
        kernel = self.kernels.store_float_to_half[grid](
            *args, **kwargs)
        self._launched(label, kernel)
        self._hit(label)
        return out

    @contextmanager
    def _methods(self):
        post, sigmoid = self.post, self.sigmoid
        saved_post, saved_sigmoid = _save_attr(post, "forward"), _save_attr(sigmoid, "forward")
        original = self.modules["post"].ResetPostBlock.forward
        if (saved_post[0] or saved_sigmoid[0]
                or not _same_method(post.forward, MethodType(original, post))):
            raise RuntimeError("Post numeric scope requires its owned original post forward")
        namespace = dict(original.__globals__)
        if self.options["post_store"] != "reference":
            namespace["store_half_rz"] = lambda value: self._store(value, False)
            namespace["store_unit_half_rz"] = lambda value: self._store(value, True)
        clone = FunctionType(original.__code__, namespace, original.__name__,
                             original.__defaults__, original.__closure__)
        clone.__kwdefaults__ = original.__kwdefaults__

        @wraps(original)
        def forward(module, features, skip, rgb, **kw):
            self._post_guard()
            if (module is not post or kw.get("sigmoid") is not sigmoid
                    or kw.get("blend_scale") is not self.blend_scale):
                raise RuntimeError("Post caller missed its owned sigmoid/blend constants")
            _tensor(self.torch, features, (384, 640, 32), self.torch.float16, self.device)
            _tensor(self.torch, skip, (*INTERNAL, 32), self.torch.float16, self.device)
            _tensor(self.torch, rgb, (*SIZE, 3), self.torch.float32, self.device)
            previous, reciprocal = kw.get("previous"), kw.get("history_reciprocal")
            if previous is not None:
                _tensor(self.torch, previous, (*SIZE, 3), self.torch.float32, self.device)
                _tensor(self.torch, reciprocal, SIZE, self.torch.float32, self.device)
            elif reciprocal is not None:
                raise RuntimeError("Reset post cannot consume a history reciprocal")
            if self._post_active:
                raise RuntimeError("Recursive post forward is forbidden")
            self._post_active = True
            self._post_branch = "reset" if previous is None else "history"
            self._return_float32 = bool(kw.get("return_float32", False))
            try:
                result = clone(module, features, skip, rgb, **kw)
            finally:
                self._post_active, self._post_branch = False, None
            self._hit("post.reset" if previous is None else "post.history")
            return result

        replacement = MethodType(forward, post)
        self._post_replacement = replacement
        post.forward = replacement
        sigmoid_replacement = None
        try:
            if self.options["post_sigmoid"] == "native":
                original_sigmoid = self.modules["sigmoid"].NativeSigmoidTable.forward

                @wraps(original_sigmoid)
                def native(module, logit):
                    if module is not sigmoid:
                        raise RuntimeError("Foreign sigmoid instance")
                    return self._sigmoid(logit)

                sigmoid_replacement = MethodType(native, sigmoid)
                self._sigmoid_replacement = sigmoid_replacement
                sigmoid.forward = sigmoid_replacement
            yield
        finally:
            try:
                if sigmoid_replacement is not None:
                    _restore_attr(sigmoid, "forward", saved_sigmoid, sigmoid_replacement)
            finally:
                _restore_attr(post, "forward", saved_post, replacement)
                self._post_replacement = self._sigmoid_replacement = None

    def _expected_capture(self, entries):
        expected = dict.fromkeys(self.capture_calls, 0)
        for entry in entries:
            branch = "reset" if entry.inputs["previous"] is None else "history"
            expected["post." + branch] += 1
            if branch == "history" and self.options["post_sigmoid"] == "native":
                expected["sigmoid.native"] += 1
            # The graph key's final bool is return_float32; that path has no store.
            if self.options["post_store"] != "reference":
                key = next(k for k, v in self.graph.entries.items() if v is entry)
                if not key[-1]:
                    expected["store." + branch] += 1
        return expected

    def _expected_frame_calls(self, entries):
        expected = super()._expected_frame_calls(entries)
        expected.update({name: 3 * count for name, count in self._expected_capture(entries).items()})
        return expected

    def snapshot(self):
        result = super().snapshot()
        result["native_rtz"] = dict(self.rtz_status)
        result["sigmoid_table_sha256"] = getattr(self, "table_sha256", None)
        result["sigmoid_constant"] = _constant_row(self.table_fingerprint)
        result["blend_scale_constant"] = _constant_row(self.blend_fingerprint)
        result["expected_sites"] = {
            "reset_body": {"post.reset": 1, "sigmoid.native": 0,
                           "store.reset": int(self.options["post_store"] != "reference")},
            "history_body": {"post.history": 1,
                             "sigmoid.native": int(self.options["post_sigmoid"] == "native"),
                             "store.history": int(self.options["post_store"] != "reference")},
            "return_float32_store_calls": 0,
            "actual_history_sampler_counts": "owned by the independent history scope",
        }
        result["theoretical_launches"] = {
            "sigmoid_candidate_per_history_body": int(self.options["post_sigmoid"] == "native"),
            "store_candidate_per_non_float_body": int(self.options["post_store"] != "reference"),
            "launches_saved": "unknown: baseline Torch graph must be counted",
            "tail_layout_fusion": False}
        result["retained_sigmoid_table_bytes"] = 65536 * 4
        return result


@contextmanager
def installed(modes, *, post_sigmoid="table", post_store="reference", native_rtz_receipt=None):
    """Schema-compatible API; preflight()/snapshot() take no required arguments."""
    if post_sigmoid not in ("table", "native"):
        raise ValueError("post_sigmoid must be table or native")
    if post_store not in ("reference", "native_rtz", "rne"):
        raise ValueError("post_store must be reference, native_rtz or rne")
    if native_rtz_receipt is not None and post_store != "native_rtz":
        raise ValueError("A native RTZ receipt requires the explicit native_rtz arm")
    if post_sigmoid == "table" and post_store == "reference":
        yield ReferenceCounters({"post_sigmoid": post_sigmoid, "post_store": post_store}, __file__)
        return
    try:
        candidate = PostNumericCounters(modes, post_sigmoid=post_sigmoid, post_store=post_store,
                                        native_rtz_receipt=native_rtz_receipt)
        with candidate.installed():
            yield candidate
    except BaseException:
        if getattr(modes, "session", None) is not None:
            modes.session._failed = True
        raise
