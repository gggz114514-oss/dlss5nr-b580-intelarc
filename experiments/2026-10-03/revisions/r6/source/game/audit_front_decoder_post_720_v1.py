"""Owned current720 fast implementation; install after combo, before numeric.

This public module is stdlib-only until installed() is entered. Runtime kernels
are preflighted before capture; exact/reference classes are restored on exit.
The main overlay owns forward-registry and capture-gate replacement registration.
"""
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from types import MethodType
from threading import RLock, get_ident
import hashlib
import inspect
import json
import sys
import weakref


GROUP = "front-decoder-post"
BASELINE_IDENTITY = "7cf38a499a25d863db90a49d19c88ab3ad98d32511c72ee54c9efb973db17d30"
MERGES = {"decoder_input": (24, 40, 512), "decoder.0.0": (48, 80, 256),
          "decoder.1.0": (96, 160, 128), "decoder.2.0": (192, 320, 64)}
_ACTIVE = None


def _same(a, b):
    return a is b or (inspect.ismethod(a) and inspect.ismethod(b)
                     and a.__self__ is b.__self__ and a.__func__ is b.__func__)


def _function(value):
    value = getattr(value, "__func__", value)
    return getattr(value, "fn", value)


def _stamp(t):
    return (id(t), t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype,
            t.device, t._version)


def _composed_source_pins(pins, overrides):
    """Main may repin composed dependencies, never the frozen owned payload."""
    modules = dict(pins)
    for name, sha in (overrides or {}).items():
        if (type(name) is not str or type(sha) is not str or len(sha) != 64
                or any(c not in "0123456789abcdef" for c in sha)):
            raise ValueError("Composed source pins require a module name and lowercase SHA256")
        owned = name.startswith("audit_front_decoder_post_") or name in (
            "front_noise_native_720_v1", "front_noise_native_720_kernel_v1")
        if owned and modules.get(name) != sha:
            raise RuntimeError("Main may override composed dependency pins, not owned snapshot bytes")
        modules[name] = sha
    return modules


class Scope:
    def __init__(self, target, *, merge_native=False, decoder_full_k=False,
                 post_entry=False, post_fusion=False, post_tail=False,
                 post_entry_mlp=False, post_native_head=False,
                 post_entry_mode=None,
                 stage_matrices=False, down_channels=(), up_channels=(), pre_average=False,
                 pool_c512_transition=True,
                 native_style=False, front_controls=False, post_sigmoid="table", source_pins=None):
        self.modes = target if hasattr(target, "session") else None
        self.session = target.session if self.modes is not None else target
        self.stack = self.session._stack
        self.model = self.stack.model
        self.decoder = self.stack.decoder_gather
        self.post = self.model.post
        if type(post_sigmoid) is not str or post_sigmoid not in ("table", "native"):
            raise ValueError("Owned post_sigmoid must be table or current native")
        if not post_tail and post_sigmoid != "table":
            raise ValueError("Owned native sigmoid requires post_tail=true; otherwise use Numeric's independent owner")
        self.post_sigmoid = self._frozen_post_sigmoid = post_sigmoid
        post_entry_mlp = post_entry_mlp or post_fusion
        post_native_head = post_native_head or post_fusion
        if post_entry_mode is not None:
            if post_entry_mode not in ("baseline", "native", "native_mlp"):
                raise ValueError("Unknown mutually exclusive post entry mode")
            if post_entry or post_entry_mlp or post_fusion:
                raise ValueError("Use post_entry_mode OR legacy entry booleans, not both")
            post_entry = post_entry_mode == "native"
            post_entry_mlp = post_entry_mode == "native_mlp"
        if post_entry and post_entry_mlp:
            raise ValueError("Native standalone entry and native entry/MLP are mutually exclusive")
        if type(stage_matrices) is not bool or type(pre_average) is not bool:
            raise ValueError("Matrix selection flags must be bools")
        self.down_channels = tuple((32, 64, 128, 256, 512) if stage_matrices else down_channels)
        self.up_channels = tuple((256, 128, 64, 32) if stage_matrices else up_channels)
        if (any(type(c) is not int or c not in (32, 64, 128, 256, 512) for c in self.down_channels)
                or any(type(c) is not int or c not in (32, 64, 128, 256) for c in self.up_channels)
                or len(set(self.down_channels)) != len(self.down_channels) or len(set(self.up_channels)) != len(self.up_channels)):
            raise ValueError("Invalid active720 down/up site tuple")
        pre_average = pre_average or stage_matrices
        stage_matrices = bool(stage_matrices or self.down_channels or self.up_channels or pre_average)
        self.flags = dict(merge_native=merge_native, decoder_full_k=decoder_full_k,
                          post_entry=post_entry, post_fusion=post_fusion,
                          post_tail=post_tail, stage_matrices=stage_matrices,
                          native_style=native_style, front_controls=front_controls,
                          post_entry_mlp=post_entry_mlp, post_native_head=post_native_head,
                          pre_average=pre_average, pool_c512_transition=pool_c512_transition)
        if any(type(v) is not bool for v in self.flags.values()):
            raise ValueError("720 implementation options must be immutable bools")
        if self.stack.provider.mode != "fp16_xmx" or self.stack.graph.entries:
            raise RuntimeError("Requires fresh current720 fp16_xmx before capture")
        if self.modes is not None and (self.modes.height != 720 or self.modes.variant != "unrounded"):
            raise RuntimeError("Owned implementation supports only unrounded720")
        self.torch = import_module("torch")
        policy = import_module("nr_backend.unround_policy")
        if policy.ENABLED != policy.FAMILIES:
            raise RuntimeError("Current720 fast profile requires all-unrounded policy")
        self.kernels = import_module("audit_front_decoder_post_720_kernels_v1")
        self.sources = {}
        self.source_symbols = []
        self.pinfile = Path(__file__).with_name("audit_front_decoder_post_720_pins_v1.json")
        self.pins = json.loads(self.pinfile.read_text(encoding="utf-8"))
        self.pins["modules"] = _composed_source_pins(self.pins["modules"], source_pins)
        self.authenticate(import_module(__name__))
        self.authenticate(self.kernels)
        for dependency in ("native_half_cubic_v1", "fused_swin_core_native_half_v1",
                           "native_half_attention_fma_v1", "nr_backend.triton_attention_weights",
                           "nr_backend.triton_attention_normalize", "nr_backend.triton_fp8"):
            self.authenticate(import_module(dependency))
        self.weights = []
        self.bindings = []
        self.contracts = {}
        self.screens, self.calls, self.capture_calls = {}, {}, {}
        self.preflight_complete = False
        self.thread, self._serial_binding, self._serial_owner = get_ident(), None, None
        self._in_dispatch = False
        self._retired = False
        self._frozen_flags = tuple(sorted(self.flags.items()))
        self._frozen_selection = (self.down_channels, self.up_channels)
        self.graph = self.stack.graph
        self.active = True
        self.front_broker = None
        self._front_broker_binding = None
        self.front_record = None
        self.front_producer_calls = 0
        self.front_controls_consumed = 0
        self.front_direct_writes = 0
        self.original_merge, self.original_input = self.decoder.merge, self.decoder.input
        self.original_head = self.post.forward_head
        head_function = getattr(self.original_head, "__func__", self.original_head)
        self.baseline_entry = inspect.getclosurevars(head_function).nonlocals.get("entry_fn")
        self.modules = dict(self.decoder.modules)
        if set(self.modules) != set(MERGES) | {"decoder.3.0"}:
            raise RuntimeError("Lost the actual five decoder transition owners")
        for name in MERGES:
            m = self.modules[name]
            self.watch_weight(m, "weight")
            self.watch_weight(m, "skip_scale" if name == "decoder_input" else "input_skip_scale")
        for name in ("input_weight", "input_skip_scale"):
            self.watch_weight(self.post, name)
        self.replacement_sites = {
            "decoder_merge": list(MERGES) if merge_native else [],
            "decoder_input": ["decoder_input.input"] if decoder_full_k else [],
            "post_entry": ["post.entry"] if post_entry else [],
            "post_mlp": ["post.body.mlp"] if post_entry_mlp else [],
            "post_head": ["post.attention_project_native_head"] if post_native_head else [],
            "post_tail": ["post.forward"] if post_tail else [],
        }

    def authenticate(self, module):
        path = Path(module.__file__).resolve()
        expected = self.pins["modules"].get(module.__name__)
        if expected is None:
            raise RuntimeError(f"Unauthenticated runtime module: {module.__name__}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"Source pin mismatch: {module.__name__}: {path}")
        self.sources[module.__name__] = {"path": str(path), "sha256": digest}
        for name, symbol in vars(module).items():
            fn = _function(symbol)
            if inspect.isfunction(fn):
                self.source_symbols.append((module, name, symbol, fn, fn.__code__))
        return module

    def watch_weight(self, owner, name):
        value = getattr(owner, name)
        self.weights.append((owner, name, value, _stamp(value)))

    def tensor(self, value, shape, *, dtype=None, positive_stride=False):
        torch = self.torch
        dtype = torch.float16 if dtype is None else dtype
        device = self.post.input_weight.device
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
                or value.dtype != dtype or value.device != device or device.type != "xpu"
                or (not positive_stride and not value.is_contiguous())
                or any(s <= 0 for s in value.stride())):
            raise RuntimeError(f"Changed 720 boundary: expected {shape}/{dtype}")
        return value

    def _guard_fixed(self):
        if (self._retired or not self.active or tuple(sorted(self.flags.items())) != self._frozen_flags
                or (self.down_channels, self.up_channels) != self._frozen_selection
                or self.post_sigmoid != self._frozen_post_sigmoid):
            raise RuntimeError("Owned720 lifecycle/profile changed")
        for module, name, symbol, fn, code in self.source_symbols:
            if getattr(module, name) is not symbol or _function(symbol) is not fn or fn.__code__ is not code:
                raise RuntimeError("Authenticated callable/code changed")
        if any(getattr(o, n) is not t or _stamp(t) != stamp for o, n, t, stamp in self.weights):
            raise RuntimeError("Owned immutable weight changed")
        if self.flags.get("post_tail", False) and self.model.sigmoid is not self.sigmoid_owner:
            raise RuntimeError("Owned post sigmoid owner changed")
        if getattr(self, "front_broker", None) is not getattr(self, "_front_broker_binding", None):
            raise RuntimeError("Owned front destination broker changed")
        if getattr(self, "front_broker", None) is not None:
            broker, acquire, acquire_code, commit, commit_code = self.front_broker
            if (broker.model is not self.model or broker.session is not self.session or broker.graph is not self.graph
                    or not _same(broker.acquire_front_720, acquire) or acquire.__func__.__code__ is not acquire_code
                    or not _same(broker.commit_front_720, commit) or commit.__func__.__code__ is not commit_code):
                raise RuntimeError("Owned front destination broker callable/owner changed")
        if any(not _same(getattr(o, n), f) for o, n, _, _, f in self.bindings):
            raise RuntimeError("Owned actual call route changed")
        if getattr(self, "curve_stamps", None):
            for key, stamp in self.curve_stamps.items():
                if _stamp(self.style_owner.curves[key]) != stamp:
                    raise RuntimeError("Owned lazy curve changed")
        for jit, fn, code in getattr(self, "kernel_symbols", ()):
            if jit.fn is not fn or fn.__code__ is not code:
                raise RuntimeError("Compiled JIT source/callee changed")
        if self.session._closed or self.session._failed:
            raise RuntimeError("Owned720 session is closed/failed")
        return True

    def guard(self):
        if get_ident() != self.thread:
            raise RuntimeError("Owned720 CPU thread owner changed; explicit process-lock transfer required")
        self._guard_fixed()
        return True

    def validate(self, *, fresh=False, sources=False):
        self.guard()
        if not fresh and not sources:
            return True
        for name, row in self.sources.items():
            module = import_module(name)
            if Path(module.__file__).resolve() != Path(row["path"]) or hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() != row["sha256"]:
                raise RuntimeError(f"Owned source changed: {name}")
        return True

    def validate_frame_context(self):
        """Memory-only frame/replay validation; no paths, SHA, JIT or GPU read."""
        return self.guard()

    def bind(self, owner, name, function, *, method=True):
        had = name in owner.__dict__
        previous = owner.__dict__.get(name) if had else getattr(owner, name, None)
        installed = MethodType(function, owner) if method else function
        setattr(owner, name, installed)
        self.bindings.append((owner, name, had, previous, installed))
        # The new owned binding deliberately supersedes an authenticated old
        # provider symbol; guard the installed callable/code at this actual slot.
        function = _function(installed)
        self.source_symbols = [
            (m, n, installed, function, function.__code__)
            if m is owner and n == name and inspect.isfunction(function) else row
            for row in self.source_symbols for m, n, _, _, _ in (row,)]

    def register_contract(self, name, outputs):
        contracts = import_module("quantization_dataflow_v1").CONTRACTS
        key = f"{self.kernels.__name__}.{name}"
        if key in contracts:
            raise RuntimeError(f"Contract collision: {key}")
        value = (tuple(outputs), ())
        contracts[key] = value
        self.contracts[key] = value

    def screen(self, site, jit, args, grid, **options):
        options = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False, **options}
        compiled = jit.warmup(*args, grid=grid, **options)
        compiled._init_handles()
        if type(compiled.n_spills) is not int or compiled.n_spills != 0:
            raise RuntimeError(f"{site}: unknown/nonzero spill metadata: {compiled.n_spills}")
        actual = compiled.kernel
        binaries = [(n, b) for n, b in compiled.asm.items()
                    if n in ("spv", "spirv", "zebin") and isinstance(b, bytes) and b is actual]
        if len(binaries) != 1:
            raise RuntimeError(f"{site}: loaded executable bytes ambiguous")
        kind, binary = binaries[0]
        self.screens[site] = (jit, compiled, grid, options, {
            "kernel_hash": compiled.hash, "binary_kind": kind,
            "binary_sha256": hashlib.sha256(binary).hexdigest(), "binary_bytes": len(binary),
            "spills": compiled.n_spills, "registers": compiled.n_regs,
            "shared_bytes": compiled.metadata.shared, "grid": list(grid),
            "launch_options": options, "source_kernel": f"{jit.fn.__module__}.{jit.fn.__name__}"})
        self.calls[site] = self.capture_calls[site] = 0
        if not hasattr(self, "kernel_symbols"):
            self.kernel_symbols = []
        if not any(saved is jit for saved, _, _ in self.kernel_symbols):
            self.kernel_symbols.append((jit, jit.fn, jit.fn.__code__))

    def launch(self, site, args):
        self.guard()
        if not self.preflight_complete or site not in self.screens:
            raise RuntimeError(f"Unscreened actual dispatch: {site}")
        jit, expected, grid, options, receipt = self.screens[site]
        if self._in_dispatch:
            raise RuntimeError("Reentrant owned kernel dispatch")
        self._in_dispatch = True
        try:
            if jit[grid](*args, **options) is not expected:
                self.session._failed = True
                raise RuntimeError(f"{site}: dispatch missed screened binary {receipt['kernel_hash']}")
        finally:
            self._in_dispatch = False
        self.calls[site] += 1
        if self.torch.xpu.is_current_stream_capturing():
            self.capture_calls[site] += 1
        return expected

    def empty(self, shape, *, dtype=None):
        return self.torch.empty(shape, dtype=dtype or self.torch.float16, device=self.post.input_weight.device)

    def merge_args(self, p, s, scale, out):
        h, w, c = s.shape
        return (p, s, scale, out, h, w, c, *p.stride(), *s.stride(), 256)

    def matrix_args(self, x, weight, out, *, pool=False):
        h, w, n = out.shape
        return (x, weight, out, h * w, weight.shape[0], n, *x.stride(), w, pool, 32, 64, 32)

    def entry_args(self, f, s, out):
        return (f, s, self.post.input_weight, self.post.input_skip_scale, out, 32)

    def preflight(self):
        if self.preflight_complete:
            self.validate_frame_context()
            return self.snapshot()["resources"]
        self.validate(fresh=True, sources=True)
        for name, shape in MERGES.items():
            if self.flags["merge_native"]:
                module = self.modules[name]
                scale = module.skip_scale if name == "decoder_input" else module.input_skip_scale
                p = self.empty((shape[0] // 2, shape[1] // 2, shape[2]))
                s, out = self.empty(shape), self.empty(shape)
                self.screen("merge:" + name, self.kernels.merge, self.merge_args(p, s, scale, out),
                            ((shape[0] * shape[1] * shape[2] + 255) // 256,))
        if self.flags["decoder_full_k"]:
            x, out = self.empty((12, 20, 1024)), self.empty((12, 20, 512))
            self.screen("decoder_input_full_k", self.kernels.matrix,
                        self.matrix_args(x, self.modules["decoder_input"].weight, out), (8, 8))
        if self.flags["post_entry"] and not self.flags["post_entry_mlp"]:
            f, s, out = self.empty((384, 640, 32)), self.empty((768, 1280, 32)), self.empty((776, 1288, 32))
            self.screen("post_entry", self.kernels.entry, self.entry_args(f, s, out),
                        ((776 * 1288 + 31) // 32,))
        self.preflight_extra()
        self.preflight_complete = True
        return self.snapshot()["resources"]

    def preflight_extra(self):
        if self.flags["post_entry_mlp"]:
            f, s = self.empty((384, 640, 32)), self.empty((768, 1280, 32))
            mlp = self.empty((776, 1288, 32))
            self.screen("post_entry_mlp", self.kernels.entry_mlp, self.entry_mlp_args(f, s, mlp),
                        ((776 * 1288 + 31) // 32,))
        if self.flags["post_entry"] or self.flags["post_entry_mlp"] or self.flags["post_native_head"]:
            mlp = self.empty((776, 1288, 32))
            windows = 97 * 161
            qkv = tuple(self.empty((97, 161, 64, 32)) for _ in range(3))
            self.screen("post_qkv", self.qkv._kernel, self.qkv_args(mlp, qkv),
                        (776 * 1288 // 32, 3))
            out = self.empty((720, 1280, 4))
            if self.flags["post_native_head"]:
                self.screen("post_attention_native_head", self.kernels.attention_native_head,
                            self.head_args(mlp, qkv, out), (2, windows))
            else:
                full = self.empty((768, 1280, 32))
                self.screen("post_attention_project_passthrough", self.attention_module._attention_project,
                            self.attention_args(mlp, qkv, full), (2, windows))
                self.screen("post_native_head_passthrough", self.baseline_head._post_half_dot,
                            self.baseline_head_args(full, out), (720 * 1280 // 32,))
            if not self.flags["post_entry"] and not self.flags["post_entry_mlp"]:
                f, s = self.empty((384, 640, 32)), self.empty((768, 1280, 32))
                self.screen("post_entry_baseline_passthrough", self.baseline_entry_kernel._entry_kernel,
                            self.baseline_entry_args(f, s, mlp), ((776 * 1288 + 31) // 32,))
        if self.flags["post_tail"]:
            head, rgb = self.empty((720, 1280, 4)), self.empty((720, 1280, 3), dtype=self.torch.float32)
            previous, reciprocal = self.empty((720, 1280, 3), dtype=self.torch.float32), self.empty((720, 1280), dtype=self.torch.float32)
            for temporal, normalize in ((False, False), (True, False), (True, True)):
                for floating in (False, True):
                    out = self.empty((720, 1280, 3), dtype=self.torch.float32 if floating else self.torch.float16)
                    args = self.tail_args(head, rgb, previous if temporal else None,
                                          reciprocal if normalize else None, out)
                    site = self.tail_site(temporal, normalize, floating, self.post_sigmoid)
                    self.screen(site, self.kernels.tail, args,
                                ((720 * 1280 * 3 + 255) // 256,))
                    self.screens[site][4]["post_sigmoid"] = self.post_sigmoid
                    self.screens[site][4]["post_store"] = "native_rne"
                    if temporal:
                        self.screens[site][4]["sigmoid_source"] = self.sources[
                            "post_numeric_suite_720_kernel_v1" if self.post_sigmoid == "native" else "nr_backend.sigmoid"]
            out = self.empty((720, 1280, 3))
            self.screen("display_native_store", self.kernels.convert, (rgb, out, 256),
                        ((720 * 1280 * 3 + 255) // 256,))
        if self.flags["stage_matrices"]:
            for site, (module, attr, shape) in self.down_sites.items():
                weight = getattr(module, attr)
                out = self.empty((shape[0] // 2, shape[1] // 2, weight.shape[1]))
                variants = [self.empty(shape)]
                sy, sx = getattr(module, "window_shift", (0, 0))
                if sy or sx:
                    parent = self.empty((shape[0] + 2 * sy, shape[1] + 2 * sx, shape[2]))
                    variants.append(parent[sy:sy + shape[0], sx:sx + shape[1]])
                for x in variants:
                    self.screen(self.pool_site(site, x), self.kernels.matrix,
                                self.matrix_args(x, weight, out, pool=True),
                                ((out.shape[0] * out.shape[1] + 31) // 32, (out.shape[2] + 63) // 64))
                    pooled = self.empty((shape[0] // 2, shape[1] // 2, shape[2]))
                    diagnostic = "diagnostic_pool:" + self.pool_site(site, x)
                    self.screen(diagnostic, self.kernels.average, self.average_args(x, pooled),
                                ((pooled.numel() + 255) // 256,))
                    self.screens[diagnostic][4]["execution_condition"] = "precompile; explicit diagnostic only; not normal eager/capture/replay"
            for name, shape in {"decoder.0.0": (24, 40, 512), "decoder.1.0": (48, 80, 256),
                                "decoder.2.0": (96, 160, 128), "decoder.3.0": (192, 320, 64)}.items():
                if self.modules[name].channels not in self.up_channels:
                    continue
                x = self.empty(shape); weight = self.modules[name].weight
                out = self.empty((*shape[:2], weight.shape[1]))
                self.screen("up:" + name, self.kernels.matrix, self.matrix_args(x, weight, out),
                            ((shape[0] * shape[1] + 31) // 32, (weight.shape[1] + 63) // 64))
            if self.flags["pre_average"]:
                x, out = self.empty((768, 1280, 32)), self.empty((384, 640, 32))
                self.screen("pre_average", self.kernels.average, self.average_args(x, out),
                            ((384 * 640 * 32 + 255) // 256,))
        if self.flags["native_style"]:
            n = self.empty((720, 1280, 3)); o = self.empty((720, 1280, 3), dtype=self.torch.float32)
            dummy = self.empty((1,), dtype=self.torch.float32)
            for style in (1, 2):
                for curves in (False, True):
                    for floating in (False, True):
                        out = self.empty((720, 1280, 3), dtype=self.torch.float32 if floating else self.torch.float16)
                        status = self.empty(((720 * 1280 + 255) // 256,), dtype=self.torch.int32)
                        self.screen(self.style_site(style, curves, floating), self.kernels.style,
                                    (n, o, dummy, out, status, 1., 1., style, curves, 256),
                                    ((720 * 1280 + 255) // 256,))

    def entry_mlp_args(self, f, s, out):
        m = self.post.body.mlp
        return (f, s, self.post.input_weight, self.post.input_skip_scale,
                m.expansion, m.contraction, m.skip_scale, out, 32)

    def qkv_args(self, mlp, qkv):
        a = self.post.body.attention
        return (mlp, a.front.qkv, a.front.scale, a.pixel_order, *qkv, 1288, 161, 32, False)

    def head_args(self, mlp, qkv, out):
        a = self.post.body.attention
        return (mlp, *qkv, a.bias, self.post.body.output_weight, self.post.body.skip_scale,
                a.pixel_order, self.post.head_weight, out)

    def attention_args(self, mlp, qkv, out):
        a = self.post.body.attention
        return (mlp, *qkv, a.bias, self.post.body.output_weight, self.post.body.skip_scale,
                a.pixel_order, out, 768, 1280, 4, 4, False)

    def baseline_head_args(self, full, out):
        return (full, self.post.head_weight, out, 720 * 1280, 1280,
                *full.stride(), *self.post.head_weight.stride(), 32)

    def baseline_entry_args(self, f, s, out):
        return (f, s, self.post.input_weight, self.post.input_skip_scale, out, 768, 1280, 32, False)

    def post_fused_head(self, features, skip):
        self.tensor(features, (384, 640, 32)); self.tensor(skip, (768, 1280, 32))
        if self.flags["post_entry_mlp"]:
            mlp = self.post_entry_mlp(self.post, features, skip)
        elif self.flags["post_entry"]:
            mlp = self.post_entry(self.post, features, skip)
        elif self.baseline_entry is not None:
            # Screen and record the exact baseline entry kernel instead of
            # impersonating its bypassed outer combo's Python entry counter.
            raw = self.empty((776, 1288, 32))
            self.launch("post_entry_baseline_passthrough", self.baseline_entry_args(features, skip, raw))
            mlp = self.post.body.mlp.forward_unquantized(raw)
        else:
            raise RuntimeError("Independent native head requires frozen actual baseline entry callback")
        qkv = tuple(self.empty((97, 161, 64, 32)) for _ in range(3))
        self.launch("post_qkv", self.qkv_args(mlp, qkv))
        out = self.empty((720, 1280, 4))
        if self.flags["post_native_head"]:
            self.launch("post_attention_native_head", self.head_args(mlp, qkv, out))
        else:
            full = self.empty((768, 1280, 32))
            self.launch("post_attention_project_passthrough", self.attention_args(mlp, qkv, full))
            self.launch("post_native_head_passthrough", self.baseline_head_args(full, out))
        return out

    def post_entry_mlp(self, post, features, skip):
        if post is not self.post:
            raise RuntimeError("Post entry/MLP owner changed")
        self.tensor(features, (384, 640, 32)); self.tensor(skip, (768, 1280, 32))
        mlp = self.empty((776, 1288, 32))
        self.launch("post_entry_mlp", self.entry_mlp_args(features, skip, mlp))
        return mlp

    @staticmethod
    def tail_site(temporal, normalize, floating, post_sigmoid="table"):
        return f"post_tail:s{post_sigmoid}:t{int(temporal)}:n{int(normalize)}:f{int(floating)}"

    def tail_args(self, head, rgb, previous, reciprocal, out):
        return (head, rgb, rgb if previous is None else previous,
                rgb if reciprocal is None else reciprocal,
                self.model.blend_scale, self.sigmoid_values, out, previous is not None,
                reciprocal is not None, self.post_sigmoid, 256)

    def post_tail(self, head, rgb, *, previous=None, history_reciprocal=None,
                  blend_scale=None, return_float32=False, destination=None):
        self.tensor(head, (720, 1280, 4)); self.tensor(rgb, (720, 1280, 3), dtype=self.torch.float32)
        if blend_scale is not None and blend_scale is not self.model.blend_scale:
            raise RuntimeError("Temporal blend-scale owner changed")
        if previous is not None:
            self.tensor(previous, (720, 1280, 3), dtype=self.torch.float32)
        if history_reciprocal is not None:
            if previous is None:
                raise ValueError("History reciprocal requires previous numerator")
            self.tensor(history_reciprocal, (720, 1280), dtype=self.torch.float32)
        dtype = self.torch.float32 if return_float32 else self.torch.float16
        out = self.empty((720, 1280, 3), dtype=dtype) if destination is None else self.tensor(destination, (720, 1280, 3), dtype=dtype)
        if any(out.data_ptr() == t.data_ptr() for t in (head, rgb, previous, history_reciprocal) if t is not None):
            raise RuntimeError("Post output cannot overwrite live inputs")
        site = self.tail_site(previous is not None, history_reciprocal is not None, return_float32, self.post_sigmoid)
        self.launch(site, self.tail_args(head, rgb, previous, history_reciprocal, out))
        return out

    def average_args(self, x, out):
        return (x, out, *out.shape, *x.stride(), 256)

    def store_half_rne(self, value):
        self.tensor(value, (720, 1280, 3), dtype=self.torch.float32)
        out = self.empty((720, 1280, 3))
        self.launch("display_native_store", (value, out, 256))
        return out

    def pool_down(self, site, full, weight):
        module, attr, shape = self.down_sites[site]
        if weight is not getattr(module, attr):
            raise RuntimeError("Pool/down weight owner changed")
        self.tensor(full, shape, positive_stride=True)
        out = self.empty((shape[0] // 2, shape[1] // 2, weight.shape[1]))
        # Shifted blocks return a view with halo pitch. Its real positive strides
        # enter the specialization; preflight screens those pitches explicitly.
        args = self.matrix_args(full, weight, out, pool=True)
        self.launch(self.pool_site(site, full), args)
        return out

    def average_pool(self, full, *, site="down:c512"):
        """Optional native pooled boundary for explicit diagnostics only.

        Actual pool/down consumes averages inside its A-loader. This method
        materializes them only when a diagnostic explicitly requests the old
        pooled tuple slot; it never runs in the normal terminal boundary.
        """
        _, _, shape = self.down_sites[site]
        self.tensor(full, shape, positive_stride=True)
        out = self.empty((shape[0] // 2, shape[1] // 2, shape[2]))
        self.launch("diagnostic_pool:" + self.pool_site(site, full), self.average_args(full, out))
        return out

    @staticmethod
    def pool_site(site, full):
        return site + ":pitch" + "x".join(str(s) for s in full.stride())

    def up_project(self, name, features, weight):
        shapes = {"decoder.0.0": (24, 40, 512), "decoder.1.0": (48, 80, 256),
                  "decoder.2.0": (96, 160, 128), "decoder.3.0": (192, 320, 64)}
        self.tensor(features, shapes[name])
        if weight is not self.modules[name].weight:
            raise RuntimeError("Upsample matrix owner changed")
        out = self.empty((*shapes[name][:2], weight.shape[1]))
        self.launch("up:" + name, self.matrix_args(features, weight, out))
        return out

    @staticmethod
    def style_site(style, curves, floating):
        return f"style{style}:curve{int(curves)}:f{int(floating)}"

    def style_process(self, neural, original, *, style, intensity=1., local_tone=1.,
                      control_mask=None, ui=None, return_float32=False):
        import math
        import struct
        if type(style) is not int or style not in (1, 2):
            raise ValueError("Game native grading supports style1/2; style0 stays in model")
        for value in (intensity, local_tone):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 2:
                raise ValueError("Style intensity/tone must be finite in [0,2]")
        if control_mask is not None or ui is not None:
            raise ValueError("Current game contract has no UI/control-mask texture")
        self.tensor(neural, (720, 1280, 3))
        self.tensor(original, (720, 1280, 3), dtype=self.torch.float32)
        f32 = lambda v: struct.unpack("<f", struct.pack("<f", v))[0]
        tone, amount = min(f32(local_tone), 1.), min(f32(intensity), 1.)
        curves = tone == 1.
        curve = self.style_owner.prepare(style, neural.device) if curves else self.style_dummy
        if curves and (style, str(neural.device)) not in self.curve_stamps:
            # Freeze the lazy curve upon first use; no curve enters body buffers.
            key = (style, str(neural.device))
            self.curve_stamps[key] = _stamp(curve)
        if curves and _stamp(curve) != self.curve_stamps[(style, str(neural.device))]:
            raise RuntimeError("Authored tone curve mutated")
        dtype = self.torch.float32 if return_float32 else self.torch.float16
        out = self.empty((720, 1280, 3), dtype=dtype)
        status = self.empty(((720 * 1280 + 255) // 256,), dtype=self.torch.int32)
        self.launch(self.style_site(style, curves, return_float32),
                    (neural, original, curve, out, status, amount, tone, style, curves, 256))
        # The conditional style path retains one finite/domain acceptance gate.
        # It never executes at default style0; it is not a claimed free readback.
        if bool(status.any().item()):
            self.session._failed = True
            raise ValueError("Style requires finite private neural and SDR original")
        return out

    def post_entry(self, module, f, s):
        if module is not self.post:
            raise RuntimeError("Post entry owner changed")
        self.tensor(f, (384, 640, 32)); self.tensor(s, (768, 1280, 32))
        out = self.empty((776, 1288, 32))
        self.launch("post_entry", self.entry_args(f, s, out))
        return self.post.body.mlp.forward_unquantized(out)

    def install(self):
        scope = self
        if self.flags["merge_native"]:
            def merge(owner, name, projected, skip, scale, shift=None):
                if name not in MERGES:
                    # P09/DecoderC32 is owned by the Swin worker.
                    return scope.original_merge(name, projected, skip, scale, shift)
                if owner is not scope.decoder or shift is not None:
                    raise RuntimeError("Four native merges must be unshifted")
                shape = MERGES[name]
                module = scope.modules[name]
                expected = module.skip_scale if name == "decoder_input" else module.input_skip_scale
                if scale is not expected:
                    raise RuntimeError("Merge scale owner changed")
                scope.tensor(projected, (shape[0] // 2, shape[1] // 2, shape[2]))
                scope.tensor(skip, shape)
                out = scope.empty(shape)
                kernel = scope.launch("merge:" + name, scope.merge_args(projected, skip, scale, out))
                owner.calls[name] = owner.calls.get(name, 0) + 1
                owner.resources[name + ":" + kernel.hash] = scope.screens["merge:" + name][4]
                if owner.probe is not None:
                    owner.probe(name, projected, skip, scale, shift, out)
                return out
            self.bind(self.decoder, "merge", merge)
            self.register_contract("merge", ("OUT",))
        if self.flags["decoder_full_k"]:
            def input(owner, module, features, skip):
                if owner is not scope.decoder or module is not scope.modules["decoder_input"]:
                    raise RuntimeError("FullK input owner changed")
                scope.tensor(features, (12, 20, 1024)); scope.tensor(skip, MERGES["decoder_input"])
                out = scope.empty((12, 20, 512))
                scope.launch("decoder_input_full_k", scope.matrix_args(features, module.weight, out))
                return owner.merge("decoder_input", out, skip, module.skip_scale)
            self.bind(self.decoder, "input", input)
            self.register_contract("matrix", ("OUT",))
        self.install_extra()

    def install_extra(self):
        scope = self
        if self.flags["post_entry"] or self.flags["post_entry_mlp"] or self.flags["post_native_head"]:
            self.qkv = self.authenticate(import_module("fused_c32_projection_native_half_v1"))
            self.attention_module = self.authenticate(import_module("post_attention_fusion_v1"))
            self.baseline_head = self.authenticate(import_module("native_k8_active_720_v1"))
            self.baseline_entry_kernel = self.authenticate(import_module("post_entry_mlp_fusion_v1"))
            m, a = self.post.body.mlp, self.post.body.attention
            for owner, names in ((m, ("expansion", "contraction", "skip_scale")),
                                 (a.front, ("qkv", "scale")), (a, ("bias", "pixel_order")),
                                 (self.post.body, ("output_weight", "skip_scale")),
                                 (self.post, ("head_weight",))):
                for name in names: self.watch_weight(owner, name)
            def forward_head(post, features, skip):
                if post is not scope.post:
                    raise RuntimeError("Native post fused head owner changed")
                return scope.post_fused_head(features, skip)
            self.bind(self.post, "forward_head", forward_head)
            if self.flags["post_entry_mlp"]:
                self.register_contract("entry_mlp", ("OUT",))
            if self.flags["post_native_head"]:
                self.register_contract("attention_native_head", ("OUT",))
            if self.flags["post_entry"] and not self.flags["post_entry_mlp"]:
                self.register_contract("entry", ("OUT",))
        if self.flags["post_tail"]:
            self.watch_weight(self.model, "blend_scale")
            self.authenticate(import_module("post_numeric_suite_720_kernel_v1"))
            self.authenticate(import_module("nr_backend.sigmoid"))
            self.sigmoid_owner = self.model.sigmoid
            self.sigmoid_values = self.sigmoid_owner.values
            self.watch_weight(self.sigmoid_owner, "values")
            self.tensor(self.sigmoid_values, (65536,), dtype=self.torch.float32)
            post_class = type(self.post)
            original = post_class.forward
            def forward_post(post, features, skip, rgb, *, previous=None, sigmoid=None,
                             blend_scale=None, history_reciprocal=None, return_float32=False):
                if post is not scope.post:
                    return original(post, features, skip, rgb, previous=previous, sigmoid=sigmoid,
                                    blend_scale=blend_scale, history_reciprocal=history_reciprocal,
                                    return_float32=return_float32)
                if previous is not None and (sigmoid is not scope.model.sigmoid or blend_scale is not scope.model.blend_scale):
                    raise RuntimeError("Post temporal owner changed")
                head = post.forward_head(features, skip)
                return scope.post_tail(head, rgb, previous=previous,
                                       history_reciprocal=history_reciprocal, blend_scale=blend_scale,
                                       return_float32=return_float32)
            self.bind(post_class, "forward", forward_post, method=False)
            self.register_contract("tail", ("OUT",))
            controlled = self.authenticate(import_module("nr_game_controlled_model"))
            self.bind(controlled, "store_half_rz", self.store_half_rne, method=False)
            self.bind(self.model, "_audit_fdp_native_display_store", self.store_half_rne, method=False)
            self.register_contract("convert", ("OUT",))
        if self.flags["stage_matrices"]:
            self.install_stages()
        if self.flags["native_style"]:
            assets = self.authenticate(import_module("audit_front_decoder_post_720_assets_v1"))
            style_owner = self.model.__dict__.get("style_post")
            if not isinstance(style_owner, assets.LazyNativeStyle):
                raise RuntimeError("Native lazy style requires construction_assets hook before model initialization")
            self.style_owner = style_owner
            self.style_dummy = self.empty((1,), dtype=self.torch.float32)
            self.curve_stamps = {}
            self.bind(style_owner, "dispatch", self.style_process, method=False)
            self.register_contract("style", ("OUT", "STATUS"))
        if self.flags["front_controls"]:
            self.bind(self.model, "_audit_fdp_front_producer", self, method=False)
            self.replacement_sites["front_controls"] = ["front_noise_native.front lanes10..14", "GameLiveControlledNR.graph_controls five fills"]

    def control_lanes(self):
        self.validate_frame_context()
        c = self.model._controls
        skin = c.local_structure if c.skin_structure is None else c.skin_structure
        return tuple(float(v) for v in (c.style / 128, c.local_tone,
                     1 if c.auto_mask else c.local_structure,
                     skin if c.auto_mask else -1, c.local_structure if c.auto_mask else -1))

    def bind_front_destination(self, broker):
        """Cold integration by history/graph owner; no history math is owned here.

        broker.acquire_front_720(rgb,previous,event) -> writable FP16 HWC16
        lease or None (eager/cold). It must pin its own graph slot, stream/event,
        reset/near/fractional context and fail instead of lending a live slot.
        broker.commit_front_720(output,event) records the actual submitted front
        kernel receipt. The broker proves replay/retirement before the next use.
        """
        self.guard()
        method = broker.acquire_front_720
        fn = getattr(method, "__func__", method)
        commit = broker.commit_front_720
        commit_fn = getattr(commit, "__func__", commit)
        self.authenticate(import_module(fn.__module__))
        if (getattr(method, "__self__", None) is not broker or broker.model is not self.model
                or broker.session is not self.session or broker.graph is not self.graph
                or getattr(commit, "__self__", None) is not broker or commit_fn.__module__ != fn.__module__
                or self.front_broker is not None):
            raise RuntimeError("Front destination broker ownership changed")
        self.front_broker = (broker, method, fn.__code__, commit, commit_fn.__code__)
        self._front_broker_binding = self.front_broker

    def front_destination(self, rgb, previous, event):
        self.validate_frame_context()
        if self.front_broker is None:
            return None
        broker, method, code, commit, commit_code = self.front_broker
        if not _same(broker.acquire_front_720, method) or method.__func__.__code__ is not code:
            raise RuntimeError("Static front broker callable changed")
        out = method(rgb, previous, event)
        if out is not None:
            self.tensor(out, (768, 1280, 16))
            if out.data_ptr() == rgb.data_ptr() or (previous is not None and out.data_ptr() == previous.data_ptr()):
                raise RuntimeError("Static front lease aliases input/history")
        return out

    def record_front(self, output, lanes, event, *, direct):
        self.validate_frame_context()
        if tuple(lanes) != self.control_lanes() or not event.get("front_kernel_hash"):
            raise RuntimeError("Front control record lacks actual launch receipt")
        if self.front_broker is not None:
            broker, method, code, commit, commit_code = self.front_broker
            if not _same(broker.commit_front_720, commit) or commit.__func__.__code__ is not commit_code:
                raise RuntimeError("Static front broker commit callable changed")
            commit(output, event)
        self.front_record = (weakref.ref(output), id(output), output.data_ptr(), tuple(lanes), event["front_kernel_hash"])
        self.front_producer_calls += 1
        self.front_direct_writes += int(direct)

    def consume_controls(self, front, controls):
        self.validate_frame_context()
        record = self.front_record
        if record is None or controls is not self.model._controls or record[0]() is not front:
            return False
        if (record[1] != id(front) or record[2] != front.data_ptr() or record[3] != self.control_lanes()):
            raise RuntimeError("Front controls/provenance changed before consumer")
        self.front_controls_consumed += 1
        self.front_record = None
        return True

    def install_stages(self):
        scope = self
        self.down_sites = {}
        for channels, group, shape in zip((32, 64, 128, 256), self.model.encoder,
                                         ((384, 640, 32), (192, 320, 64), (96, 160, 128), (48, 80, 256))):
            if channels not in self.down_channels:
                continue
            targets = [m for m in group if getattr(m, "down_weight", None) is not None]
            if len(targets) != 1:
                raise RuntimeError("Expected one active down matrix per encoder family")
            module = targets[0]; site = f"down:c{channels}"
            self.down_sites[site] = (module, "down_weight", shape)
            self.watch_weight(module, "down_weight")
            def outputs(block, features, site=site):
                full = block.forward_unquantized(features)
                return full, scope.pool_down(site, full, block.down_weight)
            self.bind(module, "forward_outputs", outputs)
        module = self.model.encoder512[-1]
        if module.final_weight is None:
            raise RuntimeError("Missing C512 encoder terminal down weight")
        if 512 in self.down_channels:
            self.down_sites["down:c512"] = (module, "final_weight", (24, 40, 512))
            self.watch_weight(module, "final_weight")
        def terminal_c512(block, x):
            # Same actual FullsizeSession providers; only the terminal pool/dot
            # boundary is replaced. No C512 FFN/attention body is rewritten.
            stack, session = scope.stack, scope.session
            split = import_module("nr_backend.split_block")
            name = stack.c512_int8.modules[id(block)]
            mlp = stack.c512_int8.ffn(name, x)
            h, w = x.shape[:2]; sy, sx = block.window_shift
            padded = scope.torch.nn.functional.pad(mlp, (0, 0, sx, (-w - sx) % 8, sy, (-h - sy) % 8))
            attended = split.q(block.attention(padded)[sy:sy + h, sx:sx + w])
            full = block.projection.forward_unquantized(attended, mlp)
            session._counts["c512"] += 1
            final = scope.pool_down("down:c512", full, block.final_weight)
            # capture_body consumes indices3/-1. Diagnostics requesting the
            # old pooled tensor must use scope.average_pool(full) explicitly.
            return mlp, mlp, attended, full, None, final
        if 512 in self.down_channels and self.flags["pool_c512_transition"]:
            function = _function(module.forward_boundaries)
            if ("forward_boundaries" in module.__dict__ or
                    function.__module__ not in ("nr_backend.split_block", "fullsize_session_v1")):
                raise RuntimeError("C512 boundary already owned; select pool_c512_transition=false for P11 consumer")
            self.bind(module, "forward_boundaries", terminal_c512)
        self.replacement_sites["c512_boundary_owner"] = (
            "front-decoder-post" if self.flags["pool_c512_transition"] else "c512-vit") if 512 in self.down_channels else "not_selected"
        pre = self.model.pre
        def pre_outputs(owner, features):
            full = owner.forward_features_unquantized(features)
            scope.tensor(full, (768, 1280, 32), positive_stride=True)
            out = scope.empty((384, 640, 32))
            scope.launch("pre_average", scope.average_args(full, out))
            return full, out
        if self.flags["pre_average"]:
            self.bind(pre, "forward_features_outputs", pre_outputs)
        decoder_mod = self.authenticate(import_module("nr_backend.decoder"))
        dot = decoder_mod.dot
        weights = {id(self.modules[n].weight): n for n in self.modules
                   if n != "decoder_input" and self.modules[n].channels in self.up_channels}
        for name in weights.values():
            if name not in MERGES:
                self.watch_weight(self.modules[name], "weight")
        def up_dot(x, weight, *, chunk_k=16, initial=None):
            name = weights.get(id(weight))
            if name is None or initial is not None:
                return dot(x, weight, chunk_k=chunk_k, initial=initial)
            return scope.up_project(name, x, weight)
        if weights:
            self.bind(decoder_mod, "dot", up_dot, method=False)
        # Explicit callable consumed by P09 if its first-body gather no longer
        # routes through decoder.dot. Main passes this instead of changing body.
        if 32 in self.up_channels:
            self.bind(self.decoder, "native_up_project", self.up_project, method=False)
        self.bind(self.model, "_audit_fdp_pool_down", self.pool_down, method=False)
        if "matrix" not in {key.rsplit(".", 1)[-1] for key in self.contracts}:
            self.register_contract("matrix", ("OUT",))
        self.register_contract("average", ("OUT",))
        self.replacement_sites["stage_matrices"] = list(self.down_sites) + ["up:" + n for n in weights.values()]

    def snapshot(self):
        return {"schema": "owned-front-decoder-post720-v1", "group": GROUP,
                "baseline_identity_sha256": BASELINE_IDENTITY, "resolution_modes": [720],
                "options": {**self.flags, "post_sigmoid": self.post_sigmoid,
                            "post_store": "native_rne" if self.flags["post_tail"] else "baseline"},
                "preflight_complete": self.preflight_complete,
                "site_selection": {"down_channels": self.down_channels, "up_channels": self.up_channels},
                "replacement_sites": self.replacement_sites, "sources": self.sources,
                "calls": dict(self.calls), "capture_calls": dict(self.capture_calls),
                "resources": {site: row[4] for site, row in self.screens.items()},
                "count_semantics": "Python eager/capture dispatch only; graph replay does not increment",
                "gpu_verified": False, "thread_owner": self.thread, "active": self.active,
                "style_assets": self.style_owner.snapshot() if hasattr(self, "style_owner") else None,
                "front_controls": {"producer_calls": self.front_producer_calls,
                                   "consumer_skips": self.front_controls_consumed,
                                   "direct_static_writes": self.front_direct_writes,
                                   "destination_broker_bound": self.front_broker is not None}}

    def capture_role_replacements(self, before):
        """Partial gate evidence for a *fresh capture*, never invented replay hits.

        passed covers only this scope's screened actual capture dispatches.
        Main unions disjoint replaced_sites and checks every remaining old site.
        It must invoke this only when its actual graph entry count grew.
        """
        post_changed = any(self.flags[n] for n in ("post_entry", "post_entry_mlp", "post_native_head"))
        if not post_changed:
            return {}
        saved = before.get("own", {}).get(__name__)
        if isinstance(saved, dict) and ("capture_calls" in saved or "calls" in saved):
            saved = saved.get("capture_calls", saved.get("calls"))
        saved = saved if isinstance(saved, dict) else {}
        entry = "post_entry_mlp" if self.flags["post_entry_mlp"] else (
            "post_entry" if self.flags["post_entry"] else "post_entry_baseline_passthrough")
        attention = "post_attention_native_head" if self.flags["post_native_head"] else "post_attention_project_passthrough"
        head = "post_attention_native_head" if self.flags["post_native_head"] else "post_native_head_passthrough"
        def result(gate, replaced, sites, remaining):
            sites = tuple(dict.fromkeys(sites))
            deltas = {site: self.capture_calls.get(site, 0) - saved[site] if site in saved else None for site in sites}
            evidence = {site: self.screens[site][4] for site in sites if site in self.screens}
            ready = (self.preflight_complete and len(evidence) == len(sites)
                     and all(r.get("spills") == 0 and isinstance(r.get("binary_sha256"), str)
                             and len(r["binary_sha256"]) == 64 for r in evidence.values()))
            return {"passed": bool(ready and all(type(v) is int and v > 0 for v in deltas.values())),
                    "replaced_sites": [str(site) for site in replaced], "new_sites": deltas,
                    "new_counter_keys": {site: site for site in sites},
                    "remaining_old_sites": remaining, "resource_evidence": evidence,
                    "count_semantics": "actual capture_calls delta; remaining sites are informational, main owns their union gate"}
        native_before = before.get("before_native_k8", before.get("native_k8"))
        native_after = getattr(self.modes, "native_k8_calls", None)
        remaining_native = {}
        if isinstance(native_before, dict) and isinstance(native_after, dict):
            remaining_native = {str(k): native_after[k] - v for k, v in native_before.items() if k != "post" and k in native_after}
        output = {
            "post_entry": result("post_entry", ["post_entry"], [entry], {}),
            "native_k8_pre_post": result("native_k8_pre_post", ["post"], [head], remaining_native),
            "post_k8": result("post_k8", ["post_k8"], [head], {}),
            "post_attention": result("post_attention", ["post_attention"], ["post_qkv", attention], {}),
            "post_contiguous_k8": result("post_contiguous_k8", ["post_contiguous_k8"], [head], {}),
        }
        if self.flags["post_entry_mlp"]:
            old = before.get("before_c32_hidden", before.get("c32_hidden"))
            owner = getattr(self.modes, "c32_hidden_calls", None)
            current = getattr(owner, "capture_calls", None)
            remaining = {}
            if isinstance(old, dict) and isinstance(current, dict):
                remaining = {str(k): current[k] - v for k, v in old.items() if k != "post" and k in current}
            output["c32_hidden_native_ten"] = result("c32_hidden_native_ten", ["post"], [entry], remaining)
        return output

    def bind_serial_owner(self, owner, *, transfer_method):
        """Cold main transaction registration; authenticate the real caller code.

        Main calls this once while creating its implementation-hooks owner.
        The owner must hold this scope in children and share modes/session/graph.
        Its module SHA must be in the composed source_pins, supplied by main.
        """
        self.guard()
        fn = getattr(transfer_method, "__func__", transfer_method)
        module = import_module(fn.__module__)
        self.authenticate(module)
        if (getattr(transfer_method, "__self__", None) is not owner
                or fn.__globals__ is not vars(module)
                or Path(fn.__code__.co_filename).resolve() != Path(module.__file__).resolve()
                or owner.modes is not self.modes or owner.session is not self.session
                or owner.graph is not self.graph
                or sum(child is self for child in owner.children.values()) != 1
                or self._serial_owner is not None):
            raise RuntimeError("Invalid implementation serial owner registration")
        self._serial_owner = (owner, fn, fn.__code__)

    def _transfer_serial_thread(self, owner, *, serial_guard, previous_thread):
        """Called only by the authenticated real process-lock transaction."""
        old = self.thread
        caller = sys._getframe(1)
        registered = self._serial_owner
        adapter = sys.modules.get("cyberpunk_nr_adapter")
        host = sys.modules.get("nr_game_pre_xess_host")
        try:
            if (registered is None or registered[0] is not owner
                    or registered[1].__code__ is not registered[2]
                    or caller.f_code is not registered[2] or caller.f_globals is not registered[1].__globals__
                    or caller.f_locals.get("self") is not owner
                    or caller.f_locals.get("serial_guard") is not serial_guard
                    or caller.f_locals.get("previous_thread") != previous_thread
                    or owner.modes is not self.modes or owner.session is not self.session or owner.graph is not self.graph
                    or sum(child is self for child in owner.children.values()) != 1
                    or type(previous_thread) is not int or previous_thread != old
                    or adapter is None or host is None or getattr(adapter, "host", None) is not host
                    or getattr(host, "_modes", None) is not self.modes or getattr(host, "_failed", False)
                    or type(serial_guard) is not type(RLock())
                    or serial_guard is not getattr(adapter, "_process_serial_lock", None)
                    or serial_guard is getattr(adapter, "_settings_lock", None) or not serial_guard._is_owned()
                    or self._in_dispatch or self.torch.xpu.is_current_stream_capturing()
                    or any(getattr(c, "_in_frame", False) or getattr(c, "_in_dispatch", False) for c in owner.children.values())):
                raise RuntimeError("Owned720 thread transfer requires idle authenticated owner and actual held process RLock")
            bridge = getattr(host, "_bridge", None)
            if (bridge is None or bridge.thread != get_ident() or getattr(host, "_thread", None) != get_ident()
                    or bridge.torch is not self.torch
                    or self.torch.xpu.current_stream().sycl_queue != bridge.stream.sycl_queue):
                raise RuntimeError("Owned720 transfer requires bridge stream already bound")
            if self._serial_binding is not None and self._serial_binding != (owner, adapter, serial_guard):
                raise RuntimeError("Owned720 serialized adapter/process guard changed")
            owner._owner()
            self._guard_fixed()
            self.thread = get_ident()
            self.validate_frame_context()
            self._serial_binding = (owner, adapter, serial_guard)
        except BaseException:
            self.thread = old
            self._retired = True
            self.session._failed = True
            raise
        return self.thread

    def close(self):
        valid = all(_same(getattr(o, n), f) for o, n, _, _, f in self.bindings)
        for owner, name, had, previous, _ in reversed(self.bindings):
            if had:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
        contracts = import_module("quantization_dataflow_v1").CONTRACTS
        valid = valid and all(contracts.get(k) == v for k, v in self.contracts.items())
        for key in self.contracts:
            contracts.pop(key, None)
        self.active = False
        self._retired = True
        if (self.graph.entries or any(self.calls.values())) and not self.session._closed:
            self.session._failed = True
            raise RuntimeError("Dispose captured graphs/close session before leaving owned implementation scope")
        if not valid:
            self.session._failed = True
            raise RuntimeError("Owned720 scope restore interference")


@contextmanager
def installed(target, **kwargs):
    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError("One owned720 scope per process; no nested installation")
    scope = Scope(target, **kwargs)
    _ACTIVE = scope
    try:
        scope.install()
        yield scope
    except BaseException:
        scope.session._failed = True
        raise
    finally:
        try:
            scope.close()
        finally:
            _ACTIVE = None


def front_controls_present(model, front, controls):
    """Only the authenticated current producer may suppress five control fills."""
    scope = getattr(model, "_audit_fdp_front_producer", None)
    if scope is None:
        return False
    if scope is not _ACTIVE or scope.model is not model or not scope.flags["front_controls"]:
        raise RuntimeError("Unknown front control producer")
    return scope.consume_controls(front, controls)
