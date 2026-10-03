"""Owned local C32--C256 overlay for the accepted fixed 1280x720 fast chain.

Install after the accepted structure/C32-hidden scopes, before the numerical
suite and any capture. This module imports only stdlib until installed() is
entered. It does not wrap model.forward or alter its numeric forward registry.
Existing Python receipts include warmups/capture; replay never reruns _body.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from types import MethodType
from threading import get_ident, RLock


BASE = {32: (384, 640), 64: (192, 320), 128: (96, 160), 256: (48, 80)}
COUNTS = {32: 4, 64: 4, 128: 6, 256: 8}
SHIFTS = ((0, 0), (4, 4), (0, 4), (4, 0))
KERNEL_SHA256 = 'b82f99091a775dab1ef8dd7373de32e26cd243f0e5538fc42243084cf3a3bde8'


@dataclass(frozen=True)
class Options:
    mlp_wide: bool = True
    fuse_pad: bool = True
    qkv_wide: bool = True
    qk_norm: str = 'fp32'
    denominator: str = 'fp32'
    wide_score: bool = True
    wide_value: bool = True
    crop_queries: bool = True
    prefer_bm32: bool = True
    attention_families: tuple[int, ...] = (32, 64, 128, 256)
    first_decoder32: bool = True

    def __post_init__(self):
        for name in ('mlp_wide', 'fuse_pad', 'qkv_wide', 'wide_score', 'wide_value',
                     'crop_queries', 'prefer_bm32', 'first_decoder32'):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f'{name} must be bool')
        if self.qk_norm not in ('ordered_half', 'fp32') or self.denominator not in ('ordered_half', 'fp32'):
            raise ValueError('Unknown local numerical variant')
        if self.fuse_pad and not self.mlp_wide:
            raise ValueError('Fused zero-gather requires the owned MLP producer/skip consumer')
        families = tuple(self.attention_families)
        if len(set(families)) != len(families) or any(type(c) is not int or c not in BASE for c in families):
            raise ValueError('Choose unique local attention families 32/64/128/256')
        object.__setattr__(self, 'attention_families', tuple(sorted(families)))

    @property
    def identity(self):
        payload = json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))
        return 'audit-swin-720-v1:' + hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class Site:
    label: str
    side: str
    group: int
    index: int
    channels: int
    height: int
    width: int
    shift: tuple[int, int]
    decoder_body: bool

    @property
    def input_shape(self):
        return (self.height, self.width, self.channels)

    @property
    def padded_shape(self):
        sy, sx = self.shift
        return (self.height + (8 if sy else 0), self.width + (8 if sx else 0), self.channels)

    @property
    def c32_combo_key(self):
        if self.channels != 32 or self.decoder_body:
            return None
        return f'block_{self.index if self.side == "encoder" else self.index + 3}'


def discover(model, *, c32_type=None, multihead_type=None):
    """Exact 44 actual sites; first decoder body, never its outer gather/merge."""
    targets = {}
    mlps = set()
    for channels, count in COUNTS.items():
        for side, group_index in (('encoder', {32: 0, 64: 1, 128: 2, 256: 3}[channels]),
                                  ('decoder', {32: 3, 64: 2, 128: 1, 256: 0}[channels])):
            group = getattr(model, side)[group_index]
            if len(group) != count:
                raise RuntimeError(f'Unexpected {side} C{channels} group length')
            for index, outer in enumerate(group):
                body = side == 'decoder' and index == 0
                block = outer.body if body else outer
                expected_type = c32_type if channels == 32 else multihead_type
                if expected_type is not None and type(block) is not expected_type:
                    raise RuntimeError('Local block class differs from the fixed model')
                shift = SHIFTS[(index + (2 if side == 'decoder' and channels == 128 else 0)) % 4]
                if tuple(block.window_shift) != shift:
                    raise RuntimeError(f'Actual shift changed: {side} C{channels} index {index}')
                if channels != 32 and (block.channels != channels or block.attention.heads != channels // 32):
                    raise RuntimeError('Local channel/head owner changed')
                if id(block) in targets or id(block.mlp) in mlps:
                    raise RuntimeError('Duplicate local block/MLP identity')
                height, width = BASE[channels]
                label = f'{side}{channels}.{index}' + ('.body' if body else '')
                targets[id(block)] = (Site(label, side, group_index, index, channels,
                                           height, width, shift, body), block)
                mlps.add(id(block.mlp))
    if len(targets) != 44:
        raise RuntimeError('Expected all 44 local bodies')
    return targets


class OwnershipLedger:
    """Restore every unchanged owned slot; retain foreign writes and fail closed."""
    def __init__(self):
        self.entries = []

    def attribute(self, obj, name, value):
        previous = (name in obj.__dict__, obj.__dict__.get(name))
        setattr(obj, name, value)
        self.entries.append(('attribute', obj, name, previous, value))

    def contract(self, mapping, name, value):
        if name in mapping:
            raise RuntimeError(f'Owned kernel contract already registered: {name}')
        mapping[name] = value
        self.entries.append(('contract', mapping, name, (False, None), value))

    def violations(self):
        return [name for kind, obj, name, _, value in self.entries
                if (obj.__dict__.get(name) if kind == 'attribute' else obj.get(name)) is not value]

    def restore(self, session, primary_exception=None):
        violations = self.violations()
        for kind, obj, name, (present, previous), value in reversed(self.entries):
            current = obj.__dict__.get(name) if kind == 'attribute' else obj.get(name)
            if current is not value:
                continue
            if kind == 'attribute':
                if present:
                    setattr(obj, name, previous)
                else:
                    delattr(obj, name)
            elif present:
                obj[name] = previous
            else:
                del obj[name]
        if violations:
            session._failed = True
            message = 'Owned Swin slots changed; all unchanged slots restored: ' + ', '.join(violations)
            if primary_exception is not None:
                primary_exception.add_note(message)
            else:
                raise RuntimeError(message)
        return violations


class Counter:
    """Cold asset/binary admission and a memory-only runtime ownership interface."""
    def __init__(self, modes, options):
        import torch
        import audit_swin_720_kernels_v1 as kernels
        import quantization_dataflow_v1 as dataflow
        import nr_backend.c32_block as c32
        import nr_backend.multihead_block as multihead
        from decoder_input_full_k_720_v1 import _Sources, _stamp
        from nr_backend.execution import current_arithmetic_backend, record_arithmetic_dispatch
        from current_dense_tiled_provider_v1 import POLICY
        import sys

        self.modes, self.options, self.session = modes, options, modes.session
        self.stack = self.session._stack
        self.model, self.graph, self.provider = self.stack.model, self.stack.graph, self.stack.provider
        self.window = self.stack.window_blocks
        self.torch, self.kernels, self.dataflow = torch, kernels, dataflow
        self.current_arithmetic_backend = current_arithmetic_backend
        self.record = record_arithmetic_dispatch
        self.policy = POLICY
        self.stamp = _stamp
        self.targets = discover(self.model, c32_type=c32.C32SwinBlock, multihead_type=multihead.MultiHeadSwinBlock)
        self.body_targets = {key: row for key, row in self.targets.items()
            if ((row[0].channels in options.attention_families and not
                 (row[0].channels == 32 and row[0].decoder_body)) or
                (row[0].channels == 32 and row[0].decoder_body and options.first_decoder32))}
        self.by_mlp = {id(block.mlp): (site, block) for site, block in self.targets.values()}
        self.by_attention = {id(block.attention): (site, block) for site, block in self.targets.values()}
        self.by_qkv = {id(block.attention.front.qkv if site.channels == 32 else block.attention.qkv): (site, block)
                       for site, block in self.targets.values()}
        self.calls = {site.label: 0 for site, _ in self.targets.values()}
        for site, block in self.targets.values():
            roles = []
            if options.mlp_wide:
                roles.append('mlp')
            if self.qkv_enabled or id(block) in self.body_targets:
                roles.append('qkv')
            if id(block) in self.body_targets:
                roles.append('tail')
            self.calls.update({site.label + ':' + role: 0 for role in roles})
        self.capture_calls = dict(self.calls)
        self.resources, self.compiled, self.payloads, self.selected = {}, {}, {}, {}
        self.actualkernelhash_by_site = {}
        self.site_kernel_hits = {}
        self.mlp_binary_evidence_by_site = {}
        self._mlp_payloads_by_site = {}
        self.kernel_hits = {}
        self.capture_kernel_hits = {}
        self.ledger = OwnershipLedger()
        self.active, self.retired, self.preflight_complete = False, False, False
        self.thread, self.in_frame = get_ident(), False
        self._frozen_options = options
        self._flags = {name: getattr(modes, name) for name in (
            'c512_qkv_library_720', 'native_k8_720', 'c128_pairwise_720',
            'c64_attention_project_720', 'c128_attention_project_720', 'c32_hidden_native_720',
            'decoder_gather_unround_720', 'c512_probability_unround_720',
            'c128_dual_qkv_720')}
        self.sources = _Sources()
        self.sources.add(sys.modules[__name__])
        self.sources.add(kernels, sha256=KERNEL_SHA256)
        self.sources.add(sys.modules[dataflow.__name__])
        for name in ('_load_input', '_join_parts', '_join64', '_mlp_pairs', '_mlp_project',
                     '_c32_mlp', '_normalize_ordered', '_normalize', '_qkv', '_weights_native', '_attention_project'):
            self.sources.function(getattr(kernels, name))
            self.sources.watch(kernels, name)
        for name in ('cubic', 'half_fma_attention', '_exp', '_halves', '_weights_pair', '_nan_left', 'rsqrt_half_clamped'):
            value = getattr(kernels, name)
            self.sources.function(value)
            self.sources.watch(kernels, name)
        for name in ('_body', 'validate', 'preflight', '_launch', '_qkv_spec', 'mlp_boundary',
                     'qkv_override', 'c32_qkv_override', '_record_site_kernel', 'record_mlp_hit',
                     '_role_proof', 'capture_role_replacements', '_require_signature',
                     '_require_installed_scope', 'frame_scope', 'transfer_serial_thread'):
            self.sources.function(getattr(type(self), name))
            self.sources.watch(type(self), name)
        self.callback_modules = []
        from importlib import import_module
        for name in ('c64_qkv_direct_pack_all_v1', 'c128_qkv_direct_pack_all_v1',
                     'c256_qkv_direct_pack_all_v1', 'fused_c32_projection_native_half_v1'):
            module = self.sources.load(name)
            self.sources.function(module.forward if name.startswith('fused_c32') else module.direct_pack)
            self.callback_modules.append(module)
        for name, symbol in (('fused_c32_projection_native_half_v1', '_kernel'),
                             ('c64_qkv_direct_pack_v2', '_qkv_direct_pack'),
                             ('c128_qkv_direct_pack_one_v1', '_direct_pack'),
                             ('c256_qkv_direct_pack_one_v1', '_direct_pack')):
            module = self.sources.load(name)
            self.sources.function(getattr(module, symbol))
            self.sources.watch(module, symbol)
        self.weight_slots = []
        for site, block in self.targets.values():
            att = block.attention
            qkv_owner = att.front if site.channels == 32 else att
            slots = [(qkv_owner, 'qkv', (site.channels, 3 * site.channels)),
                     (qkv_owner, 'scale', (site.channels // 32,)),
                     (att, 'bias', (64, 64) if site.channels == 32 else (site.channels // 32, 64, 64)),
                     (block, 'output_weight', (site.channels, site.channels)),
                     (block, 'skip_scale', (site.channels,)), (att, 'pixel_order', (64,))]
            if site.channels == 32:
                slots += [(block.mlp, 'expansion', (32, 128)), (block.mlp, 'contraction', (128, 32)),
                          (block.mlp, 'skip_scale', (32,))]
            for owner, name, shape in slots:
                tensor = getattr(owner, name)
                dtype = torch.int64 if name == 'pixel_order' else torch.float16
                if (tuple(tensor.shape) != shape or tensor.dtype != dtype or
                        tensor.device.type != 'xpu' or not tensor.is_contiguous()):
                    raise RuntimeError(f'Invalid owned tensor boundary: {site.label}.{name}')
                self.weight_slots.append((owner, name, _stamp(tensor)))
        self._device = self.weight_slots[0][2][0].device
        if any(row[2][0].device != self._device for row in self.weight_slots):
            raise RuntimeError('Local assets differ in device')
        self._combo = modes.combo_calls
        self._family_receipts = {64: modes.c64_attention_project_calls, 128: modes.c128_attention_project_calls}
        self._hidden = modes.c32_hidden_calls
        self._saved_graph_signature = self.graph.signature
        self._saved_signature_hook = self.graph._signature
        self._saved_installed = self.session._installed
        self._central_hooks_identity = getattr(modes, 'implementation_identity_720', None)
        self._central_hooks = tuple(getattr(modes, 'implementation_hooks_720', ()))
        self._identity = options.identity + ':' + hashlib.sha256(json.dumps(self.sources.receipts, sort_keys=True).encode()).hexdigest()
        self.hidden_binaries = {}
        self.tail_keys = {self._key(site) for site, _ in self.body_targets.values()}

    @property
    def mlp_enabled(self):
        return self.options.mlp_wide

    @property
    def qkv_enabled(self):
        return self.options.qkv_wide or self.options.qk_norm == 'fp32'

    def mlp_boundary(self, module):
        site, block = self.by_mlp[id(module)]
        if block.mlp is not module:
            raise RuntimeError('MLP left its fixed local body')
        if self.options.fuse_pad and id(block) in self.body_targets:
            return site.input_shape, (*site.input_shape[:2], site.padded_shape[1], *site.shift)
        hp, wp, _ = site.padded_shape
        return site.padded_shape, (hp, wp, wp, 0, 0)

    def _key(self, site):
        return (site.channels, *site.padded_shape[:2], *site.shift)

    def _require_signature(self):
        from decoder_input_full_k_720_v1 import _same
        suite = getattr(self.modes, '_numeric_cleanup_owner', None)
        if suite is None:
            if not _same(self.graph._signature, self._saved_signature_hook) or self.graph.signature != self._saved_graph_signature:
                raise RuntimeError('Cold graph signature changed before central numeric admission')
        elif (suite.graph is not self.graph or self.graph._signature is not suite.signature_hook or
              self.graph.signature != suite.signature_value or
              suite.signature_value != self._saved_graph_signature + (suite.identity,) or
              self._central_hooks_identity is None or not suite.identity.endswith(':' + self._central_hooks_identity)):
            raise RuntimeError('Unique numeric graph signature lacks the fixed implementation hooks identity')

    def _require_installed_scope(self):
        current, seen = self.session._installed, set()
        suite = getattr(self.modes, '_numeric_cleanup_owner', None)
        children = tuple(suite.children.values()) if suite is not None else ()
        while current is not self._scope_wrapper:
            if id(current) in seen or not any(current is getattr(child, name, None)
                    for child in children for name in ('wrapper', '_scope_wrapper')):
                raise RuntimeError('Owned Swin frame scope left the registered numeric wrapper chain')
            seen.add(id(current))
            current = getattr(current, '__nr_numeric_original_installed__', None)
        if self._scope_wrapper.__nr_numeric_original_installed__ is not self._saved_installed:
            raise RuntimeError('Owned Swin original installed parent changed')

    def validate(self, *, check_thread=True):
        try:
            if (not self.active or self.retired or self.modes.session is not self.session or
                    self.session._failed or self.session._closed or
                    self.session.__dict__.get('_audit_swin_720_owner') is not self or
                    self.modes.height != 720 or tuple(self.modes.source) != (720, 1280) or
                    self.modes.variant != 'unrounded' or self.provider.mode != 'fp16_xmx' or
                    self.window.probe is not None or
                    not self.window.layout.native_normalize or not self.window.layout.native_swin):
                    raise RuntimeError('Fixed 720p Swin owner/configuration changed')
            if check_thread and self.thread != get_ident():
                raise RuntimeError('Owned Swin requires the explicit actual-process serial handoff')
            if (self.options is not self._frozen_options or
                    any(getattr(self.modes, k) != v for k, v in self._flags.items()) or
                    tuple(getattr(self.modes, 'implementation_hooks_720', ())) != self._central_hooks or
                    getattr(self.modes, 'implementation_identity_720', None) != self._central_hooks_identity):
                raise RuntimeError('Owned options or accepted constructor flags changed')
            if [name for name in self.ledger.violations() if name != '_installed']:
                raise RuntimeError('Owned Swin method/signature/contract changed')
            self._require_signature()
            self._require_installed_scope()
            self.sources.verify_symbols()
            for owner, name, stamp in self.weight_slots:
                value = getattr(owner, name)
                if value is not stamp[0] or self.stamp(value)[1:] != stamp[1:]:
                    raise RuntimeError(f'Owned Swin tensor changed: {name}')
            for site, block in self.targets.values():
                outer = getattr(self.model, site.side)[site.group][site.index]
                actual = outer.body if site.decoder_body else outer
                mlp_row = self.by_mlp.get(id(block.mlp))
                if actual is not block or tuple(block.window_shift) != site.shift or mlp_row is None or mlp_row[1] is not block:
                    raise RuntimeError(f'Actual local site route changed: {site.label}')
            for role, kernel in self.compiled.items():
                if (kernel.kernel is not self.payloads[role] or kernel.hash != self.resources[role]['kernel_hash']
                        or kernel.n_spills != 0):
                    raise RuntimeError('Owned Swin loaded binary/resource changed')
            for rows, (kernel, payload, digest) in self.hidden_binaries.items():
                if (self._hidden._compiled.get(rows) is not kernel or kernel.kernel is not payload or
                        kernel.hash != digest or kernel.n_spills != 0 or kernel.src.fn is not self.kernels._c32_mlp):
                    raise RuntimeError('Owned C32 MLP binary changed')
            return True
        except BaseException:
            self.session._failed = True
            raise

    validate_frame_context = validate

    def preflight(self):
        from decoder_input_full_k_720_v1 import _screen
        from spill_preflight_v1 import select
        self.validate()
        self.sources.verify()
        if self.graph.entries or self.torch.xpu.is_current_stream_capturing():
            raise RuntimeError('Swin admission requires a fresh idle graph')
        if self.preflight_complete:
            return dict(self.resources)
        with self.torch.inference_mode(False):
            for site, block in self.targets.values():
                key = self._key(site)
                if key in self.selected:
                    continue
                if not self.qkv_enabled and id(block) not in self.body_targets:
                    continue
                c, hp, wp = site.channels, *site.padded_shape[:2]
                sy, sx = site.shift
                mlp = self.torch.empty((hp, wp, c), dtype=self.torch.float16, device=self._device)
                shape = (c // 32, hp // 8, wp // 8, 64, 32)
                q, k, v = [self.torch.empty(shape, dtype=self.torch.float16, device=self._device) for _ in range(3)]
                att = block.attention
                front = att.front if c == 32 else att
                qbm = 32 if c in (32, 256) else 16
                qjit, qargs, qgrid, qlaunch = self._qkv_spec(site, block, mlp, (q, k, v))
                kernel, row = _screen(qjit, qargs, qgrid,
                                      qlaunch)
                self._admit(('qkv', key), kernel, row, qargs, qgrid)
                if key not in self.tail_keys:
                    self.selected[key] = (qbm, None)
                    continue
                output = self.torch.empty(site.input_shape, dtype=self.torch.float16, device=self._device)
                bms = (32, 16) if self.options.prefer_bm32 else (16, 32)
                bns = (c,) if c <= 128 else (256, 128, 64)
                configs = [(bm, bn) for bn in bns for bm in bms]
                args_for = lambda cfg: (mlp, q, k, v, att.bias, block.output_weight, block.skip_scale,
                    att.pixel_order, output, hp, wp, site.height, site.width, sy, sx, c, *cfg,
                    self.options.denominator == 'fp32', self.options.wide_score,
                    self.options.wide_value, self.options.crop_queries)
                grid_for = lambda cfg: (64 // cfg[0], hp * wp // 64, (c + cfg[1] - 1) // cfg[1])
                chosen, kernel, selection = select(self.kernels._attention_project, configs, args_for, grid_for,
                    num_warps=4, num_stages=1, enable_fp_fusion=False)
                # _screen hashes the exact loaded binary that will be dispatched.
                screened, row = _screen(self.kernels._attention_project, args_for(chosen), grid_for(chosen),
                    dict(num_warps=4, num_stages=1, enable_fp_fusion=False))
                if screened is not kernel:
                    raise RuntimeError('Tail selection changed its compiled binary')
                row['selection'] = selection
                self._admit(('tail', key), kernel, row, args_for(chosen), grid_for(chosen))
                self.selected[key] = (qbm, chosen)
        if self.options.mlp_wide:
            self._hidden.preflight()
            for _, shape, family in self._hidden.targets.values():
                if family == 'c32':
                    rows = shape[0] * shape[1]
                    kernel = self._hidden._compiled[rows]
                    self.hidden_binaries[rows] = (kernel, kernel.kernel, kernel.hash)
        self.preflight_complete = True
        return dict(self.resources)

    def _admit(self, role, kernel, row, args, grid):
        self.compiled[role], self.payloads[role] = kernel, kernel.kernel
        self.resources[role] = dict(row, source_kernel=kernel.src.fn.fn.__module__ + '.' + kernel.src.fn.fn.__name__)

    def _launch(self, role, args, grid):
        kernel, row = self.compiled[role], self.resources[role]
        if (kernel.kernel is not self.payloads[role] or kernel.hash != row['kernel_hash'] or kernel.n_spills != 0):
            raise RuntimeError('Dispatch missed its admitted loaded binary')
        kernel[tuple(grid) + (1,) * (3 - len(grid))](*args)
        self.kernel_hits[role] = self.kernel_hits.get(role, 0) + 1
        if self.torch.xpu.is_current_stream_capturing():
            self.capture_kernel_hits[role] = self.capture_kernel_hits.get(role, 0) + 1
        return kernel.hash

    def _qkv_spec(self, site, block, mlp, outputs):
        c, hp, wp = site.channels, *site.padded_shape[:2]
        att = block.attention
        front = att.front if c == 32 else att
        bm = 32 if c in (32, 256) else 16
        if self.qkv_enabled:
            n = 64 if self.options.qkv_wide else 32
            args = (mlp, front.qkv, front.scale, att.pixel_order, *outputs,
                    hp * wp, wp, c, bm, self.options.qk_norm == 'fp32', self.options.qkv_wide)
            return (self.kernels._qkv, args, ((hp * wp + bm - 1) // bm, (3 * c + n - 1) // n),
                    dict(num_warps=4, num_stages=2, enable_fp_fusion=False))
        # Baseline producer specialization for independent attention/P09 arms.
        if c == 32:
            import fused_c32_projection_native_half_v1 as baseline
            args = (mlp, front.qkv, front.scale, att.pixel_order, *outputs, wp, wp // 8, bm, False)
            return baseline._kernel, args, (hp * wp // bm, 3), dict(num_warps=4, num_stages=1, enable_fp_fusion=False)
        if c == 64:
            import c64_qkv_direct_pack_v2 as baseline
            jit = baseline._qkv_direct_pack
        elif c == 128:
            import c128_qkv_direct_pack_one_v1 as baseline
            jit = baseline._direct_pack
        else:
            import c256_qkv_direct_pack_one_v1 as baseline
            jit = baseline._direct_pack
        args = (mlp, front.qkv, front.scale, att.pixel_inverse, *outputs,
                hp * wp, wp, hp * wp // 64, bm, 32, 32, False)
        return jit, args, ((hp * wp + bm - 1) // bm, 3 * (c // 32)), dict(num_warps=4, num_stages=2, enable_fp_fusion=False)

    def qkv_override(self, features, module=None, *, weight=None):
        row = self.by_attention.get(id(module)) if module is not None else self.by_qkv.get(id(weight))
        if row is None or not self.qkv_enabled:
            return None
        site, block = row
        if (not self.preflight_complete or not self.active or self.retired or not self.in_frame or
                tuple(features.shape) != site.padded_shape or not features.is_contiguous() or
                features.dtype != self.torch.float16 or features.device != self._device or
                self.current_arithmetic_backend() != 'triton'):
            raise RuntimeError('Owned direct QKV boundary/asset scope changed')
        c, hp, wp = site.channels, *site.padded_shape[:2]
        shape = (c // 32, hp // 8, wp // 8, 64, 32)
        outputs = tuple(self.torch.empty(shape, dtype=self.torch.float16, device=self._device) for _ in range(3))
        _, args, grid, _ = self._qkv_spec(site, block, features, outputs)
        role = ('qkv', self._key(site))
        digest = self._launch(role, args, grid)
        self.actualkernelhash_by_site.setdefault(site.label, {})['qkv'] = digest
        self.provider.record('audit_swin_qkv')
        self._record_site_kernel(site.label, 'qkv', digest)
        if c == 32:
            return tuple(value[0] for value in outputs), self.compiled[role]
        return outputs

    def c32_qkv_override(self, features, weight, scale, order, round_qkv):
        row = self.by_qkv.get(id(weight))
        if row is None or not self.qkv_enabled:
            return None
        site, block = row
        if (site.channels != 32 or weight is not block.attention.front.qkv or
                scale is not block.attention.front.scale or order is not block.attention.pixel_order or
                round_qkv is not False):
            raise RuntimeError('Owned C32 QKV caller changed its fixed weight/scale/order policy')
        return self.qkv_override(features, weight=weight)

    def _record_site_kernel(self, label, role, digest):
        row = self.site_kernel_hits.setdefault(label, {}).setdefault(role, {'calls': 0, 'capture_calls': 0, 'hash': digest})
        if row['hash'] != digest:
            raise RuntimeError('Actual site producer changed loaded binary')
        row['calls'] += 1
        key = label + ':' + role
        self.calls[key] = self.calls.get(key, 0) + 1
        if self.torch.xpu.is_current_stream_capturing():
            row['capture_calls'] += 1
            self.capture_calls[key] = self.capture_calls.get(key, 0) + 1

    def record_mlp_hit(self, module, kernels):
        """Notification only after the real screened MLP kernels dispatched."""
        site, block = self.by_mlp[id(module)]
        if (not self.active or self.retired or not self.in_frame or block.mlp is not module or
                self.session.__dict__.get('_audit_swin_720_owner') is not self):
            raise RuntimeError('Owned MLP receipt lacks its actual active body/serial frame')
        if site.channels == 32:
            rows = site.padded_shape[0] * site.padded_shape[1]
            if len(kernels) != 1 or kernels[0] is not self._hidden._compiled.get(rows) or kernels[0].src.fn is not self.kernels._c32_mlp:
                raise RuntimeError('C32 receipt missed the actual screened local native producer')
            resources = (self._hidden.resources[str(rows)].get('binary_proof'),)
            admitted = self.hidden_binaries.get(rows)
            if admitted is None or kernels[0].kernel is not admitted[1] or kernels[0].hash != admitted[2]:
                raise RuntimeError('C32 MLP payload changed since cold admission')
        else:
            suite = getattr(self.modes, '_numeric_cleanup_owner', None)
            branch = getattr(suite, 'children', {}).get('branch_accum')
            if (branch is None or not branch.preflight_complete or branch.audit_swin is not self or not branch._wide):
                raise RuntimeError('Shared branch receipt lacks its actual preflighted numeric child')
            first, second = branch._roles(site.padded_shape)
            if (len(kernels) != 2 or kernels[0] is not branch._compiled.get(first) or
                    kernels[1] is not branch._compiled.get(second) or
                    kernels[0].src.fn is not self.kernels._mlp_pairs or kernels[1].src.fn is not self.kernels._mlp_project):
                raise RuntimeError('Wide MLP receipt differs from actual compiled producer/consumer')
            resources = (branch.resources.get(first), branch.resources.get(second))
            if any(kernel.kernel is not branch._binary_payloads.get(role)
                   for kernel, role in zip(kernels, (first, second))):
                raise RuntimeError('Wide MLP payload changed since cold admission')
        if any(not isinstance(row, dict) or row.get('spills') != 0 or row.get('binary_bytes', 0) <= 0 or
               row.get('kernel_hash') != kernel.hash or kernel.n_spills != 0
               for kernel, row in zip(kernels, resources)):
            raise RuntimeError('MLP lacks actual cold compiled/resource/binary evidence')
        old = self.mlp_binary_evidence_by_site.get(site.label)
        hashes = tuple(kernel.hash for kernel in kernels)
        if old is None:
            self.mlp_binary_evidence_by_site[site.label] = {'hashes': hashes,
                'resources': deepcopy(resources)}
            self._mlp_payloads_by_site[site.label] = tuple((kernel, kernel.kernel, kernel.hash) for kernel in kernels)
        elif old['hashes'] != hashes:
            raise RuntimeError('Site MLP changed its admitted loaded binary')
        self._record_site_kernel(site.label, 'mlp', '|'.join(hashes))

    def _role_proof(self, label, role):
        if not self.preflight_complete:
            return False
        if role == 'mlp':
            evidence = self.mlp_binary_evidence_by_site.get(label)
            return bool(evidence and all(row['spills'] == 0 and row['binary_bytes'] > 0
                                         for row in evidence['resources']) and
                        all(kernel.kernel is payload and kernel.hash == digest and kernel.n_spills == 0
                            for kernel, payload, digest in self._mlp_payloads_by_site.get(label, ())) and
                        label in self._mlp_payloads_by_site)
        site = next(site for site, _ in self.targets.values() if site.label == label)
        index = (role, self._key(site))
        kernel, resource = self.compiled.get(index), self.resources.get(index)
        return bool(kernel is not None and resource is not None and resource.get('spills') == 0 and
                    resource.get('binary_bytes', 0) > 0 and kernel.hash == resource['kernel_hash'] and
                    kernel.kernel is self.payloads[index] and kernel.n_spills == 0)

    def capture_role_replacements(self, before):
        """Partial replacement proofs. Old remaining points are main's obligation.

        Keys in replaced_sites are the original before-counter keys. New deltas
        use actual role counters, never incrementing a bypassed legacy counter.
        Each new path also needs its cold loaded-binary/resource proof.
        """
        prior = before.get('own', {}).get('audit_swin_720_v1', {}) or {}
        combo = before.get('combo', {}) or {}
        result = {}
        def baseline_dict(*names):
            return next((before[name] or {} for name in names if name in before), {})
        def publish(gate, old, replacements, role, current_old):
            selected = {str(key): label for key, label in replacements.items() if key in old or str(key) in old}
            if not selected:
                return
            deltas = {label: self.capture_calls.get(label + ':' + role, 0) - prior.get(label + ':' + role, 0)
                      for label in selected.values()}
            remaining = {str(key): current_old.get(key, 0) - value for key, value in old.items()
                         if str(key) not in selected}
            result[gate] = {'passed': all(value == 1 and self._role_proof(label, role) for label, value in deltas.items()),
                'replaced_sites': list(selected), 'new_sites': deltas, 'remaining_old_sites': remaining,
                'new_counter_keys': {label: label + ':' + role for label in deltas},
                'role': role, 'compiled_resource_proof': {label: self._role_proof(label, role) for label in deltas}}
        for channels, gate, field in ((32, 'c32_all_seven', 'c32_by_block'), (64, 'c64_all_eight', 'c64_by_block'),
                                     (128, 'c128_all_twelve', 'c128_by_block'), (256, 'c256_all_sixteen', 'c256_by_block')):
            replacements = {}
            for site, block in self.body_targets.values():
                if site.channels != channels:
                    continue
                if channels == 32:
                    key = site.c32_combo_key
                    if key is None:
                        continue
                else:
                    key = f'{site.side}[{site.index}]'
                replacements[key] = site.label
            current = self._combo['c32'] if channels == 32 else self._combo[f'c{channels}']['by_block']
            publish(gate, combo.get(field, {}) or {}, replacements, 'tail', current)
        if self.options.mlp_wide:
            hidden_old = baseline_dict('before_c32_hidden', 'before_c32_hidden_native')
            hidden_new = {site.label.removesuffix('.body'): site.label for site, _ in self.targets.values() if site.channels == 32}
            publish('c32_hidden_native_ten', hidden_old, hidden_new, 'mlp', self._hidden.capture_calls)
            pair_old = baseline_dict('before_pairwise', 'before_c128_pairwise')
            pair_new = {f'{site.side}-128-{site.index}': site.label for site, _ in self.targets.values() if site.channels == 128}
            pair_counter = self.modes.c128_pairwise_calls
            publish('c128_pairwise_twelve', pair_old, pair_new, 'mlp', pair_counter.capture_calls)
        for channels, gate, names in ((64, 'c64_attention_project_eight', ('before_c64_attention', 'before_c64_attention_project')),
                                      (128, 'c128_attention_project_twelve', ('before_c128_attention', 'before_c128_attention_project'))):
            replacements = {f'{site.side}[{site.index}]': site.label for site, _ in self.body_targets.values()
                            if site.channels == channels}
            publish(gate, baseline_dict(*names), replacements, 'tail', self._family_receipts[channels].capture_calls)
        # Some new paths, notably the first decoder32 body and standalone QKV,
        # have no old gate of their own. Still require their actual fresh hit.
        admitted = {}
        for site, block in self.targets.values():
            roles = (['mlp'] if self.options.mlp_wide else [])
            if self.qkv_enabled or id(block) in self.body_targets:
                roles.append('qkv')
            if id(block) in self.body_targets:
                roles.append('tail')
            for role in roles:
                key = site.label + ':' + role
                delta = self.capture_calls.get(key, 0) - prior.get(key, 0)
                admitted[key] = {'delta': delta, 'compiled_resource_proof': self._role_proof(site.label, role)}
        self.last_capture_proof = admitted
        if any(row['delta'] != 1 or not row['compiled_resource_proof'] for row in admitted.values()):
            self.session._failed = True
            raise RuntimeError('Owned new capture role missed a cold admitted kernel or its actual single site hit')
        return result

    def _body(self, site, block, features):
        try:
            if (not self.preflight_complete or not self.active or self.retired or not self.in_frame or
                    self.current_arithmetic_backend() != 'triton' or
                    tuple(features.shape) != site.input_shape or features.dtype != self.torch.float16 or
                    features.device != self._device or not features.is_contiguous()):
                raise RuntimeError(f'Local input boundary changed: {site.label}')
            # No shifted input allocation in fused-gather mode. Each MLP producer
            # and its residual projection consume the same logical zero view.
            if self.options.fuse_pad and self.options.mlp_wide:
                x = features
            else:
                sy, sx = site.shift
                x = self.torch.nn.functional.pad(features, (0, 0, sx, sx, sy, sy)) if sy or sx else features
            mlp = block.mlp.forward_unquantized(x) if site.channels == 32 else block.mlp(x)
            if tuple(mlp.shape) != site.padded_shape or not mlp.is_contiguous():
                raise RuntimeError(f'MLP consumer layout changed: {site.label}')
            c, hp, wp = site.channels, *site.padded_shape[:2]
            key = self._key(site)
            qbm, (bm, bn) = self.selected[key]
            att = block.attention
            front = att.front if c == 32 else att
            packed_shape = (c // 32, hp // 8, wp // 8, 64, 32)
            q, k, v = [self.torch.empty(packed_shape, dtype=self.torch.float16, device=self._device) for _ in range(3)]
            _, qargs, qgrid, _ = self._qkv_spec(site, block, mlp, (q, k, v))
            qhash = self._launch(('qkv', key), qargs, qgrid)
            result = self.torch.empty(site.input_shape, dtype=self.torch.float16, device=self._device)
            thash = self._launch(('tail', key), (mlp, q, k, v, att.bias, block.output_weight, block.skip_scale,
                att.pixel_order, result, hp, wp, site.height, site.width, *site.shift, c, bm, bn,
                self.options.denominator == 'fp32', self.options.wide_score, self.options.wide_value,
                self.options.crop_queries), (64 // bm, hp * wp // 64, (c + bn - 1) // bn))
            self.actualkernelhash_by_site[site.label] = {'qkv': qhash, 'tail': thash}
            self._record_site_kernel(site.label, 'qkv', qhash)
            self._record_site_kernel(site.label, 'tail', thash)
            self._receipts(site, block, features, mlp)
            self.calls[site.label] += 1
            if self.torch.xpu.is_current_stream_capturing():
                self.capture_calls[site.label] += 1
            return result
        except BaseException:
            self.session._failed = True
            raise

    def _receipts(self, site, block, features, mlp):
        c = site.channels
        # Actual new launches have distinct provider roles. Do not increment
        # the bypassed combo/old fused-family kernel counters.
        self.provider.record('audit_swin_qkv')
        self.provider.record('audit_swin_attention_project')
        if c != 32:
            self.window.calls.append(dict(module=self.window.modules[id(block)],
                shape=list(features.shape), padded_shape=list(mlp.shape), shift=list(site.shift),
                fusion='audit_swin_720', owned_site=site.label))

    @contextmanager
    def capture(self):
        before = dict(self.capture_calls)
        yield self
        required = {site.label for site, _ in self.body_targets.values()}
        missed = {key: self.capture_calls[key] - value for key, value in before.items() if key in required
                  if self.capture_calls[key] - value != 1}
        if missed:
            self.session._failed = True
            raise RuntimeError(f'Owned 44-site capture incomplete: {missed}')

    @contextmanager
    def frame_scope(self):
        """Enter under main's same process-lock frame transaction, outside replay."""
        if self.in_frame:
            self.session._failed = True
            raise RuntimeError('Nested owned Swin frame')
        self.validate()
        self.in_frame = True
        try:
            yield self
        except BaseException:
            self.session._failed = True
            raise
        finally:
            self.in_frame = False

    def snapshot(self):
        if self.active:
            self.validate()
        self.sources.verify()
        return {'identity': self._identity, 'options': asdict(self.options),
                'active': self.active, 'retired': self.retired,
                'sites': [asdict(site) for site, _ in self.targets.values()],
                'calls': dict(self.calls), 'capture_calls': dict(self.capture_calls),
                'resources': {str(k): v for k, v in self.resources.items()},
                'actualkernelhash_by_site': dict(self.actualkernelhash_by_site),
                'site_kernel_hits': self.site_kernel_hits,
                'mlp_binary_evidence_by_site': self.mlp_binary_evidence_by_site,
                'last_capture_proof': getattr(self, 'last_capture_proof', {}),
                'kernel_hits': {str(k): v for k, v in self.kernel_hits.items()},
                'capture_kernel_hits': {str(k): v for k, v in self.capture_kernel_hits.items()},
                'sources': dict(self.sources.receipts), 'preflight_complete': self.preflight_complete,
                'python_counts_exclude_replays': True, 'gpu_validation': 'external-required'}

    def transfer_serial_thread(self, suite, *, game_adapter, serial_guard, previous_thread):
        """Additional participant in the actual numeric suite process-lock transaction.

        Main must call this directly inside NumericCleanupCounter.transfer_serial_thread
        and include self.thread in its existing atomic rollback, after bridge binding.
        Ordinary validate() does not accept another thread. No stream is moved here.
        """
        import sys
        try:
            numeric = sys.modules.get('numeric_cleanup_suite_720_v1')
            caller = sys._getframe(1)
            if (numeric is None or type(suite) is not numeric.NumericCleanupCounter or
                    caller.f_code is not numeric.NumericCleanupCounter.transfer_serial_thread.__code__ or
                    caller.f_locals.get('self') is not suite or
                    suite.modes is not self.modes or suite.session is not self.session or suite.graph is not self.graph):
                raise RuntimeError('Swin transfer requires the actual owned numeric transaction')
            host = getattr(game_adapter, 'host', None)
            if (game_adapter is not sys.modules.get('cyberpunk_nr_adapter') or
                    host is not sys.modules.get('nr_game_pre_xess_host') or
                    getattr(host, '_modes', None) is not self.modes or getattr(host, '_failed', True) or
                    type(serial_guard) is not type(RLock()) or
                    serial_guard is not getattr(game_adapter, '_process_serial_lock', None) or
                    serial_guard is getattr(game_adapter, '_settings_lock', None) or not serial_guard._is_owned() or
                    type(previous_thread) is not int or self.thread != previous_thread or self.in_frame or
                    self.torch.xpu.is_current_stream_capturing()):
                raise RuntimeError('Swin transfer lacks the real held process lock/idle previous owner')
            current = get_ident()
            bridge = getattr(host, '_bridge', None)
            if (bridge is None or host._thread != current or bridge.thread != current or
                    bridge.torch is not self.torch or
                    bridge.device != self.torch.device('xpu', self.torch.xpu.current_device()) or
                    bridge.torch.xpu.current_stream().sycl_queue != bridge.stream.sycl_queue):
                raise RuntimeError('Swin transfer requires the actual already-bound bridge device/stream')
            self.validate(check_thread=False)
            self.thread = current
            return current
        except BaseException:
            self.session._failed = True
            self.session._audit_swin_720_retired_owner = self
            raise


@contextmanager
def installed(modes, *, preflight=True, **kwargs):
    """before_numeric hook; numeric suite exits before this persistent scope."""
    options = Options(**kwargs)
    session = getattr(modes, 'session', None)
    required = ('c512_qkv_library_720', 'native_k8_720', 'c128_pairwise_720',
                'c64_attention_project_720', 'c128_attention_project_720', 'c32_hidden_native_720',
                'decoder_gather_unround_720', 'c512_probability_unround_720')
    if (session is None or modes.height != 720 or tuple(modes.source) != (720, 1280) or
            modes.variant != 'unrounded' or not all(getattr(modes, k, False) for k in required) or
            getattr(modes, 'c128_dual_qkv_720', False) or session._stack.graph.entries or
            '_numeric_cleanup_owner' in modes.__dict__ or '_audit_swin_720_owner' in session.__dict__):
        raise RuntimeError('Enter the owned hook after accepted720 scopes, before numeric/capture')
    families = getattr(getattr(modes, 'numeric_cleanup_720', None), 'branch_accum_families', ())
    if options.mlp_wide and set(families) != {'c64', 'c128', 'c256'}:
        raise RuntimeError('Owned wide MLP requires all three guarded BranchAccum routes')
    counter = Counter(modes, options)
    primary = None
    try:
        counter.ledger.attribute(session, '_audit_swin_720_owner', counter)
        @contextmanager
        def frame_installed():
            with counter.frame_scope(), counter._saved_installed() as original_scope:
                yield original_scope
        frame_installed.__nr_numeric_original_installed__ = counter._saved_installed
        counter._scope_wrapper = frame_installed
        counter.ledger.attribute(session, '_installed', frame_installed)
        original = counter.window.apply
        def selected_apply(self, module, features):
            row = counter.body_targets.get(id(module))
            return original(module, features) if row is None else counter._body(row[0], row[1], features)
        if any(site.channels != 32 for site, _ in counter.body_targets.values()):
            counter.ledger.attribute(counter.window, 'apply', MethodType(selected_apply, counter.window))
        for site, block in counter.body_targets.values():
            if site.channels == 32:
                def c32_forward(self, features, *, _site=site):
                    return counter._body(_site, self, features)
                counter.ledger.attribute(block, 'forward_unquantized', MethodType(c32_forward, block))
        if counter.qkv_enabled:
            for module in counter.callback_modules:
                if module._AUDIT_LOCAL is not None:
                    raise RuntimeError('Direct QKV callback already has an owner')
                counter.ledger.attribute(module, '_AUDIT_LOCAL', counter)
        # Only scope-owned contracts, never global helper/schema edits.
        for name, outputs in (('_mlp_pairs', ('LATENT',)), ('_mlp_project', ('OUT',)),
                              ('_c32_mlp', ('OUT',)), ('_qkv', ('Q', 'K', 'V')),
                              ('_attention_project', ('OUT',))):
            counter.ledger.contract(counter.dataflow.CONTRACTS, counter.kernels.__name__ + '.' + name,
                                    (outputs, ()))
        # The main NumericCleanupCounter owns the one graph signature. Its
        # identity already includes all frozen implementation HookSpecs.
        if options.mlp_wide:
            counter._hidden.configure_local(counter)
        counter.active = True
        if preflight:
            counter.preflight()
        yield counter
    except BaseException as exc:
        primary = exc
        session._failed = True
        raise
    finally:
        unwind_error = None
        counter.retired = True
        counter.active = False
        # A graph must not outlive owned methods/constants. Numeric suite should
        # already have retired it; otherwise close here before releasing hooks.
        if counter.graph.entries:
            session._failed = True
            session._audit_swin_720_retired_owner = counter
            try:
                counter.graph.close()
            except BaseException as exc:
                if primary is not None:
                    primary.add_note('Owned graph.close failed: ' + repr(exc))
                else:
                    unwind_error = exc
        if counter.graph.entries:
            # Keep all producer/consumer assets anchored if graph retirement
            # failed. Releasing methods/constants here would leave live graphs.
            message = 'Owned graph still live; producer/consumer scopes retained on failed session'
            if primary is not None:
                primary.add_note(message)
            else:
                unwind_error = unwind_error or RuntimeError(message)
        else:
            try:
                if options.mlp_wide:
                    counter._hidden.release_local(counter)
            except BaseException as exc:
                session._failed = True
                if primary is not None:
                    primary.add_note('Owned C32 release failed: ' + repr(exc))
                else:
                    unwind_error = unwind_error or exc
            try:
                counter.ledger.restore(session, primary or unwind_error)
            except BaseException as exc:
                unwind_error = unwind_error or exc
            session.__dict__.pop('_audit_swin_720_retired_owner', None)
        if primary is None and unwind_error is not None:
            raise unwind_error
