"""Independent default-off C64/C128/C256 branch accumulation candidates.

installed(modes, branch_accum={'c64': bool, 'c128': bool, 'c256': bool}) yields
a counter with preflight()/validate()/snapshot(), all without mandatory args.
None means all off and imports no tensor/model/kernel runtime. Enable after
actual720 select and before capture. Keep scope through fused-history graph
replay; leaving it retires the session and its candidate graphs.

Only CPU AST validation has been performed. The actual binary, zero-spill,
capture, numerical, timing and continuous-history video gates remain untested.
The sole numerical change is FP32 accumulation of each original per-branch
FP32 projection, in increasing branch order, before one final half. Retain
latent half, half skip initial and HWC addresses. C128's currently installed
pairwise expansion schedule and its capture receipts are preserved.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
import sys
from types import MethodType

if __package__:
    from .decoder_input_full_k_720_v1 import (
        _ReferenceCounter, _SessionCounter, _closure, _function, _nested_code, _same, _screen, _tensor)
else:
    from decoder_input_full_k_720_v1 import (
        _ReferenceCounter, _SessionCounter, _closure, _function, _nested_code, _same, _screen, _tensor)


_PINS = {
    "native_cubic_adapters_v1": "756a0a6978401dbe24aedc250296aab7b3265d9be0911aa61818ec5f58ded348",
    "batched_branched_mlp_v1": (
        "54b6a47595a53c0255cd69d41f9a6725626810b624ca0d328e91e68496d479fc",
        "4ecfc24bfc7cf9488953265cfd1a82ce42d5bd138c0674a4abf02ac8f776f6af"),
    "batched_branched_mlp_v2": "077ee397b477b7bbe63b9bf35169d08f57659cd9fb965498509c9a2d31f11207",
    "fused_branched_pairs_v1": "0eeee74e7c832f1c6e1294f12e8c91289c8f4cb3132db8c5bd2013391e437a57",
    "nr_backend.multihead_block": "12cac3afe285c68ae248a9f0a560421ba5be8a2c902afc8a505f80da6571290d",
    "branched_mlp_pairwise_720_v1": "3ba35f667b29a9d42600efbd1e358161062debc0f1a220fe95acae1a7238cd97",
    "branched_mlp_pairwise_family_720_v1": "97d54497814e647ea099b6d8e27c19a4081fc0423616c873c5c9eaa1658f6bc6",
    "native_half_cubic_v1": "d16a331f3a9d294d6e440104f6a5f681f81def067a72f1c7f8d46a9cbf33cdda",
    "cubic_lut_constant_v1": "1c2b4ef3a43124f42e11078ef74cbab661cdcd71ab64cfbdb28881676d361afe",
}
_BASE = {64: (192, 320), 128: (96, 160), 256: (48, 80)}
_EXPECTED = {64: 8, 128: 12, 256: 16}


class BranchAccumCounter(_SessionCounter):
    serial_child_label = "branch_accum"
    launches_saved = 0
    numerical_contract = ("original half latent and half skip initial; each branch dot independently "
                          "accumulates in FP32; increasing branch FP32 additions; final half; HWC unchanged")

    def __init__(self, modes, selected):
        super().__init__(modes, {"branch_accum": dict(selected)})
        self.sources.add(sys.modules[__name__])
        # Authenticate the shared helper's actual imported candidate file as well.
        self.sources.add(sys.modules[_SessionCounter.__module__])
        self.dependencies = {name: self.sources.load(name, sha256=digest)
                             for name, digest in _PINS.items()}
        self.adapters = self.dependencies["native_cubic_adapters_v1"]
        self.batched = self.dependencies["batched_branched_mlp_v1"]
        self.multihead = self.dependencies["nr_backend.multihead_block"]
        self.pairwise = self.dependencies["branched_mlp_pairwise_720_v1"]
        self.family = self.dependencies["branched_mlp_pairwise_family_720_v1"]
        self.pairs_base = self.dependencies["fused_branched_pairs_v1"]
        self.kernels = self.sources.load((__package__ + "." if __package__ else "") +
                                        "branch_accum_native_720_kernel_v1")
        self.audit_swin = getattr(self.session, "_audit_swin_720_owner", None)
        self._wide = bool(self.audit_swin is not None and self.audit_swin.mlp_enabled)
        self.wide_kernels = None
        self._latent_tiles = {}
        self._project_tiles = {}
        if self._wide:
            audit_module = self.sources.load("audit_swin_720_v1")
            if (type(self.audit_swin) is not audit_module.Counter or
                    self.audit_swin.session is not self.session or not self.audit_swin.active):
                raise RuntimeError("Wide MLP requires the active owned before-numeric scope")
            self.wide_kernels = self.sources.load("audit_swin_720_kernels_v1")
            if self.wide_kernels is not self.audit_swin.kernels:
                raise RuntimeError("Wide MLP kernel module differs from the cold admitted hook")
            for name in ("_mlp_pairs", "_mlp_project", "_load_input"):
                self.callees["owned_wide" + name] = self.sources.function(getattr(self.wide_kernels, name))
                self.sources.watch(self.wide_kernels, name)
        matches = [item for item in self.stack.components
                   if type(item) is self.adapters.FusedBatched]
        if len(matches) != 1 or matches[0].provider is not self.provider:
            raise RuntimeError("Expected exactly the model's active FusedBatched owner")
        self.owner = matches[0]
        self.all_targets = self._targets()
        if self.owner.modules != set(self.all_targets):
            raise RuntimeError("FusedBatched ownership differs from the 36 real model MLPs")
        self.targets = {key: row for key, row in self.all_targets.items()
                        if selected[f"c{row[1].channels}"]}
        self.calls = {row[0]: 0 for row in self.targets.values()}
        self.capture_calls = dict(self.calls)
        self._old_apply = self.owner.apply
        self._pairwise_counter = None
        self._schedule(record=True)
        self._pairwise_mode = modes.c128_pairwise_720
        self._configs = deepcopy(self.owner.configurations)
        expected_configs = self.dependencies["batched_branched_mlp_v2"].configurations("small")
        if self._configs != expected_configs:
            raise RuntimeError("Preserve the reviewed actual720 batched launch tiles; unknown configuration")
        self.callees["baseline_project"] = self.sources.function(self.batched._project)
        self.callees["candidate_project"] = self.sources.function(self.kernels._project_fp32)
        self.callees["c64_c256_expansion"] = self.sources.function(self.batched._pairs)
        self.callees["c128_expansion"] = self.sources.function(
            self.pairwise._pairs_pairwise if self._pairwise_counter is not None else self.batched._pairs)
        cubic = self.dependencies["native_half_cubic_v1"].cubic
        if self.batched.cubic is not cubic or self.pairwise.cubic is not cubic:
            raise RuntimeError("Branch latent producer must retain the reviewed native half cubic helper")
        self.callees["latent_cubic"] = self.sources.function(cubic)
        self.sources.watch(self.batched, "cubic")
        self.sources.watch(self.pairwise, "cubic")
        self.callees["owned_apply"] = self.sources.function(self._old_apply)
        self.sources.watch(self.adapters.FusedBatched, "apply")
        self.sources.watch(self.batched, "_project")
        self.sources.watch(self.batched, "_pairs")
        self.sources.watch(self.pairwise, "_pairs_pairwise")
        self.sources.watch(self.kernels, "_project_fp32")
        for jit in (self.batched._pairs, self.pairwise._pairs_pairwise, self.kernels._project_fp32):
            self.sources.watch(jit, "fn")
        self._forward_code = _nested_code(self.pairs_base.FusedPairs.installed,
                                         "FusedPairs.installed.<locals>.<lambda>")
        self._constant = self.owner.constant
        self._lut = self._constant.require()
        self._save_weight("model.cubic_lut", self._constant, "value")
        self.device = self._lut.device
        if (self._lut.dtype != self.torch.int16 or tuple(self._lut.shape) != (65536,) or
                self.device.type != "xpu" or not self._lut.is_contiguous()):
            raise RuntimeError("Expected the unchanged immutable owned half-bit cubic LUT")
        for label, module, shape, block in self.all_targets.values():
            c, branches = module.channels, module.channels // 32
            for attribute, weight_shape in (("expand", (branches, c, 128)),
                                             ("reduce", (branches, 128, 32)),
                                             ("project", (branches, 32, c)),
                                             ("skip_scale", (c,))):
                _tensor(self.torch, f"{label}.{attribute}", getattr(module, attribute),
                        weight_shape, self.device)
                self._save_weight(f"{label}.{attribute}", module, attribute)
        self._bound_apply = None
        self.contract = self.kernels.__name__ + "._project_fp32"
        target, name = self.multihead.BranchedMLP, "forward_unquantized"
        self._idle_slots.append((target, name, name in target.__dict__, target.__dict__.get(name),
                                 getattr(target, name)))
        contracts = [self.contract, self.batched.__name__ + "._pairs"]
        if self._pairwise_counter is not None:
            contracts.append(self.pairwise.__name__ + "._pairs_pairwise")
        self._idle_contracts = {key: (key in self.dataflow.CONTRACTS, self.dataflow.CONTRACTS.get(key))
                                for key in contracts}

    def snapshot(self):
        report = super().snapshot()
        report["expansion_modes"] = {"c64": "batched._pairs", "c256": "batched._pairs",
                                      "c128": ("pairwise._pairs_pairwise" if self._pairwise_counter is not None
                                               else "batched._pairs")}
        report["sites"] = {label: {"shape": list(shape), "shift": list(block.window_shift),
                                    "family": f"c{module.channels}", "branches": module.channels // 32,
                                    "launches_per_call": 2}
                           for label, module, shape, block in self.targets.values()}
        report["contracts"][self.batched.__name__ + "._pairs"] = [["LATENT"], []]
        if self._pairwise_counter is not None:
            report["contracts"][self.pairwise.__name__ + "._pairs_pairwise"] = [["LATENT"], []]
        if self._wide:
            report["expansion_modes"] = {f"c{c}": "audit_swin_720_kernels_v1._mlp_pairs:N64:shared2"
                                         for c in (64, 128, 256)}
            report["candidate_project"] = "audit_swin_720_kernels_v1._mlp_project"
            report["fused_zero_gather"] = self.audit_swin.options.fuse_pad
            report["selected_pair_bm"] = dict(self._latent_tiles)
            report["selected_project_bn"] = dict(self._project_tiles)
            report["legacy_pairwise_gate_replaced"] = True
        return report

    def _wide_boundary(self, module, shape):
        if not self._wide:
            return shape, ()
        if (self.session.__dict__.get("_audit_swin_720_owner") is not self.audit_swin or
                not self.audit_swin.active or self.audit_swin.retired or not self.audit_swin.mlp_enabled):
            raise RuntimeError("Owned wide/gather MLP scope changed")
        return self.audit_swin.mlp_boundary(module)

    def _targets(self):
        targets = {}
        for c, expected in _EXPECTED.items():
            start = len(targets)
            for side, group_index in (("encoder", {64: 1, 128: 2, 256: 3}[c]),
                                       ("decoder", {64: 2, 128: 1, 256: 0}[c])):
                group = getattr(self.model, side)[group_index]
                if len(group) != expected // 2:
                    raise RuntimeError(f"Unexpected actual720 {side} C{c} block count")
                for index, outer in enumerate(group):
                    block = outer.body if side == "decoder" and index == 0 else outer
                    if (type(block) is not self.multihead.MultiHeadSwinBlock or block.channels != c or
                            type(block.mlp) is not self.multihead.BranchedMLP or block.mlp.channels != c):
                        raise RuntimeError("Unexpected actual model block/MLP owner")
                    sy, sx = block.window_shift
                    if sy not in (0, 4) or sx not in (0, 4):
                        raise RuntimeError("Unexpected actual720 MLP shift")
                    h, w = _BASE[c]
                    shape = (h + (8 if sy else 0), w + (8 if sx else 0), c)
                    label = f"{side}{c}.{index}"
                    if id(block.mlp) in targets:
                        raise RuntimeError("Duplicate model MLP identity")
                    targets[id(block.mlp)] = (label, block.mlp, shape, block)
            if len(targets) - start != expected:
                raise RuntimeError(f"Expected all {expected} owned C{c} MLPs")
        return targets

    def _schedule(self, *, record=False):
        if _same(self._old_apply, self.adapters.FusedBatched.apply.__get__(self.owner)):
            if self.modes.c128_pairwise_720 or self.modes.c128_pairwise_calls is not None:
                raise RuntimeError("C128 pairwise flag is set but its actual apply owner is missing")
            return
        if record:
            self.sources.function(self._old_apply, module=self.family,
                                  qualname="installed.<locals>.selected_apply")
            self._pairwise_apply_code = _function(self._old_apply).__code__
        elif _function(self._old_apply).__code__ is not self._pairwise_apply_code:
            raise RuntimeError("Reviewed C128 pairwise apply code changed")
        counter = _closure(self._old_apply, "counter")
        base = _closure(self._old_apply, "original")
        by_id = _closure(self._old_apply, "by_id")
        expected = {key: row for key, row in self.all_targets.items() if row[1].channels == 128}
        if (not self.modes.c128_pairwise_720 or counter is not self.modes.c128_pairwise_calls or
                type(counter) is not self.family.FamilyCounter or counter.fused is not self.owner or
                not _same(base, self.adapters.FusedBatched.apply.__get__(self.owner)) or
                set(by_id) != set(expected) or set(counter.targets) !=
                {f"{label.split('128.')[0]}-128-{label.split('.')[-1]}" for label, _, _, _ in expected.values()}):
            raise RuntimeError("Unrecognized C128 pairwise apply/targets; no baseline fallback")
        for key, (label, module, shape, _) in expected.items():
            pair_label = f"{label.split('128.')[0]}-128-{label.split('.')[-1]}"
            if (by_id[key] != (pair_label, module, shape) or
                    counter.targets[pair_label] != (module, shape)):
                raise RuntimeError("C128 pairwise module/geometry ownership changed")
        self._pairwise_counter = counter

    def _validate_callees(self):
        expected_apply = self._bound_apply if self._local else self._old_apply
        current = self._targets()
        if (not _same(self.owner.apply, expected_apply) or current != self.all_targets or
                self.owner not in self.stack.components or self.owner.provider is not self.provider or
                self.owner.modules != set(self.all_targets) or self.owner.configurations != self._configs or
                self.modes.c128_pairwise_720 != self._pairwise_mode or
                self.owner.constant is not self._constant or self._constant.require() is not self._lut or
                (self._pairwise_counter is not None and
                 self.modes.c128_pairwise_calls is not self._pairwise_counter)):
            raise RuntimeError("Owned branch modules, shift, apply, configuration or constant changed")
        if self._pairwise_counter is not None:
            self._schedule()

    def _validate_live_callees(self):
        expected_apply = self._bound_apply if self._local else self._old_apply
        if (not _same(self.owner.apply, expected_apply) or
                self.owner not in self.stack.components or self.owner.provider is not self.provider or
                self.modes.c128_pairwise_720 != self._pairwise_mode or
                self.owner.constant is not self._constant or self._constant.value is not self._lut or
                (self._pairwise_counter is not None and
                 self.modes.c128_pairwise_calls is not self._pairwise_counter)):
            raise RuntimeError("Branch actual apply/provider/resource owner changed")
        if self._local:
            actual = self.multihead.BranchedMLP.forward_unquantized
            contracts = {self.contract: (("OUT",), ()),
                         self.batched.__name__ + "._pairs": (("LATENT",), ())}
            if self._pairwise_counter is not None:
                contracts[self.pairwise.__name__ + "._pairs_pairwise"] = (("LATENT",), ())
            if (_function(actual).__code__ is not self._forward_code or
                    _closure(actual, "self") is not self.owner or
                    any(self.dataflow.CONTRACTS.get(k) != v for k, v in contracts.items())):
                raise RuntimeError("Branch actual temporary forward/apply/contracts owner changed")

    def _roles(self, shape):
        key = "x".join(map(str, shape))
        return key + ":latent", key + ":project"

    def preflight(self):
        """Compile all enabled real geometries; screen both actual launches."""
        try:
            self.validate()
            self._require_session(fresh=True)
            self._preflight_sources()
            if self.torch.xpu.is_current_stream_capturing():
                raise RuntimeError("Branch preflight cannot run during capture")
            if self.preflight_complete:
                return deepcopy(self.resources)
            with self.torch.inference_mode(False):
                for label, module, shape, _ in self.targets.values():
                    first_role, second_role = self._roles(shape)
                    if first_role in self._compiled:
                        continue
                    c, config = module.channels, self._configs[module.channels]
                    m, branches = shape[0] * shape[1], c // 32
                    input_shape, gather = self._wide_boundary(module, shape)
                    sample = self.torch.empty(input_shape, dtype=self.torch.float16, device=self.device)
                    latent = self.torch.empty((branches, m, 32), dtype=self.torch.float16, device=self.device)
                    output = self.torch.empty(shape, dtype=self.torch.float16, device=self.device)
                    first_jit = (self.wide_kernels._mlp_pairs if self._wide else self.pairwise._pairs_pairwise
                                 if c == 128 and self._pairwise_counter is not None else self.batched._pairs)
                    first_args = (sample, module.expand, module.reduce, self._lut, latent,
                                  m, c, config["pair_bm"], False, *gather)
                    second_args = (sample, latent, module.project, module.skip_scale, output,
                                   m, c, config["project_bm"], config["project_bn"], *gather)
                    first_grid = ((m + config["pair_bm"] - 1) // config["pair_bm"], branches // 2 if self._wide else branches)
                    second_grid = ((m + config["project_bm"] - 1) // config["project_bm"],
                                   (c + config["project_bn"] - 1) // config["project_bn"])
                    selected_latent = selected_project = None
                    if self._wide:
                        from spill_preflight_v1 import select
                        self.sources.function(select)
                        latent_configs = list(dict.fromkeys((bm,) for bm in (config["pair_bm"], 16)))
                        latent_args_for = lambda cfg: (sample, module.expand, module.reduce, self._lut, latent,
                            m, c, cfg[0], False, *gather)
                        latent_grid_for = lambda cfg: ((m + cfg[0] - 1) // cfg[0], branches // 2)
                        chosen_latent, selected_latent, latent_selection = select(self.wide_kernels._mlp_pairs,
                            latent_configs, latent_args_for, latent_grid_for, num_warps=4, num_stages=1,
                            enable_fp_fusion=False)
                        self._latent_tiles[first_role] = chosen_latent[0]
                        first_args, first_grid = latent_args_for(chosen_latent), latent_grid_for(chosen_latent)
                        project_configs = [(bn,) for bn in (c, 128, 64) if bn <= c]
                        project_configs = list(dict.fromkeys(project_configs))
                        args_for = lambda cfg: (sample, latent, module.project, module.skip_scale, output,
                            m, c, config["project_bm"], cfg[0], *gather)
                        grid_for = lambda cfg: ((m + config["project_bm"] - 1) // config["project_bm"],
                            (c + cfg[0] - 1) // cfg[0])
                        chosen, selected_project, project_selection = select(self.wide_kernels._mlp_project,
                            project_configs, args_for, grid_for, num_warps=4, num_stages=1,
                            enable_fp_fusion=False)
                        self._project_tiles[second_role] = chosen[0]
                        second_args, second_grid = args_for(chosen), grid_for(chosen)
                    for role, jit, args, grid, stages in (
                            (first_role, first_jit, first_args, first_grid, 1 if self._wide else config["pair_stages"]),
                            (second_role, self.wide_kernels._mlp_project if self._wide else self.kernels._project_fp32,
                             second_args, second_grid, 1)):
                        kernel, resource = _screen(jit, args, grid, dict(
                            num_warps=4, num_stages=stages, enable_fp_fusion=False))
                        if self._wide and role == first_role:
                            if kernel is not selected_latent:
                                raise RuntimeError("Wide latent selection changed its loaded binary")
                            resource["selection"] = latent_selection
                            resource["selected_pair_bm"] = self._latent_tiles[first_role]
                        if self._wide and role == second_role:
                            if kernel is not selected_project:
                                raise RuntimeError("Wide projection selection changed its loaded binary")
                            resource["selection"] = project_selection
                            resource["selected_project_bn"] = self._project_tiles[second_role]
                        resource.update(source_kernel=self.sources.function(jit)["callee"],
                                        input_shape=list(input_shape), input_stride=[input_shape[1] * c, c, 1],
                                        latent_shape=[branches, m, 32], latent_stride=[m * 32, 32, 1],
                                        config=dict(config), round_reduced=False, output_layout="HWC",
                                        shared_branches=2 if self._wide else 1,
                                        fused_zero_gather=bool(self._wide and self.audit_swin.options.fuse_pad))
                        self._compiled[role], self.resources[role] = kernel, resource
                        self._binary_payloads[role] = kernel.kernel
            self.preflight_complete = True
            return deepcopy(self.resources)
        except BaseException:
            self.session._failed = True
            raise

    def _apply(self, owner, module, features):
        try:
            self._require_dispatch()
            target = self.all_targets.get(id(module))
            if owner is not self.owner or target is None or target[1] is not module:
                raise RuntimeError("Unrecognized MLP in owned branch dispatch; no silent fallback")
            label, _, shape, _ = target
            self._require_weights(tuple(f"{label}.{name}" for name in
                                       ("expand", "reduce", "project", "skip_scale")))
            if (not _same(owner.apply, self._bound_apply) or owner.provider is not self.provider or
                    owner.configurations[module.channels] != self._configs[module.channels] or
                    owner.constant is not self._constant or self._constant.require() is not self._lut):
                raise RuntimeError("Branch dispatch callee/configuration/constant changed")
            input_shape, gather = self._wide_boundary(module, shape)
            _tensor(self.torch, label, features, input_shape, self.device)
            if id(module) not in self.targets:
                # An explicitly disabled, identified family retains its actual old owner.
                return self._old_apply(module, features)
            c, config = module.channels, self._configs[module.channels]
            x = self.policy.round_multihead(c, features)
            _tensor(self.torch, label + ":boundary", x, input_shape, self.device)
            m, branches = shape[0] * shape[1], c // 32
            latent = self.torch.empty((branches, m, 32), dtype=x.dtype, device=x.device)
            output = self.torch.empty(shape, dtype=x.dtype, device=x.device)
            first_role, second_role = self._roles(shape)
            pair_bm = self._latent_tiles[first_role] if self._wide else config["pair_bm"]
            first_grid = ((m + pair_bm - 1) // pair_bm, branches // 2 if self._wide else branches)
            if self._wide:
                admitted = self.resources[first_role]
                if (pair_bm != admitted["selected_pair_bm"] or list(first_grid) != admitted["grid"] or
                        admitted["launch_options"] != dict(num_warps=4, num_stages=1, enable_fp_fusion=False)):
                    raise RuntimeError("Wide latent dispatch differs from its cold selected tile/grid/options")
            self._launch(first_role, self._compiled[first_role],
                         (x, module.expand, module.reduce, self._lut, latent,
                          m, c, pair_bm, False, *gather), first_grid)
            project_bn = self._project_tiles[second_role] if self._wide else config["project_bn"]
            self._launch(second_role, self._compiled[second_role],
                         (x, latent, module.project, module.skip_scale, output,
                          m, c, config["project_bm"], project_bn, *gather),
                         ((m + config["project_bm"] - 1) // config["project_bm"],
                          (c + project_bn - 1) // project_bn))
            self.actualkernelhash_by_site[label] = {
                "latent": self.actualkernelhash[first_role],
                "projection": self.actualkernelhash[second_role]}
            if self._wide:
                self.provider.record("audit_swin_expand")
                self.provider.record("audit_swin_mlp_project")
                self.audit_swin.record_mlp_hit(module,
                    (self._compiled[first_role], self._compiled[second_role]))
            # Preserve logical receipts, not a claim of physical FP8 launches.
            for _ in range(branches):
                for kind in ("dense", "dense", "cubic_fp8", "fp8", "dense"):
                    self.execution.record_arithmetic_dispatch(kind)
            owner.calls[str(c)] = owner.calls.get(str(c), 0) + 1
            self.calls[label] += 1
            capturing = self.torch.xpu.is_current_stream_capturing()
            if capturing:
                self.capture_calls[label] += 1
            if c == 128 and self._pairwise_counter is not None and not self._wide:
                pair_label = f"{label.split('128.')[0]}-128-{label.split('.')[-1]}"
                self._pairwise_counter.calls[pair_label] += 1
                if capturing:
                    self._pairwise_counter.capture_calls[pair_label] += 1
            return output
        except BaseException:
            self.session._failed = True
            raise

    @contextmanager
    def _local_scope(self):
        self._validate_scope_callees()
        actual = self.multihead.BranchedMLP.forward_unquantized
        if (_function(actual).__code__ is not self._forward_code or
                _closure(actual, "self") is not self.owner):
            raise RuntimeError("Previous installed scope did not select this FusedBatched owner")
        had, saved = "apply" in self.owner.__dict__, self.owner.__dict__.get("apply")
        if self.contract in self.dataflow.CONTRACTS:
            raise RuntimeError("Duplicate branch project contract owner")
        contracts = {self.contract: (("OUT",), ()),
                     self.batched.__name__ + "._pairs": (("LATENT",), ())}
        if self._pairwise_counter is not None:
            contracts[self.pairwise.__name__ + "._pairs_pairwise"] = (("LATENT",), ())
        saved_contracts = {key: self.dataflow.CONTRACTS.get(key) for key in contracts}
        replacement = MethodType(lambda owner, module, features:
                                 self._apply(owner, module, features), self.owner)
        self._bound_apply = replacement
        self.owner.apply, self._local = replacement, True
        self.dataflow.CONTRACTS.update(contracts)
        try:
            yield
        finally:
            valid = self.owner.__dict__.get("apply") is replacement
            contract_ok = all(self.dataflow.CONTRACTS.get(key) == value for key, value in contracts.items())
            if valid:
                if had:
                    self.owner.apply = saved
                else:
                    del self.owner.apply
            for key, previous in saved_contracts.items():
                if self.dataflow.CONTRACTS.get(key) == contracts[key]:
                    if previous is None:
                        del self.dataflow.CONTRACTS[key]
                    else:
                        self.dataflow.CONTRACTS[key] = previous
            self._local, self._bound_apply = False, None
            if not valid or not contract_ok:
                self.session._failed = True
                raise RuntimeError("Branch local apply/contract ownership changed")


@contextmanager
def installed(modes, *, branch_accum=None):
    """Exact mapping API matching NumericCleanupOptions.selected_scopes()."""
    families = ("c64", "c128", "c256")
    if branch_accum is None:
        selected = dict.fromkeys(families, False)
    elif (not isinstance(branch_accum, Mapping) or set(branch_accum) != set(families) or
          any(type(branch_accum[name]) is not bool for name in families)):
        raise TypeError("branch_accum must map exactly c64/c128/c256 to bools")
    else:
        selected = {name: branch_accum[name] for name in families}
    if not any(selected.values()):
        yield _ReferenceCounter({"branch_accum": selected})
        return
    try:
        counter = BranchAccumCounter(modes, selected)
    except BaseException:
        session = getattr(modes, "session", None)
        if session is not None:
            session._failed = True
        raise
    with counter.bound():
        yield counter
