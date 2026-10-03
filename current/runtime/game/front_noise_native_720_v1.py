"""Owned dynamic front noise candidates, plan 10, CPU AST development only.

API: installed(modes, front_noise="table" | "native_radius" | "native_trig" |
               "native_both") -> counters with preflight/validate/snapshot.
Select a fresh actual720 controlled unrounded C512-library/nativeK8 session
first. The main suite owns graph signatures. This child layers the real
session._installed, then selects its implementation in the shared owned front
dispatcher. All post/front route observers run before that one implementation.
The inherited _Owned720._transfer_serial_thread is an explicit suite-only idle
transaction; ordinary validation keeps its thread gate. Multiple post/front
children share this session/front entry and each observes the same frame once.
The common source gate separately accepts the exact reviewed active G capture
body; noise dispatch does not depend on E's optional encoder512 output route.

The table default is a noop. Opt-ins still execute one front kernel outside
the body graph for EACH frame, reset and history alike. uint32 hash overflow,
runtime seed, padding/reflection coordinates, RGB/history half operations and
control lanes are the reviewed fused_dynamic_front_v1 body. Only noise
radius/trig loads change. Uniform is (index+1)*2**-24 in (0,1], radius is
sqrt((log2(u)*0.69314718246459960938)*-2). Direct theta=2*pi*u sin/cos is lossy;
there is no reference-cycle integer emulation. Pairing remains
(rc*cos(d), rc*sin(d), ra*cos(b)), followed by half conversion.

The complete noise_source and all three buffers stay strongly owned. Passing
unused table pointers is NOT a release of 192 MiB. Scope exit restores the
instance method, retires its session and closes candidate graphs.

LOCAL TESTS FOR LUNA ONLY (not run): indices 0/2**24-1 and adjacent endpoints;
period seams and FP32 rounding edges; seeds 0,1,2,2**31-1,2**31,2**32-2,
2**32-1 and wrap->0, reproducible hash/index bits on padded coordinates;
multi-seed per-channel finite/mean/variance, n0/n1 radial pairing, cross-channel
correlation, angular histogram and spatial autocorrelation. Compare radius
only, trig only and both to same-seed table output BEFORE final half and by
half bit/ULP after it. Include reflected last 48 rows and all nonnoise lanes.
Use equal independent histories for 13-frame and 50-frame B-C-C-B gates, then
full continuous video review for dark texture/grain/flicker and detail.
Distribution agreement never substitutes for model/video acceptance.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys


def _common_module():
    path = Path(__file__).with_name("post_numeric_suite_720_v1.py").resolve(strict=True)
    # Share the canonical common object with the runner and post scope. A loaded
    # E module cannot serve a D snapshot (or vice versa), even with equal bytes.
    name = "post_numeric_suite_720_v1"
    module = sys.modules.get(name)
    if module is not None:
        if Path(module.__file__).resolve(strict=True) != path:
            raise RuntimeError("Front/common source belongs to a different candidate snapshot")
        return module
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Missing owned post/front lifecycle source")
    module = module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


common = _common_module()  # stdlib-only module; no tensor/model/GPU import
front_route_dispatch = common._sibling("front_route_dispatch_720_v1")
FRONT_BLOCK, FRONT_STAGES = 128, 2


class FrontNoiseCounters(common._Owned720):
    def __init__(self, modes, *, front_noise):
        super().__init__(modes, {"front_noise": front_noise},
                         __file__, "_front_noise_native_720_v1")
        self.kernels = common._sibling("front_noise_native_720_kernel_v1")
        self.sources["kernels"] = common._source("candidate_kernels", self.kernels)
        self.sources["front_dispatch"] = common._source("front_dispatch", front_route_dispatch)
        self.radius_native = front_noise in ("native_radius", "native_both")
        self.trig_native = front_noise in ("native_trig", "native_both")
        self._frozen_flags = (self.radius_native, self.trig_native)
        self.tables = {name: getattr(self.noise, name) for name in ("radius", "sine", "cosine")}
        for value in self.tables.values():
            common._tensor(self.torch, value, (1 << 24,), self.torch.float32, self.device)
        self.table_fingerprints = {n: common._fingerprint(v) for n, v in self.tables.items()}
        self.table_hashes = {}
        self.calls = dict.fromkeys(("front.reset", "front.history"), 0)
        self.capture_calls = dict.fromkeys(self.calls, 0)
        jit = self.kernels.front
        self._contracts[f"{jit.fn.__module__}.{jit.fn.__name__}"] = (("OUT",), ())

    def _validate_local(self):
        from replay_lifecycle_audit_rest_720_v1 import local_guard
        if local_guard(self):
            return
        if (self.radius_native, self.trig_native) != self._frozen_flags:
            raise RuntimeError("Front noise constexpr flags changed")
        for name, value in self.tables.items():
            if (getattr(self.noise, name) is not value
                    or common._fingerprint(value) != self.table_fingerprints[name]):
                raise RuntimeError(f"Owned complete noise table changed: {name}")
        front_route_dispatch.require_selected(self)

    def preflight(self):
        """Future Luna compile/load: both real outside-graph reset/history variants."""
        try:
            self._guard(fresh=True, sources=True)
            self._validate_local()
            if not self._live or self.torch.xpu.is_current_stream_capturing():
                raise RuntimeError("Front preflight must precede frame/capture inside installed()")
            if self._preflight_complete:
                return self.snapshot()
            self._freeze_history_sources()
            manifest_path = (Path(self.modes.exact_root) / "model-assets" /
                             "noise-sm89-v2" / "manifest.json").resolve(strict=True)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("domain_count") != 1 << 24 or manifest.get("schema") != 1:
                raise RuntimeError("Unknown owned noise scalar manifest")
            self.sources["noise_manifest"] = common._file_row(manifest_path)
            with self.torch.inference_mode(False):
                for name, filename in (("radius", "radius.f32.bin"), ("sine", "sin.f32.bin"),
                                       ("cosine", "cos.f32.bin")):
                    actual = hashlib.sha256(
                        self.tables[name].detach().cpu().numpy().tobytes()).hexdigest()
                    if actual != manifest["files"][filename]:
                        raise RuntimeError(f"Actual owned scalar noise hash mismatch: {name}")
                    self.table_hashes[name] = actual
                rgb = self.torch.empty((*common.SIZE, 3), dtype=self.torch.float32, device=self.device)
                previous = self.torch.empty_like(rgb)
                output = self.torch.empty((*common.INTERNAL, 16),
                                          dtype=self.torch.float16, device=self.device)
                # do_not_specialize avoids seed value specialization, but the
                # Python integer ABI can still differ above 2**31-1. Screen
                # both real launch signatures; TL converts either to uint32.
                for temporal, site in ((False, "front.reset"), (True, "front.history")):
                    for seed in (0, 0xffffffff):
                        label = self._label(temporal, seed)
                        args = self._args(rgb, previous if temporal else None, output, seed)
                        self._screen(label, self.kernels.front, args,
                                     (self.triton.cdiv(768 * 1280, FRONT_BLOCK),),
                                     block=FRONT_BLOCK, num_stages=FRONT_STAGES)
                        self.resources[label]["specialization"] = {
                            "site": site, "rgb_shape": [720, 1280, 3],
                            "previous_dtype": "torch.float32",
                            "output_shape": [768, 1280, 16], "seed_runtime_uint32": True,
                            "seed_abi_probe": seed, "TEMPORAL": temporal,
                            "NATIVE_RADIUS": self.radius_native, "NATIVE_TRIG": self.trig_native,
                            "noise_mode": self.options["front_noise"]}
            self._preflight_complete = True
            return self.snapshot()
        except BaseException:
            self.session._failed = True
            raise

    def _args(self, rgb, previous, output, seed):
        return (rgb, rgb if previous is None else previous,
                self.tables["radius"], self.tables["sine"], self.tables["cosine"],
                output, seed, *common.SIZE, *common.INTERNAL, previous is not None,
                self.radius_native, self.trig_native, FRONT_BLOCK)

    @staticmethod
    def _label(temporal, seed):
        return ("front.history" if temporal else "front.reset") + (
            ".seed_high" if seed >= 0x80000000 else ".seed_low")

    def _run(self, rgb, previous, options):
        self._guard(dispatch=True)
        from replay_lifecycle_audit_rest_720_v1 import local_after_guard
        local_after_guard(self)
        if (set(options) != {"padded_size", "seed", "noise_source"}
                or tuple(options["padded_size"]) != common.INTERNAL
                or options["noise_source"] is not self.noise):
            raise RuntimeError("Native front options left the actual owned callsite")
        # _observe_front checks RGB/history/seed and the real graph/history route
        # before calling this owned implementation; this path never delegates.
        if self._front_event is None:
            raise RuntimeError("Native front was called outside the observed frame route")
        output = self.torch.empty((*common.INTERNAL, 16), dtype=self.torch.float16, device=self.device)
        site = "front.reset" if previous is None else "front.history"
        label = self._label(previous is not None, options["seed"])
        args = self._args(rgb, previous, output, options["seed"])
        grid = (self.triton.cdiv(768 * 1280, FRONT_BLOCK),)
        # Same concrete arguments/specialization before submit, no compilation
        # is allowed to silently replace the binary that preflight screened.
        screened = self.kernels.front.warmup(
            *args, grid=grid, num_warps=4, num_stages=FRONT_STAGES, enable_fp_fusion=False)
        self._launched(label, screened)
        kernel = self.kernels.front[grid](
            *args, num_warps=4, num_stages=FRONT_STAGES, enable_fp_fusion=False)
        self._launched(label, kernel)
        self._hit(site)
        self.front_owner.calls += 1  # Preserve the actual FusedFront receipt.
        self._front_event.update(noise_mode=self.options["front_noise"],
                                 front_kernel_hash=str(kernel.hash),
                                 front_binary_sha256=self.resources[label]["actualbinary_sha256"])
        return output

    @contextmanager
    def _methods(self):
        with front_route_dispatch.implementation(self, self._run):
            yield

    def _expected_capture(self, entries):
        # Dynamic hash/seed/front execute outside ALL reset/history body captures.
        return dict.fromkeys(self.capture_calls, 0)

    def _expected_frame_calls(self, entries):
        expected = super()._expected_frame_calls(entries)
        site = "front.reset" if self._front_event["frontend"] == "reset" else "front.history"
        expected[site] = 1
        return expected

    def snapshot(self):
        result = super().snapshot()
        result.update(noise_mode=self.options["front_noise"], table_sha256=dict(self.table_hashes),
                      noise_constants={n: common._constant_row(v)
                                       for n, v in self.table_fingerprints.items()},
                      expected_sites={"outside_body_per_frame": 1,
                                      "reset": "front.reset", "history": "front.history",
                                      "body_capture": 0,
                                      "actual_history_sampler_counts": "independent history scope"},
                      native_radius=self.radius_native, native_trig=self.trig_native,
                      trig_reference_cycle_equivalent=False if self.trig_native else None,
                      retained_noise_table_bytes=3 * (1 << 24) * 4,
                      noise_table_memory_released=False,
                      theoretical_launches={"front_per_actual_frame": 1, "launches_saved": 0,
                                            "body_capture_front_launches": 0},
                      pairing=["rc*cos(d)", "rc*sin(d)", "ra*cos(b)"])
        return result


@contextmanager
def installed(modes, *, front_noise="table"):
    """Default table is a noop; no required preflight arguments."""
    if front_noise not in ("table", "native_radius", "native_trig", "native_both"):
        raise ValueError("Unknown front_noise mode")
    if front_noise == "table":
        yield common.ReferenceCounters({"front_noise": front_noise}, __file__)
        return
    try:
        candidate = FrontNoiseCounters(modes, front_noise=front_noise)
        with candidate.installed():
            yield candidate
    except BaseException:
        if getattr(modes, "session", None) is not None:
            modes.session._failed = True
        raise
