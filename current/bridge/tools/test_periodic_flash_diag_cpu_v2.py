"""CPU acceptance for the unapplied v2 patch; no game/GPU DLL is ever loaded."""
from __future__ import annotations

import ast
import builtins
import ctypes as C
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
PERF = Path("D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/live-web-perf-20261001")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


builder = module("flash_v2_builder", PROJECT / "tools/build_periodic_flash_diag_v2_cpu.py")
preparer = module("flash_v2_preparer", PROJECT / "tools/prepare_periodic_flash_diag_v2_cpu.py")
with mock.patch.object(C, "CDLL", side_effect=AssertionError("no DLL loading")):
    helper = module("periodic_flash_snapshot_v2", PROJECT / "game/periodic_flash_snapshot_v2.py")
temporal = module("nr_temporal_diagnostics_v1", builder.HOST.parent / "nr_temporal_diagnostics_v1.py")


class Function:
    def __init__(self, payload, returned=1):
        self.payload, self.returned, self.calls = payload, returned, 0

    def __call__(self, pointer):
        self.calls += 1
        C.memmove(pointer, bytes(self.payload), len(self.payload))
        return self.returned


class Dll:
    def __init__(self, snapshot=None, context=None):
        self.exports = {}
        self.lookups = 0
        if snapshot is not None:
            self.exports[helper.EXPORT] = snapshot
        if context is not None:
            self.exports[helper.CONTEXT_EXPORT] = context

    def __getattr__(self, name):
        self.lookups += 1
        if name in self.exports:
            return self.exports[name]
        raise AttributeError(name)


def raw_context(nr=9, sr=20, evaluation=30, *, stage_bits=0, reset=0):
    data = bytearray(136)
    struct.pack_into("<9Q", data, 0, evaluation, 0xFEDCBA9876543210, 7, 1000001, nr, sr, 900, 831, 77)
    struct.pack_into("<16I", data, 72, 0, 6, 1, 1, 1, 0x3ff, 128, 69, reset, 1, 3, 1,
                     reset, stage_bits, 60, 2)
    return data


def raw_snapshot(frames=(), events=()):
    data = bytearray(32912)
    struct.pack_into("<II7Q", data, 0, 32912, 2, len(events), 0, 0, len(frames), 0, 0, 1234567)
    struct.pack_into("<17Q", data, 64, *range(17))
    struct.pack_into("<40Q", data, 200, *range(40))
    for i, context in enumerate(frames):
        offset = 12432 + i * 160
        struct.pack_into("<2Q", data, offset, i + 1, 11000 + i)
        data[offset + 16:offset + 152] = context
        flags = struct.unpack_from("<I", context, 124)[0]
        struct.pack_into("<2I", data, offset + 152, 2 if flags & (1 << 8) else 0, 0)
    for i, (context, kind, reason) in enumerate(events):
        offset = 656 + i * 184
        struct.pack_into("<5Q", data, offset, i + 1, 10000 + i, 88, 111, 222)
        data[offset + 40:offset + 176] = context
        struct.pack_into("<2I", data, offset + 176, kind, reason)
    if frames:
        data[520:656] = frames[-1]
    return data


NATIVE_CPU = r'''
#include "periodic_flash_native_diag.h"
#include <cassert>
#include <thread>
using namespace nrb::flashdiag;
int main() {
    Recorder r;Context c{};c.eligible=c.sr_slot=1;
    for(int frame=1;frame<=3;++frame) {
        c.eval_id=r.begin();r.observe(c);r.count(Counter::eligible);
        c.stage_bits=0;c.nr_frame_id=0;
        if(frame==2) {r.fallback(c,Reason::source_state);r.finish(c,Reason::source_state);continue;}
        c.nr_frame_id=frame==1?1:2;
        c.stage_bits=bit(Stage::recorded)|bit(Stage::submitted)|bit(Stage::processor_called)|
            bit(Stage::processor_return_ok)|bit(Stage::result_valid)|bit(Stage::composite_submitted)|
            bit(Stage::sr_submitted)|bit(Stage::retired);
        r.count(Counter::nr_started);r.count(Counter::nr_processed);
        r.count(Counter::nr_composited);r.count(Counter::nr_retired);r.finish(c);
    }
    Snapshot s{};assert(r.snapshot(&s));assert(s.frames_retained==3 && s.retained==1);
    assert(s.frames[1].context.sr_sequence==2 && s.frames[1].context.nr_frame_id==0);
    assert(s.frames[2].context.sr_sequence==3 && s.frames[2].context.nr_frame_id==2);
    assert(s.counters[uint32_t(Counter::nr_processed)]==2);
    assert(s.counters[uint32_t(Counter::original_fallback)]==1);
    assert(s.counters[uint32_t(Counter::nr_skipped)]==1);
    Context tls{};assert(!current_context(&tls));
    {ActiveContext scope(c);assert(current_context(&tls) && tls.nr_frame_id==2);
        std::thread web([] {Context absent{};assert(!current_context(&absent));});web.join();
        Context nested=c;nested.nr_frame_id=99;
        {ActiveContext inner(nested);assert(current_context(&tls) && tls.nr_frame_id==99);}
        assert(current_context(&tls) && tls.nr_frame_id==2);}
    assert(!current_context(&tls));
    for(int i=0;i<300;++i) {c.eval_id=r.begin();r.observe(c);r.finish(c);}
    s=Snapshot{};assert(r.snapshot(&s));assert(s.frames_retained==128 && s.frames_overwritten==175);
    assert(s.retained==1 && s.events[0].context.sr_sequence==2); // healthy rows cannot erase anomalies
    assert(s.frames[0].serial==176 && s.frames[127].serial==303);
    s.version=1;assert(!r.snapshot(&s));s=Snapshot{};s.abi_size=8;assert(!r.snapshot(&s));
    Recorder parallel;std::thread a([&] {for(int i=0;i<1000;++i) {Context x{};x.sr_slot=1;
        x.eval_id=parallel.begin();parallel.observe(x);parallel.fallback(x,Reason::pending_full);parallel.finish(x);}});
    std::thread b([&] {for(int i=0;i<1000;++i) {Context x{};x.sr_slot=1;
        x.eval_id=parallel.begin();parallel.observe(x);parallel.fallback(x,Reason::pending_full);parallel.finish(x);}});
    a.join();b.join();s=Snapshot{};assert(parallel.snapshot(&s));
    assert(s.counters[uint32_t(Counter::seen)]==2000 && s.counters[uint32_t(Counter::nr_skipped)]==2000);
    assert(s.retained+s.overwritten+s.dropped==2000);
    static_assert(sizeof(Recorder)<40000 && sizeof(Snapshot)==32912);
}
'''

SYNC_CPU = r'''
#include "nr_sync_probe.h"
#include <cassert>
#include <cstdint>
bool gate(const nrb::SyncObservation& s) {
    return s.reset_observed && !s.closed && !s.oversized_barrier && s.color.seen &&
        s.color.flags==D3D12_RESOURCE_BARRIER_FLAG_NONE &&
        s.color.subresource==D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES &&
        s.color.after==D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE &&
        s.evaluate_sequence>s.color.sequence && s.evaluate_sequence-s.color.sequence<=64;
}
int main() {
    nrb::SyncProbe p;
    auto* game=reinterpret_cast<ID3D12GraphicsCommandList*>(uintptr_t(0x100));
    auto* other=reinterpret_cast<ID3D12GraphicsCommandList*>(uintptr_t(0x200));
    auto* color=reinterpret_cast<ID3D12Resource*>(uintptr_t(0x300));
    auto* unrelated=reinterpret_cast<ID3D12Resource*>(uintptr_t(0x400));
    D3D12_RESOURCE_BARRIER barrier{};barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags=D3D12_RESOURCE_BARRIER_FLAG_NONE;barrier.Transition.pResource=color;
    barrier.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    p.reset(game,true);p.barriers(game,1,&barrier);p.reset(other,true);
    const auto before=p.evaluate(game,color,nullptr);assert(gate(before));
    barrier.Transition.pResource=unrelated;
    for(int i=0;i<33;++i) p.barriers(other,1,&barrier);
    const auto aged=p.evaluate(game,color,nullptr);
    assert(aged.color.seen && aged.color.sequence==before.color.sequence);
    assert(aged.color.after==before.color.after && aged.generation==before.generation);
    assert(aged.evaluate_sequence-aged.color.sequence>64 && !gate(aged));
    barrier.Transition.pResource=color;p.barriers(game,1,&barrier);
    assert(gate(p.evaluate(game,color,nullptr))); // gate recovers with a fresh same-list observation
    barrier.Transition.pResource=unrelated;
    for(int i=0;i<257;++i) {p.barriers(other,1,&barrier);p.evaluate(other,unrelated,nullptr);}
    const auto evicted=p.evaluate(game,color,nullptr);assert(!evicted.color.seen && !gate(evicted));
}
'''


class NativePatchCpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = builder.patched_sources()
        cls.frozen = {name: hashlib.sha256((WORKSPACE / name).read_bytes()).hexdigest()
                      for name in cls.sources if (WORKSPACE / name).exists()}
        task = PERF / "periodic-flash-sol61-cpu-v2"
        task.mkdir(parents=True, exist_ok=True)
        cls.scratch = Path(tempfile.mkdtemp(prefix="cpu-", dir=task)).resolve()
        if cls.scratch.parent != task.resolve() or cls.scratch.is_junction():
            raise AssertionError("CPU output escaped the named D task directory")
        for name, content in cls.sources.items():
            path = cls.scratch / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
        vswhere = Path("C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe")
        vs = subprocess.check_output([str(vswhere), "-latest", "-products", "*", "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()
        setup = Path(vs) / "VC/Auxiliary/Build/vcvars64.bat"
        env = {key: value for key, value in os.environ.items() if key.lower() != "path"}
        env["Path"] = "C:/Windows/System32;C:/Windows"
        environment = subprocess.check_output(f'cmd.exe /d /s /c ""{setup}" >nul && set"',
            env=env, text=True, encoding="utf-8", errors="replace")
        env.update(line.split("=", 1) for line in environment.splitlines() if "=" in line and not line.startswith("="))
        cls.env = env
        cls.cl = shutil.which("cl.exe", path=env.get("Path", env.get("PATH")))
        if not cls.cl:
            raise AssertionError("CPU compiler unavailable")

    @classmethod
    def tearDownClass(cls):
        for name, digest in cls.frozen.items():
            assert hashlib.sha256((WORKSPACE / name).read_bytes()).hexdigest() == digest, name
        assert not (PROJECT / builder.HEADER).exists()
        # Keep this small task-owned D evidence for review; perform no cleanup.

    def run_cpu(self, args):
        result = subprocess.run(args, cwd=self.scratch, env=self.env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, errors="replace", timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_strict_patch_and_pinned_sources(self):
        self.assertEqual(builder.PATCH.read_text(encoding="utf-8"), builder.patch_text(self.sources))
        result = subprocess.run([shutil.which("git"), "apply", "--check", "--whitespace=error", str(builder.PATCH)],
                                cwd=WORKSPACE, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_recorder_bounded_buffers_and_callback_tls(self):
        source = self.scratch / "recorder.cpp"
        source.write_text(NATIVE_CPU, encoding="utf-8")
        exe = self.scratch / "recorder.exe"
        self.run_cpu([self.cl, "/nologo", "/std:c++20", "/EHsc", "/W4", "/WX",
                      "/I" + str(self.scratch / "cyberpunk-b580-nr-opt/src"), str(source), "/Fe:" + str(exe)])
        self.run_cpu([str(exe)])

    def test_original_global_age_can_reject_an_unchanged_game_list(self):
        source = self.scratch / "global-age.cpp"
        source.write_text(SYNC_CPU, encoding="utf-8")
        exe = self.scratch / "global-age.exe"
        self.run_cpu([self.cl, "/nologo", "/std:c++20", "/EHsc", "/W4", "/WX",
                      "/DNOMINMAX", "/DWIN32_LEAN_AND_MEAN",
                      "/I" + str(PROJECT / "include"), str(source),
                      str(PROJECT / "src/nr_sync_probe.cpp"), "/Fe:" + str(exe)])
        self.run_cpu([str(exe)])

    def test_modified_translation_units_compile_without_link_or_dll_load(self):
        opti = Path("D:/Codex-NR-Experiments/cyberpunk-opt/optiscaler-sol-ref")
        includes = [self.scratch / "cyberpunk-b580-nr-opt/src", PROJECT / "src", PROJECT / "include",
                    opti / "external/nvngx_dlss_sdk", opti / "OptiScaler/include"]
        macros = ["NOMINMAX", "WIN32_LEAN_AND_MEAN", "NRB_XESS_IDENTITY_TEST=1",
                  "NRB_DEFERRED_IDENTITY_TEST=1", "NRB_LIVE_NR_TEST=1", "NRB_TAIL_ORDER_PROBE=1", "NRB_MULTI_ROUTE_LIVE=1"]
        command = [self.cl, "/nologo", "/c", "/O2", "/MD", "/std:c++20", "/EHsc"]
        command += ["/I" + str(p) for p in includes] + ["/D" + value for value in macros]
        self.run_cpu(command + [str(self.scratch / "cyberpunk-b580-nr-opt/src/asi.cpp"),
                                str(self.scratch / "cyberpunk-b580-nr-opt/src/deferred_identity.cpp")])

    def test_gates_reset_gpu_calls_and_color_substitution_preserved(self):
        old = (PROJECT / "src/asi.cpp").read_text(encoding="utf-8")
        new = self.sources["cyberpunk-b580-nr-opt/src/asi.cpp"]
        for start, end in [("RouteDecision classify(", "NVSDK_NGX_Result __cdecl intercept"),
                           ("    const bool diagnostic_dlss_scope=", "#ifdef NRB_TAIL_ORDER_PROBE"),
                           ("        const bool dlss_source_proven=", "        const int hook_state=")]:
            self.assertEqual(old[old.index(start):old.index(end, old.index(start))],
                             new[new.index(start):new.index(end, new.index(start))])
        for name in ("src/asi.cpp", "src/deferred_identity.cpp"):
            old = (PROJECT / name).read_text(encoding="utf-8")
            new = self.sources["cyberpunk-b580-nr-opt/" + name]
            for pattern in (r"params->Set\(NVSDK_NGX_Parameter_Color,[^;]+;", r"WaitForSingleObject\([^)]*\)",
                            r"original_execute\([^;]+;", r"->Signal\([^;]+?\)", r"->Wait\([^;]+?\)",
                            r"->SetEventOnCompletion\([^;]+?\)"):
                self.assertEqual(re.findall(pattern, old), re.findall(pattern, new), (name, pattern))
        native = self.sources["cyberpunk-b580-nr-opt/src/deferred_identity.cpp"]
        self.assertIn("frame.frame_id=frame_serial.fetch_add(1)+1;", native)
        self.assertIn("frame.reset_history=frame.frame_id==1 || item.game_reset;", native)
        self.assertIn("if(!valid) {\n                nr_failed.store(true);", native)


class SnapshotCpu(unittest.TestCase):
    def test_abi_enum_layout_and_raw_pointer_fixture(self):
        header = builder.NATIVE_HEADER
        for enum, names in (("Counter", helper.COUNTERS), ("Reason", helper.REASONS),
                            ("EventKind", helper.EVENT_KINDS), ("Stage", helper.STAGE_BITS)):
            body = re.search(r"enum class " + enum + r" : uint32_t \{(.*?)\};", header, re.S).group(1)
            self.assertEqual(tuple(n.strip() for n in body.split(",") if n.strip() != "count"), names)
        self.assertEqual((C.sizeof(helper.Context), C.sizeof(helper.Event), C.sizeof(helper.Frame), C.sizeof(helper.Snapshot)),
                         (136, 184, 160, 32912))
        self.assertEqual({name: getattr(helper.Snapshot, name).offset for name in ("current", "events", "frames")},
                         {"current": 520, "events": 656, "frames": 12432})
        context = raw_context(stage_bits=0xff)
        snap = helper.read_snapshot(Dll(Function(raw_snapshot([context], [(context, 1, 2)]))))
        self.assertEqual(snap["status"], "ok")
        self.assertEqual(snap["frames"][0]["context"]["list"], 0xFEDCBA9876543210)
        self.assertEqual(snap["frames"][0]["context"]["record_reason"], 2)
        self.assertEqual(snap["events"][0]["reason"], "source_state")
        self.assertEqual(snap["reasons"]["retire_callback"], 39)
        self.assertFalse(snap["clock_domains_aligned"])
        self.assertEqual(json.loads(json.dumps(snap)), snap)

    def test_missing_busy_and_bad_headers_never_load_or_infer(self):
        with mock.patch.object(C, "CDLL", side_effect=AssertionError("no loading")):
            absent = Dll()
            for _ in range(3):
                self.assertEqual(helper.read_snapshot(absent)["status"], "unavailable")
            self.assertEqual(absent.lookups, 1)
            self.assertEqual(helper.read_snapshot(None)["status"], "unavailable")
            self.assertEqual(helper.read_snapshot(123)["status"], "unavailable")
            self.assertEqual(helper.read_snapshot(Dll(Function(bytes(32912), 0)))["status"], "busy")
            for offset, value in ((0, 8), (4, 1), (8, 65), (32, 129)):
                data = raw_snapshot()
                struct.pack_into("<I" if offset < 8 else "<Q", data, offset, value)
                self.assertEqual(helper.read_snapshot(Dll(Function(data)))["status"], "error")

    def test_tls_callback_id_join_rejects_stale_or_other_thread_context(self):
        rec = helper.PythonFrameRecorder()
        dll = Dll(context=Function(raw_context(nr=9)))
        self.assertTrue(rec.start(9, 0, dll)["native_join_valid"])
        self.assertFalse(rec.start(10, 0, dll)["native_join_valid"])
        outside = Dll(context=Function(raw_context(), 0))
        self.assertFalse(rec.start(9, 0, outside)["native_join_valid"])
        self.assertEqual(dll.lookups, 1)

    def test_raw_skip_and_recovery_join_even_when_nr_ids_are_contiguous(self):
        healthy = raw_context(nr=9, sr=20, evaluation=30, stage_bits=0xff)
        raw = raw_context(nr=0, sr=21, evaluation=31, stage_bits=(1 << 8) | (1 << 11))
        recovery = raw_context(nr=10, sr=22, evaluation=32, stage_bits=0xff)
        native = helper.read_snapshot(Dll(Function(raw_snapshot([healthy, raw, recovery]))))
        python = {"frames": [{"frame_id": 10, "native_join_valid": True,
                               "native": {"context": {"eval_id": 32, "sr_sequence": 22}},
                               "reset_causes": [], "observations": [], "exception": None}]}
        rows = helper.correlate(native, python)["frames"]
        self.assertTrue(rows[1]["raw_fallback"] and rows[1]["nr_skipped"])
        self.assertFalse(rows[1]["nr_composed"])
        self.assertEqual(rows[2]["python_join"], "exact")
        self.assertTrue(rows[2]["nr_composed"])
        self.assertIsNone(rows[1]["reset_causes"])

    def test_invalidated_list_is_not_evidence_of_raw_presented_frame(self):
        raw = raw_context(stage_bits=(1 << 9) | 1)
        native = helper.read_snapshot(Dll(Function(raw_snapshot([raw]))))
        result = helper.correlate(native, {"frames": []})
        self.assertTrue(result["frames"][0]["invalidated_without_submit"])
        self.assertFalse(result["frames"][0]["raw_fallback"])
        self.assertFalse(result["negative_evidence_complete"])

    def test_compact_windows_keep_anomaly_then_add_recovery(self):
        raw = raw_context(nr=0, sr=21, evaluation=31, stage_bits=(1 << 8))
        recovery = raw_context(nr=10, sr=22, evaluation=32, stage_bits=0xff)
        fn = Function(raw_snapshot([raw], [(raw, 1, 2)]))
        dll, sampler = Dll(fn), helper.WindowSampler()
        first = sampler.sample(dll, {"frames": []})
        self.assertTrue(first["changed"])
        self.assertTrue(first["windows"][0]["frames"][0]["raw_fallback"])
        same = sampler.sample(dll, {"frames": []})
        self.assertEqual(same["revision"], first["revision"])
        self.assertFalse(same["changed"])
        fn.payload = raw_snapshot([raw, recovery], [(raw, 1, 2)])
        second = sampler.sample(dll, {"frames": []})
        self.assertGreater(second["revision"], first["revision"])
        self.assertEqual(len(second["windows"][0]["frames"]), 2)
        fn.payload = raw_snapshot([recovery], [(raw, 1, 2)])
        retained = sampler.sample(dll, {"frames": []})
        self.assertEqual(len(retained["windows"][0]["frames"]), 2)  # older frame ring slot expired


def fake_modes(seed=10):
    model = NS(_next_seed=seed, _previous=object(), _controls=NS(
        style=0, intensity=1.0, local_tone=1.0, local_structure=1.0, auto_mask=False, skin_structure=None))
    graph = NS(replays=10, entries={"reset": object(), "history": object()})
    return NS(session=NS(_stack=NS(model=model, graph=graph)), last_graph_used=True)


class PythonRecorderCpu(unittest.TestCase):
    def test_stable_rollover_restart_capture_and_exception_are_distinct(self):
        rec, modes = helper.PythonFrameRecorder(), fake_modes(0xffffffff)
        row = rec.start(9, 0, Dll(context=Function(raw_context())))
        rec.prepare(row, modes, (), {"graph_requested": True})
        modes.session._stack.model._next_seed = 0
        modes.session._stack.graph.replays += 1
        rec.complete(row, modes)
        self.assertEqual(rec.snapshot()["events"], [])  # uint32 wrap is not a reset
        modes.session._stack.model._next_seed = 120
        row = rec.start(10, 0, None)
        rec.prepare(row, modes, (), {})
        modes.session._stack.model._next_seed = 1
        modes.session._stack.graph.entries["new"] = object()
        rec.complete(row, modes)
        event = rec.snapshot()["events"][0]
        self.assertIn("model_seed_restart", event["observations"])
        self.assertIn("new_graph_entry", event["observations"])
        row = rec.start(11, 0, None)
        rec.prepare(row, modes, (), {})
        rec.complete(row, modes, stage="export", exception=RuntimeError("fixture export failed"))
        failure = rec.snapshot()["events"][-1]
        self.assertFalse(failure["returned"])
        self.assertEqual(failure["stage"], "export")
        self.assertEqual(failure["exception"]["type"], "RuntimeError")

    def test_rings_are_bounded_and_snapshots_independent(self):
        rec, modes = helper.PythonFrameRecorder(), fake_modes()
        for i in range(160):
            row = rec.start(i + 1, 0, None)
            rec.prepare(row, modes, ("game_requested_reset",), {"optimizations": ["num_front_both"]})
            modes.session._stack.model._next_seed += 1
            rec.complete(row, modes)
        data = rec.snapshot()
        self.assertEqual((len(data["frames"]), len(data["events"])), (128, 32))
        data["frames"][0]["selection"]["optimizations"].clear()
        self.assertEqual(rec.snapshot()["frames"][0]["selection"]["optimizations"], ["num_front_both"])

    def test_no_tensor_truth_read_and_diagnostic_failure_is_observational(self):
        class TensorTrap:
            def __bool__(self):
                raise AssertionError("tensor truth read")
            def __getattribute__(self, name):
                raise AssertionError("tensor inspected: " + name)
        modes = fake_modes()
        modes.session._stack.model._previous = TensorTrap()
        self.assertTrue(helper.model_cpu_state(modes)["private_history_present"])
        rec = helper.PythonFrameRecorder()
        row = rec.start(1, 0, None)
        with mock.patch.object(helper, "model_cpu_state", side_effect=RuntimeError("metadata failure")):
            self.assertIsNone(rec.prepare(row, modes, (), {}))
            self.assertIsNone(rec.complete(row, modes))
        self.assertEqual(rec.snapshot()["diagnostic_errors"], 2)


class HostHooksCpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = builder.patched_sources()[builder.HOST.relative_to(WORKSPACE).as_posix()]
        tree = ast.parse(cls.source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                     node.name in ("process", "temporal_diagnostics", "periodic_flash_diagnostics")]
        cls.code = compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), "isolated-v2-host", "exec")

    def namespace(self, *, enabled=True, export_error=False):
        modes = fake_modes()
        class Image:
            def contiguous(self):
                return self
        image = Image()
        def process(*args, **kwargs):
            modes.session._stack.model._next_seed = 1 if kwargs["reset"] else modes.session._stack.model._next_seed + 1
            modes.session._stack.graph.replays += 1
            return image
        modes.process = process
        bridge = NS(width=1280, height=720, prepare=lambda frame: (image, image),
                    export=lambda *a, **k: None, output=lambda: NS(resource=7, fence=8, value=9))
        if export_error:
            def bad_export(*args, **kwargs):
                raise RuntimeError("fixture export failed")
            bridge.export = bad_export
        flags = ("c512_k8_decoder", "c512_k8_c32_native", "num_history_fractional", "num_front_both")
        setting = NS(enabled=True, input_size=720, experiment_720="c512_k8", optimizations_720=flags,
                     backend_variant="unrounded", history_mode="fused", graph_replay=True, style=0,
                     model_intensity=1.0, local_tone=1.0, local_structure=1.0, auto_mask=False,
                     skin_structure=None, display_strength=1.0)
        dll = Dll(context=Function(raw_context(nr=11, sr=101, evaluation=201)))
        imports = []
        real_import = builtins.__import__
        def imported(name, *args, **kwargs):
            imports.append(name)
            if name == "nr_texture_bridge_v1":
                return NS(SourceFrame=lambda *args: NS(args=args))
            if name == "nr_backend.controlled_temporal":
                return NS(NRControls=NS)
            if name == "periodic_flash_snapshot_v2":
                return helper
            if name == "nr_temporal_diagnostics_v1":
                return temporal
            if name == "torch" or name.startswith(("triton", "nr_backend")):
                raise AssertionError("real GPU import: " + name)
            return real_import(name, *args, **kwargs)
        values = dict(__builtins__={**vars(builtins), "__import__": imported},
                      _bridge=bridge, _modes=modes, _active_experiment="c512_k8", _active_optimizations=flags,
                      _failed=False, _failure_reason=None, _last_frame=10, _last_history_mode="fused",
                      _last_graph_replay=True, _last_backend_variant="unrounded", _temporal_diagnostics=None,
                      _periodic_flash_diagnostics=None, _periodic_flash_sampler=None, _periodic_flash_enabled=enabled,
                      _temporal_events_enabled=False, _timing_enabled=False, _timing_epoch=0,
                      _device=1, _queue=2, _thread=threading.get_ident(),
                      _panel=NS(snapshot=lambda: setting, mark_active=lambda *a, **k: None),
                      _error=lambda stage: None, sys=NS(modules={"cyberpunk_nr_web": NS(_native=dll)}),
                      threading=threading, time=time, FUSED_REPLAY_PROFILES=("c512_k8",))
        exec(self.code, values)
        return values, imports

    def test_host_reset_ast_and_gpu_expressions_are_unchanged(self):
        actual_bytes = preparer.ACTUAL_HOST.read_bytes()  # frozen actual host, never active GPU logs
        staged, proof = preparer.host_from_actual(actual_bytes, builder.HOST.read_bytes(), self.source)
        self.assertEqual(staged, self.source)
        self.assertEqual(len(proof["G_to_E_v1"]), 4)
        self.assertTrue(all(h["old_lines"] == 0 for h in proof["G_to_E_v1"]))
        self.assertTrue(proof["critical_reset_and_bridge_model_AST_preserved"])
        with self.assertRaisesRegex(AssertionError, "SHA changed"):
            preparer.host_from_actual(actual_bytes + b"\n", builder.HOST.read_bytes(), self.source)
        with self.assertRaisesRegex(AssertionError, "host base changed"):
            preparer.splice_delta("original\n", "original\nhook\n", "main edit\n")
        def reset(source):
            function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "process")
            return next(n for n in ast.walk(function) if isinstance(n, ast.Assign) and
                        any(isinstance(t, ast.Name) and t.id == "reset" for t in n.targets))
        self.assertEqual(ast.dump(reset(builder.HOST.read_text(encoding="utf-8"))), ast.dump(reset(self.source)))
        old_tree, new_tree = ast.parse(builder.HOST.read_text(encoding="utf-8")), ast.parse(self.source)
        for receiver, attr in (("_modes", "process"), ("_bridge", "prepare"), ("_bridge", "export"), ("_bridge", "output")):
            def calls(tree):
                return [ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.Call) and
                        isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and
                        (n.func.value.id, n.func.attr) == (receiver, attr)]
            self.assertEqual(calls(old_tree), calls(new_tree), attr)

    def test_disabled_host_imports_no_new_diagnostics(self):
        ns, imports = self.namespace(enabled=False)
        self.assertEqual(ns["process"](1, 2, 3, 4, 11, 0, 1280, 720), (7, 8, 9))
        self.assertNotIn("periodic_flash_snapshot_v2", imports)
        self.assertNotIn("nr_temporal_diagnostics_v1", imports)
        self.assertIsNone(ns["_periodic_flash_diagnostics"])

    def test_enabled_host_records_exact_context_after_export_and_exposes_existing_health(self):
        ns, _ = self.namespace()
        self.assertEqual(ns["process"](1, 2, 3, 4, 11, 0, 1280, 720), (7, 8, 9))
        row = ns["_periodic_flash_diagnostics"].snapshot()["frames"][0]
        self.assertTrue(row["native_join_valid"] and row["returned"])
        self.assertEqual((row["native"]["context"]["sr_sequence"], row["after"]["next_seed"]), (101, 11))
        health = ns["temporal_diagnostics"]()
        self.assertEqual(health["periodic_flash_v2"]["status"], "unavailable")  # no snapshot export in this fixture

    def test_model_success_followed_by_export_failure_remains_failed_return(self):
        ns, _ = self.namespace(export_error=True)
        with self.assertRaisesRegex(RuntimeError, "fixture export failed"):
            ns["process"](1, 2, 3, 4, 11, 0, 1280, 720)
        row = ns["_periodic_flash_diagnostics"].snapshot()["frames"][0]
        self.assertFalse(row["returned"])
        self.assertEqual(row["stage"], "export")
        self.assertEqual(row["after"]["next_seed"], 11)
        self.assertTrue(ns["_failed"])

    def test_new_diagnostic_metadata_error_cannot_disable_normal_nr(self):
        ns, _ = self.namespace()
        with mock.patch.object(helper, "model_cpu_state", side_effect=RuntimeError("optional metadata failure")):
            self.assertEqual(ns["process"](1, 2, 3, 4, 11, 0, 1280, 720), (7, 8, 9))
        self.assertFalse(ns["_failed"])
        self.assertEqual(ns["_periodic_flash_diagnostics"].snapshot()["diagnostic_errors"], 2)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    unexpected = sorted(name for name in sys.modules if name == "torch" or name.startswith(("triton", "nr_backend")))
    receipt = {"kind": "periodic-flash-unapplied-v2-cpu", "passed": result.wasSuccessful() and not unexpected,
               "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "unexpected_gpu_modules": unexpected, "native_scratch": str(getattr(NativePatchCpu, "scratch", "")),
               "sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                   builder.PATCH, PROJECT / "game/periodic_flash_snapshot_v2.py", builder.HOST,
                   preparer.ACTUAL_HOST, builder.HOST.parent / "nr_temporal_diagnostics_v1.py",
                   PROJECT / "tools/build_periodic_flash_diag_v2_cpu.py", Path(__file__),
                   PROJECT / "tools/prepare_periodic_flash_diag_v2_cpu.py")}}
    (PERF / "periodic-flash-sol61-cpu-v2.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if receipt["passed"] else 1)
