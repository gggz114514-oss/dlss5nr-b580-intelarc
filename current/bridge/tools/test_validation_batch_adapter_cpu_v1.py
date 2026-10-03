"""CPU protocol checks for the real adapter and loopback handler, without I/O.

Uses the shared stdlib validation helper, a fake host/stream and an in-memory
server. No Torch/Triton, DLL loading, socket requests or GPU work is performed.
Run with Python -I -B -S to avoid environment imports and bytecode writes.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from html.parser import HTMLParser
import importlib.util
import io
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent / ".codex-worktrees" / "re8-fp8-unround-fast-20260928"
GAME = ROOT / "game"
ADAPTER = PROJECT / "game" / "cyberpunk_nr_adapter.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MemoryServer:
    def __init__(self, address, handler):
        self.server_port = address[1] or 8766
        self.handler = handler

    def serve_forever(self):
        raise AssertionError("CPU tests must not start a listener")


class NoServerThread:
    def __init__(self, **kwargs):
        pass

    def start(self):
        pass


class Meter:
    def __init__(self, case):
        self.case = case
        self.inside = False
        self.handoffs = []
        self.walls = []

    def note_handoff(self, milliseconds):
        self.handoffs.append(milliseconds)

    @contextmanager
    def frame(self, host, frame_id):
        self.case.events.append("meter-enter")
        if self.case.meter_enter_error is not None:
            raise self.case.meter_enter_error
        self.inside = True
        started = self.case.clock_ms
        try:
            yield
        finally:
            self.walls.append(self.case.clock_ms - started)
            self.inside = False
            self.case.events.append("meter-exit")


class Owner:
    def __init__(self, case):
        self.case = case
        self.options = SimpleNamespace(active=True)
        self.session = object()
        self.graph = SimpleNamespace(closed=False)
        self.closed = False
        self.children = {name: object() for name in
                         ("history", "front", "decoder", "branch", "vit", "post")}
        self.modes = SimpleNamespace(session=self.session, numeric_cleanup_calls=self,
                                     _numeric_cleanup_owner=self)
        self.transfers = []
        self.final_checks = 0

    def _validate_suite_owner(self):
        self.case.assertTrue(self.case.adapter._process_serial_lock._is_owned())
        self.case.events.append("owner-check")

    def transfer_serial_thread(self, *, game_adapter, serial_guard, previous_thread):
        # This gate observes the caller at runtime, rather than matching source.
        self.case.assertIs(sys._getframe(1).f_code, game_adapter.process.__code__)
        self.case.assertIs(sys._getframe(1).f_globals, game_adapter.__dict__)
        self.case.assertIs(serial_guard, game_adapter._process_serial_lock)
        self.case.assertTrue(serial_guard._is_owned())
        self.case.assertEqual(game_adapter.host._thread, threading.get_ident())
        self.case.assertEqual(game_adapter.host._bridge.thread, threading.get_ident())
        if game_adapter.validation_batch_state()["enabled"] and self.options.active:
            cycle = getattr(self, "_serial_validation_frame_720", None)
            self.case.assertIsNotNone(cycle, "cycle must already be registered before handoff")
        self.transfers.append((previous_thread, threading.get_ident()))
        self.case.events.append("transfer")
        if self.case.handoff_error is not None:
            raise self.case.handoff_error

    def validate_context(self):
        self.case.assertFalse(hasattr(self, "_serial_validation_frame_720"),
                              "successful exit checks must run uncached")
        if self.case.meter_enabled:
            self.case.assertTrue(self.case.meter.inside)
        self.case.events.append("exit-validation")
        self.final_checks += len(self.children)
        self.case.clock_ms += 3.0
        if self.case.exit_error is not None:
            raise self.case.exit_error


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(sys.modules))
        self.stack.enter_context(patch.dict(
            __import__("os").environ,
            {"CYBERPUNK_NR_WEB": "0", "CYBERPUNK_NR_COST_METER": "0"}))
        self.profiles = load("numeric_game_profiles_720_v1", GAME / "numeric_game_profiles_720_v1.py")
        self.controls = load("nr_game_controls", GAME / "nr_game_controls.py")
        self.helper = load("numeric_frame_validation_720_v1", GAME / "numeric_frame_validation_720_v1.py")
        self.stack.enter_context(patch.object(self.controls, "LocalControlServer", MemoryServer))
        self.stack.enter_context(patch.object(self.controls, "threading", SimpleNamespace(
            Lock=threading.Lock, Thread=NoServerThread)))
        self.events, self.cycles = [], []
        self.clock_ms = 0.0
        self.stream_error = self.handoff_error = self.host_error = None
        self.exit_error = self.meter_enter_error = None
        self.meter_enabled = False
        self.host_action = None
        self.host = ModuleType("nr_game_pre_xess_host")
        self.host.available_experiments_720 = lambda: ("baseline", "c512_k8")
        self.host._error = lambda label: self.events.append(label)
        self.host.LOG = Path("unused-cpu-url")
        self.host._thread = threading.get_ident()
        self.host._bridge = SimpleNamespace(stream=object(), thread=self.host._thread,
            width=1280, height=720,
            torch=SimpleNamespace(xpu=SimpleNamespace(set_stream=self.set_stream)))
        self.host.process = self.host_process
        sys.modules[self.host.__name__] = self.host
        self.adapter = load("cyberpunk_nr_adapter", ADAPTER)
        self.owner = Owner(self)
        self.host._modes = self.owner.modes
        self.meter = Meter(self)
        meter_module = ModuleType("nr_numeric_cost_meter_v1")
        meter_module.optional_meter = lambda: self.meter
        sys.modules[meter_module.__name__] = meter_module
        self.before_modules = set(sys.modules)

    def tearDown(self):
        # The only 'torch' object above is a SimpleNamespace stream fixture.
        added = set(sys.modules) - self.before_modules
        self.assertFalse(any(name == "torch" or name.startswith("torch.") or
                             name == "triton" or name.startswith("triton.") for name in added))
        self.assertFalse(hasattr(self.owner, "_serial_validation_frame_720"))

    def set_stream(self, stream):
        self.assertIs(stream, self.host._bridge.stream)
        if self.owner.options.active and self.adapter.validation_batch_state()["enabled"]:
            self.assertTrue(hasattr(self.owner, "_serial_validation_frame_720"))
        self.events.append("stream")
        if self.stream_error is not None:
            raise self.stream_error

    def host_process(self, *args):
        self.assertTrue(self.adapter._process_serial_lock._is_owned())
        if self.meter_enabled:
            self.assertTrue(self.meter.inside)
        self.events.append("host")
        cycle = getattr(self.owner, "_serial_validation_frame_720", None)
        if cycle is not None:
            self.cycles.append(cycle)
            self.assertTrue(cycle.snapshot()["active"])
        self.clock_ms += 10.0
        if self.host_action is not None:
            self.host_action()
        if self.host_error is not None:
            raise self.host_error
        return ("processed", args[4])

    def frame(self, frame_id=1):
        return self.adapter.process(10, 20, 30, 40, frame_id, False, 1280, 720)

    def enable_meter(self):
        self.meter_enabled = True
        __import__("os").environ["CYBERPUNK_NR_COST_METER"] = "1"

    def panel(self, **kwargs):
        defaults = {"modes": tuple((size, style) for size in (360, 480, 540, 720)
                                   for style in (0, 1, 2)),
                    "route": "pre-xess-fullsize", "history_modes": ("reference", "fused"),
                    "graph_modes": (360, 480, 540, 720),
                    "validation_read": self.adapter.validation_batch_state,
                    "validation_apply": self.adapter.set_validation_batch}
        defaults.update(kwargs)
        return self.controls.ControlPanel(lambda *a: self.fail("validation changed native settings"),
                                          **defaults)

    def request(self, panel, path, payload=None, *, method="POST", headers=None):
        handler = object.__new__(panel._handler())
        body = json.dumps(payload).encode("utf-8")
        port = panel._server.server_port
        handler.path = path
        handler.headers = {"Host": f"127.0.0.1:{port}", "Origin": f"http://127.0.0.1:{port}",
                           "X-NR-Token": panel._token, "Content-Type": "application/json",
                           "Content-Length": str(len(body))}
        handler.headers.update(headers or {})
        handler.rfile = io.BytesIO(body)
        replies = []
        handler._reply = lambda status, text, *a: replies.append((status, json.loads(text)))
        getattr(handler, "do_" + method)()
        self.assertEqual(len(replies), 1)
        return replies[0]

    def test_same_thread_real_factory_registered_and_closed_each_frame(self):
        self.assertTrue(self.adapter.validation_batch_state()["enabled"])
        self.assertEqual(self.frame(11), ("processed", 11))
        self.assertEqual(self.frame(12), ("processed", 12))
        self.assertEqual(self.owner.transfers, [])
        self.assertEqual(self.owner.final_checks, 12)
        self.assertIsNot(self.cycles[0], self.cycles[1])
        self.assertTrue(all(not cycle.snapshot()["active"] for cycle in self.cycles))
        state = self.adapter.validation_batch_state()
        self.assertEqual(state["last_snapshot"]["frame_id"], 12)
        self.assertTrue(all(type(v) in (type(None), str, bool, int, float)
                            for v in state["last_snapshot"].values()))
        json.dumps(state, allow_nan=False)
        state["last_snapshot"]["frame_id"] = -1
        self.assertEqual(self.adapter.validation_batch_state()["last_snapshot"]["frame_id"], 12)

    def test_actual_other_thread_handoff_has_direct_caller_and_prior_cycle(self):
        old_thread = self.host._thread
        errors, result = [], []

        def worker():
            try:
                result.append(self.frame(21))
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(result, [("processed", 21)])
        self.assertEqual(self.owner.transfers, [(old_thread, thread.ident)])
        self.assertLess(self.events.index("owner-check"), self.events.index("stream"))
        self.assertLess(self.events.index("stream"), self.events.index("transfer"))
        self.assertLess(self.events.index("transfer"), self.events.index("host"))
        self.assertEqual(self.owner.final_checks, 6)

    def test_none_inactive_and_disabled_owner_are_noop(self):
        for kind in ("none", "inactive", "disabled"):
            with self.subTest(kind=kind):
                self.host._modes = self.owner.modes
                self.owner.modes.numeric_cleanup_calls = self.owner
                self.owner.options.active = kind != "inactive"
                self.adapter.set_validation_batch(kind != "disabled")
                if kind == "none":
                    self.owner.modes.numeric_cleanup_calls = None
                self.assertEqual(self.frame(), ("processed", 1))
                snapshot = self.adapter.validation_batch_state()["last_snapshot"]
                self.assertFalse(snapshot["active"])
                self.assertFalse(snapshot["enabled"])
                self.assertEqual(self.cycles, [])
                self.assertEqual(self.owner.final_checks, 0)

    def test_host_exception_kept_and_failed_frame_closed_without_exit_checks(self):
        self.host_error = RuntimeError("host sentinel")
        with self.assertRaises(RuntimeError) as caught:
            self.frame()
        self.assertIs(caught.exception, self.host_error)
        self.assertFalse(self.cycles[0].snapshot()["active"])
        self.assertEqual(self.owner.final_checks, 0)

    def test_stream_and_handoff_early_failures_abort_registered_cycle(self):
        for stage in ("stream", "handoff"):
            with self.subTest(stage=stage):
                self.events.clear()
                self.host._thread = -1
                self.stream_error = RuntimeError(stage) if stage == "stream" else None
                self.handoff_error = RuntimeError(stage) if stage == "handoff" else None
                with self.assertRaises(RuntimeError) as caught:
                    self.frame()
                self.assertIs(caught.exception, self.stream_error or self.handoff_error)
                self.assertNotIn("host", self.events)
                self.assertFalse(hasattr(self.owner, "_serial_validation_frame_720"))
                self.assertFalse(self.adapter.validation_batch_state()["last_snapshot"]["active"])
                if stage == "handoff":
                    self.assertIn("numeric-serial-thread-handoff", self.events)

    def test_exit_validation_exception_is_visible_and_cleans_cycle(self):
        self.exit_error = RuntimeError("exit validation sentinel")
        with self.assertRaises(RuntimeError) as caught:
            self.frame()
        self.assertIs(caught.exception, self.exit_error)
        self.assertFalse(self.cycles[0].snapshot()["active"])
        self.assertEqual(self.owner.final_checks, 6)

    def test_close_protocol_failure_aborts_and_preserves_host_error_with_cause(self):
        closing_error = RuntimeError("close protocol sentinel")
        close_calls = []

        def failing_close(cycle, success):
            close_calls.append(success)
            raise closing_error

        self.stack.enter_context(patch.object(self.helper._Frame, "close", failing_close))
        self.host_error = RuntimeError("original host sentinel")
        with self.assertRaises(RuntimeError) as caught:
            self.frame()
        self.assertIs(caught.exception, self.host_error)
        self.assertIs(caught.exception.__cause__, closing_error)
        self.assertEqual(close_calls, [False])
        self.assertFalse(self.cycles[0].snapshot()["active"])

    def test_retired_old_owner_is_closed_after_host_replaces_modes(self):
        self.host_action = lambda: setattr(self.host, "_modes", SimpleNamespace())
        self.frame()
        self.assertEqual(self.owner.final_checks, 0)
        self.assertTrue(self.adapter.validation_batch_state()["last_snapshot"]["retired"])
        self.assertFalse(self.cycles[0].snapshot()["active"])

    def test_meter_measures_host_plus_uncached_exit_validation(self):
        self.enable_meter()
        self.frame()
        self.assertEqual(self.meter.walls, [13.0])
        self.assertEqual(len(self.meter.handoffs), 1)
        self.assertLess(self.events.index("host"), self.events.index("exit-validation"))
        self.assertLess(self.events.index("exit-validation"), self.events.index("meter-exit"))

    def test_meter_host_failure_and_context_entry_failure_cleanup(self):
        self.enable_meter()
        self.host_error = RuntimeError("measured host failure")
        with self.assertRaises(RuntimeError) as caught:
            self.frame()
        self.assertIs(caught.exception, self.host_error)
        self.assertEqual(self.meter.walls, [10.0])
        self.host_error = None
        self.meter_enter_error = RuntimeError("meter entry failure")
        with self.assertRaises(RuntimeError) as caught:
            self.frame()
        self.assertIs(caught.exception, self.meter_enter_error)
        self.assertFalse(hasattr(self.owner, "_serial_validation_frame_720"))

    def test_switch_does_not_wait_for_serial_lock_or_modify_model_state(self):
        setting = self.adapter._panel.snapshot()
        extras = (self.adapter._extra_experiment_720, self.adapter._extra_optimizations_720)
        history, graph_key = object(), object()
        self.host._history, self.host._graph_key = history, graph_key
        completed = threading.Event()
        errors = []

        def setter():
            try:
                self.adapter.set_validation_batch(False)
            except BaseException as error:
                errors.append(error)
            finally:
                completed.set()

        with self.adapter._process_serial_lock:
            thread = threading.Thread(target=setter)
            thread.start()
            done = completed.wait(timeout=1)
        thread.join(timeout=2)
        self.assertTrue(done, "setter waited for process serial lock")
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertIs(self.adapter._panel.snapshot(), setting)
        self.assertEqual((self.adapter._extra_experiment_720,
                          self.adapter._extra_optimizations_720), extras)
        self.assertIs(self.host._history, history)
        self.assertIs(self.host._graph_key, graph_key)
        self.assertEqual(self.events, [])
        self.assertNotIn("validation_batch", asdict(setting))
        for value in (1, "false", None, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.adapter.set_validation_batch(value)

    def test_protected_validation_route_and_strict_boolean_payload(self):
        panel = self.panel()
        setting = panel.snapshot()
        for header, value in (("Host", "elsewhere"), ("Origin", "http://elsewhere"),
                              ("X-NR-Token", "bad"), ("Content-Type", "text/plain")):
            with self.subTest(header=header):
                code, _ = self.request(panel, "/api/validation", {"enabled": False},
                                       headers={header: value})
                self.assertEqual(code, 403)
                self.assertTrue(self.adapter.validation_batch_state()["enabled"])
        for body in ({}, {"enabled": 1}, {"enabled": "false"}, {"enabled": None},
                     {"enabled": False, "graph_replay": False}, []):
            with self.subTest(body=body):
                self.assertEqual(self.request(panel, "/api/validation", body)[0], 422)
        self.assertEqual(self.request(panel, "/api/validation", {"enabled": False}),
                         (200, {"enabled": False, "last_snapshot": None}))
        self.assertEqual(self.request(panel, "/api/validation", method="GET"),
                         (200, {"enabled": False, "last_snapshot": None}))
        self.assertEqual(self.request(panel, "/api/validation", method="GET",
                                      headers={"Host": "elsewhere"})[0], 403)
        self.assertIs(panel.snapshot(), setting)

    def test_validation_endpoint_does_not_query_native_status_and_timing_still_works(self):
        reads, timing = [], [True]

        def native_read():
            reads.append("native")
            return None

        panel = self.panel(metrics_read=native_read, health_read=native_read,
                           timing_read=lambda: timing[0],
                           timing_apply=lambda enabled: timing.__setitem__(0, enabled))
        self.assertEqual(self.request(panel, "/api/validation", {"enabled": False})[0], 200)
        self.assertEqual(reads, [])
        code, state = self.request(panel, "/api/timing", {"enabled": False})
        self.assertEqual(code, 200)
        self.assertFalse(state["timing_enabled"])
        self.assertEqual(reads, ["native", "native"])
        self.assertFalse(state["validation_batch"]["enabled"])
        self.assertEqual(self.request(panel, "/api/timing", {"enabled": 0})[0], 422)

    def test_optional_callbacks_preserve_legacy_panel(self):
        panel = self.panel(validation_read=None, validation_apply=None)
        self.assertIsNone(panel.status()["validation_batch"])
        self.assertEqual(self.request(panel, "/api/validation", {"enabled": False})[0], 422)
        with self.assertRaises(ValueError):
            self.panel(validation_apply=None)

    def test_adapter_wires_real_control_callbacks_and_metrics_snapshot(self):
        native = ModuleType("cyberpunk_nr_web")
        values = {"enabled": 1, "input_height": 720, "style": 0, "history": 3,
                  "graph_replay": 1, "auto_mask": 0, "skin_structure_enabled": 0,
                  "display_strength": 1.0, "model_intensity": 1.0, "local_tone": 1.0,
                  "local_structure": 1.0, "skin_structure": 0.0}
        native.get_controls = lambda: dict(values)
        native._values = lambda current: current
        native.set_controls = lambda current: self.fail("switch wrote native controls")
        native.get_timing_enabled = lambda: True
        native.set_timing_enabled = lambda enabled: None
        info = {"stages": None, "status": 0, "last_ms": 10.0, "average_ms": 10.0,
                "processed_frames": 1, "estimated_fps": 100.0}
        native.health = lambda: dict(info)
        sys.modules[native.__name__] = native
        __import__("os").environ["CYBERPUNK_NR_WEB"] = "1"
        self.stack.enter_context(patch.object(self.controls, "Path"))
        self.adapter.start_controls()
        self.assertIsInstance(self.adapter._panel, self.controls.ControlPanel)
        self.assertNotIn("web-control", self.events)
        panel = self.adapter._panel
        setting = panel.snapshot()
        panel.update_validation({"enabled": False})
        self.assertIs(panel.snapshot(), setting)
        self.assertFalse(panel.status()["processing"]["validation_batch"]["enabled"])
        self.adapter.set_validation_batch(True)
        self.frame(99)
        self.assertEqual(panel.status()["processing"]["validation_batch"]["last_snapshot"]["frame_id"], 99)

    def test_web_checkbox_is_independent_and_old_controls_preserved(self):
        class Inputs(HTMLParser):
            def __init__(self):
                super().__init__()
                self.ids, self.optimizations = {}, set()

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if "id" in attrs:
                    self.ids[attrs["id"]] = (tag, attrs)
                if "data-optimization" in attrs:
                    self.optimizations.add(attrs["data-optimization"])

        html = self.controls._page("cpu-token", "pre-xess-fullsize")
        inputs = Inputs()
        inputs.feed(html)
        self.assertEqual(inputs.ids["validationBatchEnabled"][1]["type"], "checkbox")
        self.assertIn("合并重复校验", html)
        self.assertEqual(inputs.optimizations, set(self.profiles.PROFILES) - {"baseline", "c512_k8"})
        for key in ("enabled", "timingEnabled", "backendVariant", "size", "styleButtons",
                    "modelIntensity", "localTone", "localStructure", "autoMask", "separateSkin",
                    "skinStructure", "strength", "history", "graph"):
            self.assertIn(key, inputs.ids)
        selected = html.split("function selected(){", 1)[1].split("function ", 1)[0]
        self.assertNotIn("validation", selected.lower())
        apply = html.split("async function applyValidation(){", 1)[1].split(
            "$('validationBatchEnabled').onchange", 1)[0]
        self.assertIn("fetch('/api/validation'", apply)
        self.assertNotIn("constrain()", apply)
        self.assertNotIn("apply()", apply)
        self.assertNotIn("/api/state", apply)
        self.assertIn("$('timingEnabled').onchange=applyTiming", html)
        self.assertLess(html.index('id="modelIntensity"'), html.index('id="localTone"'))
        self.assertLess(html.index('id="localTone"'), html.index('id="localStructure"'))


if __name__ == "__main__":
    unittest.main(verbosity=2)
