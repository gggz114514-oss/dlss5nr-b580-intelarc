"""CPU fakes check attribution boundaries, retirement and original guard calls."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "game/nr_numeric_cost_meter_v1.py"
spec = importlib.util.spec_from_file_location("cost_meter_cpu", SOURCE)
meter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(meter_module)


class Tests(unittest.TestCase):
    def setUp(self):
        class Graph:
            def replay(self):
                self.calls = getattr(self, "calls", 0) + 1

        class Event:
            def __init__(self, **kwargs):
                self.recorded = False

            def record(self):
                self.recorded = True

            def query(self):
                return self.recorded

            def elapsed_time(self, other):
                return 7.25

        self.Graph = Graph
        self.original_replay = Graph.replay
        self.torch = NS(xpu=NS(XPUGraph=Graph, Event=Event,
                               is_current_stream_capturing=lambda: False))
        self.calls = []

        class Guard:
            def validate(inner):
                self.calls.append("child")

        class Owner:
            def __init__(inner):
                inner.children = {"branch_accum": Guard()}

            def validate_context(inner):
                self.calls.append("suite")
                inner.children["branch_accum"].validate()

        class Network:
            def _validate(inner):
                self.calls.append("graph")

        def tensor(count):
            return NS(numel=lambda: count, element_size=lambda: 2)

        self.gpu = Graph()
        entry = NS(graph=self.gpu, inputs={"x": tensor(10), "none": None}, output=tensor(3))
        graph = Network()
        graph.entries = {1: entry}
        graph.last_entry = entry
        self.modes = NS(session=NS(_stack=NS(graph=graph)), height=720,
                        numeric_cleanup_calls=Owner(), last_combo_frame_route="replay")
        self.host = NS(_modes=self.modes, _bridge=NS(torch=self.torch),
                       _active_optimizations=("num_branch_c64",))
        self.meter = meter_module.CostMeter()

    def tearDown(self):
        self.meter.close()
        self.assertIs(self.Graph.replay, self.original_replay)

    def process(self):
        self.modes.numeric_cleanup_calls.validate_context()
        self.modes.session._stack.graph._validate()
        self.gpu.replay()

    def test_original_guards_once_and_gpu_excludes_guard_timestamp(self):
        with self.meter.frame(self.host, 10):
            self.process()
        row = self.meter.snapshot()["last"]
        self.assertEqual(self.calls, ["suite", "child", "graph"])
        self.assertEqual(row["network_graph_gpu_ms"], 7.25)
        self.assertEqual(row["network_replay_count"], 1)
        self.assertEqual(row["static_input_copy_bytes"], 20)
        self.assertEqual(row["output_clone_bytes"], 6)
        self.assertGreater(row["numeric_guard_cpu_ms"], 0)
        self.assertGreaterEqual(row["numeric_guard_cpu_ms"], row["child_branch_accum_guard_cpu_ms"])

    def test_web_retirement_during_frame_does_not_access_closed_session(self):
        with self.meter.frame(self.host, 11):
            self.modes.session = None
            self.host._modes = NS(session=None, height=None)
            self.Graph().replay()
        self.assertEqual(self.meter.snapshot()["window_frames"], 0)
        self.assertIsNone(self.meter.snapshot()["error"])

    def test_capture_and_exception_are_not_steady_samples(self):
        with self.meter.frame(self.host, 12):
            self.process()
            self.modes.last_combo_frame_route = "capture"
        self.assertEqual(self.meter.snapshot()["window_frames"], 0)
        self.modes.last_combo_frame_route = "replay"
        with self.assertRaisesRegex(RuntimeError, "guard failure"):
            with self.meter.frame(self.host, 13):
                raise RuntimeError("guard failure")
        self.assertEqual(self.meter.snapshot()["window_frames"], 0)

    def test_no_probe_wait_or_second_replay_and_restore_shadowed_slots(self):
        owner = self.modes.numeric_cleanup_calls
        original = owner.validate_context
        with self.meter.frame(self.host, 14):
            self.process()
        self.assertEqual(self.gpu.calls, 1)
        self.meter.close()
        self.assertNotIn("validate_context", owner.__dict__)
        self.assertEqual(owner.validate_context, original)

    def test_event_start_error_preserves_one_original_replay(self):
        def fail_event(**kwargs):
            raise RuntimeError("timing unsupported")
        self.torch.xpu.Event = fail_event
        with self.meter.frame(self.host, 15):
            self.process()
        self.assertEqual(self.gpu.calls, 1)
        self.assertIn("event-start", self.meter.snapshot()["error"])
        self.assertIs(self.Graph.replay, self.original_replay)

    def test_unknown_graph_never_creates_an_event(self):
        def fail_event(**kwargs):
            raise AssertionError("unknown graph must not be timestamped")
        self.torch.xpu.Event = fail_event
        with self.meter.frame(self.host, 16):
            other = self.Graph()
            other.replay()
            self.modes.last_combo_frame_route = "capture"
        self.assertEqual(other.calls, 1)
        self.assertIsNone(self.meter.snapshot()["error"])

    def test_same_modes_session_replacement_changes_meter_binding(self):
        with self.meter.frame(self.host, 17):
            self.process()
        old_epoch = self.meter.snapshot()["epoch"]
        self.modes.session = NS(_stack=self.modes.session._stack)
        with self.meter.frame(self.host, 18):
            self.process()
        self.assertGreater(self.meter.snapshot()["epoch"], old_epoch)
        self.assertEqual(self.meter.snapshot()["window_frames"], 1)

    def test_capture_original_exception_is_not_retried(self):
        calls = []
        def failed_replay(graph):
            calls.append(graph)
            raise RuntimeError("original replay failed")
        self.Graph.replay = failed_replay
        self.original_replay = failed_replay
        self.torch.xpu.is_current_stream_capturing = lambda: True
        with self.assertRaisesRegex(RuntimeError, "original replay failed"):
            with self.meter.frame(self.host, 19):
                self.gpu.replay()
        self.assertEqual(calls, [self.gpu])
        self.assertIsNone(self.meter.snapshot()["error"])

    def test_end_event_error_does_not_repeat_successful_replay(self):
        class EndFailure:
            count = 0
            def __init__(self, **kwargs):
                EndFailure.count += 1
                self.end = EndFailure.count == 2
            def record(self):
                if self.end:
                    raise RuntimeError("end timing unsupported")
        self.torch.xpu.Event = EndFailure
        with self.meter.frame(self.host, 20):
            self.process()
        self.assertEqual(self.gpu.calls, 1)
        self.assertIn("event-end", self.meter.snapshot()["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
