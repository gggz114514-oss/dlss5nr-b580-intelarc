"""Small pure CPU acceptance; no HTTP, game, native library, or G access."""
import importlib.util
import io
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("flash_v2_capture", Path(__file__).with_name("capture_periodic_flash_diag_v2_cpu.py"))
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def observation(revision=1, *, processed=10, raw=0, status="ok", dropped=0):
    return {"health_failed": False, "health_reason": None, "periodic_flash_v2": {
        "status": status, "revision": revision, "native_header": {"frames_overwritten": processed, "dropped": dropped},
        "counters": {"nr_processed": processed, "original_fallback": raw}, "windows": [{"sr_sequence": 40}]}}


class CpuCapture(unittest.TestCase):
    def test_only_existing_loopback_get_path_and_output_scope(self):
        self.assertEqual(capture.endpoint("http://127.0.0.1:8765/"), "http://127.0.0.1:8765/api/state")
        self.assertEqual(capture.endpoint("http://127.0.0.1:54321/api/state"), "http://127.0.0.1:54321/api/state")
        for url in ("https://127.0.0.1:8765/", "http://example.com:8765/", "http://127.0.0.1:8765/api/validation", "http://127.0.0.1:8765/?token=a"):
            with self.assertRaises(ValueError):
                capture.endpoint(url)
        with self.assertRaises(ValueError):
            capture.output_path("G:/flash.jsonl")

    def test_missing_disabled_or_uninitialized_remains_unknown(self):
        self.assertEqual(capture.extract({})["periodic_flash_v2"]["status"], "unavailable")
        state = {"health": {"failed": True, "reason": "fixture", "temporal_diagnostics": {
            "periodic_flash_v2": {"status": "unavailable", "reason": "no_python_nr_call_yet"}}}}
        result = capture.extract(state)
        self.assertTrue(result["health_failed"])
        self.assertEqual(result["periodic_flash_v2"]["reason"], "no_python_nr_call_yet")

    def test_normal_frames_do_not_write_but_recovery_and_loss_changes_do(self):
        stream, writer = io.StringIO(), None
        writer = capture.ChangeWriter(stream)
        writer.accept(observation())
        for frame in range(11, 120):
            writer.accept(observation(processed=frame))
        self.assertEqual(writer.records, 1)
        writer.accept(observation(revision=2, raw=1))
        writer.accept(observation(revision=3, raw=1))  # window gains its recovery row
        writer.accept(observation(revision=3, raw=1, status="busy"))
        writer.accept(observation(revision=3, raw=1, dropped=1))
        writer.finish("duration")
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual([row["kind"] for row in rows], ["start", "change", "change", "change", "change", "end"])
        self.assertFalse(rows[-1]["negative_evidence_complete"])
        self.assertEqual(rows[-1]["last_observation"]["periodic_flash_v2"]["revision"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
