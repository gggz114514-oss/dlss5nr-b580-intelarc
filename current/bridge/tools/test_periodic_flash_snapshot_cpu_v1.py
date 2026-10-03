"""CPU-only ABI fixtures: raw little-endian buffers, no compiler or real DLL."""
from __future__ import annotations

import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
from unittest import mock
import unittest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "game/periodic_flash_snapshot_v1.py"
FROZEN = [ROOT / p for p in (
    "tools/periodic_flash_native_diag_v1.patch", "src/asi.cpp", "src/deferred_identity.cpp",
    "src/deferred_identity.h", "game/cyberpunk_nr_web.py", "game/cyberpunk_nr_adapter.py",
)]
spec = importlib.util.spec_from_file_location("periodic_flash_snapshot_cpu", HELPER)
helper = importlib.util.module_from_spec(spec)
with mock.patch.object(C, "CDLL", side_effect=AssertionError("no native DLL loading")):
    spec.loader.exec_module(helper)


def fixture(*, size=8544, version=1, retained=0, overwritten=0, dropped=0):
    # Offsets are independent of helper ctypes descriptors, from the native
    # header: 32-byte prefix, 12 counters, 28 reasons, then 64 128-byte events.
    data = bytearray(8544)
    struct.pack_into("<IIQQQ", data, 0, size, version, retained, overwritten, dropped)
    return data


class FakeFunction:
    def __init__(self, payload, returned=1, exception=None):
        self.payload, self.returned, self.exception = payload, returned, exception
        self.argtypes_sets = self.restype_sets = self.calls = 0
        self.input_headers = []

    @property
    def argtypes(self):
        return self._argtypes

    @argtypes.setter
    def argtypes(self, value):
        self.argtypes_sets += 1
        self._argtypes = value

    @property
    def restype(self):
        return self._restype

    @restype.setter
    def restype(self, value):
        self.restype_sets += 1
        self._restype = value

    def __call__(self, pointer):
        self.calls += 1
        self.input_headers.append(struct.unpack("<II", C.string_at(pointer, 8)))
        if self.exception:
            raise self.exception
        C.memmove(pointer, bytes(self.payload), len(self.payload))
        return self.returned


class FakeDll:
    def __init__(self, function=None):
        self.function, self.lookups = function, 0

    def __getattr__(self, name):
        if name == "NRB_GetPeriodicFlashDiag":
            self.lookups += 1
            if self.function is not None:
                return self.function
        raise AttributeError(name)


class SnapshotCpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sha = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in FROZEN}

    @classmethod
    def tearDownClass(cls):
        for path, sha in cls.sha.items():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, path

    def setUp(self):
        self.no_load = mock.patch.object(C, "CDLL", side_effect=AssertionError("no native DLL loading"))
        self.no_load.start()
        self.addCleanup(self.no_load.stop)

    def read(self, data, **kwargs):
        return helper.read_snapshot(FakeDll(FakeFunction(data, **kwargs)))

    def test_header_enums_fields_and_native_alignment(self):
        patch = FROZEN[0].read_text(encoding="utf-8")
        header = "\n".join(line[1:] for line in patch.splitlines()
            if line.startswith("+") and not line.startswith("+++"))
        for enum, values in (("Counter", helper.COUNTERS), ("Reason", helper.REASONS),
                ("EventKind", helper.EVENT_KINDS)):
            body = re.search(r"enum class " + enum + r" : uint32_t \{(.*?)\};", header, re.S).group(1)
            names = tuple(value.strip() for value in body.split(",") if value.strip() != "count")
            self.assertEqual(names, values)
        body = re.search(r"struct Context \{(.*?)\};", header, re.S).group(1)
        fields = []
        for width, declarations in re.findall(r"uint(32|64)_t ([^;]+);", body):
            fields.extend((item.split("=")[0].strip(), int(width)) for item in declarations.split(","))
        self.assertEqual(fields, [(name, C.sizeof(kind) * 8) for name, kind in helper.Context._fields_])
        self.assertEqual((C.sizeof(helper.Context), C.sizeof(helper.Event), C.sizeof(helper.Snapshot)), (80, 128, 8544))
        self.assertEqual([C.alignment(t) for t in (helper.Context, helper.Event, helper.Snapshot)], [8, 8, 8])
        self.assertEqual({name: getattr(helper.Event, name).offset for name, _ in helper.Event._fields_},
            {"serial": 0, "steady_us": 8, "composited_before": 16, "detail": 24, "detail2": 32,
             "context": 40, "kind": 120, "reason": 124})
        self.assertEqual({name: getattr(helper.Snapshot, name).offset for name, _ in helper.Snapshot._fields_},
            {"abi_size": 0, "version": 4, "retained": 8, "overwritten": 16, "dropped": 24,
             "counters": 32, "reasons": 128, "events": 352})

    def test_absent_export_and_missing_existing_dll_are_unavailable(self):
        dll = FakeDll()
        for _ in range(3):
            result = helper.read_snapshot(dll)
            self.assertEqual(result["status"], "unavailable")
            self.assertNotIn("error", result)
        self.assertEqual(dll.lookups, 1)
        self.assertEqual(helper.read_snapshot(None)["status"], "unavailable")
        self.assertEqual(helper.read_snapshot(12345)["status"], "unavailable")

    def test_busy_ignores_unwritten_buffer_and_is_not_error(self):
        result = self.read(bytearray(8544), returned=0)
        self.assertEqual(result["status"], "busy")
        self.assertNotIn("error", result)
        self.assertEqual(result["events"], [])
        self.assertNotIn("native_header", result)

    def test_bad_abi_size_version_and_retained_are_rejected(self):
        for changes, error in (({"size": 8540}, "invalid_abi_size"),
                ({"version": 2}, "invalid_version"), ({"retained": 65}, "invalid_retained"),
                ({"retained": 2**64 - 1}, "invalid_retained")):
            with self.subTest(changes=changes):
                result = self.read(fixture(**changes))
                self.assertEqual((result["status"], result["error"]), ("error", error))
                self.assertEqual(result["counters"], {})
                self.assertEqual(result["events"], [])

    def test_full_ring_raw_pointer_fixture_fields_and_source_bits(self):
        data = fixture(retained=64, overwritten=170, dropped=7)
        for i in range(64):
            offset = 352 + i * 128
            struct.pack_into("<5Q", data, offset, 900 + i, 200000 + i, 300 + i, 400 + i, 500 + i)
            struct.pack_into("<4Q", data, offset + 40, 600 + i, 0xFEDCBA9876543210, 700 + i, 1000001)
            struct.pack_into("<11I", data, offset + 72, 0, 6, 1, 1, 1, 0x800003FF, 128, 64, 1, 1, 2)
            struct.pack_into("<2I", data, offset + 120, 1, 2)
        result = self.read(data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["native_header"]["overwritten"], 170)
        self.assertEqual(result["native_header"]["dropped"], 7)
        self.assertEqual(len(result["events"]), 64)
        self.assertEqual([e["serial"] for e in result["events"]], list(range(900, 964)))
        last = result["events"][-1]
        self.assertEqual((last["kind"], last["reason"]), ("fallback", "source_state"))
        self.assertEqual(last["native_steady_us"], 200063)
        self.assertEqual(last["context"], {
            "eval_id": 663, "list": 0xFEDCBA9876543210, "generation": 763, "feature_id": 1000001,
            "route": 0, "evidence": 6, "effective_route": 1, "eligible": 1, "sr_slot": 1,
            "source_bits": 0x800003FF, "color_after": 128, "color_age": 64,
            "game_reset": 1, "enabled": 1, "history": 2,
        })
        self.assertTrue(all(last["source_bits"]["flags"].values()))
        self.assertEqual(last["source_bits"]["unknown_mask"], 0x80000000)
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_lifetime_uint64_counts_empty_ring_and_every_reason_name(self):
        data = fixture()
        values = [(2**64 - 1) - i for i in range(12)]
        struct.pack_into("<12Q", data, 32, *values)
        struct.pack_into("<28Q", data, 128, *range(28))
        result = self.read(data)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["counters"], dict(zip(helper.COUNTERS, values)))
        self.assertEqual(result["reasons"], dict(zip(helper.REASONS, range(28))))
        self.assertEqual(result["events"], [])
        self.assertEqual(json.loads(json.dumps(result))["counters"]["seen"], 2**64 - 1)

    def test_signature_initialized_once_and_clock_domains_kept_separate(self):
        data = fixture(retained=1)
        struct.pack_into("<Q", data, 360, 99)  # native steady_us independent of Python epoch
        function = FakeFunction(data)
        dll = FakeDll(function)
        with mock.patch.object(helper.time, "monotonic_ns", side_effect=[1000, 1003, 2000, 2007]):
            first = helper.read_snapshot(dll)
            second = helper.read_snapshot(dll)
        self.assertEqual((dll.lookups, function.argtypes_sets, function.restype_sets, function.calls), (1, 1, 1, 2))
        self.assertEqual(function.argtypes, [C.POINTER(helper.Snapshot)])
        self.assertIs(function.restype, C.c_int)
        self.assertEqual(function.input_headers, [(8544, 1), (8544, 1)])
        self.assertEqual(first["python_sample_time"], {"monotonic_ns_start": 1000, "monotonic_ns_end": 1003})
        self.assertEqual(second["python_sample_time"], {"monotonic_ns_start": 2000, "monotonic_ns_end": 2007})
        self.assertEqual(first["events"][0]["native_steady_us"], 99)
        self.assertFalse(first["clock_domains_aligned"])

    def test_read_exception_and_invalid_export_result_are_errors(self):
        result = self.read(fixture(), exception=OSError("fixture failure"))
        self.assertEqual(result["status"], "error")
        self.assertIsNotNone(result["python_sample_time"])
        result = self.read(fixture(), returned=-1)
        self.assertEqual((result["status"], result["error"]), ("error", "invalid_export_result"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
