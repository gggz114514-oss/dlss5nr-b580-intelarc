"""CPU-only: apply the proposed diff in memory and exercise the actual DiskOnly hook.
No Triton/torch imports, cache manager creation, DLL loading, GPU or G writes.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

DEFAULT_SOURCE = Path(r"G:\SteamLibrary\steamapps\common\Resident Evil Village BIOHAZARD VILLAGE\nr-runtime\fast_cached_runtime_v1.py")
PATCH = Path(__file__).with_name("front_missing_cache_diag_v1.patch")


def apply_in_memory(source, diff):
    original = source.splitlines(keepends=True)
    lines = diff.splitlines(keepends=True)
    output, cursor, i = [], 0, 0
    while i < len(lines):
        match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", lines[i])
        if not match:
            i += 1
            continue
        start = int(match[1]) - 1
        output.extend(original[cursor:start])
        cursor = start
        i += 1
        while i < len(lines) and not lines[i].startswith("@@"):
            line = lines[i]
            if line[0] in " -":
                if original[cursor] != line[1:]:
                    raise AssertionError("patch context does not match source")
                if line[0] == " ":
                    output.append(line[1:])
                cursor += 1
            elif line[0] == "+":
                output.append(line[1:])
            else:
                raise AssertionError("unexpected diff line")
            i += 1
    output.extend(original[cursor:])
    return "".join(output)


class DiskOnlyCPU(unittest.TestCase):
    def setUp(self):
        self.original_calls, self.cache_lookups, self.key_inputs = [], [], []
        self.group = None
        self.env = {"CPU_TEST_ENV": "actual-supplied-value"}
        self.env_queries = []
        self.knobs = NS(compilation=NS(always_compile=False, override=False), runtime=NS(jit_cache_hook=None))
        self.modules = {}
        for name in ("triton", "triton.runtime", "triton.runtime.driver", "triton.runtime.cache",
                     "triton.compiler", "triton.compiler.compiler", "triton._C", "triton._C.libtriton", "portable_cache"):
            self.modules[name] = ModuleType(name)
        self.modules["triton"].knobs = self.knobs
        self.modules["triton.runtime.driver"].active = NS(get_current_target=lambda: (_ for _ in ()).throw(AssertionError("device query forbidden")))
        self.modules["triton.compiler.compiler"].make_backend = lambda target: NS(parse_options=lambda opts: NS(**opts))
        def cache_key(src, backend, options, env):
            self.key_inputs.append((src, options, env))
            return "recorded-cpu-key-material"
        self.modules["triton.runtime.cache"].get_cache_key = cache_key
        def manager(key):
            self.cache_lookups.append(key)
            return NS(cache_dir=Path("cpu-cache") / key, get_group=lambda name: self.group)
        self.modules["triton.runtime.cache"].get_cache_manager = manager
        def query_env():
            self.env_queries.append(True)
            return self.env
        self.modules["triton._C.libtriton"].get_cache_invalidating_env_vars = query_env
        self.modules["portable_cache"].fast_cache_options = lambda opts: {**opts, "extern_libs": (("libdevice", "identity-token"),)}
        self.module = ModuleType("cpu_diskonly_under_test")
        self.module.__file__ = str(SOURCE)
        exec(PATCHED_SOURCE, self.module.__dict__)
        def original(src, **kwargs):
            self.original_calls.append((src, kwargs))
            return NS(metadata_group={})
        self.fn = type("CPUJit", (), {})()
        self.fn.fn = NS(__module__="front_noise_native_720_kernel_v1", __qualname__="front")
        self.fn.cache_key = "reviewed-source-key"
        self.fn.compile = original
        self.specialization = json.dumps({"signature": {"SEED": "i64"}, "constant_vals": [720,1280,768,1280,False,True,False,128]})
        self.src = NS(name="front", hash=lambda: "reviewed-ast-hash")
        self.options = {"num_warps": 4, "num_stages": 2, "enable_fp_fusion": False}

    @contextmanager
    def enter(self):
        with self.module.DiskOnly() as scope:
            self.knobs.runtime.jit_cache_hook(fn=NS(jit_function=self.fn), compile={"specialization_data": self.specialization})
            yield scope

    def test_front_miss_exact_key_abi_options_environment_and_no_compile(self):
        with patch.dict(sys.modules, self.modules):
            with self.enter():
                with self.assertRaises(RuntimeError) as error:
                    self.fn.compile(self.src, target="cpu-target", options=self.options)
        detail = json.loads(str(error.exception).split("front_cache_diagnostic=",1)[1])
        self.assertEqual(detail["expected_kernel_hash"], hashlib.sha256(b"recorded-cpu-key-material").hexdigest())
        self.assertEqual(detail["cache_key_material"], "recorded-cpu-key-material")
        self.assertEqual(json.loads(detail["specialization_data"])["signature"]["SEED"], "i64")
        self.assertEqual(detail["parsed_options"]["extern_libs"], [["libdevice", "identity-token"]])
        self.assertEqual(detail["env_vars"], self.env)
        self.assertEqual(detail["source_jit_cache_key"], "reviewed-source-key")
        self.assertEqual(detail["ast_source_hash"], "reviewed-ast-hash")
        self.assertEqual(detail["group_marker"], "__grp__front.json")
        self.assertEqual(self.original_calls, [])
        self.assertIsNone(self.knobs.runtime.jit_cache_hook)

    def test_explicit_environment_is_used_without_an_extra_query(self):
        explicit = {"CPU_EXPLICIT": "provided"}
        with patch.dict(sys.modules, self.modules):
            with self.enter():
                with self.assertRaises(RuntimeError) as error:
                    self.fn.compile(self.src, target="cpu-target", options=self.options, _env_vars=explicit)
        detail = json.loads(str(error.exception).split("front_cache_diagnostic=",1)[1])
        self.assertEqual(detail["env_vars"], explicit)
        self.assertEqual(self.env_queries, [])
        self.assertEqual(self.original_calls, [])

    def test_nonfront_missing_error_is_unchanged(self):
        self.src.name = "other"
        with patch.dict(sys.modules, self.modules):
            with self.enter():
                with self.assertRaisesRegex(RuntimeError, "^Missing fast artifact; prepare offline before video processing: other$"):
                    self.fn.compile(self.src, target="cpu-target", options=self.options)
        self.assertEqual(self.original_calls, [])

    def test_cache_hit_preserves_original_options_and_counters(self):
        self.group = {"front.json": "cpu-only-fixture"}
        with patch.dict(sys.modules, self.modules):
            with self.enter() as scope:
                self.fn.compile(self.src, target="cpu-target", options=self.options)
                self.assertEqual(scope.hits, 1)
                self.assertEqual(len(scope.rows), 1)
        self.assertEqual(len(self.original_calls), 1)
        self.assertEqual(self.original_calls[0][1]["target"], "cpu-target")
        self.assertEqual(self.original_calls[0][1]["options"], {**self.options,"extern_libs": (("libdevice","identity-token"),)})
        self.assertEqual(len(self.cache_lookups), 1)
        self.assertIsNone(self.knobs.runtime.jit_cache_hook)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args, remaining = parser.parse_known_args()
    SOURCE = args.source
    original = SOURCE.read_text(encoding="utf-8")
    PATCHED_SOURCE = apply_in_memory(original, PATCH.read_text(encoding="utf-8"))
    unittest.main(argv=[sys.argv[0],*remaining])
