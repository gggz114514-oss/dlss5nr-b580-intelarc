"""Validate an unapplied native diagnostic patch; never load the ASI or a GPU API."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TASK = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\periodic-flash-20261001")
PATCH = ROOT / "tools/periodic_flash_native_diag_v1.patch"
BASE_SHA = {
    "src/asi.cpp": "4943462ef92dee321a9b705c55d3c83d278a55f3d366c8c937e62ab04913c38f",
    "src/deferred_identity.h": "de7d85f3291f049341dcfa1ec3dcc6bca8c5d4c6eb919a78f51d7bb717e6494e",
    "src/deferred_identity.cpp": "9ab59f8d2cbb1a459dd71aca09d236ab9fbcd73daf5253984dc035ac3ca910a8",
}
NEW_HEADER = "src/periodic_flash_native_diag.h"


def apply_in_memory():
    """Strict unified-diff context/line-count check, restricted to four paths."""
    lines = PATCH.read_text(encoding="utf-8").splitlines(keepends=True)
    result = {}
    i = 0
    while i < len(lines):
        if not lines[i].startswith("--- "):
            i += 1
            continue
        old_name = lines[i][4:].strip()
        new_name = lines[i + 1][4:].strip()
        if not lines[i + 1].startswith("+++ b/"):
            raise AssertionError("unexpected patch target")
        name = new_name[2:]
        if name not in {*BASE_SHA, NEW_HEADER} or name in result:
            raise AssertionError(f"unexpected or repeated path: {name}")
        if old_name == "/dev/null":
            if name != NEW_HEADER or (ROOT / name).exists():
                raise AssertionError("new header already exists")
            source = []
        else:
            if old_name != "a/" + name:
                raise AssertionError("renamed source")
            source = (ROOT / name).read_text(encoding="utf-8").splitlines(keepends=True)
        output = []
        cursor = 0
        i += 2
        while i < len(lines) and lines[i].startswith("@@ "):
            match = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*\n", lines[i])
            if not match:
                raise AssertionError("invalid hunk")
            old_start, old_count, new_start, new_count = (int(v) if v is not None else 1 for v in match.groups())
            offset = old_start - 1 if old_count else old_start
            if offset < cursor:
                raise AssertionError("overlapping hunk")
            output.extend(source[cursor:offset])
            if len(output) != (new_start - 1 if new_count else new_start):
                raise AssertionError("new hunk offset mismatch")
            cursor = offset
            consumed = added = 0
            i += 1
            while i < len(lines) and lines[i][:1] in {" ", "+", "-"} and not lines[i].startswith("--- "):
                prefix, value = lines[i][0], lines[i][1:]
                if prefix in {" ", "-"}:
                    if cursor >= len(source) or source[cursor] != value:
                        raise AssertionError(f"context mismatch in {name}:{cursor + 1}")
                    cursor += 1
                    consumed += 1
                if prefix in {" ", "+"}:
                    output.append(value)
                    added += 1
                i += 1
            if (consumed, added) != (old_count, new_count):
                raise AssertionError("hunk line count mismatch")
        output.extend(source[cursor:])
        result[name] = "".join(output)
    if set(result) != {*BASE_SHA, NEW_HEADER}:
        raise AssertionError("incomplete patch")
    return result


HARNESS = r'''
#include "periodic_flash_native_diag.h"
#include <cassert>
#include <thread>
#include <vector>
using namespace nrb::flashdiag;
uint64_t get(const Snapshot& s,Counter c) {return s.counters[uint32_t(c)];}
int main() {
    Recorder r;Context c{};c.eligible=1;c.route=0;c.effective_route=1;
    // Healthy -> source gate rejects one frame -> healthy with game Reset.
    // Delegation and compositing are separate facts; a rejected frame cannot
    // increment either successful-stage count merely from historical success.
    for(int i=0;i<3;++i) {c.eval_id=r.begin();r.count(Counter::eligible);
        r.count(Counter::delegated);r.count(Counter::submitted);r.count(Counter::nr_composited);}
    c.eval_id=r.begin();r.count(Counter::eligible);c.source_bits=0;
    const auto rejected_id=c.eval_id;r.fallback(c,Reason::source_state);
    c.eval_id=r.begin();r.count(Counter::eligible);c.game_reset=1;
    r.count(Counter::game_reset);r.event(EventKind::reset,c);
    r.count(Counter::delegated);r.count(Counter::submitted);r.count(Counter::nr_composited);
    Context fg{};fg.eval_id=r.begin();r.fallback(fg,Reason::scope_or_feature);
    Snapshot s{};assert(r.snapshot(&s));
    assert(get(s,Counter::seen)==6 && get(s,Counter::eligible)==5);
    assert(get(s,Counter::delegated)==4 && get(s,Counter::submitted)==4);
    assert(get(s,Counter::nr_composited)==4 && get(s,Counter::original_fallback)==1);
    assert(get(s,Counter::excluded_original)==1 && get(s,Counter::game_reset)==1);
    assert(s.retained==2 && s.events[0].context.eval_id==rejected_id);
    assert(s.events[0].composited_before==3 && s.events[1].context.game_reset==1);
    assert(s.events[0].context.route==0 && s.events[0].context.effective_route==1);
    // An SR slot whose caller loses the production route scope is a raw
    // fallback worth recording; FG/unrelated passthrough stays separate.
    Context untagged{};untagged.eval_id=r.begin();untagged.sr_slot=1;
    untagged.feature_id=1000001;r.fallback(untagged,Reason::scope_or_feature);
    // A submitted delegated frame may still take current-color fallback when
    // the NR failure latch is set; it is not a composited success.
    r.count(Counter::delegated);r.count(Counter::submitted);c.game_reset=0;
    r.fallback(c,Reason::latched_failure);assert(r.snapshot(&s));
    assert(get(s,Counter::nr_composited)==4 && get(s,Counter::original_fallback)==3);
    // Bounded ring with ordering and no loss of cumulative counters on wrap.
    for(int i=0;i<300;++i) {c.eval_id=r.begin();r.fallback(c,Reason::textures);}
    assert(r.snapshot(&s));assert(s.retained==capacity && s.overwritten==240);
    assert(get(s,Counter::original_fallback)==303);
    for(uint64_t i=1;i<s.retained;++i) {
        assert(s.events[i].serial==s.events[i-1].serial+1);
        assert(s.events[i].steady_us>=s.events[i-1].steady_us);
    }
    assert(!r.snapshot(nullptr));s.abi_size=0;assert(!r.snapshot(&s));
    // Concurrent producers keep every counter; contended events may be
    // dropped explicitly, never allocate more storage or block a submitter.
    Recorder concurrent;std::vector<std::thread> workers;
    for(int t=0;t<4;++t) workers.emplace_back([&] {
        Context frame{};frame.eligible=1;
        for(int i=0;i<1000;++i) {frame.eval_id=concurrent.begin();
            concurrent.fallback(frame,Reason::pending_full);}
    });
    for(auto& worker:workers) worker.join();s=Snapshot{};assert(concurrent.snapshot(&s));
    assert(get(s,Counter::seen)==4000 && get(s,Counter::original_fallback)==4000);
    assert(s.retained<=capacity && s.retained+s.overwritten+s.dropped==4000);
    assert(sizeof(Recorder)<10000 && sizeof(Snapshot)<10000);
}
'''


class NativeDiagnosticCpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name, expected in BASE_SHA.items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise AssertionError(f"live source changed; do not run this unapplied-patch test: {name}")
        cls.patched = apply_in_memory()
        TASK.mkdir(parents=True, exist_ok=True)
        cls.scratch = Path(tempfile.mkdtemp(prefix="cpu-native-diag-", dir=TASK)).resolve()
        if cls.scratch.parent != TASK.resolve():
            raise AssertionError("temporary output escaped task directory")
        for name, value in cls.patched.items():
            path = cls.scratch / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding="utf-8", newline="\n")
        vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
        vs = subprocess.check_output([str(vswhere), "-latest", "-products", "*", "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()
        setup = Path(vs) / "VC/Auxiliary/Build/vcvars64.bat"
        # Keep this compiler child isolated from the app's very long PATH.
        setup_env = {key: value for key, value in os.environ.items() if key.lower() != "path"}
        git = shutil.which("git.exe") or shutil.which("git")
        setup_env["Path"] = r"C:\Windows\System32;C:\Windows" + (";" + str(Path(git).parent) if git else "")
        environment = subprocess.check_output(
            f'cmd.exe /d /s /c ""{setup}" >nul && set"',
            env=setup_env, text=True, encoding="utf-8", errors="replace")
        cls.env = setup_env
        cls.env.update(line.split("=", 1) for line in environment.splitlines() if "=" in line and not line.startswith("="))
        cls.cl = shutil.which("cl.exe", path=cls.env.get("Path", cls.env.get("PATH")))
        if not cls.cl:
            raise AssertionError("CPU compiler missing")

    @classmethod
    def tearDownClass(cls):
        for name, expected in BASE_SHA.items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
        assert not (ROOT / NEW_HEADER).exists()
        assert cls.scratch.parent == TASK.resolve() and not cls.scratch.is_junction()
        shutil.rmtree(cls.scratch)

    def run_cpu(self, command, cwd=None):
        result = subprocess.run(command, cwd=cwd or self.scratch, env=self.env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_patch_applies_without_mutating_live_source(self):
        self.run_cpu(["git", "apply", "--check", str(PATCH)], cwd=ROOT)

    def test_real_recorder_cpu_sequences_bounded_ring_and_concurrency(self):
        path = self.scratch / "recorder_cpu.cpp"
        path.write_text(HARNESS, encoding="utf-8", newline="\n")
        exe = self.scratch / "recorder_cpu.exe"
        self.run_cpu([self.cl, "/nologo", "/std:c++20", "/EHsc", "/W4", "/WX",
            "/I" + str(self.scratch / "src"), str(path), "/Fe:" + str(exe)])
        self.run_cpu([str(exe)])

    def test_production_native_translation_units_compile_only(self):
        opti = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\optiscaler-sol-ref")
        includes = [self.scratch / "src", ROOT / "src", ROOT / "include",
            opti / "external/nvngx_dlss_sdk", opti / "OptiScaler/include"]
        macros = ["NOMINMAX", "WIN32_LEAN_AND_MEAN", "NRB_XESS_IDENTITY_TEST=1",
            "NRB_DEFERRED_IDENTITY_TEST=1", "NRB_LIVE_NR_TEST=1", "NRB_TAIL_ORDER_PROBE=1"]
        command = [self.cl, "/nologo", "/c", "/O2", "/MD", "/std:c++20", "/EHsc"]
        command += ["/I" + str(p) for p in includes] + ["/D" + v for v in macros]
        self.run_cpu(command + [str(self.scratch / "src/asi.cpp"), str(self.scratch / "src/deferred_identity.cpp")])

    def test_source_route_gates_and_color_substitution_preserved(self):
        old = (ROOT / "src/asi.cpp").read_text(encoding="utf-8")
        new = self.patched["src/asi.cpp"]
        for start, end in [("RouteDecision classify(", "NVSDK_NGX_Result __cdecl intercept"),
                ("    const bool diagnostic_dlss_scope=", "#ifdef NRB_TAIL_ORDER_PROBE"),
                ("        const bool dlss_source_proven=", "        const int hook_state=")]:
            self.assertEqual(old[old.index(start):old.index(end, old.index(start))],
                new[new.index(start):new.index(end, new.index(start))])
        color_sets = r"params->Set\(NVSDK_NGX_Parameter_Color,[^;]+;"
        self.assertEqual(re.findall(color_sets, old), re.findall(color_sets, new))
        additions = "\n".join(line[1:] for line in PATCH.read_text(encoding="utf-8").splitlines()
            if line.startswith("+") and not line.startswith("+++"))
        self.assertNotIn("borrowed.color", additions)
        self.assertNotIn("CreateFile", self.patched[NEW_HEADER])
        self.assertNotIn("WriteFile", self.patched[NEW_HEADER])
        self.assertIn("frame.reset_history=frame.frame_id==1 || item.game_reset;", self.patched["src/deferred_identity.cpp"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
