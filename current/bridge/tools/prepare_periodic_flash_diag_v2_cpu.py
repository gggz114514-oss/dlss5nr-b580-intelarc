"""Stage a frozen, diagnosis-only host delta and CMake CPU build inputs on D.

Never installs, loads an ASI, invokes a GPU, or writes G / project target sources.
The existing build_* script emits an UNAPPLIED patch; this script only prepares
an isolated CMake tree and a build-asi-cpu.ps1 for main to compile explicitly.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parents[1]
PERF = Path("D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/live-web-perf-20261001")
OUTPUT = PERF / "periodic-flash-sol61-install-v2"
ACTUAL_HOST = Path("G:/SteamLibrary/steamapps/common/Resident Evil Village BIOHAZARD VILLAGE/nr-runtime/game/nr_game_pre_xess_host.py")
G_SHA = "c76bfecfdb1a417d214b5c9132402aa4aae4848e967715f93936408dae0c30af"
E_SHA = "1fec385b544fe42418c2a20b08970589c5fec46b1be5e1c17c2d471c2cdaefbd"
V1_HELPER_SHA = "e85e2ddbc2b311242e702c585eefc9f4299aec06a687ef443e764d968c574913"
OPTI = Path("D:/Codex-NR-Experiments/cyberpunk-opt/optiscaler-sol-ref")
CMAKE = Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe")
FLAGS = {
    "NRB_XESS_IDENTITY_TEST": "ON", "NRB_DEFERRED_IDENTITY_TEST": "ON",
    "NRB_LIVE_NR_TEST": "ON", "NRB_MULTI_ROUTE_LIVE": "ON", "NRB_TAIL_ORDER_PROBE": "ON",
    "NRB_ONE_FRAME_COLOR_READBACK": "OFF", "NRB_BUILD_GENERIC_DLSS_PROBE": "OFF",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_builder():
    spec = importlib.util.spec_from_file_location("flash_v2_builder", PROJECT / "tools/build_periodic_flash_diag_v2_cpu.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def normalized(data):
    return data.decode("utf-8").replace("\r\n", "\n")


def splice_delta(before, after, current):
    """Apply only the audited line changes, with a byte-for-byte normalized base.

    No fuzzy matching or whole E-host substitution: unexpected main edits stop
    preparation. Return hunks to make both diagnostic layers reviewable.
    """
    if current != before:
        raise AssertionError("host base changed; regenerate/review against actual G host")
    old, new, lines = before.splitlines(True), after.splitlines(True), current.splitlines(True)
    changes = []
    for operation, i, j, k, end in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if operation != "equal":
            changes.append({"operation": operation, "old_line": i + 1, "old_lines": j - i,
                            "new_line": k + 1, "new_lines": end - k})
    for operation, i, j, k, end in reversed(difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes()):
        if operation != "equal":
            if lines[i:j] != old[i:j]:
                raise AssertionError("host hunk lost its exact base")
            lines[i:j] = new[k:end]
    result = "".join(lines)
    if result != after:
        raise AssertionError("host delta reconstruction mismatch")
    return result, changes


def host_from_actual(actual_bytes, e_bytes, v2_host):
    if sha(actual_bytes) != G_SHA or sha(e_bytes) != E_SHA:
        raise AssertionError("G/E host SHA changed; do not overwrite using this handoff")
    original, v1 = normalized(actual_bytes), normalized(e_bytes)
    with_v1, hunks_v1 = splice_delta(original, v1, original)
    if any(h["old_lines"] for h in hunks_v1) or len(hunks_v1) != 4:
        raise AssertionError("expected exactly four addition-only v1 diagnostic hunks")
    final, hunks_v2 = splice_delta(v1, v2_host, with_v1)
    # Business call expressions are exact AST matches, including reset semantics.
    def critical(source):
        process = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "process")
        reset = next(n for n in ast.walk(process) if isinstance(n, ast.Assign) and
                     any(isinstance(t, ast.Name) and t.id == "reset" for t in n.targets))
        calls = [ast.dump(n) for n in ast.walk(process) if isinstance(n, ast.Call) and
                 isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and
                 n.func.value.id in ("_modes", "_bridge")]
        return ast.dump(reset), calls
    if critical(original) != critical(final):
        raise AssertionError("diagnostic delta changed a business/GPU call or reset")
    compile(final, "staged-actual-G-diagnostic-host", "exec")
    return final, {"G_to_E_v1": hunks_v1, "E_v1_to_v2": hunks_v2,
                   "critical_reset_and_bridge_model_AST_preserved": True}


def diff(before, after, path):
    return "diff --git a/" + path + " b/" + path + "\n" + "".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile="a/" + path, tofile="b/" + path))


def safe_path(path):
    for part in (path, *path.parents):
        if part.exists() and (part.is_symlink() or part.is_junction()):
            raise AssertionError("no junction/symlink traversal: " + str(part))


def copy_file(source, destination):
    safe_path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def main():
    builder = load_builder()
    receipt_path = PERF / "periodic-flash-sol61-cpu-v2.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not (receipt["passed"] and receipt["tests"] == 19 and not receipt["unexpected_gpu_modules"]):
        raise AssertionError("final 19 CPU tests must pass first")
    for name, digest in receipt["sources"].items():
        if sha(Path(name).read_bytes()) != digest:
            raise AssertionError("CPU receipt is stale: " + name)
    actual_bytes, e_bytes = ACTUAL_HOST.read_bytes(), builder.HOST.read_bytes()
    sources = builder.patched_sources()
    if builder.PATCH.read_text(encoding="utf-8") != builder.patch_text(sources):
        raise AssertionError("v2 patch differs from the CPU-checked generated sources")
    final_host, proof = host_from_actual(actual_bytes, e_bytes, sources[builder.HOST.relative_to(builder.WORKSPACE).as_posix()])
    helper_v1 = builder.HOST.parent / "nr_temporal_diagnostics_v1.py"
    if sha(helper_v1.read_bytes()) != V1_HELPER_SHA:
        raise AssertionError("v1 reset helper changed")
    safe_path(OUTPUT)
    if OUTPUT.exists() or OUTPUT.resolve().parent != PERF.resolve() or OUTPUT.drive.lower() != "d:":
        raise AssertionError("stage only into the fresh named D output; never reuse/clean a directory")
    OUTPUT.mkdir()
    source, build, payload = OUTPUT / "source", OUTPUT / "build", OUTPUT / "payload"
    copy_file(PROJECT / "CMakeLists.txt", source / "CMakeLists.txt")
    for directory in ("src", "include", "tests", "shaders"):
        for path in sorted((PROJECT / directory).rglob("*")):
            safe_path(path)
            if path.is_file():
                copy_file(path, source / path.relative_to(PROJECT))
    # CMake POST_BUILD requires these files. They are NOT installation payloads;
    # installing them would overwrite main's independent public CPU fixes.
    for name in ("cyberpunk_nr_adapter.py", "cyberpunk_nr_web.py"):
        copy_file(PROJECT / "game" / name, source / "game" / name)
    for name, value in sources.items():
        if name.startswith("cyberpunk-b580-nr-opt/"):
            (source / name.removeprefix("cyberpunk-b580-nr-opt/")).write_text(value, encoding="utf-8", newline="\n")
    (payload / "game").mkdir(parents=True)
    (payload / "game/nr_game_pre_xess_host.py").write_text(final_host, encoding="utf-8", newline="\n")
    copy_file(helper_v1, payload / "game" / helper_v1.name)
    copy_file(PROJECT / "game/periodic_flash_snapshot_v2.py", payload / "game/periodic_flash_snapshot_v2.py")
    host_path = "game/nr_game_pre_xess_host.py"
    for name, before, after in (
        ("G-actual-to-E-v1-host.patch", normalized(actual_bytes), normalized(e_bytes)),
        ("E-v1-to-v2-host.patch", normalized(e_bytes), final_host),
        ("G-actual-to-v2-host.patch", normalized(actual_bytes), final_host),
    ):
        (OUTPUT / name).write_text(diff(before, after, host_path), encoding="utf-8", newline="\n")
    configure = [str(CMAKE), "-S", str(source), "-B", str(build), "-G", "Visual Studio 17 2022", "-A", "x64",
                 "-DOPTISCALER_SOURCE=" + str(OPTI)] + ["-D" + k + "=" + v for k, v in FLAGS.items()]
    compile_command = [str(CMAKE), "--build", str(build), "--config", "Release", "--target", "CyberpunkNRBridge"]
    build_ps = "$ErrorActionPreference = 'Stop'\n" + "\n".join(
        "& " + " ".join(ps_quote(a) for a in command) +
        "\nif ($LASTEXITCODE -ne 0) { throw 'CPU CMake build failed' }" for command in (configure, compile_command)) + "\n"
    (OUTPUT / "build-asi-cpu.ps1").write_text(build_ps, encoding="utf-8", newline="\n")
    runtime = ACTUAL_HOST.parent.parent
    installs = [{"from": str(payload / "game" / name), "to": str(runtime / "game" / name),
                 "sha256": sha((payload / "game" / name).read_bytes())}
                for name in ("nr_game_pre_xess_host.py", "nr_temporal_diagnostics_v1.py", "periodic_flash_snapshot_v2.py")]
    installs.insert(0, {"from_after_main_build": str(build / "Release/CyberpunkNRBridge.asi"),
                       "to": "G:/epic/Cyberpunk2077/bin/x64/plugins/CyberpunkNRBridge.asi", "sha256": "compute_after_build"})
    manifest = {"kind": "periodic-flash-v2-CPU-preparation-only", "installed": False,
                "linked_asi": False, "gpu_executed": False, "flags": FLAGS,
                "G_host_base": {"path": str(ACTUAL_HOST), "sha256": G_SHA},
                "E_v1_host_base": {"path": str(builder.HOST), "sha256": E_SHA},
                "host_delta_proof": proof, "cpu_receipt": str(receipt_path),
                "configure_argv": configure, "compile_argv": compile_command, "install_exactly": installs,
                "launch_process_environment": {"NR_DIAG_PERIODIC_FLASH_V2": "1"},
                "patches": {p.name: sha(p.read_bytes()) for p in OUTPUT.glob("*.patch")},
                "must_not_install_cmake_postbuild_adapter_or_web": True}
    if sha(ACTUAL_HOST.read_bytes()) != G_SHA or sha(builder.HOST.read_bytes()) != E_SHA:
        raise AssertionError("host base changed during preparation; output is not installable")
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prepared": str(OUTPUT), "manifest": str(OUTPUT / "manifest.json"),
                      "main_runs_cpu_build": str(OUTPUT / "build-asi-cpu.ps1"),
                      "G_host_diagnostic_delta": str(OUTPUT / "G-actual-to-v2-host.patch")}, indent=2))


if __name__ == "__main__":
    main()
