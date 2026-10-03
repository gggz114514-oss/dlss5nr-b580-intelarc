"""Prepare startup observability and a diagnostic shadow of the new gate.

No game, GPU, DLL or installed runtime is opened by this preparer.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PERF = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001")
BASE = PROJECT / "artifacts/periodic-flash-state-proof-v1-build-source-20261001"
SOURCE = PROJECT / "artifacts/periodic-flash-startup-shadow-v1-source-20261001"
BUILD = PERF / "periodic-flash-startup-shadow-v1-native-build"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError("expected one exact source edit: " + before[:100])
    return text.replace(before, after, 1)


def main():
    parent = json.loads((PERF / "periodic-flash-state-proof-v1-native-build/build-receipt.json").read_text(encoding="utf-8"))
    if parent["status"] != "cpu_build_passed":
        raise RuntimeError("completed parent build required")
    for name, digest in parent["source_files"].items():
        if sha((BASE / name).read_bytes()) != digest:
            raise RuntimeError("parent source changed: " + name)
    if SOURCE.exists() or BUILD.exists():
        raise RuntimeError("fresh task-owned stage and build directory required")
    original = {p.relative_to(BASE).as_posix(): p.read_bytes()
                for p in BASE.rglob("*") if p.is_file()}
    host_name = "game/nr_game_pre_xess_host.py"
    if sha(original[host_name]) != "0ab98f559e0b81cb4c92113dc7e9e4763931c1657b94de049ee9748e730fbc75":
        raise RuntimeError("unexpected host base")
    sources = {name: data.decode("utf-8").replace("\r\n", "\n") for name, data in original.items()}
    header_name = "include/nr_sync_probe.h"
    header = sources[header_name]
    header = replace_once(header,
        'inline constexpr char kDlssColorStateScope[]="legacy-direct-reviewed-v1";',
        'inline constexpr char kDlssColorStateScope[]="legacy-direct-reviewed-v1";\n'
        'inline constexpr char kDlssColorStateShadowScope[]="legacy-direct-shadow-v1";\n'
        'inline bool dlss_color_state_shadow_scope(std::string_view value) noexcept {\n'
        '    return value==kDlssColorStateShadowScope;\n}\n')
    header = replace_once(header, 'return value==kDlssColorStateScope;',
                          'return value==kDlssColorStateScope || dlss_color_state_shadow_scope(value);')
    header = replace_once(header,
        'inline bool dlss_color_source(const SyncObservation& s,bool implementation_known) noexcept {\n'
        '    if(!s.color_state.enabled) return legacy_dlss_color_source(s);',
        'inline bool dlss_color_source(const SyncObservation& s,bool implementation_known,\n'
        '                              bool shadow_only=false) noexcept {\n'
        '    // Shadow diagnostics retain the previously working gate, including its age bound.\n'
        '    if(shadow_only || !s.color_state.enabled) return legacy_dlss_color_source(s);')
    sources[header_name] = header
    asi_name = "src/asi.cpp"
    asi = sources[asi_name]
    asi = replace_once(asi, 'nrb::ProofAcquire acquire_color_identity(',
        'bool color_state_scope_shadow() noexcept {\n'
        '    char value[64]{};\n'
        '    const auto size=GetEnvironmentVariableA("NRB_DLSS_COLOR_STATE_PROOF_SCOPE",value,sizeof(value));\n'
        '    return size>0 && size<sizeof(value) && nrb::dlss_color_state_shadow_scope(value);\n}\n'
        'const bool color_state_shadow_mode=color_state_scope_shadow();\n'
        'nrb::ProofAcquire acquire_color_identity(')
    asi = replace_once(asi,
        'nrb::dlss_color_source(sync,color_implementation_known);',
        'nrb::dlss_color_source(sync,color_implementation_known,color_state_shadow_mode);')
    asi = replace_once(asi,
        '(unsigned(sync.color_state.known)<<11) | (unsigned(sync.color_state.reason)<<16);',
        '(unsigned(sync.color_state.known)<<11) | (unsigned(color_state_shadow_mode)<<12) |\n'
        '            (unsigned(sync.color_state.reason)<<16);')
    sources[asi_name] = asi
    test_name = "tests/state_proof_cpu.cpp"
    test = sources[test_name]
    test = replace_once(test, 'void unrelated_churn_and_ring_loss() {',
        'void shadow_gate_contract() {\n'
        '    CHECK(nrb::dlss_color_state_scope("legacy-direct-shadow-v1"));\n'
        '    CHECK(nrb::dlss_color_state_shadow_scope("legacy-direct-shadow-v1"));\n'
        '    CHECK(!nrb::dlss_color_state_shadow_scope("legacy-direct-reviewed-v1"));\n'
        '    CHECK(!nrb::dlss_color_state_shadow_scope("legacy-direct-shadow-v1 "));\n'
        '    Fixture f;nrb::SyncProbe p(f.ids.ops(),true);f.begin(p);\n'
        '    p.unknown_recording(f.list());auto s=f.observe(p);\n'
        '    rejected(s,Reason::implementation);\n'
        '    CHECK(nrb::legacy_dlss_color_source(s));\n'
        '    CHECK(nrb::dlss_color_source(s,false,true));\n'
        '    s.color.after=uav;CHECK(!nrb::dlss_color_source(s,true,true));\n'
        '    s.color.after=psr;s.closed=true;CHECK(!nrb::dlss_color_source(s,true,true));\n'
        '    s.closed=false;s.evaluate_sequence=s.color.sequence+65;\n'
        '    CHECK(!nrb::dlss_color_source(s,true,true));\n'
        '}\n'
        'void unrelated_churn_and_ring_loss() {')
    test = replace_once(test, '{"default_off_and_exact_enable_scope",disabled_contract},',
        '{"default_off_and_exact_enable_scope",disabled_contract},\n'
        '        {"shadow_retains_legacy_gate_and_reports_rejection",shadow_gate_contract},')
    sources[test_name] = test
    host = sources[host_name]
    host = replace_once(host,
        '    if _periodic_flash_diagnostics is None:\n'
        '        return {"status": "unavailable", "reason": "no_python_nr_call_yet"}\n'
        '    from periodic_flash_snapshot_v2 import WindowSampler',
        '    from periodic_flash_snapshot_v2 import PythonFrameRecorder, WindowSampler\n'
        '    # Native rejection evidence must also be visible before the first NR callback.\n'
        '    python_observed = _periodic_flash_diagnostics is not None\n'
        '    python_snapshot = (_periodic_flash_diagnostics.snapshot() if python_observed\n'
        '                       else PythonFrameRecorder().snapshot())')
    host = replace_once(host,
        '    return _periodic_flash_sampler.sample(getattr(web, "_native", None),\n'
        '                                         _periodic_flash_diagnostics.snapshot())',
        '    result = _periodic_flash_sampler.sample(getattr(web, "_native", None), python_snapshot)\n'
        '    result["python_nr_call_observed"] = python_observed\n'
        '    return result')
    # Every process/reset/model/bridge call remains in an identical AST.
    before_ast, after_ast = ast.parse(sources[host_name]), ast.parse(host)
    for tree in (before_ast, after_ast):
        tree.body = [node for node in tree.body if not
                     (isinstance(node, ast.FunctionDef) and node.name == "periodic_flash_diagnostics")]
    if ast.dump(before_ast) != ast.dump(after_ast):
        raise RuntimeError("host changed outside passive diagnostics")
    sources[host_name] = host
    changed = {header_name, asi_name, test_name, host_name}
    SOURCE.mkdir(parents=True)
    for name, data in original.items():
        target = SOURCE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(sources[name].encode("utf-8") if name in changed else data)
    BUILD.mkdir(parents=True)
    receipt = {
        "status": "prepared_cpu_only", "source_root": str(SOURCE),
        "source_files": {name: sha((SOURCE / name).read_bytes()) for name in original},
        "parent_build_receipt_sha256": sha((PERF / "periodic-flash-state-proof-v1-native-build/build-receipt.json").read_bytes()),
        "changed_files": sorted(changed), "host_process_math_reset_AST_unchanged": True,
        "runtime_scope": "legacy-direct-shadow-v1", "GPU_executed": False,
        "installation": "not installed", "shadow_limitation": "Old age gate remains; this is startup diagnosis, not flash-fix acceptance.",
    }
    (BUILD / "source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "changed": receipt["changed_files"],
                      "source": str(SOURCE), "build": str(BUILD)}))


if __name__ == "__main__":
    main()
