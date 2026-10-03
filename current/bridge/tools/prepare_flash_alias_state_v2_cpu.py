"""Narrow the fix to recorded access state and extend the working source gate.

Aliasing barriers do not provide a replacement StateBefore/StateAfter pair.
The engine's resource contents/alias ownership and our existing same-frame
queue/copy/lifetime contract remain separate from this legacy state observer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PERF = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001")
BASE = PROJECT / "artifacts/periodic-flash-startup-shadow-v1-source-20261001"
SOURCE = PROJECT / "artifacts/periodic-flash-alias-state-v2-source-20261001"
BUILD = PERF / "periodic-flash-alias-state-v2-native-build"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError("expected one exact source edit: " + before[:100])
    return text.replace(before, after, 1)


def main():
    parent_path = PERF / "periodic-flash-startup-shadow-v1-native-build/build-receipt.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent["status"] != "cpu_build_passed":
        raise RuntimeError("completed parent build required")
    original = {}
    for name, digest in parent["source_files"].items():
        data = (BASE / name).read_bytes()
        if sha(data) != digest:
            raise RuntimeError("parent changed: " + name)
        original[name] = data
    evidence_path = PERF / "periodic-flash-startup-shadow-v1-live-20261001/one-shot-completed.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not evidence.get("completed") or evidence["summary"]["reason_high16"] != 7:
        raise RuntimeError("completed live aliasing rejection required")
    if SOURCE.exists() or BUILD.exists():
        raise RuntimeError("fresh stage required")
    text = {name: data.decode("utf-8").replace("\r\n", "\n") for name, data in original.items()}
    cpp_name = "src/nr_sync_probe.cpp"
    text[cpp_name] = replace_once(text[cpp_name],
        '    if(b.Type==D3D12_RESOURCE_BARRIER_TYPE_ALIASING) {invalidate_all(ColorProofReason::aliasing);return;}',
        '    // Aliasing synchronizes overlapping memory usages; it is not an access-state\n'
        '    // transition and cannot rewrite every list\'s recorded StateAfter. The validated\n'
        '    // direct DLSS scope supplies the current live resource; actual copies and\n'
        '    // prefix/private/suffix queue ordering remain checked separately.\n'
        '    if(b.Type==D3D12_RESOURCE_BARRIER_TYPE_ALIASING) return;')
    header_name = "include/nr_sync_probe.h"
    text[header_name] = replace_once(text[header_name], 'struct SubmitObservation {',
        '// Extend the existing accepted source path; a diagnostic tracking miss must\n'
        '// not make all previously valid frames disappear. The new proof is needed\n'
        '// only where unrelated command traffic aged the old observation out.\n'
        'inline bool dlss_color_processing_source(const SyncObservation& s,\n'
        '        bool implementation_known,bool shadow_only=false) noexcept {\n'
        '    return legacy_dlss_color_source(s) ||\n'
        '        (!shadow_only && dlss_color_source(s,implementation_known));\n'
        '}\n'
        'struct SubmitObservation {')
    asi_name = "src/asi.cpp"
    text[asi_name] = replace_once(text[asi_name],
        'nrb::dlss_color_source(sync,color_implementation_known,color_state_shadow_mode);',
        'nrb::dlss_color_processing_source(sync,color_implementation_known,color_state_shadow_mode);')
    test_name = "tests/state_proof_cpu.cpp"
    test = text[test_name]
    test = replace_once(test,
        '        if(variant<2) p.barriers(f.list(1),1,&b); // list not Reset-observed; named unrelated alias still unknown',
        '        if(variant<2) p.barriers(f.list(1),1,&b); // alias alone does not replace recorded access state')
    test = replace_once(test,
        '        const auto why=variant<2?Reason::aliasing:Reason::unknown_event;\n'
        '        rejected(f.observe(p),why);f.barrier(p,0,0,psr,psr);rejected(f.observe(p),why);',
        '        if(variant<2) {\n'
        '            CHECK(allowed(f.observe(p)));\n'
        '            f.barrier(p,0,0,psr,uav);rejected(f.observe(p),Reason::not_psr);\n'
        '            f.barrier(p,0,0,uav,psr);CHECK(allowed(f.observe(p)));\n'
        '        } else {\n'
        '            rejected(f.observe(p),Reason::unknown_event);\n'
        '            f.barrier(p,0,0,psr,psr);rejected(f.observe(p),Reason::unknown_event);\n'
        '        }')
    test = replace_once(test, 'void unrelated_churn_and_ring_loss() {',
        'void processing_gate_extension() {\n'
        '    Fixture f;nrb::SyncProbe p(f.ids.ops(),true);f.begin(p);\n'
        '    p.unknown_recording(f.list());auto s=f.observe(p);\n'
        '    CHECK(!allowed(s));CHECK(nrb::legacy_dlss_color_source(s));\n'
        '    CHECK(nrb::dlss_color_processing_source(s,false));\n'
        '    s.evaluate_sequence=s.color.sequence+65;\n'
        '    CHECK(!nrb::dlss_color_processing_source(s,true));\n'
        '    p.reset(f.list(),true);f.barrier(p);s=f.observe(p);\n'
        '    s.evaluate_sequence=s.color.sequence+65;\n'
        '    CHECK(!nrb::legacy_dlss_color_source(s));CHECK(allowed(s));\n'
        '    CHECK(nrb::dlss_color_processing_source(s,true));\n'
        '    CHECK(!nrb::dlss_color_processing_source(s,true,true));\n'
        '    CHECK(!nrb::dlss_color_processing_source(s,false));\n'
        '    s.closed=true;CHECK(!nrb::dlss_color_processing_source(s,true));\n'
        '    s.closed=false;s.reset_observed=false;\n'
        '    CHECK(!nrb::dlss_color_processing_source(s,true));\n'
        '    s.reset_observed=true;s.color.after=uav;s.color_state.transition.after=uav;\n'
        '    CHECK(!nrb::dlss_color_processing_source(s,true));\n'
        '}\n'
        'void unrelated_churn_and_ring_loss() {')
    test = replace_once(test, '{"shadow_retains_legacy_gate_and_reports_rejection",shadow_gate_contract},',
        '{"shadow_retains_legacy_gate_and_reports_rejection",shadow_gate_contract},\n'
        '        {"processing_gate_preserves_working_path_and_extends_age",processing_gate_extension},')
    test = replace_once(test, '"alias_unknown_null_oversized_no_revival"',
                        '"alias_access_state_unchanged_unknown_batch_still_rejected"')
    text[test_name] = test
    changed = {cpp_name, header_name, asi_name, test_name}
    SOURCE.mkdir(parents=True)
    for name, data in original.items():
        path = SOURCE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text[name].encode("utf-8") if name in changed else data)
    BUILD.mkdir(parents=True)
    receipt = {"status": "prepared_cpu_only", "source_root": str(SOURCE),
        "source_files": {name: sha((SOURCE / name).read_bytes()) for name in original},
        "changed_files": sorted(changed), "parent_build_receipt_sha256": sha(parent_path.read_bytes()),
        "completed_live_evidence_sha256": sha(evidence_path.read_bytes()),
        "runtime_scope": "legacy-direct-reviewed-v1", "GPU_executed": False,
        "host_process_math_reset_AST_unchanged": True,
        "source_contract": "Existing accepted gate OR scoped recorded-state extension; aliasing is not an access-state transition.",
        "official_basis": [
            "https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-resource-barriers-to-synchronize-resource-states-in-direct3d-12",
            "https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_resource_aliasing_barrier"],
        "limitation": "Recorded legacy access state only; relies on the established same-frame direct engine resource/copy/queue contract, not a physical alias-heap audit.",
        "installation": "not installed"}
    (BUILD / "source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "changed_files": sorted(changed)}))


if __name__ == "__main__":
    main()
