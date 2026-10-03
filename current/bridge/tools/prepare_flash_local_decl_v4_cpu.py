"""Keep each recorded access-state observation on its own command list.

CPU recording order cannot establish execution ordering between command lists.
The existing reviewed direct-route submission/copy/fence contract remains the
execution contract; the observer describes that list's explicit declaration.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PERF = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001")
BASE = PROJECT / "artifacts/periodic-flash-list-state-v3-source-20261001"
SOURCE = PROJECT / "artifacts/periodic-flash-local-decl-v4-source-20261001"
BUILD = PERF / "periodic-flash-local-decl-v4-native-build"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError("expected one exact source edit: " + before[:100])
    return text.replace(before, after, 1)


def main():
    parent_path = PERF / "periodic-flash-list-state-v3-native-build/build-receipt.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent["status"] != "cpu_build_passed":
        raise RuntimeError("completed parent build required")
    original = {}
    for name, digest in parent["source_files"].items():
        data = (BASE / name).read_bytes()
        if sha(data) != digest:
            raise RuntimeError("parent changed: " + name)
        original[name] = data
    evidence_path = PERF / "periodic-flash-list-state-v3-live-20261001/one-shot-completed.json"
    evidence_bytes = evidence_path.read_bytes()
    evidence = json.loads(evidence_bytes)
    summary = evidence["summary"]
    current = summary["current"]
    if (not evidence.get("completed") or current["reason_high16"] != 8
            or current["known_bit11"] is not False or current["shadow_bit12"] is not False
            or summary["source_geometry"] != {"width": 1280, "height": 720}
            or summary["health_failed"] is not False
            or summary["native_counters"]["original_fallback"] == 0):
        raise RuntimeError("completed cross-list rejection on actual direct source required")
    if SOURCE.exists() or BUILD.exists():
        raise RuntimeError("fresh stage required")
    cpp_name = "src/nr_sync_probe.cpp"
    cpp = original[cpp_name].decode("utf-8").replace("\r\n", "\n")
    start = cpp.index("    // A same-resource declaration on another recording list makes its old\n")
    end = cpp.index("    if(acquired!=ProofAcquire::acquired) return;", start)
    cpp = replace_once(cpp, cpp[start:end], '''    // This observer describes each list's recorded local declaration. Other
    // lists may be recorded in parallel and cannot overwrite that declaration
    // by CPU arrival order. GPU execution ordering remains the reviewed direct
    // route's existing prefix/private/suffix queue and fence contract.
    for(auto& list:lists_) if(list.generation && !list.closed) for(auto& c:list.colors) {
        if(c.identity.cookie && c.object==resource &&
                (acquired!=ProofAcquire::acquired || c.identity.cookie!=identity.cookie))
            list.rejected=ColorProofReason::identity;
    }
''')
    cpp = replace_once(cpp, '''    // Only a full local transition can renew an externally stale observation.
    // Unknown/identity/capacity/partial/split/contradictory-chain failures retain
    // their existing sticky rejection and cannot be cleared by another list.
    const bool renewed=plain && color->rejected==ColorProofReason::cross_list;
    if(renewed) color->rejected=ColorProofReason::ready;
    if(!plain) color->rejected=ColorProofReason::partial_split; // sticky until successful Reset
    else if(!renewed && color->transition.seen && color->rejected==ColorProofReason::ready &&
''', '''    if(!plain) color->rejected=ColorProofReason::partial_split; // sticky until successful Reset
    else if(color->transition.seen && color->rejected==ColorProofReason::ready &&
''')
    test_name = "tests/state_proof_cpu.cpp"
    test = original[test_name].decode("utf-8").replace("\r\n", "\n")
    start = test.index("void same_resource_cross_list() {")
    end = test.index("void capacity_sticky() {", start)
    test = replace_once(test, test[start:end], '''void same_resource_cross_list() {
    for(int variant=0;variant<3;++variant) {
        Fixture f;nrb::SyncProbe p(f.ids.ops(),true);f.begin(p);
        if(variant==0) p.reset(f.list(1),true);
        if(variant<2) f.barrier(p,1,0,psr,uav);
        else {D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;
            b.UAV.pResource=f.resource();p.barriers(f.list(1),1,&b);}
        CHECK(allowed(f.observe(p))); // another recording is not this list's state
        if(variant==0) {
            rejected(f.observe(p,1),Reason::not_psr);
            f.barrier(p,1,0,uav,psr);CHECK(allowed(f.observe(p,1)));
        }
        f.barrier(p,0,0,psr,uav);rejected(f.observe(p),Reason::not_psr);
        if(variant==0) CHECK(allowed(f.observe(p,1)));
        f.barrier(p,0,0,uav,psr);CHECK(allowed(f.observe(p)));
    }
}
void cross_list_is_resource_local_and_can_extend_age() {
    Fixture f;nrb::SyncProbe p(f.ids.ops(),true);f.begin(p);
    f.barrier(p,0,1);p.reset(f.list(1),true);
    for(int i=0;i<80;++i) {
        f.barrier(p,1,0,psr,uav);f.barrier(p,1,0,uav,psr);
    }
    auto s=f.observe(p);
    CHECK(s.incomplete);CHECK(allowed(s));CHECK(allowed(f.observe(p,0,1)));
    CHECK(!nrb::legacy_dlss_color_source(s));
    CHECK(nrb::dlss_color_processing_source(s,true));
    CHECK(!nrb::dlss_color_processing_source(s,false));
    CHECK(!nrb::dlss_color_processing_source(s,true,true));
}
void cross_list_cannot_erase_permanent_rejections() {
    for(int variant=0;variant<2;++variant) {
        Fixture f;nrb::SyncProbe p(f.ids.ops(),true);f.begin(p);
        const auto bad=(variant==0?f.transition(0,psr,uav,0):f.transition(0,copy_dest,uav));
        p.barriers(f.list(),1,&bad);
        const auto reason=variant==0?Reason::partial_split:Reason::state_chain;
        rejected(f.observe(p),reason);
        p.reset(f.list(1),true);f.barrier(p,1,0,uav,psr);
        rejected(f.observe(p),reason);
        f.barrier(p,0,0,psr,uav);rejected(f.observe(p),reason);
        f.barrier(p,0,0,uav,psr);rejected(f.observe(p),reason);
    }
}
''')
    test = replace_once(test,
        '"same_resource_external_state_requires_fresh_local_transition"',
        '"other_recording_does_not_rewrite_this_lists_declaration"')
    test = replace_once(test, '"cross_list_resource_local_and_strict_age_extension"',
                        '"same_resource_other_recording_churn_preserves_strict_age_extension"')
    changed = {cpp_name: cpp.encode("utf-8"), test_name: test.encode("utf-8")}
    SOURCE.mkdir(parents=True)
    for name, data in original.items():
        path = SOURCE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(changed.get(name, data))
    BUILD.mkdir(parents=True)
    frozen_evidence = BUILD / "completed-evidence-input.json"
    frozen_evidence.write_bytes(evidence_bytes)
    receipt = {"status": "prepared_cpu_only", "source_root": str(SOURCE),
        "source_files": {name: sha((SOURCE / name).read_bytes()) for name in original},
        "changed_files": sorted(changed), "parent_build_receipt_sha256": sha(parent_path.read_bytes()),
        "completed_live_evidence": str(frozen_evidence),
        "completed_live_evidence_sha256": sha(evidence_bytes),
        "runtime_scope": "legacy-direct-reviewed-v1", "GPU_executed": False,
        "source_contract": "Own-list recorded state only; CPU recording order of peers is not an execution timeline. Existing accepted gate and direct submission/copy/fence contract unchanged.",
        "CPU_cases": 17,
        "official_basis": [
            "https://learn.microsoft.com/en-us/windows/win32/direct3d12/porting-from-direct3d-11-to-direct3d-12",
            "https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-resourcebarrier"],
        "limitation": "Scoped recorded legacy access state only; not a GPU state/alias audit or a new execution-order proof. Actual same-frame route remains required.",
        "installation": "not installed"}
    (BUILD / "source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "changed_files": sorted(changed)}))


if __name__ == "__main__":
    main()
