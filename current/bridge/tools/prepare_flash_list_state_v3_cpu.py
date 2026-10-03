"""Refresh scoped recorded state after an explicit local transition.

The observer does not establish GPU execution order from CPU recording order.
An external same-resource record invalidates the old resource observation, but
must not permanently reject the list recording a new explicit state declaration.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PERF = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001")
BASE = PROJECT / "artifacts/periodic-flash-alias-state-v2-source-20261001"
SOURCE = PROJECT / "artifacts/periodic-flash-list-state-v3-source-20261001"
BUILD = PERF / "periodic-flash-list-state-v3-native-build"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError("expected one exact source edit: " + before[:100])
    return text.replace(before, after, 1)


def main():
    parent_path = PERF / "periodic-flash-alias-state-v2-native-build/build-receipt.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent["status"] != "cpu_build_passed":
        raise RuntimeError("completed parent build required")
    original = {}
    for name, digest in parent["source_files"].items():
        data = (BASE / name).read_bytes()
        if sha(data) != digest:
            raise RuntimeError("parent changed: " + name)
        original[name] = data
    evidence_path = PERF / "periodic-flash-alias-state-v2-live-20261001/one-shot-completed.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    summary = evidence["summary"]
    active = evidence["full_response"]["active_mode"]
    if (not evidence.get("completed") or summary["native_current"]["reason_high16"] != 8
            or active["input_size"] != 720 or active["experiment_720"] != "c512_k8"
            or summary["health_failed"] is not False
            or not any(row["raw_fallback"] and row["color_age"] > 64
                       and row["source_mask"] >> 16 == 8 for row in summary["recent_frames"])):
        raise RuntimeError("completed actual720 cross-list fallback evidence required")
    if SOURCE.exists() or BUILD.exists():
        raise RuntimeError("fresh stage required")
    cpp_name = "src/nr_sync_probe.cpp"
    cpp = original[cpp_name].decode("utf-8").replace("\r\n", "\n")
    start = cpp.index("    bool cross_list=false;\n")
    end = cpp.index("    if(acquired!=ProofAcquire::acquired) return;", start)
    old = cpp[start:end]
    if "own->rejected=ColorProofReason::cross_list" not in old:
        raise RuntimeError("expected the observed whole-list cross rejection")
    cpp = replace_once(cpp, old, '''    // A same-resource declaration on another recording list makes its old
    // resource observation stale. It does not reject the list recording the
    // new declaration: CPU recording order is not GPU execution order.
    for(auto& list:lists_) if(list.generation && !list.closed) for(auto& c:list.colors) {
        if(!c.identity.cookie) continue;
        if(c.object==resource && (acquired!=ProofAcquire::acquired || c.identity.cookie!=identity.cookie))
            list.rejected=ColorProofReason::identity;
        if(acquired==ProofAcquire::acquired && c.identity.cookie==identity.cookie && &list!=own &&
                c.rejected==ColorProofReason::ready)
            c.rejected=ColorProofReason::cross_list;
    }
''')
    cpp = replace_once(cpp, '''    if(!plain) color->rejected=ColorProofReason::partial_split; // sticky until successful Reset
    else if(color->transition.seen && color->rejected==ColorProofReason::ready &&
''', '''    // Only a full local transition can renew an externally stale observation.
    // Unknown/identity/capacity/partial/split/contradictory-chain failures retain
    // their existing sticky rejection and cannot be cleared by another list.
    const bool renewed=plain && color->rejected==ColorProofReason::cross_list;
    if(renewed) color->rejected=ColorProofReason::ready;
    if(!plain) color->rejected=ColorProofReason::partial_split; // sticky until successful Reset
    else if(!renewed && color->transition.seen && color->rejected==ColorProofReason::ready &&
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
        rejected(f.observe(p),Reason::cross_list);
        // A fresh list's own explicit state was recorded, not rejected because
        // the old list also holds this resource. Its declared UAV remains unsafe.
        if(variant==0) rejected(f.observe(p,1),Reason::not_psr);
        if(variant==2) {
            f.barrier(p,0,0,psr,uav);rejected(f.observe(p),Reason::not_psr);
        }
        f.barrier(p,0,0,uav,psr);CHECK(allowed(f.observe(p)));
        const auto refreshed=f.observe(p);
        CHECK(refreshed.color_state.reason==Reason::ready);
        if(variant==0) {
            rejected(f.observe(p,1),Reason::cross_list);
            f.barrier(p,1,0,psr,copy_dest);rejected(f.observe(p,1),Reason::not_psr);
            f.barrier(p,1,0,copy_dest,psr);CHECK(allowed(f.observe(p,1)));
            rejected(f.observe(p),Reason::cross_list);
        }
    }
}
void cross_list_is_resource_local_and_can_extend_age() {
    Fixture f;nrb::SyncProbe p(f.ids.ops(),true);f.begin(p);
    f.barrier(p,0,1);p.reset(f.list(1),true);f.barrier(p,1,0,psr,uav);
    rejected(f.observe(p),Reason::cross_list);
    CHECK(allowed(f.observe(p,0,1))); // unrelated color in same list stays usable
    f.barrier(p,0,0,uav,psr);auto s=f.observe(p);
    s.evaluate_sequence=s.color.sequence+65;
    CHECK(!nrb::legacy_dlss_color_source(s));CHECK(allowed(s));
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
        rejected(f.observe(p),reason); // cross event cannot replace permanent reason
        f.barrier(p,0,0,psr,uav);rejected(f.observe(p),reason);
        f.barrier(p,0,0,uav,psr);rejected(f.observe(p),reason);
    }
}
''')
    test = replace_once(test,
        '        {"same_resource_other_list_or_uav_reject",same_resource_cross_list},',
        '        {"same_resource_external_state_requires_fresh_local_transition",same_resource_cross_list},\n'
        '        {"cross_list_resource_local_and_strict_age_extension",cross_list_is_resource_local_and_can_extend_age},\n'
        '        {"cross_list_cannot_erase_partial_or_bad_chain",cross_list_cannot_erase_permanent_rejections},')
    changed = {cpp_name: cpp.encode("utf-8"), test_name: test.encode("utf-8")}
    SOURCE.mkdir(parents=True)
    for name, data in original.items():
        path = SOURCE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(changed.get(name, data))
    BUILD.mkdir(parents=True)
    receipt = {"status": "prepared_cpu_only", "source_root": str(SOURCE),
        "source_files": {name: sha((SOURCE / name).read_bytes()) for name in original},
        "changed_files": sorted(changed), "parent_build_receipt_sha256": sha(parent_path.read_bytes()),
        "completed_live_evidence_sha256": sha(evidence_path.read_bytes()),
        "runtime_scope": "legacy-direct-reviewed-v1", "GPU_executed": False,
        "source_contract": "Recorded-state observation renewed only by explicit full local transition; previous observation invalidated per resource, not permanent rejection of the new declaring list.",
        "CPU_cases": 17,
        "official_basis": [
            "https://learn.microsoft.com/en-us/windows/win32/direct3d12/porting-from-direct3d-11-to-direct3d-12",
            "https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-resourcebarrier"],
        "limitation": "Scoped recorded legacy access state only; established same-frame direct resource/copy/queue contract remains required; this is not an execution-timeline or physical-alias audit.",
        "installation": "not installed"}
    (BUILD / "source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "changed_files": sorted(changed)}))


if __name__ == "__main__":
    main()
