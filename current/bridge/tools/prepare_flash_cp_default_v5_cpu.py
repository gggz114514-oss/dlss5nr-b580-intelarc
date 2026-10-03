"""Stage the reviewed local observer for normal Cyberpunk launches; no install."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PERF = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001")
BASE = PROJECT / "artifacts/periodic-flash-local-decl-v4-source-20261001"
SOURCE = PROJECT / "artifacts/periodic-flash-cp-default-v5-source-20261001"
BUILD = PERF / "periodic-flash-cp-default-v5-native-build"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError("expected one exact edit: " + before[:100])
    return text.replace(before, after, 1)


def main():
    parent_path = PERF / "periodic-flash-local-decl-v4-native-build/build-receipt.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent["status"] != "cpu_build_passed":
        raise RuntimeError("completed parent build required")
    original = {}
    for name, digest in parent["source_files"].items():
        data = (BASE / name).read_bytes()
        if sha(data) != digest:
            raise RuntimeError("parent changed: " + name)
        original[name] = data
    if SOURCE.exists() or BUILD.exists():
        raise RuntimeError("fresh stage required")
    header_name = "include/nr_sync_probe.h"
    header = original[header_name].decode("utf-8").replace("\r\n", "\n")
    header = replace_once(header, "inline bool dlss_color_candidate(const D3D12_RESOURCE_DESC& d) noexcept {", '''// Only the reviewed Cyberpunk profile is enabled on an ordinary launch.
// An explicit environment selection takes precedence, including disable/unknown.
enum class ColorStateScopeMode {disabled, reviewed, shadow};
inline bool dlss_color_default_profile(std::wstring_view executable) noexcept {
    const auto separator=executable.find_last_of(L"\\\\/");
    const auto name=separator==std::wstring_view::npos?executable:executable.substr(separator+1);
    constexpr std::wstring_view expected=L"cyberpunk2077.exe";
    if(name.size()!=expected.size()) return false;
    for(size_t i=0;i<name.size();++i) {
        const auto ch=name[i]>=L'A' && name[i]<=L'Z'?name[i]+(L'a'-L'A'):name[i];
        if(ch!=expected[i]) return false;
    }
    return true;
}
inline ColorStateScopeMode dlss_color_select_scope(bool override_present,
        std::string_view override_value,std::wstring_view executable) noexcept {
    if(override_present) {
        if(override_value==kDlssColorStateScope) return ColorStateScopeMode::reviewed;
        if(dlss_color_state_shadow_scope(override_value)) return ColorStateScopeMode::shadow;
        return ColorStateScopeMode::disabled;
    }
    return dlss_color_default_profile(executable)?ColorStateScopeMode::reviewed:
        ColorStateScopeMode::disabled;
}
inline bool dlss_color_candidate(const D3D12_RESOURCE_DESC& d) noexcept {''')
    cpp_name = "src/asi.cpp"
    cpp = original[cpp_name].decode("utf-8").replace("\r\n", "\n")
    start = cpp.index("// Opt-in recorded legacy/direct scope only. This is not execution-state proof.")
    end = cpp.index("nrb::ProofAcquire acquire_color_identity", start)
    cpp = replace_once(cpp, cpp[start:end], '''// Recorded legacy/direct scope only; not GPU execution-state proof.
// Enable the exact reviewed Cyberpunk profile without a special launcher.
nrb::ColorStateScopeMode select_color_state_scope() noexcept {
    char value[64]{};
    SetLastError(ERROR_SUCCESS);
    const auto size=GetEnvironmentVariableA("NRB_DLSS_COLOR_STATE_PROOF_SCOPE",value,sizeof(value));
    const auto error=GetLastError();
    const bool override_present=size!=0 || error!=ERROR_ENVVAR_NOT_FOUND;
    if(size>=sizeof(value)) return nrb::ColorStateScopeMode::disabled;
    wchar_t executable[1024]{};
    const auto length=GetModuleFileNameW(nullptr,executable,1024);
    const std::wstring_view path=length>0 && length<1024?
        std::wstring_view(executable,length):std::wstring_view{};
    return nrb::dlss_color_select_scope(override_present,
        std::string_view(value,size),path);
}
const auto color_state_scope_mode=select_color_state_scope();
bool color_state_scope_enabled() noexcept {
    return color_state_scope_mode!=nrb::ColorStateScopeMode::disabled;
}
const bool color_state_shadow_mode=color_state_scope_mode==nrb::ColorStateScopeMode::shadow;
''')
    test_name = "tests/state_proof_cpu.cpp"
    test = original[test_name].decode("utf-8").replace("\r\n", "\n")
    test = replace_once(test, "void shadow_gate_contract() {", '''void normal_launch_profile_and_override() {
    using Mode=nrb::ColorStateScopeMode;
    constexpr auto cp=L"G:/epic/Cyberpunk2077/bin/x64/Cyberpunk2077.exe";
    CHECK(nrb::dlss_color_select_scope(false,"",cp)==Mode::reviewed);
    CHECK(nrb::dlss_color_select_scope(false,"",L"C:\\\\GAME\\\\CYBERPUNK2077.EXE")==Mode::reviewed);
    CHECK(nrb::dlss_color_select_scope(false,"",L"re8.exe")==Mode::disabled);
    CHECK(nrb::dlss_color_select_scope(false,"",L"")==Mode::disabled);
    CHECK(!nrb::dlss_color_default_profile(L"Cyberpunk2077.exe.bak"));
    CHECK(!nrb::dlss_color_default_profile(L"Cyberpunk2077.exe/re8.exe"));
    for(const auto value:{"", "0", "disabled", "unexpected", "legacy-direct-reviewed-v1 "})
        CHECK(nrb::dlss_color_select_scope(true,value,cp)==Mode::disabled);
    CHECK(nrb::dlss_color_select_scope(true,nrb::kDlssColorStateShadowScope,cp)==Mode::shadow);
    CHECK(nrb::dlss_color_select_scope(true,nrb::kDlssColorStateScope,cp)==Mode::reviewed);
    // Explicit reviewed opt-in for other executables retains the old opt-in contract.
    CHECK(nrb::dlss_color_select_scope(true,nrb::kDlssColorStateScope,L"re8.exe")==Mode::reviewed);
}
void shadow_gate_contract() {''')
    test = replace_once(test, '        {"shadow_retains_legacy_gate_and_reports_rejection",shadow_gate_contract},',
        '        {"normal_launch_exact_cp_profile_and_explicit_override",normal_launch_profile_and_override},\n'
        '        {"shadow_retains_legacy_gate_and_reports_rejection",shadow_gate_contract},')
    changed = {header_name: header.encode("utf-8"), cpp_name: cpp.encode("utf-8"),
               test_name: test.encode("utf-8")}
    SOURCE.mkdir(parents=True)
    for name, data in original.items():
        path = SOURCE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(changed.get(name, data))
    BUILD.mkdir(parents=True)
    receipt = {"status": "prepared_cpu_only", "source_root": str(SOURCE),
        "source_files": {name: sha((SOURCE / name).read_bytes()) for name in original},
        "changed_files": sorted(changed), "parent_build_receipt_sha256": sha(parent_path.read_bytes()),
        "runtime_scope": "Exact Cyberpunk2077.exe profile defaults to reviewed; explicit override wins",
        "CPU_cases": 18, "GPU_executed": False, "installation": "not installed",
        "source_contract": "Only startup scope selection changes. Local observer, NR math, queue/copy/fence/lifetime and other route gates byte-identical to v4.",
        "acceptance_pending": "Do not install or claim flash fixed before v4 live and user visual acceptance."}
    (BUILD / "source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "changed_files": sorted(changed)}))


if __name__ == "__main__":
    main()
