"""CPU plan/check for the accepted serial GPU bridge source integration.

Never load a DLL/Torch, run a compiler/GPU/API, write G:, or modify frozen stage.
Only an explicit future 'promote' of a SHA-reviewed ready plan can write the
25 allowlisted canonical E: targets. All backup/journal outputs are on D:.
"""
import argparse
import base64
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import uuid

sys.dont_write_bytecode = True
PROJECT = Path(r"E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt")
SHARED = PROJECT.parent / ".codex-worktrees/re8-fp8-unround-fast-20260928"
SERIAL = PROJECT / "artifacts/gpu-handoff-serial-owner-v2-20261002/payload"
OWNER = PROJECT / "artifacts/gpu-handoff-owner-transfer-v1-20261002/payload"
LAB = Path(r"D:/Codex-NR-Experiments/cyberpunk-opt")
SERIAL_D = LAB / "gpu-handoff-serial-owner-v2-20261002"
OWNER_D = LAB / "gpu-handoff-owner-transfer-v1-20261002"
LIVE = LAB / "gpu-handoff-live-pair-v1-20261002"
OUTPUT = SERIAL_D / "source-integration-v2"
DOC = PROJECT / "docs/SERIAL_GPU_BRIDGE_SOURCE_INTEGRATION_20261002.md"
HOST_LABEL = "shared/game/nr_game_pre_xess_host.py"

# These are immutable historical reviews, not a new GPU/Present measurement.
PROOFS = {
    "install": (SERIAL_D / "install-01/installed.json", "ac6ec3000e8913a2208539c24690b2871b1129d7cae92f49a8fb4ee7dc4ddc9e"),
    "serial_build": (SERIAL_D / "cpu-build-03/CPU_BUILD_RECEIPT.json", "2f5b8e21b07fa829a3a7bd487c0e26f77bd27c855da996139c5d1a1586bbcdbd"),
    "serial_cpu": (SERIAL_D / "cpu-sidecar-02/MAIN_CPU_REVIEW.json", "12a5f31f76b59ae22c98df549680f1303fa1644460299787db17c9e842917372"),
    "serial_sidecar": (SERIAL_D / "cpu-sidecar-02/CPU_SIDECAR_RECEIPT.json", "cfbb163baa01a2df151eb389dd4b5c2623eef8f8cbb0e8fd3e550184948383f6"),
    "owner_build": (OWNER_D / "cpu-build-03c41unu/CPU_BUILD_RECEIPT.json", "bf4a866255cd81c17150c6589362fe6e35bd013bb198281dd0c90ac43601a512"),
    "owner_runtime": (OWNER_D / "runtime-2thread-02/MAIN_RUNTIME_REVIEW.json", "4337fb39d13cab657dcf807f1c6f711264430a3ba36e95d10b0ded21bbb4145c"),
    "run05": (LIVE / "serial-driver-run05/MAIN_PROTOCOL_REVIEW.json", "61e8fc25bb6d3b736b93ef8d522c5cb461a42d8c6cfb0d77ba3a0aec9d3ca0ec"),
    "run05_result": (LIVE / "serial-driver-run05/RESULT.json", "a9d4c36105b3265c7003962eb644d9ed2f6a3f0f2356c914ab526f13a452c83f"),
    "combined": (LIVE / "combined-serial-run01/MAIN_PROTOCOL_REVIEW.json", "c499ef571b810fc12f50fd782263aaf0b0e13b1be45f850fc3e91d33d8bd8dbf"),
    "raw": (LIVE / "combined-serial-run01/MAIN_COMBINED_RAW_REVIEW.json", "6467c6cd856935415a8c558adaeb8e485660ac5dce85b402a6f9501294791bdc"),
    "combined_result": (LIVE / "combined-serial-run01/RESULT.json", "6ffb43186f111427fe00af5d0d0b00f3dbb352c80a89de66e4c80d025c548bf4"),
    "human": (LIVE / "combined-serial-run01/HUMAN_ACCEPTANCE.json", "ae51b2a39b2c940707fe2ddbb193ce86364a2bdc84506697a75f2f92d04d855e"),
    "restored_on": (LIVE / "combined-serial-run01/ON_FOR_USER_REVIEW_03.json", "523eb8456398379d099394177d7b71346bb5cddba2c668894eac370e337b5a27"),
}
OLD_HOST = LAB / "gpu-handoff-product-trial-v1-20261002/20261002T032023010926Z/4-nr_game_pre_xess_host.py"
OLD_HOST_SHA = "d613d1ba88dd5c60833e48a2bc00778b9a274f775459319d46df4c548f0a7100"

MAPPING = (
    ("project/CMakeLists.txt","serial","CMakeLists.txt"),
    ("project/game/cyberpunk_nr_adapter.py","owner","plugins/cyberpunk_nr_adapter.py"),
    ("project/game/cyberpunk_nr_web.py","serial","game/cyberpunk_nr_web.py"),
    ("shared/game/nr_game_controls.py","serial","game/nr_game_controls.py"),
    ("shared/game/nr_game_pre_xess_host.py","serial","game/nr_game_pre_xess_host.py"),
    ("shared/game/nr_gpu_handoff_host_v1.py","serial","game/nr_gpu_handoff_host_v1.py"),
    ("project/include/nr_gpu_handoff_api.h","serial","include/nr_gpu_handoff_api.h"),
    ("project/include/nr_hdr_resource_pool_api.h","serial","include/nr_hdr_resource_pool_api.h"),
    ("project/src/asi.cpp","serial","src/asi.cpp"),
    ("project/src/deferred_identity.cpp","serial","src/deferred_identity.cpp"),
    ("project/src/nr_gpu_handoff.cpp","serial","src/nr_gpu_handoff.cpp"),
    ("project/src/nr_gpu_handoff.h","serial","src/nr_gpu_handoff.h"),
    ("project/src/nr_gpu_handoff_reuse.h","serial","src/nr_gpu_handoff_reuse.h"),
    ("project/src/nr_hdr_proxy.cpp","serial","src/nr_hdr_proxy.cpp"),
    ("project/src/nr_hdr_proxy.h","serial","src/nr_hdr_proxy.h"),
    ("project/src/nr_hdr_resource_pool.h","serial","src/nr_hdr_resource_pool.h"),
    ("project/tests/gpu_handoff_cpu.cpp","serial","tests/gpu_handoff_cpu.cpp"),
    ("project/tests/gpu_handoff_host_cpu.py","serial","tests/gpu_handoff_host_cpu.py"),
    ("project/tests/hdr_resource_pool_controls_cpu.cpp","serial","tests/hdr_resource_pool_controls_cpu.cpp"),
    ("project/tests/hdr_resource_pool_cpu.cpp","serial","tests/hdr_resource_pool_cpu.cpp"),
    ("project/tests/hdr_resource_pool_gpu.cpp","serial","tests/hdr_resource_pool_gpu.cpp"),
    ("shared/game/nr_texture_bridge_v1.py","owner","game/nr_texture_bridge_v1.py"),
    ("shared/native/nr_gpu_handoff_lease.h","owner","native/nr_gpu_handoff_lease.h"),
    ("shared/native/nr_texture_bridge_re8_v1.cpp","owner","native/nr_texture_bridge_re8_v1.cpp"),
    ("shared/native/nr_texture_bridge_v1.h","owner","native/nr_texture_bridge_v1.h"),
)
EXPECTED_BEFORE = {
    "project/CMakeLists.txt": "ad3119ec1d923f0801466bfda0389f9ab1336d27c719a34535b14d118dd951f1",
    "project/game/cyberpunk_nr_adapter.py": "e05c5749ff5c0978874ea3f2987469c70c0d3859650ac90012de2369f5209938",
    "project/game/cyberpunk_nr_web.py": "8589fbb00318c7300f7edd5205de2647345d10356c841ed45024256fb6729093",
    "shared/game/nr_game_controls.py": "8e4a75d295eff4a184df539bda6c96f79415df275bc89c562deb0b25add38400",
    "shared/game/nr_game_pre_xess_host.py": "1fec385b544fe42418c2a20b08970589c5fec46b1be5e1c17c2d471c2cdaefbd",
    "shared/game/nr_gpu_handoff_host_v1.py": None,
    "project/include/nr_gpu_handoff_api.h": None,
    "project/include/nr_hdr_resource_pool_api.h": None,
    "project/src/asi.cpp": "1db0a22f54a46a80ff020415e5f6cadd72f1e675dc039ffa04ae6ecbca2011b8",
    "project/src/deferred_identity.cpp": "d425b66fa31e08129fa92b8da19b54a14a33af62bda641e560a5724a651a0b2e",
    "project/src/nr_gpu_handoff.cpp": None,
    "project/src/nr_gpu_handoff.h": None,
    "project/src/nr_gpu_handoff_reuse.h": None,
    "project/src/nr_hdr_proxy.cpp": "397303b0ca8739b90d5c5aa3b5894bbc44975ea9ab9f517e40c11fd657a2c8eb",
    "project/src/nr_hdr_proxy.h": "5f6a3cbc97adbef488c341e0a9485721d4994114ec1fa7f81ba4c5acec86e29e",
    "project/src/nr_hdr_resource_pool.h": None,
    "project/tests/gpu_handoff_cpu.cpp": None,
    "project/tests/gpu_handoff_host_cpu.py": None,
    "project/tests/hdr_resource_pool_controls_cpu.cpp": None,
    "project/tests/hdr_resource_pool_cpu.cpp": None,
    "project/tests/hdr_resource_pool_gpu.cpp": None,
    "shared/game/nr_texture_bridge_v1.py": "01bb3bc05594bfcdf769177595b9d7854136419996a1216abf8dfecddf004662",
    "shared/native/nr_gpu_handoff_lease.h": None,
    "shared/native/nr_texture_bridge_re8_v1.cpp": "e8bf06e5c6e37da4ab94c030ba8d8829bb8ab4152ca8b7a60b14f314f0cbc375",
    "shared/native/nr_texture_bridge_v1.h": "fe66075ec1e32943fe58aad32021bc1d45471f49135ffcc2303083027c182076",
}


# Four uniquely anchored GPU-interface changes from the actual tested host.
# No periodic-flash block, model call, reset/history/control math is copied.
HOST_PATCHES = (
    ("            from nr_texture_bridge_v1 import TextureBridge\n"
     "            _bridge.close()",
     "            from nr_texture_bridge_v1 import TextureBridge\n"
     "            from nr_gpu_handoff_host_v1 import ensure_idle_before_reinitialize\n"
     "            ensure_idle_before_reinitialize(_bridge)\n"
     "            _bridge.close()"),
    ("        from nr_texture_bridge_v1 import SourceFrame",
     "        from nr_texture_bridge_v1 import SourceFrame\n"
     "        from nr_gpu_handoff_host_v1 import source_handoff\n"
     "        producer_point = source_handoff(_bridge, device_ptr, queue_ptr, color_ptr,\n"
     "                                        motion_ptr, frame_id)"),
    ("                            frame_id - 1, reset)",
     "                            frame_id - 1, reset, **producer_point)"),
    ("        return output.resource, output.fence, output.value\n    except BaseException as exc:",
     "        return output.resource, output.fence, output.value\n    except BaseException as exc:\n"
     "        from nr_gpu_handoff_host_v1 import invalidate_after_failure\n"
     "        invalidate_after_failure()"),
)


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def physical(path):
    """Reject every redirect before resolve/open, including missing-file parents."""
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(info.st_mode) and
                not (getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT),
                "Redirect/junction forbidden: " + str(part))
    return path


def readable(path):
    path = Path(os.path.abspath(path))
    require(any(path.is_relative_to(root) for root in (PROJECT, SHARED, LAB)),
            "Read outside pinned E:/D: source/evidence roots: " + str(path))
    path = physical(path)
    require(path.is_file(), "Missing regular source/evidence: " + str(path))
    return path.read_bytes()


def pinned(path, expected):
    data = readable(path)
    require(sha(data) == expected, "Frozen pin changed: " + str(path))
    return data


def current(path):
    path = Path(os.path.abspath(path))
    require(any(path.is_relative_to(root) for root in (PROJECT, SHARED)),
            "Canonical read outside E: allowlist roots: " + str(path))
    path = physical(path)
    if not path.exists():
        return None
    require(path.is_file(), "Canonical target is not a regular file: " + str(path))
    require(path.stat().st_nlink == 1, "Canonical hardlink forbidden: " + str(path))
    return path.read_bytes()


def digest(data):
    return None if data is None else sha(data)


def target_for(label):
    require(label in EXPECTED_BEFORE, "Unrelated target refused: " + label)
    domain, relative = label.split("/", 1)
    require(".." not in Path(relative).parts, "Target traversal refused")
    root = PROJECT if domain == "project" else SHARED
    target = physical(root / relative)
    require(target.is_relative_to(root), "Target escaped E: canonical root")
    require("artifacts" not in target.relative_to(root).parts,
            "Frozen artifacts cannot be a promote target")
    return target


def nl(data):
    return data.replace(b"\r\n", b"\n")


def replace_lines(data, old, new):
    lines = data.decode("utf-8").splitlines(keepends=True)
    plain = [line.rstrip("\r\n") for line in lines]
    old_lines = old.split("\n")
    hits = [i for i in range(len(lines)-len(old_lines)+1)
            if plain[i:i+len(old_lines)] == old_lines]
    require(len(hits) == 1, "GPU host anchor not unique: " + old_lines[0])
    start = hits[0]
    ending = lines[start][len(plain[start]):] or "\n"
    return ("".join(lines[:start]) +
            "".join(line + ending for line in new.split("\n")) +
            "".join(lines[start+len(old_lines):])).encode("utf-8")


def patch_host(data, reverse=False):
    patches = tuple(reversed(HOST_PATCHES)) if reverse else HOST_PATCHES
    for before, after in patches:
        data = replace_lines(data, after if reverse else before, before if reverse else after)
    return data


def verify_evidence():
    evidence = {name: json.loads(pinned(path, pin).decode("utf-8-sig"))
                for name, (path, pin) in PROOFS.items()}
    b, cpu, side = (evidence[name] for name in ("serial_build", "serial_cpu", "serial_sidecar"))
    ob, runtime = evidence["owner_build"], evidence["owner_runtime"]
    require(b["status"] == "serial_handoff_native_CPU_build_passed_GPU_UNTESTED" and
            cpu["status"] == "serial_gpu_handoff_CPU_reviewed" and
            side["status"] == "serial_sidecar_CPP_and_host_CPU_passed", "CPU reviews not accepted")
    require(len(b["source_pins"]) == 69 and cpu["CPP_completion"]["checks"] == 540 and
            cpu["CPP_completion"]["persistent_host_threads"] == 2 and cpu["host_tests"] == 23 and
            cpu["old_strict_arm_passed"] and side["real_gate_two_CPU_threads_passed"] and
            cpu["CPU_build_receipt_sha256"] == PROOFS["serial_build"][1] and
            cpu["CPU_sidecar_receipt_sha256"] == PROOFS["serial_sidecar"][1], "CPU chain mismatch")
    for path, pin in b["source_pins"].items():
        pinned(Path(path), pin)
        require(cpu["review_pins"][path] == pin and side["source_pins"][path] == pin,
                "Source build/review/sidecar disagree: " + path)
    for path, pin in ob["source_pins"].items():
        pinned(Path(path), pin)
        require(runtime["verified_hashes"][path] == pin, "Owner runtime/source chain mismatch")
    require(ob["old_exports_preserved"] and ob["original_data_ABI_and_HLSL_unchanged"] and
            runtime["status"] == "runtime_owner_transfer_reviewed" and
            runtime["byte_checks_passed"] and runtime["two_actual_threads_observed"] and
            runtime["normal_exit"] and runtime["busy_rejections"] == 15 and runtime["migrations"] == 4,
            "Owner protocol not accepted")
    for field in ("runtime_receipt", "child_result"):
        pinned(Path(runtime[field]), runtime[field+"_sha256"])
    for build, field in ((b, "candidate_ASI"), (ob, "candidate_DLL")):
        pinned(Path(build[field]), build["candidate_sha256"])
    require(cpu["candidate_ASI_sha256"] == b["candidate_sha256"] and
            runtime["candidate_DLL_sha256"] == ob["candidate_sha256"], "Binary chain mismatch")

    run, combined, raw = evidence["run05"], evidence["combined"], evidence["raw"]
    require(run["status"] == "serial_handoff_live_protocol_reviewed" and run["source_pin_count"] == 83 and
            run["unique_timed_frames"] == 270 and run["serial_actual_capability_held"] and
            run["continuity_seconds"] >= 90 and run["continuity_new_completed_frames"] == 1559,
            "Run05 actual serial protocol proof missing")
    require(combined["status"] == "combined_serial_live_protocol_reviewed" and
            combined["source_pin_count"] == 86 and combined["unique_timed_frames"] == 120 and
            raw["status"] == "main_raw_statistics_reviewed", "Combined protocol/raw review missing")
    for review in (run, combined):
        for path, pin in review["evidence_sha256"].items():
            pinned(Path(path), pin)
    for arm in raw["arms"]:
        pinned(Path(arm["input"]), arm["sha256"])
    for metric, saving in (("record_last_ms", 1.12383), ("record_to_retire_span_last_ms", 2.97105)):
        pair = raw["paired"][metric]
        require(pair["both_on_below_both_off"] and abs(pair["saved_ms"]-saving) < 1e-8,
                "Paired raw gain changed: " + metric)

    install = evidence["install"]
    require(install["status"] == "installed_trial_default_off" and len(install["files"]) == 9 and
            install["GPU_handoff_default_requested"] is False and
            install["resource_pool_default_enabled"] is False, "Nine-file OFF-default trial not pinned")
    for result_name, before, after, count in (
            ("run05_result", "source_hashes_before", "source_hashes_after", 83),
            ("combined_result", "source_hashes_before", "source_hashes_after", 86),
            ("restored_on", "source_pins_before", "source_pins_after", 83)):
        result = evidence[result_name]
        require(len(result[before]) == count and result[before] == result[after],
                "Historical pin continuity failed: " + result_name)
        for installed in install["files"]:
            # G: paths here are only JSON identity keys. Never open/stat G:.
            require(result[before][installed["target"]] == installed["new_sha256"],
                    "Installed nine bytes disagree with actual run: " + installed["target"])
            pinned(Path(installed["source"]), installed["new_sha256"])

    human, on = evidence["human"], evidence["restored_on"]
    require(human["status"] == "human_visual_stability_accepted" and
            human["visual_stability_accepted"] is True and human["keep_actual_combined_mode"] is True and
            human["agreed_frame_generation_for_session"] == "off" and
            human["startup_default_changed"] is False and
            human["automatic_Present_trace_recorded"] is False and
            human["RTSS_samples_paired_with_API_frames"] is False and
            human["attribute_all_RTSS_improvement_to_this_pair"] is False,
            "Human acceptance scope changed")
    require(on["status"] == "healthy_actual_combined_on" and on["source_pins_unchanged"] and
            on["first_recording_frame_id"] == 35224 and on["final_recording_frame_id"] == 35234 and
            on["counter_deltas"]["bridge_resource_pool"]["hits"] == 10 and
            on["counter_deltas"]["gpu_handoff"]["prepared_bypasses"] == 10, "Actual combined ON restoration missing")
    # Prove the four hook edits are exactly the candidate's changes to its own
    # installed parent. LF normalization is comparison-only, never a source write.
    old = pinned(OLD_HOST, OLD_HOST_SHA)
    tested = readable(SERIAL / "game/nr_game_pre_xess_host.py")
    require(nl(patch_host(old)) == nl(tested), "Projected GPU hooks differ from tested candidate hooks")
    return evidence


def cmake_custom_commands(cmake):
    """Inspect this SHA-pinned recipe; refuse an unreviewed generator shape."""
    calls = re.findall(r"add_custom_command\s*\((.*?)\)", cmake, re.DOTALL)
    require(len(calls) == 1, "CMake custom-command closure changed")
    tokens = re.findall(r'"[^"]*"|[^\s]+', calls[0])
    tokens = [token.strip('"') for token in tokens]
    expected = ["TARGET", "CyberpunkNRBridge", "POST_BUILD"]
    copies = []
    for name in ("cyberpunk_nr_adapter.py", "cyberpunk_nr_web.py"):
        source = "${CMAKE_CURRENT_SOURCE_DIR}/game/" + name
        output = "$<TARGET_FILE_DIR:CyberpunkNRBridge>/" + name
        expected.extend(["COMMAND", "${CMAKE_COMMAND}", "-E", "copy_if_different", source, output])
        copies.append({"authored_input": "game/" + name, "build_copy_output": output})
    require(tokens == expected, "Unreviewed CMake OUTPUT/DEPENDS or POST_BUILD inputs")
    return {"kind": "TARGET_POST_BUILD_copy", "target": "CyberpunkNRBridge",
            "OUTPUT_declarations": [], "DEPENDS_declarations": [],
            "generated_header_outputs": [], "copies": copies}


def dependency_closure(proposed, build):
    cmake = proposed["project/CMakeLists.txt"].decode("utf-8")
    local_cmake = cmake.replace("${CMAKE_CURRENT_SOURCE_DIR}/", "")
    references = sorted(set(re.findall(
        r"(?<![/\w.-])(?:src|include|tests|shaders|game)/[\w/.-]+\.(?:cpp|hlsl|py|h)\b",
        local_cmake)))
    custom = cmake_custom_commands(cmake)
    checked = {}
    # All non-promoted project inputs remain exactly the tested candidate.
    for source, pin in build["source_pins"].items():
        source_path = Path(source)
        relative = source_path.relative_to(SERIAL).as_posix()
        if relative in ("game/nr_game_controls.py", "game/nr_game_pre_xess_host.py", "game/nr_gpu_handoff_host_v1.py"):
            label = "shared/" + relative
        else:
            label = "project/" + relative
        if label in proposed:
            continue
        target = physical(PROJECT / relative)
        actual = digest(current(target))
        require(actual == pin, "Unmapped/unrelated canonical dependency differs: " + str(target) +
                " expected=" + pin + " actual=" + str(actual))
        checked[str(target)] = pin
    for relative in references:
        label = "project/" + relative
        require(str(SERIAL / relative) in build["source_pins"],
                "Unpinned authored CMake dependency: " + relative)
        require(label in proposed or current(PROJECT / relative) is not None,
                "Missing CMake dependency: " + relative)
    require("EXCLUDE_FROM_ALL tests/hdr_resource_pool_gpu.cpp" in
            proposed["project/CMakeLists.txt"].decode("utf-8"), "GPU probe must remain excluded")
    require(sha(readable(PROJECT / "shaders/nr_hdr_proxy.hlsl")) ==
            build["source_pins"][str(SERIAL / "shaders/nr_hdr_proxy.hlsl")], "Shader contract changed")
    require("shaders/nr_hdr_proxy.hlsl" in references and "shaders/nr_hdr_proxy.h" not in references,
            "Authored HLSL path was truncated to a nonexistent header")
    return {"CMake_references": references, "unchanged_canonical_pins": checked,
            "CMake_sha256": sha(proposed["project/CMakeLists.txt"]),
            "custom_commands": [custom],
            "authored_shader_inputs": {"shaders/nr_hdr_proxy.hlsl":
                build["source_pins"][str(SERIAL / "shaders/nr_hdr_proxy.hlsl")]},
            "authored_header_inputs": {"src/nr_hdr_proxy.h":
                build["source_pins"][str(SERIAL / "src/nr_hdr_proxy.h")]},
            "optional_existing_external_test_condition":
                "${CMAKE_CURRENT_SOURCE_DIR}/../re8-b580-nr-xess/game/nr_game_controls.py",
            "external_SDK_requirements_preserved": [
                "OptiScaler commit 7534ad00bf9e590eedb99e8dd9fd8c89dae3654f",
                "${OPTISCALER_SOURCE}/external/nvngx_dlss_sdk/nvsdk_ngx.h",
                "${OPTISCALER_SOURCE}/OptiScaler/library/detours/detours.lib"],
            "generated_build_outputs_promoted": False}


def make_plan(policy):
    ev = verify_evidence()
    rows, proposed, snapshots, conflicts = [], {}, {}, []
    install = ev["install"]["files"]
    for label, origin, relative in MAPPING:
        source = (SERIAL if origin == "serial" else OWNER) / relative
        build_name = "serial_build" if origin == "serial" else "owner_build"
        source_pin = ev[build_name]["source_pins"][str(source)]
        data = pinned(source, source_pin)
        target = target_for(label)
        before = current(target)
        mode = "exact_tested_source"
        if label == HOST_LABEL:
            mode = "four_tested_GPU_hooks_only"
            if digest(before) != EXPECTED_BEFORE[label]:
                raise RuntimeError("Canonical host baseline changed: " + str(target) +
                                   " expected=" + EXPECTED_BEFORE[label] + " actual=" + str(digest(before)))
            data = patch_host(before)
            require(patch_host(data, reverse=True) == before, "Projection changed unrelated host bytes")
            require(b"periodic_flash_snapshot_v2" not in data,
                    "Unrelated diagnostics entered minimal host projection")
            if policy != "gpu-hooks-only":
                conflicts.append({"label": label, "reason": "main_host_projection_decision_required",
                                  "candidate_unrelated_change": "periodic-flash diagnostic blocks",
                                  "safe_choice": "--pre-xess-policy gpu-hooks-only"})
        if digest(before) not in (EXPECTED_BEFORE[label], sha(data)):
            conflicts.append({"label": label, "reason": "canonical_unrelated_change",
                              "expected": EXPECTED_BEFORE[label], "actual": digest(before)})
        if target.suffix == ".py":
            compile(data, str(target), "exec", dont_inherit=True)
        proposed[label], snapshots[label] = data, before
        rows.append({"label": label, "source": str(source), "target": str(target), "mode": mode,
                     "candidate_source_sha256": source_pin, "built_source_sha256": source_pin,
                     "build_receipt": str(PROOFS[build_name][0]), "build_receipt_sha256": PROOFS[build_name][1],
                     "expected_before_sha256": EXPECTED_BEFORE[label], "actual_before_sha256": digest(before),
                     "proposed_after_sha256": sha(data), "changed": before != data,
                     "installed_targets": [r["target"] for r in install if r["new_sha256"] == source_pin]})
    require(len(proposed) == len(MAPPING), "Host baseline conflict; no complete promotable plan")
    closure = dependency_closure(proposed, ev["serial_build"])
    compile(readable(Path(__file__)), str(__file__), "exec", dont_inherit=True)
    protected = closure["unchanged_canonical_pins"]
    original = snapshots[HOST_LABEL]
    projected = proposed[HOST_LABEL]
    parent = pinned(OLD_HOST, OLD_HOST_SHA)
    tested = readable(SERIAL / "game/nr_game_pre_xess_host.py")
    return {
        "schema": "serial-gpu-bridge-source-integration-v2",
        "status": "blocked_main_decision_or_conflict" if conflicts else "CPU_dry_check_ready",
        "pre_xess_policy": policy, "conflicts": conflicts, "mapping": rows,
        "projection_base64": base64.b64encode(proposed[HOST_LABEL]).decode("ascii"),
        "host_projection_proof": {
            "hook_count": len(HOST_PATCHES), "before_sha256": sha(original),
            "after_sha256": sha(projected),
            "reverse_to_original_bytes": patch_host(projected, reverse=True) == original,
            "reverse_sha256": sha(patch_host(projected, reverse=True)),
            "tested_parent_path": str(OLD_HOST), "tested_parent_sha256": OLD_HOST_SHA,
            "tested_candidate_sha256": sha(tested),
            "patch_parent_matches_tested_candidate_LF": nl(patch_host(parent)) == nl(tested),
            "copy_unrelated_candidate_diagnostics": False},
        "canonical_snapshot": {label: digest(before) for label, before in snapshots.items()},
        "protected_canonical_pins": protected, "dependency_closure": closure,
        "acceptance": {"human_visual_stability_accepted": True, "keep_actual_combined_mode": True,
                       "source_startup_defaults_OFF": True, "run05_pins": 83, "run05_timed_frames": 270,
                       "continuity_seconds": ev["run05"]["continuity_seconds"], "continuity_frames": 1559,
                       "combined_pins": 86, "combined_timed_frames": 120,
                       "record_saved_ms": ev["raw"]["paired"]["record_last_ms"]["saved_ms"],
                       "span_saved_ms": ev["raw"]["paired"]["record_to_retire_span_last_ms"]["saved_ms"],
                       "automatic_Present_pairing": False, "all_RTSS_gain_attributed": False,
                       "historical_GPU_bytes_relabelled": False,
                       "projected_host_is_full_tested_file": False},
        "proof_pins": {name: {"path": str(path), "sha256": pin} for name, (path, pin) in PROOFS.items()},
        "tool_sha256": sha(readable(Path(__file__))), "doc_sha256": sha(readable(DOC)),
        "Python_compiled_in_memory": [r["target"] for r in rows if r["target"].endswith(".py")] + [str(Path(__file__))],
        "CXX_compiled": False, "DLL_loaded": False, "GPU_executed": False, "API_called": False,
        "G_read_or_write": False, "canonical_E_writes": False,
    }, proposed, snapshots


def d_output(path):
    path = Path(os.path.abspath(path))
    require(path.is_relative_to(OUTPUT) and path != OUTPUT,
            "Outputs/backups must be a new child of " + str(OUTPUT))
    return physical(path)


def write_new(path, data):
    physical(path.parent).mkdir(parents=True, exist_ok=True)
    physical(path)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    require(path.read_bytes() == data, "D output readback mismatch: " + str(path))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def save_check(plan, proposed, snapshots, output):
    output = d_output(output)
    physical(output).mkdir(parents=True, exist_ok=False)
    diffs = []
    for row in plan["mapping"]:
        label = row["label"]
        before = (snapshots[label] or b"").decode("utf-8").splitlines(keepends=True)
        after = proposed[label].decode("utf-8").splitlines(keepends=True)
        diffs.extend(difflib.unified_diff(before, after, fromfile="before/"+label, tofile="proposed/"+label))
    diff_data = "".join(diffs).encode("utf-8")
    plan["diff_sha256"] = sha(diff_data)
    write_new(output / "DIFF.patch", diff_data)
    write_new(output / "PLAN.json", json_bytes(plan))
    return output / "PLAN.json"


def assert_plan(plan):
    require(plan["schema"] == "serial-gpu-bridge-source-integration-v2" and
            plan["status"] == "CPU_dry_check_ready" and plan["pre_xess_policy"] == "gpu-hooks-only" and
            not plan["conflicts"], "Only a ready, explicitly resolved CPU plan can promote")
    require(plan["tool_sha256"] == sha(readable(Path(__file__))) and
            plan["doc_sha256"] == sha(readable(DOC)), "Reviewed integration tool/doc changed")
    require(plan["proof_pins"] == {name: {"path": str(path), "sha256": pin}
                                  for name, (path, pin) in PROOFS.items()}, "Historical proof chain changed")
    require([row["label"] for row in plan["mapping"]] == [row[0] for row in MAPPING],
            "Promote target allowlist changed")
    require(plan["acceptance"]["source_startup_defaults_OFF"] and
            plan["acceptance"]["human_visual_stability_accepted"] and
            plan["acceptance"]["keep_actual_combined_mode"], "Acceptance gates changed")


def plan_payload(plan, evidence):
    output = {}
    for row, (label, origin, relative) in zip(plan["mapping"], MAPPING):
        target, source = target_for(label), (SERIAL if origin == "serial" else OWNER) / relative
        build = evidence["serial_build" if origin == "serial" else "owner_build"]
        expected = build["source_pins"][str(source)]
        require(row["target"] == str(target) and row["source"] == str(source) and
                row["expected_before_sha256"] == EXPECTED_BEFORE[label] and
                row["candidate_source_sha256"] == expected and row["built_source_sha256"] == expected,
                "Reviewed row identity changed: " + label)
        source_data = pinned(source, expected)
        data = base64.b64decode(plan["projection_base64"], validate=True) if label == HOST_LABEL else source_data
        require(sha(data) == row["proposed_after_sha256"], "Reviewed after bytes changed: " + label)
        before = current(target)
        if label == HOST_LABEL:
            require(digest(patch_host(data, reverse=True)) == EXPECTED_BEFORE[label],
                    "Projected host contains unrelated edits")
        require(digest(before) in (EXPECTED_BEFORE[label], sha(data)),
                "Canonical changed since CPU plan: " + label)
        if target.suffix == ".py":
            compile(data, str(target), "exec", dont_inherit=True)
        output[label] = (target, before, data)
    for path, pin in plan["protected_canonical_pins"].items():
        require(digest(current(Path(path))) == pin, "Protected canonical dependency changed: " + path)
    closure = dependency_closure({label: data for label, (_, _, data) in output.items()},
                                 evidence["serial_build"])
    require(plan["dependency_closure"] == closure and
            plan["protected_canonical_pins"] == closure["unchanged_canonical_pins"],
            "Reviewed authored/protected dependency closure changed")
    return output


def replace_canonical(target, data, expected):
    # Atomic replace avoids partial source writes. No recursive delete/move.
    target = physical(target)
    require(target in {target_for(label) for label, _, _ in MAPPING}, "E write outside allowlist")
    require(target.parent.is_dir(), "Canonical parent disappeared")
    temp = target.with_name("."+target.name+".serialgpu-"+uuid.uuid4().hex+".tmp")
    physical(temp)
    try:
        with temp.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        require(current(target) == expected, "Concurrent canonical edit before replace")
        os.replace(temp, target)
    finally:
        if temp.exists():
            physical(temp).unlink()


def promote(plan_file, expected_plan_sha, backup_root):
    plan_file = d_output(plan_file)
    plan = json.loads(pinned(plan_file, expected_plan_sha).decode("utf-8-sig"))
    assert_plan(plan)
    pinned(plan_file.with_name("DIFF.patch"), plan["diff_sha256"])
    evidence = verify_evidence()
    files = plan_payload(plan, evidence)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    backup = d_output(d_output(backup_root) / tag)
    backup.mkdir(parents=True, exist_ok=False)
    journal = {"status": "backed_up", "plan": str(plan_file), "plan_sha256": expected_plan_sha,
               "changes": [], "G_writes": False, "GPU_executed": False, "API_called": False}
    for label, (target, before, data) in files.items():
        if before == data:
            continue
        saved = backup / "before" / label
        if before is not None:
            write_new(saved, before)
        journal["changes"].append({"label": label, "target": str(target),
                                   "before_sha256": digest(before), "after_sha256": sha(data),
                                   "backup": str(saved) if before is not None else None})
    write_new(backup / "before.json", json_bytes(journal))
    written = []
    try:
        # Recheck every protected/before/source pin after all D backups exist.
        files = plan_payload(plan, verify_evidence())
        for row in journal["changes"]:
            label = row["label"]
            target, before, data = files[label]
            written.append((label, target, before, data))
            replace_canonical(target, data, before)
            require(current(target) == data, "Canonical readback mismatch: " + label)
        for label, (target, before, data) in files.items():
            require(current(target) == data, "Final canonical readback mismatch: " + label)
        for path, pin in plan["protected_canonical_pins"].items():
            require(digest(current(Path(path))) == pin, "Protected dependency changed after writes")
        journal.update(status="canonical_E_source_promoted", E_changes=len(written),
                       canonical_after_pins={label: sha(data) for label, (_, _, data) in files.items()})
        write_new(backup / "promoted.json", json_bytes(journal))
    except BaseException as exc:
        rollback_errors = []
        for label, target, before, data in reversed(written):
            try:
                now = current(target)
                if now == before:
                    continue
                require(now == data, "External canonical edit; preserve it and use D backup")
                if before is None:
                    require(target == target_for(label), "Rollback deletion escaped allowlist")
                    physical(target).unlink()
                else:
                    replace_canonical(target, before, data)
                require(current(target) == before, "Rollback readback mismatch")
            except BaseException as error:
                rollback_errors.append({"label": label, "error": str(error)})
        journal.update(status="rollback_incomplete" if rollback_errors else "rolled_back",
                       error=str(exc), rollback_errors=rollback_errors)
        write_new(backup / "rollback.json", json_bytes(journal))
        raise
    return {"status": journal["status"], "E_changes": len(written), "receipt": str(backup / "promoted.json"),
            "GPU_executed": False, "API_called": False, "G_writes": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "promote"))
    parser.add_argument("--pre-xess-policy", choices=("pending-main", "gpu-hooks-only"), default="pending-main")
    parser.add_argument("--output-dir", type=Path, help="Fresh D: check-plan directory; otherwise stdout only")
    parser.add_argument("--plan", type=Path, help="Future promote: reviewed D: PLAN.json")
    parser.add_argument("--plan-sha256", help="Future promote: exact SHA of reviewed PLAN.json")
    parser.add_argument("--backup-root", type=Path, default=OUTPUT / "promotions")
    args = parser.parse_args()
    try:
        require(Path(__file__).absolute() == PROJECT / "tools/promote_serial_gpu_bridge_source_v2_cpu.py",
                "Run the canonical integration tool path")
        if args.command == "promote":
            require(args.plan and args.plan_sha256 and not args.output_dir,
                    "Explicit promote requires --plan and --plan-sha256, not --output-dir")
            print(json.dumps(promote(args.plan, args.plan_sha256, args.backup_root), ensure_ascii=False))
            return 0
        require(not args.plan and not args.plan_sha256, "check does not consume a promote authorization")
        plan, proposed, snapshots = make_plan(args.pre_xess_policy)
        # CPU check must not alter any canonical file, including preserved inputs.
        require(all(digest(current(target_for(label))) == pin for label, pin in plan["canonical_snapshot"].items()),
                "Canonical changed during dry check")
        require(all(digest(current(Path(path))) == pin for path, pin in plan["protected_canonical_pins"].items()),
                "Protected dependency changed during dry check")
        location = save_check(plan, proposed, snapshots, args.output_dir) if args.output_dir else None
        result = {"status": plan["status"], "mapping_count": len(plan["mapping"]),
                  "planned_E_changes": sum(row["changed"] for row in plan["mapping"]),
                  "Python_in_memory_compiled": len(plan["Python_compiled_in_memory"]),
                  "conflicts": plan["conflicts"], "CXX_compiled": False,
                  "canonical_E_writes": False, "GPU_executed": False, "API_called": False, "G_read_or_write": False,
                  "acceptance": plan["acceptance"], "plan": str(location) if location else None,
                  "diff_sha256": plan.get("diff_sha256"),
                  "plan_sha256": sha(location.read_bytes()) if location else None}
        if location is None:
            result["mapping"] = plan["mapping"]
        print(json.dumps(result, ensure_ascii=False))
        return 2 if plan["conflicts"] else 0
    except Exception as exc:
        print(json.dumps({"status": "CPU_rejected", "error": str(exc), "GPU_executed": False,
                          "API_called": False, "G_read_or_write": False}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
