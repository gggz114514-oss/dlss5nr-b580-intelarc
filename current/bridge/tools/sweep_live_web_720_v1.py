"""Serial live-web screens; no GPU imports or runtime/source/cache writes.

The already-running game performs inference. All reported times are the web's
overlapping 32-frame average, not game FPS or independently sampled GPU time.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_live_web_720_v1 as base

PERF = base.OUT.parent
PLAN = PERF / "installed-sweep-plan.json"
CONTROL_KEYS = ("enabled", "display_strength", "style", "input_size", "history_mode",
                "graph_replay", "model_intensity", "local_tone", "local_structure",
                "auto_mask", "skin_structure", "backend_variant", "experiment_720")


def actual_hits(st, selected, reg):
    h = st.get("health") or {}
    options = reg.combined_mode_options(list(selected))
    need = {"c512_library_720_active", "native_k8_720_active"}
    for key in selected:
        need.update(reg.PROFILES[key].hits)
    missing = [key for key in sorted(need) if h.get(key) is not True]
    if h.get("structure_combo_frame_route") != "replay":
        missing.append("structure_combo_frame_route!=replay")
    numeric = options.get("numeric_cleanup_720", {})
    expected = set()
    for key in numeric:
        if key == "decoder_input_full_k": expected.add("decoder_input")
        elif key == "branch_accum_families": expected.add("branch_accum")
        elif key.startswith("vit_"): expected.add("vit")
        elif key.startswith("history_"): expected.add("history")
        elif key.startswith("post_"): expected.add("post")
        elif key == "front_noise": expected.add("front")
        else: raise RuntimeError("Unknown numeric child mapping: " + key)
    sites = h.get("numeric_cleanup_sites") or {}
    conditional = (numeric.get("history_value") == "fp32_fractional" and
                   "history" in sites and sites["history"] is False)
    if expected != set(sites):
        missing.append("numeric child set differs from selected implementations")
    for site in expected:
        if site == "history" and conditional: continue
        if sites.get(site) is not True: missing.append("numeric_cleanup_sites." + site)
    if numeric and not conditional and h.get("numeric_cleanup_720_active") is not True:
        missing.append("numeric_cleanup_720_active")
    if missing:
        raise RuntimeError("actual hit/graph gate failed: " + ", ".join(missing))
    return {"mandatory_hits": {key: h.get(key) for key in sorted(need)},
            "numeric_cleanup_sites": sites, "resolved_mode_options": options,
            "history_fractional_status": "conditional_unexercised" if conditional else
                ("hit" if "num_history_fractional" in selected else "not_selected"),
            "graph_route": h.get("structure_combo_frame_route")}


class FixedControlsAPI(base.API):
    def __init__(self):
        super().__init__()
        self.controls = None

    def call(self, path, payload=None):
        st = super().call(path, payload)
        if self.controls is not None and isinstance(st, dict) and "settings" in st:
            settings = st["settings"] or {}
            changed = [k for k, v in self.controls.items() if settings.get(k) != v]
            if changed:
                raise RuntimeError("Fixed scene/control changed: " + ", ".join(changed))
        return st


def comparison(candidate, before, after):
    value = candidate["measurement"]["average_ms_mean"]
    left = before["measurement"]["average_ms_mean"]
    right = after["measurement"]["average_ms_mean"]
    conditional = candidate["hit_evidence"]["history_fractional_status"] == "conditional_unexercised"
    label = ("conditional_unexercised" if conditional else
             "faster_than_both_anchors" if value < min(left, right) else
             "slower_than_both_anchors" if value > max(left, right) else "within_anchor_drift")
    return {"optimizations": candidate["target_optimizations"], "average_ms": value,
            "baseline_before_ms": left, "baseline_after_ms": right,
            "baseline_mean_ms": (left + right) / 2,
            "delta_vs_anchor_mean_ms": value - (left + right) / 2,
            "baseline_drift_ms": right - left, "screen_classification": label,
            "requires_repeat_for_recommendation": True}


def load_plan():
    plan = json.loads(PLAN.read_text(encoding="utf-8-sig"))
    reg = base.registry()
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(Path(reg.__file__)) != plan["registry_sha256"]:
        raise RuntimeError("Installed registry changed after screen-plan preparation")
    cfg = base.GAME.parent / "data" / "product-v1" / "local-runtime-v1.json"
    if digest(cfg) != plan["availability_sha256"]:
        raise RuntimeError("Installed candidate availability changed")
    if {r["key"] for r in plan["single_candidates"]} != set(reg.PROFILES):
        raise RuntimeError("Screen plan does not cover actual installed registry")
    return plan, reg


def final_selection(reg, supplied):
    selected = list(base.OLD) if supplied is None else supplied.split(",")
    if len(selected) != len(set(selected)) or set(selected) - set(reg.PROFILES):
        raise ValueError("Unknown or duplicate final selection")
    reg.combined_mode_options(selected)
    return selected


def self_test():
    _, reg = load_plan()
    assert final_selection(reg, None) == list(base.OLD)
    chosen = list(base.OLD) + ["c512_k8_probability", "c512_k8_merge", "c512_k8_post_fma32"]
    assert final_selection(reg, ",".join(chosen)) == chosen
    for supplied in ("unknown", "c512_k8_decoder,c512_k8_decoder"):
        try: final_selection(reg, supplied)
        except ValueError: pass
        else: raise AssertionError("Invalid final selection accepted")
    health = {"c512_library_720_active": True, "native_k8_720_active": True,
              "structure_combo_frame_route": "replay", "numeric_cleanup_sites": {}}
    actual_hits({"health": health}, [], reg)
    for changes, selected in (({"c512_library_720_active": False}, []),
                              ({"structure_combo_frame_route": "capture"}, []),
                              ({}, ["num_vit_qkv_full_k"])):
        try: actual_hits({"health": {**health, **changes}}, selected, reg)
        except RuntimeError: pass
        else: raise AssertionError("Missing-hit/capture gate accepted invalid state")
    fractional = {**health, "numeric_cleanup_sites": {"history": False},
                  "numeric_cleanup_720_active": False}
    assert actual_hits({"health": fractional}, ["num_history_fractional"], reg)[
        "history_fractional_status"] == "conditional_unexercised"
    fractional["numeric_cleanup_sites"] = {"history": True}
    fractional["numeric_cleanup_720_active"] = True
    assert actual_hits({"health": fractional}, ["num_history_fractional"], reg)[
        "history_fractional_status"] == "hit"
    controls = {key: index for index, key in enumerate(CONTROL_KEYS)}
    saved = copy.deepcopy(controls)
    controls.pop("experiment_720")
    assert base.payload(controls, base.OLD)["style"] == saved["style"]
    assert controls == {k: v for k, v in saved.items() if k != "experiment_720"}
    fixture = lambda value: {"measurement": {"average_ms_mean": value},
              "target_optimizations": ["x"], "hit_evidence": {"history_fractional_status": "not_selected"}}
    assert comparison(fixture(49.9), fixture(50), fixture(50.1))["screen_classification"] == "faster_than_both_anchors"
    assert comparison(fixture(50.05), fixture(50), fixture(50.1))["screen_classification"] == "within_anchor_drift"
    assert comparison(fixture(50.2), fixture(50), fixture(50.1))["screen_classification"] == "slower_than_both_anchors"
    for pair in reg.conflicting_profiles():
        try: reg.combined_mode_options(list(pair))
        except ValueError: pass
        else: raise AssertionError("Conflicting pair accepted")
    return {"status": "passed", "candidates": len(reg.PROFILES), "network_touched": False,
            "gpu_touched": False, "controls_preserved": True,
            "checks": ["legacy and graph hits", "numeric child selection", "conditional history",
                       "sub-ms gains and anchor drift", "mathematical conflicts"], "utc": base.utc()}


def run(args):
    plan, reg = load_plan()
    if not 30 <= args.measure_frames <= 1500:
        raise ValueError("Measure 30 through 1500 new completed frames per arm")
    base.MEASURE_FRAMES = args.measure_frames
    output = args.output.resolve()
    if not output.is_relative_to(PERF.resolve()) or output.exists():
        raise ValueError("Use a new output directory inside the task live-web-perf directory")
    output.mkdir(parents=True)
    base.actual_hits = actual_hits  # Reuse corrected frame sampling without modifying its source.
    api = FixedControlsAPI()
    arms, blocks = [], []
    initial, initial_settings = None, None
    started = base.utc()

    def arm(target, role):
        target = list(target)
        reg.combined_mode_options(target)
        st = api.call("/api/state"); base.fatal(st)
        if not base.matches(st, target):
            st = api.call("/api/state", base.payload(initial_settings, target)); base.fatal(st)
        result = base.measure_arm(api, target, len(arms) + 1)
        # Require actual sites again after measurement, not just at warmup.
        st = api.call("/api/state"); base.fatal(st)
        if not base.matches(st, target): raise RuntimeError("Mode changed at arm completion")
        result["final_hit_evidence"] = actual_hits(st, target, reg)
        result["role"] = role
        arms.append(result)
        print(json.dumps({"event": "arm_completed", "arm": result["arm"], "role": role,
              "target": target, "average_ms": result["measurement"]["average_ms_mean"]},
              ensure_ascii=False), flush=True)
        return result

    try:
        api.connect()
        if api.checkboxes != set(reg.PROFILES): raise RuntimeError("Actual webpage registry mismatch")
        initial = api.call("/api/state"); base.fatal(initial)
        if args.prepare_base:
            geometry = initial.get("source_geometry") or {}
            if (initial.get("route") != "pre-xess-fullsize" or
                    (geometry.get("width"), geometry.get("height")) != (1280, 720)):
                raise RuntimeError("Cannot prepare base on a different route/source geometry")
            settings = copy.deepcopy(initial.get("settings") or {})
            settings.update(enabled=True, input_size=720, history_mode="fused", graph_replay=True,
                            backend_variant="unrounded", experiment_720="c512_k8")
            prepare_target = list(base.OLD) if args.command == "restore-old" else []
            api.call("/api/state", base.payload(settings, prepare_target))
            prepared = base.wait_target(api, prepare_target)
            s, _ = base.validate_base(prepared)
        else:
            s, _ = base.validate_base(initial)
        initial_settings = copy.deepcopy(s)
        api.controls = {key: s[key] for key in CONTROL_KEYS}
        if not set(reg.PROFILES) <= set(initial.get("available_experiments_720") or []):
            raise RuntimeError("Actual API omits installed candidates")
        if initial.get("timing_enabled") is False:
            st = api.call("/api/timing", {"enabled": True}); base.fatal(st)
            if st.get("timing_enabled") is not True: raise RuntimeError("Timing did not enable")
        elif initial.get("timing_enabled") is not True:
            raise RuntimeError("Timing control unavailable")

        if args.command == "singles":
            keys = [r["key"] for r in plan["single_candidates"]]
            if args.keys:
                requested = args.keys.split(",")
                if set(requested) - set(keys) or len(requested) != len(set(requested)):
                    raise ValueError("Unknown or duplicate screen keys")
                keys = requested
            left = arm([], "baseline-open")
            for offset in range(0, len(keys), 4):
                rows = [arm([key], "single") for key in keys[offset:offset + 4]]
                right = arm([], "baseline-close")
                block = {"status": "completed", "block": len(blocks) + 1,
                         "before": left, "candidates": rows, "after": right,
                         "comparisons": [comparison(row, left, right) for row in rows]}
                blocks.append(block)
                path = output / f"block-{len(blocks):02d}-completed.json"
                base.write_json(path, block)
                print(json.dumps({"event": "block_completed", "checkpoint": str(path),
                                  "comparisons": block["comparisons"]}, ensure_ascii=False), flush=True)
                left = right
            final_target = final_selection(reg, args.final_optimizations)
            summary = {"status": "completed", "phase": "single_screens",
                       "comparisons": [row for b in blocks for row in b["comparisons"]],
                       "single_deltas_are_not_additive": True}
        elif args.command == "restore-old":
            final_target = list(base.OLD)
            summary = {"status": "completed", "phase": "restore_old_three_only",
                       "performance_measurements_taken": False}
        else:
            spec = json.loads(args.targets.read_text(encoding="utf-8-sig"))
            reference, targets = spec["reference"], spec["targets"]
            records = []
            for target in targets:
                rows = [arm(reference, "reference-open"), arm(target, "candidate-1"),
                        arm(target, "candidate-2"), arm(reference, "reference-close")]
                b = statistics.fmean(r["measurement"]["average_ms_mean"] for r in (rows[0], rows[3]))
                c = statistics.fmean(r["measurement"]["average_ms_mean"] for r in (rows[1], rows[2]))
                changes = [comparison(rows[1], rows[0], rows[3]), comparison(rows[2], rows[0], rows[3])]
                confirmed = all(x["screen_classification"] == "faster_than_both_anchors" for x in changes)
                item = {"status": "completed", "reference": reference, "target": target,
                        "baseline_ms": b, "candidate_ms": c, "delta_ms": c-b,
                        "confirmed_direction": confirmed, "arms": rows, "comparisons": changes}
                records.append(item)
                path = output / f"pair-{len(records):02d}-completed.json"
                base.write_json(path, item)
                print(json.dumps({"event": "pair_completed", "checkpoint": str(path),
                      "delta_ms": c-b, "confirmed_direction": confirmed}, ensure_ascii=False), flush=True)
            winners = [r for r in records if r["confirmed_direction"]]
            final_target = min(winners, key=lambda r: r["delta_ms"])["target"] if winners else reference
            summary = {"status": "completed", "phase": "paired_confirmations", "records": records}

        st = api.call("/api/state"); base.fatal(st)
        frame = st["processing"]["frames"]
        if not base.matches(st, final_target):
            api.call("/api/state", base.payload(initial_settings, final_target))
        final, hits, frames = base.wait_restore_evidence(api, final_target, reg, frame)
        summary.update(started_utc=started, finished_utc=base.utc(), arms=arms,
                       initial_state=base.clean(initial), post_restore_actual_state=base.clean(final),
                       final_optimizations=final_target, final_actual_hits=hits, final_frame_evidence=frames,
                       timing_metric="overlapping recent-32 completed-frame web averages; not game FPS",
                       controls_preserved=True, frame_generation_touched=False,
                       code_or_cache_changed=False, token_saved_or_printed=False)
        path = output / "completed-checkpoint.json"
        base.write_json(path, summary)
        print(json.dumps({"event": "completed", "checkpoint": str(path), "applied": final_target},
                         ensure_ascii=False), flush=True)
    except Exception as exc:
        stopped = {"status": "stopped", "started_utc": started, "finished_utc": base.utc(),
                   "error": f"{type(exc).__name__}: {str(exc)[:400]}", "completed_arms": arms,
                   "completed_blocks": blocks, "code_or_cache_changed": False}
        # Restore only on a healthy progressing API; do not hide a failed GPU/session.
        try:
            if initial_settings is not None:
                st = api.call("/api/state"); base.fatal(st)
                api.call("/api/state", base.payload(initial_settings, base.OLD))
                st, hits, proof = base.wait_restore_evidence(api, base.OLD, reg, st["processing"]["frames"])
                stopped["healthy_restore"] = {"state": base.clean(st), "hits": hits, "frames": proof}
        except Exception as restore_exc:
            stopped["restore_error"] = f"{type(restore_exc).__name__}: {str(restore_exc)[:240]}"
        base.write_json(output / "stopped-checkpoint.json", stopped)
        print(json.dumps({"event": "stopped", "checkpoint": str(output / "stopped-checkpoint.json"),
                          "error": stopped["error"]}, ensure_ascii=False), flush=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-test", "singles", "confirm", "restore-old"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keys")
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--measure-frames", type=int, default=30,
                        help="New completed frames per arm (default: 30)")
    parser.add_argument("--final-optimizations",
                        help="Comma-separated already measured selection to restore after singles")
    parser.add_argument("--prepare-base", action="store_true",
                        help="Use authorized 720p C512+K8 base, preserving style/strength controls")
    args = parser.parse_args()
    if args.command == "self-test":
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
    else:
        if not args.output or (args.command == "confirm" and not args.targets):
            parser.error("--output required; confirm also requires --targets")
        run(args)
