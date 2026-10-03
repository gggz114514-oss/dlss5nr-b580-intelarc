"""Prepare installed 720p checkbox targets without touching a running game."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = args.runtime / "game/numeric_game_profiles_720_v1.py"
    config_path = args.runtime / "data/product-v1/local-runtime-v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    installed = set(config["available_720_experiments"])
    spec = importlib.util.spec_from_file_location("live_sweep_installed_registry", registry)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    selected = [name for name in module.PROFILES if name in installed]
    historical_three = ["c512_k8_decoder", "c512_k8_c32_native", "num_history_fractional"]
    if not set(historical_three) <= set(selected):
        raise RuntimeError("Previously reviewed three-box combination is not installed")
    plan = {
        "kind": "cpu-only-live-web-sweep-plan",
        "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
        "availability_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "baseline": {"input_size": 720, "experiment_720": "c512_k8",
                     "backend_variant": "unrounded", "history_mode": "fused",
                     "graph_replay": True, "optimizations_720": []},
        "historical_three": historical_three,
        "warmup": {"min_seconds": 8, "min_new_completed_frames": 35},
        "measurement": {"min_new_completed_frames": 30, "poll_seconds": 0.2,
                        "rolling_average_window": 32},
        "single_candidates": [
            {"key": name, "label": module.PROFILES[name].label,
             "optimizations_720": [name],
             "resolved_mode_options": module.combined_mode_options([name]),
             "health_fields": list(module.HIT_FIELDS[name]),
             "conditional_fractional_site": name == "num_history_fractional"}
            for name in selected],
        "true_conflicts": [list(pair) for pair in module.conflicting_profiles()
                           if set(pair) <= set(selected)],
        "notes": [
            "Installed availability determines eligibility; stale pending_validation metadata does not disable installed items.",
            "A motion-dependent station not exercised in the fixed scene is not evidence of an algorithmic slowdown.",
            "New combinations require runtime cache hits; a missing specialization is not a performance result.",
            "Do not add individual deltas, or describe rolling web averages as full game frame times.",
        ],
    }
    if any(name == "torch" or name.startswith(("triton", "nr_backend")) for name in sys.modules):
        raise RuntimeError("Unexpected GPU module import while preparing CPU plan")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidates": len(selected), "conflicts": len(plan["true_conflicts"]),
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
