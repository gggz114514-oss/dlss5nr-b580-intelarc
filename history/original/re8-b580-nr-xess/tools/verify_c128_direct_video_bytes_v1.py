"""Check every float RGB frame with twelve C128 QKV direct-write producers.

Both arms retain separate temporal history and the same frozen DIS vectors.
No video is encoded; only hashes and graph-route evidence are written to D:.
This is a correctness check, not a latency benchmark or game-scene review.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import zlib

from render_c32_unrounded_face540_v1 import PRIOR, SOURCE_SIZE, read_frame, sha


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(r"G:\SteamLibrary\steamapps\common\Resident Evil Village BIOHAZARD VILLAGE\nr-runtime")
BASE = Path(r"D:\Codex-NR-Experiments\nr-b580")
FRAMES = 243
SIZES = {480: (480, 864), 540: (540, 960)}


def main(args):
    report = args.report.resolve()
    if report.drive.casefold() != "d:" or not report.is_relative_to(BASE.resolve()):
        raise ValueError("Report must be under the D: NR experiment directory")
    runtime, cache = args.runtime.resolve(), args.cache.resolve()
    if not cache.is_dir():
        raise FileNotFoundError(f"Precompiled cache absent: {cache}")
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    source_manifest = Path(prior["source"])
    source = source_manifest.resolve()
    if len(prior["frames"]) != FRAMES or sha(source) != prior["sources"][str(source_manifest)]:
        raise RuntimeError("Frozen face source/manifest changed")
    ffmpeg = Path(next(p for p in prior["sources"] if Path(p).name == "ffmpeg.exe"))
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(runtime / "game"), str(runtime), str(runtime / "modules"),
                    str(runtime / "fast/backend"), str(ROOT / "game")]
    from runtime_environment import isolate
    isolate()
    from fast_cached_runtime_v1 import DiskOnly, bootstrap
    bootstrap()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from triton import knobs
    from nr_backend.controlled_temporal import NRControls
    from nr_game_fullsize import FullsizeGameModes
    from c128_qkv_direct_pack_all_v1 import installed as candidate_scope

    config = json.loads((runtime / "data/product-v1/local-runtime-v1.json")
                        .read_text(encoding="utf-8-sig"))
    combo_modes = tuple(config.get("structure_combo_modes", ()))
    if combo_modes != (480, 540):
        raise RuntimeError("Expected current 480/540 structure combo")
    output_size = SIZES[args.height]
    previous_cache = (os.environ.get("TRITON_CACHE_DIR"), knobs.cache.dir)
    os.environ["TRITON_CACHE_DIR"] = str(cache)
    knobs.cache.dir = str(cache)
    result = {"passed": False, "height": args.height, "render_size": output_size,
              "frames_expected": FRAMES, "source": str(source),
              "source_sha256": sha(source), "runtime": str(runtime), "cache": str(cache),
              "candidate_sha256": sha(ROOT / "game/c128_qkv_direct_pack_all_v1.py"),
              "kernel_sha256": sha(ROOT / "game/c128_qkv_direct_pack_one_v1.py"),
              "arms": {}}
    try:
        for arm in ("baseline", "candidate"):
            decoder = subprocess.Popen(
                [str(ffmpeg), "-v", "error", "-threads", "2", "-i", str(source),
                 "-vf", prior["decode_filter"], "-frames:v", str(FRAMES),
                 "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            modes = None
            hashes = []
            routes = []
            capture_gates = []
            try:
                modes = FullsizeGameModes(runtime / "exact",
                                          runtime / "data/product-v1/profile-v1.json",
                                          config["profile_sha256"], controlled=True,
                                          graph_capture_policy="all", combo_modes=combo_modes)
                with DiskOnly() as disk:
                    modes.select(args.height, output_size)
                    scope = (candidate_scope(modes.session._stack.window_blocks,
                                             modes.session._stack.model,
                                             height=args.height)
                             if arm == "candidate" else nullcontext(None))
                    with scope as calls, torch.inference_mode():
                        for i in range(FRAMES):
                            pixels = np.frombuffer(read_frame(decoder.stdout,
                                                               SOURCE_SIZE[0] * SOURCE_SIZE[1] * 3),
                                                   dtype=np.uint8).reshape(*SOURCE_SIZE, 3)
                            motion_row = prior["frames"][i]["motion"]
                            compressed = Path(motion_row["path"]).read_bytes()
                            if hashlib.sha256(compressed).hexdigest() != motion_row["sha256"]:
                                raise RuntimeError(f"Motion file changed at frame {i}")
                            flow = np.load(io.BytesIO(zlib.decompress(compressed)))
                            if flow.shape != (*SOURCE_SIZE, 2) or flow.dtype != np.float16:
                                raise RuntimeError(f"Invalid frozen motion at frame {i}")
                            if args.height == 540:
                                pixels = np.asarray(Image.fromarray(pixels).resize(
                                    (output_size[1], output_size[0]),
                                    resample=Image.Resampling.BICUBIC))
                                flow = F.interpolate(
                                    torch.from_numpy(flow.astype(np.float32).transpose(2, 0, 1))
                                    .unsqueeze(0), size=output_size, mode="bilinear",
                                    align_corners=False)[0].permute(1, 2, 0).contiguous().numpy()
                                flow *= np.array([output_size[1] / SOURCE_SIZE[1],
                                                  output_size[0] / SOURCE_SIZE[0]], dtype=np.float32)
                            else:
                                flow = flow.astype(np.float32)
                            rgb = torch.from_numpy(pixels.copy()).to("xpu", dtype=torch.float32) / 255
                            mv = torch.from_numpy(flow.copy()).to("xpu")
                            before_c128 = (dict(calls["by_block"]) if calls is not None else None)
                            output = modes.process(rgb, mv, height=args.height, reset=i == 0,
                                                   history_warp="fused", graph_replay=True,
                                                   controls=NRControls())
                            torch.xpu.synchronize()
                            route = modes.last_combo_frame_route
                            routes.append(route)
                            if route == "capture":
                                gate = modes.last_combo_capture_gate
                                if gate is None or not all(gate.values()):
                                    raise RuntimeError(f"Combo capture gate failed at frame {i}")
                                capture_gates.append(dict(gate))
                                if before_c128 is not None and not all(
                                        calls["by_block"][name] > previous
                                        for name, previous in before_c128.items()):
                                    raise RuntimeError(f"A C128 producer missed captured frame {i}")
                            if i >= 16 and (route != "replay" or not modes.last_graph_used):
                                raise RuntimeError(f"Frame {i} did not replay the model graph")
                            if not bool(torch.isfinite(output).all().item()):
                                raise RuntimeError(f"Nonfinite output at frame {i}")
                            hashes.append(hashlib.sha256(
                                output.detach().contiguous().cpu().numpy().tobytes()).hexdigest())
                    if arm == "candidate" and (
                            calls["direct_pack"] < 12 or len(calls["by_block"]) != 12 or
                            not all(n > 0 for n in calls["by_block"].values())):
                        raise RuntimeError("Candidate missed C128 blocks")
                    if len(capture_gates) != 2:
                        raise RuntimeError(f"Expected two gated captures for {arm}")
                    hits = disk.hits
                stderr = decoder.stderr.read().decode("utf-8", "replace")
                if decoder.wait(timeout=30) != 0:
                    raise RuntimeError(f"Video decode failed for {arm}: {stderr[-1000:]}")
                result["arms"][arm] = {"frame_hashes": hashes,
                                       "routes": {name: routes.count(name) for name in set(routes)},
                                       "capture_gates": capture_gates,
                                       "candidate_calls": calls, "disk_only_hits": hits}
            finally:
                if modes is not None:
                    modes.close()
                if decoder.poll() is None:
                    decoder.terminate()
                if decoder.stdout:
                    decoder.stdout.close()
                if decoder.stderr:
                    decoder.stderr.close()
        left, right = (result["arms"][arm]["frame_hashes"] for arm in ("baseline", "candidate"))
        differences = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
        result["differing_frame_indices"] = differences
        result["passed"] = len(left) == len(right) == FRAMES and not differences
        for row in result["arms"].values():
            hashes = row.pop("frame_hashes")
            row["first_sha256"] = hashes[0]
            row["last_sha256"] = hashes[-1]
            row["all_hashes_sha256"] = hashlib.sha256("".join(hashes).encode("ascii")).hexdigest()
        if not result["passed"]:
            raise RuntimeError(f"Video RGB mismatch on {len(differences)} frames")
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-12000:]
    finally:
        knobs.cache.dir = previous_cache[1]
        if previous_cache[0] is None:
            os.environ.pop("TRITON_CACHE_DIR", None)
        else:
            os.environ["TRITON_CACHE_DIR"] = previous_cache[0]
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "height": args.height,
                      "error": result.get("error"),
                      "differing_frames": len(result.get("differing_frame_indices", []))},
                     ensure_ascii=False), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, choices=(480, 540), required=True)
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
