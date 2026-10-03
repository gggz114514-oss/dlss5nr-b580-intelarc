"""Time only the two identical DXBC compiler calls used by the installed bridge.

No D3D device, GPU work, process injection, game/control mutation or PSO creation.
The result is an offline CPU call cost, not a whole-game frame-time saving.
"""
from pathlib import Path
import ctypes as C
import hashlib
import json
import math
import os
import statistics
import time

ROOT = Path("E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt")
SHADER = Path("G:/epic/Cyberpunk2077/bin/x64/plugins/nr_hdr_proxy.hlsl")
EXPECTED = "5aabd2fda6062983a8976a888ad90ec692997ce653dd8637489efa564fcae5ad"
OUT = Path("D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/"
           "live-web-perf-20261001/hdr-shader-compile-cpu-20261001")
FLAGS = (1 << 15) | (1 << 13)  # SDK: optimization level 3, IEEE strictness.


def physical(path):
    if any(p.is_symlink() or p.is_junction() for p in (path, *path.parents)):
        raise RuntimeError("Redirected path: " + str(path))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method(blob, index, result, *args):
    table = C.cast(blob, C.POINTER(C.POINTER(C.c_void_p))).contents
    return C.WINFUNCTYPE(result, C.c_void_p, *args)(table[index])


def blob_bytes(blob):
    address = method(blob, 3, C.c_void_p)(blob)
    length = method(blob, 4, C.c_size_t)(blob)
    return C.string_at(address, length)


def release(blob):
    if blob:
        method(blob, 2, C.c_uint32)(blob)


def compile_entry(compiler, entry):
    code, errors = C.c_void_p(), C.c_void_p()
    try:
        started = time.perf_counter_ns()
        status = compiler(str(SHADER), None, C.c_void_p(1), entry.encode("ascii"),
                          b"cs_5_0", FLAGS, 0, C.byref(code), C.byref(errors))
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        if status < 0 or not code:
            error = blob_bytes(errors).decode("utf-8", "replace") if errors else "no error blob"
            raise RuntimeError(f"Compile {entry} failed: HRESULT {status & 0xffffffff:08x}: {error}")
        bytecode = blob_bytes(code)
        return {"entry": entry, "compiler_call_ms": elapsed,
                "bytecode_sha256": hashlib.sha256(bytecode).hexdigest(),
                "bytecode_bytes": len(bytecode)}
    finally:
        release(errors)
        release(code)


def percentile(values, q):
    items = sorted(values)
    pos = (len(items) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    return items[low] + (items[high] - items[low]) * (pos - low)


def summary(rows, key):
    values = [r[key] for r in rows]
    return {"mean_ms": statistics.fmean(values), "p50_ms": percentile(values, .5),
            "p95_ms": percentile(values, .95), "min_ms": min(values), "max_ms": max(values)}


def main():
    physical(SHADER)
    physical(OUT)
    if sha(SHADER) != EXPECTED or sha(ROOT / "shaders/nr_hdr_proxy.hlsl") != EXPECTED:
        raise RuntimeError("The installed/source shader changed")
    if OUT.exists():
        raise RuntimeError("Preserving prior result directory: " + str(OUT))
    library_path = Path(os.environ["WINDIR"]) / "System32/d3dcompiler_47.dll"
    compiler = C.WinDLL(str(library_path)).D3DCompileFromFile
    compiler.argtypes = (C.c_wchar_p, C.c_void_p, C.c_void_p, C.c_char_p, C.c_char_p,
                         C.c_uint32, C.c_uint32, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p))
    compiler.restype = C.c_int32
    rows, hashes = [], {}
    for sample in range(35):
        entries = [compile_entry(compiler, e) for e in ("prepare", "composite")]
        for entry in entries:
            name = entry["entry"]
            hashes.setdefault(name, entry["bytecode_sha256"])
            if hashes[name] != entry["bytecode_sha256"]:
                raise RuntimeError("Non-deterministic bytecode for " + name)
        rows.append({"sample": sample, "warmup": sample < 5,
                     "prepare_ms": entries[0]["compiler_call_ms"],
                     "composite_ms": entries[1]["compiler_call_ms"],
                     "pair_ms": sum(e["compiler_call_ms"] for e in entries), "entries": entries})
    if sha(SHADER) != EXPECTED:
        raise RuntimeError("Shader changed while measuring")
    measured = rows[5:]
    result = {"completed": True, "cpu_only": True, "gpu_used": False,
              "writes_game": False, "control_requests": 0, "samples": 30, "warmup_pairs": 5,
              "shader": {"path": str(SHADER), "sha256": EXPECTED},
              "compiler": {"path": str(library_path), "sha256": sha(library_path)},
              "source": {"path": str(ROOT / "src/nr_hdr_proxy.cpp"),
                         "sha256": sha(ROOT / "src/nr_hdr_proxy.cpp")},
              "profile": "cs_5_0", "flags": FLAGS, "bytecode_sha256": hashes,
              "summary": {key: summary(measured, key) for key in ("prepare_ms", "composite_ms", "pair_ms")},
              "cold_first_pair": rows[0], "rows": rows,
              "limits": "Offline compiler-call cost only; not actual Evaluate critical-path measurement. "
                        "No PSO/root/allocator timing or RTSS/Present comparison. "
                        "Do not claim this explains or saves the whole observed 15 ms."}
    OUT.mkdir(parents=True)
    (OUT / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed": True, "cpu_only": True, "samples": 30,
                      "summary": result["summary"], "result": str(OUT / "RESULT.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
