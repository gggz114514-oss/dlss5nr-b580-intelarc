"""Build a separate byte/fence probe that explicitly records missing debug layer."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
WORKER = PROJECT / "artifacts/bridge-resource-pool-v1-20261001"
STAGE = PROJECT / "artifacts/bridge-resource-pool-byte-probe-v2-20261002"
DATA = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\bridge-resource-pool-byte-probe-v2-20261002")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def one(text, old, new):
    if text.count(old) != 1: raise RuntimeError("Changed source anchor: " + old[:100])
    return text.replace(old, new)


def main():
    if STAGE.exists(): raise RuntimeError("Fresh E stage required")
    paths = {"worker_source": WORKER / "payload/tests/hdr_resource_pool_gpu.cpp",
             "hdr_source": WORKER / "payload/src/nr_hdr_proxy.cpp",
             "hdr_header": WORKER / "payload/src/nr_hdr_proxy.h",
             "pool_header": WORKER / "payload/src/nr_hdr_resource_pool.h",
             "shader": WORKER / "payload/shaders/nr_hdr_proxy.hlsl",
             "worker_manifest": WORKER / "source-manifest.json"}
    pins = {str(path): sha(path) for path in paths.values()}
    text = paths["worker_source"].read_text(encoding="utf-8")
    text = one(text, "void debug_clean(ID3D12InfoQueue* info) {", "void debug_clean(ID3D12InfoQueue* info) {\n    if(!info) return; // Absence is reported, never counted as zero debug errors.")
    text = one(text, '''        if(FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(debug.GetAddressOf())))) {std::puts("SKIP: D3D12 debug layer unavailable; no verified GPU run");return 77;}
        debug->EnableDebugLayer();ComPtr<IDXGIFactory6> factory;hr(CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf())),"DXGI factory");''', '''        const bool debug_available=SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(debug.GetAddressOf())));
        if(debug_available) debug->EnableDebugLayer();
        else std::puts("NOTE: D3D12 debug layer unavailable; byte/fence checks continue, debug coverage is unavailable.");
        ComPtr<IDXGIFactory6> factory;hr(CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf())),"DXGI factory");''')
    text = one(text, 'ComPtr<ID3D12InfoQueue> info;hr(device.As(&info),"Debug info queue");', 'ComPtr<ID3D12InfoQueue> info;if(debug_available) hr(device.As(&info),"Debug info queue");')
    text = one(text, r'  \"status\":\"passed\",', r'  \"status\":\"passed_byte_fence_checks\",')
    debug_line = next(line for line in text.splitlines() if 'debug_errors' in line)
    text = one(text, debug_line, r'''            <<",\n  \"debug_layer_available\":"<<(debug_available?"true":"false")
            <<",\n  \"debug_errors\":"<<(debug_available?"0":"null")
            <<",\n  \"samples\":[\n"<<records<<"\n  ]\n}\n";''')
    text = one(text, 'debug errors=0; synthetic consumer only\\n",comparisons,stats.hits);', 'debug coverage=%s; synthetic consumer only\\n",comparisons,stats.hits,debug_available?"checked":"unavailable");')
    STAGE.mkdir()
    source = STAGE / "hdr_resource_pool_byte_v2.cpp"
    source.write_text(text, encoding="utf-8", newline="\n")
    DATA.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="cpu-", dir=DATA))
    spec = importlib.util.spec_from_file_location("cpu_build_environment", PROJECT / "artifacts/dx12-xpu-fence-probe-v5-20261002/build_cpu.py")
    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    env, cl, dumpbin, setup = helper.environment(scratch)
    exe = scratch / "nr_hdr_resource_pool_byte_v2.exe"
    command = [cl, "/nologo", "/std:c++20", "/O2", "/EHsc", "/MD", "/W4", "/DNOMINMAX", "/DWIN32_LEAN_AND_MEAN",
               "/I"+str(WORKER / "payload/src"), "/I"+str(WORKER / "payload/include"),
               str(source), str(paths["hdr_source"]), "/Fe"+str(exe),
               "/link", "d3d12.lib", "dxgi.lib", "d3dcompiler.lib", "bcrypt.lib"]
    result = subprocess.run(command, cwd=scratch, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", timeout=60)
    (scratch / "build.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode: raise RuntimeError(result.stdout[-8000:])
    for path, expected in pins.items():
        if sha(Path(path)) != expected: raise RuntimeError("Frozen worker changed: " + path)
    receipt = {"status": "CPU_compiled_GPU_unverified", "input_pins": pins,
               "source": str(source), "source_sha256": sha(source), "exe": str(exe), "exe_sha256": sha(exe),
               "command": command, "exception_unwind": "/EHsc", "msvc_setup": setup,
               "GPU_executed": False, "G_writes": False, "scope": "Original96byte/fence/binding/counter checks unchanged; debug presence and errors explicitly boolean/null. Not original full-debug acceptance."}
    (scratch / "CPU_COMPILE_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "receipt": str(scratch / "CPU_COMPILE_RECEIPT.json"), "exe": str(exe)}))


if __name__ == "__main__": main()
