// Process-local diagnostics for the owned NR reference harness. No GPU inputs
// or parameters are changed. NVAPI private signatures follow public runtime
// research at taowen/dlss5-as-inpainting f8e18d3 (reshade_capture tool).
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <cstdint>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cstring>
#include "MinHook.h"

namespace {
using Query = void* (__cdecl*)(unsigned);
using ModuleFn = int (__stdcall*)(void*, const void*, uint32_t, uint64_t*);
using FunctionFn = int (__stdcall*)(void*, uint64_t, const char*, uint64_t*);
struct Dim3 { uint32_t x, y, z; };
struct Kernel { uint64_t function; Dim3 grid, block; uint32_t shared; const void* params; uint32_t size; };
static_assert(sizeof(Kernel) == 56, "NVAPI launch ABI size");
using LaunchFn = int (__stdcall*)(void*, const Kernel*, uint32_t);
Query query_original = nullptr;
ModuleFn module_original = nullptr;
FunctionFn function_original = nullptr;
LaunchFn launch_original = nullptr;
std::mutex io_mutex;
std::ofstream log_file;
std::filesystem::path output;
std::unordered_map<uint64_t, std::string> names;
std::atomic<uint32_t> sequence{0};
uint64_t captured_bytes = 0;
bool initialized = false;
HMODULE nvapi = nullptr;
ID3D12Device* capture_device = nullptr;
using CreateResourceFn = HRESULT (STDMETHODCALLTYPE*)(ID3D12Device*, const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS,
    const D3D12_RESOURCE_DESC*, D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE*, REFIID, void**);
using CreateUavFn = void (STDMETHODCALLTYPE*)(ID3D12Device*, ID3D12Resource*, ID3D12Resource*,
    const D3D12_UNORDERED_ACCESS_VIEW_DESC*, D3D12_CPU_DESCRIPTOR_HANDLE);
CreateResourceFn create_resource_original = nullptr;
CreateUavFn create_uav_original = nullptr;
std::mutex resource_mutex;
std::vector<ID3D12Resource*> resources;
struct Snapshot { ID3D12Resource* readback; uint64_t size; std::string file; };
std::vector<Snapshot> snapshots;
ID3D12Resource* pre_arena = nullptr;
unsigned trace_first=13, trace_last=16;

std::string quote(const std::string& value) {
    std::string result = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { result += '\\'; result += c; }
        else if (c >= 32 && c < 127) result += c;
        else { char encoded[8]; sprintf_s(encoded, "\\u%04x", c); result += encoded; }
    }
    return result + '"';
}
void log(const std::string& line) {
    std::lock_guard<std::mutex> lock(io_mutex);
    log_file << line << '\n';
    log_file.flush();
}
bool dump(const std::string& name, const void* data, size_t size) {
    if (!data || size == 0 || size > 64 * 1024 * 1024) return false;
    std::lock_guard<std::mutex> lock(io_mutex);
    if (captured_bytes + size > 256ull * 1024 * 1024) return false;
    std::vector<char> copy(size);
    SIZE_T read = 0;
    if (!ReadProcessMemory(GetCurrentProcess(), data, copy.data(), size, &read) || read != size) return false;
    std::ofstream file(output / name, std::ios::binary | std::ios::trunc);
    file.write(copy.data(), size);
    if (!file) return false;
    captured_bytes += size;
    return true;
}
void remember_resource(ID3D12Resource* resource) {
    if (!resource) return;
    const auto desc = resource->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || desc.Width < 1024 * 1024 ||
        desc.Width > 256ull * 1024 * 1024 || !(desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) return;
    std::lock_guard<std::mutex> lock(resource_mutex);
    if (std::find(resources.begin(), resources.end(), resource) != resources.end()) return;
    resource->AddRef();
    resources.push_back(resource);
    log("{\"event\":\"resource\",\"gpu_va\":" + std::to_string(resource->GetGPUVirtualAddress()) +
        ",\"bytes\":" + std::to_string(desc.Width) + '}');
}
HRESULT STDMETHODCALLTYPE create_resource_hook(ID3D12Device* device, const D3D12_HEAP_PROPERTIES* heap,
    D3D12_HEAP_FLAGS flags, const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES state,
    const D3D12_CLEAR_VALUE* clear, REFIID iid, void** result) {
    const auto status = create_resource_original(device, heap, flags, desc, state, clear, iid, result);
    if (SUCCEEDED(status) && device == capture_device && result && *result) {
        try {
            ID3D12Resource* resource = nullptr;
            if (SUCCEEDED(reinterpret_cast<IUnknown*>(*result)->QueryInterface(IID_PPV_ARGS(&resource)))) {
                remember_resource(resource);
                resource->Release();
            }
        } catch (...) {}
    }
    return status;
}
void STDMETHODCALLTYPE create_uav_hook(ID3D12Device* device, ID3D12Resource* resource, ID3D12Resource* counter,
    const D3D12_UNORDERED_ACCESS_VIEW_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE handle) {
    create_uav_original(device, resource, counter, desc, handle);
    if (device == capture_device) { try { remember_resource(resource); } catch (...) {} }
}
void schedule_snapshot(ID3D12GraphicsCommandList* list, ID3D12Resource* source, unsigned seq, const char* label) {
    if (!source || !list) return;
    const auto desc = source->GetDesc();
    // Bounded diagnostic arena only. The much larger model allocation is excluded.
    if (desc.Width > 32ull * 1024 * 1024 || snapshots.size() >= 7) return;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC readback_desc = desc;
    readback_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    ID3D12Resource* readback = nullptr;
    const auto hr = capture_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback));
    if (FAILED(hr)) {
        log("{\"event\":\"buffer_failed\",\"hresult\":" + std::to_string(hr) + '}');
        return;
    }
    const std::string file = "arena-" + std::to_string(seq) + '-' + label + ".bin";
    snapshots.push_back({readback, desc.Width, file});
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = source;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(readback, 0, source, 0, desc.Width);
    std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
    list->ResourceBarrier(1, &barrier);
    log("{\"event\":\"buffer_scheduled\",\"sequence\":" + std::to_string(seq) +
        ",\"label\":" + quote(label) + ",\"file\":" + quote(file) + ",\"gpu_va\":" +
        std::to_string(source->GetGPUVirtualAddress()) + ",\"bytes\":" + std::to_string(desc.Width) + '}');
}
int __stdcall module_hook(void* device, const void* data, uint32_t size, uint64_t* handle) {
    const unsigned seq = sequence++;
    const std::string file = "module-" + std::to_string(seq) + ".bin";
    bool written = false;
    try { written = dump(file, data, size); } catch (...) {}
    const int status = module_original(device, data, size, handle);
    try {
        std::ostringstream s;
        s << "{\"event\":\"module\",\"sequence\":" << seq << ",\"size\":" << size
          << ",\"status\":" << status << ",\"handle\":" << (status == 0 && handle ? *handle : 0)
          << ",\"file\":" << quote(file) << ",\"written\":" << (written ? "true" : "false") << '}';
        log(s.str());
    } catch (...) {}
    return status;
}
int __stdcall function_hook(void* device, uint64_t module, const char* name, uint64_t* handle) {
    const int status = function_original(device, module, name, handle);
    try {
        std::string safe_name = name ? name : "";
        if (status == 0 && handle) {
            std::lock_guard<std::mutex> lock(io_mutex);
            names[*handle] = safe_name;
        }
        std::ostringstream s;
        s << "{\"event\":\"function\",\"module\":" << module << ",\"name\":" << quote(safe_name)
          << ",\"status\":" << status << ",\"handle\":" << (status == 0 && handle ? *handle : 0) << '}';
        log(s.str());
    } catch (...) {}
    return status;
}
int __stdcall launch_hook(void* command_list, const Kernel* kernels, uint32_t count) {
    const unsigned seq = sequence++;
    std::string first_name;
    uint64_t pre_skip_address = 0;
    try {
        if (kernels && count > 0 && count <= 256) {
            std::vector<Kernel> copy(count);
            SIZE_T read = 0;
            if (ReadProcessMemory(GetCurrentProcess(), kernels, copy.data(), count * sizeof(Kernel), &read)
                && read == count * sizeof(Kernel)) {
                for (unsigned i = 0; i < count; ++i) {
                    const auto& k = copy[i];
                    const std::string file = "launch-" + std::to_string(seq) + "-" + std::to_string(i) + ".bin";
                    const bool written = k.size <= 1024 * 1024 && dump(file, k.params, k.size);
                    std::string name;
                    { std::lock_guard<std::mutex> lock(io_mutex); name = names[k.function]; }
                    if (i == 0) {
                        first_name = name;
                        if (name == "cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8" && k.size == 264) {
                            SIZE_T copied = 0;
                            ReadProcessMemory(GetCurrentProcess(), static_cast<const char*>(k.params) + 0xd8,
                                &pre_skip_address, sizeof(pre_skip_address), &copied);
                            if (copied != sizeof(pre_skip_address)) pre_skip_address = 0;
                        }
                    }
                    std::ostringstream s;
                    s << "{\"event\":\"launch\",\"sequence\":" << seq << ",\"item\":" << i
                      << ",\"count\":" << count << ",\"name\":" << quote(name) << ",\"function\":" << k.function
                      << ",\"command_list\":" << reinterpret_cast<uintptr_t>(command_list)
                      << ",\"grid\":[" << k.grid.x << ',' << k.grid.y << ',' << k.grid.z
                      << "],\"block\":[" << k.block.x << ',' << k.block.y << ',' << k.block.z
                      << "],\"shared_bytes\":" << k.shared << ",\"parameter_bytes\":" << k.size
                      << ",\"file\":" << quote(file) << ",\"written\":" << (written ? "true" : "false") << '}';
                    log(s.str());
                }
            }
        }
    } catch (...) {}
    if (first_name == "cc_tinlayout_fused_post_block_swin_1h_32_fp8") {
        try { schedule_snapshot(static_cast<ID3D12GraphicsCommandList*>(command_list), pre_arena, seq, "before-post"); } catch (...) {}
    }
    if (count == 1 && seq == trace_first) {
        try { schedule_snapshot(static_cast<ID3D12GraphicsCommandList*>(command_list), pre_arena, seq, "before-selected"); } catch (...) {}
    }
    const int status = launch_original(command_list, kernels, count);
    if (status == 0 && count == 1 && pre_skip_address) {
        try {
            std::lock_guard<std::mutex> lock(resource_mutex);
            for (auto* resource : resources) {
                const auto address = resource->GetGPUVirtualAddress();
                if (address <= pre_skip_address && pre_skip_address - address < resource->GetDesc().Width) {
                    pre_arena = resource;
                    schedule_snapshot(static_cast<ID3D12GraphicsCommandList*>(command_list), resource, seq, "after-pre");
                    break;
                }
            }
        } catch (...) {}
    }
    // v5: at most four selected launches, with exact incoming arena state.
    if (status == 0 && count == 1 && seq >= trace_first && seq <= trace_last) {
        try { schedule_snapshot(static_cast<ID3D12GraphicsCommandList*>(command_list), pre_arena, seq, "after-selected"); } catch (...) {}
    }
    try { log("{\"event\":\"launch_result\",\"sequence\":" + std::to_string(seq) + ",\"status\":" + std::to_string(status) + '}'); } catch (...) {}
    return status;
}
void* __cdecl query_hook(unsigned id) {
    void* result = query_original(id);
    if (!result) return result;
    if (id == 0xad1a677d) { module_original = reinterpret_cast<ModuleFn>(result); return reinterpret_cast<void*>(&module_hook); }
    if (id == 0xe2436e22) { function_original = reinterpret_cast<FunctionFn>(result); return reinterpret_cast<void*>(&function_hook); }
    if (id == 0x24973538) { launch_original = reinterpret_cast<LaunchFn>(result); return reinterpret_cast<void*>(&launch_hook); }
    return result;
}
}

extern "C" __declspec(dllexport) int __cdecl NrTraceInitialize(const wchar_t* directory, ID3D12Device* device) {
    if (initialized || !directory || !device) return 0;
    try {
        output = std::filesystem::path(directory) / "nvapi-trace";
        char first[16]{},last[16]{};
        if(GetEnvironmentVariableA("CODEX_NR_TRACE_FIRST",first,sizeof(first)))trace_first=std::stoul(first);
        if(GetEnvironmentVariableA("CODEX_NR_TRACE_LAST",last,sizeof(last)))trace_last=std::stoul(last);
        if(trace_first<13 || trace_last<trace_first || trace_last-trace_first>3 || trace_last>164)return 0;
        if (std::filesystem::exists(output)) return 0;
        std::filesystem::create_directories(output);
        log_file.open(output / "events.jsonl", std::ios::out);
        if (!log_file) return 0;
        nvapi = LoadLibraryExW(L"nvapi64.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!nvapi) return 0;
        void* target = reinterpret_cast<void*>(GetProcAddress(nvapi, "nvapi_QueryInterface"));
        if (!target || MH_Initialize() != MH_OK) return 0;
        auto status = MH_CreateHook(target, reinterpret_cast<void*>(&query_hook), reinterpret_cast<void**>(&query_original));
        capture_device = device;
        device->AddRef();
        auto** vtable = *reinterpret_cast<void***>(device);
        if (status == MH_OK) status = MH_CreateHook(vtable[27], reinterpret_cast<void*>(&create_resource_hook), reinterpret_cast<void**>(&create_resource_original));
        if (status == MH_OK) status = MH_CreateHook(vtable[19], reinterpret_cast<void*>(&create_uav_hook), reinterpret_cast<void**>(&create_uav_original));
        if (status == MH_OK) status = MH_EnableHook(MH_ALL_HOOKS);
        log("{\"event\":\"initialize\",\"minhook_status\":" + std::to_string(status) + '}');
        initialized = status == MH_OK;
        return initialized ? 1 : 0;
    } catch (...) { return 0; }
}
// Called by the owned host only after its queue fence has completed.
extern "C" __declspec(dllexport) int __cdecl NrTraceFlushCompleted() {
    int written_count = 0;
    for (auto& snapshot : snapshots) {
        void* data = nullptr;
        D3D12_RANGE range{0, static_cast<SIZE_T>(snapshot.size)};
        bool written = false;
        if (SUCCEEDED(snapshot.readback->Map(0, &range, &data))) {
            try { written = dump(snapshot.file, data, snapshot.size); } catch (...) {}
            D3D12_RANGE no_writes{0, 0};
            snapshot.readback->Unmap(0, &no_writes);
        }
        log("{\"event\":\"buffer_written\",\"file\":" + quote(snapshot.file) +
            ",\"written\":" + (written ? "true" : "false") + '}');
        if (written) ++written_count;
        snapshot.readback->Release();
    }
    snapshots.clear();
    return written_count;
}
extern "C" __declspec(dllexport) void __cdecl NrTraceShutdown() {
    if (!initialized) return;
    MH_DisableHook(MH_ALL_HOOKS);
    MH_Uninitialize();
    for (auto* resource : resources) resource->Release();
    resources.clear();
    for (auto& snapshot : snapshots) snapshot.readback->Release();
    snapshots.clear();
    if (capture_device) { capture_device->Release(); capture_device = nullptr; }
    log("{\"event\":\"shutdown\",\"captured_bytes\":" + std::to_string(captured_bytes) + '}');
    log_file.close();
    initialized = false;
}
BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) DisableThreadLibraryCalls(instance);
    return TRUE;
}
