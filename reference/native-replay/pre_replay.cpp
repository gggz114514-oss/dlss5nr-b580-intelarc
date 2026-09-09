// Independent CUDA Driver API replay of the captured SM89 NR pre kernel.
// This is a reference experiment on the RTX 4060, not an Intel backend.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cuda.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void check(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    const char* name = nullptr;
    const char* text = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &text);
    throw std::runtime_error(std::string(operation) + ": " + (name ? name : "unknown") + " (" +
        std::to_string(result) + "): " + (text ? text : ""));
}
#define CUDA_CHECK(call) check((call), #call)
std::vector<char> read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Missing input file: " + path.string());
    const auto size = file.tellg();
    if (size < 0 || size > 64 * 1024 * 1024) throw std::runtime_error("Input size out of bounds");
    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0);
    file.read(data.data(), data.size());
    if (!file) throw std::runtime_error("Input read failed");
    return data;
}
void write_file(const std::filesystem::path& path, const char* data, size_t size) {
    std::ofstream file(path, std::ios::binary);
    file.write(data, size);
    if (!file) throw std::runtime_error("Output write failed");
}
template<class T> T get(const std::vector<char>& data, size_t offset) {
    if (offset + sizeof(T) > data.size()) throw std::runtime_error("Parameter range out of bounds");
    T value;
    std::memcpy(&value, data.data() + offset, sizeof(T));
    return value;
}
template<class T> void set(std::vector<char>& data, size_t offset, T value) {
    if (offset + sizeof(T) > data.size()) throw std::runtime_error("Parameter range out of bounds");
    std::memcpy(data.data() + offset, &value, sizeof(T));
}

int wmain(int argc, wchar_t** argv) {
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUarray input_array = nullptr;
    CUtexObject texture = 0;
    CUdeviceptr arena = 0, weights = 0;
    CUdevice device = 0;
    auto cleanup = [&]() {
        if (texture) cuTexObjectDestroy(texture);
        if (input_array) cuArrayDestroy(input_array);
        if (arena) cuMemFree(arena);
        if (weights) cuMemFree(weights);
        if (module) cuModuleUnload(module);
        if (context) cuDevicePrimaryCtxRelease(device);
    };
    try {
        if (argc != 3) throw std::runtime_error("Usage: pre_replay.exe INPUT_DIRECTORY NEW_OUTPUT_DIRECTORY");
        const std::filesystem::path input(argv[1]), output(argv[2]);
        if (std::filesystem::exists(output)) throw std::runtime_error("Output directory already exists");
        std::filesystem::create_directories(output);
        auto cubin = read_file(input / "pre-sm89.cubin");
        auto payload = read_file(input / "pre-parameters.bin");
        auto weight_data = read_file(input / "pre-weights.bin");
        auto rgba16 = read_file(input / "input.rgba16.bin");
        if (payload.size() != 264 || weight_data.size() != 21696 || rgba16.size() != 256 * 256 * 8)
            throw std::runtime_error("Wrong fixed reference input sizes");
        for (auto offset : {0xd0u, 0xd4u}) if (get<uint32_t>(payload, offset) != 256) throw std::runtime_error("Wrong input dimensions");
        for (auto offset : {0xf0u, 0xf4u}) if (get<uint32_t>(payload, offset) != 320) throw std::runtime_error("Wrong padded dimensions");
        for (auto offset : {0x100u, 0x104u}) if (get<uint32_t>(payload, offset) != 160) throw std::runtime_error("Wrong downsample dimensions");
        // Only the first texture is bound in the captured Reset=1 contract.
        for (size_t offset = 8; offset < 0x90; offset += 8)
            if (get<uint64_t>(payload, offset)) throw std::runtime_error("Unexpected additional texture bindings");
        CUDA_CHECK(cuInit(0));
        CUDA_CHECK(cuDeviceGet(&device, 0));
        char device_name[256]{};
        CUDA_CHECK(cuDeviceGetName(device_name, sizeof(device_name), device));
        int major = 0, minor = 0;
        CUDA_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
        CUDA_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
        std::cout << "GPU: " << device_name << " SM " << major << minor << std::endl;
        if (major != 8 || minor != 9 || std::string(device_name).find("4060") == std::string::npos)
            throw std::runtime_error("This recorded replay requires the RTX 4060 reference GPU");
        CUDA_CHECK(cuDevicePrimaryCtxRetain(&context, device));
        CUDA_CHECK(cuCtxSetCurrent(context));
        CUDA_CHECK(cuModuleLoadData(&module, cubin.data()));
        CUfunction function = nullptr;
        CUDA_CHECK(cuModuleGetFunction(&function, module, "cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8"));
        int registers = 0, static_shared = 0;
        CUDA_CHECK(cuFuncGetAttribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, function));
        CUDA_CHECK(cuFuncGetAttribute(&static_shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function));
        std::cout << "Kernel loaded: registers=" << registers << " static_shared=" << static_shared << std::endl;

        constexpr size_t arena_size = 15711232;
        constexpr size_t skip_offset = 0x16c00, skip_size = 320 * 320 * 32;
        constexpr size_t down_offset = 0x336c00, down_size = 160 * 160 * 32;
        CUDA_CHECK(cuMemAlloc(&arena, arena_size));
        CUDA_CHECK(cuMemsetD8(arena, 0, arena_size));
        CUDA_CHECK(cuMemAlloc(&weights, 256 * 1024));
        CUDA_CHECK(cuMemsetD8(weights, 0, 256 * 1024));
        CUDA_CHECK(cuMemcpyHtoD(weights, weight_data.data(), weight_data.size()));
        CUDA_ARRAY_DESCRIPTOR array_desc{};
        array_desc.Width = 256;
        array_desc.Height = 256;
        array_desc.Format = CU_AD_FORMAT_HALF;
        array_desc.NumChannels = 4;
        CUDA_CHECK(cuArrayCreate(&input_array, &array_desc));
        CUDA_MEMCPY2D copy{};
        copy.srcMemoryType = CU_MEMORYTYPE_HOST;
        copy.srcHost = rgba16.data();
        copy.srcPitch = 256 * 8;
        copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copy.dstArray = input_array;
        copy.WidthInBytes = 256 * 8;
        copy.Height = 256;
        CUDA_CHECK(cuMemcpy2D(&copy));
        CUDA_RESOURCE_DESC resource_desc{};
        resource_desc.resType = CU_RESOURCE_TYPE_ARRAY;
        resource_desc.res.array.hArray = input_array;
        CUDA_TEXTURE_DESC texture_desc{};
        texture_desc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
        texture_desc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
        texture_desc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
        texture_desc.filterMode = CU_TR_FILTER_MODE_POINT;
        texture_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;
        CUDA_CHECK(cuTexObjectCreate(&texture, &resource_desc, &texture_desc, nullptr));
        set<uint64_t>(payload, 0x00, texture);
        set<uint64_t>(payload, 0xd8, arena + skip_offset);
        set<uint64_t>(payload, 0xe0, weights);
        set<uint64_t>(payload, 0xf8, arena + down_offset);
        size_t parameter_size = payload.size();
        void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, payload.data(), CU_LAUNCH_PARAM_BUFFER_SIZE,
                        &parameter_size, CU_LAUNCH_PARAM_END};
        std::cout << "Launching captured pre kernel (40x40 blocks, 32 threads)..." << std::endl;
        CUDA_CHECK(cuLaunchKernel(function, 40, 40, 1, 32, 1, 1, 0, nullptr, nullptr, extra));
        CUDA_CHECK(cuCtxSynchronize());
        std::vector<char> result(arena_size);
        CUDA_CHECK(cuMemcpyDtoH(result.data(), arena, result.size()));
        write_file(output / "arena.bin", result.data(), result.size());
        write_file(output / "pre_skip.raw", result.data() + skip_offset, skip_size);
        write_file(output / "pre_down_candidate.raw", result.data() + down_offset, down_size);
        write_file(output / "bound-parameters.bin", payload.data(), payload.size());
        std::ofstream metadata(output / "replay.json");
        metadata << "{\"cuda_success\":true,\"kernel_registers\":" << registers << ",\"kernel_static_shared\":" << static_shared
            << ",\"texture_filter\":\"point\",\"texture_normalized_coordinates\":true,\"native_equivalence_validated\":false}\n";
        if (!metadata) throw std::runtime_error("Metadata write failed");
        cleanup();
        std::cout << "Replay completed; raw outputs require independent comparison." << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        cleanup();
        return 1;
    }
}
