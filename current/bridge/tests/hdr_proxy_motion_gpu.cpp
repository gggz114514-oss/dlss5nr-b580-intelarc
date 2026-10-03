#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cwchar>
#include <cstring>
#include <stdexcept>
#include <string>

#include "nr_hdr_proxy.h"

using Microsoft::WRL::ComPtr;

namespace {
constexpr UINT kWidth = 1280;
constexpr UINT kHeight = 720;

void require(HRESULT hr, const char* operation, ID3D12Device* device = nullptr) {
    if (SUCCEEDED(hr)) return;
    char message[256]{};
    const HRESULT removed = device ? device->GetDeviceRemovedReason() : S_OK;
    sprintf_s(message, "%s failed: HRESULT=0x%08X device_removed=0x%08X",
              operation, static_cast<unsigned>(hr), static_cast<unsigned>(removed));
    throw std::runtime_error(message);
}

ComPtr<ID3D12Resource> make_texture(ID3D12Device* device, UINT width,
                                    UINT height, DXGI_FORMAT format) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    ComPtr<ID3D12Resource> resource;
    require(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
        IID_PPV_ARGS(resource.GetAddressOf())), "CreateCommittedResource", device);
    return resource;
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource,
                                   D3D12_RESOURCE_STATES before,
                                   D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    return barrier;
}

float half_to_float(uint16_t h) {
    const bool negative = (h & 0x8000u) != 0;
    const unsigned exponent = (h >> 10) & 0x1fu;
    const unsigned mantissa = h & 0x03ffu;
    float value;
    if (exponent == 0) {
        value = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exponent == 31) {
        value = mantissa ? NAN : INFINITY;
    } else {
        value = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f,
                           static_cast<int>(exponent) - 15);
    }
    return negative ? -value : value;
}

ComPtr<IDXGIAdapter> find_b580(IDXGIFactory6* factory, std::wstring& name) {
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> candidate;
        const HRESULT hr = factory->EnumAdapters1(index, candidate.GetAddressOf());
        if (hr == DXGI_ERROR_NOT_FOUND) break;
        require(hr, "EnumAdapters1");
        DXGI_ADAPTER_DESC1 desc{};
        require(candidate->GetDesc1(&desc), "GetDesc1");
        name = desc.Description;
        if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            name.find(L"B580") != std::wstring::npos) {
            ComPtr<IDXGIAdapter> base;
            require(candidate.As(&base), "QueryInterface IDXGIAdapter");
            return base;
        }
    }
    return {};
}

void run_case(ID3D12Device* device, ID3D12CommandQueue* queue,
              ID3D12Resource* source, ID3D12Resource* motion,
              ID3D12Resource* xess_input, const std::wstring& shader_path,
              bool use_game_motion, D3D12_RESOURCE_STATES motion_state) {
    nrb::HdrProxy proxy;
    if (!proxy.initialize(device, source, motion, xess_input, shader_path))
        throw std::runtime_error("HdrProxy initialization/prepare+composite shader compilation failed");
    if (!proxy.game_motion_available())
        throw std::runtime_error("RG16F game-motion input was rejected by HdrProxy");

    ComPtr<ID3D12DescriptorHeap> clear_heap;
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = 2;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    require(device->CreateDescriptorHeap(&heap_desc,
        IID_PPV_ARGS(clear_heap.GetAddressOf())), "CreateDescriptorHeap", device);
    const UINT stride = device->GetDescriptorHandleIncrementSize(heap_desc.Type);
    auto cpu = clear_heap->GetCPUDescriptorHandleForHeapStart();
    auto gpu = clear_heap->GetGPUDescriptorHandleForHeapStart();
    D3D12_UNORDERED_ACCESS_VIEW_DESC source_uav{};
    source_uav.Format = DXGI_FORMAT_R11G11B10_FLOAT;
    source_uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(source, nullptr, &source_uav, cpu);
    D3D12_UNORDERED_ACCESS_VIEW_DESC motion_uav{};
    motion_uav.Format = DXGI_FORMAT_R16G16_FLOAT;
    motion_uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    cpu.ptr += stride;
    device->CreateUnorderedAccessView(motion, nullptr, &motion_uav, cpu);
    gpu.ptr += stride;

    const auto motion_desc = proxy.nr_motion()->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0, total_bytes = 0;
    device->GetCopyableFootprints(&motion_desc, 0, 1, 0, &footprint, &rows,
                                  &row_bytes, &total_bytes);
    if (rows != kHeight || row_bytes != UINT64(kWidth) * 4 || !total_bytes)
        throw std::runtime_error("Unexpected RG16F readback footprint");
    D3D12_RESOURCE_DESC readback_desc{};
    readback_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    readback_desc.Width = total_bytes;
    readback_desc.Height = 1;
    readback_desc.DepthOrArraySize = 1;
    readback_desc.MipLevels = 1;
    readback_desc.SampleDesc.Count = 1;
    readback_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_HEAP_PROPERTIES readback_heap{};
    readback_heap.Type = D3D12_HEAP_TYPE_READBACK;
    ComPtr<ID3D12Resource> readback;
    require(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE,
        &readback_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(readback.GetAddressOf())), "Create readback buffer", device);

    ComPtr<ID3D12CommandAllocator> allocator;
    require(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(allocator.GetAddressOf())), "CreateCommandAllocator", device);
    ComPtr<ID3D12GraphicsCommandList> list;
    require(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        allocator.Get(), nullptr, IID_PPV_ARGS(list.GetAddressOf())),
        "CreateCommandList", device);

    ID3D12DescriptorHeap* heaps[] = {clear_heap.Get()};
    list->SetDescriptorHeaps(1, heaps);
    const float source_clear[4] = {0.25f, 0.5f, 0.75f, 1.0f};
    const float motion_clear[4] = {0.01f, -0.02f, 0.0f, 0.0f};
    auto source_cpu = clear_heap->GetCPUDescriptorHandleForHeapStart();
    auto source_gpu = clear_heap->GetGPUDescriptorHandleForHeapStart();
    list->ClearUnorderedAccessViewFloat(source_gpu, source_cpu, source,
                                        source_clear, 0, nullptr);
    if (motion_state == D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) {
        const auto to_uav = transition(motion, motion_state,
                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->ResourceBarrier(1, &to_uav);
    }
    list->ClearUnorderedAccessViewFloat(gpu, cpu, motion, motion_clear, 0, nullptr);
    const auto motion_ready = transition(motion,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    list->ResourceBarrier(1, &motion_ready);

    proxy.record_prepare(list.Get(), source, use_game_motion, 1280.0f, 720.0f);
    const auto output_copy = transition(proxy.nr_motion(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &output_copy);
    D3D12_TEXTURE_COPY_LOCATION src_location{};
    src_location.pResource = proxy.nr_motion();
    src_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src_location.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION dst_location{};
    dst_location.pResource = readback.Get();
    dst_location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst_location.PlacedFootprint = footprint;
    list->CopyTextureRegion(&dst_location, 0, 0, 0, &src_location, nullptr);
    require(list->Close(), "Close command list", device);

    ID3D12CommandList* submitted[] = {list.Get()};
    queue->ExecuteCommandLists(1, submitted);
    ComPtr<ID3D12Fence> fence;
    require(device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(fence.GetAddressOf())), "CreateFence", device);
    require(queue->Signal(fence.Get(), 1), "Signal", device);
    HANDLE event_handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event_handle) throw std::runtime_error("CreateEvent failed");
    const HRESULT wait_hr = fence->SetEventOnCompletion(1, event_handle);
    if (FAILED(wait_hr)) {
        CloseHandle(event_handle);
        require(wait_hr, "SetEventOnCompletion", device);
    }
    const DWORD wait = WaitForSingleObject(event_handle, 30000);
    CloseHandle(event_handle);
    if (wait != WAIT_OBJECT_0)
        throw std::runtime_error("GPU completion fence timed out");

    void* mapped = nullptr;
    D3D12_RANGE read_range{0, static_cast<SIZE_T>(total_bytes)};
    require(readback->Map(0, &read_range, &mapped), "Map readback", device);
    const auto* bytes = static_cast<const uint8_t*>(mapped);
    uint16_t first_half[2]{};
    std::memcpy(first_half, bytes + footprint.Offset, sizeof(first_half));
    const float first_x = half_to_float(first_half[0]);
    const float first_y = half_to_float(first_half[1]);
    uint64_t failures = 0;
    float max_x_error = 0.0f, max_y_error = 0.0f;
    for (UINT y = 0; y < kHeight; ++y) {
        const auto* row = bytes + footprint.Offset +
            SIZE_T(y) * footprint.Footprint.RowPitch;
        for (UINT x = 0; x < kWidth; ++x) {
            uint16_t half[2]{};
            std::memcpy(half, row + SIZE_T(x) * 4, sizeof(half));
            const float actual_x = half_to_float(half[0]);
            const float actual_y = half_to_float(half[1]);
            if (use_game_motion) {
                const float error_x = std::fabs(actual_x - 12.8f);
                const float error_y = std::fabs(actual_y + 14.4f);
                if (error_x > max_x_error) max_x_error = error_x;
                if (error_y > max_y_error) max_y_error = error_y;
                if (!std::isfinite(actual_x) || !std::isfinite(actual_y) ||
                    error_x > 0.02f || error_y > 0.02f) ++failures;
            } else if (actual_x != 0.0f || actual_y != 0.0f) {
                ++failures;
            }
        }
    }
    const D3D12_RANGE no_writes{0, 0};
    readback->Unmap(0, &no_writes);
    if (failures) {
        char message[256]{};
        sprintf_s(message, "%s motion: %llu/%llu pixels failed; max_error=(%.6f,%.6f)",
            use_game_motion ? "scaled" : "zero-mode",
            static_cast<unsigned long long>(failures),
            static_cast<unsigned long long>(kWidth) * kHeight,
            max_x_error, max_y_error);
        throw std::runtime_error(message);
    }
    std::printf("%s: all %u x %u pixels passed; first=(%.6f, %.6f) max_error=(%.6f, %.6f)\n",
        use_game_motion ? "scaled motion" : "zero-motion mode",
        kWidth, kHeight, first_x, first_y, max_x_error, max_y_error);
}
}

int wmain(int argc, wchar_t** argv) {
    if (argc != 2) {
        std::fwprintf(stderr, L"usage: hdr_proxy_motion_gpu.exe <nr_hdr_proxy.hlsl>\n");
        return 2;
    }
    try {
        ComPtr<IDXGIFactory6> factory;
        require(CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf())),
                "CreateDXGIFactory1");
        std::wstring adapter_name;
        auto adapter = find_b580(factory.Get(), adapter_name);
        if (!adapter) {
            std::fwprintf(stderr, L"SKIP: Intel Arc B580 adapter not found\n");
            return 77;
        }
        ComPtr<ID3D12Device> device;
        require(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
            IID_PPV_ARGS(device.GetAddressOf())), "D3D12CreateDevice");
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        ComPtr<ID3D12CommandQueue> queue;
        require(device->CreateCommandQueue(&queue_desc,
            IID_PPV_ARGS(queue.GetAddressOf())), "CreateCommandQueue", device.Get());
        auto source = make_texture(device.Get(), kWidth, kHeight,
                                  DXGI_FORMAT_R11G11B10_FLOAT);
        auto motion = make_texture(device.Get(), kWidth, kHeight,
                                  DXGI_FORMAT_R16G16_FLOAT);
        auto xess_input = make_texture(device.Get(), kWidth, kHeight,
                                       DXGI_FORMAT_R11G11B10_FLOAT);
        std::printf("adapter: Intel Arc B580; testing HdrProxy prepare path\n");
        const std::wstring shader_path = argv[1];
        run_case(device.Get(), queue.Get(), source.Get(), motion.Get(),
                 xess_input.Get(), shader_path, true,
                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        run_case(device.Get(), queue.Get(), source.Get(), motion.Get(),
                 xess_input.Get(), shader_path, false,
                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        std::puts("PASS: HdrProxy shader compile, scaled motion, and zero-motion mode");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
