#include "nr_hdr_proxy.h"
#include <d3dcompiler.h>
#include <array>
#include <bit>
#include <algorithm>
#include <limits>
#include <mutex>
#include <bcrypt.h>
#pragma comment(lib,"bcrypt.lib")

namespace nrb {
namespace {
using Microsoft::WRL::ComPtr;
constexpr UINT hdr_compile_flags=D3DCOMPILE_OPTIMIZATION_LEVEL3|D3DCOMPILE_IEEE_STRICTNESS;
constexpr uint32_t hdr_root_abi_version=1;
bool serialize_root(ComPtr<ID3DBlob>& blob,HdrProxyCacheStats* stats) {
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0]={D3D12_DESCRIPTOR_RANGE_TYPE_SRV,2,0,0,0};
    ranges[1]={D3D12_DESCRIPTOR_RANGE_TYPE_UAV,3,0,0,0};
    D3D12_ROOT_PARAMETER params[3]{};
    for(unsigned i=0;i<2;i++) {
        params[i].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[i].DescriptorTable={1,&ranges[i]};
    }
    params[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants={0,0,5};
    D3D12_ROOT_SIGNATURE_DESC rs{};
    rs.NumParameters=3;rs.pParameters=params;
    ComPtr<ID3DBlob> errors;
    if(stats) ++stats->root_serializations;
    return SUCCEEDED(D3D12SerializeRootSignature(&rs,D3D_ROOT_SIGNATURE_VERSION_1,
        blob.GetAddressOf(),errors.GetAddressOf()));
}
bool create_root(ID3D12Device* device,ID3DBlob* blob,
                 ComPtr<ID3D12RootSignature>& root,HdrProxyCacheStats* stats) {
    if(FAILED(device->CreateRootSignature(0,blob->GetBufferPointer(),
        blob->GetBufferSize(),IID_PPV_ARGS(root.GetAddressOf())))) return false;
    if(stats) ++stats->root_signature_creates;
    return true;
}
bool create_pso(ID3D12Device* device,ID3D12RootSignature* root,ID3DBlob* code,
                ComPtr<ID3D12PipelineState>& pipeline,HdrProxyCacheStats* stats) {
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature=root;
    desc.CS={code->GetBufferPointer(),code->GetBufferSize()};
    if(FAILED(device->CreateComputePipelineState(&desc,
        IID_PPV_ARGS(pipeline.GetAddressOf())))) return false;
    if(stats) ++stats->pso_creates;
    return true;
}
// SHA256 is a cold-load/audit operation only. No hashing in cached_acquire.
bool sha256(ID3DBlob* blob,std::string& result) {
    if(blob->GetBufferSize()>std::numeric_limits<ULONG>::max()) return false;
    BCRYPT_ALG_HANDLE algorithm=nullptr;
    if(BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0)
        return false;
    struct AlgorithmOwner {
        BCRYPT_ALG_HANDLE handle;
        ~AlgorithmOwner() {BCryptCloseAlgorithmProvider(handle,0);}
    } owner{algorithm};
    DWORD object_size=0,written=0;
    if(BCryptGetProperty(algorithm,BCRYPT_OBJECT_LENGTH,
        reinterpret_cast<PUCHAR>(&object_size),sizeof(object_size),&written,0)<0)
        return false;
    std::vector<UCHAR> object(object_size);
    BCRYPT_HASH_HANDLE hash=nullptr;
    if(BCryptCreateHash(algorithm,&hash,object.data(),object_size,nullptr,0,0)<0)
        return false;
    struct HashOwner {
        BCRYPT_HASH_HANDLE handle;
        ~HashOwner() {BCryptDestroyHash(handle);}
    } hash_owner{hash};
    std::array<UCHAR,32> digest{};
    if(BCryptHashData(hash,static_cast<PUCHAR>(blob->GetBufferPointer()),
        static_cast<ULONG>(blob->GetBufferSize()),0)<0 ||
       BCryptFinishHash(hash,digest.data(),static_cast<ULONG>(digest.size()),0)<0)
        return false;
    constexpr char hex[]="0123456789abcdef";
    result.resize(64);
    for(size_t i=0;i<digest.size();++i) {
        result[2*i]=hex[digest[i]>>4];result[2*i+1]=hex[digest[i]&15];
    }
    return true;
}
bool utf8(const std::wstring& input,std::string& output) {
    if(input.size()>size_t(std::numeric_limits<int>::max())) return false;
    const int size=WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,input.data(),
        static_cast<int>(input.size()),nullptr,0,nullptr,nullptr);
    if(size<=0) return false;
    output.resize(size_t(size));
    return WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,input.data(),
        static_cast<int>(input.size()),output.data(),size,nullptr,nullptr)==size;
}
bool absolute_path(const std::wstring& input,std::wstring& output) {
    const DWORD size=GetFullPathNameW(input.c_str(),0,nullptr,nullptr);
    if(!size) return false;
    std::vector<wchar_t> buffer(size);
    const DWORD written=GetFullPathNameW(input.c_str(),size,buffer.data(),nullptr);
    if(!written || written>=size) return false;
    output.assign(buffer.data(),written);
    return true;
}
D3D12_RESOURCE_BARRIER transition(ID3D12Resource* r,D3D12_RESOURCE_STATES before,
                                  D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition={r,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after};
    return b;
}
ComPtr<ID3D12Resource> texture(ID3D12Device* device,UINT width,UINT height,
                               DXGI_FORMAT format) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width=width;desc.Height=height;desc.DepthOrArraySize=1;
    desc.MipLevels=1;desc.SampleDesc.Count=1;desc.Format=format;
    desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;
    ComPtr<ID3D12Resource> result;
    if(FAILED(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,
        &desc,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,
        IID_PPV_ARGS(result.GetAddressOf())))) result.Reset();
    return result;
}
void srv(ID3D12Device* device,ID3D12Resource* resource,DXGI_FORMAT format,
         D3D12_CPU_DESCRIPTOR_HANDLE handle) {
    D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.Format=format;desc.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;
    desc.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    desc.Texture2D.MipLevels=1;
    device->CreateShaderResourceView(resource,&desc,handle);
}
void uav(ID3D12Device* device,ID3D12Resource* resource,DXGI_FORMAT format,
         D3D12_CPU_DESCRIPTOR_HANDLE handle) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
    desc.Format=format;desc.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(resource,nullptr,&desc,handle);
}
bool make_pipeline(ID3D12Device* device,ID3D12RootSignature* root,
                   const std::wstring& path,const char* entry,
                   ComPtr<ID3D12PipelineState>& pipeline,HdrProxyCacheStats* stats) {
    ComPtr<ID3DBlob> code,errors;
    if(stats) {++stats->shader_compile_calls;++stats->source_reads;}
    if(FAILED(D3DCompileFromFile(path.c_str(),nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,entry,"cs_5_0",
        hdr_compile_flags,0,code.GetAddressOf(),errors.GetAddressOf()))) {
        if(stats) ++stats->shader_compile_failures;
        return false;
    }
    return create_pso(device,root,code.Get(),pipeline,stats);
}
bool uncached_bundle(ID3D12Device* device,const std::wstring& path,
    ComPtr<ID3D12RootSignature>& root,ComPtr<ID3D12PipelineState>& prepare,
    ComPtr<ID3D12PipelineState>& composite,HdrProxyCacheStats* stats) {
    ComPtr<ID3DBlob> blob;
    ComPtr<ID3D12RootSignature> new_root;
    ComPtr<ID3D12PipelineState> new_prepare,new_composite;
    if(!serialize_root(blob,stats) || !create_root(device,blob.Get(),new_root,stats) ||
       !make_pipeline(device,new_root.Get(),path,"prepare",new_prepare,stats) ||
       !make_pipeline(device,new_root.Get(),path,"composite",new_composite,stats))
        return false;
    root=std::move(new_root);prepare=std::move(new_prepare);composite=std::move(new_composite);
    return true;
}
}

struct HdrProxyCache::Impl {
    struct Program {
        // Blobs never leave Impl except as caller-owned audit copies.
        ComPtr<ID3DBlob> prepare,composite,root;
        HdrProxyShaderIdentity identity;
        std::wstring request_path;
    };
    struct DeviceEntry {
        // Strong canonical COM identity prevents pointer-reuse collisions.
        ComPtr<IUnknown> identity;
        ComPtr<ID3D12Device> device;
        ComPtr<ID3D12RootSignature> root;
        ComPtr<ID3D12PipelineState> prepare,composite;
        uint64_t generation=0,last_use=0;
    };
    mutable std::mutex mutex;
    HdrProxyCacheStats counters;
    std::unique_ptr<Program> program;
    std::vector<DeviceEntry> devices;
    size_t max_devices;
    uint64_t generation=1,use_serial=0;
    explicit Impl(size_t capacity):max_devices(std::clamp(capacity,size_t(1),size_t(4))) {
        devices.reserve(max_devices);
    }
    void invalidate() {
        devices.clear();program.reset();++generation;++counters.invalidations;
    }
    void forget(IUnknown* identity) {
        std::erase_if(devices,[&](const DeviceEntry& entry) {
            return entry.identity.Get()==identity;
        });
    }
    bool ensure_shader(const std::wstring& path) {
        if(path.empty() || path.find(L'\0')!=std::wstring::npos) return false;
        if(program) {
            // Explicit generation changes, not frame-time disk polling.
            if(program->request_path!=path) return false;
            ++counters.bytecode_hits;
            return true;
        }
        auto candidate=std::make_unique<Program>();
        candidate->request_path=path;
        if(!absolute_path(path,candidate->identity.shader_path)) return false;
        std::string source_name;
        if(!utf8(candidate->identity.shader_path,source_name)) return false;
        ComPtr<ID3DBlob> source;
        ++counters.source_reads;
        if(FAILED(D3DReadFileToBlob(candidate->identity.shader_path.c_str(),
            source.GetAddressOf()))) return false;
        // Bound this fixed standalone shader's cold memory footprint.
        if(!source->GetBufferSize() || source->GetBufferSize()>1024*1024) return false;
        ++counters.source_hashes;
        if(!sha256(source.Get(),candidate->identity.source_sha256)) return false;
        const auto compile=[&](const char* entry,ComPtr<ID3DBlob>& code) {
            ComPtr<ID3DBlob> errors;
            ++counters.shader_compile_calls;
            // Compile the exact hashed snapshot. Includes are unsupported so
            // no untracked dependency can silently survive a generation.
            const HRESULT result=D3DCompile(source->GetBufferPointer(),source->GetBufferSize(),
                source_name.c_str(),nullptr,nullptr,entry,"cs_5_0",hdr_compile_flags,0,
                code.GetAddressOf(),errors.GetAddressOf());
            if(FAILED(result)) ++counters.shader_compile_failures;
            return SUCCEEDED(result);
        };
        if(!compile("prepare",candidate->prepare) || !compile("composite",candidate->composite) ||
           !serialize_root(candidate->root,&counters) ||
           !sha256(candidate->prepare.Get(),candidate->identity.prepare_sha256) ||
           !sha256(candidate->composite.Get(),candidate->identity.composite_sha256) ||
           !sha256(candidate->root.Get(),candidate->identity.root_signature_sha256)) return false;
        candidate->identity.compile_flags=hdr_compile_flags;
        candidate->identity.target="cs_5_0";
        candidate->identity.root_abi_version=hdr_root_abi_version;
        candidate->identity.shader_generation=generation;
        program=std::move(candidate); // Only a complete immutable pair is published.
        return true;
    }
    bool cached_acquire(ID3D12Device* device,const std::wstring& path,
        ComPtr<ID3D12RootSignature>& root,ComPtr<ID3D12PipelineState>& prepare,
        ComPtr<ID3D12PipelineState>& composite) {
        if(!device) return false;
        ComPtr<IUnknown> identity;
        if(FAILED(device->QueryInterface(IID_PPV_ARGS(identity.GetAddressOf())))) return false;
        if(FAILED(device->GetDeviceRemovedReason())) {
            forget(identity.Get());++counters.device_loss_failures;return false;
        }
        if(!ensure_shader(path)) return false;
        for(auto& entry:devices) {
            if(entry.identity.Get()!=identity.Get() || entry.generation!=generation) continue;
            ++counters.pipeline_hits;entry.last_use=++use_serial;
            root=entry.root;prepare=entry.prepare;composite=entry.composite;
            return true;
        }
        ++counters.pipeline_misses;
        DeviceEntry candidate;
        candidate.identity=identity;candidate.device=device;candidate.generation=generation;
        if(!create_root(device,program->root.Get(),candidate.root,&counters) ||
           !create_pso(device,candidate.root.Get(),program->prepare.Get(),candidate.prepare,&counters) ||
           !create_pso(device,candidate.root.Get(),program->composite.Get(),candidate.composite,&counters)) {
            if(FAILED(device->GetDeviceRemovedReason())) {
                forget(identity.Get());++counters.device_loss_failures;
            }
            return false;
        }
        if(FAILED(device->GetDeviceRemovedReason())) {
            forget(identity.Get());++counters.device_loss_failures;return false;
        }
        candidate.last_use=++use_serial;
        // Eviction drops cache refs only. Pending/HdrProxy retains its own
        // immutable COM refs until the original completion retirement.
        if(devices.size()==max_devices) {
            const auto victim=std::min_element(devices.begin(),devices.end(),
                [](const DeviceEntry& a,const DeviceEntry& b) {return a.last_use<b.last_use;});
            *victim=std::move(candidate);++counters.device_evictions;
            root=victim->root;prepare=victim->prepare;composite=victim->composite;
        } else {
            devices.push_back(std::move(candidate));
            root=devices.back().root;prepare=devices.back().prepare;composite=devices.back().composite;
        }
        return true;
    }
};
HdrProxyCache::HdrProxyCache(size_t max_devices):impl_(std::make_unique<Impl>(max_devices)) {}
HdrProxyCache::~HdrProxyCache()=default;
void HdrProxyCache::set_enabled(bool enabled) {
    std::lock_guard guard(impl_->mutex);impl_->counters.enabled=enabled;
}
bool HdrProxyCache::enabled() const {
    std::lock_guard guard(impl_->mutex);return impl_->counters.enabled;
}
bool HdrProxyCache::warmup_shader(const std::wstring& path) {
    std::lock_guard guard(impl_->mutex);
    try {if(impl_->ensure_shader(path)) return true;} catch(...) {}
    ++impl_->counters.failures;return false;
}
bool HdrProxyCache::warmup(ID3D12Device* device,const std::wstring& path) {
    std::lock_guard guard(impl_->mutex);
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> prepare,composite;
    try {if(impl_->cached_acquire(device,path,root,prepare,composite)) return true;} catch(...) {}
    ++impl_->counters.failures;return false;
}
bool HdrProxyCache::reload_shader(const std::wstring& path) {
    std::lock_guard guard(impl_->mutex);impl_->invalidate();
    try {if(impl_->ensure_shader(path)) return true;} catch(...) {}
    ++impl_->counters.failures;return false;
}
void HdrProxyCache::invalidate_shader() {
    std::lock_guard guard(impl_->mutex);impl_->invalidate();
}
void HdrProxyCache::forget_device(ID3D12Device* device) {
    if(!device) return;
    ComPtr<IUnknown> identity;
    if(FAILED(device->QueryInterface(IID_PPV_ARGS(identity.GetAddressOf())))) return;
    std::lock_guard guard(impl_->mutex);impl_->forget(identity.Get());
}
HdrProxyCacheStats HdrProxyCache::stats() const {
    std::lock_guard guard(impl_->mutex);
    auto result=impl_->counters;result.shader_generation=impl_->generation;
    result.shader_ready=bool(impl_->program);result.resident_device_entries=impl_->devices.size();
    return result;
}
HdrProxyShaderIdentity HdrProxyCache::shader_identity() const {
    std::lock_guard guard(impl_->mutex);
    return impl_->program?impl_->program->identity:HdrProxyShaderIdentity{};
}
bool HdrProxyCache::copy_shader_bytecode(std::vector<uint8_t>& prepare,
                                       std::vector<uint8_t>& composite) const {
    std::lock_guard guard(impl_->mutex);
    if(!impl_->program) {prepare.clear();composite.clear();return false;}
    const auto copy=[](ID3DBlob* blob,std::vector<uint8_t>& out) {
        const auto* bytes=static_cast<const uint8_t*>(blob->GetBufferPointer());
        out.assign(bytes,bytes+blob->GetBufferSize());
    };
    copy(impl_->program->prepare.Get(),prepare);copy(impl_->program->composite.Get(),composite);
    return true;
}
bool HdrProxyCache::acquire(ID3D12Device* device,const std::wstring& path,
    ComPtr<ID3D12RootSignature>& root,ComPtr<ID3D12PipelineState>& prepare,
    ComPtr<ID3D12PipelineState>& composite) {
    std::lock_guard guard(impl_->mutex);
    try {
        if(impl_->counters.enabled) {
            if(impl_->cached_acquire(device,path,root,prepare,composite)) return true;
        } else {
            ++impl_->counters.uncached_initializations;
            if(uncached_bundle(device,path,root,prepare,composite,&impl_->counters)) return true;
        }
    } catch(...) {}
    ++impl_->counters.failures;return false;
}
D3D12_CPU_DESCRIPTOR_HANDLE HdrProxy::cpu(UINT index) const {
    auto handle=heap_->GetCPUDescriptorHandleForHeapStart();
    handle.ptr+=SIZE_T(index)*stride_;
    return handle;
}
D3D12_GPU_DESCRIPTOR_HANDLE HdrProxy::gpu(UINT index) const {
    auto handle=heap_->GetGPUDescriptorHandleForHeapStart();
    handle.ptr+=UINT64(index)*stride_;
    return handle;
}
bool HdrProxy::initialize(ID3D12Device* device,ID3D12Resource* source,
    ID3D12Resource* motion,ID3D12Resource* xess_input,
    const std::wstring& shader_path) {
    return initialize(device,source,motion,xess_input,shader_path,nullptr);
}
bool HdrProxy::initialize(ID3D12Device* device,ID3D12Resource* source,
    ID3D12Resource* motion,ID3D12Resource* xess_input,
    const std::wstring& shader_path,HdrProxyCache* cache) {
    return initialize(device,source,motion,xess_input,shader_path,cache,nullptr);
}
bool HdrProxy::initialize(ID3D12Device* device,ID3D12Resource* source,
    ID3D12Resource* motion,ID3D12Resource* xess_input,
    const std::wstring& shader_path,HdrProxyCache* cache,HdrProxyResourcePool* pool) {
    if(!device || !source || !xess_input || shader_path.empty()) return false;
    // A frame-owned heap/texture set must never be rebound in flight. On
    // partial failure discard the instance as the existing caller does.
    if(device_) return false;
    const auto s=source->GetDesc(),d=xess_input->GetDesc();
    if(s.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
       s.Width!=1280 || s.Height!=720 ||
       s.Format!=DXGI_FORMAT_R11G11B10_FLOAT ||
       d.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
       d.Width!=s.Width || d.Height!=s.Height ||
       d.Format!=s.Format ||
       !(d.Flags&D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) return false;
    D3D12_FEATURE_DATA_FORMAT_SUPPORT support{};
    support.Format=DXGI_FORMAT_R11G11B10_FLOAT;
    if(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT,
        &support,sizeof(support))) ||
       !(support.Support2&D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE)) return false;
    device_=device;width_=UINT(s.Width);height_=s.Height;
    if(motion) {
        const auto m=motion->GetDesc();
        ComPtr<ID3D12Device> motion_device;
        game_motion_available_=m.Dimension==D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
            m.Width==s.Width && m.Height==s.Height &&
            m.DepthOrArraySize==1 && m.MipLevels==1 && m.SampleDesc.Count==1 &&
            m.Format==DXGI_FORMAT_R16G16_FLOAT &&
            SUCCEEDED(motion->GetDevice(IID_PPV_ARGS(motion_device.GetAddressOf()))) &&
            motion_device.Get()==device;
    }

    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;hd.NumDescriptors=5;
    hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if(pool && pool->enabled()) {
        ComPtr<IUnknown> identity;
        const auto same_owner=[&](ID3D12Resource* resource) {
            ComPtr<ID3D12Device> owner;ComPtr<IUnknown> canonical;
            return SUCCEEDED(resource->GetDevice(IID_PPV_ARGS(owner.GetAddressOf()))) &&
                SUCCEEDED(owner->QueryInterface(IID_PPV_ARGS(canonical.GetAddressOf()))) &&
                canonical.Get()==identity.Get();
        };
        const auto plain=[](const D3D12_RESOURCE_DESC& desc) {
            return desc.DepthOrArraySize==1 && desc.MipLevels==1 &&
                desc.SampleDesc.Count==1 && desc.SampleDesc.Quality==0;
        };
        if(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(identity.GetAddressOf()))) &&
           plain(s) && plain(d) && same_owner(source) && same_owner(xess_input) &&
           (!game_motion_available_ || same_owner(motion))) {
            HdrResourceKey key;key.device_identity=identity.Get();key.source=s;key.xess=d;
            key.motion_available=game_motion_available_;key.heap=hd;
            if(game_motion_available_) key.game_motion=motion->GetDesc();
            key.proxy.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            key.proxy.Width=width_;key.proxy.Height=height_;key.proxy.DepthOrArraySize=1;
            key.proxy.MipLevels=1;key.proxy.SampleDesc.Count=1;
            key.proxy.Format=DXGI_FORMAT_R32G32B32A32_FLOAT;
            key.proxy.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            key.nr_motion=key.proxy;key.nr_motion.Format=DXGI_FORMAT_R16G16_FLOAT;
            resource_lease_=pool->acquire(key,[&]() -> std::unique_ptr<HdrResourceBundle> {
                auto bundle=std::make_unique<HdrResourceBundle>();
                bundle->identity=identity;bundle->device=device;
                const auto p=device->GetResourceAllocationInfo(0,1,&key.proxy);
                const auto m=device->GetResourceAllocationInfo(0,1,&key.nr_motion);
                if(!p.SizeInBytes || !m.SizeInBytes || p.SizeInBytes==UINT64_MAX ||
                   m.SizeInBytes==UINT64_MAX || p.SizeInBytes>HdrProxyResourcePool::max_texture_bytes ||
                   m.SizeInBytes>HdrProxyResourcePool::max_texture_bytes-p.SizeInBytes) return {};
                bundle->texture_bytes=p.SizeInBytes+m.SizeInBytes;
                bundle->proxy=texture(device,width_,height_,DXGI_FORMAT_R32G32B32A32_FLOAT);
                bundle->motion=texture(device,width_,height_,DXGI_FORMAT_R16G16_FLOAT);
                if(!bundle->proxy || !bundle->motion ||
                   FAILED(device->CreateDescriptorHeap(&hd,IID_PPV_ARGS(bundle->heap.GetAddressOf())))) return {};
                bundle->stride=device->GetDescriptorHandleIncrementSize(hd.Type);
                bundle->descriptor_bytes_estimate=uint64_t(bundle->stride)*5;
                return bundle;
            });
        }
    }
    if(resource_lease_) {
        auto& bundle=resource_lease_->bundle();
        proxy_=bundle.proxy;nr_motion_=bundle.motion;heap_=bundle.heap;stride_=bundle.stride;
        resource_initial_=resource_lease_->initial_state();
        bundle.bound_source=source;bundle.bound_motion=game_motion_available_?motion:nullptr;
        bundle.bound_xess=xess_input;bundle.bound_nr_result.Reset();
    } else {
        // Pool disabled, busy, incompatible or failed: original unique allocation.
        proxy_=texture(device,width_,height_,DXGI_FORMAT_R32G32B32A32_FLOAT);
        nr_motion_=texture(device,width_,height_,DXGI_FORMAT_R16G16_FLOAT);
        if(!proxy_ || !nr_motion_ ||
           FAILED(device->CreateDescriptorHeap(&hd,IID_PPV_ARGS(heap_.GetAddressOf())))) return false;
        stride_=device->GetDescriptorHandleIncrementSize(hd.Type);
    }
    // Refresh all five slots on every exclusive rebind. Composite overwrites
    // slot 1 with NR output; next prepare must restore the motion/null SRV.
    srv(device,source,DXGI_FORMAT_R11G11B10_FLOAT,cpu(0));
    srv(device,game_motion_available_?motion:nullptr,
        game_motion_available_?DXGI_FORMAT_R16G16_FLOAT:DXGI_FORMAT_R32G32B32A32_FLOAT,cpu(1));
    uav(device,proxy_.Get(),DXGI_FORMAT_R32G32B32A32_FLOAT,cpu(2));
    uav(device,nr_motion_.Get(),DXGI_FORMAT_R16G16_FLOAT,cpu(3));
    uav(device,xess_input,DXGI_FORMAT_R11G11B10_FLOAT,cpu(4));

    if(cache) return cache->acquire(device,shader_path,root_,prepare_,composite_);
    return uncached_bundle(device,shader_path,root_,prepare_,composite_,nullptr);
}
void HdrProxy::bind(ID3D12GraphicsCommandList* list,ID3D12PipelineState* pipeline,
                    bool use_game_motion,float scale_x,float scale_y,
                    ID3D12DescriptorHeap* private_heap) {
    ID3D12DescriptorHeap* heaps[]={private_heap?private_heap:heap_.Get()};
    list->SetDescriptorHeaps(1,heaps);
    list->SetComputeRootSignature(root_.Get());
    auto srv_table=private_heap?private_heap->GetGPUDescriptorHandleForHeapStart():gpu(0);
    auto uav_table=srv_table;uav_table.ptr+=2*stride_;
    list->SetComputeRootDescriptorTable(0,srv_table);
    list->SetComputeRootDescriptorTable(1,uav_table);
    const UINT constants[]={width_,height_,UINT(use_game_motion),
        std::bit_cast<UINT>(scale_x),std::bit_cast<UINT>(scale_y)};
    list->SetComputeRoot32BitConstants(2,5,constants,0);
    list->SetPipelineState(pipeline);
    list->Dispatch((width_+7)/8,(height_+7)/8,1);
}
void HdrProxy::record_prepare(ID3D12GraphicsCommandList* list,
                              ID3D12Resource* source,bool use_game_motion,
                              float scale_x,float scale_y,
                              D3D12_RESOURCE_STATES source_state) {
    if(resource_lease_) {
        resource_lease_->proof().prepare_recorded();
        const auto reuse=hdr_reuse_transitions(proxy_.Get(),nr_motion_.Get(),resource_initial_);
        if(reuse.count) list->ResourceBarrier(reuse.count,reuse.barriers.data());
    }
    auto to_read=transition(source,source_state,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    list->ResourceBarrier(1,&to_read);
    bind(list,prepare_.Get(),use_game_motion && game_motion_available_,scale_x,scale_y);
    D3D12_RESOURCE_BARRIER done[3]={
        transition(source,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                   source_state),
        transition(proxy_.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        transition(nr_motion_.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)};
    list->ResourceBarrier(3,done);
}
bool HdrProxy::record_composite(ID3D12GraphicsCommandList* list,
    ID3D12Resource* source,ID3D12Resource* nr_result,
    ID3D12Resource* xess_input,D3D12_RESOURCE_STATES source_state,
    D3D12_RESOURCE_STATES xess_input_state,bool private_descriptors) {
    if(!list || !source || !nr_result || !xess_input || !composite_ ||
       nr_result->GetDesc().Format!=DXGI_FORMAT_R32G32B32A32_FLOAT ||
       nr_result->GetDesc().Width!=width_ ||
       nr_result->GetDesc().Height!=height_) return false;
    if(resource_lease_) {
        if(!(private_descriptors?resource_lease_->proof().can_record_private_composite():
             resource_lease_->proof().can_rewrite_descriptors())) return false;
        resource_lease_->proof().composite_recorded(private_descriptors);
        resource_lease_->bundle().bound_nr_result=nr_result;
    }
    if(private_descriptors) {
        if(composite_heap_) return false;
        D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,5,
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,0};
        if(FAILED(device_->CreateDescriptorHeap(&desc,IID_PPV_ARGS(composite_heap_.GetAddressOf())))) return false;
        const auto handle=[&](UINT slot) {auto h=composite_heap_->GetCPUDescriptorHandleForHeapStart();
            h.ptr+=SIZE_T(slot)*stride_;return h;};
        srv(device_.Get(),source,DXGI_FORMAT_R11G11B10_FLOAT,handle(0));
        srv(device_.Get(),nr_result,DXGI_FORMAT_R32G32B32A32_FLOAT,handle(1));
        uav(device_.Get(),proxy_.Get(),DXGI_FORMAT_R32G32B32A32_FLOAT,handle(2));
        uav(device_.Get(),nr_motion_.Get(),DXGI_FORMAT_R16G16_FLOAT,handle(3));
        uav(device_.Get(),xess_input,DXGI_FORMAT_R11G11B10_FLOAT,handle(4));
    } else srv(device_.Get(),nr_result,DXGI_FORMAT_R32G32B32A32_FLOAT,cpu(1));
    D3D12_RESOURCE_BARRIER ready[2]={
        transition(source,source_state,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        transition(xess_input,xess_input_state,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS)};
    list->ResourceBarrier(2,ready);
    bind(list,composite_.Get(),false,1,1,private_descriptors?composite_heap_.Get():nullptr);
    D3D12_RESOURCE_BARRIER done[2]={
        transition(source,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                   source_state),
        transition(xess_input,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   xess_input_state)};
    list->ResourceBarrier(2,done);
    return true;
}
}
