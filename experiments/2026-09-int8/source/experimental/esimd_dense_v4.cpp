// B580-only experimental FP16 dense projection on the caller's SYCL queue.
// DPAS operand layout: A row-major 8x16, B VNNI [8,16,2].
// Static B is packed once; dynamic A packing is separately timed by the caller.
#define NOMINMAX
#include <windows.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace es = sycl::ext::intel::esimd;
using half = sycl::half;
template<bool PackedA> class NRDpasDenseV4;
struct Resources { uint32_t spill_bytes, private_bytes, local_bytes, max_group, registers; };
static thread_local std::string error_text;

static sycl::queue& checked_queue(void* pointer) {
    if (!pointer) throw std::runtime_error("Missing caller queue");
    auto& q = *static_cast<sycl::queue*>(pointer);
    if (!q.is_in_order() || q.get_backend()!=sycl::backend::ext_oneapi_level_zero)
        throw std::runtime_error("Requires caller's in-order Level Zero queue");
    if (q.get_device().get_info<sycl::info::device::name>().find("B580")==std::string::npos)
        throw std::runtime_error("Unvalidated device");
    return q;
}

template<bool PackedA>
static void launch(sycl::queue& q, const half* a, const half* b,
                   const half* initial, half* out, int m, int k, int n) {
    const int nk=n/16, kt=k/16;
    const int tiles=((m+7)/8)*nk;
    constexpr int WG=16;
    q.parallel_for<NRDpasDenseV4<PackedA>>(
        sycl::nd_range<1>(sycl::range<1>((tiles+WG-1)/WG*WG),sycl::range<1>(WG)),
        [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
            const int tile=item.get_global_id(0);
            if(tile>=tiles)return;
            const int rb=tile/nk, cb=tile%nk;
            const half* ap=a+static_cast<size_t>(rb)*8*k;
            const half* bp=b+static_cast<size_t>(cb)*kt*256;
            es::simd<float,128> c(0.0f);
            // Match the observed Triton lowering that seeds the first DPAS
            // with INITIAL. General equivalence is separately tested.
            if(initial) {
                #pragma unroll
                for(int r=0;r<8;++r) {
                    if(rb*8+r<m) {
                        const size_t offset=static_cast<size_t>(rb*8+r)*n+cb*16;
                        es::simd<half,16> h=es::block_load<half,16>(initial+offset);
                        es::simd<float,16> f=h;
                        c.template select<16,1>(r*16)=f;
                    }
                }
            }
            for(int z=0;z<kt;++z) {
                es::simd<half,128> av;
                if constexpr(PackedA) {
                    av=es::block_load<half,128>(ap);
                    ap+=128;
                } else {
                    #pragma unroll
                    for(int r=0;r<8;++r) {
                        es::simd<half,16> row(0.0f);
                        if(rb*8+r<m)row=es::block_load<half,16>(ap+r*k);
                        av.template select<16,1>(r*16)=row;
                    }
                    ap+=16;
                }
                es::simd<half,256> bv;
                bv.template select<128,1>(0)=es::block_load<half,128>(bp);
                bv.template select<128,1>(128)=es::block_load<half,128>(bp+128);
                bp+=256;
                c=es::xmx::dpas<8,8,float>(c,bv,av);
            }
            #pragma unroll
            for(int r=0;r<8;++r) {
                if(rb*8+r<m) {
                    const size_t offset=static_cast<size_t>(rb*8+r)*n+cb*16;
                    es::simd<float,16> row=c.template select<16,1>(r*16);

                    es::simd<half,16> h=row;
                    es::block_store<half,16>(out+offset,h);
                }
            }
        });
}

template<bool PackedA> static Resources resources(sycl::queue& q) {
    const auto id=sycl::get_kernel_id<NRDpasDenseV4<PackedA>>();
    auto bundle=sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(),{q.get_device()},{id});
    auto kernel=bundle.get_kernel(id);
    auto native=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(kernel);
    using Query=ze_result_t(ZE_APICALL*)(ze_kernel_handle_t,ze_kernel_properties_t*);
    auto module=GetModuleHandleW(L"ze_loader.dll");
    auto query=module?reinterpret_cast<Query>(GetProcAddress(module,"zeKernelGetProperties")):nullptr;
    if(!query)throw std::runtime_error("Kernel resource query unavailable");
    ze_kernel_properties_t p={};p.stype=ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
    if(query(native,&p)!=ZE_RESULT_SUCCESS)throw std::runtime_error("Kernel resource query failed");
    const auto group=kernel.template get_info<sycl::info::kernel_device_specific::work_group_size>(q.get_device());
    uint32_t registers=0;
    try {registers=kernel.template get_info<sycl::info::kernel_device_specific::ext_codeplay_num_regs>(q.get_device());}catch(...){}
    return {p.spillMemSize,p.privateMemSize,p.localMemSize,static_cast<uint32_t>(group),registers};
}

extern "C" __declspec(dllexport) const char* nr_esimd_error(){return error_text.c_str();}
extern "C" __declspec(dllexport) int nr_esimd_resources(void* queue,int packed,Resources* out) {
    try {
        if(!out || (packed!=0&&packed!=1))throw std::runtime_error("Invalid resource request");
        auto& q=checked_queue(queue);*out=packed?resources<true>(q):resources<false>(q);return 0;
    }catch(const std::exception& e){error_text=e.what();return 1;}
}
extern "C" __declspec(dllexport) int nr_esimd_dense(void* queue,const void* a,const void* b,
        const void* initial,void* out,int m,int k,int n,int packed) {
    try {
        if(!queue||!a||!b||!out||m<=0||m>1048576||k<16||k>4096||k%16||n<16||n>4096||n%16||
            (packed!=0&&packed!=1))throw std::runtime_error("Invalid dense contract");
        auto& q=*static_cast<sycl::queue*>(queue);
        if(packed)launch<true>(q,static_cast<const half*>(a),static_cast<const half*>(b),
            static_cast<const half*>(initial),static_cast<half*>(out),m,k,n);
        else launch<false>(q,static_cast<const half*>(a),static_cast<const half*>(b),
            static_cast<const half*>(initial),static_cast<half*>(out),m,k,n);
        return 0;
    }catch(const std::exception& e){error_text=e.what();return 1;}
}

extern "C" __declspec(dllexport) int nr_esimd_binary(void* queue,void* output,size_t capacity,size_t* needed) {
    try {
        if(!needed)throw std::runtime_error("Missing binary size result");
        auto& q=checked_queue(queue);
        auto id=sycl::get_kernel_id<NRDpasDenseV4<true>>();
        auto bundle=sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(),{q.get_device()},{id});
        auto modules=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(bundle);
        if(modules.size()!=1)throw std::runtime_error("Expected one AOT module");
        using Query=ze_result_t(ZE_APICALL*)(ze_module_handle_t,size_t*,uint8_t*);
        auto loader=GetModuleHandleW(L"ze_loader.dll");
        auto query=loader?reinterpret_cast<Query>(GetProcAddress(loader,"zeModuleGetNativeBinary")):nullptr;
        if(!query || query(modules[0],needed,nullptr)!=ZE_RESULT_SUCCESS)
            throw std::runtime_error("Native binary size query failed");
        if(!output)return 0;
        if(capacity<*needed)throw std::runtime_error("Native binary buffer too small");
        if(query(modules[0],needed,static_cast<uint8_t*>(output))!=ZE_RESULT_SUCCESS)
            throw std::runtime_error("Native binary query failed");
        return 0;
    }catch(const std::exception& e){error_text=e.what();return 1;}
}
