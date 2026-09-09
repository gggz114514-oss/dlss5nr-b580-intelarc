// Bounded replay of the four recorded C32 encoder calls, after native pre.
// Runs only in the owned RTX4060 reference process; this is not an Intel backend.
#define wmain recorded_pre_main
#include "pre_replay.cpp"
#undef wmain

int wmain(int argc, wchar_t** argv) {
    CUcontext context=nullptr; CUmodule module=nullptr;
    CUdeviceptr arena=0, weights=0; CUdevice device=0;
    auto cleanup=[&]() {
        if(arena) cuMemFree(arena); if(weights) cuMemFree(weights);
        if(module) cuModuleUnload(module); if(context) cuDevicePrimaryCtxRelease(device);
    };
    try {
        if(argc!=3) throw std::runtime_error("Usage: c32_replay.exe INPUT NEW_OUTPUT");
        const std::filesystem::path input(argv[1]), output(argv[2]);
        if(std::filesystem::exists(output)) throw std::runtime_error("Output already exists");
        std::filesystem::create_directories(output);
        auto cubin=read_file(input/"c32-sm89.cubin");
        auto initial=read_file(input/"arena-initial.bin");
        auto original_base=read_file(input/"arena-base.bin");
        constexpr size_t arena_size=15711232, layer_size=160*160*32, down_size=80*80*64;
        if(initial.size()!=arena_size || original_base.size()!=8) throw std::runtime_error("Wrong fixed arena contract");
        const auto base=get<uint64_t>(original_base,0);
        CUDA_CHECK(cuInit(0)); CUDA_CHECK(cuDeviceGet(&device,0));
        char name[256]{}; int major=0,minor=0;
        CUDA_CHECK(cuDeviceGetName(name,sizeof(name),device));
        CUDA_CHECK(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device));
        CUDA_CHECK(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device));
        if(major!=8 || minor!=9 || std::string(name).find("4060")==std::string::npos)
            throw std::runtime_error("Requires reference RTX4060 SM89");
        std::cout<<name<<std::endl;
        CUDA_CHECK(cuDevicePrimaryCtxRetain(&context,device)); CUDA_CHECK(cuCtxSetCurrent(context));
        CUDA_CHECK(cuModuleLoadData(&module,cubin.data()));
        CUDA_CHECK(cuMemAlloc(&arena,arena_size)); CUDA_CHECK(cuMemcpyHtoD(arena,initial.data(),arena_size));
        CUDA_CHECK(cuMemAlloc(&weights,256*1024));
        const char* kernels[]={"cc_tinlayout_fused_swin_1h_32_1_inpview_tilesync_fp8",
            "cc_tinlayout_fused_swin_1h_32_1_chained_fp8", "cc_tinlayout_fused_swin_1h_32_1_chained_fp8",
            "cc_tinlayout_fused_swin_1h_32_1_ds_wait_fp8"};
        const unsigned gx[]={20,21,21,20}, gy[]={20,21,20,21};
        for(unsigned i=0;i<4;++i) {
            const std::string prefix="layer"+std::to_string(i+1);
            auto params=read_file(input/(prefix+"-parameters.bin"));
            auto weight=read_file(input/(prefix+"-weights.bin"));
            if(params.size()!=96 || weight.size()!=(i==3?22720:20672) ||
               get<uint32_t>(params,24)!=160 || get<uint32_t>(params,28)!=160)
                throw std::runtime_error("Wrong C32 layer contract");
            auto offset=[&](size_t at,size_t length) {
                const auto va=get<uint64_t>(params,at);
                if(va<base || va-base>arena_size || length>arena_size-(va-base))
                    throw std::runtime_error("Original arena pointer out of bounds");
                return static_cast<size_t>(va-base);
            };
            const auto out_offset=offset(8,layer_size);
            const auto down_offset=i==3?offset(64,down_size):0;
            for(auto at:{0u,8u,40u,56u,64u}) {
                if(get<uint64_t>(params,at)) set<uint64_t>(params,at,arena+offset(at,at<16?layer_size:4));
            }
            CUDA_CHECK(cuMemsetD8(weights,0,256*1024));
            CUDA_CHECK(cuMemcpyHtoD(weights,weight.data(),weight.size())); set<uint64_t>(params,16,weights);
            CUfunction fn=nullptr; CUDA_CHECK(cuModuleGetFunction(&fn,module,kernels[i]));
            size_t bytes=params.size(); void* extra[]={CU_LAUNCH_PARAM_BUFFER_POINTER,params.data(),
                CU_LAUNCH_PARAM_BUFFER_SIZE,&bytes,CU_LAUNCH_PARAM_END};
            std::cout<<"Launch "<<i+13<<": "<<kernels[i]<<std::endl;
            CUDA_CHECK(cuLaunchKernel(fn,gx[i],gy[i],1,32,1,1,0,nullptr,nullptr,extra));
            CUDA_CHECK(cuCtxSynchronize());
            std::vector<char> result(arena_size); CUDA_CHECK(cuMemcpyDtoH(result.data(),arena,arena_size));
            write_file(output/(prefix+"-arena.bin"),result.data(),result.size());
            write_file(output/(prefix+".raw"),result.data()+out_offset,layer_size);
            write_file(output/(prefix+"-bound-parameters.bin"),params.data(),params.size());
            if(i==3)write_file(output/"layer4-down.raw",result.data()+down_offset,down_size);
        }
        cleanup(); std::cout<<"Four calls completed; independent native comparison required."<<std::endl; return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;cleanup();return 1;}
}
