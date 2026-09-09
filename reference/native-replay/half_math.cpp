// Enumerate the complete 24-bit uniform domain with original scalar SASS.
#define wmain recorded_pre_main
#include "pre_replay.cpp"
#undef wmain
int wmain(int argc,wchar_t**argv){
 CUcontext context=nullptr;CUdevice device=0;CUmodule module=nullptr;CUdeviceptr output_gpu=0;
 auto cleanup=[&](){if(output_gpu)cuMemFree(output_gpu);if(module)cuModuleUnload(module);if(context)cuDevicePrimaryCtxRelease(device);};
 try{
  if(argc!=3)throw std::runtime_error("Usage: half_math.exe INPUT NEW_OUTPUT");std::filesystem::path input(argv[1]),output(argv[2]);if(std::filesystem::exists(output))throw std::runtime_error("Output exists");std::filesystem::create_directories(output);
  auto cubin=read_file(input/"half-math-sm89.cubin");CUDA_CHECK(cuInit(0));CUDA_CHECK(cuDeviceGet(&device,0));char name[256]{};int major=0,minor=0;CUDA_CHECK(cuDeviceGetName(name,sizeof(name),device));CUDA_CHECK(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device));CUDA_CHECK(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device));if(major!=8||minor!=9||std::string(name).find("4060")==std::string::npos)throw std::runtime_error("Requires reference RTX4060 SM89");
  CUDA_CHECK(cuDevicePrimaryCtxRetain(&context,device));CUDA_CHECK(cuCtxSetCurrent(context));CUDA_CHECK(cuModuleLoadData(&module,cubin.data()));CUfunction fn=nullptr;CUDA_CHECK(cuModuleGetFunction(&fn,module,"cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8"));
  constexpr size_t table_bytes=(size_t(1)<<15)*4;CUDA_CHECK(cuMemAlloc(&output_gpu,table_bytes*2));CUDA_CHECK(cuMemsetD8(output_gpu,0xcd,table_bytes*2));std::vector<char>params(264);set<uint64_t>(params,216,output_gpu);size_t size=params.size();void*extra[]={CU_LAUNCH_PARAM_BUFFER_POINTER,params.data(),CU_LAUNCH_PARAM_BUFFER_SIZE,&size,CU_LAUNCH_PARAM_END};
  CUDA_CHECK(cuLaunchKernel(fn,1024,1,1,32,1,1,0,nullptr,nullptr,extra));CUDA_CHECK(cuCtxSynchronize());
  std::vector<char>table(table_bytes);const char*names[]={"reciprocal.f32.bin","rsqrt.f32.bin"};for(size_t i=0;i<2;++i){CUDA_CHECK(cuMemcpyDtoH(table.data(),output_gpu+i*table_bytes,table_bytes));write_file(output/names[i],table.data(),table.size());}
  cleanup();std::cout<<"Normal half scalar domain enumerated; validate arithmetic boundaries."<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;cleanup();return 1;}
}
