// Bounded replay of model decoder calls. Native texture/post processing excluded.
#define wmain recorded_pre_main
#include "pre_replay.cpp"
#undef wmain
#include "decoder_calls.h"
int wmain(int argc,wchar_t**argv) {
 CUcontext context=nullptr;CUdevice device=0;CUdeviceptr arena=0,weights=0;std::vector<CUmodule>modules;
 auto cleanup=[&](){if(arena)cuMemFree(arena);if(weights)cuMemFree(weights);for(auto m:modules)cuModuleUnload(m);if(context)cuDevicePrimaryCtxRelease(device);};
 try {
  if(argc!=3)throw std::runtime_error("Usage: decoder_replay.exe INPUT NEW_OUTPUT");
  const std::filesystem::path input(argv[1]),output(argv[2]);
  if(std::filesystem::exists(output))throw std::runtime_error("Output already exists");std::filesystem::create_directories(output);
  auto initial=read_file(input/"arena-initial.bin"),original_base=read_file(input/"arena-base.bin");
  constexpr size_t arena_size=15711232,weight_capacity=4*1024*1024;
  if(initial.size()!=arena_size||original_base.size()!=8)throw std::runtime_error("Wrong arena contract");const auto base=get<uint64_t>(original_base,0);
  CUDA_CHECK(cuInit(0));CUDA_CHECK(cuDeviceGet(&device,0));char name[256]{};int major=0,minor=0;
  CUDA_CHECK(cuDeviceGetName(name,sizeof(name),device));CUDA_CHECK(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device));CUDA_CHECK(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device));
  if(major!=8||minor!=9||std::string(name).find("4060")==std::string::npos)throw std::runtime_error("Requires reference RTX4060 SM89");std::cout<<name<<std::endl;
  CUDA_CHECK(cuDevicePrimaryCtxRetain(&context,device));CUDA_CHECK(cuCtxSetCurrent(context));
  for(auto file:module_files){auto data=read_file(input/file);CUmodule m=nullptr;CUDA_CHECK(cuModuleLoadData(&m,data.data()));modules.push_back(m);}
  CUDA_CHECK(cuMemAlloc(&arena,arena_size));CUDA_CHECK(cuMemcpyHtoD(arena,initial.data(),arena_size));CUDA_CHECK(cuMemAlloc(&weights,weight_capacity));
  for(const auto&call:calls) {
   auto prefix="call"+std::to_string(call.sequence);auto params=read_file(input/(prefix+"-parameters.bin")),weight=read_file(input/(prefix+"-weights.bin"));
   if(params.size()!=call.parameter_bytes||weight.size()!=call.weight_bytes||weight.size()>weight_capacity||call.module>=modules.size())throw std::runtime_error("Wrong recorded call contract");
   for(unsigned at=0;at+8<=params.size();at+=8)if(call.pointer_mask&(1u<<(at/8))){auto va=get<uint64_t>(params,at);if(va<base||va-base>arena_size-4)throw std::runtime_error("Arena pointer outside bounds");set<uint64_t>(params,at,arena+va-base);}
   CUDA_CHECK(cuMemsetD8(weights,0,weight_capacity));CUDA_CHECK(cuMemcpyHtoD(weights,weight.data(),weight.size()));set<uint64_t>(params,call.weight_at,weights);
   CUfunction fn=nullptr;CUDA_CHECK(cuModuleGetFunction(&fn,modules[call.module],call.name));size_t bytes=params.size();void*extra[]={CU_LAUNCH_PARAM_BUFFER_POINTER,params.data(),CU_LAUNCH_PARAM_BUFFER_SIZE,&bytes,CU_LAUNCH_PARAM_END};
   std::cout<<call.sequence<<" "<<call.name<<std::endl;CUDA_CHECK(cuLaunchKernel(fn,call.grid[0],call.grid[1],call.grid[2],call.block[0],call.block[1],call.block[2],0,nullptr,nullptr,extra));CUDA_CHECK(cuCtxSynchronize());
   std::vector<char>result(arena_size);CUDA_CHECK(cuMemcpyDtoH(result.data(),arena,arena_size));write_file(output/(prefix+"-arena.bin"),result.data(),result.size());write_file(output/(prefix+"-bound-parameters.bin"),params.data(),params.size());
  }
  cleanup();std::cout<<"Decoder body completed; independent native comparisons required."<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;cleanup();return 1;}
}
