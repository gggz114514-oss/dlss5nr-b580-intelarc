// Replay eight captured split-Swin calls in the owned reference process.
#define wmain recorded_pre_main
#include "pre_replay.cpp"
#undef wmain
#include "vit_calls.h"
int wmain(int argc,wchar_t**argv) {
 CUcontext context=nullptr;CUmodule module=nullptr;CUdeviceptr arena=0,weights=0;CUdevice device=0;
 auto cleanup=[&](){if(arena)cuMemFree(arena);if(weights)cuMemFree(weights);if(module)cuModuleUnload(module);if(context)cuDevicePrimaryCtxRelease(device);};
 try {
  if(argc!=3)throw std::runtime_error("Usage: vit_replay.exe INPUT NEW_OUTPUT");
  const std::filesystem::path input(argv[1]),output(argv[2]);
  if(std::filesystem::exists(output))throw std::runtime_error("Output already exists");
  std::filesystem::create_directories(output);
  auto cubin=read_file(input/"vit-sm89.cubin"),initial=read_file(input/"arena-initial.bin"),original_base=read_file(input/"arena-base.bin");
  constexpr size_t arena_size=15711232,weight_capacity=5*1024*1024;
  if(initial.size()!=arena_size||original_base.size()!=8)throw std::runtime_error("Wrong arena contract");
  const auto base=get<uint64_t>(original_base,0);
  CUDA_CHECK(cuInit(0));CUDA_CHECK(cuDeviceGet(&device,0));
  char name[256]{};int major=0,minor=0;
  CUDA_CHECK(cuDeviceGetName(name,sizeof(name),device));
  CUDA_CHECK(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device));
  CUDA_CHECK(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device));
  if(major!=8||minor!=9||std::string(name).find("4060")==std::string::npos)throw std::runtime_error("Requires reference RTX4060 SM89");
  std::cout<<name<<std::endl;
  CUDA_CHECK(cuDevicePrimaryCtxRetain(&context,device));CUDA_CHECK(cuCtxSetCurrent(context));
  CUDA_CHECK(cuModuleLoadData(&module,cubin.data()));CUDA_CHECK(cuMemAlloc(&arena,arena_size));
  CUDA_CHECK(cuMemcpyHtoD(arena,initial.data(),arena_size));CUDA_CHECK(cuMemAlloc(&weights,weight_capacity));
  for(const auto& call:calls) {
   auto prefix="call"+std::to_string(call.sequence);
   auto params=read_file(input/(prefix+"-parameters.bin")),weight=read_file(input/(prefix+"-weights.bin"));
   if(params.size()!=call.parameter_bytes||weight.size()!=call.weight_bytes||weight.size()>weight_capacity)throw std::runtime_error("Wrong captured call contract");
   auto offset=[&](size_t at,size_t length){auto va=get<uint64_t>(params,at);if(va<base||va-base>arena_size||length>arena_size-(va-base))throw std::runtime_error("Arena pointer outside bounds");return static_cast<size_t>(va-base);};
   auto output_offset=offset(call.output_at,call.output_bytes);
   const bool is_qkv=call.sequence>=71&&call.sequence<=106&&(call.sequence-71)%5==0;auto k_offset=is_qkv?offset(16,65536):0;auto v_offset=is_qkv?offset(24,65536):0;
   for(unsigned at=0;at+8<=params.size();at+=8)if(call.pointer_mask&(1u<<(at/8)))set<uint64_t>(params,at,arena+offset(at,4));
   if(call.weight_bytes){CUDA_CHECK(cuMemsetD8(weights,0,weight_capacity));CUDA_CHECK(cuMemcpyHtoD(weights,weight.data(),weight.size()));
   set<uint64_t>(params,call.weight_at,weights);}
   CUfunction fn=nullptr;CUDA_CHECK(cuModuleGetFunction(&fn,module,call.name));
   size_t bytes=params.size();void*extra[]={CU_LAUNCH_PARAM_BUFFER_POINTER,params.data(),CU_LAUNCH_PARAM_BUFFER_SIZE,&bytes,CU_LAUNCH_PARAM_END};
   std::cout<<call.sequence<<" "<<call.name<<std::endl;
   CUDA_CHECK(cuLaunchKernel(fn,call.grid[0],call.grid[1],call.grid[2],call.block[0],call.block[1],call.block[2],0,nullptr,nullptr,extra));
   CUDA_CHECK(cuCtxSynchronize());std::vector<char>result(arena_size);CUDA_CHECK(cuMemcpyDtoH(result.data(),arena,arena_size));
   write_file(output/(prefix+"-arena.bin"),result.data(),result.size());
   write_file(output/(prefix+".raw"),result.data()+output_offset,call.output_bytes);
   write_file(output/(prefix+"-bound-parameters.bin"),params.data(),params.size());
   if(is_qkv){write_file(output/(prefix+"-k.raw"),result.data()+k_offset,65536);write_file(output/(prefix+"-v.raw"),result.data()+v_offset,65536);}
  }
  cleanup();std::cout<<"Captured split calls completed; native comparison required."<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;cleanup();return 1;}
}
