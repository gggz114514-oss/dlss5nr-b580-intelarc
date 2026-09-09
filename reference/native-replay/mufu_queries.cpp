// Diagnostic scalar oracle: supplied FP32 operands, no model/image substitution.
#define wmain recorded_pre_main
#include "pre_replay.cpp"
#undef wmain
int wmain(int argc,wchar_t**argv){
 CUcontext context=nullptr;CUdevice device=0;CUmodule module=nullptr;CUdeviceptr input_gpu=0,output_gpu=0;
 auto cleanup=[&](){if(input_gpu)cuMemFree(input_gpu);if(output_gpu)cuMemFree(output_gpu);if(module)cuModuleUnload(module);if(context)cuDevicePrimaryCtxRelease(device);};
 try{
  if(argc!=3)throw std::runtime_error("Usage: mufu_queries.exe INPUT NEW_OUTPUT");
  std::filesystem::path input(argv[1]),output(argv[2]);if(std::filesystem::exists(output))throw std::runtime_error("Output exists");
  auto operands=read_file(input/"input.u32x4.bin"),cubin=read_file(input/"mufu-sm89.cubin");
  if(operands.empty()||operands.size()%512||operands.size()>size_t(512)*1024*1024)throw std::runtime_error("Input must contain bounded groups of 32 aligned float4 operands");
  const size_t count=operands.size()/16,output_bytes=count*4;
  CUDA_CHECK(cuInit(0));CUDA_CHECK(cuDeviceGet(&device,0));char name[256]{};int major=0,minor=0;
  CUDA_CHECK(cuDeviceGetName(name,sizeof(name),device));CUDA_CHECK(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device));CUDA_CHECK(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device));
  if(major!=8||minor!=9||std::string(name).find("4060")==std::string::npos)throw std::runtime_error("Requires reference RTX4060 SM89");
  CUDA_CHECK(cuDevicePrimaryCtxRetain(&context,device));CUDA_CHECK(cuCtxSetCurrent(context));CUDA_CHECK(cuModuleLoadData(&module,cubin.data()));CUfunction fn=nullptr;
  CUDA_CHECK(cuModuleGetFunction(&fn,module,"cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8"));
  CUDA_CHECK(cuMemAlloc(&input_gpu,operands.size()));CUDA_CHECK(cuMemcpyHtoD(input_gpu,operands.data(),operands.size()));
  CUDA_CHECK(cuMemAlloc(&output_gpu,output_bytes));CUDA_CHECK(cuMemsetD8(output_gpu,0xcd,output_bytes));
  std::vector<char>params(264);set<uint64_t>(params,216,output_gpu);set<uint64_t>(params,224,input_gpu);size_t size=params.size();
  void*extra[]={CU_LAUNCH_PARAM_BUFFER_POINTER,params.data(),CU_LAUNCH_PARAM_BUFFER_SIZE,&size,CU_LAUNCH_PARAM_END};
  CUDA_CHECK(cuLaunchKernel(fn,static_cast<unsigned>(count/32),1,1,32,1,1,0,nullptr,nullptr,extra));CUDA_CHECK(cuCtxSynchronize());
  std::vector<char>result(output_bytes);CUDA_CHECK(cuMemcpyDtoH(result.data(),output_gpu,output_bytes));
  std::filesystem::create_directories(output);write_file(output/"output.f32.bin",result.data(),result.size());
  cleanup();std::cout<<"Scalar operands executed: "<<count<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;cleanup();return 1;}
}
