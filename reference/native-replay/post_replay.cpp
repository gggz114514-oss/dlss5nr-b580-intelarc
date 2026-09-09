// Original post kernel with owned CUDA arrays; bounded reset-frame contract.
#define wmain recorded_pre_main
#include "pre_replay.cpp"
#undef wmain
int wmain(int argc,wchar_t**argv){
 CUcontext context=nullptr;CUdevice device=0;CUdeviceptr arena=0,weights=0;CUmodule module=nullptr;CUarray input_array=nullptr,output_array=nullptr;CUtexObject texture=0;CUsurfObject surface=0;
 auto cleanup=[&](){if(surface)cuSurfObjectDestroy(surface);if(texture)cuTexObjectDestroy(texture);if(output_array)cuArrayDestroy(output_array);if(input_array)cuArrayDestroy(input_array);if(arena)cuMemFree(arena);if(weights)cuMemFree(weights);if(module)cuModuleUnload(module);if(context)cuDevicePrimaryCtxRelease(device);};
 try{
  if(argc!=3)throw std::runtime_error("Usage: post_replay.exe INPUT NEW_OUTPUT");std::filesystem::path input(argv[1]),output(argv[2]);if(std::filesystem::exists(output))throw std::runtime_error("Output already exists");std::filesystem::create_directories(output);
  auto initial=read_file(input/"arena-initial.bin"),params=read_file(input/"parameters.bin"),base_data=read_file(input/"arena-base.bin"),raw=read_file(input/"post-weights.bin"),blend=read_file(input/"blend-weights.bin"),rgba=read_file(input/"input.rgba16f.bin"),cubin=read_file(input/"post-sm89.cubin");
  if(initial.size()!=15711232||params.size()!=184||base_data.size()!=8||raw.size()!=21808||blend.size()!=2||rgba.size()!=256*256*8)throw std::runtime_error("Wrong recorded input sizes");
  if(get<uint64_t>(params,88)||get<uint64_t>(params,96)||get<uint64_t>(params,120)||get<uint32_t>(params,112)!=1||get<uint32_t>(params,172)!=256||get<uint32_t>(params,176)!=256)throw std::runtime_error("Only bounded reset reference supported");
  CUDA_CHECK(cuInit(0));CUDA_CHECK(cuDeviceGet(&device,0));char name[256]{};int major=0,minor=0;CUDA_CHECK(cuDeviceGetName(name,sizeof(name),device));CUDA_CHECK(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device));CUDA_CHECK(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device));if(major!=8||minor!=9||std::string(name).find("4060")==std::string::npos)throw std::runtime_error("Requires reference RTX4060 SM89");
  CUDA_CHECK(cuDevicePrimaryCtxRetain(&context,device));CUDA_CHECK(cuCtxSetCurrent(context));CUDA_CHECK(cuModuleLoadData(&module,cubin.data()));CUfunction fn=nullptr;CUDA_CHECK(cuModuleGetFunction(&fn,module,"cc_tinlayout_fused_post_block_swin_1h_32_fp8"));
  CUDA_CHECK(cuMemAlloc(&arena,initial.size()));CUDA_CHECK(cuMemcpyHtoD(arena,initial.data(),initial.size()));CUDA_CHECK(cuMemAlloc(&weights,32768));CUDA_CHECK(cuMemsetD8(weights,0,32768));CUDA_CHECK(cuMemcpyHtoD(weights,raw.data(),raw.size()));CUDA_CHECK(cuMemcpyHtoD(weights+24576,blend.data(),blend.size()));
  CUDA_ARRAY_DESCRIPTOR desc{};desc.Width=256;desc.Height=256;desc.Format=CU_AD_FORMAT_HALF;desc.NumChannels=4;CUDA_CHECK(cuArrayCreate(&input_array,&desc));
  CUDA_MEMCPY2D copy{};copy.srcMemoryType=CU_MEMORYTYPE_HOST;copy.srcHost=rgba.data();copy.srcPitch=256*8;copy.dstMemoryType=CU_MEMORYTYPE_ARRAY;copy.dstArray=input_array;copy.WidthInBytes=256*8;copy.Height=256;CUDA_CHECK(cuMemcpy2D(&copy));
  CUDA_RESOURCE_DESC res{};res.resType=CU_RESOURCE_TYPE_ARRAY;res.res.array.hArray=input_array;CUDA_TEXTURE_DESC tex{};for(int i=0;i<3;++i)tex.addressMode[i]=CU_TR_ADDRESS_MODE_CLAMP;tex.filterMode=CU_TR_FILTER_MODE_POINT;tex.flags=CU_TRSF_NORMALIZED_COORDINATES;CUDA_CHECK(cuTexObjectCreate(&texture,&res,&tex,nullptr));
  CUDA_ARRAY3D_DESCRIPTOR outdesc{};outdesc.Width=320;outdesc.Height=320;outdesc.Format=CU_AD_FORMAT_HALF;outdesc.NumChannels=4;outdesc.Flags=CUDA_ARRAY3D_SURFACE_LDST;CUDA_CHECK(cuArray3DCreate(&output_array,&outdesc));res.res.array.hArray=output_array;CUDA_CHECK(cuSurfObjectCreate(&surface,&res));
  const auto base=get<uint64_t>(base_data,0);for(unsigned at:{0u,8u}){auto va=get<uint64_t>(params,at);if(va<base||va-base>=initial.size())throw std::runtime_error("Arena pointer bounds");set<uint64_t>(params,at,arena+va-base);}
  set<uint64_t>(params,16,surface);set<uint64_t>(params,24,weights);set<uint64_t>(params,56,texture);set<uint64_t>(params,104,weights+24576);
  size_t size=params.size();void*extra[]={CU_LAUNCH_PARAM_BUFFER_POINTER,params.data(),CU_LAUNCH_PARAM_BUFFER_SIZE,&size,CU_LAUNCH_PARAM_END};CUDA_CHECK(cuLaunchKernel(fn,41,41,1,32,1,1,0,nullptr,nullptr,extra));CUDA_CHECK(cuCtxSynchronize());
  std::vector<char>result(320*320*8);copy={};copy.srcMemoryType=CU_MEMORYTYPE_ARRAY;copy.srcArray=output_array;copy.dstMemoryType=CU_MEMORYTYPE_HOST;copy.dstHost=result.data();copy.dstPitch=320*8;copy.WidthInBytes=320*8;copy.Height=320;CUDA_CHECK(cuMemcpy2D(&copy));write_file(output/"post320.rgba16f.bin",result.data(),result.size());
  std::vector<char>crop(256*256*8);for(size_t row=0;row<256;++row)std::memcpy(crop.data()+row*256*8,result.data()+row*320*8,256*8);write_file(output/"post256.rgba16f.bin",crop.data(),crop.size());write_file(output/"bound-parameters.bin",params.data(),params.size());
  CUDA_CHECK(cuMemcpyDtoH(initial.data(),arena,initial.size()));write_file(output/"arena.bin",initial.data(),initial.size());cleanup();std::cout<<"Post reference completed; RGB and alpha require original-output comparison."<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;cleanup();return 1;}
}
