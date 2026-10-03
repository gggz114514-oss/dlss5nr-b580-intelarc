#include <nvsdk_ngx.h>
#include <windows.h>
#include <fstream>
#include <iostream>
#include <string>

class EmptyParams final : public NVSDK_NGX_Parameter {
public:
    void Set(const char*,unsigned long long) override {}
    void Set(const char*,float) override {}
    void Set(const char*,double) override {}
    void Set(const char*,unsigned int) override {}
    void Set(const char*,int) override {}
    void Set(const char*,ID3D11Resource*) override {}
    void Set(const char*,ID3D12Resource*) override {}
    void Set(const char*,void*) override {}
    NVSDK_NGX_Result Get(const char*,unsigned long long*) const override { return NVSDK_NGX_Result_Fail; }
    NVSDK_NGX_Result Get(const char*,float*) const override { return NVSDK_NGX_Result_Fail; }
    NVSDK_NGX_Result Get(const char*,double*) const override { return NVSDK_NGX_Result_Fail; }
    NVSDK_NGX_Result Get(const char*,unsigned int*) const override { return NVSDK_NGX_Result_Fail; }
    NVSDK_NGX_Result Get(const char*,int*) const override { return NVSDK_NGX_Result_Fail; }
    NVSDK_NGX_Result Get(const char*,ID3D11Resource**) const override { return NVSDK_NGX_Result_Fail; }
    NVSDK_NGX_Result Get(const char*,ID3D12Resource**) const override { return NVSDK_NGX_Result_Fail; }
    NVSDK_NGX_Result Get(const char*,void**) const override { return NVSDK_NGX_Result_Fail; }
    void Reset() override {}
};

int main() {
    wchar_t path[MAX_PATH]{};
    const auto length=GetModuleFileNameW(nullptr,path,MAX_PATH);
    if(!length || length>=MAX_PATH) return 1;
    wchar_t* slash=wcsrchr(path,L'\\');
    if(!slash) return 2;
    *(slash+1)=L'\0';
    const std::wstring base=path;
    const auto log_path=base+L"GenericDLSSProbe.log";
    DeleteFileW(log_path.c_str());
    HMODULE mock=LoadLibraryW((base+L"OptiScaler.dll").c_str());
    HMODULE probe=LoadLibraryW((base+L"GenericDLSSProbe.asi").c_str());
    if(!mock || !probe) return 3;
    auto init=reinterpret_cast<void(*)()>(GetProcAddress(probe,"InitializeASI"));
    auto state=reinterpret_cast<int(*)()>(GetProcAddress(probe,"NRB_InitState"));
    auto eval=reinterpret_cast<NVSDK_NGX_Result(*)(ID3D12GraphicsCommandList*,
        const NVSDK_NGX_Handle*,NVSDK_NGX_Parameter*,PFN_NVSDK_NGX_ProgressCallback)>(
        GetProcAddress(mock,"NVSDK_NGX_D3D12_EvaluateFeature"));
    if(!init || !state || !eval) return 4;
    init();
    for(int i=0;i<500 && state()==1;++i) Sleep(10);
    if(state()!=2) return 5;
    EmptyParams params;
    NVSDK_NGX_Handle handle{};
    handle.Id=123;
    for(int i=0;i<12;++i) if(eval(reinterpret_cast<ID3D12GraphicsCommandList*>(0x1000),
        &handle,&params,nullptr)!=NVSDK_NGX_Result_Success) return 6;
    std::ifstream log(log_path);
    if(!log.good()) return 7;
    unsigned samples=0;
    std::string line;
    while(std::getline(log,line)) {
        if(line.find("\"event\":\"pre_sr_input\"")==std::string::npos) continue;
        if(line.find("\"route\":\"DLSS\"")==std::string::npos ||
           line.find("\"source\":[0,0]")==std::string::npos ||
           line.find("\"bypass\":\"producer_or_consumer_order_unproven\"")==std::string::npos)
            return 8;
        ++samples;
    }
    if(samples!=8) return 9;
    std::cout << "Passive probe forwarded 12 calls and bounded the DLSS log at 8 samples\n";
    return 0;
}
