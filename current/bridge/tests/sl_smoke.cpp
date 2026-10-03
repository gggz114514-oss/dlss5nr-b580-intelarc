#include "nr_bridge.h"
#include <nvsdk_ngx.h>
#include <windows.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#define CHECK(x) do { if(!(x)) {std::cerr<<"CHECK failed: " #x "\n";std::abort();} } while(false)

class EmptyParams final:public NVSDK_NGX_Parameter {
public:
    void Set(const char*,unsigned long long) override {}
    void Set(const char*,float) override {}
    void Set(const char*,double) override {}
    void Set(const char*,unsigned int) override {}
    void Set(const char*,int) override {}
    void Set(const char*,ID3D11Resource*) override {}
    void Set(const char*,ID3D12Resource*) override {}
    void Set(const char*,void*) override {}
    NVSDK_NGX_Result Get(const char*,unsigned long long*) const override {return NVSDK_NGX_Result_Fail;}
    NVSDK_NGX_Result Get(const char*,float*) const override {return NVSDK_NGX_Result_Fail;}
    NVSDK_NGX_Result Get(const char*,double*) const override {return NVSDK_NGX_Result_Fail;}
    NVSDK_NGX_Result Get(const char*,unsigned int*) const override {return NVSDK_NGX_Result_Fail;}
    NVSDK_NGX_Result Get(const char*,int*) const override {return NVSDK_NGX_Result_Fail;}
    NVSDK_NGX_Result Get(const char*,ID3D11Resource**) const override {return NVSDK_NGX_Result_Fail;}
    NVSDK_NGX_Result Get(const char*,ID3D12Resource**) const override {return NVSDK_NGX_Result_Fail;}
    NVSDK_NGX_Result Get(const char*,void**) const override {return NVSDK_NGX_Result_Fail;}
    void Reset() override {}
};

int main() {
    wchar_t path[MAX_PATH]{};
    const auto length=GetModuleFileNameW(nullptr,path,MAX_PATH);
    CHECK(length && length<MAX_PATH);
    auto* slash=wcsrchr(path,L'\\');CHECK(slash);*(slash+1)=L'\0';
    const std::wstring base=path;
    const auto log_path=base+L"CyberpunkNRBridge.log";
    DeleteFileW(log_path.c_str());
    CHECK(LoadLibraryW((base+L"OptiScaler.dll").c_str()));
    HMODULE sl=LoadLibraryW((base+L"sl.interposer.dll").c_str());CHECK(sl);
    HMODULE asi=LoadLibraryW((base+L"CyberpunkNRBridge.asi").c_str());CHECK(asi);
    auto init=reinterpret_cast<void(*)()>(GetProcAddress(asi,"InitializeASI"));
    auto state=reinterpret_cast<int(*)()>(GetProcAddress(asi,"NRB_InitState"));
    auto count=reinterpret_cast<uint64_t(*)(NRB_Route)>(GetProcAddress(asi,"NRB_InterceptCount"));
    using Eval=int32_t (*)(uint32_t,const void*,const void**,uint32_t,void*);
    auto eval=reinterpret_cast<Eval>(GetProcAddress(sl,"slEvaluateFeature"));
    CHECK(init && state && count && eval);
    init();for(int i=0;i<500 && state()==1;i++) Sleep(10);
    CHECK(state()==2);
    EmptyParams params;
    NVSDK_NGX_Handle handle{};handle.Id=1000001;
    const void* inputs[]={&handle,&params};
    auto* list=reinterpret_cast<ID3D12GraphicsCommandList*>(0x1000);
    CHECK(eval(0,nullptr,inputs,2,list)==0);
    CHECK(count(NRB_ROUTE_DLSS)==1 && count(NRB_ROUTE_UNKNOWN)==0);
    CHECK(eval(2,nullptr,inputs,2,list)==0);
    CHECK(count(NRB_ROUTE_DLSS)==1 && count(NRB_ROUTE_UNKNOWN)==1);
    std::ifstream log(log_path);CHECK(log.good());
    std::string line;bool saw_evidence=false;
    while(std::getline(log,line))
        if(line.find("\"route_evidence\":\"streamline_feature_dlss\"")!=std::string::npos)
            saw_evidence=true;
    CHECK(saw_evidence);
    std::cout<<"Streamline DLSS feature tag survives sl.common-style NGX caller\n";
}
