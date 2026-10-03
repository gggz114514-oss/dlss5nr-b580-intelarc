#include "nr_bridge.h"
#include <nvsdk_ngx.h>
#include <windows.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#define CHECK(x) do { if (!(x)) { std::cerr << "CHECK failed: " #x << "\n"; std::abort(); } } while(false)

class EmptyParams final : public NVSDK_NGX_Parameter {
public:
    bool fsr = false;
    bool xess = false;
    void Set(const char*, unsigned long long) override {}
    void Set(const char*, float) override {}
    void Set(const char*, double) override {}
    void Set(const char*, unsigned int) override {}
    void Set(const char*, int) override {}
    void Set(const char*, ID3D11Resource*) override {}
    void Set(const char*, ID3D12Resource*) override {}
    void Set(const char*, void*) override {}
    NVSDK_NGX_Result Get(const char*, unsigned long long*) const override { return NVSDK_NGX_Result_Fail; }
    NVSDK_NGX_Result Get(const char* key, float* value) const override {
        if (fsr && strcmp(key,"FSR.frameTimeDelta")==0) {
            *value=16.0f; return NVSDK_NGX_Result_Success;
        }
        return NVSDK_NGX_Result_Fail;
    }
    NVSDK_NGX_Result Get(const char*, double*) const override { return NVSDK_NGX_Result_Fail; }
    NVSDK_NGX_Result Get(const char*, unsigned int*) const override { return NVSDK_NGX_Result_Fail; }
    NVSDK_NGX_Result Get(const char* key, int* value) const override {
        if (xess && strcmp(key,"XeSS.ExposureScaleTexture")==0) {
            *value=1; return NVSDK_NGX_Result_Success;
        }
        return NVSDK_NGX_Result_Fail;
    }
    NVSDK_NGX_Result Get(const char*, ID3D11Resource**) const override { return NVSDK_NGX_Result_Fail; }
    NVSDK_NGX_Result Get(const char*, ID3D12Resource**) const override { return NVSDK_NGX_Result_Fail; }
    NVSDK_NGX_Result Get(const char*, void**) const override { return NVSDK_NGX_Result_Fail; }
    void Reset() override {}
};

int main() {
    wchar_t path[MAX_PATH]{};
    auto length=GetModuleFileNameW(nullptr,path,MAX_PATH);
    CHECK(length && length<MAX_PATH);
    auto* slash=wcsrchr(path,L'\\');
    CHECK(slash);
    *(slash+1)=L'\0';
    std::wstring base=path;
    const auto log_path=base+L"CyberpunkNRBridge.log";
    DeleteFileW(log_path.c_str());
    HMODULE mock=LoadLibraryW((base+L"OptiScaler.dll").c_str());
    CHECK(mock);
    HMODULE asi=LoadLibraryW((base+L"CyberpunkNRBridge.asi").c_str());
    CHECK(asi);
    auto init=reinterpret_cast<void(*)()>(GetProcAddress(asi,"InitializeASI"));
    auto init_state=reinterpret_cast<int(*)()>(GetProcAddress(asi,"NRB_InitState"));
    auto count=reinterpret_cast<uint64_t(*)(NRB_Route)>(GetProcAddress(asi,"NRB_InterceptCount"));
    auto last_status=reinterpret_cast<NRB_Status(*)()>(GetProcAddress(asi,"NRB_LastStatus"));
    auto eval=reinterpret_cast<NVSDK_NGX_Result(*)(ID3D12GraphicsCommandList*,
        const NVSDK_NGX_Handle*,NVSDK_NGX_Parameter*,PFN_NVSDK_NGX_ProgressCallback)>(
            GetProcAddress(mock,"NVSDK_NGX_D3D12_EvaluateFeature"));
    auto internal=reinterpret_cast<NVSDK_NGX_Result(*)(ID3D12GraphicsCommandList*,
        const NVSDK_NGX_Handle*,NVSDK_NGX_Parameter*)>(GetProcAddress(mock,"MockInternalEvaluate"));
    CHECK(init && init_state && count && last_status && eval && internal);
    init();
    for(int i=0;i<500 && init_state()==1;i++) Sleep(10);
    CHECK(init_state()==2);
    EmptyParams p;
    NVSDK_NGX_Handle handle{};
    handle.Id=1000001;
    for(int i=0;i<10;i++) CHECK(eval(reinterpret_cast<ID3D12GraphicsCommandList*>(0x1000),
        &handle,&p,nullptr)==NVSDK_NGX_Result_Success);
    CHECK(count(NRB_ROUTE_DLSS)==10);
    p.fsr=true;
    for(int i=0;i<10;i++) CHECK(internal(reinterpret_cast<ID3D12GraphicsCommandList*>(0x1000),
        &handle,&p)==NVSDK_NGX_Result_Success);
    CHECK(count(NRB_ROUTE_FSR)==10);
    CHECK(last_status()==NRB_BYPASS_UNKNOWN_ROUTE); // marker is diagnostic only
    p.fsr=false;
    p.xess=true;
    for(int i=0;i<10;i++) CHECK(internal(reinterpret_cast<ID3D12GraphicsCommandList*>(0x1000),
        &handle,&p)==NVSDK_NGX_Result_Success);
    CHECK(count(NRB_ROUTE_XESS)==10);
    CHECK(last_status()==NRB_BYPASS_UNKNOWN_ROUTE); // optional marker is not proof
    p.xess=false;
    for(int i=0;i<3;i++) CHECK(internal(reinterpret_cast<ID3D12GraphicsCommandList*>(0x1000),
        &handle,&p)==NVSDK_NGX_Result_Success);
    CHECK(count(NRB_ROUTE_UNKNOWN)==3);
    handle.Id=2000001;
    CHECK(eval(reinterpret_cast<ID3D12GraphicsCommandList*>(0x1000),
        &handle,&p,nullptr)==NVSDK_NGX_Result_Success);
    CHECK(count(NRB_ROUTE_DLSS)==10); // FG remains outside NR path
    handle.Id=10;
    CHECK(eval(reinterpret_cast<ID3D12GraphicsCommandList*>(0x1000),
        &handle,&p,nullptr)==NVSDK_NGX_Result_Success);
    CHECK(count(NRB_ROUTE_DLSS)==10); // native passthrough remains outside NR
    std::ifstream log(log_path);
    CHECK(log.good());
    std::string line;
    int dlss=0,fsr=0,xess=0,unknown=0,fg=0;
    while(std::getline(log,line)) if(line.find("\"event\":\"pre_sr_input\"")!=std::string::npos) {
        if(line.find("\"route\":\"DLSS\"")!=std::string::npos) ++dlss;
        if(line.find("\"route\":\"FSR\"")!=std::string::npos) ++fsr;
        if(line.find("\"route\":\"XeSS\"")!=std::string::npos) ++xess;
        if(line.find("\"route\":\"unknown\"")!=std::string::npos) ++unknown;
        if(line.find("FG")!=std::string::npos) ++fg;
        CHECK(line.find("\"queue\":")!=std::string::npos);
        CHECK(line.find("\"bypass\":")!=std::string::npos);
        CHECK(line.find("\"route_evidence\":")!=std::string::npos);
    }
    CHECK(dlss==8 && fsr==8 && xess==8 && unknown==3 && fg==0);
    std::cout << "ASI detour classified three routes; diagnostic cap 8 per route\n";
}
