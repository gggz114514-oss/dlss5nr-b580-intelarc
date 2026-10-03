#define NVSDK_NGX
#include <nvsdk_ngx.h>
#include <windows.h>
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_EvaluateFeature(
    ID3D12GraphicsCommandList*, const NVSDK_NGX_Handle*, NVSDK_NGX_Parameter*,
    PFN_NVSDK_NGX_ProgressCallback) {
    return NVSDK_NGX_Result_Success;
}

extern "C" __declspec(dllexport) NVSDK_NGX_Result MockInternalEvaluate(
    ID3D12GraphicsCommandList* list, const NVSDK_NGX_Handle* handle,
    NVSDK_NGX_Parameter* params) {
    auto module=GetModuleHandleW(L"OptiScaler.dll");
    auto address=GetProcAddress(module,"NVSDK_NGX_D3D12_EvaluateFeature");
    auto fn=reinterpret_cast<NVSDK_NGX_Result(*)(ID3D12GraphicsCommandList*,
        const NVSDK_NGX_Handle*,NVSDK_NGX_Parameter*,PFN_NVSDK_NGX_ProgressCallback)>(address);
    volatile NVSDK_NGX_Result result=fn(list,handle,params,nullptr);
    return result;
}
