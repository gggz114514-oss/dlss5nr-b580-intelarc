#include <nvsdk_ngx.h>
#include <windows.h>
#include <cstdint>

extern "C" __declspec(dllexport) int32_t __cdecl slEvaluateFeature(
    uint32_t,const void*,const void** inputs,uint32_t count,void* command_buffer) {
    if(!inputs || count!=2 || !command_buffer) return -1;
    HMODULE opti=GetModuleHandleW(L"OptiScaler.dll");
    if(!opti) return -2;
    using Nested=NVSDK_NGX_Result (*)(ID3D12GraphicsCommandList*,
        const NVSDK_NGX_Handle*,NVSDK_NGX_Parameter*);
    auto nested=reinterpret_cast<Nested>(GetProcAddress(opti,"MockInternalEvaluate"));
    if(!nested) return -3;
    return nested(static_cast<ID3D12GraphicsCommandList*>(command_buffer),
        static_cast<const NVSDK_NGX_Handle*>(inputs[0]),
        static_cast<NVSDK_NGX_Parameter*>(const_cast<void*>(inputs[1])))==
        NVSDK_NGX_Result_Success ? 0 : -4;
}
