#include "nr_bridge.h"
#include "nr_core.h"
#include "nr_diag.h"
#include "nr_route_adapters.h"
#include <nvsdk_ngx.h>
#include <d3d12.h>
#include <windows.h>
#include <detours/detours.h>
#include <intrin.h>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace {
using Evaluate = NVSDK_NGX_Result (__cdecl *)(ID3D12GraphicsCommandList*,
    const NVSDK_NGX_Handle*,NVSDK_NGX_Parameter*,PFN_NVSDK_NGX_ProgressCallback);
Evaluate original=nullptr;
HMODULE host=nullptr;
std::atomic<int> init_state{0};
std::atomic<unsigned> sample_counts[2]{}; // direct/Streamline DLSS; other NGX calls
std::mutex log_mutex;
thread_local bool inside=false;

void log_line(const char* message) {
    std::lock_guard guard(log_mutex);
    HMODULE self=nullptr;
    if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&log_line),&self)) return;
    wchar_t path[MAX_PATH]{};
    const DWORD size=GetModuleFileNameW(self,path,MAX_PATH);
    if(!size || size>=MAX_PATH) return;
    wchar_t* slash=wcsrchr(path,L'\\');
    if(!slash) return;
    wcscpy_s(slash+1,MAX_PATH-(slash+1-path),L"GenericDLSSProbe.log");
    HANDLE file=CreateFileW(path,FILE_APPEND_DATA,FILE_SHARE_READ|FILE_SHARE_WRITE,
        nullptr,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(file==INVALID_HANDLE_VALUE) return;
    DWORD written=0;
    WriteFile(file,message,static_cast<DWORD>(strlen(message)),&written,nullptr);
    CloseHandle(file);
}

nrb::ResourceMeta meta(ID3D12Resource* resource) {
    if(!resource) return {};
    const auto desc=resource->GetDesc();
    return {static_cast<uint32_t>(desc.Format),static_cast<uint32_t>(desc.Width),desc.Height};
}

NRB_RouteEvidence caller_evidence(void* caller) {
    HMODULE module=nullptr;
    if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(caller),&module)) return NRB_EVIDENCE_NONE;
    if(module==GetModuleHandleW(nullptr)) return NRB_EVIDENCE_GAME_NGX;
    HMODULE streamline=GetModuleHandleW(L"sl.interposer.dll");
    if(streamline && module==streamline) return NRB_EVIDENCE_STREAMLINE_NGX;
    // A call from sl.common.dll or another proxy still needs route validation.
    return module==host?NRB_EVIDENCE_INTERNAL_UNTAGGED:NRB_EVIDENCE_EXTERNAL_UNTAGGED;
}

void caller_name(void* caller,char (&name)[MAX_PATH]) {
    HMODULE module=nullptr;
    wchar_t path[MAX_PATH]{};
    if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(caller),&module) ||
        !GetModuleFileNameW(module,path,MAX_PATH)) {
        strcpy_s(name,"unknown");
        return;
    }
    const wchar_t* slash=wcsrchr(path,L'\\');
    const wchar_t* basename=slash?slash+1:path;
    if(!WideCharToMultiByte(CP_UTF8,0,basename,-1,name,MAX_PATH,nullptr,nullptr))
        strcpy_s(name,"unprintable");
}

NVSDK_NGX_Result __cdecl intercept(ID3D12GraphicsCommandList* list,
    const NVSDK_NGX_Handle* handle,NVSDK_NGX_Parameter* params,
    PFN_NVSDK_NGX_ProgressCallback callback) {
    if(!original) return NVSDK_NGX_Result_Fail;
    if(inside || !list || !handle || !params) return original(list,handle,params,callback);
    void* caller=_ReturnAddress();
    inside=true;
    const auto evidence=caller_evidence(caller);
    const bool dlss=evidence==NRB_EVIDENCE_GAME_NGX ||
        evidence==NRB_EVIDENCE_STREAMLINE_NGX;
    const unsigned group=dlss?0:1;
    const unsigned index=sample_counts[group].fetch_add(1,std::memory_order_relaxed);
    // Sample menu/transition sizes and later gameplay without unbounded logs.
    const bool sampled=index<8 || (index>=16 && index<=8192 && (index&(index-1))==0);
    if(sampled) {
        NRB_Frame frame{};
        nrb::extract_common_ngx(params,dlss?NRB_ROUTE_DLSS:NRB_ROUTE_UNKNOWN,list,frame);
        frame.route_evidence=evidence;
        const auto line=nrb::format_diagnostic(index+1,frame,meta(frame.color),
            meta(frame.motion),meta(frame.depth),meta(frame.exposure),meta(frame.output),
            {},NRB_BYPASS_UNSYNCED);
        char caller_module[MAX_PATH]{};
        caller_name(caller,caller_module);
        char prefix[512]{};
        sprintf_s(prefix,"passive_probe handle=%llu group=%u caller=%s (no NR or resource writes)\r\n",
            static_cast<unsigned long long>(handle->Id),group,caller_module);
        log_line(prefix);
        log_line(line.c_str());
    }
    // No processor, queue hook, command-list command, or resource substitution.
    const auto result=original(list,handle,params,callback);
    inside=false;
    return result;
}

DWORD WINAPI initialize_worker(void*) {
    const wchar_t* names[]={L"OptiScaler.dll",L"dxgi.dll",L"version.dll",
        L"winmm.dll",L"winhttp.dll",L"OptiScaler.asi"};
    for(const auto name:names) {
        HMODULE candidate=GetModuleHandleW(name);
        if(!candidate) continue;
        auto address=GetProcAddress(candidate,"NVSDK_NGX_D3D12_EvaluateFeature");
        if(address) {host=candidate;original=reinterpret_cast<Evaluate>(address);break;}
    }
    if(!original) {
        log_line("passive_probe: OptiScaler NGX evaluator export not found\r\n");
        init_state.store(3,std::memory_order_release);
        return 0;
    }
    LONG rc=DetourTransactionBegin();
    if(rc==NO_ERROR) rc=DetourUpdateThread(GetCurrentThread());
    if(rc==NO_ERROR) rc=DetourAttach(reinterpret_cast<void**>(&original),
        reinterpret_cast<void*>(intercept));
    if(rc==NO_ERROR) rc=DetourTransactionCommit();
    else DetourTransactionAbort();
    if(rc!=NO_ERROR) {
        original=nullptr;
        log_line("passive_probe: NGX evaluator hook failed\r\n");
        init_state.store(3,std::memory_order_release);
        return 0;
    }
    log_line("passive_probe: NGX evaluator hook attached; all frames forwarded unchanged\r\n");
    init_state.store(2,std::memory_order_release);
    return 0;
}
}

extern "C" __declspec(dllexport) void InitializeASI() {
    int expected=0;
    if(!init_state.compare_exchange_strong(expected,1,std::memory_order_acq_rel)) return;
    HANDLE worker=CreateThread(nullptr,0,initialize_worker,nullptr,0,nullptr);
    if(worker) CloseHandle(worker);
    else init_state.store(3,std::memory_order_release);
}
extern "C" __declspec(dllexport) int NRB_InitState() {
    return init_state.load(std::memory_order_acquire);
}
BOOL WINAPI DllMain(HINSTANCE,DWORD,LPVOID) { return TRUE; }
