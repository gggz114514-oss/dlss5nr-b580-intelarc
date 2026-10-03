#pragma once
#include "nr_bridge.h"
#include <windows.h>
#include <string>
#include <cstdint>

namespace nrb {
using LogFunction = void (*)(const char*);
// Dynamic CPython 3.13 ABI adapted from RE8 nr_live_present.cpp PythonApi.
// The interpreter is deliberately process-lifetime; unloading would invalidate
// PyObjects and native runtime modules still owned by CPython.
class PythonProcessor {
public:
    explicit PythonProcessor(LogFunction logger): log_(logger) {}
    static int process_callback(void*, const NRB_Frame*, NRB_Result*);
    static int retire_callback(void*, ID3D12Fence*, uint64_t);
    bool assets_available();
    bool start_controls();
private:
    using Object=void;
    using Size=intptr_t;
    struct PreConfig { int fields[11]; };
    struct Status { int type; const char* function; const char* error; int exit_code; };
    static_assert(sizeof(PreConfig)==44 && sizeof(Status)==32);
    LogFunction log_;
    HMODULE dll_=nullptr;
    std::wstring home_, game_, shim_;
    bool attempted_=false, initialized_=false;
    int (__cdecl *is_initialized_)()=nullptr;
    void (__cdecl *init_preconfig_)(PreConfig*)=nullptr;
    Status (__cdecl *preinitialize_)(const PreConfig*)=nullptr;
    void (__cdecl *initialize_ex_)(int)=nullptr;
    void (__cdecl *set_home_)(const wchar_t*)=nullptr;
    void* (__cdecl *save_thread_)()=nullptr;
    int (__cdecl *gil_ensure_)()=nullptr;
    void (__cdecl *gil_release_)(int)=nullptr;
    Object* (__cdecl *sys_get_)(const char*)=nullptr;
    Object* (__cdecl *wide_string_)(const wchar_t*,Size)=nullptr;
    Object* (__cdecl *utf8_string_)(const char*)=nullptr;
    const char* (__cdecl *as_utf8_)(Object*)=nullptr;
    int (__cdecl *list_append_)(Object*,Object*)=nullptr;
    Object* (__cdecl *import_)(const char*)=nullptr;
    Object* (__cdecl *attr_)(Object*,const char*)=nullptr;
    int (__cdecl *callable_)(Object*)=nullptr;
    Object* (__cdecl *tuple_new_)(Size)=nullptr;
    int (__cdecl *tuple_set_)(Object*,Size,Object*)=nullptr;
    Size (__cdecl *tuple_size_)(Object*)=nullptr;
    Object* (__cdecl *tuple_get_)(Object*,Size)=nullptr;
    Object* (__cdecl *long_new_)(unsigned long long)=nullptr;
    unsigned long long (__cdecl *long_get_)(Object*)=nullptr;
    Object* (__cdecl *call_)(Object*,Object*)=nullptr;
    Object* (__cdecl *str_)(Object*)=nullptr;
    void (__cdecl *err_fetch_)(Object**,Object**,Object**)=nullptr;
    Object* (__cdecl *err_occurred_)()=nullptr;
    void (__cdecl *err_clear_)()=nullptr;
    void (__cdecl *decref_)(Object*)=nullptr;
    Object *process_=nullptr, *retire_=nullptr, *configure_=nullptr, *start_controls_=nullptr;
    bool find_assets();
    bool initialize();
    bool append_path(const std::wstring&);
    bool append_integer(Object*,Size,uint64_t);
    bool call_configure(const NRB_Controls&);
    void python_error(const char* stage);
    void log(const char* message) const { if(log_) log_(message); }
    int process(const NRB_Frame*, NRB_Result*);
    int retire(ID3D12Fence*,uint64_t);
};
NRB_Status validate_python_contract(const NRB_Frame&);
}
