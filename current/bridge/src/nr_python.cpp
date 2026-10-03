#include "nr_python.h"
#include "nr_contract.h"
#include <d3d12.h>
#include <dxgi.h>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <locale>

namespace nrb {
namespace {
bool file_exists(const std::wstring& path) {
    const DWORD flags=GetFileAttributesW(path.c_str());
    return flags!=INVALID_FILE_ATTRIBUTES && !(flags&FILE_ATTRIBUTE_DIRECTORY);
}
bool directory_exists(const std::wstring& path) {
    const DWORD flags=GetFileAttributesW(path.c_str());
    return flags!=INVALID_FILE_ATTRIBUTES && (flags&FILE_ATTRIBUTE_DIRECTORY);
}
}
NRB_Status validate_python_contract(const NRB_Frame& f) {
    if (!f.color || !f.motion) return NRB_BYPASS_UNSUPPORTED;
    const auto c=f.color->GetDesc(), m=f.motion->GetDesc();
    SourceTextures textures{c.Format,m.Format,static_cast<uint32_t>(c.Width),c.Height,
        static_cast<uint32_t>(m.Width),m.Height};
    return plan_re8_adaptations(f,textures)==RE8_DIRECT ? NRB_OK : NRB_BYPASS_UNSUPPORTED;
}

bool PythonProcessor::find_assets() {
    wchar_t root[32768]{};
    DWORD n=GetEnvironmentVariableW(L"CYBERPUNK_NR_RUNTIME",root,_countof(root));
    std::wstring base;
    if (n && n<_countof(root)) base=root;
    else {
        HMODULE self=nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                reinterpret_cast<LPCWSTR>(&PythonProcessor::process_callback),&self)) return false;
        n=GetModuleFileNameW(self,root,_countof(root));
        if (!n || n>=_countof(root)) return false;
        base.assign(root);
        base=base.substr(0,base.find_last_of(L"\\/"));
        shim_=base;
        base+=L"\\nr-runtime";
    }
    if (shim_.empty()) {
        HMODULE self=nullptr; wchar_t path[32768]{};
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                reinterpret_cast<LPCWSTR>(&PythonProcessor::process_callback),&self)) return false;
        n=GetModuleFileNameW(self,path,_countof(path));
        if (!n || n>=_countof(path)) return false;
        shim_=std::wstring(path).substr(0,std::wstring(path).find_last_of(L"\\/"));
    }
    home_=base+L"\\python"; game_=base+L"\\game";
    return directory_exists(game_) &&
        file_exists(home_+L"\\python313.dll") &&
        file_exists(home_+L"\\Library\\bin\\sycl9.dll") &&
        file_exists(game_+L"\\nr_game_pre_xess_host.py") &&
        file_exists(game_+L"\\nr_game_controls.py") &&
        file_exists(game_+L"\\nr_game_fullsize.py") &&
        file_exists(game_+L"\\nr_texture_bridge_v1.py") &&
        file_exists(game_+L"\\rows_fscache_guard_v1.py") &&
        file_exists(base+L"\\fast_cached_runtime_v1.py") &&
        file_exists(base+L"\\native\\nr_texture_bridge_re8_v1.dll") &&
        file_exists(base+L"\\data\\product-v1\\local-runtime-v1.json") &&
        file_exists(base+L"\\data\\product-v1\\profile-v1.json") &&
        file_exists(shim_+L"\\cyberpunk_nr_adapter.py") &&
        file_exists(shim_+L"\\cyberpunk_nr_web.py");
}
bool PythonProcessor::assets_available() { return find_assets(); }

bool PythonProcessor::append_path(const std::wstring& path) {
    Object* list=sys_get_("path"), *item=wide_string_(path.c_str(),-1);
    if (!list || !item) { if(item) decref_(item); return false; }
    const bool ok=list_append_(list,item)==0;
    decref_(item); return ok;
}
void PythonProcessor::python_error(const char* stage) {
    if (!err_occurred_()) return;
    Object *type=nullptr,*value=nullptr,*trace=nullptr;
    err_fetch_(&type,&value,&trace);
    Object* description=value?str_(value):nullptr;
    const char* raw=description?as_utf8_(description):nullptr;
    char line[640]{};
    snprintf(line,sizeof(line),"PythonApi %s: %.480s\r\n",stage,raw?raw:"unknown exception");
    log(line);
    if (description) decref_(description);
    if (type) decref_(type); if (value) decref_(value); if (trace) decref_(trace);
    if (err_occurred_()) err_clear_();
}
bool PythonProcessor::initialize() {
    if (initialized_) return true;
    if (attempted_) return false;
    attempted_=true;
    if (!find_assets()) { log("PythonApi runtime assets missing; NR disabled\r\n"); return false; }
    dll_=LoadLibraryExW((home_+L"\\python313.dll").c_str(),nullptr,LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!dll_) { log("PythonApi python313.dll load failed\r\n"); return false; }
#define LOAD(member,name) member=reinterpret_cast<decltype(member)>(GetProcAddress(dll_,name)); if(!member) {log("PythonApi missing export " name "\r\n");return false;}
    LOAD(is_initialized_,"Py_IsInitialized"); LOAD(init_preconfig_,"PyPreConfig_InitPythonConfig");
    LOAD(preinitialize_,"Py_PreInitialize"); LOAD(initialize_ex_,"Py_InitializeEx");
    LOAD(set_home_,"Py_SetPythonHome"); LOAD(save_thread_,"PyEval_SaveThread");
    LOAD(gil_ensure_,"PyGILState_Ensure"); LOAD(gil_release_,"PyGILState_Release");
    LOAD(sys_get_,"PySys_GetObject"); LOAD(wide_string_,"PyUnicode_FromWideChar");
    LOAD(utf8_string_,"PyUnicode_FromString"); LOAD(as_utf8_,"PyUnicode_AsUTF8");
    LOAD(list_append_,"PyList_Append"); LOAD(import_,"PyImport_ImportModule");
    LOAD(attr_,"PyObject_GetAttrString"); LOAD(callable_,"PyCallable_Check");
    LOAD(tuple_new_,"PyTuple_New"); LOAD(tuple_set_,"PyTuple_SetItem");
    LOAD(tuple_size_,"PyTuple_Size"); LOAD(tuple_get_,"PyTuple_GetItem");
    LOAD(long_new_,"PyLong_FromUnsignedLongLong"); LOAD(long_get_,"PyLong_AsUnsignedLongLong");
    LOAD(call_,"PyObject_CallObject"); LOAD(str_,"PyObject_Str");
    LOAD(err_fetch_,"PyErr_Fetch"); LOAD(err_occurred_,"PyErr_Occurred");
    LOAD(err_clear_,"PyErr_Clear"); LOAD(decref_,"Py_DecRef");
#undef LOAD
    const bool owned=!is_initialized_();
    if (owned) {
        PreConfig config{}; init_preconfig_(&config); config.fields[8]=1;
        if (preinitialize_(&config).type!=0) {log("PythonApi preinitialization failed\r\n");return false;}
        set_home_(home_.c_str()); initialize_ex_(0);
    }
    if (!is_initialized_()) {log("PythonApi initialization failed\r\n");return false;}
    const int gil=owned?0:gil_ensure_();
    bool ok=false;
    do {
        if (!append_path(game_) || !append_path(shim_)) break;
        Object* module=import_("cyberpunk_nr_adapter");
        if (!module) break;
        process_=attr_(module,"process"); retire_=attr_(module,"retire");
        configure_=attr_(module,"configure");
        start_controls_=attr_(module,"start_controls"); decref_(module);
        ok=process_ && retire_ && configure_ && start_controls_ && callable_(process_) &&
            callable_(retire_) && callable_(configure_) && callable_(start_controls_);
    } while(false);
    python_error("import");
    if (owned) save_thread_(); else gil_release_(gil);
    initialized_=ok;
    if (ok) log("PythonApi portable RE8 runtime imported\r\n");
    else log("PythonApi portable RE8 runtime import failed\r\n");
    return ok;
}
bool PythonProcessor::start_controls() {
    if (!initialize()) return false;
    const int gil=gil_ensure_();
    Object* args=tuple_new_(0);
    Object* result=args?call_(start_controls_,args):nullptr;
    const bool ok=result!=nullptr;
    if(result) decref_(result);
    if(args) decref_(args);
    python_error("start_controls");
    gil_release_(gil);
    return ok;
}
bool PythonProcessor::append_integer(Object* args,Size index,uint64_t value) {
    Object* item=long_new_(value);
    return item && tuple_set_(args,index,item)==0;
}
bool PythonProcessor::call_configure(const NRB_Controls& c) {
    std::ostringstream s; s.imbue(std::locale::classic());
    s << "{\"enabled\":" << c.enabled << ",\"input_height\":" << c.input_height
      << ",\"style\":" << c.style << ",\"history\":" << c.history
      << ",\"graph_replay\":" << c.graph_replay << ",\"auto_mask\":" << c.auto_mask
      << ",\"skin_structure_enabled\":" << c.skin_structure_enabled
      << ",\"display_strength\":" << c.display_strength
      << ",\"model_intensity\":" << c.model_intensity
      << ",\"local_tone\":" << c.local_tone
      << ",\"local_structure\":" << c.local_structure
      << ",\"skin_structure\":" << c.skin_structure << '}';
    Object* args=tuple_new_(1);
    if (!args) return false;
    Object* item=utf8_string_(s.str().c_str());
    const bool filled=item && tuple_set_(args,0,item)==0;
    Object* result=filled?call_(configure_,args):nullptr;
    if (result) decref_(result);
    decref_(args);
    return result!=nullptr;
}
int PythonProcessor::process_callback(void* self,const NRB_Frame* frame,NRB_Result* result) {
    return static_cast<PythonProcessor*>(self)->process(frame,result);
}
int PythonProcessor::retire_callback(void* self,ID3D12Fence* fence,uint64_t value) {
    return static_cast<PythonProcessor*>(self)->retire(fence,value);
}
int PythonProcessor::process(const NRB_Frame* f,NRB_Result* out) {
    if (!f || !out || validate_python_contract(*f)!=NRB_OK || !initialize()) return -1;
    const int gil=gil_ensure_();
    bool ok=false; Object* args=tuple_new_(8);
    if (args && call_configure(f->controls) &&
        append_integer(args,0,reinterpret_cast<uintptr_t>(f->device)) &&
        append_integer(args,1,reinterpret_cast<uintptr_t>(f->queue)) &&
        append_integer(args,2,reinterpret_cast<uintptr_t>(f->color)) &&
        append_integer(args,3,reinterpret_cast<uintptr_t>(f->motion)) &&
        append_integer(args,4,f->frame_id) && append_integer(args,5,f->reset_history?1:0) &&
        append_integer(args,6,f->render_width) && append_integer(args,7,f->render_height)) {
        Object* result=call_(process_,args);
        if (result) {
            if (tuple_size_(result)==3) {
                const uint64_t resource=long_get_(tuple_get_(result,0));
                const uint64_t fence=long_get_(tuple_get_(result,1));
                const uint64_t value=long_get_(tuple_get_(result,2));
                if (!err_occurred_() && resource && fence && value) {
                    out->abi_size=sizeof(*out);
                    out->color=reinterpret_cast<ID3D12Resource*>(static_cast<uintptr_t>(resource));
                    out->ready_fence=reinterpret_cast<ID3D12Fence*>(static_cast<uintptr_t>(fence));
                    out->ready_value=value;
                    out->color->AddRef(); out->ready_fence->AddRef(); ok=true;
                }
            }
            decref_(result);
        }
    }
    if (args) decref_(args);
    python_error("process"); gil_release_(gil);
    return ok?0:-1;
}
int PythonProcessor::retire(ID3D12Fence* fence,uint64_t value) {
    if (!initialized_ || !fence || !value) return -1;
    const int gil=gil_ensure_();
    Object* args=tuple_new_(2); bool ok=false;
    if (args && append_integer(args,0,reinterpret_cast<uintptr_t>(fence)) &&
        append_integer(args,1,value)) {
        Object* result=call_(retire_,args);
        if (result) {decref_(result);ok=true;}
    }
    if (args) decref_(args);
    python_error("retire"); gil_release_(gil);
    return ok?0:-1;
}
}
