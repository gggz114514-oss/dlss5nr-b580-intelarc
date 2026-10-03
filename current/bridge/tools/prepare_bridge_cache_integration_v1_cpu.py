"""Compose reviewed cache with outer timing; stage only, no game writes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
OUTER=PROJECT/"artifacts/bridge-outer-timing-v1-20261001/payload"
CACHE=PROJECT/"artifacts/bridge-cache-v1-20261001"
STAGE=PROJECT/"artifacts/bridge-cache-integrated-v1-20261001/payload"
BUILD=Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001\bridge-outer-timing-v1-build")

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def one(text,old,new):
    if text.count(old)!=1:raise RuntimeError("Source anchor changed: "+old[:90])
    return text.replace(old,new)
def physical(path):
    for part in (path,*path.parents):
        if part.is_symlink() or part.is_junction():raise RuntimeError("Redirected path: "+str(part))

def main():
    for path in (OUTER,CACHE,STAGE):physical(path)
    if STAGE.exists():raise RuntimeError("Fresh integration stage required")
    if not (CACHE/"REPORT.md").exists():raise RuntimeError("Cache worker final report required before composition")
    worker_receipt=json.loads((CACHE/"SOURCE_PIN.json").read_text(encoding="utf-8"))
    for name,value in worker_receipt["source_files"].items():
        if sha(CACHE/name)!=value:raise RuntimeError("Frozen cache worker source changed: "+name)
    receipt=json.loads((BUILD/"build-receipt.json").read_text(encoding="utf-8"))
    if receipt["status"]!="cpu_build_passed":raise RuntimeError("Outer timing CPU validation missing")
    STAGE.mkdir(parents=True)
    for name,value in receipt["source_pins"].items():
        if sha(OUTER/name)!=value:raise RuntimeError("Outer stage changed: "+name)
        dest=STAGE/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((OUTER/name).read_bytes())
    worker_pins={}
    for name in ("src/nr_hdr_proxy.cpp","src/nr_hdr_proxy.h"):
        worker=CACHE/"payload"/name;physical(worker);worker_pins[name]=sha(worker)
        (STAGE/name).write_bytes(worker.read_bytes())
    def read(name):return (STAGE/name).read_text(encoding="utf-8")
    def write(name,text):(STAGE/name).write_text(text,encoding="utf-8",newline="\n")

    public=read("include/nr_bridge.h")
    fields=("shader_compile_calls","shader_compile_failures","source_reads","source_hashes","pso_creates","root_signature_creates","pipeline_hits","pipeline_misses","uncached_initializations","shader_generation","resident_device_entries")
    api="// Immutable pipeline cache; toggles affect new frames, not in-flight resources.\nstruct NRB_HdrCacheStats {\n    uint32_t abi_size, enabled;\n"+"".join(f"    uint64_t {name};\n" for name in fields)+"};\nNRB_API int NRB_GetHdrCacheStats(NRB_HdrCacheStats* stats);\nNRB_API int NRB_SetHdrCacheEnabled(int enabled);\n\n"
    write("include/nr_bridge.h",one(public,"// The built-in processor delegates",api+"// The built-in processor delegates"))
    h=read("src/deferred_identity.h")
    write("src/deferred_identity.h",one(h,"bool deferred_nr_record_times(NRB_RecordTimes* times);","bool deferred_nr_record_times(NRB_RecordTimes* times);\nbool deferred_hdr_cache_stats(NRB_HdrCacheStats* stats);\nvoid deferred_hdr_set_cache_enabled(bool enabled);\nbool deferred_hdr_warmup_shader();"))
    source=read("src/deferred_identity.cpp")
    session="""HdrProxyCache& hdr_cache_session() {
    static auto cache=[] {
        auto result=std::make_unique<HdrProxyCache>(2);
        result->set_enabled(true);return result;
    }();
    return *cache;
}
"""
    source=one(source,"std::atomic<uint64_t> nr_completed{0};",session+"std::atomic<uint64_t> nr_completed{0};")
    source=one(source,"entry.color_copy.Get(),shader)) entry.hdr.reset();","entry.color_copy.Get(),shader,&hdr_cache_session())) entry.hdr.reset();")
    assignments="".join(f"    out.{name}=value.{name};\n" for name in fields)
    functions="""bool deferred_hdr_cache_stats(NRB_HdrCacheStats* stats) {
    if(!stats || stats->abi_size!=sizeof(NRB_HdrCacheStats)) return false;
    NRB_HdrCacheStats out{};out.abi_size=sizeof(out);
#ifdef NRB_LIVE_NR_TEST
    const auto value=hdr_cache_session().stats();out.enabled=value.enabled?1:0;
__ASSIGN__#endif
    *stats=out;return true;
}
void deferred_hdr_set_cache_enabled(bool enabled) {
#ifdef NRB_LIVE_NR_TEST
    hdr_cache_session().set_enabled(enabled);
#else
    (void)enabled;
#endif
}
bool deferred_hdr_warmup_shader() {
#ifdef NRB_LIVE_NR_TEST
    HMODULE module=nullptr;wchar_t path[MAX_PATH]{};
    if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&record_deferred_identity),&module) ||
        !GetModuleFileNameW(module,path,MAX_PATH)) return false;
    std::wstring shader=path;const auto slash=shader.find_last_of(L"\\\\/");
    if(slash==std::wstring::npos) return false;
    shader.resize(slash+1);shader+=L"nr_hdr_proxy.hlsl";
    return hdr_cache_session().warmup_shader(shader);
#else
    return false;
#endif
}
""".replace("__ASSIGN__",assignments)
    source=one(source,"bool deferred_nr_record_times(NRB_RecordTimes* times) {",functions+"bool deferred_nr_record_times(NRB_RecordTimes* times) {")
    write("src/deferred_identity.cpp",source)
    asi=read("src/asi.cpp")
    exports="""NRB_API int NRB_GetHdrCacheStats(NRB_HdrCacheStats* stats) {
#ifdef NRB_LIVE_NR_TEST
    return nrb::deferred_hdr_cache_stats(stats)?1:0;
#else
    if(!stats || stats->abi_size!=sizeof(NRB_HdrCacheStats)) return 0;
    *stats=NRB_HdrCacheStats{};stats->abi_size=sizeof(NRB_HdrCacheStats);return 0;
#endif
}
NRB_API int NRB_SetHdrCacheEnabled(int enabled) {
#ifdef NRB_LIVE_NR_TEST
    if(enabled!=0 && enabled!=1) return 0;
    nrb::deferred_hdr_set_cache_enabled(enabled!=0);return 1;
#else
    (void)enabled;return 0;
#endif
}
"""
    asi=one(asi,"NRB_API int NRB_GetTimingEnabled() {",exports+"NRB_API int NRB_GetTimingEnabled() {")
    asi=one(asi,"    runtime_assets_available=python_processor.assets_available();","""#ifdef NRB_LIVE_NR_TEST
    if(!nrb::deferred_hdr_warmup_shader())
        log_line("HDR shader cache warmup unavailable; first frame will retry\\r\\n");
#endif
    runtime_assets_available=python_processor.assets_available();""")
    write("src/asi.cpp",asi)
    write("CMakeLists.txt",one(read("CMakeLists.txt"),"target_link_libraries(nr_hdr_proxy PUBLIC d3d12 d3dcompiler)","target_link_libraries(nr_hdr_proxy PUBLIC d3d12 d3dcompiler bcrypt)"))

    web=read("game/cyberpunk_nr_web.py")
    cls="class HdrCacheStats(C.Structure):\n    _fields_ = [(\"abi_size\", C.c_uint32), (\"enabled\", C.c_uint32)]\n    _fields_ += [(name, C.c_uint64) for name in (\n"+"".join(f'        "{name}",\n' for name in fields)+"    )]\n\n\n"
    web=one(web,"_names = [",cls+"_names = [")
    web=one(web,"        _native.NRB_GetTimingEnabled.restype = C.c_int","""        if hasattr(_native, "NRB_GetHdrCacheStats"):
            _native.NRB_GetHdrCacheStats.argtypes = [C.POINTER(HdrCacheStats)]
            _native.NRB_GetHdrCacheStats.restype = C.c_int
            _native.NRB_SetHdrCacheEnabled.argtypes = [C.c_int]
            _native.NRB_SetHdrCacheEnabled.restype = C.c_int
        _native.NRB_GetTimingEnabled.restype = C.c_int""")
    functions="""def get_hdr_cache_state():
    native=_api()
    if not hasattr(native, "NRB_GetHdrCacheStats"):
        return None
    stats=HdrCacheStats();stats.abi_size=C.sizeof(HdrCacheStats)
    if not native.NRB_GetHdrCacheStats(C.byref(stats)):
        return None
    state={name:getattr(stats,name) for name,_ in HdrCacheStats._fields_ if name!="abi_size"}
    state["enabled"]=bool(state["enabled"])
    return state


def set_hdr_cache_enabled(enabled):
    if type(enabled) is not bool:
        raise ValueError("HDR cache enabled must be boolean")
    if not _api().NRB_SetHdrCacheEnabled(int(enabled)):
        raise RuntimeError("Native HDR cache switch rejected setting")


"""
    write("game/cyberpunk_nr_web.py",one(web,"def health():",functions+"def health():"))
    adapter=read("game/cyberpunk_nr_adapter.py")
    adapter=one(adapter,"            lifecycle_options =", """            bridge_cache_options = ({"bridge_cache_read": native.get_hdr_cache_state,
                                     "bridge_cache_apply": native.set_hdr_cache_enabled}
                                    if "bridge_cache_read" in inspect.signature(ControlPanel).parameters
                                    else {})
            lifecycle_options =""")
    adapter=one(adapter,"**timing_options, **validation_options, **lifecycle_options)","**timing_options, **validation_options, **lifecycle_options, **bridge_cache_options)")
    write("game/cyberpunk_nr_adapter.py",adapter)
    panel=read("game/nr_game_controls.py")
    panel=one(panel,"                 lifecycle_read=None, lifecycle_apply=None,","                 lifecycle_read=None, lifecycle_apply=None,\n                 bridge_cache_read=None, bridge_cache_apply=None,")
    panel=one(panel,"        self._lifecycle_apply = lifecycle_apply","""        self._lifecycle_apply = lifecycle_apply
        if (bridge_cache_read is None) != (bridge_cache_apply is None):
            raise ValueError("Bridge cache callbacks must be paired")
        self._bridge_cache_read=bridge_cache_read
        self._bridge_cache_apply=bridge_cache_apply""")
    panel=one(panel,"        status[\"lifecycle_audit\"] = self.lifecycle_audit_state()","        status[\"lifecycle_audit\"] = self.lifecycle_audit_state()\n        status[\"bridge_cache\"] = self._bridge_cache_read() if self._bridge_cache_read else None")
    methods="""    def update_bridge_cache(self, values):
        if self._bridge_cache_apply is None:
            raise ValueError("Bridge cache is unavailable on this route")
        if not isinstance(values, dict) or set(values)!={"enabled"} or type(values["enabled"]) is not bool:
            raise ValueError("Expected a boolean bridge cache setting")
        self._bridge_cache_apply(values["enabled"])
        return self._bridge_cache_read()

"""
    panel=one(panel,"    def update_timing(self, values):",methods+"    def update_timing(self, values):")
    panel=one(panel,'"/api/validation", "/api/lifecycle-audit")','"/api/validation", "/api/lifecycle-audit", "/api/bridge-cache")')
    panel=one(panel,'                              else panel.update_lifecycle(values) if self.path == "/api/lifecycle-audit"','                              else panel.update_bridge_cache(values) if self.path == "/api/bridge-cache"\n                              else panel.update_lifecycle(values) if self.path == "/api/lifecycle-audit"')
    panel=one(panel,'<label id="timingSwitch" hidden>', '<label id="bridgeCacheSwitch" hidden><input id="bridgeCacheEnabled" type="checkbox" checked> 复用桥的着色器与管线</label>\n<p id="bridgeCacheStatus" class="hint" hidden></p>\n<label id="timingSwitch" hidden>')
    panel=one(panel,"let validationBusy=false;","let validationBusy=false, bridgeCacheBusy=false;")
    panel=one(panel,"    updateLifecycleView(data);","    updateLifecycleView(data);\n    updateBridgeCacheView(data);")
    js="""function updateBridgeCacheView(data){
  const c=data.bridge_cache, box=$('bridgeCacheEnabled');
  $('bridgeCacheSwitch').hidden=c==null;$('bridgeCacheStatus').hidden=c==null;
  if(c==null)return;
  if(!bridgeCacheBusy)box.checked=c.enabled;
  $('bridgeCacheStatus').textContent=(c.enabled?'管线缓存开启':'管线缓存关闭（对照）')+
    '；累计编译 '+c.shader_compile_calls+' 次，创建管线 '+c.pso_creates+' 次，缓存命中 '+c.pipeline_hits+' 次';
}
async function applyBridgeCache(){
  const box=$('bridgeCacheEnabled'), enabled=box.checked;
  bridgeCacheBusy=true;box.disabled=true;
  try{
    const response=await fetch('/api/bridge-cache',{method:'POST',headers:{'Content-Type':'application/json','X-NR-Token':token},body:JSON.stringify({enabled})}), data=await response.json();
    if(!response.ok)throw Error(data.error);
    box.checked=data.enabled;show(enabled?'桥管线复用已开启':'桥管线复用关闭，后续帧作对照');await active();
  }catch(e){box.checked=!enabled;show('桥缓存切换失败：'+e.message)}
  finally{bridgeCacheBusy=false;box.disabled=false}
}
$('bridgeCacheEnabled').onchange=applyBridgeCache;
"""
    panel=one(panel,"async function applyTiming(){",js+"async function applyTiming(){")
    write("game/nr_game_controls.py",panel)
    pins={path.relative_to(STAGE).as_posix():sha(path) for path in STAGE.rglob("*") if path.is_file() and "__pycache__" not in path.parts}
    report={"status":"integrated_staged","outer_receipt":str(BUILD/"build-receipt.json"),"worker_pins":worker_pins,"source_pins":pins,"G_writes":False,"GPU_executed":False}
    (STAGE.parent/"source-manifest.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],"source":str(STAGE)},ensure_ascii=False))

if __name__=="__main__":main()
