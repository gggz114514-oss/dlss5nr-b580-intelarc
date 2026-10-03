"""Stage same-frame outer bridge timing without touching the running product."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
STAGE = PROJECT / "artifacts/bridge-outer-timing-v1-20261001/payload"
SHARED = PROJECT.parent / ".codex-worktrees/re8-fp8-unround-fast-20260928/game"


def physical(path):
    for part in (path, *path.parents):
        if part.is_symlink() or part.is_junction():
            raise RuntimeError("Redirected path: " + str(part))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("Source anchor changed: " + old[:100])
    return text.replace(old, new)


def main():
    physical(STAGE)
    if STAGE.exists():
        raise RuntimeError("Fresh source stage required")
    files = {}
    for directory in ("include", "src", "tests", "shaders", "game"):
        for path in (PROJECT / directory).rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            physical(path)
            files[path.relative_to(PROJECT).as_posix()] = path.read_bytes()
    files["CMakeLists.txt"] = (PROJECT / "CMakeLists.txt").read_bytes()
    pins = {name: sha(data) for name, data in files.items()}
    STAGE.mkdir(parents=True)
    for name, data in files.items():
        dest = STAGE / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    def read(name):
        return (STAGE / name).read_text(encoding="utf-8")

    def write(name, text):
        (STAGE / name).write_text(text, encoding="utf-8", newline="\n")

    public = read("include/nr_bridge.h")
    fields = ("record", "resources", "hdr_initialize", "xess_record", "record_to_submit_gap", "record_to_retire_span")
    struct = "// Completed-frame CPU recording intervals; gap/span include game work.\nstruct NRB_RecordTimes {\n    uint32_t abi_size, reserved;\n    uint64_t frames, last_nr_frame_id;\n"
    struct += "".join(f"    double {name}_last_ms, {name}_average_ms;\n" for name in fields)
    struct += "};\nNRB_API int NRB_GetRecordTimes(NRB_RecordTimes* times);\n\n"
    public = replace(public, "// The built-in processor delegates", struct + "// The built-in processor delegates")
    write("include/nr_bridge.h", public)
    header = read("src/deferred_identity.h")
    write("src/deferred_identity.h", replace(header, "bool deferred_nr_stage_times(NRB_StageTimes* times);", "bool deferred_nr_stage_times(NRB_StageTimes* times);\nbool deferred_nr_record_times(NRB_RecordTimes* times);"))

    # Caller holds the existing timing mutex. Every channel has the same valid
    # completed-frame window; a disabled/changed epoch never publishes zeros.
    pairs = "".join(f"        out.{name}_last_ms=last_[{i}]; out.{name}_average_ms=sums_[{i}]/samples;\n" for i, name in enumerate(fields))
    window = """#pragma once
#include "nr_bridge.h"
#include <array>
#include <algorithm>
#include <cmath>
namespace nrb {
class RecordTimingWindow {
    std::array<std::array<double,32>,6> rows_{};
    std::array<double,6> sums_{},last_{};
    uint64_t count_=0, frame_=0;
public:
    void reset() {rows_={}; sums_={}; last_={}; count_=frame_=0;}
    bool append(uint64_t frame,const std::array<double,6>& values) {
        if(!frame || frame<=frame_) return false;
        for(double v:values) if(!std::isfinite(v) || v<0) return false;
        if(values[1]+values[2]+values[3]>values[0]+0.001 ||
           values[4]+values[0]>values[5]+0.001) return false;
        const size_t slot=count_%32;
        for(size_t i=0;i<6;i++) {
            if(count_>=32) sums_[i]-=rows_[i][slot];
            rows_[i][slot]=values[i]; sums_[i]+=values[i]; last_[i]=values[i];
        }
        ++count_; frame_=frame; return true;
    }
    bool snapshot(NRB_RecordTimes* times) const {
        if(!times || times->abi_size!=sizeof(NRB_RecordTimes)) return false;
        NRB_RecordTimes out{};out.abi_size=sizeof(out);
        out.frames=count_;out.last_nr_frame_id=frame_;
        if(count_) {
            const double samples=double(std::min<uint64_t>(32,count_));
__PAIRS__        }
        *times=out;return count_>0;
    }
};
}
""".replace("__PAIRS__", pairs)
    write("src/nr_record_timing.h", window)

    source = read("src/deferred_identity.cpp")
    source = replace(source, '#include "periodic_flash_native_diag.h"', '#include "periodic_flash_native_diag.h"\n#include "nr_record_timing.h"')
    source = replace(source, "    bool nr_timed=false;", "    bool nr_timed=false;\n    bool record_timed=false;\n    uint64_t record_timing_epoch=0;\n    double record_resources_ms=0, record_hdr_ms=0, record_xess_ms=0;\n    std::chrono::steady_clock::time_point record_started{},record_finished{};")
    source = replace(source, "std::mutex nr_timing_mutex;", "std::mutex nr_timing_mutex;\nRecordTimingWindow record_timing_window;")
    source = replace(source, "    flashdiag::Context diag=diagnostic?*diagnostic:flashdiag::Context{};", """#ifdef NRB_LIVE_NR_TEST
    const bool record_timed=nr_timing_enabled.load(std::memory_order_acquire);
    const uint64_t record_epoch=nr_timing_epoch.load(std::memory_order_acquire);
    const auto record_started=record_timed?std::chrono::steady_clock::now():
        std::chrono::steady_clock::time_point{};
#endif
    flashdiag::Context diag=diagnostic?*diagnostic:flashdiag::Context{};""")
    source = replace(source, "    entry.diagnostic=diag;", """    entry.diagnostic=diag;
#ifdef NRB_LIVE_NR_TEST
    entry.record_timed=record_timed;entry.record_timing_epoch=record_epoch;
    entry.record_started=record_started;
#endif""")
    start = "    if(FAILED(entry.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,"
    source = replace(source, start, """#ifdef NRB_LIVE_NR_TEST
    const auto resource_started=record_timed?std::chrono::steady_clock::now():
        std::chrono::steady_clock::time_point{};
#endif
""" + start)
    source = replace(source, "    auto* prep=entry.prep_list.Get();", """#ifdef NRB_LIVE_NR_TEST
    if(record_timed) entry.record_resources_ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-resource_started).count();
#endif
    auto* prep=entry.prep_list.Get();""")
    source = replace(source, "    wchar_t module_path[MAX_PATH]{};", """    const auto hdr_started=record_timed?std::chrono::steady_clock::now():
        std::chrono::steady_clock::time_point{};
    wchar_t module_path[MAX_PATH]{};""")
    source = replace(source, "    if(entry.game_motion && !entry.hdr)", """    if(record_timed) entry.record_hdr_ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-hdr_started).count();
    if(entry.game_motion && !entry.hdr)""")
    source = replace(source, "    result=original(list,handle,params,callback);", """#ifdef NRB_LIVE_NR_TEST
    const auto xess_started=record_timed?std::chrono::steady_clock::now():
        std::chrono::steady_clock::time_point{};
#endif
    result=original(list,handle,params,callback);
#ifdef NRB_LIVE_NR_TEST
    if(record_timed) entry.record_xess_ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-xess_started).count();
#endif""")
    source = replace(source, "        pending.push_back(std::move(owned));", """#ifdef NRB_LIVE_NR_TEST
        if(record_timed) entry.record_finished=std::chrono::steady_clock::now();
#endif
        pending.push_back(std::move(owned));""")
    source = replace(source, "                const unsigned count=++nr_timing_count;", """                if(item.record_timed && item.record_timing_epoch==item.nr_timing_epoch) {
                    record_timing_window.append(item.diagnostic.nr_frame_id,{
                        millis(item.record_started,item.record_finished),
                        item.record_resources_ms,item.record_hdr_ms,item.record_xess_ms,
                        millis(item.record_finished,item.nr_started),
                        millis(item.record_started,ended)});
                }
                const unsigned count=++nr_timing_count;""")
    getter = """bool deferred_nr_record_times(NRB_RecordTimes* times) {
    if(!times || times->abi_size!=sizeof(NRB_RecordTimes)) return false;
#ifdef NRB_LIVE_NR_TEST
    std::lock_guard guard(nr_timing_mutex);
    return record_timing_window.snapshot(times);
#else
    *times=NRB_RecordTimes{};times->abi_size=sizeof(NRB_RecordTimes);return false;
#endif
}
"""
    source = replace(source, "bool deferred_nr_stage_times(NRB_StageTimes* times) {", getter + "bool deferred_nr_stage_times(NRB_StageTimes* times) {")
    source = replace(source, "    nr_timing_count=0;", "    nr_timing_count=0;\n    record_timing_window.reset();")
    write("src/deferred_identity.cpp", source)

    export = """NRB_API int NRB_GetRecordTimes(NRB_RecordTimes* times) {
#ifdef NRB_DEFERRED_IDENTITY_TEST
    return nrb::deferred_nr_record_times(times)?1:0;
#else
    if(!times || times->abi_size!=sizeof(NRB_RecordTimes)) return 0;
    *times=NRB_RecordTimes{};times->abi_size=sizeof(NRB_RecordTimes);return 0;
#endif
}
"""
    write("src/asi.cpp", replace(read("src/asi.cpp"), "NRB_API int NRB_GetTimingEnabled() {", export + "NRB_API int NRB_GetTimingEnabled() {"))
    web = read("game/cyberpunk_nr_web.py")
    cls = "class RecordTimes(C.Structure):\n    _fields_ = [(\"abi_size\", C.c_uint32), (\"reserved\", C.c_uint32),\n                (\"frames\", C.c_uint64), (\"last_nr_frame_id\", C.c_uint64)]\n    _fields_ += [(name, C.c_double) for name in (\n"
    cls += "".join(f'        "{name}_last_ms", "{name}_average_ms",\n' for name in fields) + "    )]\n\n\n"
    web = replace(web, "_names = [", cls + "_names = [")
    web = replace(web, "        _native.NRB_GetTimingEnabled.restype = C.c_int", """        if hasattr(_native, "NRB_GetRecordTimes"):
            _native.NRB_GetRecordTimes.argtypes = [C.POINTER(RecordTimes)]
            _native.NRB_GetRecordTimes.restype = C.c_int
        _native.NRB_GetTimingEnabled.restype = C.c_int""")
    web = replace(web, "    return {\"init_state\": native.NRB_InitState(),", """    outer = None
    if timing is not None and hasattr(native, "NRB_GetRecordTimes"):
        row = RecordTimes();row.abi_size = C.sizeof(RecordTimes)
        if native.NRB_GetRecordTimes(C.byref(row)):
            outer = {name: getattr(row, name) for name, _ in RecordTimes._fields_
                     if name not in ("abi_size", "reserved")}
    if timing is not None:
        timing["recording"] = outer
    return {"init_state": native.NRB_InitState(),""")
    write("game/cyberpunk_nr_web.py", web)

    # Shared webpage is staged here only; install explicitly into physical
    # nr-runtime/game after review, never through the plugins junction.
    physical(SHARED / "nr_game_controls.py")
    panel = (SHARED / "nr_game_controls.py").read_text(encoding="utf-8")
    pins["shared/game/nr_game_controls.py"] = sha((SHARED / "nr_game_controls.py").read_bytes())
    panel = replace(panel, "      const r=stages.runtime;", "      const r=stages.runtime, o=stages.recording;")
    panel = replace(panel, "      bridge.textContent=", """      bridge.textContent=
        (o?'网页外 CPU 录制 '+o.record_average_ms.toFixed(2)+' ms（资源 '+o.resources_average_ms.toFixed(2)+
        '，颜色/运动管线准备 '+o.hdr_initialize_average_ms.toFixed(2)+'，XeSS 命令录制 '+o.xess_record_average_ms.toFixed(2)+'）\\\\n'+
        '录制到提交间隔 '+o.record_to_submit_gap_average_ms.toFixed(2)+' ms；录制到完成跨度 '+o.record_to_retire_span_average_ms.toFixed(2)+
        ' ms（包含游戏工作与等待，不等于完整桥耗时或游戏帧时间）\\\\n':'')+""")
    write("game/nr_game_controls.py", panel)
    changed = {path.relative_to(STAGE).as_posix(): sha(path.read_bytes())
               for path in STAGE.rglob("*") if path.is_file()}
    receipt = {"status": "outer_timing_staged", "source": str(STAGE),
               "base_pins": pins, "staged_pins": changed,
               "GPU_executed": False, "G_writes": False}
    (STAGE.parent / "source-manifest.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "source": str(STAGE)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
