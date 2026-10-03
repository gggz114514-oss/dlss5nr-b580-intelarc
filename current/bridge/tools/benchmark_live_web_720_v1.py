from __future__ import annotations
import datetime as dt
import importlib
import json
import math
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

GAME = Path(r"G:\SteamLibrary\steamapps\common\Resident Evil Village BIOHAZARD VILLAGE\nr-runtime\game")
OUT = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001\first-three")
URL = "http://127.0.0.1:8765"
OLD = ("c512_k8_decoder", "c512_k8_c32_native", "num_history_fractional")
ARMS = ((), OLD, OLD, ())
POLL = 0.2
WARM_FRAMES, WARM_SECONDS, MEASURE_FRAMES, STALL_SECONDS = 35, 8.0, 30, 15.0

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

def registry():
    sys.path.insert(0, str(GAME))
    return importlib.import_module("numeric_game_profiles_720_v1")

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)

def self_test():
    reg = registry()
    assert len(reg.PROFILES) == 30
    effective = reg.combined_mode_options(OLD)
    assert effective["decoder_gather_unround_720"] is True
    assert effective["c32_hidden_native_720"] is True
    assert effective["numeric_cleanup_720"]["history_value"] == "fp32_fractional"
    seq, unique, skipped, prev = (100, 100, 103, 104), [], 0, None
    for frame in seq:
        if frame in unique:
            continue
        unique.append(frame)
        if prev is not None and frame - prev > 1:
            skipped += frame - prev - 1
        prev = frame
    assert unique == [100, 103, 104] and skipped == 2
    return {"status":"passed","registry_checkbox_count":len(reg.PROFILES),
            "old_three":list(OLD),"effective_options":effective,
            "frame_dedup_simulation":{"input":list(seq),"unique":unique,"skipped":skipped},
            "network_touched":False,"gpu_touched":False,"utc":utc()}

class API:
    def __init__(self):
        self.token = None
        self.checkboxes = set()
    def connect(self):
        req = urllib.request.Request(URL + "/", headers={"Accept":"text/html"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                page = r.read().decode("utf-8")
        except Exception as e:
            raise RuntimeError(f"{type(e).__name__} opening control page") from None
        m = re.search(r"\bconst\s+token\s*=\s*'([0-9a-f]{48})'\s*;", page)
        if not m:
            raise RuntimeError("CSRF token missing from control page")
        self.token = m.group(1)
        self.checkboxes = set(re.findall(r"""data-optimization=["']([^"']+)["']""", page))
    def call(self, path, payload=None):
        data = None if payload is None else json.dumps(payload,separators=(",",":")).encode()
        headers = {"Accept":"application/json"}
        if data is not None:
            headers.update({"Content-Type":"application/json","Origin":URL,"X-NR-Token":self.token})
        req = urllib.request.Request(URL+path,data=data,headers=headers,method="GET" if data is None else "POST")
        try:
            with urllib.request.urlopen(req,timeout=5) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail=""
            try: detail=str(json.loads(e.read().decode()).get("error",""))
            except Exception: pass
            if self.token: detail=detail.replace(self.token,"<redacted>")
            raise RuntimeError(f"HTTP {e.code} {path}: {detail[:240]}") from None
        except Exception as e:
            raise RuntimeError(f"{type(e).__name__} requesting {path}") from None

def clean(state):
    p=state.get("processing") or {}
    return {"settings":state.get("settings"),"active_mode":state.get("active_mode"),
            "processing":{k:p.get(k) for k in ("frames","last_ms","average_ms","estimated_fps")},
            "health":state.get("health"),"source_geometry":state.get("source_geometry"),
            "timing_enabled":state.get("timing_enabled"),"route":state.get("route"),
            "available_experiments_720":state.get("available_experiments_720")}

def validate_base(st, stable=True):
    s,m=st.get("settings") or {},st.get("active_mode") or {}
    if st.get("route")!="pre-xess-fullsize": raise RuntimeError("wrong API route")
    required={"enabled":True,"input_size":720,"history_mode":"fused","graph_replay":True,
              "backend_variant":"unrounded","experiment_720":"c512_k8"}
    if any(s.get(k)!=v for k,v in required.items()):
        raise RuntimeError("current settings do not match 720p C512+K8/unrounded/fused/replay base")
    g=st.get("source_geometry") or {}
    if (g.get("width"),g.get("height"))!=(1280,720): raise RuntimeError("source geometry is not 1280x720")
    h=st.get("health")
    if not isinstance(h,dict) or h.get("failed") is not False: raise RuntimeError("health.failed true or unavailable")
    if stable:
        if not m: raise RuntimeError("active_mode unavailable")
        for k in ("input_size","style","history_mode","graph_replay","backend_variant","experiment_720"):
            if m.get(k)!=s.get(k): raise RuntimeError("settings/active_mode mismatch: "+k)
        controls=m.get("model_controls") or {}
        for setting_key,active_key in (("model_intensity","intensity"),("local_tone","local_tone"),
                                       ("local_structure","local_structure"),("auto_mask","auto_mask"),
                                       ("skin_structure","skin_structure")):
            if setting_key in s and controls.get(active_key)!=s.get(setting_key):
                raise RuntimeError("settings/active_mode mismatch: "+setting_key)
        if tuple(sorted(m.get("optimizations_720") or ()))!=tuple(sorted(s.get("optimizations_720") or ())):
            raise RuntimeError("settings/active_mode optimization mismatch")
    return s,m

def matches(st,selected):
    try: s,m=validate_base(st)
    except RuntimeError: return False
    target=tuple(sorted(selected))
    return tuple(sorted(s.get("optimizations_720") or ()))==target and tuple(sorted(m.get("optimizations_720") or ()))==target

def fatal(st):
    h=st.get("health") or {}
    if h.get("failed") is not False: raise RuntimeError("health.failed or health unavailable")
    for name,obj in (("health",h),("processing",st.get("processing") or {})):
        for k,v in obj.items():
            low=str(k).lower()
            if v and any(x in low for x in ("cache_miss","cache_missing","out_of_memory","oom","allocation_failed")):
                raise RuntimeError(f"runtime failure indicator {name}.{k}")

def payload(settings,selected):
    keys=("enabled","display_strength","style","input_size","history_mode","graph_replay",
          "model_intensity","local_tone","local_structure","auto_mask","skin_structure","backend_variant")
    if any(k not in settings for k in keys): raise RuntimeError("settings lacks required control field")
    out={k:settings[k] for k in keys}
    out.update(experiment_720="c512_k8",optimizations_720=list(selected))
    return out

def wait_target(api,selected):
    deadline=time.monotonic()+45
    while time.monotonic()<deadline:
        st=api.call("/api/state"); fatal(st)
        if matches(st,selected): return st
        time.sleep(POLL)
    raise RuntimeError("settings/active_mode did not converge to target")

def actual_hits(st,selected,reg):
    h=st.get("health") or {}
    need=["c512_library_720_active","native_k8_720_active"]
    if "c512_k8_decoder" in selected: need.append("decoder_gather_unround_720_active")
    if "c512_k8_c32_native" in selected: need.append("c32_hidden_native_720_active")
    miss=[k for k in need if h.get(k) is not True]
    if h.get("structure_combo_frame_route")!="replay": miss.append("structure_combo_frame_route!=replay")
    sites=h.get("numeric_cleanup_sites") or {}
    conditional="num_history_fractional" in selected and sites.get("history") is not True
    if "num_history_fractional" in selected:
        miss += ["numeric_cleanup_sites."+k for k,v in sites.items() if k!="history" and v is not True]
    if miss: raise RuntimeError("actual hit/graph gate failed: "+", ".join(miss))
    return {"mandatory_hits":{k:h.get(k) for k in need},"numeric_cleanup_sites":sites,
            "history_fractional_status":"conditional_unexercised" if conditional else ("hit" if "num_history_fractional" in selected else "not_selected"),
            "graph_route":h.get("structure_combo_frame_route")}

def wait_restore_evidence(api,selected,reg,baseline_frame):
    deadline=time.monotonic()+10.0
    newest=baseline_frame
    while time.monotonic()<deadline:
        st=api.call("/api/state"); fatal(st)
        if matches(st,selected):
            p=st.get("processing") or {}; frame=p.get("frames")
            if type(frame) is int and frame>baseline_frame:
                newest=max(newest,frame)
                if (st.get("health") or {}).get("structure_combo_frame_route")=="replay":
                    try:
                        hits=actual_hits(st,selected,reg)
                    except RuntimeError:
                        time.sleep(POLL)
                        continue
                    return st,hits,{"baseline_frame":baseline_frame,"verified_frame":newest,
                                    "new_completed_frames":newest-baseline_frame,
                                    "wait_seconds":round(10.0-(deadline-time.monotonic()),3)}
        time.sleep(POLL)
    raise RuntimeError("restore did not produce a new replay frame with mandatory hits within 10 seconds")

def measure_arm(api,selected,index):
    st=wait_target(api,selected)
    start_frame=(st.get("processing") or {}).get("frames")
    if type(start_frame) is not int: raise RuntimeError("processing.frames unavailable")
    last=start_frame; advanced=time.monotonic(); warm0=time.monotonic(); warm=0
    while warm<WARM_FRAMES or time.monotonic()-warm0<WARM_SECONDS:
        time.sleep(POLL); st=api.call("/api/state"); fatal(st)
        if not matches(st,selected): raise RuntimeError("actual mode changed during warmup")
        f=(st.get("processing") or {}).get("frames")
        if type(f) is not int: raise RuntimeError("processing.frames unavailable during warmup")
        if f>last: warm+=f-last; last=f; advanced=time.monotonic()
        if time.monotonic()-advanced>STALL_SECONDS: raise RuntimeError("frame counter stalled during warmup")
    hits=actual_hits(st,selected,registry())
    measure0=time.monotonic(); warm_seconds=measure0-warm0; measured=0; polls=0; skipped=0; samples={}
    while measured<MEASURE_FRAMES:
        time.sleep(POLL); st=api.call("/api/state"); fatal(st)
        if not matches(st,selected): raise RuntimeError("actual mode changed during measurement")
        p=st.get("processing") or {}; f=p.get("frames")
        if type(f) is not int: raise RuntimeError("processing.frames unavailable during measurement")
        polls+=1
        if f>last:
            d=f-last; measured+=d; skipped+=max(0,d-1); last=f; advanced=time.monotonic()
            samples[str(f)]={"frame":f,"last_ms":p.get("last_ms"),"average_ms":p.get("average_ms")}
        elif f<last: raise RuntimeError("processing.frames reset during arm")
        if time.monotonic()-advanced>STALL_SECONDS: raise RuntimeError("frame counter stalled during measurement")
        if time.monotonic()-measure0>180: raise RuntimeError("measurement timed out")
    vals=lambda key:[x[key] for x in samples.values() if isinstance(x.get(key),(int,float)) and math.isfinite(x[key])]
    av=vals("average_ms"); lv=vals("last_ms")
    if not av: raise RuntimeError("processing.average_ms unavailable")
    return {"arm":index,"target_optimizations":list(selected),
            "warmup":{"new_completed_frames":warm,"seconds":round(warm_seconds,3)},
            "measurement":{"new_completed_frames":measured,"seconds":round(time.monotonic()-measure0,3),
             "poll_seconds":POLL,"polls":polls,"unique_frame_samples":len(samples),"skipped_frame_ids":skipped,
             "last_ms_sample_count":len(lv),"last_ms_mean":statistics.fmean(lv) if lv else None,
             "average_ms_sample_count":len(av),"average_ms_mean":statistics.fmean(av),
             "average_ms_semantics":"overlapping recent-32-completed-frame rolling metric; not whole-game FPS or independent per-frame P95",
             "samples":list(samples.values())},
            "hit_evidence":hits,"finished_utc":utc()}

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    started=utc(); arms=[]; initial=None; timer_changed=False
    try:
        reg=registry()
        assert len(reg.PROFILES)==30 and set(OLD)<=set(reg.PROFILES)
        reg.combined_mode_options(OLD)
        api=API(); api.connect()
        if api.checkboxes!=set(reg.PROFILES):
            raise RuntimeError("control-page checkbox set differs from runtime profile registry")
        initial=api.call("/api/state"); fatal(initial)
        s,_=validate_base(initial)
        available=set(initial.get("available_experiments_720") or ())
        if not set(OLD)<=available: raise RuntimeError("old-three controls unavailable in API")
        if initial.get("timing_enabled") is None: raise RuntimeError("timing control unavailable")
        initial_clean=clean(initial)
        before=initial.get("timing_enabled")
        if before is False:
            initial=api.call("/api/timing",{"enabled":True}); fatal(initial)
            if initial.get("timing_enabled") is not True: raise RuntimeError("could not enable timing")
            timer_changed=True
        for i,target in enumerate(ARMS,1):
            st=api.call("/api/state"); fatal(st)
            if not matches(st,target):
                api.call("/api/state",payload(st["settings"],target))
            arm=measure_arm(api,target,i); arms.append(arm)
            print(json.dumps({"event":"arm_completed","arm":i,"target":list(target),
               "frames":arm["measurement"]["new_completed_frames"],
               "average_ms_mean":arm["measurement"]["average_ms_mean"],
               "history_fractional_status":arm["hit_evidence"]["history_fractional_status"]},ensure_ascii=False),flush=True)
        st=api.call("/api/state"); fatal(st)
        baseline_frame=(st.get("processing") or {}).get("frames")
        if type(baseline_frame) is not int:
            raise RuntimeError("processing.frames unavailable before final restore verification")
        if not matches(st,OLD):
            st=api.call("/api/state",payload(st["settings"],OLD)); fatal(st)
        final,final_hits,restore_frames=wait_restore_evidence(api,OLD,reg,baseline_frame)
        b=[arms[0]["measurement"]["average_ms_mean"],arms[3]["measurement"]["average_ms_mean"]]
        c=[arms[1]["measurement"]["average_ms_mean"],arms[2]["measurement"]["average_ms_mean"]]
        result={"status":"completed","phase":"old_three_bccb","started_utc":started,"finished_utc":utc(),
          "endpoint":URL,"initial_state":initial_clean,"timing_enabled_before":before,
          "timing_enabled_after":final.get("timing_enabled"),"timing_enabled_by_script":timer_changed,
          "invariants":{"input_size":720,"experiment_720":"c512_k8","backend_variant":"unrounded",
             "history_mode":"fused","graph_replay":True,"style_strength_preserved":True,
             "frame_generation_setting_touched":False,"other_controls_touched":False},
          "arm_order":["B-empty","C-old-three","C-old-three-repeat","B-empty-repeat"],"arms":arms,
          "comparison":{"baseline_average_ms_mean":statistics.fmean(b),"old_three_average_ms_mean":statistics.fmean(c),
             "delta_c_minus_b_ms":statistics.fmean(c)-statistics.fmean(b),
             "baseline_drift_b2_minus_b1_ms":b[1]-b[0],
             "interpretation":"stage-one paired screen; overlapping web average_ms, not whole-game FPS"},
          "restored_final_optimizations":list(OLD),"restore_actual_hit_evidence":final_hits,
          "restore_frame_evidence":restore_frames,
          "token_saved_or_printed":False}
        path=OUT/"completed-checkpoint.json"; write_json(path,result)
        print(json.dumps({"phase_status":"completed","checkpoint":str(path),
          "comparison":result["comparison"],"restored":list(OLD)},ensure_ascii=False),flush=True)
    except Exception as e:
        stopped={"status":"stopped","phase":"old_three_bccb","started_utc":started,"finished_utc":utc(),
          "error":f"{type(e).__name__}: {str(e)[:400]}","completed_arms":arms,
          "initial_state":clean(initial) if isinstance(initial,dict) else None,
          "timing_enabled_by_script":timer_changed,"token_saved_or_printed":False}
        path=OUT/"stopped-checkpoint.json"; write_json(path,stopped)
        print(json.dumps({"phase_status":"stopped","checkpoint":str(path),
          "error":stopped["error"],"completed_arms":len(arms)},ensure_ascii=False),flush=True)
        raise

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=("self-test","first-three"))
    a=p.parse_args()
    if a.command=="self-test":
        result=self_test(); write_json(OUT/"cpu-self-test.json",result)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    else: run()

if __name__=="__main__": main()
