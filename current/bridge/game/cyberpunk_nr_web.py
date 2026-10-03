"""Loopback-only control page backed by the ASI's validated controls ABI."""
import ctypes as C
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading


class Controls(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in (
        "abi_size", "enabled", "input_height", "style", "history",
        "graph_replay", "auto_mask", "skin_structure_enabled")]
    _fields_ += [(name, C.c_float) for name in (
        "display_strength", "model_intensity", "local_tone",
        "local_structure", "skin_structure")]


class ProducerPoint(C.Structure):
    _fields_ = [("abi_size", C.c_uint32), ("version", C.c_uint32)]
    _fields_ += [(name, C.c_void_p) for name in
                ("device", "queue", "fence", "color", "motion", "processor_context", "processor_process")]
    _fields_ += [(name, C.c_uint64) for name in
                ("value", "frame_id", "registry_epoch", "request_epoch", "callback_cookie", "owner_thread")]
    _fields_ += [("prepared_cpu_waited", C.c_uint32), ("reserved", C.c_uint32)]


class NativeHandoffInfo(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in (
        "abi_size", "version", "enabled", "healthy", "reuse_safe_idle", "active", "poisoned",
        "queue_context_equal", "in_order", "native_luid_matched", "max_active_frames", "private_records",
        "forward_import_per_wait", "producer_cpu_wait_bypass_supported",
        "producer_cpu_wait_bypass_configured", "producer_fence_required")]
    _fields_ += [(name, C.c_uint64) for name in (
        "borrowed_sycl_queue", "native_context", "native_device", "d3d12_device", "d3d12_queue",
        "active_frame", "retained_producer_fence", "retained_producer_value")]


class HandoffCapability(C.Structure):
    _fields_ = [("abi_size", C.c_uint32), ("version", C.c_uint32),
                ("point", ProducerPoint), ("native_info", NativeHandoffInfo), ("native_helper", C.c_uint64)]


class GpuHandoffStats(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in
                ("abi_size", "version", "requested", "armed", "cap_healthy", "pending", "last_rejection", "reserved")]
    _fields_ += [(name, C.c_uint64) for name in (
        "request_epoch", "registry_epoch", "arms", "disarms", "arm_rejections", "source_queries",
        "source_rejections", "prepared_cpu_waits", "prepared_bypasses", "motion_readback_waits",
        "identity_mismatch_waits", "previous_consumer_waits", "previous_consumer_wait_failures",
        "process_failures", "native_helper", "device", "queue", "native_context", "borrowed_sycl_queue", "owner_thread")]


def _bind_handoff(native):
    signatures = {
        "NRB_GetCurrentProducerPoint": [C.POINTER(ProducerPoint)],
        "NRB_ArmGpuHandoffCapability": [C.POINTER(HandoffCapability)],
        "NRB_ArmSerialGpuHandoffCapability": [C.POINTER(HandoffCapability), C.c_uint64],
        "NRB_GetSerialGpuHandoffArmed": [],
        "NRB_ClearGpuHandoffCapability": [],
        "NRB_SetGpuHandoffRequested": [C.c_int],
        "NRB_GetGpuHandoffStats": [C.POINTER(GpuHandoffStats)]}
    for name, args in signatures.items():
        if hasattr(native, name):
            getattr(native, name).argtypes = args
            getattr(native, name).restype = C.c_int


class StageTimes(C.Structure):
    _fields_ = [("abi_size", C.c_uint32), ("reserved", C.c_uint32),
                ("frames", C.c_uint64)]
    _fields_ += [(name, C.c_double) for name in (
        "prep_last_ms", "prep_average_ms", "host_last_ms", "host_average_ms",
        "composite_submit_last_ms", "composite_submit_average_ms",
        "tail_last_ms", "tail_average_ms")]


class RecordTimes(C.Structure):
    _fields_ = [("abi_size", C.c_uint32), ("reserved", C.c_uint32),
                ("frames", C.c_uint64), ("last_nr_frame_id", C.c_uint64)]
    _fields_ += [(name, C.c_double) for name in (
        "record_last_ms", "record_average_ms",
        "resources_last_ms", "resources_average_ms",
        "hdr_initialize_last_ms", "hdr_initialize_average_ms",
        "xess_record_last_ms", "xess_record_average_ms",
        "record_to_submit_gap_last_ms", "record_to_submit_gap_average_ms",
        "record_to_retire_span_last_ms", "record_to_retire_span_average_ms",
    )]


class HdrCacheStats(C.Structure):
    _fields_ = [("abi_size", C.c_uint32), ("enabled", C.c_uint32)]
    _fields_ += [(name, C.c_uint64) for name in (
        "shader_compile_calls",
        "shader_compile_failures",
        "source_reads",
        "source_hashes",
        "pso_creates",
        "root_signature_creates",
        "pipeline_hits",
        "pipeline_misses",
        "uncached_initializations",
        "shader_generation",
        "resident_device_entries",
    )]


class HdrResourcePoolStats(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in (
        "abi_size", "enabled", "capacity", "shutdown")]
    _fields_ += [(name, C.c_uint64) for name in (
        "acquires", "hits", "creates", "disabled_fallbacks", "busy_fallbacks", "creation_failures",
        "retire_returns", "unsubmitted_returns", "quarantines", "evictions", "generation",
        "resident_slots", "free_slots", "leased_slots", "quarantined_slots",
        "texture_bytes", "peak_texture_bytes", "descriptor_bytes_estimate", "max_texture_bytes")]


_names = [name for name, _ in Controls._fields_ if name != "abi_size"]
_native = None
_server = None


def _api():
    global _native
    if _native is None:
        _native = C.CDLL(str(Path(__file__).with_name("CyberpunkNRBridge.asi")))
        _native.NRB_GetControls.argtypes = [C.POINTER(Controls)]
        _native.NRB_GetControls.restype = C.c_int
        _native.NRB_SetControls.argtypes = [C.POINTER(Controls)]
        _native.NRB_SetControls.restype = C.c_int
        _native.NRB_InitState.restype = C.c_int
        _native.NRB_LastStatus.restype = C.c_uint32
        _native.NRB_InterceptCount.argtypes = [C.c_uint32]
        _native.NRB_InterceptCount.restype = C.c_uint64
        _native.NRB_ProcessedCount.restype = C.c_uint64
        _native.NRB_LastProcessMs.restype = C.c_double
        _native.NRB_AverageProcessMs.restype = C.c_double
        _native.NRB_GetStageTimes.argtypes = [C.POINTER(StageTimes)]
        _native.NRB_GetStageTimes.restype = C.c_int
        if hasattr(_native, "NRB_GetRecordTimes"):
            _native.NRB_GetRecordTimes.argtypes = [C.POINTER(RecordTimes)]
            _native.NRB_GetRecordTimes.restype = C.c_int
        if hasattr(_native, "NRB_GetHdrCacheStats"):
            _native.NRB_GetHdrCacheStats.argtypes = [C.POINTER(HdrCacheStats)]
            _native.NRB_GetHdrCacheStats.restype = C.c_int
            _native.NRB_SetHdrCacheEnabled.argtypes = [C.c_int]
            _native.NRB_SetHdrCacheEnabled.restype = C.c_int
        if (hasattr(_native, "NRB_GetHdrResourcePoolStats") and
                hasattr(_native, "NRB_SetHdrResourcePoolEnabled")):
            _native.NRB_GetHdrResourcePoolStats.argtypes = [C.POINTER(HdrResourcePoolStats)]
            _native.NRB_GetHdrResourcePoolStats.restype = C.c_int
            _native.NRB_SetHdrResourcePoolEnabled.argtypes = [C.c_int]
            _native.NRB_SetHdrResourcePoolEnabled.restype = C.c_int
        _native.NRB_GetTimingEnabled.restype = C.c_int
        _native.NRB_SetTimingEnabled.argtypes = [C.c_int]
        _native.NRB_SetTimingEnabled.restype = C.c_int
        _bind_handoff(_native)
    return _native


def get_controls():
    c = Controls()
    c.abi_size = C.sizeof(Controls)
    if not _api().NRB_GetControls(C.byref(c)):
        raise RuntimeError("NRB_GetControls failed")
    return c


def set_controls(values):
    if set(values) != set(_names):
        raise ValueError("Incomplete controls")
    c = get_controls()
    for name in _names:
        value = values[name]
        if type(value) not in (int, float, bool):
            raise ValueError("Non-numeric controls")
        setattr(c, name, value)
    if not _api().NRB_SetControls(C.byref(c)):
        raise ValueError("Unsupported control value")
    return get_controls()


def _values(c):
    return {name: getattr(c, name) for name in _names}


def get_timing_enabled():
    return bool(_api().NRB_GetTimingEnabled())


def set_timing_enabled(enabled):
    if type(enabled) is not bool:
        raise ValueError("timing enabled must be boolean")
    if not _api().NRB_SetTimingEnabled(int(enabled)):
        raise RuntimeError("Native timing switch rejected the setting")


def get_hdr_cache_state():
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


def get_hdr_resource_pool_state():
    native = _api()
    if not (hasattr(native, "NRB_GetHdrResourcePoolStats") and
            hasattr(native, "NRB_SetHdrResourcePoolEnabled")):
        return None
    stats = HdrResourcePoolStats(); stats.abi_size = C.sizeof(HdrResourcePoolStats)
    if not native.NRB_GetHdrResourcePoolStats(C.byref(stats)):
        return None
    state = {name: getattr(stats, name) for name, _ in HdrResourcePoolStats._fields_
             if name != "abi_size"}
    state["enabled"] = bool(state["enabled"])
    state["shutdown"] = bool(state["shutdown"])
    return state


def set_hdr_resource_pool_enabled(enabled):
    if type(enabled) is not bool:
        raise ValueError("HDR resource pool enabled must be boolean")
    native = _api()
    if (not hasattr(native, "NRB_SetHdrResourcePoolEnabled") or
            not native.NRB_SetHdrResourcePoolEnabled(int(enabled))):
        raise RuntimeError("Native HDR resource pool switch rejected setting")


def get_gpu_handoff_state():
    native = _api()
    if not hasattr(native, "NRB_GetGpuHandoffStats"):
        return None
    stats = GpuHandoffStats(); stats.abi_size = C.sizeof(stats)
    if not native.NRB_GetGpuHandoffStats(C.byref(stats)):
        return None
    state = {name: getattr(stats, name) for name, _ in GpuHandoffStats._fields_
             if name not in ("abi_size", "reserved")}
    for name in ("requested", "armed", "cap_healthy", "pending"):
        state[name] = bool(state[name])
    state['serial_owner_transfer_armed'] = bool(
        hasattr(native, 'NRB_GetSerialGpuHandoffArmed') and native.NRB_GetSerialGpuHandoffArmed())
    # Cached host data only. HTTP never touches native helper/GPU/stream state.
    import sys
    host_handoff = sys.modules.get("nr_gpu_handoff_host_v1")
    state["host"] = host_handoff.host_snapshot() if host_handoff else None
    if state["host"] is not None:
        state["pending"] = state["pending"] or state["requested"] != state["host"]["applied"]
    return state


def set_gpu_handoff_requested(requested):
    if type(requested) is not bool:
        raise ValueError("GPU handoff requested must be boolean")
    native = _api()
    if (not hasattr(native, "NRB_SetGpuHandoffRequested") or
            not native.NRB_SetGpuHandoffRequested(int(requested))):
        raise RuntimeError("Native GPU handoff request rejected")


def current_producer_point(device, queue, color, motion, frame_id):
    native = _api()
    if not hasattr(native, "NRB_GetCurrentProducerPoint"):
        return None
    point = ProducerPoint(); point.abi_size = C.sizeof(point)
    if not native.NRB_GetCurrentProducerPoint(C.byref(point)):
        return None
    if (point.version != 1 or point.device != device or point.queue != queue or
            point.color != color or point.motion != motion or point.frame_id != frame_id or
            not point.fence or not 0 < point.value < 2**64-1):
        raise RuntimeError("Scoped producer point does not match the actual source callback")
    return point


def clear_gpu_handoff_capability():
    native = _api()
    return bool(hasattr(native, "NRB_ClearGpuHandoffCapability") and native.NRB_ClearGpuHandoffCapability())


def arm_gpu_handoff_capability(point, native_helper, info):
    capability = HandoffCapability(); capability.abi_size = C.sizeof(capability); capability.version = 1
    capability.point = point; capability.native_helper = native_helper
    for name, _ in NativeHandoffInfo._fields_:
        setattr(capability.native_info, name, info[name])
    native = _api()
    return bool(hasattr(native, "NRB_ArmGpuHandoffCapability") and
                native.NRB_ArmGpuHandoffCapability(C.byref(capability)))


def serial_gpu_handoff_supported():
    native = _api()
    return (hasattr(native, 'NRB_ArmSerialGpuHandoffCapability') and
            hasattr(native, 'NRB_GetSerialGpuHandoffArmed'))


def arm_serial_gpu_handoff_capability(point, bridge, info):
    bridge._ready()  # actual native/Python owner and original Torch stream
    if (not getattr(bridge, '_owner_transfer_supported', False) or
            not callable(getattr(bridge, 'transfer_serial_thread', None)) or
            not serial_gpu_handoff_supported()):
        return False
    transfer = C.cast(bridge.dll.nr_texture_transfer_owner, C.c_void_p).value
    if not transfer:
        return False
    capability = HandoffCapability(); capability.abi_size = C.sizeof(capability); capability.version = 1
    capability.point = point; capability.native_helper = bridge.handle
    for name, _ in NativeHandoffInfo._fields_:
        setattr(capability.native_info, name, info[name])
    return bool(_api().NRB_ArmSerialGpuHandoffCapability(C.byref(capability), transfer))


def health():
    native = _api()
    average = native.NRB_AverageProcessMs()
    stages = StageTimes()
    stages.abi_size = C.sizeof(StageTimes)
    stage_available = bool(native.NRB_GetStageTimes(C.byref(stages)))
    timing = None
    if stage_available:
        timing = {"frames": stages.frames,
                  "prep_last_ms": stages.prep_last_ms,
                  "prep_average_ms": stages.prep_average_ms,
                  "host_last_ms": stages.host_last_ms,
                  "host_average_ms": stages.host_average_ms,
                  "composite_submit_last_ms": stages.composite_submit_last_ms,
                  "composite_submit_average_ms": stages.composite_submit_average_ms,
                  "tail_last_ms": stages.tail_last_ms,
                  "tail_average_ms": stages.tail_average_ms,
                  "bridge_pre_sr_last_ms": (
                      stages.prep_last_ms + stages.composite_submit_last_ms),
                  "bridge_pre_sr_average_ms": (
                      stages.prep_average_ms + stages.composite_submit_average_ms)}
    outer = None
    if timing is not None and hasattr(native, "NRB_GetRecordTimes"):
        row = RecordTimes();row.abi_size = C.sizeof(RecordTimes)
        if native.NRB_GetRecordTimes(C.byref(row)):
            outer = {name: getattr(row, name) for name, _ in RecordTimes._fields_
                     if name not in ("abi_size", "reserved")}
    if timing is not None:
        timing["recording"] = outer
    return {"init_state": native.NRB_InitState(),
            "timing_enabled": get_timing_enabled(),
            "status": native.NRB_LastStatus(),
            "intercepts": {name: native.NRB_InterceptCount(route)
                           for route, name in ((1, "DLSS"), (2, "FSR"), (3, "XeSS"))},
            "processed_frames": native.NRB_ProcessedCount(),
            "last_ms": native.NRB_LastProcessMs(),
            "average_ms": average,
            "stages": timing,
            "gpu_handoff": get_gpu_handoff_state(),
            "estimated_fps": 1000.0 / average if average > 0 else 0.0}


_PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cyberpunk B580 NR 控制</title><style>body{font:16px system-ui;background:#101827;color:#eaf2ff;max-width:620px;margin:30px auto;padding:0 16px}
label{display:block;margin:15px 0}input,select{margin-left:10px;background:#26344b;color:white;padding:6px}button{padding:10px 18px;background:#26344b;color:white;border:1px solid #537096;border-radius:5px;cursor:pointer}.style-choice{padding:6px 10px;margin:0 4px}.active{background:#2475c7}small{color:#aebed2}</style>
<h1>B580 NR → XeSS SR</h1><small>可预设参数；只有源纹理与队列条件通过时 NR 才会运行。桥前置计时包含准备等待与合成命令提交，不含后续 GPU 合成执行；尾段包含 GPU 合成、XeSS 和完成等待。估算吞吐由整段处理时间换算，不是游戏帧率。</small>
<div id="fields"></div><label id="gpuSwitch" hidden><input id="gpuRequested" type="checkbox"> GPU 队列交接（实验）</label><p id="gpuState"></p><button id="apply">应用</button><p id="status"></p><p id="health">等待游戏输入…</p>
<script>const token='__TOKEN__';const fields=[['enabled','启用','check'],['input_height','NR 高度','select:360,480,540,720'],
['style','风格','style3'],['history','历史','select:0,1,2,3'],['graph_replay','计算图重放','check'],
['auto_mask','自动遮罩','check'],['skin_structure_enabled','单独皮肤结构','check'],
['display_strength','显示混合 0–1','number'],['model_intensity','模型强度 0–2','number'],
['local_tone','局部色调 0–2','number'],['local_structure','局部结构 0–2','number'],['skin_structure','皮肤结构 0–2','number']];
const box=document.getElementById('fields');let current={};for(const [key,label,type] of fields){let row=document.createElement('label');
row.textContent=label;let e;if(type==='style3'){e=document.createElement('span');for(const [v,name] of [['0','标准'],['1','自然'],['2','电影']]){
let b=document.createElement('button');b.type='button';b.className='style-choice';b.dataset.value=v;b.textContent=name;
b.onclick=()=>{e.dataset.value=v;for(const x of e.children)x.classList.toggle('active',x===b)};e.append(b)}}
else if(type.startsWith('select')){e=document.createElement('select');for(const v of type.split(':')[1].split(',')){
let o=document.createElement('option');o.value=v;o.textContent=v; e.append(o)}}else{e=document.createElement('input');
e.type=type==='check'?'checkbox':'number';if(type==='number'){e.step='0.05';e.min=0;e.max=key==='display_strength'?1:2}}
e.id=key;row.append(e);box.append(row)}
async function load(){let r=await fetch('/api');current=await r.json();for(const [key,,type] of fields){let e=document.getElementById(key);
if(type==='check')e.checked=!!current[key];else if(type==='style3'){e.dataset.value=current[key];for(const x of e.children)x.classList.toggle('active',x.dataset.value==current[key])}else e.value=current[key]}}
document.getElementById('apply').onclick=async()=>{let v={};for(const [key,,type] of fields){let e=document.getElementById(key);
v[key]=type==='check'?Number(e.checked):Number(type==='style3'?e.dataset.value:e.value)}let r=await fetch('/api',{method:'POST',headers:{'Content-Type':'application/json',
'X-NR-Token':token},body:JSON.stringify(v)});let x=await r.json();document.getElementById('status').textContent=r.ok?'已应用':'失败：'+x.error;
if(r.ok)load()};async function updateHealth(){try{let r=await fetch('/health');let h=await r.json();
const names=['已合成 NR','NR 关闭','输入格式/尺寸不支持','源提交或消费顺序未证实','入口未知','NR 处理错误','便携运行时缺失'];
document.getElementById('health').textContent=`状态：${names[h.status]||h.status}；已处理 ${h.processed_frames} 帧；`+
`上一帧 ${h.last_ms.toFixed(2)} ms；平均 ${h.average_ms.toFixed(2)} ms；估算 ${h.estimated_fps.toFixed(1)} 帧/秒；`+
(h.stages?`桥前置与合成提交 ${h.stages.bridge_pre_sr_average_ms.toFixed(2)} ms（准备 ${h.stages.prep_average_ms.toFixed(2)}，提交 ${h.stages.composite_submit_average_ms.toFixed(2)}）；NR 运行时 ${h.stages.host_average_ms.toFixed(2)} ms；含 XeSS 的尾段 ${h.stages.tail_average_ms.toFixed(2)} ms；`:'')+
`入口计数 DLSS ${h.intercepts.DLSS} / FSR ${h.intercepts.FSR} / XeSS ${h.intercepts.XeSS}`}
catch(e){document.getElementById('health').textContent=e.message}}
let gpuBusy=false;
async function gpuState(){try{const c=await(await fetch('/api/gpu-handoff')).json();
document.getElementById('gpuSwitch').hidden=c==null;if(c==null)return;
if(!gpuBusy)document.getElementById('gpuRequested').checked=c.requested;
document.getElementById('gpuState').textContent=(c.requested?'已请求开启':'已请求关闭')+'；'+(c.armed&&c.cap_healthy?'能力已确认':'继续 CPU 准备等待')+'；跳过准备等待 '+c.prepared_bypasses+' 次，前帧复用等待 '+c.previous_consumer_waits+' 次';}catch(e){document.getElementById('gpuState').textContent=e.message}}
document.getElementById('gpuRequested').onchange=async()=>{const box=document.getElementById('gpuRequested'), requested=box.checked;gpuBusy=true;box.disabled=true;
try{const r=await fetch('/api/gpu-handoff',{method:'POST',headers:{'Content-Type':'application/json','X-NR-Token':token},body:JSON.stringify({requested})});if(!r.ok)throw Error('请求失败');}
catch(e){box.checked=!requested;document.getElementById('status').textContent=e.message}finally{gpuBusy=false;box.disabled=false;gpuState()}};
gpuState();setInterval(gpuState,1000);load().catch(e=>document.getElementById('status').textContent=e.message);
updateHealth();setInterval(updateHealth,1000)</script></html>"""


def start(log_dir):
    global _server
    if _server is not None:
        return _server.server_port
    token = secrets.token_hex(24)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _reply(self, code, body, kind="application/json"):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", kind + "; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                return self._reply(200, _PAGE.replace("__TOKEN__", token), "text/html")
            if self.path == "/api":
                try:
                    return self._reply(200, json.dumps(_values(get_controls())))
                except Exception as exc:
                    return self._reply(500, json.dumps({"error": str(exc)}))
            if self.path == "/health":
                try:
                    return self._reply(200, json.dumps(health()))
                except Exception as exc:
                    return self._reply(500, json.dumps({"error": str(exc)}))
            if self.path == "/api/gpu-handoff":
                return self._reply(200, json.dumps(get_gpu_handoff_state()))
            self._reply(404, "{}")

        def do_POST(self):
            if self.path not in ("/api", "/api/gpu-handoff") or self.headers.get("X-NR-Token") != token:
                return self._reply(403, "{}")
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096:
                    raise ValueError("Invalid request size")
                values = json.loads(self.rfile.read(size))
                if self.path == "/api/gpu-handoff":
                    if not isinstance(values, dict) or set(values)!={"requested"}:
                        raise ValueError("Expected GPU handoff request")
                    set_gpu_handoff_requested(values["requested"])
                    self._reply(200, json.dumps(get_gpu_handoff_state()))
                else:
                    self._reply(200, json.dumps(_values(set_controls(values))))
            except (ValueError, TypeError, RuntimeError) as exc:
                self._reply(400, json.dumps({"error": str(exc)}))

    class Server(ThreadingHTTPServer):
        allow_reuse_address = False

    _server = Server(("127.0.0.1", 0), Handler)
    _server.daemon_threads = True
    thread = threading.Thread(target=_server.serve_forever, name="Cyberpunk NR controls", daemon=True)
    thread.start()
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "cyberpunk-nr-control-url.txt").write_text(
        f"http://127.0.0.1:{_server.server_port}/\n", encoding="ascii")
    return _server.server_port
