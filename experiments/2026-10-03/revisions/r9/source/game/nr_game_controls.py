"""Loopback-only controls for the experimental RE8 NR host.

HTTP threads only update a small validated configuration. They never touch XPU,
Triton, Python model state, or game frame textures.
"""
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import secrets
import threading

from html import escape
from numeric_game_profiles_720_v1 import (
    EXPERIMENTS_720, EXPERIMENT_LABELS, FUSED_REPLAY_PROFILES, HIT_FIELDS,
    PROFILES, combined_mode_options, conflicting_profiles)


class LocalControlServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR can permit two live listeners on the same port.
    # A second NR host must choose a different port, never attach to a game
    # process's existing control endpoint.
    allow_reuse_address = False


@dataclass(frozen=True)
class Settings:
    enabled: bool = True
    display_strength: float = 1.0
    style: int = 0
    input_size: int = 256
    history_mode: str = "reference"
    graph_replay: bool = False
    model_intensity: float = 1.0
    local_tone: float = 1.0
    local_structure: float = 1.0
    auto_mask: bool = False
    skin_structure: float | None = None
    backend_variant: str = "standard"
    experiment_720: str = "baseline"
    optimizations_720: tuple[str, ...] = ()


class ControlPanel:
    def __init__(self, native_apply, *, modes=((256, 0),), url_file=None,
                 metrics_read=None, route="post-xess-legacy",
                 history_modes=("reference", "zero_motion", "reset"),
                 graph_available=False, graph_modes=None, health_read=None,
                 geometry_read=None, timing_read=None, timing_apply=None,
                 validation_read=None, validation_apply=None,
                 lifecycle_read=None, lifecycle_apply=None,
                 bridge_cache_read=None, bridge_cache_apply=None,
                 available_experiments_720=("baseline",),
                 bridge_pool_read=None, bridge_pool_apply=None,
                 gpu_handoff_read=None, gpu_handoff_apply=None):
        self._native_apply = native_apply
        self._metrics_read = metrics_read
        self._health_read = health_read
        self._geometry_read = geometry_read
        if (timing_read is None) != (timing_apply is None):
            raise ValueError("Timing read and apply callbacks must be paired")
        self._timing_read = timing_read
        self._timing_apply = timing_apply
        if (validation_read is None) != (validation_apply is None):
            raise ValueError("Validation read and apply callbacks must be paired")
        self._validation_read = validation_read
        self._validation_apply = validation_apply
        if (lifecycle_read is None) != (lifecycle_apply is None):
            raise ValueError("Lifecycle read and apply callbacks must be paired")
        self._lifecycle_read = lifecycle_read
        self._lifecycle_apply = lifecycle_apply
        if (bridge_cache_read is None) != (bridge_cache_apply is None):
            raise ValueError("Bridge cache callbacks must be paired")
        self._bridge_cache_read=bridge_cache_read
        self._bridge_cache_apply=bridge_cache_apply
        if (bridge_pool_read is None) != (bridge_pool_apply is None):
            raise ValueError("Bridge resource pool callbacks must be paired")
        self._bridge_pool_read=bridge_pool_read
        self._bridge_pool_apply=bridge_pool_apply
        if (gpu_handoff_read is None) != (gpu_handoff_apply is None):
            raise ValueError("GPU handoff callbacks must be paired")
        self._gpu_handoff_read=gpu_handoff_read
        self._gpu_handoff_apply=gpu_handoff_apply
        self._modes = frozenset(modes)
        if route not in ("post-xess-legacy", "pre-xess-fullsize"):
            raise ValueError("Unknown NR route")
        if route == "post-xess-legacy" and (256, 0) not in self._modes:
            raise ValueError("The active NR256 mode must be available")
        if route == "pre-xess-fullsize" and not {(360, 0), (480, 0), (540, 0), (720, 0)} <= self._modes:
            raise ValueError("The original-size 360p/480p/540p/720p modes must be available")
        self._route = route
        available = tuple(dict.fromkeys(available_experiments_720))
        if (not available or available[0] != "baseline" or
                set(available) - set(EXPERIMENTS_720)):
            raise ValueError("Invalid 720p experiment availability")
        self._experiments_720 = available if route == "pre-xess-fullsize" else ("baseline",)
        self._history_modes = tuple(history_modes) if route == "pre-xess-fullsize" else ("reference",)
        self._graph_modes = (frozenset(graph_modes) if graph_modes is not None else
                             frozenset((360,)) if graph_available else frozenset())
        if route != "pre-xess-fullsize":
            self._graph_modes = frozenset()
        if any(type(size) is not int or (size, 0) not in self._modes
               for size in self._graph_modes):
            raise ValueError("Graph modes must be available game input heights")
        self._graph_available = bool(self._graph_modes)
        if (not self._history_modes or "reference" not in self._history_modes or
                set(self._history_modes) - {"reference", "zero_motion", "reset", "fused"}):
            raise ValueError("Invalid game history modes")
        self._lock = threading.Lock()
        if route == "pre-xess-fullsize":
            # Prefer the visually accepted live route on new game processes.
            # The old 480p reference-motion route remains available for A/B.
            self._settings = (Settings(input_size=360, history_mode="fused",
                                       graph_replay=True)
                              if self._graph_available and "fused" in self._history_modes
                              else Settings(input_size=480,
                                            history_mode="zero_motion" if
                                            "zero_motion" in self._history_modes else "reference"))
        else:
            self._settings = Settings()
        self._active_mode = None
        self._token = secrets.token_hex(24)
        try:
            self._server = LocalControlServer(("127.0.0.1", 8765), self._handler())
        except OSError:
            self._server = LocalControlServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="NR loopback controls", daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{self._server.server_port}/"
        if url_file is not None:
            Path(url_file).write_text(self.url + "\n", encoding="ascii")

    def snapshot(self):
        with self._lock:
            return self._settings

    def status(self):
        with self._lock:
            status = {"settings": asdict(self._settings),
                    "active_mode": self._active_mode,
                    "available_modes": [{"input_size": size, "style": style}
                                        for size, style in sorted(self._modes)],
                    "available_history_modes": self._history_modes,
                    "graph_available": self._graph_available,
                    "graph_modes": sorted(self._graph_modes),
                    "available_experiments_720": self._experiments_720,
                    "route": self._route,
                    "resolution_note": (
                        "Game render size is detected per XeSS source frame; NR uses a "
                        "fixed 1280x720, 960x540, 854x480 or 640x360 model image before XeSS."
                        if self._route == "pre-xess-fullsize" else
                        "NR network input is square; 256 uses 256x144 active pixels, 512 uses 512x288.")}
        # Native timing must be queried after releasing the settings lock: the
        # game thread can hold the GPU host lock while it marks a mode active.
        status["processing"] = self._metrics_read() if self._metrics_read else None
        status["health"] = (self._health_read() if self._health_read else
                            {"failed": False, "reason": None})
        status["source_geometry"] = self._geometry_read() if self._geometry_read else None
        status["timing_enabled"] = self._timing_read() if self._timing_read else None
        status["validation_batch"] = self.validation_batch_state()
        status["lifecycle_audit"] = self.lifecycle_audit_state()
        status["bridge_cache"] = self._bridge_cache_read() if self._bridge_cache_read else None
        status["bridge_resource_pool"] = self._bridge_pool_read() if self._bridge_pool_read else None
        status["gpu_handoff"] = self._gpu_handoff_read() if self._gpu_handoff_read else None
        return status

    def validation_batch_state(self):
        return self._validation_read() if self._validation_read else None

    def update_validation(self, values):
        if self._validation_apply is None:
            raise ValueError("Validation switch is unavailable on this route")
        if not isinstance(values, dict) or set(values) != {"enabled"} or type(values["enabled"]) is not bool:
            raise ValueError("Expected a boolean validation setting")
        self._validation_apply(values["enabled"])
        # Avoid status/native callbacks on this CPU-only settings transaction.
        return self.validation_batch_state()

    def lifecycle_audit_state(self):
        return self._lifecycle_read() if self._lifecycle_read else None

    def update_lifecycle(self, values):
        if self._lifecycle_apply is None:
            raise ValueError("Lifecycle switch is unavailable on this route")
        if not isinstance(values, dict) or set(values) != {"enabled"} or type(values["enabled"]) is not bool:
            raise ValueError("Expected a boolean lifecycle setting")
        self._lifecycle_apply(values["enabled"])
        return self.lifecycle_audit_state()

    def update_bridge_cache(self, values):
        if self._bridge_cache_apply is None:
            raise ValueError("Bridge cache is unavailable on this route")
        if not isinstance(values, dict) or set(values)!={"enabled"} or type(values["enabled"]) is not bool:
            raise ValueError("Expected a boolean bridge cache setting")
        self._bridge_cache_apply(values["enabled"])
        return self._bridge_cache_read()

    def update_bridge_pool(self, values):
        if self._bridge_pool_apply is None:
            raise ValueError("Bridge resource pool is unavailable on this route")
        if not isinstance(values, dict) or set(values)!={"enabled"} or type(values["enabled"]) is not bool:
            raise ValueError("Expected a boolean bridge resource pool setting")
        self._bridge_pool_apply(values["enabled"])
        return self._bridge_pool_read()

    def update_timing(self, values):
        if self._timing_apply is None:
            raise ValueError("Timing switch is unavailable on this route")
        if not isinstance(values, dict) or set(values) != {"enabled"} or type(values["enabled"]) is not bool:
            raise ValueError("Expected a boolean timing setting")
        self._timing_apply(values["enabled"])
        return self.status()

    def update_gpu_handoff(self, values):
        if self._gpu_handoff_apply is None:
            raise ValueError("GPU handoff is unavailable on this route")
        if not isinstance(values, dict) or set(values)!={"requested"} or type(values["requested"]) is not bool:
            raise ValueError("Expected a boolean GPU handoff request")
        self._gpu_handoff_apply(values["requested"])
        return self._gpu_handoff_read()

    def mark_active(self, size: int, style: int, history_mode: str = "reference",
                    graph_replay: bool = False, model_controls=None,
                    graph_requested: bool | None = None, backend_variant="standard",
                    experiment_720="baseline", optimizations_720=()):
        with self._lock:
            self._active_mode = {"input_size": size, "style": style,
                                 "history_mode": history_mode,
                                 "graph_replay": graph_replay,
                                 "graph_requested": (graph_replay if graph_requested is None
                                                     else graph_requested),
                                 "model_controls": model_controls}
            self._active_mode["backend_variant"] = backend_variant
            self._active_mode["experiment_720"] = experiment_720
            self._active_mode["optimizations_720"] = tuple(optimizations_720)

    def update(self, values):
        if self._health_read and self._health_read().get("failed"):
            raise RuntimeError("NR 推理已停止；当前进程无法恢复 GPU，请保存并重启游戏")
        required = {"enabled", "display_strength", "style", "input_size"}
        if self._route == "pre-xess-fullsize":
            required.update(("history_mode", "graph_replay", "model_intensity",
                             "local_tone", "local_structure", "auto_mask",
                             "skin_structure", "backend_variant"))
        allowed = required | ({"experiment_720", "optimizations_720"}
                              if self._route == "pre-xess-fullsize" else set())
        if not isinstance(values, dict) or not required <= set(values) or not set(values) <= allowed:
            raise ValueError("Expected the controls for this NR route")
        enabled, strength = values["enabled"], values["display_strength"]
        style, size = values["style"], values["input_size"]
        if type(enabled) is not bool:
            raise ValueError("enabled must be true or false")
        if type(strength) not in (float, int) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("display_strength must be between 0 and 1")
        if type(style) is not int or type(size) is not int or (size, style) not in self._modes:
            raise ValueError("This style/input resolution has not passed offline precompile and GPU validation")
        history_mode = values.get("history_mode", "reference")
        if type(history_mode) is not str or history_mode not in self._history_modes:
            raise ValueError("This history mode has not passed validation")
        graph_replay = values.get("graph_replay", False)
        if type(graph_replay) is not bool or (graph_replay and not self._graph_available):
            raise ValueError("Game graph replay has not passed validation")
        if graph_replay and size not in self._graph_modes:
            raise ValueError("Game graph replay has not passed validation at this input height")
        model_values = {}
        if self._route == "pre-xess-fullsize":
            if values["backend_variant"] not in ("standard", "unrounded"):
                raise ValueError("Unknown fast backend variant")
            experiment = values.get("experiment_720", "baseline")
            if type(experiment) is not str or experiment not in self._experiments_720:
                raise ValueError("This 720p experiment has not passed offline validation")
            if experiment != "baseline" and (size != 720 or values["backend_variant"] != "unrounded"):
                raise ValueError("720p experiments require 720p and the unrounded fast backend")
            if experiment == "history_compact" and history_mode != "fused":
                raise ValueError("Compact history requires fused sampling and real motion")
            optimizations = values.get("optimizations_720", ())
            combined_mode_options(optimizations)
            if set(optimizations) - set(self._experiments_720):
                raise ValueError("勾选项尚未准备好本机缓存")
            if optimizations and (size != 720 or values["backend_variant"] != "unrounded" or experiment != "c512_k8"):
                raise ValueError("勾选优化需要 720p 去舍入版的 C512＋K8 基准")
            if experiment in FUSED_REPLAY_PROFILES or optimizations:
                if history_mode != "fused" or not graph_replay:
                    raise ValueError("该数值实验需要融合采样＋真实运动和计算图重放")
                geometry = self._geometry_read() if self._geometry_read else None
                if geometry and (geometry.get("width"), geometry.get("height")) != (1280, 720):
                    raise ValueError("该数值实验需要游戏实际提供 1280×720 输入")
            if type(values["auto_mask"]) is not bool:
                raise ValueError("auto_mask must be true or false")
            model_values["auto_mask"] = values["auto_mask"]
            for name in ("model_intensity", "local_tone", "local_structure", "skin_structure"):
                value = values[name]
                if name == "skin_structure" and value is None:
                    model_values[name] = None
                    continue
                if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 2:
                    raise ValueError(name + " must be between 0 and 2")
                model_values[name] = float(value)
        setting = Settings(enabled, float(strength), style, size, history_mode,
                           graph_replay, **model_values,
                           backend_variant=values.get("backend_variant", "standard"),
                           experiment_720=values.get("experiment_720", "baseline"),
                           optimizations_720=tuple(sorted(values.get("optimizations_720", ()))))
        with self._lock:
            if not self._native_apply(setting.enabled, setting.display_strength):
                raise RuntimeError("Native NR control bridge rejected the setting")
            self._settings = setting
            self._active_mode = None
        return self.status()

    def close(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    def _handler(self):
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def _reply(self, status, body, content_type="application/json; charset=utf-8"):
                data = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Content-Security-Policy",
                    "default-src 'none'; connect-src 'self'; style-src 'unsafe-inline'; "
                    f"script-src 'nonce-{panel._token}'; frame-ancestors 'none'")
                self.end_headers()
                self.wfile.write(data)

            def _valid_host(self):
                return self.headers.get("Host") == f"127.0.0.1:{panel._server.server_port}"

            def do_GET(self):
                if not self._valid_host():
                    self._reply(403, '{"error":"Loopback host required"}')
                elif self.path == "/api/state":
                    self._reply(200, json.dumps(panel.status(), ensure_ascii=False))
                elif self.path == "/api/lifecycle-audit":
                    self._reply(200, json.dumps(panel.lifecycle_audit_state(), ensure_ascii=False))
                elif self.path == "/api/validation":
                    self._reply(200, json.dumps(panel.validation_batch_state(), ensure_ascii=False))
                elif self.path == "/":
                    self._reply(200, _page(panel._token, panel._route), "text/html; charset=utf-8")
                else:
                    self._reply(404, '{"error":"Not found"}')

            def do_POST(self):
                origin = f"http://127.0.0.1:{panel._server.server_port}"
                if (not self._valid_host() or self.path not in ("/api/state", "/api/timing", "/api/validation", "/api/lifecycle-audit", "/api/bridge-cache", "/api/bridge-resource-pool", "/api/gpu-handoff") or
                    self.headers.get("Origin") != origin or
                    self.headers.get("X-NR-Token") != panel._token or
                    self.headers.get("Content-Type") != "application/json"):
                    self._reply(403, '{"error":"Invalid local control request"}')
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 1 <= length <= 2048:
                        raise ValueError("Invalid request size")
                    values = json.loads(self.rfile.read(length))
                    result = (panel.update_timing(values) if self.path == "/api/timing"
                              else panel.update_validation(values) if self.path == "/api/validation"
                              else panel.update_bridge_cache(values) if self.path == "/api/bridge-cache"
                              else panel.update_bridge_pool(values) if self.path == "/api/bridge-resource-pool"
                              else panel.update_gpu_handoff(values) if self.path == "/api/gpu-handoff"
                              else panel.update_lifecycle(values) if self.path == "/api/lifecycle-audit"
                              else panel.update(values))
                except (ValueError, UnicodeError, json.JSONDecodeError, RuntimeError) as error:
                    self._reply(422, json.dumps({"error": str(error)}, ensure_ascii=False))
                except Exception as error:
                    self._reply(503, json.dumps({"error": str(error)}, ensure_ascii=False))
                else:
                    self._reply(200, json.dumps(result, ensure_ascii=False))

        return Handler


def _page(token, route="post-xess-legacy"):
    if route == "pre-xess-fullsize":
        return _fullsize_page(token)
    return """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>B580 NR 实机控制</title><style>
body{font:16px system-ui,sans-serif;background:#101827;color:#eef4ff;max-width:660px;margin:40px auto;padding:0 20px}
h1{font-size:24px}label{display:block;margin:22px 0 8px}select,input[type=range]{width:100%}
select{padding:10px;background:#26344b;color:white;border:1px solid #64758d;border-radius:6px}
.row{display:flex;justify-content:space-between;gap:15px}.hint{color:#aebed2;font-size:14px}
.panel{background:#1b283b;padding:24px;border-radius:12px}button{padding:12px 18px;background:#4b89d4;color:white;border:0;border-radius:6px;cursor:pointer}
#status{min-height:2em;margin-top:20px}#telemetry{white-space:pre-line;font-size:18px;margin-top:18px}
</style><div class="panel"><h1>B580 NR 旧链调试</h1>
<p class="hint">当前仍是 XeSS 后的 256／512 方形输入实验链。960×540 原尺寸 NR、480p／360p 输入滑块与 XeSS 前接入尚未启用。</p>
<label><input id="enabled" type="checkbox" checked> 启用 NR</label>
<label for="strength">显示混合强度 <span id="strengthValue">1.00</span></label>
<input id="strength" type="range" min="0" max="1" step="0.05" value="1">
<p class="hint">0 为原画，1 为完整 NR 显示结果；不改变模型内部强度。</p>
<label for="style">NR 风格</label><select id="style"><option value="0">标准（0）</option><option value="1">自然（1）</option><option value="2">电影（2）</option></select>
<label for="size">NR 网络输入尺寸</label><select id="size"><option value="256">256×256（有效画面 256×144）</option><option value="512">512×512（有效画面 512×288）</option></select>
<p class="hint">256／风格 0 是当前优化过的快速档。其他档位使用较慢的参考执行链，可能明显降低帧率；首次切换需加载模型，但不会在游戏里编译。</p>
<p class="hint">仅可选已离线编译并通过验证的组合；不支持的组合会明确拒绝。</p>
<button id="apply">应用设置</button><div id="status" role="status"></div><div id="active" class="hint"></div>
<div id="telemetry" role="status" aria-live="off">等待处理计时</div>
<p class="hint">统计最近最多 32 个完成的 NR 帧，包含帧复制、NR、合成和 GPU 等待；不含游戏渲染和 Present。吞吐帧率是 1000 ÷ 平均耗时，不代表游戏实际 FPS。切换设置后重新统计。</p></div>
<script nonce=""" + token + """>const token='""" + token + """';
const $=id=>document.getElementById(id);
const show=s=>{$('status').textContent=s};
let available=[];
function values(){return {enabled:$('enabled').checked,display_strength:Number($('strength').value),style:Number($('style').value),input_size:Number($('size').value)}}
function filterModes(){const size=Number($('size').value);let styles=available.filter(m=>m.input_size===size).map(m=>m.style);
for(const option of $('style').options)option.disabled=!styles.includes(Number(option.value));
if(!styles.includes(Number($('style').value)))$('style').value=String(styles[0]);}
async function load(){let r=await fetch('/api/state');let x=await r.json();let s=x.settings;
available=x.available_modes;
for(const option of $('size').options)option.disabled=!available.some(m=>m.input_size===Number(option.value));
$('enabled').checked=s.enabled;$('strength').value=s.display_strength;$('strengthValue').textContent=s.display_strength.toFixed(2);
$('style').value=s.style;$('size').value=s.input_size;filterModes();
show('可用组合：'+x.available_modes.map(m=>m.input_size+' / 风格 '+m.style).join('，'))}
async function active(){try{let x=await(await fetch('/api/state')).json();let m=x.active_mode;
$('active').textContent=m?'游戏实际执行：'+m.input_size+' / 风格 '+m.style:'等待游戏首帧执行';
let p=x.processing;
$('telemetry').textContent=!x.settings.enabled?'NR 已关闭，计时暂停':
    p&&p.frames>0?'处理耗时：'+p.average_ms.toFixed(2)+' ms/帧（最近 '+Math.min(p.frames,32)+' 帧；上一帧 '+p.last_ms.toFixed(2)+' ms）；估算吞吐：'+p.estimated_fps.toFixed(1)+' 帧/秒':
    '等待完成的 NR 帧'}catch(e){$('active').textContent='游戏连接已断开';$('telemetry').textContent='计时不可用'}}
$('strength').oninput=()=>$('strengthValue').textContent=Number($('strength').value).toFixed(2);
$('size').onchange=()=>{filterModes();apply()};
async function apply(){try{let r=await fetch('/api/state',{method:'POST',headers:{'Content-Type':'application/json','X-NR-Token':token},body:JSON.stringify(values())});let x=await r.json();if(!r.ok)throw Error(x.error);show('已应用：'+(x.settings.enabled?'NR 开':'NR 关')+'，强度 '+x.settings.display_strength.toFixed(2)+'，风格 '+x.settings.style+'，输入 '+x.settings.input_size);active()}catch(e){show('未应用：'+e.message)}}
$('enabled').onchange=apply;$('style').onchange=apply;$('strength').onchange=apply;$('apply').onclick=apply;
load().then(active).catch(e=>show(e.message));setInterval(active,1000);</script></html>"""


def _fullsize_page(token):
    """UI for the pre-XeSS source-frame route; never labels the legacy route as this one."""
    options = "".join(f'<option value="{escape(key)}">{escape(label)}</option>'
                      for key, label in EXPERIMENT_LABELS.items())
    names_json = json.dumps(dict(EXPERIMENT_LABELS), ensure_ascii=False)
    hits_json = json.dumps(dict(HIT_FIELDS))
    fused_json = json.dumps(sorted(FUSED_REPLAY_PROFILES))
    checkboxes = "".join(
        f'<label><input type="checkbox" disabled data-optimization="{escape(key)}" '
        f'data-pending-validation="{str(profile.pending_validation).lower()}"> '
        f'{escape(profile.label.removeprefix("基准＋"))}'
        f'<span class="hint" data-readiness> — 等待离线准备</span></label>'
        for key, profile in PROFILES.items())
    conflicts_json = json.dumps(conflicting_profiles())
    return """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>B580 NR 游戏输入控制</title><style>
body{font:16px system-ui,sans-serif;background:#101827;color:#eef4ff;max-width:660px;margin:40px auto;padding:0 20px}
h1{font-size:24px}label{display:block;margin:22px 0 8px}input[type=range],select{width:100%}
[hidden]{display:none!important}
select{padding:10px;background:#26344b;color:white;border:1px solid #64758d;border-radius:6px}
.hint{color:#aebed2;font-size:14px}.panel{background:#1b283b;padding:24px;border-radius:12px}
button{padding:12px 18px;background:#4b89d4;color:white;border:0;border-radius:6px;cursor:pointer}
.styles{display:flex;gap:8px;margin-top:8px}.styles button{flex:1;background:#26344b;border:1px solid #64758d}
.styles button[aria-pressed="true"]{background:#4b89d4;border-color:#a5d0ff}.styles button:disabled{opacity:.45;cursor:not-allowed}
fieldset{border:0;padding:0;margin:20px 0}legend{margin-bottom:8px}details{margin:20px 0}
#status{min-height:2em;margin-top:20px}#telemetry{white-space:pre-line;font-size:18px;margin-top:18px}
</style><div class="panel"><h1>B580 NR 游戏输入控制</h1>
<p>游戏 <strong id="sourceGeometry">等待输入</strong> → 可选尺寸快速 NR → XeSS 超分</p>
<label><input id="enabled" type="checkbox" checked> 启用 NR</label>
<label id="bridgeCacheSwitch" hidden><input id="bridgeCacheEnabled" type="checkbox" checked> 复用桥的着色器与管线</label>
<p id="bridgeCacheStatus" class="hint" hidden></p>
<label id="bridgePoolSwitch" hidden><input id="bridgePoolEnabled" type="checkbox"> 复用桥的中间纹理（实验）</label>
<p id="bridgePoolStatus" class="hint" hidden></p>
<label id="gpuHandoffSwitch" hidden><input id="gpuHandoffRequested" type="checkbox"> GPU 队列交接（实验）</label>
<p id="gpuHandoffStatus" class="hint" hidden></p>
<label id="timingSwitch" hidden><input id="timingEnabled" type="checkbox" checked> 记录逐帧计时</label>
<p id="timingHint" class="hint" hidden>关闭后 NR 继续运行；网页不再记录处理耗时。请用 RTSS 在同一场景比较开关前后的游戏帧率。</p>
<label id="validationSwitch" hidden><input id="validationBatchEnabled" type="checkbox" checked> 合并重复校验</label>
<p id="validationHint" class="hint" hidden>合并同一帧的重复校验；与下方固定检查迁移开关分别比较。</p>
<label id="lifecycleSwitch" hidden><input id="lifecycleEnabled" type="checkbox"> 固定检查移到加载／建图（实验）</label>
<p id="lifecycleHint" class="hint" hidden>切换会重新选档、重建计算图并重置一次历史；稳定重放跳过固定权重、源码和配置扫描。动态输入与同步约束保留。</p>
<p id="lifecycleStatus" class="hint" hidden></p>
<label for="backendVariant">快速后端</label>
<select id="backendVariant"><option value="standard">现役版：保留 FP8 舍入</option>
<option value="unrounded">实验版：去舍入组合</option></select>
<p class="hint">切换后会重建模型并重置 NR 历史，第一次切换可能停顿。480p／540p 包含解码激活去舍入；720p 可通过下方独立勾选启用。本选项并不代表全网所有 FP8 规则都已移除。两版画面可能不同，请分别看画面和处理耗时。</p>
<select id="experiment720" hidden>""" + options + """</select>
<fieldset id="optimizations720"><legend>720p 独立优化（可叠加勾选）</legend>
<p class="hint">底座：C512＋K8。全部不勾就是对照基准。</p>""" + checkboxes + """</fieldset>
<p class="hint">不同计算步骤可以叠加；有包含关系的选项合并执行一次。同一步运算的不同算法需二选一，勾选时会自动取消另一种。灰色项目尚未完成离线准备，不能启用。画面可能变化；用同一场景的 RTSS 帧时间比较。改变勾选会释放旧模型、重建计算图并重置历史，首帧可能暂停。</p>
<label for="size">NR 输入高度 <strong id="sizeValue">480p</strong></label>
<input id="size" type="range" min="0" max="3" step="1" value="1" list="sizeTicks">
<datalist id="sizeTicks"><option value="0" label="360p"></option><option value="1" label="480p"></option><option value="2" label="540p"></option><option value="3" label="720p"></option></datalist>
<p class="hint">NR 模型档位：360p = 640×360；480p = 854×480；540p = 960×540；720p = 1280×720。NR 结果会合回检测到的游戏输入尺寸，再交给 XeSS。若游戏当前是 540p，选择 720p 会先放大输入，无法凭空增加原画细节，且处理更慢。720p 内部画布是实验推导档，请关注画面和耗时。</p>
<fieldset><legend>NR 模型风格</legend><div class="styles" id="styleButtons">
<button type="button" data-style="0" aria-pressed="true">标准</button>
<button type="button" data-style="1" aria-pressed="false">自然</button>
<button type="button" data-style="2" aria-pressed="false">电影</button></div></fieldset>
<label for="modelIntensity">模型内强度 <strong id="modelIntensityValue">1.00</strong></label>
<input id="modelIntensity" type="range" min="0" max="2" step="0.05" value="1">
<label for="localTone">局部色调 <strong id="localToneValue">1.00</strong></label>
<input id="localTone" type="range" min="0" max="2" step="0.05" value="1">
<label for="localStructure">局部结构 <strong id="localStructureValue">1.00</strong></label>
<input id="localStructure" type="range" min="0" max="2" step="0.05" value="1">
<label><input id="autoMask" type="checkbox"> 自动遮罩</label>
<label><input id="separateSkin" type="checkbox"> 单独调整皮肤结构</label>
<label for="skinStructure">皮肤结构 <strong id="skinStructureValue">1.00</strong></label>
<input id="skinStructure" type="range" min="0" max="2" step="0.05" value="1" disabled>
<p class="hint">皮肤结构只在自动遮罩启用时生效；未单独调整时继承局部结构。模型参数改变时可能重置 NR 历史；强度单独改变会保留历史。标准风格的模型内强度在 1 及以上不会继续增强。</p>
<details><summary>输出后混合（额外实验控制）</summary>
<label for="strength">显示效果混合 <strong id="strengthValue">1.00</strong></label>
<input id="strength" type="range" min="0" max="1" step="0.05" value="1">
<p class="hint">0 为游戏原画，1 为完整 NR 结果；与模型内强度不同。</p></details>
<label for="history">历史与运动处理</label>
<select id="history"><option value="reference">标准：真实运动＋原历史采样</option>
<option value="zero_motion">诊断：保留历史、忽略运动</option>
<option value="reset">诊断：每帧重置历史</option>
<option value="fused">融合采样＋真实运动</option></select>
<p class="hint">切换时会重置一次历史；诊断档会改变画面。融合采样的 720p 画布需单独验证。</p>
<label><input id="graph" type="checkbox"> 计算图重放</label>
<p class="hint">独立于分辨率和历史模式。切换输入高度时先回到普通执行；在所选高度开启图重放会建立该尺寸的图，首帧可能停顿数秒。标准风格模型强度低于 1 的新精度变体会安全回退普通执行；实际路径见下方状态。</p>
<button id="apply">应用设置</button><div id="status" role="status"></div><div id="active" class="hint"></div>
<div id="telemetry" role="status" aria-live="off">等待处理计时</div>
<div id="bridgeTelemetry" class="hint" style="white-space:pre-line" hidden></div>
<p class="hint">处理时间覆盖 NR 输入准备、推理、合成和 GPU 等待；估算吞吐 = 1000 ÷ 平均毫秒，并非游戏实际 FPS。改变档位后重新统计。</p></div>
<script nonce=""" + token + """>const token='""" + token + """';
const $=id=>document.getElementById(id), heights=[360,480,540,720], styleNames=['标准','自然','电影'];
const experimentNames=""" + names_json + """;
const hitFields=""" + hits_json + """, fusedReplayProfiles=""" + fused_json + """;
const conflicts=""" + conflicts_json + """;
const optimizationBoxes=()=>[...$('optimizations720').querySelectorAll('[data-optimization]')];
let available=[], historyAvailable=[], graphModes=[], experimentAvailable=['baseline'], busy=false, pendingApply=null, selectedStyle=0;
let validationBusy=false, bridgeCacheBusy=false, bridgePoolBusy=false, gpuHandoffBusy=false;
let lifecycleBusy=false;
const show=s=>{$('status').textContent=s};
function refreshLabels(){
  $('sizeValue').textContent=heights[Number($('size').value)]+'p';
  $('strengthValue').textContent=Number($('strength').value).toFixed(2);
  for(const [control,label] of [['modelIntensity','modelIntensityValue'],
       ['localTone','localToneValue'],['localStructure','localStructureValue'],
       ['skinStructure','skinStructureValue']])
    $(label).textContent=Number($(control).value).toFixed(2);
  $('skinStructure').disabled=!$('separateSkin').checked||!$('autoMask').checked;
  for(const button of $('styleButtons').querySelectorAll('button'))
    button.setAttribute('aria-pressed',String(Number(button.dataset.style)===selectedStyle));
}
function selected(){return {enabled:$('enabled').checked,display_strength:Number($('strength').value),
  style:selectedStyle,input_size:heights[Number($('size').value)],backend_variant:$('backendVariant').value,
  experiment_720:$('experiment720').value,
  optimizations_720:optimizationBoxes().filter(box=>box.checked).map(box=>box.dataset.optimization).sort(),
  history_mode:$('history').value,
  graph_replay:$('graph').checked,model_intensity:Number($('modelIntensity').value),
  local_tone:Number($('localTone').value),local_structure:Number($('localStructure').value),
  auto_mask:$('autoMask').checked,
  skin_structure:$('separateSkin').checked?Number($('skinStructure').value):null}}
function supported(size,style){return available.some(m=>m.input_size===size&&m.style===style)}
function constrain(){
  const size=heights[Number($('size').value)], styles=available.filter(m=>m.input_size===size).map(m=>m.style);
  if(!styles.includes(selectedStyle))selectedStyle=styles[0]??0;
  for(const button of $('styleButtons').querySelectorAll('button'))
    button.disabled=!styles.includes(Number(button.dataset.style));
  for(const option of $('history').options)
    option.disabled=!historyAvailable.includes(option.value);
  if($('history').selectedOptions[0]?.disabled)$('history').value='reference';
  $('graph').disabled=!graphModes.includes(size);
  if($('graph').disabled)$('graph').checked=false;
  const experimental=size===720&&$('backendVariant').value==='unrounded';
  $('experiment720').disabled=!experimental;
  if(!experimental)$('experiment720').value='baseline';
  if(experimental&&experimentAvailable.includes('c512_k8'))$('experiment720').value='c512_k8';
  for(const box of optimizationBoxes()){
    const installed=experimentAvailable.includes(box.dataset.optimization);
    box.parentElement.hidden=false;box.disabled=!experimental||!installed;
    const readiness=box.parentElement.querySelector('[data-readiness]');
    if(readiness)readiness.textContent=installed?'':box.dataset.pendingValidation==='true'?' — 待验证':' — 等待离线准备';
    if(!experimental||!installed)box.checked=false;
  }
  if($('history').value!=='fused'&&$('experiment720').value==='history_compact')
    $('experiment720').value='baseline';
  for(const option of $('experiment720').options){
    option.disabled=!experimentAvailable.includes(option.value);
    option.hidden=option.disabled;
  }
  if($('experiment720').selectedOptions[0]?.disabled)$('experiment720').value='baseline';
  if(fusedReplayProfiles.includes($('experiment720').value)&&
     ($('history').value!=='fused'||!$('graph').checked))
    $('experiment720').value=experimentAvailable.includes('c512_k8')?'c512_k8':'baseline';
  if($('history').value!=='fused'||!$('graph').checked)
    for(const box of optimizationBoxes())box.checked=false;
  refreshLabels();
}
async function load(){
  const response=await fetch('/api/state'), data=await response.json();
  if(data.route!=='pre-xess-fullsize')throw Error('页面与游戏运行链路不一致');
  available=data.available_modes;historyAvailable=data.available_history_modes;
  graphModes=data.graph_modes;experimentAvailable=data.available_experiments_720??['baseline'];
  const s=data.settings, index=heights.indexOf(s.input_size);
  if(index<0)throw Error('游戏返回了未登记的 NR 输入高度');
  $('enabled').checked=s.enabled;$('strength').value=s.display_strength;
  $('size').value=index;selectedStyle=s.style;$('history').value=s.history_mode;
  $('backendVariant').value=s.backend_variant;
  $('experiment720').value=s.experiment_720??'baseline';
  for(const box of optimizationBoxes())box.checked=(s.optimizations_720??[]).includes(box.dataset.optimization);
  $('modelIntensity').value=s.model_intensity;$('localTone').value=s.local_tone;
  $('localStructure').value=s.local_structure;$('autoMask').checked=s.auto_mask;
  $('separateSkin').checked=s.skin_structure!==null;
  $('skinStructure').value=s.skin_structure??s.local_structure;
  $('graph').checked=s.graph_replay;constrain();
  $('timingSwitch').hidden=data.timing_enabled===null;
  $('timingHint').hidden=data.timing_enabled===null;
  if(data.timing_enabled!==null)$('timingEnabled').checked=data.timing_enabled;
  $('validationSwitch').hidden=data.validation_batch==null;
  $('validationHint').hidden=data.validation_batch==null;
  if(data.validation_batch!=null)$('validationBatchEnabled').checked=data.validation_batch.enabled;
  updateLifecycleView(data);
  show('可用 NR 输入：'+[...new Set(available.map(m=>m.input_size))].join('p、')+'p');
}
async function active(){
  try{
    const data=await(await fetch('/api/state')).json(), m=data.active_mode,p=data.processing;
    $('sourceGeometry').textContent=data.source_geometry?data.source_geometry.width+'×'+data.source_geometry.height:'等待输入';
    if(data.health?.failed){
      $('active').textContent='NR 已停止：'+(data.health.reason==='GPU_DEVICE_LOST'?'显卡设备丢失':'推理失败')+'；请保存并重启游戏';
      $('telemetry').textContent='当前进程不能继续处理 NR';$('bridgeTelemetry').hidden=true;return;
    }
    const checks=m?.optimizations_720??[];
    const requiredHits=[...(hitFields[m?.experiment_720]??[]),
      ...checks.flatMap(key=>hitFields[key]??[])];
    const hit=requiredHits.length>0&&requiredHits.every(key=>data.health?.[key]);
    $('active').textContent=m?'游戏实际执行：'+m.input_size+'p / '+(m.backend_variant==='unrounded'?'去舍入组合版':'现役版')+' / '+(experimentNames[m.experiment_720]??'现役快速版')+
      (m.experiment_720==='baseline'?'':hit?'（新算子已命中）':'（等待新算子首次命中）')+' / '+styleNames[m.style]+' / '+m.history_mode+
      (checks.length?' / 已勾选：'+checks.map(key=>experimentNames[key]?.replace('基准＋','')??key).join('、'):'')+
      (m.graph_replay?' / 图重放':m.graph_requested?' / 图未就绪，已安全回退普通执行':' / 普通执行')+
      (m.model_controls?' / 模型强度 '+m.model_controls.intensity.toFixed(2):''):'等待游戏首帧执行';
    if(data.timing_enabled!==null)$('timingEnabled').checked=data.timing_enabled;
    if(!validationBusy&&data.validation_batch!=null)
      $('validationBatchEnabled').checked=data.validation_batch.enabled;
    updateLifecycleView(data);
    updateBridgeCacheView(data);
    updateBridgePoolView(data);
    updateGpuHandoffView(data);
    $('telemetry').textContent=!data.settings.enabled?'NR 已关闭，计时暂停':
      data.timing_enabled===false?'逐帧计时已关闭，NR 仍运行；已处理 '+(p?.frames??0)+' 帧。请看 RTSS 帧率':
      p&&p.average_ms>0?'处理耗时：'+p.average_ms.toFixed(2)+' ms/帧（上一帧 '+p.last_ms.toFixed(2)+' ms）；估算吞吐：'+p.estimated_fps.toFixed(1)+' 帧/秒':
      '等待完成的 NR 帧';
    const stages=p?.stages, bridge=$('bridgeTelemetry');
    bridge.hidden=!stages||!data.settings.enabled||data.timing_enabled===false;
    if(!bridge.hidden){
      const r=stages.runtime, o=stages.recording;
      bridge.textContent=
        (o?'网页外 CPU 录制 '+o.record_average_ms.toFixed(2)+' ms（资源 '+o.resources_average_ms.toFixed(2)+
        '，颜色/运动管线准备 '+o.hdr_initialize_average_ms.toFixed(2)+'，XeSS 命令录制 '+o.xess_record_average_ms.toFixed(2)+'）\\n'+
        '录制到提交间隔 '+o.record_to_submit_gap_average_ms.toFixed(2)+' ms；录制到完成跨度 '+o.record_to_retire_span_average_ms.toFixed(2)+
        ' ms（包含游戏工作与等待，不等于完整桥耗时或游戏帧时间）\\n':'')+
        '桥已知阶段：'+(stages.bridge_known_average_ms==null?'等待运行时分段':stages.bridge_known_average_ms.toFixed(2)+' ms/帧')+
        '（输入准备 '+stages.prep_average_ms.toFixed(2)+'＋纹理导入 '+(r?r.prepare_average_ms.toFixed(2):'…')+
        '＋纹理导出 '+(r?r.export_average_ms.toFixed(2):'…')+'＋合成提交 '+stages.composite_submit_average_ms.toFixed(2)+'）\\n'+
        'NR 模型调用：'+(r?r.model_average_ms.toFixed(2):'…')+' ms/帧；运行时总调用：'+stages.host_average_ms.toFixed(2)+
        ' ms/帧；尾段：'+stages.tail_average_ms.toFixed(2)+' ms/帧（含 GPU 合成、XeSS 与完成等待）\\n'+
        '各段取最近 32 帧均值；桥已知阶段不含未归属运行时与尾段 GPU 工作，不能作为完整桥开销。';
    }
  }catch(e){$('active').textContent='游戏连接已断开';$('telemetry').textContent='计时不可用';$('bridgeTelemetry').hidden=true}
}
async function apply(){
  constrain();
  // Capture every control before the busy check; newer edits replace this snapshot.
  pendingApply=selected();
  if(busy){show('新设置已排队，等待当前请求完成');return;}
  busy=true;
  const wasDisabled=$('apply').disabled;
  $('apply').disabled=true;
  try{
    while(pendingApply){
      const s=pendingApply;pendingApply=null;
      try{
        if(!supported(s.input_size,s.style))throw Error('该分辨率／风格尚未验证');
        const response=await fetch('/api/state',{method:'POST',
          headers:{'Content-Type':'application/json','X-NR-Token':token},
          body:JSON.stringify(s)}), data=await response.json();
        if(!response.ok)throw Error(data.error);
        if(!pendingApply){
          show('已应用：'+s.input_size+'p / '+(s.backend_variant==='unrounded'?'去舍入组合版':'现役版')+' / '+experimentNames[s.experiment_720]+' / '+styleNames[s.style]+' / 模型强度 '+s.model_intensity.toFixed(2)+
            ' / 色调 '+s.local_tone.toFixed(2)+' / 结构 '+s.local_structure.toFixed(2)+
            ' / '+s.history_mode+(s.graph_replay?' / 图重放':' / 普通执行'));
          await active();
        }
      }catch(e){show('未应用：'+e.message)}
    }
  }finally{
    busy=false;
    $('apply').disabled=wasDisabled;
  }
}
function updateBridgeCacheView(data){
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
function updateBridgePoolView(data){
  const c=data.bridge_resource_pool, box=$('bridgePoolEnabled');
  $('bridgePoolSwitch').hidden=c==null;$('bridgePoolStatus').hidden=c==null;
  if(c==null)return;
  if(!bridgePoolBusy)box.checked=c.enabled;
  $('bridgePoolStatus').textContent=(c.enabled?'中间纹理复用开启':'中间纹理复用关闭（对照）')+
    '；累计创建 '+c.creates+' 次，复用 '+c.hits+' 次；占用 '+(c.texture_bytes/1048576).toFixed(1)+
    ' MiB；等待完成 '+c.leased_slots+' 槽，隔离 '+c.quarantined_slots+' 槽';
}
async function applyBridgePool(){
  const box=$('bridgePoolEnabled'), enabled=box.checked;
  bridgePoolBusy=true;box.disabled=true;
  try{
    const response=await fetch('/api/bridge-resource-pool',{method:'POST',headers:{'Content-Type':'application/json','X-NR-Token':token},body:JSON.stringify({enabled})}), data=await response.json();
    if(!response.ok)throw Error(data.error);
    box.checked=data.enabled;show(enabled?'中间纹理复用已开启':'中间纹理复用已关闭，后续帧作对照');await active();
  }catch(e){box.checked=!enabled;show('中间纹理复用切换失败：'+e.message)}
  finally{bridgePoolBusy=false;box.disabled=false}
}
$('bridgePoolEnabled').onchange=applyBridgePool;
function updateGpuHandoffView(data){
  const c=data.gpu_handoff, box=$('gpuHandoffRequested');
  $('gpuHandoffSwitch').hidden=c==null;$('gpuHandoffStatus').hidden=c==null;
  if(c==null)return;
  if(!gpuHandoffBusy)box.checked=c.requested;
  $('gpuHandoffStatus').textContent=(c.requested?'已请求开启':'已请求关闭')+
    '；'+(c.armed&&c.cap_healthy?'能力已确认':c.requested?'等待线程确认，继续 CPU 等待':'CPU 准备等待已恢复')+
    '；已跳过准备等待 '+c.prepared_bypasses+' 次，前帧复用等待 '+c.previous_consumer_waits+' 次';
}
async function applyGpuHandoff(){
  const box=$('gpuHandoffRequested'), requested=box.checked;
  gpuHandoffBusy=true;box.disabled=true;
  try{
    const response=await fetch('/api/gpu-handoff',{method:'POST',headers:{'Content-Type':'application/json','X-NR-Token':token},body:JSON.stringify({requested})}), data=await response.json();
    if(!response.ok)throw Error(data.error);
    box.checked=data.requested;show(requested?'GPU 交接请求已保存，等待 NR 线程确认':'GPU 交接请求已关闭，准备阶段恢复 CPU 等待');await active();
  }catch(e){box.checked=!requested;show('GPU 交接请求失败：'+e.message)}
  finally{gpuHandoffBusy=false;box.disabled=false}
}
$('gpuHandoffRequested').onchange=applyGpuHandoff;
async function applyTiming(){
  const enabled=$('timingEnabled').checked;
  try{
    const response=await fetch('/api/timing',{method:'POST',
      headers:{'Content-Type':'application/json','X-NR-Token':token},
      body:JSON.stringify({enabled})}), data=await response.json();
    if(!response.ok)throw Error(data.error);
    show(enabled?'逐帧计时已开启':'逐帧计时已关闭，NR 保持运行');
    await active();
  }catch(e){$('timingEnabled').checked=!enabled;show('计时切换失败：'+e.message)}
}
$('timingEnabled').onchange=applyTiming;
async function applyValidation(){
  const box=$('validationBatchEnabled'), enabled=box.checked;
  validationBusy=true;box.disabled=true;
  try{
    const response=await fetch('/api/validation',{method:'POST',
      headers:{'Content-Type':'application/json','X-NR-Token':token},
      body:JSON.stringify({enabled})}), data=await response.json();
    if(!response.ok)throw Error(data.error);
    box.checked=data.enabled;
    show(enabled?'合并重复校验已开启；成功帧保留完整结束校验':'合并重复校验已关闭；保留当前计算图和历史');
    await active();
  }catch(e){box.checked=!enabled;show('校验开关未应用：'+e.message)}
  finally{validationBusy=false;box.disabled=false}
}
$('validationBatchEnabled').onchange=applyValidation;
function updateLifecycleView(data){
  const s=data.lifecycle_audit, box=$('lifecycleEnabled');
  $('lifecycleSwitch').hidden=s==null;
  $('lifecycleHint').hidden=s==null;
  $('lifecycleStatus').hidden=s==null;
  if(s==null)return;
  if(!lifecycleBusy)box.checked=s.enabled;
  $('lifecycleStatus').textContent=s.pending?'等待游戏下一帧重新选档':
    s.applied?'固定检查迁移已应用，切档次数 '+s.epoch:'固定检查迁移关闭（对照）';
}
async function applyLifecycle(){
  const box=$('lifecycleEnabled'), enabled=box.checked;
  lifecycleBusy=true;box.disabled=true;
  try{
    const response=await fetch('/api/lifecycle-audit',{method:'POST',
      headers:{'Content-Type':'application/json','X-NR-Token':token},
      body:JSON.stringify({enabled})}), data=await response.json();
    if(!response.ok)throw Error(data.error);
    box.checked=data.enabled;
    show('请求已提交；游戏下一帧重新选档后生效');
    await active();
  }catch(e){box.checked=!enabled;show('固定检查迁移未应用：'+e.message)}
  finally{lifecycleBusy=false;box.disabled=false}
}
$('lifecycleEnabled').onchange=applyLifecycle;
$('size').oninput=()=>{constrain()};
$('size').onchange=()=>{
  $('graph').checked=false;
  if(heights[Number($('size').value)]===720&&$('backendVariant').value==='unrounded'&&
     experimentAvailable.includes('c512_k8'))$('experiment720').value='c512_k8';
  apply();
};
$('backendVariant').onchange=()=>{constrain();apply()};
$('experiment720').onchange=()=>{
  if(fusedReplayProfiles.includes($('experiment720').value)){
    $('history').value='fused';$('graph').checked=true;
  }
  constrain();apply();
};
for(const box of optimizationBoxes())box.onchange=()=>{
  if(box.checked){
    for(const pair of conflicts)if(pair.includes(box.dataset.optimization))
      for(const other of optimizationBoxes())if(other!==box&&pair.includes(other.dataset.optimization))other.checked=false;
    $('experiment720').value='c512_k8';$('history').value='fused';$('graph').checked=true;
  }
  constrain();apply();
};
$('strength').oninput=refreshLabels;$('strength').onchange=apply;
for(const button of $('styleButtons').querySelectorAll('button'))button.onclick=()=>{
  selectedStyle=Number(button.dataset.style);refreshLabels();apply()};
for(const id of ['modelIntensity','localTone','localStructure','skinStructure']){
  $(id).oninput=refreshLabels;$(id).onchange=apply;
}
$('autoMask').onchange=()=>{refreshLabels();apply()};
$('separateSkin').onchange=()=>{refreshLabels();apply()};
$('history').onchange=()=>{constrain();apply()};$('graph').onchange=()=>{constrain();apply()};
$('enabled').onchange=apply;$('apply').onclick=apply;
load().then(active).catch(e=>show(e.message));setInterval(active,1000);
</script></html>"""
