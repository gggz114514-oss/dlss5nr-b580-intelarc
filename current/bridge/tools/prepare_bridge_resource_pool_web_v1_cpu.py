"""Stage an independent pool checkbox without changing the frozen native worker."""
import ast
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SHARED = PROJECT.parent / ".codex-worktrees/re8-fp8-unround-fast-20260928/game"
STAGE = PROJECT / "artifacts/bridge-resource-pool-web-v1-20261002"
WORKER = PROJECT / "artifacts/bridge-resource-pool-v1-20261001"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("Changed source anchor: " + old[:100])
    return text.replace(old, new)


def main():
    if STAGE.exists() or not STAGE.resolve().is_relative_to(PROJECT.resolve()):
        raise RuntimeError("Fresh E stage required")
    paths = {"cyberpunk_nr_web.py": PROJECT / "game/cyberpunk_nr_web.py",
             "cyberpunk_nr_adapter.py": PROJECT / "game/cyberpunk_nr_adapter.py",
             "nr_game_controls.py": SHARED / "nr_game_controls.py"}
    pins = {str(path): sha(path) for path in paths.values()}
    texts = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}
    web = texts["cyberpunk_nr_web.py"]
    declaration = '''class HdrResourcePoolStats(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in (
        "abi_size", "enabled", "capacity", "shutdown")]
    _fields_ += [(name, C.c_uint64) for name in (
        "acquires", "hits", "creates", "disabled_fallbacks", "busy_fallbacks", "creation_failures",
        "retire_returns", "unsubmitted_returns", "quarantines", "evictions", "generation",
        "resident_slots", "free_slots", "leased_slots", "quarantined_slots",
        "texture_bytes", "peak_texture_bytes", "descriptor_bytes_estimate", "max_texture_bytes")]


'''
    web = one(web, "_names = [name for name, _ in Controls._fields_", declaration + "_names = [name for name, _ in Controls._fields_")
    web = one(web, "        _native.NRB_GetTimingEnabled.restype = C.c_int", '''        if (hasattr(_native, "NRB_GetHdrResourcePoolStats") and
                hasattr(_native, "NRB_SetHdrResourcePoolEnabled")):
            _native.NRB_GetHdrResourcePoolStats.argtypes = [C.POINTER(HdrResourcePoolStats)]
            _native.NRB_GetHdrResourcePoolStats.restype = C.c_int
            _native.NRB_SetHdrResourcePoolEnabled.argtypes = [C.c_int]
            _native.NRB_SetHdrResourcePoolEnabled.restype = C.c_int
        _native.NRB_GetTimingEnabled.restype = C.c_int''')
    web = one(web, "def health():", '''def get_hdr_resource_pool_state():
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


def health():''')
    texts["cyberpunk_nr_web.py"] = web
    adapter = texts["cyberpunk_nr_adapter.py"]
    adapter = one(adapter, "            lifecycle_options = (", '''            bridge_pool_options = ({"bridge_pool_read": native.get_hdr_resource_pool_state,
                                    "bridge_pool_apply": native.set_hdr_resource_pool_enabled}
                                   if "bridge_pool_read" in inspect.signature(ControlPanel).parameters
                                   else {})
            lifecycle_options = (''')
    adapter = one(adapter, "**lifecycle_options, **bridge_cache_options)", "**lifecycle_options, **bridge_cache_options, **bridge_pool_options)")
    texts["cyberpunk_nr_adapter.py"] = adapter
    panel = texts["nr_game_controls.py"]
    panel = one(panel, '                 available_experiments_720=("baseline",)):', '''                 available_experiments_720=("baseline",),
                 bridge_pool_read=None, bridge_pool_apply=None):''')
    panel = one(panel, "        self._bridge_cache_apply=bridge_cache_apply", '''        self._bridge_cache_apply=bridge_cache_apply
        if (bridge_pool_read is None) != (bridge_pool_apply is None):
            raise ValueError("Bridge resource pool callbacks must be paired")
        self._bridge_pool_read=bridge_pool_read
        self._bridge_pool_apply=bridge_pool_apply''')
    panel = one(panel, '        status["bridge_cache"] = self._bridge_cache_read() if self._bridge_cache_read else None', '''        status["bridge_cache"] = self._bridge_cache_read() if self._bridge_cache_read else None
        status["bridge_resource_pool"] = self._bridge_pool_read() if self._bridge_pool_read else None''')
    panel = one(panel, "    def update_timing(self, values):", '''    def update_bridge_pool(self, values):
        if self._bridge_pool_apply is None:
            raise ValueError("Bridge resource pool is unavailable on this route")
        if not isinstance(values, dict) or set(values)!={"enabled"} or type(values["enabled"]) is not bool:
            raise ValueError("Expected a boolean bridge resource pool setting")
        self._bridge_pool_apply(values["enabled"])
        return self._bridge_pool_read()

    def update_timing(self, values):''')
    panel = one(panel, '"/api/lifecycle-audit", "/api/bridge-cache")', '"/api/lifecycle-audit", "/api/bridge-cache", "/api/bridge-resource-pool")')
    panel = one(panel, '                              else panel.update_bridge_cache(values) if self.path == "/api/bridge-cache"', '''                              else panel.update_bridge_cache(values) if self.path == "/api/bridge-cache"
                              else panel.update_bridge_pool(values) if self.path == "/api/bridge-resource-pool"''')
    panel = one(panel, '<p id="bridgeCacheStatus" class="hint" hidden></p>', '''<p id="bridgeCacheStatus" class="hint" hidden></p>
<label id="bridgePoolSwitch" hidden><input id="bridgePoolEnabled" type="checkbox"> 复用桥的中间纹理（实验）</label>
<p id="bridgePoolStatus" class="hint" hidden></p>''')
    panel = one(panel, "let validationBusy=false, bridgeCacheBusy=false;", "let validationBusy=false, bridgeCacheBusy=false, bridgePoolBusy=false;")
    panel = one(panel, "    updateBridgeCacheView(data);", "    updateBridgeCacheView(data);\n    updateBridgePoolView(data);")
    panel = one(panel, "async function applyTiming(){", '''function updateBridgePoolView(data){
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
async function applyTiming(){''')
    texts["nr_game_controls.py"] = panel
    for name, text in texts.items():
        ast.parse(text, filename=name)
    target = STAGE / "payload/game"
    target.mkdir(parents=True)
    for name, text in texts.items():
        (target / name).write_text(text, encoding="utf-8", newline="\n")
    receipt = {"status": "source_staged_CPU_protocol_check_pending", "source_pins": pins,
               "payload_pins": {name: sha(target / name) for name in texts},
               "native_worker_manifest": str(WORKER / "source-manifest.json"),
               "native_worker_manifest_sha256": sha(WORKER / "source-manifest.json"),
               "native_pool_stats_size": 168, "default_pool_enabled": False,
               "independent_of_model_rebuild_and_pipeline_cache": True,
               "GPU_executed": False, "G_writes": False}
    (STAGE / "STAGE_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "staged", "path": str(STAGE), "GPU_executed": False}))


if __name__ == "__main__":
    main()
