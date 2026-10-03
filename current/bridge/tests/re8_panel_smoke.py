"""Exercise the actual RE8 panel against Cyberpunk's native control mapping."""
import ctypes as C
import http.client
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import types
from urllib.parse import urlsplit


root = Path(__file__).resolve().parents[1]
re8_game = root.parent / "re8-b580-nr-xess" / "game"
sys.path.insert(0, str(re8_game))

host = types.ModuleType("nr_game_pre_xess_host")
host.LOG = Path(tempfile.mkdtemp(prefix="nr-re8-panel-"))
host._bridge = None
host._failed = False
host._failure_reason = None
host._error = lambda _stage: None
host._stage_resets = []
host.reset_stage_times = lambda: host._stage_resets.append(True)
host.stage_times = lambda: {
    "frames": 12, "prepare_average_ms": 3.0,
    "model_average_ms": 63.0, "export_average_ms": 4.0}
sys.modules[host.__name__] = host

spec = importlib.util.spec_from_file_location("cyberpunk_nr_web", root / "game/cyberpunk_nr_web.py")
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


class MockNative:
    def __init__(self):
        self.timing_enabled = True
        self.current = native.Controls()
        self.current.abi_size = C.sizeof(native.Controls)
        self.current.enabled = 1
        self.current.input_height = 540
        self.current.history = 3
        self.current.graph_replay = 1
        for name in ("display_strength", "model_intensity", "local_tone", "local_structure"):
            setattr(self.current, name, 1.0)

    def NRB_GetControls(self, pointer):
        C.memmove(pointer, C.byref(self.current), C.sizeof(self.current))
        return 1

    def NRB_SetControls(self, pointer):
        C.memmove(C.byref(self.current), pointer, C.sizeof(self.current))
        return 1

    def NRB_InitState(self):
        return 2

    def NRB_LastStatus(self):
        return 0

    def NRB_InterceptCount(self, _route):
        return 0

    def NRB_ProcessedCount(self):
        return 12

    def NRB_LastProcessMs(self):
        return 80.0

    def NRB_AverageProcessMs(self):
        return 80.0

    def NRB_GetStageTimes(self, pointer):
        stages = native.StageTimes()
        stages.abi_size = C.sizeof(stages)
        stages.frames = 12
        stages.prep_last_ms = stages.prep_average_ms = 3.0
        stages.host_last_ms = stages.host_average_ms = 70.0
        stages.composite_submit_last_ms = stages.composite_submit_average_ms = 2.0
        stages.tail_last_ms = stages.tail_average_ms = 5.0
        C.memmove(pointer, C.byref(stages), C.sizeof(stages))
        return 1

    def NRB_GetTimingEnabled(self):
        return int(self.timing_enabled)

    def NRB_SetTimingEnabled(self, enabled):
        self.timing_enabled = bool(enabled)
        return 1


native._native = MockNative()
sys.modules[native.__name__] = native
os.environ.pop("CYBERPUNK_NR_WEB", None)
spec = importlib.util.spec_from_file_location("cyberpunk_nr_adapter", root / "game/cyberpunk_nr_adapter.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
adapter.start_controls()
panel = adapter._panel
url = urlsplit(panel.url)
conn = http.client.HTTPConnection(url.hostname, url.port, timeout=2)
try:
    conn.request("GET", "/")
    response = conn.getresponse()
    page = response.read().decode("utf-8")
    assert response.status == 200
    assert "B580 NR 游戏输入控制" in page and "模型内强度" in page
    token = re.search(r"const token='([0-9a-f]+)'", page).group(1)

    conn.request("GET", "/api/state")
    response = conn.getresponse()
    state = json.loads(response.read())
    assert state["route"] == "pre-xess-fullsize"
    assert state["settings"]["input_size"] == 540
    assert {mode["input_size"] for mode in state["available_modes"]} == {360, 480, 540, 720}
    assert state["graph_modes"] == [360, 480, 540, 720]
    assert state["settings"]["history_mode"] == "fused"
    if "backend_variant" in state["settings"]:
        assert state["settings"]["backend_variant"] == "unrounded"
    assert state["available_history_modes"] == ["reference", "zero_motion", "reset", "fused"]
    assert state["processing"]["frames"] == 12
    assert state["processing"]["stages"]["bridge_pre_sr_average_ms"] == 5.0
    assert state["processing"]["stages"]["bridge_known_average_ms"] == 12.0
    assert state["processing"]["stages"]["runtime"]["model_average_ms"] == 63.0

    values = state["settings"]
    values.update(input_size=480, style=2, model_intensity=1.25,
                  display_strength=0.8, auto_mask=True, skin_structure=1.1)
    headers = {"Content-Type": "application/json", "X-NR-Token": token,
               "Origin": panel.url.rstrip("/")}
    conn.request("POST", "/api/state", json.dumps(values), headers)
    response = conn.getresponse()
    updated = json.loads(response.read())
    assert response.status == 200, updated
    c = native._native.current
    assert (c.input_height, c.style, c.history, c.auto_mask,
            c.skin_structure_enabled) == (480, 2, 3, 1, 1)
    assert abs(c.model_intensity - 1.25) < 1e-5
    assert adapter._panel.snapshot().input_size == 480
    assert len(host._stage_resets) == 1

    values["history_mode"] = "reference"
    conn.request("POST", "/api/state", json.dumps(values), headers)
    response = conn.getresponse()
    response.read()
    assert response.status == 200 and c.history == 0
    values["history_mode"] = "fused"
    conn.request("POST", "/api/state", json.dumps(values), headers)
    response = conn.getresponse()
    response.read()
    assert response.status == 200 and c.history == 3
finally:
    conn.close()
    panel.close()
