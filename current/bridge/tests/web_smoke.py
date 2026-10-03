"""Offline loopback controls check with a mock native ABI; no game or GPU."""
import ctypes as C
import http.client
import importlib.util
import json
from pathlib import Path
import re
import sys


root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cyberpunk_nr_web", root / "game/cyberpunk_nr_web.py")
web = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web)
assert C.sizeof(web.Controls) == 52
assert C.sizeof(web.StageTimes) == 80


class MockNative:
    def __init__(self):
        self.timing_enabled = True
        self.current = web.Controls()
        self.current.abi_size = 52
        self.current.input_height = 360
        self.current.history = 3
        self.current.display_strength = 1.0

    def NRB_GetControls(self, pointer):
        C.memmove(pointer, C.byref(self.current), 52)
        return 1

    def NRB_SetControls(self, pointer):
        value = C.cast(pointer, C.POINTER(web.Controls)).contents
        if value.input_height not in (360, 480, 540, 720) or not 0 <= value.display_strength <= 1:
            return 0
        C.memmove(C.byref(self.current), pointer, 52)
        return 1

    def NRB_InitState(self):
        return 2

    def NRB_LastStatus(self):
        return 3

    def NRB_InterceptCount(self, route):
        return (4, 5, 6)[route - 1]

    def NRB_ProcessedCount(self):
        return 0

    def NRB_LastProcessMs(self):
        return 0.0

    def NRB_AverageProcessMs(self):
        return 0.0

    def NRB_GetStageTimes(self, _pointer):
        return 0

    def NRB_GetTimingEnabled(self):
        return int(self.timing_enabled)

    def NRB_SetTimingEnabled(self, enabled):
        self.timing_enabled = bool(enabled)
        return 1


web._native = MockNative()
log_dir = Path(sys.argv[1])
port = web.start(log_dir)
assert (log_dir / "cyberpunk-nr-control-url.txt").read_text().strip().endswith(f":{port}/")
conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
try:
    conn.request("GET", "/")
    response = conn.getresponse()
    page = response.read().decode()
    assert response.status == 200
    token = re.search(r"const token='([0-9a-f]+)'", page).group(1)
    conn.request("GET", "/api")
    response = conn.getresponse()
    settings = json.loads(response.read())
    assert settings["input_height"] == 360
    conn.request("GET", "/health")
    response = conn.getresponse()
    health = json.loads(response.read())
    assert response.status == 200 and health["status"] == 3
    assert health["intercepts"] == {"DLSS": 4, "FSR": 5, "XeSS": 6}
    assert health["stages"] is None
    settings["enabled"] = 1
    settings["input_height"] = 720
    settings["style"] = 2
    payload = json.dumps(settings)
    conn.request("POST", "/api", payload, {"X-NR-Token": "wrong"})
    response = conn.getresponse()
    response.read()
    assert response.status == 403
    conn.request("POST", "/api", payload, {"X-NR-Token": token})
    response = conn.getresponse()
    actual = json.loads(response.read())
    assert response.status == 200 and actual["input_height"] == 720 and actual["style"] == 2
finally:
    conn.close()
    web._server.shutdown()
    web._server.server_close()
