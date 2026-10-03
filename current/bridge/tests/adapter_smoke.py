"""Offline ABI controls/delegation check with an inert stand-in for RE8 host."""
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import sys
import types


@dataclasses.dataclass
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


host = types.ModuleType("nr_game_pre_xess_host")
os.environ["CYBERPUNK_NR_WEB"] = "0"
host.process = lambda *args: (11, 22, args[4])
host.retire = lambda fence, value: (fence, value)
controls = types.ModuleType("nr_game_controls")
controls.Settings = Settings
sys.modules[host.__name__] = host
sys.modules[controls.__name__] = controls
script = Path(__file__).resolve().parents[1] / "game/cyberpunk_nr_adapter.py"
spec = importlib.util.spec_from_file_location("cyberpunk_nr_adapter", script)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
assert (adapter._panel.snapshot().input_size, adapter._panel.snapshot().history_mode,
        adapter._panel.snapshot().graph_replay,
        adapter._panel.snapshot().backend_variant) == (540, "fused", True, "unrounded")
adapter.configure(json.dumps({
    "enabled": 1, "input_height": 540, "style": 2, "history": 3,
    "graph_replay": 1, "auto_mask": 1, "skin_structure_enabled": 1,
    "display_strength": .65, "model_intensity": 1.2,
    "local_tone": .8, "local_structure": 1.3, "skin_structure": 1.1,
}))
setting = host._panel.snapshot()
assert (setting.input_size, setting.style, setting.history_mode, setting.graph_replay) == (540, 2, "fused", True)
assert setting.backend_variant == "unrounded"
assert (setting.display_strength, setting.model_intensity, setting.local_tone,
        setting.local_structure, setting.auto_mask, setting.skin_structure) == (.65, 1.2, .8, 1.3, True, 1.1)
assert adapter.process(1, 2, 3, 4, 5, 1, 960, 540) == (11, 22, 5)
assert adapter.retire(22, 5) == (22, 5)
