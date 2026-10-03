"""Exercise the installed loader and complete VIDEO node, isolated from live ComfyUI."""
import importlib.util
import json
from pathlib import Path
import sys
import traceback

BASE = Path(__file__).resolve().parents[3]
OUT = Path('D:/Codex-NR-Experiments/nr-b580/product-worker/comfy-node-v1')
sys.path.insert(0, str(BASE / 'ComfyUI-aki-v3-IntelArc/ComfyUI'))
sys.argv = ['smoke', '--cpu']
try:
    path = BASE / 'ComfyUI-aki-v3-IntelArc/ComfyUI/custom_nodes/NR-B580-Local/__init__.py'
    spec = importlib.util.spec_from_file_location('nr_installed_smoke', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from comfy_api.latest import InputImpl
    node = module.NODE_CLASS_MAPPINGS['NRB580Video']()
    video, filename = node.execute(InputImpl.VideoFromFile('E:/下载/480p (1).mp4'), '快速版（类 DLSS5）')
    assert video.get_stream_source() == filename and Path(filename).is_file()
    result = dict(passed=True, output=filename, class_name='NRB580Video', frames=243)
except BaseException:
    result = dict(passed=False, error=traceback.format_exc())
    traceback.print_exc()
(OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
sys.exit(0 if result['passed'] else 1)
