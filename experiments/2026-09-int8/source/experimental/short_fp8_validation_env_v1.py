"""Authenticated setup for frozen ShortFP8 complete-call experiments."""
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXACT = ROOT.parent / 'nr-b580'
R = EXACT / 'reference'
D = Path('D:/Codex-NR-Experiments/nr-b580/reference')
TOOLCHAIN = D / 'toolchains/triton-xpu-3.8.0-git1e2d42a0'
os.environ['TRITON_CACHE_DIR'] = str(D / 'triton-cache-c32-triton38-v1')
sys.path[:0] = [str(TOOLCHAIN / 'site'), str(R), str(ROOT / 'backend')]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
js = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate = authenticate_main()
body_path = D / 'experimental/short-fp8-body-v2/validation.json'
assert sha(body_path) == '48ea3ee90be2f6e8166125a4917cfee1a46248a138f73bde90a1757b1a06aead'
body = js(body_path)
assert body['passed'] and js(body_path.parent.with_suffix('.log.lease.json'))['returncode'] == 0
sources = dict(body['sources'])
sources[str(body_path)] = sha(body_path)
for p in (Path(__file__), HERE / 'short_fp8_graph_v1.py', HERE / 'nr256_selected_stack_v2.py'):
    sources[str(p)] = sha(p)
assert all(sha(p) == h for p, h in sources.items())
provision = TOOLCHAIN / 'provision-v1.json'
assert sha(provision) == 'e94757afcb5962cacbe208cbb808b3516de210059335f1ee7ab7b40d6bfe2c07'
assert all(sha(TOOLCHAIN / 'site' / p) == h for p, h in js(provision)['files'].items())
with urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=5) as response:
    queue = json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']

def receipt(path, expected):
    path = Path(path)
    assert sha(path) == expected
    result = js(path)
    assert result['passed']
    for p, h in result.get('sources', {}).items():
        assert sha(p) == h
        if p in sources:
            assert sources[p] == h
        sources[p] = h
    sources[str(path)] = expected
    return result

def finalize_sources():
    assert all(sha(p) == h for p, h in sources.items())
    authenticate_main()
