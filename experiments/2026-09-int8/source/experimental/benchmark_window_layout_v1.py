"""All half bit-pattern permutations and paired window layout timing, no NR inference."""
import hashlib, json, os, statistics, sys, time, traceback, urllib.request
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DREF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT = DREF / 'experimental/window-layout-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR'] = str(DREF / 'triton-cache-sm89-v1')
sys.path.insert(0, str(ROOT / 'backend'))
import numpy as np
import torch
from window_layout_v1 import pack, unpack
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths = [Path(__file__), HERE / 'Run-WindowLayoutV1.cmd', HERE / 'window_layout_v1.py']
sources = {str(p): sha(p) for p in paths}
with urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=5) as response:
    queue = json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
OUT.mkdir()
report = dict(scope=__doc__, passed=False, sources=sources, cases=[], complete_migration=False,
              data='Deterministic cyclic raw uint16 pattern, viewed as half. Includes all 65536 encodings in sufficiently large cases. No arithmetic is applied.',
              timing='Six alternating prewarmed 40-call rounds; ordinary wrappers and synchronized batches of graph replay measured separately.')
raw = lambda t: t.cpu().numpy().tobytes()

def save():
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')

try:
    torch.set_num_threads(2)
    with torch.inference_mode():
        order = torch.tensor([base + g % 4 + 8 * (g // 4) + 16 * word for base in (0, 4, 32, 36)
                              for word in range(2) for g in range(8)], device='xpu')
        inverse = torch.argsort(order)
        for h, w, heads, strided in [(8, 8, 1, False), (16, 24, 2, False), (72, 80, 4, True),
                                     (128, 224, 8, False), (512, 896, 1, False)]:
            shape = (h + 2, w + 2, heads, 64) if strided else (h, w, heads, 32)
            values = (np.arange(np.prod(shape), dtype='u4') % 65536).astype('u2').view('f2').reshape(shape)
            parent = torch.from_numpy(values).to('xpu')
            x = parent[1:h + 1, 1:w + 1, :, ::2] if strided else parent
            expected = x.reshape(h // 8, 8, w // 8, 8, heads, 32).permute(4, 0, 2, 1, 3, 5).reshape(heads, h // 8, w // 8, 64, 32)[..., order, :]
            value = pack(x, order)
            assert raw(value) == raw(expected)
            assert raw(unpack(value, inverse)) == raw(x)
            windows = expected
            functions = {
                'old_pack': lambda: x.reshape(h // 8, 8, w // 8, 8, heads, 32).permute(4, 0, 2, 1, 3, 5).reshape(heads, h // 8, w // 8, 64, 32)[..., order, :],
                'new_pack': lambda: pack(x, order),
                'old_unpack': lambda: windows[..., inverse, :].reshape(heads, h // 8, w // 8, 8, 8, 32).permute(1, 3, 2, 4, 0, 5).reshape(h, w, heads, 32),
                'new_unpack': lambda: unpack(windows, inverse)}
            graphs = {}
            for name, fn in functions.items():
                stream = torch.xpu.Stream()
                with stream:
                    for _ in range(4):
                        warm = fn()
                torch.xpu.synchronize()
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph, stream=stream):
                    output = fn()
                graphs[name] = graph, output
            samples = {name + suffix: [] for name in functions for suffix in ('', '_graph')}
            for repetition in range(6):
                names = list(samples)
                offset = repetition % len(names)
                names = names[offset:] + names[:offset]
                if repetition % 2:
                    names.reverse()
                for name in names:
                    torch.xpu.synchronize()
                    started = time.perf_counter()
                    for _ in range(40):
                        if name.endswith('_graph'):
                            graphs[name[:-6]][0].replay()
                        else:
                            output = functions[name]()
                    torch.xpu.synchronize()
                    samples[name].append((time.perf_counter() - started) / 40)
                    check = graphs[name[:-6]][1] if name.endswith('_graph') else output
                    assert raw(check) == raw(x if 'unpack' in name else expected)
            assert raw(parent) == values.tobytes()
            row = dict(shape=[h, w, heads, 32], strided=strided, byte_equal=True, source_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                       output_sha256=hashlib.sha256(raw(expected)).hexdigest(), input_unchanged=True,
                       samples_seconds=samples, median_seconds={k: statistics.median(v) for k, v in samples.items()})
            report['cases'].append(row)
            for graph, output in graphs.values():
                graph.reset()
            save()
            print(json.dumps(dict(shape=row['shape'], strided=strided, times=row['median_seconds'])), flush=True)
        report['passed'] = True
except Exception as error:
    report.update(error=repr(error), traceback=traceback.format_exc())
    raise
finally:
    assert all(sha(p) == h for p, h in sources.items())
    save()
print(json.dumps(dict(passed=report['passed'], cases=len(report['cases']))), flush=True)
