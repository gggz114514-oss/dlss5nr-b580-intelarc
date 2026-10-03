"""Synthetic finite-half layouts covering Swin, ViT, offset, broadcast and tail masks."""
import hashlib, json, os, statistics, sys, time, traceback, urllib.request
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DREF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT = DREF / 'experimental/strided-batched-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR'] = str(DREF / 'triton-cache-sm89-v1')
sys.path.insert(0, str(ROOT / 'backend'))
import numpy as np
import torch
from fast_matrices_v3 import dot as previous
from strided_batched_v1 import dot as candidate
import compressed_arrays_v1 as arrays
import immutable_artifacts_v1 as artifacts
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths = [Path(__file__), HERE / 'Run-StridedBatchedV1.cmd', HERE / 'strided_batched_v1.py',
         HERE / 'fast_matrices_v3.py', HERE / 'compressed_arrays_v1.py', HERE / 'immutable_artifacts_v1.py']
sources = {str(p): sha(p) for p in paths}
with urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=5) as response:
    queue = json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
OUT.mkdir()
report = dict(scope=__doc__, sources=sources, passed=False, cases=[], seed=9092603,
              timing_scope='Six alternating rounds of 40 prewarmed calls; wrapper, allocations, copies, dot and final sync included. GPU graph timing measured separately for the same primitive. Not complete-model speed.',
              complete_migration=False)
rng = np.random.default_rng(report['seed'])
raw = lambda t: t.cpu().numpy().tobytes()

def save():
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')

def random(shape):
    x = rng.uniform(-1, 1, shape).astype('f2')
    x.flat[::211] = np.float16(0)
    x.flat[::257] = np.float16(-0.)
    x.flat[::313] = np.float16(2 ** -24)
    return torch.from_numpy(x).to('xpu')

try:
    torch.set_num_threads(2)
    with torch.inference_mode():
        cases = [('swin', 7, 64, 32, 64), ('swin', 1024, 64, 32, 64),
                 ('value', 7, 64, 64, 32), ('value', 1024, 64, 64, 32),
                 ('vit', 32, 96, 32, 128), ('vit', 32, 640, 32, 640),
                 ('vit_value', 32, 640, 640, 32), ('offset', 3, 17, 96, 37),
                 ('broadcast', 5, 31, 32, 19)]
        for kind, b, m, k, n in cases:
            if kind == 'vit':
                a = random((m, b, k)).transpose(0, 1)
            elif kind == 'offset':
                a = random((b + 1, m + 2, k * 2 + 4))[1:, 1:m + 1, 2:2 + 2 * k:2]
            elif kind == 'broadcast':
                a = random((m, k)).expand(b, m, k)
            else:
                a = random((b, m, k))
            w = random((b, n, k)).transpose(-1, -2) if kind in ('swin', 'vit') else random((b, k, n))
            if kind == 'vit_value':
                w = random((k, b, n)).transpose(0, 1)
            if kind == 'offset':
                w = random((b + 1, k * 2 + 3, n + 2))[1:, 1:1 + 2 * k:2, 1:n + 1]
            initial = random((m, n)).expand(b, m, n) if kind in ('swin', 'broadcast') else None
            if kind == 'offset':
                initial = random((b, n + 1, m + 1))[:, :n, :m].transpose(-1, -2)
            before = [raw(t) if t is not None else None for t in (a, w, initial)]
            operands = {name: None if t is None else dict(array=arrays.save(t.cpu().numpy()), stride=list(t.stride()), storage_offset=t.storage_offset())
                        for name, t in [('a', a), ('w', w), ('initial', initial)]}
            expected, _ = previous(a, w, initial=initial, batched=True)
            actual, compiled = candidate(a, w, initial=initial)
            equal = raw(expected) == raw(actual)
            row = dict(kind=kind, shape=[b, m, k, n], operands=operands, byte_equal=equal,
                       expected=arrays.save(expected.cpu().numpy()), actual=arrays.save(actual.cpu().numpy()))
            report['cases'].append(row)
            row['ttgir'] = artifacts.text(str(compiled.asm['ttgir']), 'ttgir')
            assert 'ttig.dpas' in str(compiled.asm['ttgir'])
            save()
            assert equal, (kind, b, m, k, n)
            functions = {'previous': lambda: previous(a, w, initial=initial, batched=True),
                         'strided': lambda: candidate(a, w, initial=initial)}
            graph_entries = {}
            for name, fn in functions.items():
                stream = torch.xpu.Stream()
                with stream:
                    for _ in range(4):
                        warm, _ = fn()
                torch.xpu.synchronize()
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph, stream=stream):
                    output, _ = fn()
                graph.replay()
                torch.xpu.synchronize()
                assert raw(output) == raw(expected)
                graph_entries[name] = graph, output
            samples = {name: [] for name in ('previous', 'strided', 'previous_graph', 'strided_graph')}
            for repetition in range(6):
                order = list(samples)
                offset = repetition % len(order)
                order = order[offset:] + order[:offset]
                if repetition % 2:
                    order.reverse()
                for name in order:
                    torch.xpu.synchronize()
                    started = time.perf_counter()
                    for _ in range(40):
                        if name.endswith('_graph'):
                            graph_entries[name[:-6]][0].replay()
                        else:
                            output, _ = functions[name]()
                    torch.xpu.synchronize()
                    samples[name].append((time.perf_counter() - started) / 40)
                    if not name.endswith('_graph'):
                        assert raw(output) == raw(expected)
            assert before == [raw(t) if t is not None else None for t in (a, w, initial)]
            for graph, output in graph_entries.values():
                assert raw(output) == raw(expected)
                graph.reset()
            row.update(samples_seconds=samples, medians_seconds={k: statistics.median(v) for k, v in samples.items()}, operands_unchanged=True)
            save()
            print(json.dumps(dict(kind=kind, shape=row['shape'], times=row['medians_seconds'], byte_equal=True)), flush=True)
        report['passed'] = True
except Exception as error:
    report.update(error=repr(error), traceback=traceback.format_exc())
    raise
finally:
    assert all(sha(p) == h for p, h in sources.items())
    save()
print(json.dumps(dict(passed=report['passed'], cases=len(report['cases']))), flush=True)
