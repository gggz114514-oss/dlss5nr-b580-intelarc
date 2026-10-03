"""Paired graph timings of explicit tiles on real captured 480p NR operands.

Each graph contains eight same-operand calls, amortizing replay host overhead.
Five rotated/reversed rounds of twelve graph replays; no whole-model speed claim.
Candidates with any output-byte difference are recorded and excluded.
"""
import argparse, hashlib, json, os, statistics, sys, time, traceback, urllib.request
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DREF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['fp16_xmx', 'int8_fused_v2'], required=True)
MODE = p.parse_args().mode
OUT = DREF / f'experimental/dense-tiles-{MODE}-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR'] = str(DREF / 'triton-cache-sm89-v1')
sys.path.insert(0, str(ROOT / 'backend'))
import numpy as np
import torch
from fast_matrices_v3 import dot as previous, quantize
from dense_tiles_v1 import dot as candidate
from fused_activation_int8_v1 import dot as fused
import compressed_arrays_v1 as arrays
import immutable_artifacts_v1 as artifacts
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
js = lambda path: json.loads(Path(path).read_text(encoding='utf-8-sig'))
capture_path = DREF / f'experimental/dense-operands-{MODE}-v1/validation.json'
capture = js(capture_path)
assert capture['passed'] and capture['byte_equal_prior']
assert js(capture_path.parent.with_suffix('.log.lease.json'))['returncode'] == 0
assert all(sha(path) == digest for path, digest in capture['sources'].items())
paths = [Path(__file__), HERE / 'Run-DenseTilesV1.cmd', HERE / 'dense_tiles_v1.py',
         HERE / 'fast_matrices_v3.py', HERE / 'fused_activation_int8_v1.py',
         HERE / 'compressed_arrays_v1.py', HERE / 'immutable_artifacts_v1.py', capture_path]
frozen = {str(path): sha(path) for path in paths}
with urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=5) as response:
    queue = json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
OUT.mkdir()
report = dict(scope=__doc__, sources=frozen, passed=False, mode=MODE, cases=[],
              case_selection='Top 24 captured geometries by call_count*M*K*N; this is a work-size heuristic, not a measured time ranking.',
              calls_per_graph=8, replays_per_sample=12, rounds=5, complete_migration=False,
              selection='Byte-identical, median at least 5% faster and at least 2% faster in every measured round; exact observed geometries only.')
raw = lambda value: value.cpu().numpy().tobytes()

def save():
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')

try:
    torch.set_num_threads(2)
    with torch.inference_mode():
        cases = [(key, row) for key, row in capture['cases'].items() if row['captured']]
        cases.sort(key=lambda item: item[1]['count'] * int(np.prod(item[1]['shape'])), reverse=True)
        report['selected_work_fraction'] = sum(row['count'] * int(np.prod(row['shape'])) for _, row in cases[:24]) / sum(row['count'] * int(np.prod(row['shape'])) for _, row in cases)
        for key, old in cases[:24]:
            operands = {name: None if meta is None else arrays.load(meta) for name, meta in old['operands'].items()}
            a, w, initial = (None if operands[name] is None else torch.from_numpy(operands[name]).to('xpu') for name in ('a', 'w', 'initial'))
            packed = None
            is_int8 = MODE.startswith('int8')
            if is_int8:
                q, scale, _ = quantize(w, columns=True)
                packed = q, scale
            is_fused = is_int8 and old['fused_bn'] is not None
            if is_fused:
                bn = old['fused_bn']
                configs = [(16, bn, 0, 4), (32, bn, 0, 4), (32, 64, 0, 4), (32, 128, 0, 4), (64, 64, 0, 4)]
                make = lambda c: lambda: fused(a, w, initial=initial, packed=packed, bm=c[0], bn=c[1], warps=c[3])
                target, _ = fused(a, w, initial=initial, packed=packed, bn=bn)
            else:
                configs = [(16, 32, 32, 4), (16, 64, 32, 4), (32, 32, 32, 4), (32, 64, 32, 4), (32, 128, 32, 4), (64, 64, 32, 4)]
                if is_int8:
                    configs += [(32, 64, 64, 4), (32, 64, 128, 4)]
                make = lambda c: lambda: candidate(a, w, initial=initial, packed=packed, int8=is_int8, tile=c)
                target, _ = previous(a, w, initial=initial, packed=packed, int8=is_int8)
            configs = list(dict.fromkeys(configs))
            expected = raw(target)
            row = dict(key=key, shape=old['shape'], initialized=old['initialized'], count=old['count'], fused=is_fused,
                       operands=old['operands'], expected=arrays.save(target.cpu().numpy()), candidates=[], samples_seconds={})
            report['cases'].append(row)
            entries = {}
            for config in configs:
                name = 'x'.join(map(str, config))
                fn = make(config)
                value, kernel = fn()
                equal = raw(value) == expected
                item = dict(name=name, tile=config, byte_equal=equal,
                            ttgir=artifacts.text(str(kernel.asm['ttgir']), 'ttgir'))
                row['candidates'].append(item)
                assert 'ttig.dpas' in str(kernel.asm['ttgir'])
                if not equal:
                    item['actual'] = arrays.save(value.cpu().numpy())
                    save()
                    continue
                stream = torch.xpu.Stream()
                with stream:
                    for _ in range(4):
                        value, _ = fn()
                torch.xpu.synchronize()
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph, stream=stream):
                    for _ in range(report['calls_per_graph']):
                        output, _ = fn()
                graph.replay()
                torch.xpu.synchronize()
                assert raw(output) == expected
                item['graph_byte_equal'] = True
                entries[name] = graph, output
                row['samples_seconds'][name] = []
            base = 'x'.join(map(str, configs[0]))
            assert base in entries
            names = list(entries)
            row['orders'] = []
            for repetition in range(report['rounds']):
                offset = repetition % len(names)
                order = names[offset:] + names[:offset]
                if repetition % 2:
                    order.reverse()
                row['orders'].append(order)
                for name in order:
                    graph, output = entries[name]
                    torch.xpu.synchronize()
                    started = time.perf_counter()
                    for _ in range(report['replays_per_sample']):
                        graph.replay()
                    torch.xpu.synchronize()
                    seconds = (time.perf_counter() - started) / (report['replays_per_sample'] * report['calls_per_graph'])
                    row['samples_seconds'][name].append(seconds)
                    assert raw(output) == expected
            medians = {name: statistics.median(samples) for name, samples in row['samples_seconds'].items()}
            eligible = [name for name in names if name != base and medians[name] < medians[base] * .95
                        and all(b / c > 1.02 for b, c in zip(row['samples_seconds'][base], row['samples_seconds'][name]))]
            best = min(eligible, key=medians.get) if eligible else base
            row.update(baseline=base, selected=best, median_seconds=medians,
                       speedup=medians[base] / medians[best], approximate_weighted_saving_seconds=old['count'] * (medians[base] - medians[best]))
            for name, t in [('a', a), ('w', w), ('initial', initial)]:
                assert t is None or raw(t) == operands[name].tobytes()
            row['operands_unchanged'] = True
            for graph, output in entries.values():
                assert raw(output) == expected
                graph.reset()
            entries.clear()
            save()
            print(json.dumps(dict(key=key, selected=best, speedup=row['speedup'], weighted_ms=row['approximate_weighted_saving_seconds'] * 1000)), flush=True)
        report['passed'] = True
        report['selected_changes'] = sum(row['selected'] != row['baseline'] for row in report['cases'])
        report['primitive_weighted_saving_seconds'] = sum(row['approximate_weighted_saving_seconds'] for row in report['cases'])
except Exception as error:
    report.update(error=repr(error), traceback=traceback.format_exc())
    raise
finally:
    assert all(sha(path) == digest for path, digest in frozen.items())
    save()
print(json.dumps({key: report[key] for key in ('passed', 'selected_changes', 'primitive_weighted_saving_seconds')}), flush=True)
