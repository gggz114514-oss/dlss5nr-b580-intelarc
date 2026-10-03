"""Bounded feasibility probe for timing tags inside a replayed XPU graph.

No NR model or profiler layer. A handled unsupported-capture result is negative
evidence, not a usable stage timer. Outside-graph tags and changed-input replay
provide controls; no stage attribution is claimed from host enqueue intervals.
"""
import hashlib
import json
from pathlib import Path
import time
import traceback
import urllib.request

HERE = Path(__file__).resolve().parent
OUT = Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental/xpu-graph-events-v1')
assert not OUT.exists()
with urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=5) as response:
    queue = json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']

import torch

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
torch_root = Path(torch.__file__).parent
paths = [Path(__file__), HERE / 'Run-XpuGraphEventsV1.cmd',
         torch_root / 'xpu/graphs.py', torch_root / 'xpu/streams.py',
         torch_root / 'include/c10/xpu/XPUEvent.h']
OUT.mkdir()
report = dict(scope=__doc__, sources={str(p): sha(p) for p in paths},
              passed=False, complete_migration=False, model_modified=False,
              torch_version=torch.__version__, captured_tags_usable=False,
              captured_samples=[], untagged_samples=[])


def save():
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def body(source, temporary, output, tags=None):
    if tags is not None:
        tags[0].record()
    torch.add(source, 1, out=temporary)
    for _ in range(15):
        temporary.add_(1)
    if tags is not None:
        tags[1].record()
    for _ in range(64):
        temporary.add_(1)
    output.copy_(temporary)
    if tags is not None:
        tags[2].record()


def expected(output, initial):
    # Copy/check the entire result, not a spot sample. Inputs are exact integers.
    actual = output.cpu()
    assert torch.equal(actual, torch.full_like(actual, initial + 80))
    return sha_bytes(actual.numpy().tobytes())


sha_bytes = lambda b: hashlib.sha256(b).hexdigest()
graph = None
try:
    torch.set_num_threads(2)
    report['device'] = torch.xpu.get_device_name()
    source = torch.zeros(262144, device='xpu', dtype=torch.float32)
    temporary, output = torch.empty_like(source), torch.empty_like(source)
    stream = torch.xpu.Stream()
    torch.xpu.synchronize()
    tags = [torch.xpu.Event(enable_timing=True) for _ in range(3)]
    with stream:
        body(source, temporary, output)
        body(source, temporary, output, tags)
    torch.xpu.synchronize()
    report['outside_control'] = dict(first_ms=tags[0].elapsed_time(tags[1]),
                                    second_ms=tags[1].elapsed_time(tags[2]),
                                    output_sha256=expected(output, 0))
    assert min(report['outside_control'][k] for k in ('first_ms', 'second_ms')) > 0
    save()
    try:
        report['capture_phase'] = 'record'
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph, stream=stream):
            body(source, temporary, output, tags)
        report['capture_phase'] = 'replay_and_query'
        for initial in (0, 7, -3):
            source.fill_(initial)
            anchor = torch.xpu.Event(enable_timing=True)
            anchor.record()
            start = time.perf_counter()
            graph.replay()
            torch.xpu.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            first = tags[0].elapsed_time(tags[1])
            second = tags[1].elapsed_time(tags[2])
            since_anchor = anchor.elapsed_time(tags[0])
            assert 0 <= since_anchor < elapsed * 2
            assert first > 0 and second > 0
            report['captured_samples'].append(dict(initial=initial, first_ms=first,
                second_ms=second, host_ms=elapsed, since_anchor_ms=since_anchor,
                output_sha256=expected(output, initial)))
        report['captured_tags_usable'] = True
    except Exception:
        report['capture_error'] = traceback.format_exc()
        print(report['capture_error'], flush=True)
    finally:
        if graph is not None:
            graph.reset()
            graph = None
    # Verify that ordinary graph capture/replay still works after the probe.
    torch.xpu.synchronize()
    assert not torch.xpu.is_current_stream_capturing()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph, stream=stream):
        body(source, temporary, output)
    for initial in (0, 7, -3):
        source.fill_(initial)
        graph.replay()
        torch.xpu.synchronize()
        report['untagged_samples'].append(dict(initial=initial, output_sha256=expected(output, initial)))
    graph.reset()
    graph = None
    report['passed'] = True
except Exception:
    report['error'] = traceback.format_exc()
    raise
finally:
    if graph is not None:
        graph.reset()
    assert all(sha(p) == h for p, h in report['sources'].items())
    save()
print(json.dumps({k: v for k, v in report.items() if k != 'sources'}, indent=2), flush=True)
