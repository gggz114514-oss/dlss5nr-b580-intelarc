"""Copied PHASE1 save_array/raw_graph helpers; tensor imports occur only in the child."""
from __future__ import annotations
import hashlib
import contextlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import runner_common as c
from cache_policy import CachePolicy, json_value
from module_sources import collect as collect_module_sources

_FAILED_RETIREMENTS = []
_VIRTUAL_SOURCE_MODULES = {}
FRONT_MODULES = frozenset({'nr_game_controls', 'nr_game_pre_xess_host', 'nr_gpu_handoff_host_v1',
                           'nr_texture_bridge_v1', 'cyberpunk_nr_web', 'cyberpunk_nr_adapter'})
SELECTED_CAPTURE_GATES = (
    'c32_hidden_native_ten', 'c512_decoder_all_eight', 'c512_probability_all_sixteen',
    'c64_all_eight', 'c128_all_twelve', 'c128_pairwise_twelve',
    'c64_attention_project_eight', 'c128_attention_project_twelve',
)


def progress(tag, phase, **fields):
    print(json.dumps(dict(event='CURRENT_PROVIDER_PROGRESS', tag=tag, phase=phase, **fields),
                     ensure_ascii=False, allow_nan=False), flush=True)


def buffer_sha(value):
    return hashlib.sha256(memoryview(value).cast('B')).hexdigest()


def save_array(np, tensor, path):
    path = c.output_path(path)
    copy_started = time.perf_counter()
    value = tensor.detach().cpu().contiguous().numpy()
    d2h_cpu_ms = (time.perf_counter() - copy_started) * 1000.0
    c.require(value.dtype.kind == 'f' and bool(np.isfinite(value).all()), 'Nonfinite/nonfloat full-model result')
    write_started = time.perf_counter()
    with path.open('xb') as stream:
        np.save(stream, value, allow_pickle=False)
    write_ms = (time.perf_counter() - write_started) * 1000.0
    hash_started = time.perf_counter()
    file_record = c.record(path)
    hash_readback_ms = (time.perf_counter() - hash_started) * 1000.0
    return dict(file_record, shape=list(value.shape), dtype=value.dtype.str,
                raw_sha256=buffer_sha(value), finite=True, min=float(value.min()), max=float(value.max()),
                save_array_io=dict(d2h_and_cpu_materialization_ms=d2h_cpu_ms,
                                   npy_write_wall_ms=write_ms, file_hash_readback_wall_ms=hash_readback_ms))


def loaded_sources():
    allowed = c.allowed_source_map()
    result, virtual = collect_module_sources(sys.modules, c.RUNTIME, c.source_root('PARENT_OFF'),
                                            allowed, FRONT_MODULES, c.record, c.require)
    _VIRTUAL_SOURCE_MODULES.clear()
    _VIRTUAL_SOURCE_MODULES.update(virtual)
    expected = {}
    root = c.source_root('PARENT_OFF')
    for item in c.manifest()['source_files']:
        rel = Path(item['relative_path'])
        if rel.suffix != '.py': continue
        parts = rel.parts
        if parts[0] == 'game' or 'modules' in parts:
            name = rel.stem
        elif 'nr_backend' in parts:
            tail = parts[parts.index('nr_backend'):]
            name = '.'.join(tail[:-1] if rel.stem == '__init__' else (*tail[:-1],rel.stem))
        else: continue
        expected[name] = root / rel
    for name,path in expected.items():
        module = sys.modules.get(name)
        if module is not None:
            file = vars(module).get('__file__')
            c.require(file and Path(file).resolve() == path.resolve() and
                      c.sha(path) == allowed[str(path.resolve())], 'Loaded model/provider source is outside PHASE2 pins: '+name)
    return result


def child_retirement(suite):
    reports = {}
    for label,child in suite.children.items():
        active = getattr(child, 'active', None)
        retired_flag = getattr(child, 'retired', None)
        live = getattr(child, '_live', None)
        retired = (retired_flag is True and active is False) or live is False
        reports[label] = dict(retired=retired, active=active, retired_flag=retired_flag, live=live)
        c.require(retired, 'Numerical child did not retire: ' + label)
    return reports


def counter_map(snapshot):
    return {label: {name: dict(child[name]) for name in ('calls', 'capture_calls', 'eager_calls') if isinstance(child.get(name), dict)}
            for label, child in snapshot['children'].items()}


def raw_graph(torch, entry, graph, policy, child, output, np, out):
    """No input changes, Python model dispatch, file scan or conversion in raw loop."""
    c.require(sys.getprofile() is None, 'Observer may not be installed during raw timing')
    pairs = [(torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)) for _ in range(50)]
    for begin,end in pairs: begin.record(); end.record()
    torch.xpu.synchronize()
    before = dict(host_cache_dispatches=policy.hits, graph_replays=graph.replays,
                  entry_replays=entry.replays, child_calls=None if child is None else dict(child.calls))
    # This is the complete captured network body including layout and the V6
    # pool-external publication copy. Front/history preparation and game/Present
    # costs are excluded and are not inferred from this device-only metric.
    for begin,end in pairs:
        begin.record()
        entry.graph.replay()
        end.record()
    torch.xpu.synchronize()
    timings = [begin.elapsed_time(end) for begin,end in pairs]
    c.require(policy.hits == before['host_cache_dispatches'] and graph.replays == before['graph_replays'] and
              entry.replays == before['entry_replays'] and
              (child is None or child.calls == before['child_calls']), 'Raw replay unexpectedly re-entered host/model/JIT')
    after = save_array(np, entry.output, out / 'raw-final-body.npy')
    c.require(after['raw_sha256'] == output['raw_sha256'], 'Identical immutable body inputs changed output on raw replay')
    return dict(**c.stats(timings), metric='complete captured network body + layout + V6 publication copy; XPU device events',
                excludes=['front/history preparation and commit', 'geometry/compositing', 'game/bridge/Present', 'host wall latency'],
                auxiliary_direct_graph_replays=50, python_replay_counters_intentionally_unchanged=True,
                host_cache_dispatches_before=before['host_cache_dispatches'], host_cache_dispatches_after=policy.hits,
                no_host_scan_or_matrix_reformat_in_raw_loop=True, final_body=after)


