"""Actual XPU child implementation. This module's import is stdlib-only."""
from __future__ import annotations
from contextlib import ExitStack, nullcontext
from dataclasses import asdict
import copy
import hashlib
import importlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback
import release_contracts as c
from portable_runtime import write_json, report_value
from reproduction_cache import CachePhase

_FAILED_OWNERS = []
ROOT_STATES = {
    ('audit_swin_720_v1', 'Counter'): ('active', 'retired'),
    ('audit_front_decoder_post_720_v1', 'Scope'): ('active', '_retired'),
    ('audit_vit_c512_scope_720_v1', 'CompleteCounter'): ('live', 'retired'),
    ('audit_history_host_720_v1', 'HistoryAdmissionCounter'): ('active', None),
    ('audit_vit_c512_720_v1', 'NumericStageHook'): (None, None),
}

def retired_state(child, *, root=False):
    fields = {k:v for k,v in vars(child).items() if k in
              ('active','retired','_retired','live','_live','closed') and type(v) is bool}
    if root:
        rule = ROOT_STATES.get((type(child).__module__,type(child).__name__))
        c.require(rule is not None, 'unsupported real root child lifecycle schema')
        live, retired = rule
    else:
        live = next((k for k in ('active','_live','live') if k in fields), None)
        retired = next((k for k in ('retired','_retired') if k in fields), None)
        c.require(live is not None, 'numeric child has no actual lifecycle field')
    if live: c.require(fields.get(live) is False, 'child still live after close')
    if retired: c.require(fields.get(retired) is True, 'child retirement flag absent')
    return dict(class_module=type(child).__module__, class_name=type(child).__name__,
                fields=fields, live_field=live, retired_field=retired,
                adapter_metadata=live is None)

def dispatch_counts(value):
    out = {}
    def walk(v, path):
        if isinstance(v,dict):
            for key,child in v.items():
                name = path+'/'+str(key)
                if key in ('validate_calls','validation_calls','preflight_calls'): continue
                if isinstance(key,str) and (key.endswith('calls') or key.endswith('hits') or
                    key in ('frame_sequence','body_graph_replays')) and isinstance(child,(dict,list,int)):
                    out[name] = copy.deepcopy(child)
                elif key not in ('sources','preflight','preflights','resources'):
                    walk(child,name)
    walk(value,'')
    return out

def inventory(root):
    return {name:c.sha(path) for name,path in c.all_files(root).items()}

def stats(values):
    c.require(values and all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in values), 'invalid event samples')
    return dict(count=len(values), samples_ms=values, mean_ms=statistics.mean(values), median_ms=statistics.median(values))

def array_summary(np, tensor, output, name, *, persist):
    started = time.perf_counter()
    a = tensor.detach().cpu().contiguous().numpy()
    c.require(a.dtype.kind == 'f' and bool(np.isfinite(a).all()), 'nonfinite/nonfloat output/history')
    row = dict(shape=list(a.shape), dtype=a.dtype.str, finite=True,
               raw_sha256=hashlib.sha256(memoryview(a).cast('B')).hexdigest(), min=float(a.min()), max=float(a.max()))
    if persist:
        target = c.under(output, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            np.save(stream, a, allow_pickle=False)
        row.update(path=name, file_sha256=c.sha(target), bytes=target.stat().st_size)
    row['readback_hash_save_wall_ms'] = (time.perf_counter()-started)*1000
    return row

def inputs(np, seed, external):
    if external:
        path = c.plain_path(external)
        c.validate_inputs(path)
        fixture = c.read_json(path)
        def load(ordinal):
            row = fixture['frames'][ordinal]
            a = np.load(c.under(path.parent, row['rgb']['path']), allow_pickle=False)
            m = np.load(c.under(path.parent, row['motion']['path']), allow_pickle=False)
            return a, m
        return load, dict(scenario=fixture.get('scenario', 'external NPY'),
                         contract='current-to-previous pixel displacement', external_manifest=c.file_record(path),
                         private_face_video=False, input_provenance='user external NPY; no inherited quality qualification')
    # Per-frame integer seed makes reset cycles exactly reproduce inputs, without
    # retaining another 13-frame copy or conflating same-frame raw body timing.
    def load(ordinal):
        rng = np.random.default_rng(np.random.SeedSequence([seed, ordinal]))
        y, x = np.mgrid[0:720, 0:1280].astype(np.float32)
        base = np.stack(((x+ordinal)/1293, (y+ordinal)/733,
                         (x+y+ordinal)/2025), axis=-1)
        rgb = np.clip(base+0.025*rng.standard_normal(base.shape, dtype=np.float32), 0, 1).astype('<f4')
        motion = np.empty((720, 1280, 2), dtype='<f4')
        motion[..., 0] = 0.75*np.sin(y/97+ordinal/13)
        motion[..., 1] = 0.5*np.cos(x/113+ordinal/13)
        return np.ascontiguousarray(rgb), motion
    return load, dict(scenario='synthetic fixed-seed SDR gradients + noise + pixel motion', seed=seed,
                     generator='numpy.default_rng(SeedSequence([seed, ordinal]))',
                     contract='current-to-previous pixel displacement', private_face_video=False,
                     input_provenance='synthetic smoke; no private-video/NVIDIA/game quality claim')

def process_frame(torch, np, modes, controls, load, out, ordinal, cycle, persist):
    started = time.perf_counter()
    rgb, motion = load(ordinal)
    c.require(rgb.shape == (720,1280,3) and motion.shape == (720,1280,2) and
              rgb.dtype.str == motion.dtype.str == '<f4' and rgb.flags.c_contiguous and motion.flags.c_contiguous
              and np.isfinite(rgb).all() and np.isfinite(motion).all() and (rgb >= 0).all() and (rgb <= 1).all(), 'input layout/finite/range failure')
    input_wall = (time.perf_counter()-started)*1000
    input_hash = dict(rgb=hashlib.sha256(memoryview(rgb).cast('B')).hexdigest(),
                      motion=hashlib.sha256(memoryview(motion).cast('B')).hexdigest())
    model = modes.session._stack.model
    before_seed, before_history = model.next_seed, model._previous
    c.require(ordinal == 0 or before_seed == ordinal and before_history is not None, 'history/seed input sequence changed')
    started = time.perf_counter()
    rgb_xpu, motion_xpu = torch.from_numpy(rgb).to('xpu'), torch.from_numpy(motion).to('xpu')
    torch.xpu.synchronize()
    h2d_wall = (time.perf_counter()-started)*1000
    begin, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    begin.record(); end.record(); torch.xpu.synchronize()
    started = time.perf_counter()
    begin.record()
    output = modes.process(rgb_xpu, motion_xpu, height=720, reset=ordinal == 0,
                           history_warp='fused', graph_replay=True, controls=controls, variant='unrounded')
    end.record()
    call_wall = (time.perf_counter()-started)*1000
    started = time.perf_counter()
    torch.xpu.synchronize()
    sync_wall = (time.perf_counter()-started)*1000
    event = float(begin.elapsed_time(end))
    c.require(event > 0 and tuple(output.shape) == (720,1280,3) and output.dtype == torch.float32 and
              model.next_seed == ordinal+1 and model._previous is not None and
              model._previous is not before_history and model._previous.data_ptr() != output.data_ptr(), 'output/private history/seed commit failure')
    return dict(ordinal=ordinal, cycle=cycle, reset=ordinal == 0, seed_before=before_seed,
                seed_after=model.next_seed, inputs=input_hash, motion_nonzero=bool(np.count_nonzero(motion)),
                route=modes.last_combo_frame_route,
                capture_gate=copy.deepcopy(modes.last_combo_capture_gate),
                implementation_capture_evidence=copy.deepcopy(getattr(modes, 'last_implementation_capture_evidence_720', None)),
                implementation_provider_status=copy.deepcopy(getattr(modes, 'implementation_provider_status_720', None)),
                output=array_summary(np, output, out, f'output-{ordinal:02d}.npy', persist=persist),
                history=array_summary(np, model._previous, out, f'history-{ordinal:02d}.npy', persist=persist),
                full_process_xpu_ms=event, host=dict(input_load_generate_wall_ms=input_wall,
                    input_h2d_sync_wall_ms=h2d_wall, process_call_wall_ms=call_wall, post_event_sync_wall_ms=sync_wall))

def observations(modes):
    graph = modes.session._stack.graph
    suite = modes.numeric_cleanup_calls
    return dict(graph=report_value(graph.metadata()), graph_replays=graph.replays, graph_entries=len(graph.entries),
        numeric=None if suite is None else suite.snapshot(),
        implementation={stage:owner.snapshot() for stage,owner in getattr(modes, 'implementation_calls_720', {}).items()},
        original_capture_counters={name:report_value(getattr(getattr(modes, name, None), 'capture_calls', None))
            for name in ('c32_hidden_calls','c512_probability_calls','c128_pairwise_calls',
                         'c64_attention_project_calls','c128_attention_project_calls')},
        c512_library_calls=copy.deepcopy(modes.c512_library_calls), native_k8_calls=copy.deepcopy(modes.native_k8_calls))

def raw_events(torch, np, modes, policy, out):
    graph = modes.session._stack.graph
    entry, model = graph.last_entry, modes.session._stack.model
    c.require(entry is not None, 'raw timing needs an actual captured graph entry')
    before = dispatch_counts(observations(modes))
    replay_counts = (graph.replays,entry.replays,len(graph.entries))
    history_before = array_summary(np, model._previous, out, '', persist=False)
    body_before = array_summary(np, entry.output, out, '', persist=False)
    hits = policy.hits
    pairs = [(torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)) for _ in range(50)]
    for a,b in pairs: a.record(); b.record()
    torch.xpu.synchronize()
    for a,b in pairs:
        a.record(); entry.graph.replay(); b.record()
    torch.xpu.synchronize()
    values = [float(a.elapsed_time(b)) for a,b in pairs]
    c.require(before == dispatch_counts(observations(modes)) and
              replay_counts == (graph.replays,entry.replays,len(graph.entries)) and hits == policy.hits and
              array_summary(np, model._previous, out, '', persist=False)['raw_sha256'] == history_before['raw_sha256'] and
              array_summary(np, entry.output, out, '', persist=False)['raw_sha256'] == body_before['raw_sha256'] and
              model.next_seed == 13, 'raw body changed host counters, immutable output or committed history')
    return dict(stats=stats(values), direct_graph_replays=50,
        metric='same-frame actual captured model/math/layout/publication body',
        excludes=['front/history prep and commit','full modes geometry/composite','input I/O/H2D/readback/save','game/bridge/Present'],
        host_dispatch_counters_unchanged=True, private_history_unchanged=True, game_FPS=None)

def loaded(view, expected):
    result = {}
    for name, module in list(sys.modules.items()):
        file = vars(module).get('__file__') if module is not None else None
        if file:
            path = Path(file).resolve()
            if path.is_relative_to(view) and path.suffix in ('.py', '.pyd', '.dll', '.so'):
                rel = path.relative_to(view).as_posix()
                c.require(expected.get(rel) == c.sha(path), 'loaded source/binary provenance drift: '+name)
                result[name] = dict(relative_path=rel, sha256=expected[rel])
    for name, rel in (('nr_game_fullsize','game/nr_game_fullsize.py'),
                      ('nr_backend','experimental/fp8_unround_overlay/nr_backend/__init__.py'),
                      ('graph_front_v6','experimental/fp8_unround_overlay/modules/graph_front_v6.py')):
        c.require(name in result and result[name]['relative_path'] == rel, 'loaded provider resolved outside selected source: '+name)
    c.require('nr_game_pre_xess_host' not in sys.modules and 'cyberpunk_nr_web' not in sys.modules, 'unnecessary game bridge was loaded')
    return result

def driver_info():
    if os.name == 'nt':
        proc = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command',
            'Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate | ConvertTo-Json -Compress'],
            capture_output=True, text=True, timeout=30)
        return dict(method='Win32_VideoController', returncode=proc.returncode,
                    actual_output=proc.stdout.strip(), error=proc.stderr.strip())
    return dict(method='platform/sysfs', platform=platform.platform(),
                kernel=platform.release(), driver_version='not independently available')

def game_profiles(profiles, combine, constructor):
    selected=[]
    for name,profile in profiles.items():
        legacy,numeric=tuple(profile.legacy),tuple(profile.numeric)
        if legacy and all(k in c.FLAGS and constructor.get(k) == v for k,v in legacy) and not numeric:
            selected.append(name)
        if numeric and not legacy and all(k in c.NUMERIC_BASE and
                report_value(constructor['numeric_cleanup_720'].get(k)) == report_value(v) for k,v in numeric):
            selected.append(name)
    # These three switches are initializer configuration fields in the actual
    # game host, rather than optimization names in PROFILES.
    structural={k:constructor[k] for k in ('c128_pairwise_720','c64_attention_project_720','c128_attention_project_720')}
    combined=dict(structural,**combine(selected))
    return selected,combined,structural

def run_child(config_path, phase):
    config_path = c.plain_path(config_path)
    config = c.read_json(config_path)
    view, output, cache = (c.plain_path(config[k]) for k in ('view','output','cache'))
    out = c.under(output, phase)
    out.mkdir()
    result = dict(schema='nr-release-reproduction-child-v1', phase=phase, completed=False, GPU_executed=False,
        scenario=None, claim_scope='offline source/graph smoke and same-input pair; no private-video/game/NVIDIA qualification',
        game_FPS=None, frames=[], cycles=[], failures=[], config=c.file_record(config_path))
    torch = modes = session = graph = policy = None
    suite = None; owners = {}; numeric_children = {}; root_children = {}
    failure_category = 'SOURCE_FAILURE'
    started = time.perf_counter()
    try:
        c.require(inventory(view) == config['view_files'], 'prepared runtime view bytes changed')
        for name, expected in config['runner_files'].items():
            c.require(c.sha(c.under(c.HERE,name)) == expected, 'reproduction runner changed after preparation: '+name)
        before_cache = inventory(cache)
        result['cache_before'] = before_cache
        os.environ['NR_FAST_UNROUND_RUNTIME_ROOT'] = str(view)
        os.environ['NR_FAST_UNROUND'] = ''
        sys.path[:0] = [str(view), str(view/'game')]
        # This is the dedicated portable runtime's real isolation function.
        # A normal Torch-XPU venv can instead supply its own loader environment.
        environment = importlib.import_module('runtime_environment')
        handles = []
        if os.name == 'nt' and (Path(sys.executable).parent/'Library/bin/sycl9.dll').is_file():
            directory, library, env = environment.isolate()
            handles.extend((directory, library)); result['isolation'] = env
        else:
            result['isolation'] = dict(mode='explicit user Torch-XPU environment', Python=sys.executable)
        bootstrap = importlib.import_module('fast_cached_runtime_v1')
        bootstrap.CACHE = cache
        bootstrap.bootstrap()
        spec = importlib.util.spec_from_file_location('_release_overlay', view/'experimental/fp8_unround_overlay/bootstrap.py')
        overlay = importlib.util.module_from_spec(spec); spec.loader.exec_module(overlay)
        result['overlay'] = overlay.activate(view, '')
        torch = importlib.import_module('torch'); np = importlib.import_module('numpy')
        triton = importlib.import_module('triton')
        failure_category = 'ENVIRONMENT_FAILURE'
        c.require(torch.xpu.is_available(), 'Torch XPU unavailable')
        result['GPU_executed'] = True
        torch.xpu.set_device(0)
        torch.set_num_threads(8); torch.set_num_interop_threads(1)
        from triton.runtime import driver
        result['environment'] = dict(Python=sys.version, executable=sys.executable, platform=platform.platform(),
            torch_version=str(torch.__version__), triton_version=getattr(triton,'__version__', 'unknown'),
            numpy_version=np.__version__, triton_origin=str(Path(triton.__file__).resolve()),
            device_name=torch.xpu.get_device_name(0), device_properties=str(torch.xpu.get_device_properties(0)),
            actual_target=report_value(driver.active.get_current_target()), driver=driver_info())
        from nr_game_fullsize import FullsizeGameModes
        from nr_backend.controlled_temporal import NRControls
        from nr_game_controls import Settings
        from numeric_game_profiles_720_v1 import PROFILES,combined_mode_options
        controls = NRControls(**config['controls'])
        selected_profiles,accepted_game_options,structural=game_profiles(PROFILES,combined_mode_options,config['constructor'])
        c.require((all(accepted_game_options.get(k) is True for k in c.FLAGS) and
                  report_value(accepted_game_options.get('numeric_cleanup_720')) == report_value(config['constructor']['numeric_cleanup_720'])) or
                  config['source']['experiment_reference'] is not None,
                  'actual game profile registry cannot express the accepted numeric/six-flag settings')
        settings = Settings(input_size=720, history_mode='fused', graph_replay=True,
            backend_variant='unrounded', experiment_720='c512_k8', style=config['controls']['style'],
            optimizations_720=tuple(selected_profiles),
            model_intensity=config['controls']['intensity'], local_tone=config['controls']['local_tone'],
            local_structure=config['controls']['local_structure'], auto_mask=config['controls']['auto_mask'],
            skin_structure=config['controls']['skin_structure'])
        result['installed_settings'] = asdict(settings)
        result['installed_structure_switches'] = structural
        result['selected_constructor'] = config['constructor']
        result['conditional_coverage']=dict(style=config['controls']['style'],
            unexecuted_styles=[s for s in (0,1,2) if s != config['controls']['style']],
            low_strength='unexecuted; separate real adapter harness required',
            quality_qualification=False)
        load, scenario = inputs(np, config['seed'], config.get('inputs'))
        result['scenario'] = scenario
        from rows_fscache_guard_v1 import install
        install()
        # r18 helpers may batch when an explicit CPU compile session is active.
        # This reproduction deliberately keeps both compilation and GPU work
        # serial; source selection/cold fallback order remains unchanged.
        parallel = view/'game/parallel_compile_720_v1.py'
        cold = (importlib.import_module('parallel_compile_720_v1').ColdCompileSession(
                precompile=False, workers=1) if parallel.is_file() else nullcontext())
        prepared_result = None if phase == 'precompile' else c.read_json(output/'precompile/RESULT.json')
        prepared = None if prepared_result is None else prepared_result['cache_gate']['actual_compiler_keys']
        c.require(prepared_result is None or prepared_result['completed'], 'precompile did not complete')
        failure_category = 'CACHE_FAILURE'
        with cold, CachePhase(bootstrap, readonly=phase == 'readonly', prepared=prepared) as policy:
            with torch.inference_mode(False):
                failure_category = 'ADMISSION_FAILURE'
                modes = FullsizeGameModes(view/'exact', Path(config['profile']['path']), config['profile']['sha256'],
                                          **config['constructor'])
                modes.select(720, source=(720,1280), variant='unrounded')
                session = modes.session; graph = session._stack.graph; suite = modes.numeric_cleanup_calls
                c.require(suite is not None and modes.numeric_cleanup_720.identity == config['numeric_identity']['base_identity'] and
                          suite.snapshot()['identity'] == config['numeric_identity']['suite_identity'], 'actual numeric/suite identity mismatch')
                result['numeric_identity'] = config['numeric_identity']
                owners = dict(getattr(modes,'implementation_calls_720', {}))
                numeric_children = {} if suite is None else dict(suite.children)
                root_children = {stage:dict(owner.children) for stage,owner in owners.items()}
                c.require(tuple(session.fullsize_geometry) == (720,1280) and
                          tuple(session.fullsize_padding['padding']) == (768,1280) and
                          type(graph).__module__ == 'graph_front_v6' and len(session._stack.model.vit) == 8,
                          'actual720 padded/L8/GraphFrontV6 route mismatch')
                failure_category = 'COMPILE_FAILURE'
                if suite is not None: suite.preflight()
                for owner in owners.values(): owner.preflight()
            with torch.inference_mode():
                failure_category = 'ADMISSION_FAILURE'
                for ordinal in range(13):
                    row = process_frame(torch,np,modes,controls,load,out,ordinal,'initial13',True)
                    result['frames'].append(row)
                    if ordinal < 2:
                        gate = row['capture_gate']
                        c.require(isinstance(gate,dict) and all(gate.values()) and
                                  all(gate.get(k) is True for k in c.GATES), 'actual accepted six/C512K8 graph capture gate failed')
                c.require([r['route'] for r in result['frames']] == ['capture']*2+['replay']*11 and
                          len(graph.entries) == 2 and graph.replays == 13, 'initial13 route/object counter mismatch')
                failure_category = 'NUMERIC_FAILURE'
                for cycle in range(2+config['repeats']):
                    rows = []
                    for ordinal in range(13):
                        row = process_frame(torch,np,modes,controls,load,out,ordinal,cycle,False)
                        rows.append(row)
                        c.require(row['route'] == 'replay' and len(graph.entries) == 2, 'warm sequence added a graph/capture')
                        for role in ('output','history'):
                            c.require(row[role]['raw_sha256'] == result['frames'][ordinal][role]['raw_sha256'],
                                      'repeated reset/sequence output or private history changed: '+role)
                    result['cycles'].append(dict(cycle=cycle, measured=cycle >= 2, frames=rows,
                        full_process=stats([r['full_process_xpu_ms'] for r in rows])))
                total = 13*(1+2+config['repeats'])
                c.require(graph.replays == total and len(graph.entries) == 2, 'actual graph object replay counter mismatch')
                result['graph_routes'] = dict(initial_captures=2, frame_replay_routes=11+13*(2+config['repeats']),
                                              actual_graph_object_replays=graph.replays, entries=len(graph.entries))
                result['actual_evidence'] = observations(modes)
                measured = [r['full_process_xpu_ms'] for cycle in result['cycles'] if cycle['measured'] for r in cycle['frames']]
                result['full_process_timing'] = dict(stats=stats(measured),
                    boundary='same-stream XPU events around the complete FullsizeGameModes.process call',
                    includes=['preparation','history warp','model/graph','history/seed commit','geometry/composite','stream idle while Python submits'],
                    excludes=['input generation/load/H2D','event initialization','post-event wait','readback/hash/save','game/bridge/Present'],
                    private_stage_device_times='unknown', CPU_wall_separate=True, game_FPS=None)
                if phase == 'readonly': result['raw_body_timing'] = raw_events(torch,np,modes,policy,out)
            result['loaded_modules'] = loaded(view, config['view_files'])
            c.require(policy.hits > 0 and (phase != 'readonly' or policy.disk.hits > 0 and not policy.write_attempts), 'compiler/DiskOnly evidence absent')
            if prepared_result is not None:
                c.require(prepared_result['environment'] == result['environment'] and
                          prepared_result['loaded_modules'] == result['loaded_modules'], 'fresh readonly runtime provenance changed')
            result['completed'] = True
    except BaseException:
        result['failures'].append(dict(category=failure_category, stack=traceback.format_exc()))
    finally:
        # Preserve original failures/actual owners; no cleared field is used to
        # disguise failed retirement. All close failures stop the supervisor.
        if modes is not None:
            try: result['failure_or_final_observation'] = observations(modes)
            except BaseException: result['observation_error'] = traceback.format_exc()
        try:
            if torch is not None: torch.xpu.synchronize()
            if modes is not None: modes.close()
            states = {'numeric':{}, 'implementation':{}}
            for label,child in numeric_children.items():
                states['numeric'][label] = retired_state(child)
            for stage,children in root_children.items():
                states['implementation'][stage] = {}
                for label,child in children.items():
                    states['implementation'][stage][label] = retired_state(child, root=True)
            c.require(modes is None or modes.session is None and session._closed and graph.closed
                      and not graph.entries and graph.last_entry is None and (suite is None or suite.closed),
                      'session/graph/suite retirement incomplete')
            c.require(all(owner.active is False and owner.retired is True for owner in owners.values()), 'implementation owner retirement incomplete')
            result['retirement'] = dict(verified=True, actual_child_states=states)
        except BaseException:
            result['failures'].append(dict(category='RETIREMENT_FAILURE',stack=traceback.format_exc()))
            _FAILED_OWNERS.append((modes,session,graph,suite,owners,numeric_children,root_children))
        if policy is not None: result['cache_gate'] = policy.snapshot()
        result['cache_after'] = inventory(cache)
        if phase == 'readonly' and result.get('cache_before') != result['cache_after']:
            result['failures'].append(dict(category='CACHE_FAILURE',stack='readonly cache inventory changed'))
        result['completed'] = bool(result['completed'] and not result['failures'])
        result['status'] = 'OFFLINE_REPRODUCTION_COMPLETED' if result['completed'] else 'FAILED_NO_FURTHER_LAUNCH'
        result['phase_wall_ms'] = (time.perf_counter()-started)*1000
        write_json(out/'RESULT.json', result)
    return 0 if result['completed'] else 1

def compare_saved(left, right):
    """CPU NumPy array comparison, called after all GPU children have exited."""
    import numpy as np
    a, b = c.read_json(left), c.read_json(right)
    ca, cb = c.read_json(a['config']['path']), c.read_json(b['config']['path'])
    c.require(a['completed'] and b['completed'] and ca['controls'] == cb['controls'] and
              len(a['frames']) == len(b['frames']) == 13 and
              a['scenario'] == b['scenario'], 'baseline/candidate condition mismatch')
    rows = []
    for first,second in zip(a['frames'],b['frames']):
        c.require(all(first[k] == second[k] for k in ('ordinal','reset','seed_after','inputs')), 'paired input/sequence mismatch')
        row = dict(ordinal=first['ordinal'])
        for role in ('output','history'):
            arrays = []
            for result_path,record in ((Path(left),first[role]),(Path(right),second[role])):
                path = c.under(result_path.parent,record['path'])
                c.require(c.sha(path) == record['file_sha256'], 'saved pair array SHA changed')
                value = np.load(path,allow_pickle=False)
                c.require(list(value.shape) == record['shape'] and value.dtype.str == record['dtype'] and
                          np.isfinite(value).all() and hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast('B')).hexdigest() == record['raw_sha256'], 'saved pair payload contract changed')
                arrays.append(value)
            c.require(arrays[0].shape == arrays[1].shape, 'pair array shape differs')
            delta = arrays[1].astype(np.float64)-arrays[0].astype(np.float64)
            mse = float(np.mean(delta*delta))
            row[role] = dict(finite=True, MAE=float(np.mean(np.abs(delta))), max_error=float(np.max(np.abs(delta))),
                MSE=mse, PSNR_db=None if mse == 0 else 10*math.log10(1/mse),
                PSNR_infinite=mse == 0, PSNR_peak=1.0, changed_components=int(np.count_nonzero(delta)),
                changed_pixels=int(np.count_nonzero(np.any(delta != 0,axis=-1))) if delta.ndim == 3 else None,
                byte_identical=first[role]['raw_sha256'] == second[role]['raw_sha256'])
        rows.append(row)
    timing=None
    if 'full_process_timing' in a and 'full_process_timing' in b:
        base=a['full_process_timing']['stats']['mean_ms'];candidate=b['full_process_timing']['stats']['mean_ms']
        timing=dict(full_process_baseline_mean_ms=base,full_process_candidate_mean_ms=candidate,
                    full_process_candidate_minus_baseline_ms=candidate-base,
                    sample_count_baseline=a['full_process_timing']['stats']['count'],
                    sample_count_candidate=b['full_process_timing']['stats']['count'],
                    metric='full FullsizeGameModes.process XPU events; CPU wall separate; no game FPS',
                    raw_body_candidate_minus_baseline_ms=None)
        c.require(timing['sample_count_baseline']==timing['sample_count_candidate'],'paired number of measured frames differs')
        if a.get('raw_body_timing') and b.get('raw_body_timing'):
            timing['raw_body_candidate_minus_baseline_ms']=b['raw_body_timing']['stats']['mean_ms']-a['raw_body_timing']['stats']['mean_ms']
    return dict(schema='nr-release-saved-pair-v1', baseline=c.file_record(left), candidate=c.file_record(right),
        frames=rows, all_finite=True, output_all_byte_identical=all(r['output']['byte_identical'] for r in rows),
        history_all_byte_identical=all(r['history']['byte_identical'] for r in rows),
        quality_threshold_applied=False, quality_acceptance=False, timing_comparison=timing,
        scenario=a['scenario'], game_FPS=None)
