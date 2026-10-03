"""Luna subprocess only: fixed720 complete model, graph and full modes events.

Import is CPU/stdlib only. Tensor/device imports are inside run(), after every
source, manifest, READY, workload and process-lock proof has been checked.
"""
from __future__ import annotations
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import runner_common as c
from cache_policy import CachePolicy, json_value
import evidence as ev
import failure_policy as fp
from worker_support import buffer_sha, save_array, raw_graph, loaded_sources, FRONT_MODULES

_FAILED_RETIREMENTS = []

def progress(stage, **fields):
    print(json.dumps(dict(event='PHASE2_PROGRESS', arm=c.ARM, stage=stage, **fields),
                     ensure_ascii=False, allow_nan=False), flush=True)

def process_one(torch, np, modes, controls, fixture_frame, out, *, cycle, ordinal, persist, expected_frame=None):
    model = modes.session._stack.model
    load_start = time.perf_counter()
    with np.load(c.checked(dict(path=fixture_frame['npz'], sha256=fixture_frame['npz_sha256'])), allow_pickle=False) as source:
        rgb, motion = source['rgb'], source['motion']
    load_ms = (time.perf_counter() - load_start)*1000
    c.require(rgb.shape == (720,1280,3) and motion.shape == (720,1280,2)
              and rgb.dtype.str == motion.dtype.str == '<f4' and rgb.flags.c_contiguous and motion.flags.c_contiguous,
              'Original actual720 input/motion layout changed')
    c.require(buffer_sha(rgb) == fixture_frame['rgb_float_sha256'] and buffer_sha(motion) == fixture_frame['motion_float_sha256']
              and np.isfinite(rgb).all() and np.isfinite(motion).all(), 'Frozen real material SHA/finite failure')
    before_seed, before_history = model.next_seed, model._previous
    c.require((ordinal == 0 and fixture_frame['reset']) or
              (before_seed == ordinal and before_history is not None), 'History/seed entry sequence changed')
    h2d = time.perf_counter()
    rgb_xpu, motion_xpu = torch.from_numpy(rgb).to('xpu'), torch.from_numpy(motion).to('xpu')
    torch.xpu.synchronize()
    h2d_ms = (time.perf_counter() - h2d)*1000
    # Prime both events before the boundary. Input loading/H2D, event creation,
    # persistence, and completion wait are outside the measured process region.
    begin, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    begin.record(); end.record(); torch.xpu.synchronize()
    started = time.perf_counter()
    begin.record()
    output = modes.process(rgb_xpu, motion_xpu, height=720, reset=fixture_frame['reset'],
                           history_warp='fused', graph_replay=True, controls=controls, variant='unrounded')
    end.record()
    call_ms = (time.perf_counter() - started)*1000
    sync = time.perf_counter()
    torch.xpu.synchronize()
    sync_ms = (time.perf_counter() - sync)*1000
    event_ms = float(begin.elapsed_time(end))
    c.require(event_ms > 0 and tuple(output.shape) == c.FRAME_SHAPE and str(output.dtype) == 'torch.float32'
              and model.next_seed == ordinal + 1 and model._previous is not None
              and model._previous is not before_history and model._previous.data_ptr() != output.data_ptr(),
              'Full modes output/history/seed commit or event boundary failed')
    started = time.perf_counter()
    if persist:
        output_row = save_array(np, output, out / f'output-{ordinal:02d}.npy')
        history_row = save_array(np, model._previous, out / f'history-{ordinal:02d}.npy')
    else:
        def summary(tensor, role):
            data = tensor.detach().cpu().contiguous().numpy()
            c.require(np.isfinite(data).all(), 'Nonfinite warmup output/history')
            row = dict(shape=list(data.shape), dtype=data.dtype.str, finite=True, raw_sha256=buffer_sha(data))
            if expected_frame is not None and row['raw_sha256'] != expected_frame[role]['raw_sha256']:
                from failure_evidence import persist_mismatch
                row['failure_diagnostic'] = persist_mismatch(np, data, expected_frame[role],
                    out / ('mismatch-%02d-%s.npy' % (ordinal, role)), c.checked, c.output_path, c.record)
            return row
        output_row, history_row = summary(output, 'output'), summary(model._previous, 'history')
    persist_ms = (time.perf_counter() - started)*1000
    return dict(frame_id=fixture_frame['frame_id'], cycle=cycle, reset=fixture_frame['reset'],
        seed_before=before_seed, seed_after=model.next_seed, effective_seed=ordinal,
        reset_discards_previous=fixture_frame['reset'],
        original_rgb_raw_sha256=buffer_sha(rgb), original_motion_raw_sha256=buffer_sha(motion),
        motion_nonzero=bool(np.count_nonzero(motion)), route=modes.last_combo_frame_route,
        combo_capture_gate=copy.deepcopy(modes.last_combo_capture_gate),
        implementation_capture_evidence=copy.deepcopy(modes.last_implementation_capture_evidence_720),
        implementation_provider_status=copy.deepcopy(modes.implementation_provider_status_720),
        output=output_row, history=history_row, whole_modes_xpu_ms=event_ms,
        host_timing=dict(input_npz_load_wall_ms=load_ms, input_h2d_and_sync_wall_ms=h2d_ms,
                         process_call_wall_ms=call_ms, completion_sync_wall_ms=sync_ms,
                         output_and_history_persistence_wall_ms=persist_ms))

def run(args):
    c.require(args.execute_gpu, 'Luna must explicitly select --execute-gpu')
    c.require(Path(sys.executable).resolve() == (c.RUNTIME / 'python/python.exe').resolve()
              and sys.flags.isolated and sys.flags.utf8_mode and sys.dont_write_bytecode,
              'Child needs pinned runtime Python -I -X utf8 -B')
    c.configure(args.manifest, args.arm, args.manifest_sha256)
    lock = c.read(c.checked(dict(path=args.lock, sha256=args.lock_sha256)))
    c.require(lock['pid'] == os.getppid() and lock['manifest_sha256'] == c.MANIFEST_SHA256
              and lock['schema'] == 'b580-phase2-serial-lock-v1', 'Worker lacks supervisor process lock')
    c.require(c.output_path(args.lock).parent == c.DATA.resolve(), 'Wrong execution lock location')
    out, cache = c.output_path(args.output), c.output_path(args.cache)
    c.require(out.is_dir() and cache.is_dir() and not (out / 'RESULT.json').exists(), 'Use fresh allocated child output')
    precompile = args.phase == 'precompile'
    c.require((args.tag in c.PREPARE_TAGS) is precompile, 'Child phase/tag mismatch')
    result = dict(schema='b580-phase2-fullmodel-child-v1', arm=c.ARM, phase=args.phase, tag=args.tag,
        manifest_sha256=c.MANIFEST_SHA256, mode_options=c.mode_options(),
        constructor_sha256=c.digest_obj(c.mode_options()), numeric_options=c.effective_numeric(),
        source_tree=str(c.source_root()), workload=c.workload(), completed=False, status='FAILED_UNACCEPTED',
        GPU_executed=False, game_or_API_calls=0, game_acceptance=False, frames=[], cleanup=None,
        whole_modes_timing=dict(status='UNKNOWN', reason='Full modes boundary not yet completed'), raw=None,
        failure_events=[])
    modes = session = graph = suite = torch = policy = None
    owners = {}; numeric_children = {}; root_children = {}; resources = []; failures = []
    started = time.monotonic()
    failure_category, failure_operation = 'SOURCE_FAILURE', 'source and fixture pin verification'
    try:
        result['sources'] = c.verify_sources()
        fixture, _ = c.fixture()
        failure_category, failure_operation = 'CACHE_FAILURE', 'same-arm prepared cache verification'
        prepared = None if precompile else c.verify_prepared(args.prepared)
        c.require(precompile or Path(prepared['cache']).resolve() == cache, 'Prepared cache path differs')
        expected = None if precompile else prepared['compiler_keys']
        result['cache_inventory_before'] = c.cache_inventory(cache)
        temp = out / 'temp'; temp.mkdir()
        for key in ('TEMP','TMP','TMPDIR','SYCL_CACHE_DIR','TORCH_EXTENSIONS_DIR','XDG_CACHE_HOME'):
            os.environ[key] = str(temp)
        for key in ('TRITON_DUMP_DIR','TRITON_OVERRIDE_DIR','TRITON_HOME','TORCHINDUCTOR_CACHE_DIR'):
            os.environ[key] = str(temp / key.lower())
        os.environ['TRITON_CACHE_DIR'] = str(cache)
        progress('runtime_bootstrap_begin')
        failure_category, failure_operation = 'SOURCE_FAILURE', 'pinned runtime and source bootstrap'
        sys.path.insert(0, str(c.RUNTIME))
        runtime_environment = importlib.import_module('runtime_environment')
        resources.append(runtime_environment.isolate())
        bootstrap = importlib.import_module('fast_cached_runtime_v1')
        bootstrap.CACHE = cache; bootstrap.bootstrap()
        overlay = c.import_file('_phase2_overlay', c.source_root() / 'experimental/fp8_unround_overlay/bootstrap.py')
        result['overlay'] = overlay.activate(c.RUNTIME, '')
        import torch as actual_torch
        import numpy as np
        torch = actual_torch
        c.require(Path(torch.__file__).resolve().is_relative_to(c.RUNTIME/'python')
                  and Path(np.__file__).resolve().is_relative_to(c.RUNTIME/'python'), 'Foreign tensor/ComfyUI environment')
        torch.set_num_threads(14); torch.set_num_interop_threads(1)
        result['GPU_executed'] = True
        c.require(torch.xpu.is_available(), 'XPU unavailable')
        torch.xpu.set_device(0)
        result['device'] = dict(name=torch.xpu.get_device_name(0), index=0,
            properties=str(torch.xpu.get_device_properties(0)), torch_version=torch.__version__, numpy_version=np.__version__)
        c.require('B580' in result['device']['name'].upper(), 'Requires actual B580')
        from nr_game_fullsize import FullsizeGameModes
        from nr_backend.controlled_temporal import NRControls
        from triton.runtime import driver
        # The frozen offline model accepts this exact class. No LiveControls
        # class exists in this source tree; no game/frontend API is imported.
        controls = NRControls(**c.selected_controls())
        c.require(c.selected_controls()['intensity'] >= 1.0,
                  'P17 low strength requires real adapter validation/native harness; never fabricate a cold token')
        result['selected_controls'] = c.selected_controls()
        result['controls_constructor'] = 'nr_backend.controlled_temporal.NRControls'
        result['actual_target'] = json_value(driver.active.get_current_target())
        failure_category, failure_operation = 'CACHE_FAILURE', 'exact compiler cache policy'
        with CachePolicy(precompile=precompile, bootstrap=bootstrap, expected_keys=expected,
                         source_tree=c.source_root(), progress=progress) as policy:
            with torch.inference_mode(False):
                failure_category, failure_operation = 'ADMISSION_FAILURE', 'actual complete model construction'
                modes = FullsizeGameModes(c.RUNTIME/'exact', c.RUNTIME/'data/product-v1/profile-v1.json',
                    fixture['current_model']['profile_sha256'], **result['mode_options'])
                resources.append(modes)
                modes.select(720, source=(720,1280), variant='unrounded')
                session = modes.session; graph = session._stack.graph; suite = modes.numeric_cleanup_calls
                resources.extend((session, graph, suite))
                owners = dict(modes.implementation_calls_720)
                root_children = {stage: dict(owner.children) for stage, owner in owners.items()}
                numeric_children = {} if suite is None else dict(suite.children)
                c.require(tuple(session.fullsize_geometry) == (720,1280)
                          and tuple(session.fullsize_padding['padding']) == (768,1280)
                          and type(graph).__module__ == 'graph_front_v6' and len(session._stack.model.vit) == 8,
                          'Wrong actual720/L8/V6 model')
                if suite is not None:
                    failure_category, failure_operation = 'COMPILE_FAILURE', 'actual numeric/root compiler preflight'
                    suite.preflight()
                for owner in owners.values():
                    owner.preflight()
            with torch.inference_mode():
                failure_category, failure_operation = 'ADMISSION_FAILURE', 'fixed13 actual modes dispatch and graph admission'
                for ordinal, item in enumerate(fixture['frames']):
                    frame = process_one(torch, np, modes, controls, item, out, cycle='fixed13', ordinal=ordinal, persist=True)
                    result['frames'].append(frame)
                    if ordinal < 2:
                        ev.check_capture(frame, result['mode_options'])
                    progress('fixed_frame_done', ordinal=ordinal, route=frame['route'])
                c.require([f['route'] for f in result['frames']] == ['capture']*2 + ['replay']*11
                          and len(graph.entries) == 2 and graph.replays == 13, 'Fixed13 actual graph route/counter contract failed')
                result['fixed13_evidence'] = ev.owner_snapshot(modes, suite)
                failure_category, failure_operation = 'NUMERIC_FAILURE', 'warm repeated13 history/reset and finite/hash proof'
                warmup_receipts, warm_cycles = [], []
                # Repeat the SAME actual13 sequence. Only reset=True frame0
                # restarts private history/seed; no field mutation, synthetic
                # motion, graph-off fallback, or duplicated same-frame timing.
                for cycle in range(c.workload()['warmup_cycles'] + c.workload()['measured_cycles']):
                    measure = cycle >= c.workload()['warmup_cycles']
                    cycle_dir = out / ('warm-%02d' % (cycle - c.workload()['warmup_cycles']))
                    frames = []
                    for ordinal, item in enumerate(fixture['frames']):
                        frame = process_one(torch, np, modes, controls, item, cycle_dir,
                            cycle=cycle, ordinal=ordinal, persist=False, expected_frame=result['frames'][ordinal])
                        c.require(frame['route'] == 'replay' and len(graph.entries) == 2, 'Warm sequence captured new entries')
                        # Replaying reset+same full inputs must reproduce this
                        # arm's complete outputs AND committed private history.
                        for role in ('output', 'history'):
                            if frame[role]['raw_sha256'] != result['frames'][ordinal][role]['raw_sha256']:
                                result['warm_sequence_failure'] = dict(
                                    cycle=cycle, ordinal=ordinal, role=role,
                                    frame={key: frame[key] for key in ('frame_id', 'reset', 'seed_before', 'seed_after',
                                        'effective_seed', 'original_rgb_raw_sha256', 'original_motion_raw_sha256', 'route')},
                                    cold_hashes={key: result['frames'][ordinal][key]['raw_sha256'] for key in ('output', 'history')},
                                    warm_hashes={key: frame[key]['raw_sha256'] for key in ('output', 'history')},
                                    diagnostics={key: frame[key].get('failure_diagnostic') for key in ('output', 'history')})
                            c.require(frame[role]['raw_sha256'] == result['frames'][ordinal][role]['raw_sha256'],
                                      'Warm sequence differs from matching cold sequence: ' + role)
                        frames.append(frame)
                    receipt = dict(cycle=cycle, frames=frames,
                                   events=c.stats([f['whole_modes_xpu_ms'] for f in frames]),
                                   sequence_sha256=c.digest_obj([{k:f[k] for k in ('frame_id','reset','seed_after',
                                       'original_rgb_raw_sha256','original_motion_raw_sha256')} for f in frames]))
                    if measure: warm_cycles.append(receipt)
                    else: warmup_receipts.append(receipt)
                    progress('warm_sequence_done', cycle=cycle, measured=measure)
                result['whole_modes_timing'] = dict(status='MEASURED_XPU_STREAM_EVENTS',
                    metric='same-stream events surrounding the full FullsizeGameModes.process call',
                    includes=['front and history preparation', 'complete graph/model work',
                              'history and seed commit', 'geometry and composite within modes.process',
                              'stream idle intervals while Python submits work'],
                    excludes=['input file loading', 'input H2D and synchronization', 'output persistence',
                              'event initialization', 'post-boundary completion wait', 'game/bridge/Present'],
                    CPU_wall_reported_separately=True, private_stage_device_times='unknown',
                    fixed13=[dict(frame_id=f['frame_id'], route=f['route'], xpu_ms=f['whole_modes_xpu_ms'],
                                  host_timing=f['host_timing']) for f in result['frames']],
                    warmup_cycles=warmup_receipts, warm_cycles=warm_cycles,
                    warm_events=c.stats([f['whole_modes_xpu_ms'] for cycle in warm_cycles for f in cycle['frames']]),
                    event_boundary_independent_of_body=True, game_FPS=None)
                result['evidence'] = ev.owner_snapshot(modes, suite)
                result['conditional_coverage'] = dict(style=c.selected_controls()['style'],
                    style1_dispatch='unexecuted' if c.selected_controls()['style'] != 1 else 'actual counters required',
                    style2_dispatch='unexecuted' if c.selected_controls()['style'] != 2 else 'actual counters required',
                    low_strength='requires separate real adapter harness',
                    per_scope_zero_dispatch_sites={stage:{name:[key for key,value in child.get('calls',{}).items() if value==0]
                        for name,child in owner['children'].items()} for stage,owner in result['evidence']['implementation'].items()},
                    scope='compilation/resources can be screened without dispatch; zero sites receive no numerical qualification')
                ev.check_numeric(result['evidence']['numeric'], result['evidence']['numeric_states'], result['numeric_options'])
                ev.check_implementation(result['evidence'], result['mode_options'])
                result['graph_metadata'] = graph.metadata()
                result['graph_route_counts'] = ev.check_graph_routes(result['frames'],warmup_receipts,warm_cycles,
                    entries=len(graph.entries),graph_replays=graph.replays,workload=c.workload())
                before_raw = ev.counters(result['evidence'])
                failure_category, failure_operation = 'ADMISSION_FAILURE', 'raw graph replay and actual counters'
                result['body_output'] = save_array(np, graph.last_entry.output, out/'body-before-raw.npy')
                if not precompile:
                    result['raw'] = raw_graph(torch, graph.last_entry, graph, policy, None, result['body_output'], np, out)
                final_history = save_array(np, session._stack.model._previous, out/'final-history.npy')
                result['final_history'] = final_history
                c.require(final_history['raw_sha256'] == result['frames'][-1]['history']['raw_sha256']
                          and session._stack.model.next_seed == 13, 'Raw replay modified committed history/seed')
                after_raw = ev.counters(ev.owner_snapshot(modes, suite))
                c.require(before_raw == after_raw, 'Raw graph replay incremented actual host dispatch counters')
                result['raw_replay_counter_guard'] = dict(counters_before=before_raw, counters_after=after_raw,
                                                         unchanged=True, graph_off_requested=False)
            failure_category, failure_operation = 'SOURCE_FAILURE', 'loaded source provenance'
            result['loaded_modules'] = loaded_sources()
            allowed = c.allowed_source_map()
            for name, row in result['loaded_modules'].items():
                path = Path(row['path']).resolve()
                if path.is_relative_to(c.source_root()):
                    c.require(allowed.get(str(path)) == row['sha256'], 'Loaded source lacks PHASE2 pin: ' + name)
            if prepared is not None:
                previous = c.read(c.checked(prepared['child']))
                c.require(previous['device'] == result['device'] and previous['actual_target'] == result['actual_target']
                          and previous['loaded_modules'] == result['loaded_modules'], 'Fresh DiskOnly runtime/loaded source provenance drift')
            failure_category, failure_operation = 'CACHE_FAILURE', 'actual compiler/DiskOnly final hits'
            c.require(policy.hits > 0 and (precompile or (policy.disk.hits > 0 and not policy.readonly_write_attempts)),
                      'Actual compiler/DiskOnly hits absent')
            result['completed'] = True
            result['status'] = 'PRECOMPILED_ARM_UNACCEPTED' if precompile else 'READONLY_ARM_CHECKS_PASSED_GAME_PENDING'
    except BaseException as exc:
        result['error'] = traceback.format_exc(); failures.append(result['error'])
        result['failure_events'].append(fp.describe(exc,category=failure_category,
            operation=failure_operation,source_root=result['source_tree']))
        result['completed'] = False; result['status'] = 'FAILED_UNACCEPTED'
    finally:
        progress('retirement_begin', completed=result['completed'])
        if failures:
            # Preserve observations before attempting close. Validation itself
            # may fail; that original error is additional evidence, not a reason
            # to erase stacks/counters/owners.
            try:
                result['failure_live_observation'] = dict(
                    graph_entries=None if graph is None else len(graph.entries),
                    graph_replays=None if graph is None else graph.replays,
                    graph_closed=None if graph is None else graph.closed,
                    numeric_child_fields={name:{k:v for k,v in vars(child).items() if k in ('calls','capture_calls','active','retired','_live')}
                                          for name,child in numeric_children.items()},
                    root_child_fields={stage:{name:{k:v for k,v in vars(child).items() if k in ('calls','capture_calls','active','retired','_retired','live')}
                                             for name,child in children.items()} for stage,children in root_children.items()})
                result['failure_loaded_modules'] = loaded_sources()
            except BaseException:
                result['failure_observation_error'] = traceback.format_exc()
        cleanup = dict(verified=False, synchronized=False, modes_closed=False, graph_closed=False,
                       session_closed=False, numeric_states={}, implementation_states={})
        result['cleanup'] = cleanup
        try:
            if torch is not None:
                torch.xpu.synchronize(); cleanup['synchronized'] = True
                if modes is not None:
                    modes.close()
                cleanup.update(modes_closed=modes is None or modes.session is None,
                    session_closed=session is None or session._closed,
                    graph_closed=graph is None or graph.closed,
                    graph_entries_remaining=0 if graph is None else len(graph.entries),
                    last_entry_released=graph is None or graph.last_entry is None,
                    suite_closed=suite is None or suite.closed,
                    model_forward_registry_remaining={} if session is None else session.__dict__.get('_numeric_model_forward_registry_720', {}))
                cleanup['numeric_states'] = {name: ev.live_state(child) for name, child in numeric_children.items()}
                cleanup['implementation_states'] = {stage: {name: ev.live_state(child, root=True) for name, child in children.items()}
                                                    for stage, children in root_children.items()}
                for row in cleanup['numeric_states'].values(): ev.check_state(row, retired=True)
                for rows in cleanup['implementation_states'].values():
                    for row in rows.values(): ev.check_state(row, retired=True)
                cleanup['implementation_owners'] = {stage: dict(active=owner.active, retired=owner.retired) for stage, owner in owners.items()}
                c.require(all(not owner.active and owner.retired for owner in owners.values()), 'Implementation owner did not retire')
                c.require(cleanup['modes_closed'] and cleanup['session_closed'] and cleanup['graph_closed'] and cleanup['suite_closed']
                          and not cleanup['graph_entries_remaining'] and cleanup['last_entry_released']
                          and not cleanup['model_forward_registry_remaining'], 'Graph/session/ExitStack retirement incomplete')
                if graph is not None:
                    c.require('_forward_front' not in graph.model.__dict__ and 'dense' not in graph.arithmetic.__dict__, 'Graph/provider alias remains')
                if 'nr_backend.triton_math' in sys.modules:
                    arithmetic = sys.modules['nr_backend.triton_math']
                    c.require(getattr(arithmetic.fused_dot,'__self__',None) is None and
                              getattr(arithmetic.fused_batched_dot,'__self__',None) is None, 'Global arithmetic provider leak')
                result['loaded_gpu_libraries'] = {name:c.record(path) for name,path in runtime_environment.loaded_libraries().items()}
                pinned = {r['path']:r['sha256'] for r in c.runtime_pins()['runtime_assets']}
                c.require('sycl9.dll' in result['loaded_gpu_libraries'], 'Loaded SYCL evidence absent')
                for row in result['loaded_gpu_libraries'].values():
                    c.require(pinned.get(row['path']) == row['sha256'], 'Loaded GPU library bytes not pinned')
                cleanup['verified'] = True
            else:
                cleanup.update(no_GPU_initialized=True, no_session_created=modes is None)
        except BaseException as exc:
            result['retirement_error'] = traceback.format_exc(); failures.append(result['retirement_error'])
            result['failure_events'].append(fp.describe(exc,category='RETIREMENT_FAILURE',
                operation='actual ExitStack/graph/child/provider retirement',source_root=result['source_tree']))
            # Keep every owner reachable; failed close is never converted to
            # a successful retirement or retried by clearing provider fields.
            _FAILED_RETIREMENTS.append(dict(modes=modes, session=session, graph=graph, suite=suite,
                owners=owners, resources=resources, torch=torch, numeric_children=numeric_children, root_children=root_children))
            result['completed'] = False; result['status'] = 'FAILED_UNACCEPTED'
        if policy is not None:
            result['cache_gate'] = policy.snapshot()
        result['cache_inventory_after'] = c.cache_inventory(cache)
        if not precompile and result.get('cache_inventory_before') != result['cache_inventory_after']:
            failures.append('DiskOnly cache inventory changed'); result['cache_inventory_error'] = failures[-1]
            result['failure_events'].append(dict(category='CACHE_FAILURE',operation='readonly final cache inventory',
                fatal=True,source_failure=False,message=failures[-1],action='STOP_RUN_NO_FURTHER_LAUNCH'))
            result['completed'] = False; result['status'] = 'FAILED_UNACCEPTED'
        result['child_wall_seconds'] = time.monotonic()-started
        result['no_game_frontend_modules_loaded'] = not bool(FRONT_MODULES & set(sys.modules))
        result['failure_stacks'] = failures
        c.write(out/'RESULT.json', result)
    if failures:
        raise RuntimeError('PHASE2 child failed; durable RESULT.json contains original execution/retirement evidence')
    return dict(status=result['status'], result=c.record(out/'RESULT.json'))
