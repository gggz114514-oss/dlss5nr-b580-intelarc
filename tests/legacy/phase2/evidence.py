"""CPU-only checks over actual child snapshots; no synthetic provider counts."""
from __future__ import annotations
import copy
import inspect
from pathlib import Path
import runner_common as c
import failure_policy as fp

ROOT_STATES = {
    ('audit_swin_720_v1', 'Counter'): ('active', 'retired'),
    ('audit_front_decoder_post_720_v1', 'Scope'): ('active', '_retired'),
    ('audit_vit_c512_scope_720_v1', 'CompleteCounter'): ('live', 'retired'),
    ('audit_history_host_720_v1', 'HistoryAdmissionCounter'): ('active', None),
    ('audit_vit_c512_720_v1', 'NumericStageHook'): (None, None),
}

def counters(snapshot):
    """Extract actual maps, including nested history/front and kernel launches."""
    result = {}
    def walk(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                name = path + '/' + str(key)
                if key in ('validate_calls','validation_calls','preflight_calls'):
                    continue
                if (key.endswith('calls') or key.endswith('hits') or key in ('frame_sequence', 'body_graph_replays')):
                    if isinstance(child, (dict, list, int)):
                        result[name] = copy.deepcopy(child)
                elif key not in ('sources', 'preflight', 'preflights', 'resources'):
                    walk(child, name)
    walk(snapshot, '')
    return result

def live_state(child, *, root=False):
    module, name = type(child).__module__, type(child).__name__
    path = Path(inspect.getfile(type(child))).resolve()
    expected = c.allowed_source_map()
    c.require(str(path) in expected and c.sha(path) == expected[str(path)], 'Unpinned child class: ' + str(path))
    values = {key: vars(child)[key] for key in ('active', 'retired', '_retired', 'live', '_live', 'closed')
              if key in vars(child) and type(vars(child)[key]) is bool}
    if root:
        rule = ROOT_STATES.get((module, name))
        c.require(rule is not None, 'Unsupported root child state schema: ' + module + '.' + name)
        active_field, retired_field = rule
        if active_field:
            c.require(active_field in values, 'Root missing actual live field')
        if retired_field:
            c.require(retired_field in values, 'Root missing actual retired field')
    else:
        if 'active' in values:
            active_field = 'active'
        elif '_live' in values:
            active_field = '_live'
        elif 'live' in values:
            active_field = 'live'
        else:
            c.require(False, 'Numeric child lacks an explicit actual state field')
        retired_field = 'retired' if 'retired' in values else '_retired' if '_retired' in values else None
    return dict(class_module=module, class_name=name, class_source=c.record(path),
                fields=values, live_field=active_field, retired_field=retired_field,
                adapter_metadata=active_field is None)

def check_state(row, retired=False):
    values = row['fields']
    if row['live_field'] is not None:
        c.require(values.get(row['live_field']) is (not retired), 'Child live state differs from actual lifecycle')
    if row['retired_field'] is not None:
        c.require(values.get(row['retired_field']) is retired, 'Child retired flag differs from actual lifecycle')

def owner_snapshot(modes, suite):
    numeric = None if suite is None else suite.snapshot()
    owners = dict(modes.implementation_calls_720)
    roots = {stage: owner.snapshot() for stage, owner in owners.items()}
    states = {stage: {name: live_state(child, root=True) for name, child in owner.children.items()}
              for stage, owner in owners.items()}
    numerical_states = {} if suite is None else {name: live_state(child) for name, child in suite.children.items()}
    return dict(numeric=numeric, implementation=roots, implementation_states=states,
                numeric_states=numerical_states,
                provider_status=copy.deepcopy(modes.implementation_provider_status_720))

def positive_map(value, label, *, allow_empty=False):
    c.require(isinstance(value, dict) and (value or allow_empty), 'Missing actual counters: ' + label)
    c.require(all(type(v) is int and v >= 0 for v in value.values()), 'Invalid actual counters: ' + label)
    if value:
        c.require(any(v > 0 for v in value.values()), 'Zero actual dispatch counters: ' + label)

def resource_rows(value):
    if isinstance(value, dict):
        if any(k in value for k in ('spills', 'n_spills', 'binary_sha256', 'actualbinarysha256', 'actual_binary_sha256')):
            yield value
        for key, item in value.items():
            # A selection log describes tried/rejected configurations. The
            # enclosing admitted resource row owns the actually loaded binary.
            # Do not reinterpret unselected compact trial rows as executed code.
            if key != 'selection':
                yield from resource_rows(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from resource_rows(item)

@fp.classified('COMPILE_FAILURE', 'actual compiler resources/binary proof')
def check_resources(snapshot, label, *, required=True):
    rows = list(resource_rows(snapshot.get('resources', {})))
    c.require(rows or not required, 'Missing actual compiled resource evidence: ' + label)
    for row in rows:
        spill = row.get('spills', row.get('n_spills'))
        c.require(type(spill) is int and spill == 0, 'Actual spill gate failed/unknown: ' + label)
        binary = next((row[k] for k in ('binary_sha256', 'actualbinarysha256', 'actual_binary_sha256',
                                      'actualbinary_sha256') if row.get(k)), None)
        c.require(type(binary) is str and len(binary) == 64, 'Missing actual compiled binary SHA: ' + label)
    return rows

@fp.classified('NUMERIC_FAILURE', 'actual numeric child identity/dispatch proof')
def check_numeric(snapshot, states, expected):
    if snapshot is None:
        c.require(not expected['children'], 'Selected numerical suite is missing')
        return
    c.require(snapshot['identity'] == expected['suite_identity'] and set(snapshot['children']) == set(expected['children'])
              and set(snapshot['preflight']) == set(snapshot['children']), 'Numeric selection/preflight mismatch')
    for name, child in snapshot['children'].items():
        check_state(states[name])
        positive_map(child.get('calls'), name)
        captures = child.get('capture_calls')
        c.require(isinstance(captures, dict), 'Missing visible numerical capture counters: ' + name)
        if name not in ('history', 'front'):
            positive_map(captures, name + ' capture')
        else:
            c.require(all(type(v) is int and v >= 0 for v in captures.values()), 'Invalid outside-body capture map')
        c.require(child.get('preflight_complete', True) is True, 'Numerical preflight incomplete')
    history = snapshot['children'].get('history')
    if history:
        required = {'axes_fp16', 'axes_fp32', 'five_tap_debug_false', 'five_tap_debug_true'}
        if history.get('implementation', {}).get('fused_near') is True:
            required.update(('near_five_tap_debug_false', 'near_five_tap_debug_true'))
        else:
            if history['options']['history_value'] == 'fp32_all_paths':
                required.add('near_counts_fp32')
            if history['options']['history_reciprocal'] == 'native':
                required.add('near_div_rn')
        if history.get('implementation', {}).get('input_summary') is True:
            required.update(('input_fp16_color', 'input_fp32_color', 'input_fp16_motion',
                             'input_fp32_motion', 'input_flags_motion', 'input_flags_color'))
        c.require(required <= set(history['required_specializations']), 'Missing selected DEBUG/near/axes/five-tap specializations')
        c.require(history['calls'].get('fractional.total', 0) > 0, 'Frozen real-motion history dispatch absent')
        c.require(history.get('outside_body_graph', True) is True, 'Unexpected history route placement')
        if 'launch_calls_by_specialization' in history:
            positive_map(history['launch_calls_by_specialization'], 'history successful launches')

@fp.classified('ADMISSION_FAILURE', 'actual implementation owner/dispatch admission')
def check_implementation(evidence, constructor):
    specs = constructor.get('implementation_hooks_720') or []
    selected = {h['module'] for h in specs}
    seen = set()
    for stage, owner in evidence['implementation'].items():
        c.require(owner['active'] is True and owner['retired'] is False, 'Implementation owner is not live')
        expected = {h['module'] for h in specs if h.get('stage', 'before_numeric') == stage}
        c.require(set(owner['children']) == expected and set(owner['preflights']) == expected, 'Selected hook stage/preflight mismatch')
        for name, child in owner['children'].items():
            seen.add(name)
            check_state(evidence['implementation_states'][stage][name])
            if name == 'audit_history_host_720_v1':
                c.require(child['active'] is True and child['single_history_executor'] is True
                          and child['metadata_owns_sampler_thread'] is False and child['preflight_calls'] > 0,
                          'History admission is not bound to the sole real sampler')
                actual = child.get('actual_child')
                c.require(isinstance(actual, dict) and actual['active'] is True and actual['retired'] is False,
                          'History admission lost the actual selected child')
                positive_map(actual['calls'], 'admitted history')
            elif name == 'audit_vit_c512_720_v1' and 'child' in child:
                c.require(child['owner'] == 'main NumericCleanup ViT child' and child['child'] is not None,
                          'Metadata numeric hook lacks real numeric child')
                positive_map(child['child']['calls'], 'delegated ViT')
            else:
                c.require(child.get('preflight_complete') is True, 'Root preflight incomplete: ' + name)
                calls = child.get('calls')
                inactive_style = (name == 'audit_front_decoder_post_720_v1' and
                    c.selected_controls()['style'] == 0 and isinstance(calls,dict) and calls
                    and all(key.startswith(('style1:','style2:')) for key in calls)
                    and child.get('options',{}).get('native_style') is True)
                if inactive_style:
                    c.require(all(type(v) is int and v == 0 for v in calls.values()), 'Style0 unexpectedly dispatched style1/2')
                else:
                    positive_map(calls, name)
                # A scope containing exclusively outside-body front/style kernels
                # legitimately has zero capture counters. Preserve them explicitly.
                capture = child.get('capture_calls')
                c.require(isinstance(capture, dict), 'Root capture map absent')
                c.require(all(type(v) is int and v >= 0 for v in capture.values()), 'Bad root capture map')
                check_resources(child, name)
                if name == 'audit_vit_c512_720_v1':
                    positive_map(capture, 'complete ViT capture')
                    c.require(child['capture_gates'] and all(r['passed'] is True for r in child['capture_gates']),
                              'Complete ViT actual capture/resource gate failed')
    c.require(seen == selected, 'Missing implementation scope')

@fp.classified('ROLE_FAILURE', 'actual capture gates and Main replacement provider roles')
def check_capture(frame, constructor):
    gates = frame.get('combo_capture_gate')
    c.require(isinstance(gates, dict) and all(gates.get(name) is True for name in c.CAPTURE_GATES)
              and all(type(v) is bool and v for v in gates.values()), 'Actual transformed capture gate failed')
    selected = {h['module'] for h in constructor.get('implementation_hooks_720') or []}
    proof = frame.get('implementation_capture_evidence') or {}
    status = frame.get('implementation_provider_status')
    if selected:
        c.require(isinstance(status, dict) and set(status['selected_modules']) == selected
                  and status['replacement_gates_passed'] is True and set(status['captured_roles']) == set(proof),
                  'Main implementation_provider_status_720 does not match actual capture evidence')
        providers = set()
        for role, row in proof.items():
            c.require(role in gates and gates[role] is True and row['passed'] is True
                      and row['new_paths_passed'] is True and row['providers'], 'Failed replacement role')
            c.require(all(type(v) is int and v > 0 for v in row['remaining_old_sites'].values()), 'Unreplaced old sites missed capture')
            occupied = set()
            for provider in row['providers']:
                module = provider['module']
                providers.add(module)
                c.require(module in selected and provider['passed'] is True, 'Foreign/failed replacement provider')
                sites = set(provider['replaced_sites'])
                c.require(sites and not occupied & sites, 'Overlap/missing role sites')
                occupied |= sites
                positive_map(provider['new_sites'], 'replacement ' + role)
            c.require(occupied == set(row['replaced']), 'Replacement site union differs')
        c.require(providers == set(status['providers']), 'Main provider list differs from actual proofs')

@fp.classified('ADMISSION_FAILURE', 'complete child qualification receipt')
def validate_worker(result, arm, phase, expected_numeric):
    constructor = c.manifest()['arms'][arm]['constructor']
    c.require(result.get('schema') == 'b580-phase2-fullmodel-child-v1' and result.get('completed') is True
              and result.get('status') == ('PRECOMPILED_ARM_UNACCEPTED' if phase == 'precompile' else 'READONLY_ARM_CHECKS_PASSED_GAME_PENDING'),
              'Worker failed or returned wrong schema/status')
    c.require(result['arm'] == arm and result['phase'] == phase and result['mode_options'] == constructor
              and result['manifest_sha256'] == c.MANIFEST_SHA256
              and result['constructor_sha256'] == c.digest_obj(constructor)
              and result['workload'] == c.workload() and result['numeric_options'] == expected_numeric,
              'Worker identity/workload changed')
    c.require(result['GPU_executed'] is True and result['game_or_API_calls'] == 0 and result['game_acceptance'] is False
              and result['no_game_frontend_modules_loaded'] is True, 'Wrong scope or acceptance claim')
    frames = result['frames']
    c.require(len(frames) == 13 and [f['route'] for f in frames] == ['capture']*2 + ['replay']*11,
              'Fixed actual13 capture/replay semantics changed')
    fixture, _ = c.fixture()
    for i, (frame, item) in enumerate(zip(frames, fixture['frames'])):
        c.require(frame['frame_id'] == item['frame_id'] and frame['reset'] == item['reset']
                  and frame['original_rgb_raw_sha256'] == item['rgb_float_sha256']
                  and frame['original_motion_raw_sha256'] == item['motion_float_sha256']
                  and frame['seed_before'] == i and frame['seed_after'] == i + 1, 'Actual frame/history/seed sequence drift')
        for role in ('output', 'history'):
            with fp.guard('NUMERIC_FAILURE', 'saved complete output/history finite and shape'):
                c.require(frame[role]['finite'] is True and frame[role]['shape'] == list(c.FRAME_SHAPE), 'Nonfinite/wrong-sized saved result')
        if i < 2:
            check_capture(frame, constructor)
    c.require(any(f['motion_nonzero'] for f in frames[1:]), 'Frozen real motion absent')
    check_numeric(result['evidence']['numeric'], result['evidence']['numeric_states'], expected_numeric)
    check_implementation(result['evidence'], constructor)
    counts = check_graph_routes(frames, result['whole_modes_timing']['warmup_cycles'],
        result['whole_modes_timing']['warm_cycles'], entries=result['graph_route_counts']['entries'],
        graph_replays=result['graph_route_counts']['replays'], workload=result['workload'])
    c.require(result['graph_route_counts'] == counts, 'Reported frame routes/raw graph counter differ from actual trace')
    c.require(len(result['graph_metadata']) == 2 and all(row['persistent_inputs_and_output_verified_outside_pool']
              for row in result['graph_metadata']), 'Actual V6 persistent resource/admission evidence failed')
    with fp.guard('RETIREMENT_FAILURE', 'actual child/graph/provider retirement'):
        c.require(result['cleanup']['verified'] is True, 'Close/graph/provider retirement unverified')
    if phase == 'readonly':
        c.require(result['raw']['count'] == 50 and len(result['raw']['samples_ms']) == 50
                  and result['raw_replay_counter_guard']['unchanged'] is True, 'Missing raw50 actual replay proof')
        warm = result['whole_modes_timing']
        c.require(warm['status'] == 'MEASURED_XPU_STREAM_EVENTS' and len(warm['fixed13']) == 13
                  and len(warm['warm_cycles']) == c.workload()['measured_cycles'], 'Missing full modes timing')
        for cycle in warm['warm_cycles']:
            c.require(len(cycle['frames']) == 13 and all(f['route'] == 'replay' for f in cycle['frames']), 'Warm sequence/frame semantics changed')
            for i, frame in enumerate(cycle['frames']):
                c.require(frame['reset'] is (i == 0) and frame['seed_after'] == i+1
                          and frame['original_rgb_raw_sha256'] == frames[i]['original_rgb_raw_sha256']
                          and frame['original_motion_raw_sha256'] == frames[i]['original_motion_raw_sha256'], 'Warm input/history/seed sequence mismatch')
                for role in ('output','history'):
                    with fp.guard('NUMERIC_FAILURE', 'warm complete output/history finite and raw hashes'):
                        c.require(frame[role]['finite'] is True and frame[role]['raw_sha256'] == frames[i][role]['raw_sha256']
                                  and 'path' not in frame[role], 'Warm result not proved by finite/hash-only receipt')
        cache = result['cache_gate']
        with fp.guard('CACHE_FAILURE', 'fresh DiskOnly no-write exact key gate'):
            c.require(cache['readonly_write_attempts'] == 0 and cache['misses'] == [] and cache['new_candidate_kernels'] == []
                      and cache['original_disk_only_hits'] > 0, 'Fresh DiskOnly gate failed')


@fp.classified('ADMISSION_FAILURE', 'actual fixed/warm frame routes and raw graph replay counter')
def check_graph_routes(frames, warmup, measured, *, entries, graph_replays, workload):
    expected = c.expected_graph_counts(workload)
    c.require(len(frames) == workload['fixed_frames'] and
              [f['route'] for f in frames] == ['capture']*workload['captures']+['replay']*workload['replays'],
              'Fixed frame capture/replay trace differs')
    c.require(len(warmup) == workload['warmup_cycles'] and len(measured) == workload['measured_cycles'],
              'Warmup/measurement cycle count differs')
    trace = [dict(cycle='fixed13',captures=sum(f['route']=='capture' for f in frames),
                  replays=sum(f['route']=='replay' for f in frames))]
    for kind,cycles in (('warmup',warmup),('measured',measured)):
        for index,cycle in enumerate(cycles):
            routes = [f['route'] for f in cycle['frames']]
            c.require(len(routes) == workload['frames_per_cycle'] and all(route=='replay' for route in routes),
                      'Repeated sequence changed capture/replay semantics')
            trace.append(dict(cycle=kind,index=index,captures=0,replays=len(routes)))
    captures = sum(row['captures'] for row in trace); replays = sum(row['replays'] for row in trace)
    c.require(captures == expected['capture_routes'] and replays == expected['replay_routes']
              and entries == expected['entries'] and graph_replays == expected['graph_replay_counter'],
              'Actual frame routes or raw GraphFront replay counter differ')
    return dict(entries=entries,replays=graph_replays,frame_capture_routes=captures,frame_replay_routes=replays,
                expected=expected,actual_route_trace=trace,
                counter_semantics='GraphFront.forward counts replay on every process frame, including capture frames; rawbody50 is separate')
