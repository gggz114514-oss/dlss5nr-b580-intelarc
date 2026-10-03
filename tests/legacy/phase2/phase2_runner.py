"""CPU supervisor. Luna explicitly opts in; exactly one GPU child at a time."""
from __future__ import annotations
import argparse
import contextlib
import csv
import json
import math
import os
from pathlib import Path
import statistics
import shutil
import subprocess
import sys
import time
import traceback
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner_common as c
import evidence as ev
from metrics import compare_frames
from cache_seed import initialize as initialize_cache
from artifact_cleanup import cleanup as cleanup_arrays
import process_policy
import failure_policy as fp

_ACTIVE_CHILD = None
_LAUNCH_AUDIT = []

@fp.classified('PROCESS_FAILURE', 'process inventory')
def inventory():
    """OS CPU process enumeration; never initializes a GPU library/device."""
    shell = r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
    query = (
        "$ErrorActionPreference='Stop'; "
        "$items=@(Get-CimInstance Win32_Process | Select-Object Name,ProcessId,ParentProcessId,ExecutablePath,CommandLine, "
        "@{Name='CreationTime';Expression={if ($_.CreationDate) {$_.CreationDate.ToUniversalTime().ToString('o')} else {$null}}}); "
        "ConvertTo-Json -InputObject $items -Compress -Depth 3"
    )
    process = subprocess.run([shell, '-NoProfile', '-Command', query], capture_output=True,
        text=True, timeout=25, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    c.require(process.returncode == 0, 'Could not inventory CPU/game/GPU processes: ' + process.stderr)
    output = process.stdout.strip()
    rows = [] if output in ('', 'null') else json.loads(output)
    if isinstance(rows, dict): rows = [rows]
    classified = process_policy.classify_inventory(rows)
    conflicts = [row for row in classified if row['blocking']]
    result = dict(checked_unix_seconds=time.time(), all_processes=classified, conflicts=conflicts,
                  query_exit_code=process.returncode, authorization_receipt=process_policy.AUTHORIZATIONS)
    if conflicts:
        # Durable inventory includes every process and its exact classification.
        c.write(c.DATA/'process-inventories'/('%d-%s.json' % (time.time_ns(),uuid.uuid4().hex)),result)
        c.require(False, 'Serial Luna launch blocked: ' + json.dumps([dict(pid=r['ProcessId'],name=r['Name'],
                  classification=r['classification']) for r in conflicts]))
    return result

@contextlib.contextmanager
def execution_lock():
    lock_path = c.output_path(c.DATA / 'GPU_EXECUTION.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    value = dict(schema='b580-phase2-serial-lock-v1', pid=os.getpid(), token=token,
                 manifest_sha256=c.MANIFEST_SHA256, started_unix_seconds=time.time())
    with lock_path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream)
    row = c.record(lock_path)
    try:
        yield row
    finally:
        # Never delete a replacement/stale foreign lock or follow a link.
        c.require(_ACTIVE_CHILD is None or _ACTIVE_CHILD.poll() is not None,
                  'A child is still alive; retaining serial HOLD lock')
        path = c.checked(row)
        c.require(c.read(path)['token'] == token and c.read(path)['pid'] == os.getpid(), 'Execution lock ownership changed')
        path.unlink()

def child_environment(out, cache):
    env = dict(os.environ)
    for name in ('NR_PHASE1_ARM','NR_PHASE1_SEED_RECEIPT_PATH','PYTHONPATH','PYTHONHOME'):
        env.pop(name, None)
    env.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', TRITON_CACHE_DIR=str(cache))
    temp = out / 'temp'
    for name in ('TEMP','TMP','TMPDIR','SYCL_CACHE_DIR','TORCH_EXTENSIONS_DIR','XDG_CACHE_HOME'):
        env[name] = str(temp)
    for name in ('TRITON_DUMP_DIR','TRITON_OVERRIDE_DIR','TRITON_HOME','TORCHINDUCTOR_CACHE_DIR'):
        env[name] = str(temp / name.lower())
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name] = '14'
    return env

def kill_child(process):
    if process.poll() is None:
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'], capture_output=True, timeout=30,
                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        process.wait(timeout=30)

@fp.classified('CHILD_FAILURE', 'serial child launch/wait/result')
def call_child(arm, phase, out, cache, prepared, lock, timeout_seconds):
    global _ACTIVE_CHILD
    out = c.output_path(out)
    c.require(not out.exists(), 'Preserving previous child output: ' + str(out))
    out.mkdir(parents=True)
    disk = shutil.disk_usage(out)
    c.require(disk.free >= 512*1024**2, 'Insufficient D: space for fresh qualification; preserving existing evidence/cache')
    command = [str(c.RUNTIME / 'python/python.exe'), '-I', '-X', 'utf8', '-B',
        str(c.SCRIPT_DIR / 'phase2_child.py'), '--manifest', str(c.MANIFEST_PATH), '--manifest-sha256', c.MANIFEST_SHA256,
        '--arm', arm, '--tag', 'PRECOMPILE_PARENT' if phase == 'precompile' else 'PARENT_OFF',
        '--phase', phase, '--output', str(out), '--cache', str(cache),
        '--lock', lock['path'], '--lock-sha256', lock['sha256'], '--execute-gpu']
    if prepared is not None: command += ['--prepared', str(prepared)]
    receipt = dict(schema='b580-phase2-process-v1', arm=arm, phase=phase, command=command,
                   pid=None, returncode=None, timed_out=False, timeout_seconds=timeout_seconds,
                   quiescence=inventory(), lock=lock)
    with fp.guard('SOURCE_FAILURE', 'verify pinned source and READY before launch'):
        c.checked(lock); c.manifest(); c.verify_ready()
    log_path = out / 'stdout.log'
    process = None; started = time.monotonic()
    try:
        with log_path.open('xb') as log:
            process = subprocess.Popen(command, cwd=str(c.SCRIPT_DIR), env=child_environment(out, cache),
                stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            receipt['pid'] = process.pid
            _ACTIVE_CHILD = process
            launch = dict(arm=arm, phase=phase, pid=process.pid, process_path=str(out/'PROCESS.json'),
                          returncode=None, GPU_executed=None)
            _LAUNCH_AUDIT.append(launch)
            while process.poll() is None:
                elapsed = time.monotonic()-started
                if elapsed >= timeout_seconds:
                    receipt['timed_out'] = True; kill_child(process); break
                try: process.wait(timeout=min(25, timeout_seconds-elapsed))
                except subprocess.TimeoutExpired:
                    print(json.dumps(dict(event='PHASE2_WORKER_RUNNING', arm=arm, phase=phase,
                        pid=process.pid, elapsed_seconds=round(time.monotonic()-started, 1))), flush=True)
            receipt['returncode'] = process.returncode
    except BaseException:
        receipt['supervisor_error'] = traceback.format_exc()
        if process is not None:
            kill_child(process); receipt['returncode'] = process.returncode
        raise
    finally:
        receipt['wall_seconds'] = time.monotonic()-started
        receipt['stdout_log'] = c.record(log_path) if log_path.exists() else None
        receipt['worker_result'] = c.record(out/'RESULT.json') if (out/'RESULT.json').is_file() else None
        c.write(out/'PROCESS.json', receipt)
        if process is not None:
            launch['returncode'] = receipt['returncode']
        if process is not None and process.poll() is not None: _ACTIVE_CHILD = None
    refs = dict(process=c.record(out/'PROCESS.json'), result=receipt['worker_result'], stdout=receipt['stdout_log'])
    with fp.guard('CHILD_FAILURE', 'parse durable child result'):
        result = c.read(out/'RESULT.json') if receipt['worker_result'] else None
    if result is not None:
        launch['GPU_executed'] = result.get('GPU_executed')
    expected_status = 'PRECOMPILED_ARM_UNACCEPTED' if phase == 'precompile' else 'READONLY_ARM_CHECKS_PASSED_GAME_PENDING'
    if (receipt['returncode'] != 0 or receipt['timed_out'] or not isinstance(result,dict)
            or result.get('completed') is not True or result.get('status') != expected_status
            or any(result.get(key) for key in ('error','retirement_error','cache_inventory_error','failure_stacks','failure_events'))):
        raise fp.child_failure(result, operation=phase+' child', evidence=refs,
                               timed_out=receipt['timed_out'], source_root=c.source_root())
    return result, c.record(out/'PROCESS.json')

@fp.classified('CACHE_FAILURE', 'prepared compiler receipt')
def prepared_receipt(directory, arm, cache, result_path, process):
    directory = c.output_path(directory); directory.mkdir(parents=True, exist_ok=True)
    result = c.read(result_path)
    ev.validate_worker(result, arm, 'precompile', c.effective_numeric())
    keys = result['cache_gate']['actual_compiler_keys']
    c.require(keys, 'No actual compiler keys observed')
    for row in keys.values():
        for path, digest in row['metadata_files'].items(): c.checked(dict(path=path,sha256=digest))
    value = dict(schema='b580-phase2-precompiled-ready-v1', status='PRECOMPILED_READY_NOT_ACCEPTED',
        arm=arm, manifest_sha256=c.MANIFEST_SHA256, constructor_sha256=c.digest_obj(c.mode_options()),
        numeric_identity=c.effective_numeric()['identity'], source_root=str(c.source_root()), workload=c.workload(),
        child=c.record(result_path), process=process, compiler_keys=keys, cache=str(cache), cache_inventory=c.cache_inventory(cache),
        copied_acceptance=False, game_acceptance=False)
    c.write(directory/'READY.json', value)
    return value

@fp.classified('CACHE_FAILURE', 'fresh DiskOnly key set and inventory')
def readonly(arm, directory, cache, precompile, lock, timeout_seconds):
    prepared = directory / 'prepared'
    ready = prepared_receipt(prepared, arm, cache, c.checked(precompile['child']), precompile['process'])
    result, process = call_child(arm, 'readonly', directory/'readonly', cache, prepared, lock, timeout_seconds)
    ev.validate_worker(result, arm, 'readonly', c.effective_numeric())
    c.require(result['cache_gate']['actual_compiler_keys'] == ready['compiler_keys'], 'Fresh DiskOnly observed different exact key set')
    c.require(c.cache_inventory(cache) == ready['cache_inventory'], 'Readonly cache writes/inventory drift')
    return dict(arm=arm, result=c.record(directory/'readonly/RESULT.json'), process=process,
                prepared=c.record(prepared/'READY.json'))

def prepare_arm(arm, directory, cache, lock, timeout_seconds):
    before = c.cache_inventory(cache)
    c.write(directory/'INVOCATION.json', dict(arm=arm, constructor=c.mode_options(), manifest=c.record(c.MANIFEST_PATH),
        cache=str(cache), cache_inventory_before=before, cache_policy='compile only missing actual compiler keys', workload=c.workload()))
    result, process = call_child(arm, 'precompile', directory/'precompile', cache, None, lock, timeout_seconds)
    ev.validate_worker(result, arm, 'precompile', c.effective_numeric())
    after = c.cache_inventory(cache)
    with fp.guard('CACHE_FAILURE', 'precompile changes only missing exact keys'):
        c.require(all(after.get(p) == digest for p,digest in before.items()), 'Precompile changed existing validated cache bytes')
        c.require(all(row['compiler_key'] in result['cache_gate']['actual_compiler_keys'] for row in result['cache_gate']['misses']), 'Missing-key receipt mismatch')
    precompile = dict(child=c.record(directory/'precompile/RESULT.json'), process=process)
    run = readonly(arm, directory/'measurement', cache, precompile, lock, timeout_seconds)
    measured = c.read(c.checked(run['result']))
    with fp.guard('NUMERIC_FAILURE', 'precompile versus fresh DiskOnly saved complete outputs/history'):
        identical = compare_frames(result['frames'], measured['frames'])
        c.require(all(identical['aggregate'][role]['all_byte_identical'] for role in ('output','history')),
                  'Precompile vs fresh DiskOnly initial13 differs; preserve all diagnostic arrays')
    proof = dict(schema='b580-phase2-same-arm-bytes-v1',
                 same_arm_initial13_byte_identity=True, arm=arm, manifest_sha256=c.MANIFEST_SHA256,
                 precompile=precompile['child'], readonly=run['result'], full_error_check=identical)
    c.write(directory/'SAME_ARM_BYTES.json', proof)
    run['precompile'] = precompile
    run['same_arm_bytes'] = c.record(directory/'SAME_ARM_BYTES.json')
    run['new_compiler_keys'] = sorted({row['compiler_key'] for row in result['cache_gate']['misses']})
    c.write(directory/'ARM_RESULT.json', run)
    return run

def pair_timing(reference, candidate):
    left = reference['whole_modes_timing']['warm_cycles']
    right = candidate['whole_modes_timing']['warm_cycles']
    c.require(len(left) == len(right) == c.workload()['measured_cycles'], 'Unequal warm sequences')
    cycle_deltas = []
    for a,b in zip(left,right):
        c.require(a['sequence_sha256'] == b['sequence_sha256'] and len(a['frames']) == len(b['frames']) == 13,
                  'Full modes comparison sequence changed')
        cycle_deltas.append(statistics.fmean(x['whole_modes_xpu_ms']-y['whole_modes_xpu_ms'] for x,y in zip(a['frames'],b['frames'])))
    mean = statistics.fmean(cycle_deltas)
    # Screening only. The three cycle means are not a proof of independent
    # GPU trials; repeat B-C-C-B provides bracketing evidence on useful arms.
    margin = 2*statistics.stdev(cycle_deltas)/math.sqrt(len(cycle_deltas)) if len(cycle_deltas)>1 else None
    return dict(metric='full_modes_warm_sequence_mean_ms', baseline_minus_candidate_ms=mean,
                cycle_deltas_ms=cycle_deltas, screening_interval_ms=[mean-margin,mean+margin] if margin is not None else None,
                repeat_eligible=mean>0 or margin is None or abs(mean)<=margin,
                decision_basis='positive full process gain or interval includes zero; no 1ms threshold',
                quality_accepted=False, game_FPS=None)

@fp.classified('NUMERIC_FAILURE', 'same-controls baseline complete output/history comparison')
def comparison(reference, candidate, arm_info):
    c.require(reference['workload'] == candidate['workload'] and reference['manifest_sha256'] == candidate['manifest_sha256'],
              'Comparison uses different manifest/control/workload')
    cold = compare_frames(reference['frames'], candidate['frames'])
    warm = []
    for cycle_a,cycle_b in zip(reference['whole_modes_timing']['warm_cycles'], candidate['whole_modes_timing']['warm_cycles']):
        c.require(cycle_a['sequence_sha256'] == cycle_b['sequence_sha256'], 'Warm input sequences differ')
        proof = []
        for i,(a,b) in enumerate(zip(cycle_a['frames'],cycle_b['frames'])):
            for row,initial in ((a,reference['frames'][i]),(b,candidate['frames'][i])):
                for role in ('output','history'):
                    c.require(row[role]['finite'] is True and row[role]['raw_sha256'] == initial[role]['raw_sha256'],
                              'Warm hashes do not reproduce corresponding saved initial13')
            proof.append(dict(frame_id=a['frame_id'], reset=a['reset'], seed_after=a['seed_after'],
                baseline_output_raw_sha256=a['output']['raw_sha256'], candidate_output_raw_sha256=b['output']['raw_sha256'],
                baseline_history_raw_sha256=a['history']['raw_sha256'], candidate_history_raw_sha256=b['history']['raw_sha256'], finite=True))
        warm.append(dict(cycle=cycle_a['cycle'], frame_count=len(proof), per_frame_hashes=proof,
                         numerical_errors='identical to saved initial13 by each arm raw hash equality; no warm tensor dump'))
    preserving = arm_info.get('mathematically_preserving') is True
    if preserving:
        c.require(all(cold['aggregate'][role]['all_byte_identical'] for role in ('output','history')),
                  'Manifest declares mathematically preserving candidate but saved bytes differ')
    return dict(baseline_error_comparison_passed=True, fixed13=cold, warm_cycles=warm, timing=pair_timing(reference,candidate),
                byte_identity_required=preserving, numeric_differences_allowed=not preserving,
                numerical_quality_status='REQUIRES_MAIN_QUALITY_AND_USER_VISUAL_REVIEW',
                raw_body=dict(baseline=reference['raw'], candidate=candidate['raw'],
                              scope='compute/layout body; excludes front/history/geometry; not complete NR or game FPS'),
                package_timing_rule='P05/P14/P18 use full modes events; raw body alone cannot judge these packages')

def sequence(manifest, requested):
    arms = manifest['arms']
    baselines = [name for name,item in arms.items() if item['baseline']]
    if requested == 'all':
        selected = manifest.get('priority_order', list(arms)) + manifest.get('remaining_order', [])
    elif type(requested) is str and requested in ('MainPriority','priority'):
        selected = list(manifest['priority_order'])
    elif isinstance(requested, list):
        selected = [name for group in requested for name in group.split(',')]
        if len(selected) == 1 and selected[0].casefold() in ('mainpriority','priority','priority_order'):
            selected = list(manifest['priority_order'])
    else:
        selected = [requested]
    c.require(len(selected) == len(set(selected)), 'Duplicate requested arms')
    c.require(all(a in arms for a in selected), 'Unknown arm')
    # A single requested control condition brings only its matching baseline.
    needed = set()
    for name in selected:
        if arms[name]['baseline']:
            needed.add(name); continue
        control = dict(c.CONTROLS, **arms[name].get('controls',{}))
        matches = [b for b in baselines if dict(c.CONTROLS, **arms[b].get('controls',{})) == control]
        explicit = arms[name].get('matching_baseline')
        if explicit is not None:
            c.require(explicit in matches, 'Explicit baseline does not share controls')
            needed.add(explicit)
        else:
            c.require(len(matches) == 1, 'Each condition needs exactly one same-controls baseline')
            needed.add(matches[0])
    baselines = [b for b in baselines if b in needed]
    order = baselines + [name for name in selected if name not in baselines]
    return order, baselines

def failure_receipt(exc, output, arm, stage, *, partial_bracket=None):
    value = c._MANIFEST or {}
    row = fp.describe(exc, operation=stage, source_root=value.get('source_root'))
    row.update(arm=arm, stage=stage, launch_count_at_stop=len(_LAUNCH_AUDIT),
               partial_bracket=partial_bracket or [], diagnostic_cleanup_performed=False)
    reports = []
    for path in sorted(output.rglob('*')):
        if path.is_file() and (path.name in ('RESULT.json','PROCESS.json','stdout.log','COMPARISON.json',
                                            'SAME_ARM_BYTES.json','INVOCATION.json','BRACKET.json')
                               or path.name.startswith('COMPARISON-')):
            try:
                reports.append(c.record(path))
            except Exception as evidence_error:
                reports.append(dict(path=str(path), recording_error=str(evidence_error)))
    row['preserved_reports'] = reports
    return row


def execute_run(args, output, cache, manifest, order, baselines, plan):
    """One terminal exception boundary; failures never return to either loop."""
    global _LAUNCH_AUDIT
    _LAUNCH_AUDIT = []
    failures, runs, comparisons, repeats, skips = [], {}, {}, {}, {}
    arm = None; stage = 'lock/process/cache seed'; bracket_rows = []
    try:
        with execution_lock() as lock:
            inventory()
            with fp.guard('CACHE_FAILURE', 'initialize validated exact PHASE1 compiler cache'):
                seed = initialize_cache(cache, output/'CACHE_SEED.json')
                c.write(output/'CACHE_SEED_REFERENCE.json', seed)
            for arm in order:
                c.ARM = arm
                stage = 'pre-launch condition'
                if c.selected_controls()['intensity'] < 1.0:
                    skips[arm] = fp.pending_skip('LOW_STRENGTH_REQUIRES_REAL_ADAPTER_NATIVE_HARNESS', c.selected_controls())
                    c.write(output/arm/'CONDITION_PENDING.json', skips[arm])
                    continue
                c.write(output/'PROGRESS.json', dict(active_arm=arm, completed=list(runs), failures=failures), replace=True)
                stage = 'prepare/readonly/admission'
                with fp.guard('ADMISSION_FAILURE', stage):
                    run = prepare_arm(arm, output/arm, cache, lock, args.timeout_seconds)
                # An arm is completed only after all qualification and numeric
                # comparisons succeed. Partial results stay in their child dirs.
                stage = 'same-controls baseline comparison'
                if not manifest['arms'][arm]['baseline']:
                    with fp.guard('NUMERIC_FAILURE', stage):
                        baseline = manifest['arms'][arm].get('matching_baseline')
                        if baseline is None:
                            matches = [b for b in baselines if dict(c.CONTROLS,**manifest['arms'][b].get('controls',{})) == c.selected_controls()]
                            c.require(len(matches) == 1, 'Missing matching same-controls baseline')
                            baseline = matches[0]
                        c.require(baseline in runs and manifest['arms'][baseline]['baseline'], 'Matching baseline failed/missing')
                        reference = c.read(c.checked(runs[baseline]['result']))
                        candidate = c.read(c.checked(run['result']))
                        comp = comparison(reference,candidate,manifest['arms'][arm])
                        comp['matching_baseline'] = baseline
                        c.write(output/arm/'COMPARISON.json', comp)
                        comparisons[arm] = comp
                else:
                    with fp.guard('NUMERIC_FAILURE', 'baseline saved-byte self comparison'):
                        c.write(output/arm/'COMPARISON.json', dict(baseline_error_comparison_passed=True,
                            comparison_scope='baseline self comparison; same-arm complete saved-byte proof already passed',
                            fixed13=c.read(c.checked(run['same_arm_bytes']))['full_error_check']))
                stage = 'qualified array cleanup'
                cleanup_arrays(output/arm, run['precompile']['child'], run['result'],
                    qualification=run['result'], comparison=c.record(output/arm/'COMPARISON.json'),
                    same_arm_byte_proof=run['same_arm_bytes'], keep_outputs=True,
                    keep_history=manifest['arms'][arm]['baseline'])
                run['array_cleanup'] = c.record(output/arm/'CLEANUP.json')
                c.write(output/arm/'ARM_RESULT_FINAL.json',run)
                runs[arm] = run
            # Only fully qualified performance results reach this screen. A
            # negative delta is a valid result and does not stop the next arm.
            stage = 'repeat screen'
            selected = [name for name in comparisons if comparisons[name]['timing']['repeat_eligible']]
            selected.sort(key=lambda name: comparisons[name]['timing']['baseline_minus_candidate_ms'],reverse=True)
            if args.repeat_policy == 'none': selected = []
            for arm in selected[:args.max_repeat_arms]:
                baseline = comparisons[arm]['matching_baseline']
                bracket_rows = []
                stage = 'B-C-C-B child/admission'
                for ordinal, name in enumerate((baseline,arm,arm,baseline)):
                    c.ARM = name
                    run = readonly(name, output/'brackets'/arm/('%02d-%s' % (ordinal,name)),cache,
                                   runs[name]['precompile'],lock,args.timeout_seconds)
                    bracket_rows.append(run)
                a,b,cand,d = [c.read(c.checked(row['result'])) for row in bracket_rows]
                repeats[arm] = dict(order=[baseline,arm,arm,baseline], runs=bracket_rows,
                    first_candidate_vs_opening_baseline=pair_timing(a,b),
                    second_candidate_vs_closing_baseline=pair_timing(d,cand),
                    baseline_drift=pair_timing(a,d), game_FPS=None)
                c.write(output/'brackets'/arm/'BRACKET.json', repeats[arm])
                stage = 'B-C-C-B complete output/history comparison'
                for index, (ref,candidate) in enumerate(((a,b),(d,cand))):
                    comp = comparison(ref,candidate,manifest['arms'][arm])
                    c.write(output/'brackets'/arm/('COMPARISON-%d.json' % index), comp)
                stage = 'B-C-C-B regeneration and qualified cleanup'
                for index, row in enumerate(bracket_rows):
                    compared = output/'brackets'/arm/('COMPARISON-%d.json' % (0 if index<2 else 1))
                    proof_path = Path(row['result']['path']).parent.parent/'SAME_ARM_BYTES.json'
                    actual = c.read(c.checked(row['result']))
                    original = c.read(c.checked(runs[row['arm']]['result']))
                    with fp.guard('NUMERIC_FAILURE', 'bracket regeneration complete output/history hashes'):
                        c.require(all(all(actual['frames'][i][role]['raw_sha256'] == original['frames'][i][role]['raw_sha256']
                                          for role in ('output','history')) for i in range(13)), 'Bracket regeneration differs')
                    c.write(proof_path,dict(same_arm_initial13_byte_identity=True,
                                           reference=runs[row['arm']]['result'],readonly=row['result'],
                                           proof='same manifest/constructor, regenerated complete output/history raw hashes equal byte-qualified initial13'))
                    cleanup_arrays(proof_path.parent, runs[row['arm']]['precompile']['child'],row['result'],
                        qualification=row['result'],comparison=c.record(compared),same_arm_byte_proof=c.record(proof_path),
                        keep_outputs=False,keep_history=False,clean_precompile=False)
    except BaseException as exc:
        failures.append(failure_receipt(exc,output,arm,stage,partial_bracket=bracket_rows))
        c.write(output/'FAILURES.json',dict(failures=failures,run_stopped=True,
                                          further_launches_permitted=False),replace=True)
    # Final verification never launches a child and cannot replace the first
    # failure. Cache/source drift is another fatal receipt, not a successful skip.
    cache_final = None; source_final = None
    for category, operation, check in (
        ('CACHE_FAILURE','final cache inventory',lambda:c.cache_inventory(cache)),
        ('SOURCE_FAILURE','final pinned source/READY verification',c.verify_sources)):
        try:
            with fp.guard(category,operation): value = check()
            if category == 'CACHE_FAILURE': cache_final = value
            else: source_final = value
        except BaseException as exc:
            failures.append(failure_receipt(exc,output,arm,operation,partial_bracket=bracket_rows))
    final = dict(schema='b580-phase2-results-v1',
        status=('FAILED_UNACCEPTED' if failures else 'OFFLINE_MODEL_CHECKS_PASSED_WITH_PENDING_CONDITIONS_GAME_PENDING'
                if skips else 'OFFLINE_MODEL_CHECKS_PASSED_GAME_PENDING'),
        manifest=plan['manifest'], workload=plan['workload'], runs=runs, comparisons=comparisons, brackets=repeats,
        failures=failures, run_stopped=bool(failures), further_launches_permitted=False,
        failure_policy='Any exception stops this invocation; only qualified performance results and explicit pre-launch pending conditions continue',
        launch_audit=_LAUNCH_AUDIT, launch_count=len(_LAUNCH_AUDIT), successful_skips=skips,
        all_requested_arms_qualified=len(runs)==len(order), completed_arm_count=len(runs),
        GPU_executed=any(row['GPU_executed'] is True for row in _LAUNCH_AUDIT),
        GPU_launch_attempted=bool(_LAUNCH_AUDIT), GPU_execution_unknown=any(row['GPU_executed'] is None for row in _LAUNCH_AUDIT),
        supervisor_GPU_access=False, game_acceptance=False, cache_final_inventory=cache_final,
        untested_arms=[name for name in order if name not in runs],
        timing_scope='Full modes warm sequences and raw body50 reported separately. No game FPS claim.',
        separate_qualification=plan['separate_qualification'], qualification_completed=plan['qualification_completed'],
        final_source_verification=source_final)
    if failures:
        c.write(output/'FAILURES.json',dict(failures=failures,run_stopped=True,further_launches_permitted=False),replace=True)
    c.write(output/'RESULTS.json',final)
    c.write(output/'PROGRESS.json',dict(status=final['status'],completed=list(runs),failure_count=len(failures),
                                      run_stopped=bool(failures),launch_count=len(_LAUNCH_AUDIT)),replace=True)
    print(json.dumps(dict(status=final['status'],report=str(output/'RESULTS.json'),failure_count=len(failures)),indent=2))
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', help='Optional independently supplied Main manifest SHA')
    select = parser.add_mutually_exclusive_group()
    select.add_argument('--arm', help='Manifest arm name or all (serial)')
    select.add_argument('--arms', nargs='+', help='Space/comma separated manifest names, or MainPriority')
    select.add_argument('--all', action='store_true')
    select.add_argument('--priority-only', action='store_true', help='Main mandatory priority_order, serial')
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--shared-cache', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--execute-gpu', action='store_true', help='Only Luna may set this opt-in')
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    parser.add_argument('--repeat-policy', choices=('promising-or-ambiguous','none'), default='none')
    parser.add_argument('--max-repeat-arms', type=int, default=3)
    parser.add_argument('--process-authorization-receipt', help='Main stable CPU background receipt for otherwise unknown Python')
    args = parser.parse_args()
    if not args.preflight_only and not args.execute_gpu:
        parser.error('CPU-only default; Luna must explicitly pass --execute-gpu')
    value = c.read(args.manifest)
    requested = ('MainPriority' if args.priority_only else args.arms if args.arms else args.arm or 'all')
    order, baselines = sequence(value, requested)
    manifest = c.configure(args.manifest, order[0],args.manifest_sha256)
    if args.execute_gpu:
        c.require(c.verify_ready().get('candidate_GPU_HOLD') is not True,
                  'Private runner4 CPU candidate HOLD: Main must freeze its final target manifest before GPU use')
    process_policy.load_authorizations(args.process_authorization_receipt)
    output, cache = c.output_path(args.output_root), c.output_path(args.shared_cache)
    c.require(output != cache and not output.is_relative_to(cache) and not cache.is_relative_to(output), 'Output/cache paths overlap')
    c.require(not output.exists() and args.timeout_seconds > 0 and args.max_repeat_arms >= 0, 'Use fresh output-root/positive timeout')
    output.mkdir(parents=True)
    plan = dict(schema='b580-phase2-cpu-preflight-v1', status='CPU_PREFLIGHT_ONLY', GPU_executed=False,
        manifest=c.record(c.MANIFEST_PATH), sources=c.verify_sources(), workload=c.workload(),
        arm_order=order, baseline_arms=baselines, output_root=str(output), shared_cache=str(cache),
        cache_seed_root=str(c.SEED_ROOT), source_manifest_sha256=manifest['source_manifest_sha256'])
    plan['separate_qualification'] = manifest.get('separate_qualification',
        ['style1/2 and low strength controls', 'native bridge HDR', 'game Present and quality'])
    plan['qualification_completed'] = dict(style0_defaults_only=True, style1=False, style2=False, low_strength=False,
                                         native_bridge_HDR=False, game_Present=False)
    fixture, protocols = c.fixture(scan=True)
    plan['fixture_frame_count'] = len(protocols)
    plan['identities'] = {}
    for arm in order:
        c.ARM = arm
        plan['identities'][arm] = dict(constructor_sha256=c.digest_obj(c.mode_options()), numeric=c.effective_numeric())
    c.write(output/'CPU_PREFLIGHT.json', plan)
    if args.preflight_only:
        print(json.dumps(dict(status=plan['status'], report=str(output/'CPU_PREFLIGHT.json')),indent=2)); return 0
    return execute_run(args, output, cache, manifest, order, baselines, plan)

if __name__ == '__main__':
    raise SystemExit(main())
