"""Freeze a CPU-verified runner. Never imports Torch or launches a GPU child."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import runner_common as c
from process_policy import MAIN_CPU
from failure_policy import CATEGORIES, SKIP_REASONS

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cpu-report',required=True)
    parser.add_argument('--replace-ready-sha256',help='Explicit authorized runner repair; preserve the exact previous READY bytes')
    args=parser.parse_args();report=c.record(c.output_path(args.cpu_report));cpu=c.read(c.checked(report))
    c.require(cpu['status']=='CPU_CHECKS_PASSED_GPU_UNEXECUTED' and cpu['tests']['passed'] is True
              and cpu['GPU_executed'] is False and cpu['Torch_executed'] is False,'No passed CPU-only verification')
    ready_path=c.no_reparse(c.SCRIPT_DIR/'READY.json')
    previous=None;revision=1
    if ready_path.exists():
        c.require(args.replace_ready_sha256 and c.sha(ready_path)==args.replace_ready_sha256,
                  'An authorized repair needs the exact previous READY SHA')
        old=c.read(ready_path);revision=old.get('runner_revision',1)+1
        archive=c.output_path(c.DATA/'cpu/runner-ready-history'/(args.replace_ready_sha256+'-READY.json'))
        archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists():
            with archive.open('xb') as stream:stream.write(ready_path.read_bytes())
        c.require(c.sha(archive)==args.replace_ready_sha256,'Previous READY archive bytes differ')
        previous=c.record(archive)
    else:
        c.require(args.replace_ready_sha256 is None,'Cannot replace a missing READY')
    files={p.relative_to(c.SCRIPT_DIR).as_posix():c.sha(c.no_reparse(p)) for p in sorted(c.SCRIPT_DIR.rglob('*'))
           if p.is_file() and p.name!='READY.json'}
    for name,digest in cpu['actual_manifest']['runner_AST_scripts'].items():
        c.require(files.get(name)==digest,'Runner source changed after CPU verification: '+name)
    target=cpu['actual_manifest']['manifest'];manifest=c.read(c.checked(target))
    deps=[c.record(c.V4/name) for name in ('FREEZE.json','SOURCE_PINS.json','FIXTURE_MANIFEST.json','runner_common.py')]
    value=dict(schema='b580-phase2-runner-ready-v1',status='CPU_READY_FOR_LUNA_SERIAL_QUALIFICATION_GPU_UNEXECUTED',
        runner_revision=revision,previous_READY=previous,
        created_utc=datetime.now(timezone.utc).isoformat(),files=files,source_manifest_sha256=c.digest_obj(files),
        target_manifest=target,target_manifest_revision=manifest.get('revision',1),
        target_source_root=manifest['source_root'],target_source_manifest_sha256=manifest['source_manifest_sha256'],
        source_file_count=len(manifest['source_files']),source_python_count=cpu['actual_manifest']['source_python_AST_count'],
        arm_count=len(manifest['arms']),priority_count=len(manifest['priority_order']),
        source_revisions='Main phase2-complete/source or phase2-complete/revisions/rN/source; immutable archive SHA required',
        cpu_checks=report,read_only_dependencies=deps,
        main_cpu_script_pins={name:c.sha(c.ROOT/name) for name in MAIN_CPU if (c.ROOT/name).is_file()},
        workload=c.WORKLOAD,CLI=dict(core='phase2_runner.py --manifest --arm/--arms/--priority-only --output-root --shared-cache',
                                   default_order='priority_order + remaining_order, baseline first',GPU_opt_in='--execute-gpu; Luna only'),
        cache_seed_root=str(c.SEED_ROOT),cache_policy='byte-validated copy only; never mutate PHASE1; compile missing actual keys; fresh same-arm DiskOnly denies writes',
        cleanup_policy='verified exact task-owned array paths/SHA; after same-arm bytes and baseline error comparison; keep baseline13 outputs/history and candidate13 outputs',
        timing_policy='full modes fixed13 + warm2/measure3 identical13; raw graph-body50 separate; readback/persistence outside events; no game FPS',
        failure_policy=dict(any_exception='STOP_RUN_NO_FURTHER_LAUNCH',categories=sorted(CATEGORIES),
            successful_skip_reason_codes=sorted(SKIP_REASONS),
            skip_only_before_launch=True,source_failure_never_skips=True,
            failed_child_exit_zero_is_fatal=True,failed_owners_and_original_stacks_preserved=True,
            slower_qualified_performance_result_can_continue=True,
            CPU_fake_child_launch_audit_tests=8,CPU_fake_tests_are_not_GPU_qualification=True),
        repeat_policy='none by default; bounded B-C-C-B only explicitly selected after Main screen',
        numeric_identity_policy='child identity is the unchanged NumericCleanupOptions base; suite identity independently appends frozen parse_hooks/hooks_identity for selected constructor specs',
        graph_counter_policy='fixed13 routes capture2/replay11; warm2+measure3 replay65; frame replay routes76; frozen GraphFront raw counter78 includes its2 capture-frame graph replay calls; preserve both',
        saved_r4_result_evidence_recheck=cpu['actual_manifest'].get('saved_r4_result_evidence_recheck'),
        GPU_executed=False,Torch_executed=False,GPU_launch_authorized_for_this_CPU_worker=False,
        separate_qualification=manifest['separate_qualification'],
        pending=['style1/2 and low-strength actual dispatch qualification',
                 'P17 real adapter cold token/native harness', 'native bridge HDR and game Present/quality',
                 'future controls use frozen NRControls; LiveControls is absent from r1 source and no game frontend is imported',
                 'unknown Python needs Main stable PID/creation/executable/command SHA CPU authorization receipt'],
        phase1_writes=0,game_or_runtime_writes=0,output_root=str(c.DATA))
    next_ready=c.no_reparse(c.SCRIPT_DIR/'READY.json.next')
    with next_ready.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    c.require(not ready_path.exists() if previous is None else c.sha(ready_path)==args.replace_ready_sha256,
              'READY changed during authorized repair freeze')
    os.replace(next_ready,ready_path)
    print(json.dumps(dict(status=value['status'],READY=c.record(c.SCRIPT_DIR/'READY.json'),runner_files=len(files),
                         CPU_tests=cpu['tests']['tests_run'],GPU_executed=False),indent=2))

if __name__=='__main__':main()
