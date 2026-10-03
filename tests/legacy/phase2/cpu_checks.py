"""Verify Main's actual frozen manifest and runner contracts; no device import."""
from __future__ import annotations
import argparse
import ast
import builtins
import importlib
import json
import subprocess
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import runner_common as c
import evidence as ev
import phase2_runner as runner
from metrics_subprocess import runtime_contract, worker_environment, unique_json


def run_tests():
    # Contract regressions deliberately read immutable r4 receipts. Keep their
    # module cache in a separate CPU process so r6/r7 admission is truly cold;
    # runner_common.import_file's foreign-source guard remains unchanged.
    runtime=runtime_contract()
    script=c.SCRIPT_DIR/'cpu_tests.py'
    command=[runtime['python']['path'],'-I','-X','utf8','-B',str(script)]
    process=subprocess.run(command,capture_output=True,text=True,timeout=60,env=worker_environment(),
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
    lines=process.stdout.rstrip().splitlines()
    c.require(lines,'CPU regression process produced no report: '+process.stderr)
    row=unique_json(lines[-1])
    c.require(process.returncode==0 and row['passed'] and row['GPU_executed'] is False
              and row['Torch_imported'] is False,'CPU regression child failed: '+process.stdout+process.stderr)
    row.update(details='\n'.join(lines[:-1]),CPU_test_command=command,CPU_interpreter=runtime['python'],
               CPU_test_script=c.record(script),isolated_test_namespace=True)
    return row

def recheck_saved_r4():
    path=c.EVIDENCE_DATA/'attempt-02-r4-full-groups/accepted_baseline/precompile/RESULT.json'
    receipt=c.record(path);result=c.read(c.checked(receipt));previous_arm=c.ARM
    try:
        c.ARM=result['arm']
        c.require(result['manifest_sha256']==c.MANIFEST_SHA256 and result['mode_options']==c.mode_options()
                  and result['workload']==c.workload(),'Saved r4 evidence uses different source/constructor/workload')
        expected=c.effective_numeric();snapshot=result['evidence']
        c.require(result['numeric_options']['identity']==expected['identity'],'Saved child base identity differs')
        ev.check_numeric(snapshot['numeric'],snapshot['numeric_states'],expected)
        timing=result['whole_modes_timing'];observed=result['failure_live_observation']
        routes=ev.check_graph_routes(result['frames'],timing['warmup_cycles'],timing['warm_cycles'],
            entries=observed['graph_entries'],graph_replays=observed['graph_replays'],workload=result['workload'])
        return dict(result=receipt,numeric_snapshot_recheck_passed=True,graph_route_recheck_passed=True,
            child_base_identity=expected['identity'],independently_computed_suite_identity=expected['suite_identity'],
            suite_identity_source=expected['suite_identity_source'],actual_suite_identity=snapshot['numeric']['identity'],
            selected_children=expected['children'],graph_route_counts=routes,
            scope='Only the two mechanical evidence checks; original failed RESULT remains unchanged and unaccepted',
            original_worker_completed=result['completed'],full_GPU_qualification_passed=False,GPU_started=False)
    finally:
        c.ARM=previous_arm


def check_actual(path, expected_sha256):
    path=Path(path).resolve(strict=True)
    c.require(c.sha(path)==expected_sha256,'Supplied Main manifest SHA mismatch')
    value=c.read(path);source_check=c.validate_manifest(value)
    c._MANIFEST=value;c.MANIFEST_PATH=path;c.MANIFEST_SHA256=expected_sha256;c.ARM=next(iter(value['arms']))
    c.require(c.checked(value['parent_manifest'])==c.PHASE1_MANIFEST.resolve()
              and c.sha(c.PHASE1_MANIFEST)==c.PHASE1_SHA256,'PHASE1 parent SHA mismatch')
    c.require(c.no_reparse(value['cache_seed_root'])==c.SEED_ROOT.resolve(),'Wrong PHASE1 cache seed root')
    c.checked(value['composition'])
    archive=c.ROOT/('phase2-complete/PHASE2-r%d.json'%value.get('revision',1))
    c.require(c.sha(archive)==expected_sha256,'Main immutable manifest archive differs')
    source=Path(value['source_root']);sys.path.insert(0,str(source/'game'))
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        c.require(name.split('.')[0] not in ('torch','triton','numpy','nr_backend'), 'CPU tried to import device/tensor code: '+name)
        return original(name,*args,**kwargs)
    checks={}
    with patch('builtins.__import__',side_effect=guard):
        profiles=importlib.import_module('full_implementation_profiles_720_v1')
        hooks=importlib.import_module('implementation_game_profiles_720_v1')
        registry=importlib.import_module('numeric_game_profiles_720_v1')
        for name,item in value['arms'].items():
            c.ARM=name
            expected=profiles.ARMS['accepted_baseline'].mode_options()
            if item['selectors']:
                expected.update(registry.combined_mode_options(item['selectors']))
            else:
                expected['implementation_hooks_720']=[h.to_dict() for h in hooks.COMMON_ADMISSION]
            expected=json.loads(json.dumps(expected))
            c.require(expected==item['constructor'],'Exact frozen registry constructor differs: '+name)
            numeric=c.effective_numeric()
            checks[name]=dict(constructor_sha256=c.digest_obj(item['constructor']),numeric_identity=numeric['identity'],
                             numeric_suite_identity=numeric['suite_identity'],
                             selectors=item['selectors'],packages=item['packages'],controls=c.selected_controls())
        for name in ('full_implementation_profiles_720_v1','implementation_game_profiles_720_v1',
                     'implementation_hooks_720_v1','numeric_game_profiles_720_v1'):
            module=sys.modules[name];file=Path(module.__file__).resolve()
            c.require(file.is_relative_to(source) and c.allowed_source_map().get(str(file))==c.sha(file),'CPU registry loaded foreign bytes')
        fixture,protocols=c.fixture(scan=True)
        source_python=0
        for row in value['source_files']:
            p=source/row['relative_path']
            if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p));source_python+=1
        scripts={}
        for p in c.SCRIPT_DIR.glob('*.py'):
            ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p));scripts[p.name]=c.sha(p)
        root_states={}
        for (module,name),(live,retired) in ev.ROOT_STATES.items():
            path=source/'game'/(module+'.py');tree=ast.parse(path.read_text(encoding='utf-8-sig'))
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
            attributes={n.attr for n in ast.walk(cls) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='self'}
            c.require((live is None or live in attributes) and (retired is None or retired in attributes),'Root lifecycle fields changed')
            root_states[module+'.'+name]=dict(live_field=live,retired_field=retired,source=c.record(path))
        saved_recheck=recheck_saved_r4() if value.get('revision')==4 else None
    c.require('torch' not in sys.modules and 'triton' not in sys.modules,'CPU verification loaded a device library')
    return dict(manifest=c.record(path if False else c.MANIFEST_PATH),actual_inventory=source_check,
                source_python_AST_count=source_python,arm_count=len(checks),priority_count=len(value['priority_order']),
                arm_registry_checks=checks,fixture_manifest=c.record(c.V4/'FIXTURE_MANIFEST.json'),
                fixture_count=len(protocols),fixture_finite_scanned=True,nonzero_motion=any(p['motion']['nonzero'] for p in protocols),
                root_actual_lifecycle_schemas=root_states,runner_AST_scripts=scripts,
                saved_r4_result_evidence_recheck=saved_recheck,
                separate_qualification=value['separate_qualification'],GPU_executed=False,source_writes=0,
                phase1_cache_access='none in CPU preflight; readonly validated copy deferred to Luna execution')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True)
    parser.add_argument('--manifest-sha256',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args();started=time.monotonic()
    result=dict(schema='b580-phase2-cpu-checks-v1',status='FAILED_CPU_ONLY',GPU_executed=False,Torch_executed=False)
    try:
        result['tests']=run_tests()
        c.require(result['tests']['passed'],'CPU contract tests failed')
        result['actual_manifest']=check_actual(args.manifest,args.manifest_sha256)
        result['status']='CPU_CHECKS_PASSED_GPU_UNEXECUTED'
    except BaseException:
        result['error']=traceback.format_exc()
    result['wall_seconds']=time.monotonic()-started
    c.write(args.report,result)
    print(json.dumps(dict(status=result['status'],tests=result.get('tests',{}).get('tests_run'),
                         report=str(args.report),error=result.get('error')),indent=2))
    return 0 if result['status']=='CPU_CHECKS_PASSED_GPU_UNEXECUTED' else 1

if __name__=='__main__':raise SystemExit(main())
