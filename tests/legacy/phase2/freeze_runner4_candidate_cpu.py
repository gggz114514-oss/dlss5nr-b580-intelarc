"""Seal this private CPU candidate and compatible READY; no GPU execution."""
from pathlib import Path
import argparse
import ast
import copy
from datetime import datetime,timezone
import json
import shutil
import subprocess
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).absolute().parent
sys.path.insert(0,str(HERE))
import runner_common as c
import metrics_subprocess as rpc

def new_json(path,row):
    path=c.no_reparse(path);c.require(path.is_relative_to(HERE),'Source output outside private candidate')
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(row,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    return c.record(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cpu-report',required=True)
    parser.add_argument('--integration-report',required=True);args=parser.parse_args()
    cpu_pin=c.record(c.output_path(args.cpu_report));cpu=c.read(c.checked(cpu_pin))
    integration_pin=c.record(c.output_path(args.integration_report));integration=c.read(c.checked(integration_pin))
    c.require(cpu['status']=='CPU_CHECKS_PASSED_GPU_UNEXECUTED' and cpu['tests']['passed']
              and cpu['tests']['tests_run']==46 and cpu['GPU_executed'] is False,'Missing passed CPU target admission')
    c.require(integration['passed'] and integration['tests_run']==13 and integration['GPU_executed'] is False
              and integration['Torch_imported'] is False and integration['NumPy_imported'] is False,'Missing CPU integration checks')
    provenance=c.read(HERE/'BASE_PROVENANCE.json');base=c.read(HERE/'BASE_RUNNER3_READY.json')
    c.require(c.sha(HERE/'BASE_RUNNER3_READY.json')==provenance['base_runner3_READY']['sha256'],'Base snapshot changed')
    for row in provenance['runner3_files'].values():c.checked(row)
    c.checked(provenance['base_runner3_READY']);c.checked(provenance['comparison_READY'])
    c.checked(provenance['comparison_snapshot']);target=cpu['actual_manifest']['manifest']
    manifest=c.read(c.checked(target));c.require(len(manifest['source_files'])==258,'Unexpected Main source count')
    runtime=rpc.runtime_contract()
    names=sorted(p.relative_to(HERE).as_posix() for p in HERE.rglob('*') if p.is_file()
        and p.suffix in ('.py','.json','.md') and p.name!='READY.json')
    files={name:c.sha(c.no_reparse(HERE/name)) for name in names}
    for name,pin in cpu['actual_manifest']['runner_AST_scripts'].items():
        c.require(files.get(name)==pin,'Source changed since r6 CPU check: '+name)
    for name in names:
        if name.endswith('.py'):ast.parse((HERE/name).read_text(encoding='utf-8-sig'),filename=name)
    snapshot=c.no_reparse(HERE/'stage1');c.require(not snapshot.exists(),'Immutable snapshot already exists')
    snapshot.mkdir();source_files=[]
    for name,pin in files.items():
        target_file=c.no_reparse(snapshot/name);target_file.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(c.no_reparse(HERE/name),target_file)
        c.require(c.sha(target_file)==pin,'Snapshot copy changed: '+name)
        source_files.append(dict(c.record(target_file),relative_path=name))
    snapshot_pin=new_json(snapshot/'SNAPSHOT.json',dict(schema='runner4-private-CPU-snapshot-v1',
        files=source_files,source_bytes=sum(row['bytes'] for row in source_files),no_NPY_copied=True,
        do_not_edit=True,GPU_executed=False))
    value=copy.deepcopy(base)
    value.update(runner_revision=4,previous_READY=c.record(HERE/'BASE_RUNNER3_READY.json'),
        created_utc=datetime.now(timezone.utc).isoformat(),files=files,source_manifest_sha256=c.digest_obj(files),
        status='PRIVATE_RUNNER4_CPU_CANDIDATE_PASSED_GPU_UNEXECUTED',target_manifest=cpu['actual_manifest']['manifest'],
        target_manifest_revision=manifest['revision'],target_source_root=manifest['source_root'],
        target_source_manifest_sha256=manifest['source_manifest_sha256'],source_file_count=len(manifest['source_files']),
        source_python_count=cpu['actual_manifest']['source_python_AST_count'],arm_count=len(manifest['arms']),
        priority_count=len(manifest['priority_order']),cpu_checks=cpu_pin,comparison_integration_checks=integration_pin,
        candidate_GPU_HOLD=True,frozen_snapshot=str(snapshot),snapshot_manifest=snapshot_pin,source_files=source_files,
        saved_r4_result_evidence_recheck=cpu['actual_manifest'].get('saved_r4_result_evidence_recheck'),
        output_root=str(c.DATA),GPU_executed=False,Torch_executed=False,GPU_launch_authorized_for_this_CPU_worker=False,
        base_runner3_provenance=c.record(HERE/'BASE_PROVENANCE.json'),
        comparison_backend=dict(name='vectorized-cpu-subprocess',API='metrics.compare_pair/compare_frames',
            CPU_python=runtime['python'],NumPy=runtime['numpy'],worker=dict(path='cpu_comparison/metrics_cpu_worker.py',
                sha256=runtime['worker_files']['metrics_cpu_worker.py']),explicit_bundle=True,
            parent_imports_only_stdlib=True,same_exe_parent_still_subprocess=True,
            GPU_runtime_default_unchanged=True,GPU_child_comparison_dependency_added=False,
            CPU_comparison_time_counted_as_NR_gain=False),
        process_policy_addition=dict(classification='exact_bundled_CPU_comparison',
            requires=['exact script path/SHA','exact CPU exe path/SHA','exact -I -X utf8 -B command'],
            unknown_python_relaxed=False),
        numerical_compatibility=dict(complete_report_exact=True,observed_float64_max_ULP_difference=0,
            changed_values_exact=True,changed_pixels_exact=True,frame_source_reset_motion_seed_checks_preserved=True,
            full13_actual_Main_comparison_report_exact=True),
        implementation_notes=str(snapshot/'IMPLEMENTATION.md'),
        explicit_future_archive_support='Original configure accepts matching Main rN archive with supplied manifest SHA; re-run CPU checks and freeze for r7',
        test_namespace_fix='r4 contract tests are isolated CPU subprocess; r6 admission cold; import_file foreign guard unchanged',
        failure_semantics=dict(NUMERIC_FAILURE='same original stop/retain-output path',CPU_child_reaped=True,
            actual_bad_history_raw_SHA_error_preserved=True,no_NPY_mutations=True),
        pending=['Main final target manifest/READY review before GPU GO',*base['pending']])
    root_ready=new_json(HERE/'READY.json',value)
    frozen_ready=new_json(snapshot/'READY.json',value)
    # New paths and READY policy are exercised from the immutable snapshot.
    code='import runner_common as c, process_policy as p, metrics_subprocess as m, metrics; '
    code+='r=c.verify_ready(); t=p.trusted_cpu_scripts(); assert t[str(m.WORKER.resolve()).casefold()]==m.runtime_contract()["worker_files"]["metrics_cpu_worker.py"]; '
    code+='assert r["candidate_GPU_HOLD"] is True; import sys; assert not any(x in sys.modules for x in ("numpy","torch","triton")); print("FROZEN_PRIVATE_POLICY_READY_PASS")'
    command=[sys.executable,'-B','-c',code]
    run=subprocess.run(command,cwd=snapshot,capture_output=True,text=True,timeout=15,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    c.require(run.returncode==0,'Frozen READY/policy check failed: '+run.stdout+run.stderr)
    inputs=c.read(c.checked(next(row for row in c.read(c.checked(provenance['comparison_READY']))['source_files']
        if row['relative_path']=='INPUTS.json')))
    first=inputs['reference_frames'][0]['output']
    request=dict(reference=first,candidate=first,shape=[720,1280,3])
    call=subprocess.run([runtime['python']['path'],'-I','-X','utf8','-B',str(snapshot/'runner4_cpu_probe.py')],
        input=json.dumps(request).encode(),capture_output=True,timeout=20,env=rpc.worker_environment(),
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
    c.require(call.returncode==0,'Frozen CPU probe failed: '+call.stderr.decode('utf-8',errors='replace'))
    probe=rpc.unique_json(call.stdout)
    c.require(probe['result']['byte_identical'] and probe['result']['changed_pixels']==0 and probe['call']['child_exited']
        and probe['call']['pid']!=probe['parent_pid'] and not probe['parent_numpy_imported']
        and not probe['parent_torch_imported'],'Frozen real NPY delegation failed')
    for row in source_files:c.checked(row)
    for row in provenance['runner3_files'].values():c.checked(row)
    c.checked(provenance['base_runner3_READY']);c.checked(provenance['comparison_READY'])
    c.checked(target)
    result=c.write(c.DATA/'FINAL_FROZEN_SELFCHECK-runner4-stage1.json',dict(schema='runner4-final-frozen-selfcheck-v1',
        passed=True,root_READY=root_ready,frozen_READY=frozen_ready,snapshot_manifest=snapshot_pin,
        verified_source_files=len(source_files),all_SHA_verified=True,all_Python_AST_parsed=True,
        actual_frozen_bundle_parent_cpu_child=probe,CPU_only=True,GPU_executed=False,
        live_runner3_READY_and_files_unchanged=True,previous_vectorized_READY_unchanged=True,
        candidate_GPU_HOLD=True,target_manifest=target,r6_source_count=258,r6_arm_count=len(manifest['arms']),
        CPU_comparison_time_counted_as_NR_gain=False))
    print(json.dumps(dict(READY=root_ready,frozen_READY=frozen_ready,selfcheck=result,
        source_files=len(source_files),CPU_checks=46,integration_checks=13,target_manifest=target,
        GPU_executed=False),ensure_ascii=False))

if __name__=='__main__':main()
