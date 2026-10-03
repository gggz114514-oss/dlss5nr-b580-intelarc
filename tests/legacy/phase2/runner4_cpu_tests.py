"""Runner4 CPU-only integration, exact process policy and saved-array checks."""
from pathlib import Path
import argparse
import ast
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
HERE=Path(__file__).absolute().parent
sys.path.insert(0,str(HERE))
import runner_common as c
import phase2_runner as runner
import metrics
import metrics_subprocess as rpc
import metrics_streaming_reference as old
import process_policy as policy
import failure_policy as fp
from cpu_tests import npy

def exact(a,b,where='report'):
    if type(a) is float:
        assert type(b) is float and a.hex()==b.hex(),(where,a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b),where
        for key in a:exact(a[key],b[key],where+'.'+key)
    elif isinstance(a,list):
        assert len(a)==len(b),where
        for i,(x,y) in enumerate(zip(a,b)):exact(x,y,where+'.'+str(i))
    else:assert type(a) is type(b) and a==b,(where,a,b)

def trusted_row(command=None):
    command=rpc.worker_command() if command is None else command
    row=dict(Name='python.exe',ProcessId=os.getpid()+100000,CreationTime='CPU synthetic policy input',
        ExecutablePath=str(rpc.CPU_PYTHON),CommandLine=subprocess.list2cmdline(command))
    trust={str(rpc.WORKER.resolve()).casefold():c.sha(rpc.WORKER)}
    return row,trust

class Integration(unittest.TestCase):
    def setUp(self):
        base=c.DATA/'cpu/runner4-tests';base.mkdir(parents=True,exist_ok=True)
        self.temp=Path(tempfile.mkdtemp(prefix=self._testMethodName+'-',dir=c.no_reparse(base)))
    def tearDown(self):
        c.no_reparse(self.temp);assert self.temp.is_relative_to(c.DATA)
        for path in self.temp.rglob('*'):c.no_reparse(path)
        shutil.rmtree(self.temp)
    def pair(self,bad=False):
        a=npy(self.temp/'reference.npy',[0,0,0,1,1,1])
        b=npy(self.temp/'candidate.npy',[0,.25,0,1,math.inf if bad else .5,1])
        return a,b
    def test_parent_import_and_actual_runner_binding_stay_stdlib(self):
        self.assertIs(runner.compare_frames,metrics.compare_frames)
        self.assertEqual(metrics.compare_frames.__module__,'metrics_subprocess')
        for name in ('numpy','torch','triton'):self.assertNotIn(name,sys.modules)
    def test_policy_accepts_exact_bundled_script_interpreter_and_flags(self):
        row,trust=trusted_row();got=policy.classify(row,trust)
        self.assertFalse(got['blocking']);self.assertEqual(got['classification'],'exact_bundled_CPU_comparison')
        self.assertEqual(got['interpreter']['sha256'],rpc.CPU_PYTHON_SHA256)
    def test_policy_rejects_unknown_interpreter_and_missing_executable(self):
        row,trust=trusted_row()
        for executable in ('C:/unknown/python.exe','G:/never-stat-python.exe',None,''):
            wrong=dict(row,ExecutablePath=executable)
            self.assertTrue(policy.classify(wrong,trust)['blocking'])
        wrong=dict(row,CommandLine=subprocess.list2cmdline(['C:/unknown/python.exe',*rpc.worker_command()[1:]]))
        self.assertTrue(policy.classify(wrong,trust)['blocking'])
    def test_policy_rejects_cli_variants_c_m_extra_args_and_pythonw(self):
        wanted=rpc.worker_command()
        for tokens in ([wanted[0],*wanted[2:]],wanted+['--execute-gpu'],
                       [wanted[0],'-c','pass',wanted[-1]],[wanted[0],'-m','numpy',wanted[-1]],
                       [wanted[0],'-I','-X','dev','-B',wanted[-1]]):
            row,trust=trusted_row(tokens);self.assertTrue(policy.classify(row,trust)['blocking'])
        row,trust=trusted_row();self.assertTrue(policy.classify(dict(row,Name='pythonw.exe'),trust)['blocking'])
    def test_policy_rejects_missing_script_pin_and_executable_byte_drift(self):
        row,trust=trusted_row();self.assertTrue(policy.classify(row,{})['blocking'])
        self.assertTrue(policy.classify(row,{next(iter(trust)):'0'*64})['blocking'])
        original=c.sha
        with patch.object(c,'sha',side_effect=lambda p:'0'*64 if Path(p)==rpc.CPU_PYTHON else original(p)):
            self.assertTrue(policy.classify(row,trust)['blocking'])
    def test_unknown_python_and_known_gpu_remain_blocked(self):
        row,trust=trusted_row();self.assertTrue(policy.classify(dict(row,CommandLine='python.exe -c "pass"'),trust)['blocking'])
        gpu=dict(row,CommandLine=subprocess.list2cmdline([str(rpc.CPU_PYTHON),'-I','-B',str(c.SCRIPT_DIR/'phase2_child.py'),'--execute-gpu']))
        self.assertEqual(policy.classify(gpu,trust)['classification'],'known_GPU_job')
    def test_cpu_child_environment_has_one_thread_and_no_pythonpath(self):
        with patch.dict(os.environ,{'PYTHONHOME':'forbidden','PYTHONPATH':'forbidden','OMP_NUM_THREADS':'14'}):
            env=rpc.worker_environment()
        self.assertNotIn('PYTHONHOME',env);self.assertNotIn('PYTHONPATH',env)
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            self.assertEqual(env[name],'1')
    def test_same_bundled_parent_still_uses_different_cpu_child_pid(self):
        a,b=self.pair();request=dict(reference=a,candidate=b,shape=[1,2,3])
        run=subprocess.run([str(rpc.CPU_PYTHON),'-I','-X','utf8','-B',str(HERE/'runner4_cpu_probe.py')],
            input=json.dumps(request).encode(),capture_output=True,timeout=20,env=rpc.worker_environment(),
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
        self.assertEqual(run.returncode,0,run.stderr.decode());row=rpc.unique_json(run.stdout)
        exact(old.compare_pair(a,b,shape=(1,2,3)),row['result'])
        self.assertNotEqual(row['parent_pid'],row['call']['pid'])
        self.assertTrue(row['call']['child_exited']);self.assertFalse(row['parent_numpy_imported']);self.assertFalse(row['parent_torch_imported'])
        self.assertEqual(Path(row['call']['child_CPU_python']['path']),rpc.CPU_PYTHON)
    def test_actual_nonfinite_error_closes_child_and_keeps_array_bytes(self):
        a,b=self.pair(bad=True);pins=[c.record(row['path']) for row in (a,b)]
        with self.assertRaisesRegex(RuntimeError,'Nonfinite'):metrics.compare_pair(a,b,shape=(1,2,3))
        self.assertTrue(rpc.LAST_CALL['child_exited']);self.assertNotEqual(rpc.LAST_CALL['returncode'],0)
        self.assertIn('Nonfinite',rpc.LAST_CALL['original_CPU_response']['error'])
        for pin in pins:c.checked(pin)
    def test_actual_timeout_reaps_cpu_child_and_keeps_array_bytes(self):
        a,b=self.pair();pins=[c.record(row['path']) for row in (a,b)]
        with patch.object(rpc,'TIMEOUT_SECONDS',.001):
            with self.assertRaises(subprocess.TimeoutExpired):metrics.compare_pair(a,b,shape=(1,2,3))
        self.assertTrue(rpc.LAST_CALL['child_exited']);self.assertIsNotNone(rpc.LAST_CALL['returncode'])
        for pin in pins:c.checked(pin)
    def test_response_duplicate_nonfinite_and_request_guard_reject(self):
        with self.assertRaises(RuntimeError):rpc.unique_json('{"ok":true,"ok":false}')
        with self.assertRaises(ValueError):rpc.unique_json('{"ok":NaN}')
        with patch.object(rpc.subprocess,'Popen',side_effect=AssertionError('Must reject before launch')):
            with self.assertRaisesRegex(RuntimeError,'Oversized'):metrics.compare_pair({'padding':'x'*(rpc.REQUEST_LIMIT+1)},{})
            for module in ('numpy','torch','triton'):
                with patch.dict(sys.modules,{module:object()}):
                    with self.assertRaisesRegex(RuntimeError,'stdlib'):metrics.compare_pair({},{})
    def test_original_execution_numeric_role_cache_retirement_guards_are_unchanged(self):
        provenance=c.read(HERE/'BASE_PROVENANCE.json')
        for name in ('evidence.py','failure_policy.py','phase2_child.py','gpu_worker.py','worker_support.py',
                     'cache_policy.py','artifact_cleanup.py','module_sources.py'):
            self.assertEqual(c.sha(HERE/name),provenance['runner3_files'][name]['sha256'])
        seed_base=Path(provenance['runner3_files']['cache_seed.py']['path'])
        seed_expected=seed_base.read_text(encoding='utf-8').replace(
            "c.DATA.parent.joinpath('phase1')", "c.EVIDENCE_DATA.parent.joinpath('phase1')")
        self.assertEqual(ast.dump(ast.parse(seed_expected),include_attributes=False),
                         ast.dump(ast.parse((HERE/'cache_seed.py').read_text(encoding='utf-8')),include_attributes=False))
        base=Path(provenance['runner3_files']['phase2_runner.py']['path'])
        before=ast.parse(base.read_text(encoding='utf-8'));after=ast.parse((HERE/'phase2_runner.py').read_text(encoding='utf-8'))
        old_defs={n.name:ast.dump(n,include_attributes=False) for n in before.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        new_defs={n.name:ast.dump(n,include_attributes=False) for n in after.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        for name,node in old_defs.items():
            if name!='main':self.assertEqual(node,new_defs[name],name)
    def test_private_write_scope_rejects_live_and_seed_outputs(self):
        if c.SCRIPT_DIR.is_relative_to(c.ROOT/'reviews/metrics-vectorized'):
            self.assertTrue(c.DATA.is_relative_to(Path('D:/Codex-NR-Experiments/cyberpunk-opt/b580-full-implementation-v1-20261003/reviews/metrics-vectorized')))
            outside = (c.EVIDENCE_DATA/'do-not-write',)
        elif c.SCRIPT_DIR == c.ROOT/'luna/phase2-e':
            self.assertEqual(c.DATA,c.E_DATA)
            self.assertEqual(c.DATA.drive.upper(),'E:')
            outside = (c.EVIDENCE_DATA/'do-not-write',c.DATA.parent/'do-not-write')
        else:
            self.assertEqual(c.SCRIPT_DIR, c.ROOT/'luna/phase2')
            self.assertEqual(c.DATA, c.EVIDENCE_DATA)
            outside = (c.DATA.parent/'do-not-write',)
        for path in (*outside,c.SEED_ROOT/'do-not-write',c.ROOT/'integrated/do-not-write'):
            with self.assertRaises(RuntimeError):c.output_path(path)

def real_comparisons():
    source=c.read(c.checked(c.read(HERE/'BASE_PROVENANCE.json')['comparison_READY']))
    input_path=Path(source['frozen_snapshot'])/'INPUTS.json'
    inputs=c.read(c.checked(next(row for row in source['source_files'] if row['relative_path']=='INPUTS.json')))
    previous=c.read(c.checked(source['results']['benchmark']))
    rows=[]
    for selected,expected in zip(inputs['cases'],previous['cases']):
        c.checked(selected['candidate_receipt']);started=time.perf_counter()
        report=runner.compare_frames(inputs['reference_frames'],selected['candidate_frames'])
        exact(expected['vectorized_report'],report)
        receipt=copy.deepcopy(rpc.LAST_CALL);policy_input,trust=trusted_row(receipt['command'])
        policy_input['ProcessId']=receipt['pid'];classification=policy.classify(policy_input,trust)
        c.require(not classification['blocking'],'Actual CPU command was not admitted')
        rows.append(dict(case=selected['name'],frames=2,roles=['output','history'],wall_s=time.perf_counter()-started,
            complete_report_exact=True,float64_max_observed_ULP_difference=0,report=report,
            actual_CPU_call=receipt,process_classification=classification,proof='Real NPY/JSON CPU calculation only; not model qualification'))
    baseline=c.read(c.checked(inputs['baseline_receipt']))
    manifest=c.checked(c.read(HERE/'BASE_PROVENANCE.json')['target_manifest'])
    c._MANIFEST=c.read(manifest);c.MANIFEST_PATH=manifest;c.MANIFEST_SHA256=c.sha(manifest);c.ARM=baseline['arm']
    started=time.perf_counter();report=runner.comparison(baseline,baseline,dict(mathematically_preserving=True))
    vector_wall=time.perf_counter()-started;receipt=copy.deepcopy(rpc.LAST_CALL)
    started=time.perf_counter()
    with patch.object(runner,'compare_frames',old.compare_frames):
        original=runner.comparison(baseline,baseline,dict(mathematically_preserving=True))
    streaming_wall=time.perf_counter()-started;exact(original,report)
    rows.append(dict(case='actual_Main_comparison_baseline13_self',frames=13,complete_report_exact=True,
        float64_max_observed_ULP_difference=0,wall_s=vector_wall,original_streaming_wall_s=streaming_wall,
        actual_CPU_call=receipt,report=report,
        original_result_completed=baseline['completed'],model_GPU_qualified=False,
        proof='Actual Main comparison function and saved full13/warm hashes; failed historical result remains unaccepted'))
    pins=[c.record(frame[role]['path']) for frame in inputs['reference_frames'] for role in ('output','history')]
    broken=copy.deepcopy(baseline);broken['frames'][0]['history']['raw_sha256']='0'*64
    try:runner.comparison(baseline,broken,{})
    except fp.Failure as exc:
        c.require(exc.category=='NUMERIC_FAILURE' and 'Raw SHA' in str(exc),'Remote numeric failure changed category')
        failure=dict(category=exc.category,original_error=str(exc),actual_CPU_call=copy.deepcopy(rpc.LAST_CALL),
                     input_arrays_preserved=all(c.checked(pin) for pin in pins),GPU_executed=False)
    else:raise AssertionError('Bad actual history raw SHA was accepted')
    c.require('numpy' not in sys.modules and 'torch' not in sys.modules and 'triton' not in sys.modules,'Parent imported a tensor/runtime')
    return dict(input_selection=c.record(input_path),cases=rows,actual_failure=failure,
                parent_numpy_imported=False,parent_torch_imported=False,parent_triton_imported=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',required=True);args=parser.parse_args()
    stream=__import__('io').StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Integration))
    report=dict(schema='runner4-vectorized-CPU-integration-tests-v1',passed=result.wasSuccessful(),
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),details=stream.getvalue(),
        GPU_executed=False,Torch_imported='torch' in sys.modules,NumPy_imported='numpy' in sys.modules)
    if result.wasSuccessful():report['real_NPY_comparisons']=real_comparisons()
    c.write(args.report,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('details','real_NPY_comparisons')},indent=2))
    if not result.wasSuccessful():print(report['details'])
    return 0 if result.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(main())
