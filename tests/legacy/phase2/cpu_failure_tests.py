"""Real CPU subprocess launch audits; these fixtures never execute GPU code."""
from __future__ import annotations
from contextlib import ExitStack, contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import runner_common as c
import evidence as ev
import failure_policy as fp
import phase2_runner as runner


FAKE_CHILD = r'''
import json, os, pathlib, sys
config=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))
arm,phase,directory=sys.argv[2:5]
out=pathlib.Path(directory)
with pathlib.Path(config['audit']).open('a',encoding='utf-8') as stream:
    stream.write(json.dumps(dict(arm=arm,phase=phase,pid=os.getpid(),actual_command=sys.argv,
        executable=sys.executable,GPU_executed=False,test_only_cpu_fake_child=True))+'\n')
is_bracket='brackets' in out.parts
failed=(config.get('bad_phase')==phase and arm==config.get('bad_arm','accepted_baseline')
        and is_bracket==config.get('bad_bracket',False))
result=dict(test_only_cpu_fake_child=True,GPU_executed=False,arm=arm,phase=phase,
    completed=not failed,status=('PRECOMPILED_ARM_UNACCEPTED' if phase=='precompile'
        else 'READONLY_ARM_CHECKS_PASSED_GAME_PENDING') if not failed else 'FAILED_UNACCEPTED',
    frames=[],cache_gate=dict(actual_compiler_keys={'CPU_FAKE_NOT_A_GPU_COMPILER_KEY':{'metadata_files':{}}},misses=[]))
if failed:
    result['error']=config.get('original_stack','CPU fake child deliberate failure')
    result['failure_stacks']=[result['error']]
    if config.get('completed_with_failure'):result.update(completed=True,status='PRECOMPILED_ARM_UNACCEPTED')
if config.get('cache_mutation') and phase=='precompile':
    pathlib.Path(config['cache_mutation']).write_text('changed by CPU fake child',encoding='utf-8')
if not (failed and config.get('missing_result')):
    (out/'RESULT.json').write_text('{broken' if failed and config.get('malformed_result') else json.dumps(result),encoding='utf-8')
print(json.dumps(dict(CPU_fake_child=True,GPU_executed=False,arm=arm,phase=phase,failed=failed)),flush=True)
sys.exit(config.get('bad_exit',2) if failed else 0)
'''


class LaunchAudits(unittest.TestCase):
    def setUp(self):
        base=c.DATA/'cpu/failure-launch-audits';base.mkdir(parents=True,exist_ok=True)
        self.temp=Path(tempfile.mkdtemp(prefix=self._testMethodName+'-',dir=c.no_reparse(base))).resolve()
        self.saved=(c._MANIFEST,c.MANIFEST_PATH,c.MANIFEST_SHA256,c.ARM,runner._LAUNCH_AUDIT)
        self.index=0

    def tearDown(self):
        c._MANIFEST,c.MANIFEST_PATH,c.MANIFEST_SHA256,c.ARM,runner._LAUNCH_AUDIT=self.saved
        self.assertIsNone(runner._ACTIVE_CHILD)
        # Small CPU fake reports/audits remain as durable regression evidence.

    def run_case(self, *, config=None, gate=None, compare_failure=False, repeats=False, low_strength=False):
        self.index+=1;case=self.temp/('case-%02d'%self.index);case.mkdir()
        output,cache=case/'run',case/'cache';output.mkdir();cache.mkdir()
        fake=case/'CPU_FAKE_CHILD.py';fake.write_text(FAKE_CHILD,encoding='utf-8')
        cfg=dict(audit=str(case/'CPU_LAUNCH_AUDIT.jsonl'),**(config or {}))
        if gate=='CACHE_FAILURE':
            owned=cache/'existing-test-cache-bytes';owned.write_text('original',encoding='utf-8')
            cfg['cache_mutation']=str(owned)
        config_path=case/'CPU_FAKE_CONFIG.json';config_path.write_text(json.dumps(cfg),encoding='utf-8')
        names=['accepted_baseline','candidate_one','candidate_two']
        controls={'intensity':.5} if low_strength else {}
        manifest=dict(source_root=str(c.ROOT/'phase2-complete/source'),
            arms={name:dict(baseline=i==0,controls=controls,constructor={},numeric_identity=None)
                  for i,name in enumerate(names)})
        manifest_path=case/'CPU_FAKE_MANIFEST.json';manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
        c._MANIFEST=manifest;c.MANIFEST_PATH=manifest_path;c.MANIFEST_SHA256=c.sha(manifest_path);c.ARM=names[0]
        plan=dict(manifest=c.record(manifest_path),workload={'CPU_fake_fixture':True},
                  separate_qualification=['GPU unexecuted'],qualification_completed={'GPU_qualified':False})
        args=SimpleNamespace(timeout_seconds=10,repeat_policy='promising-or-ambiguous' if repeats else 'none',max_repeat_arms=3)
        real_popen=subprocess.Popen

        def cpu_popen(command, **kwargs):
            self.assertEqual(Path(command[5]).name,'phase2_child.py')
            self.assertIn('--execute-gpu',command)
            arm=command[command.index('--arm')+1];phase=command[command.index('--phase')+1]
            directory=command[command.index('--output')+1]
            actual=[sys.executable,'-I','-B',str(fake),str(config_path),arm,phase,directory]
            # The only executed interpreter is the current stdlib CPU Python.
            return real_popen(actual,**kwargs)

        @contextmanager
        def cpu_lock():
            path=case/'CPU_FAKE_LOCK.json'
            c.write(path,dict(pid=os.getpid(),CPU_fake_only=True))
            yield c.record(path)

        def qualify(result,*unused):
            self.assertTrue(result['test_only_cpu_fake_child']);self.assertFalse(result['GPU_executed'])
            if result['phase']!='precompile':return
            if gate=='COMPILE_FAILURE':ev.check_resources({},'CPU fake missing resource proof')
            elif gate=='ROLE_FAILURE':ev.check_capture({'combo_capture_gate':{}},{})
            elif gate=='ADMISSION_FAILURE':raise fp.Failure(gate,'CPU fake admission','Deliberate missing admission evidence')

        def same_arm(*unused):
            with fp.guard('NUMERIC_FAILURE','CPU fake saved output/history comparison'):
                if gate=='NUMERIC_FAILURE':raise RuntimeError('CPU fake finite/history mismatch')
            return dict(aggregate={role:dict(all_byte_identical=True) for role in ('output','history')},CPU_fake_only=True)

        def compare(*unused):
            if compare_failure:raise RuntimeError('CPU fake baseline quality comparison failure')
            return dict(baseline_error_comparison_passed=True,CPU_fake_only=True,
                timing=dict(repeat_eligible=repeats,baseline_minus_candidate_ms=1 if repeats else -3))

        def cleanup(directory,*unused,**options):
            c.write(directory/'CLEANUP.json',dict(status='CPU_FAKE_NO_ARRAYS_DELETED',deleted=[]))

        with ExitStack() as patches:
            for obj,name,value in (
                (runner,'execution_lock',cpu_lock),(runner,'inventory',lambda:{'CPU_fake_only':True}),
                (runner,'initialize_cache',lambda *a:{'CPU_fake_only':True}),
                (runner.subprocess,'Popen',cpu_popen),(c,'verify_ready',lambda:{'CPU_fake_only':True}),
                (c,'verify_sources',lambda:{'CPU_fake_only':True}),
                (c,'effective_numeric',lambda:{'identity':None}),
                (ev,'validate_worker',qualify),(runner,'compare_frames',same_arm),
                (runner,'comparison',compare),(runner,'cleanup_arrays',cleanup)):
                patches.enter_context(patch.object(obj,name,value))
            with redirect_stdout(io.StringIO()):
                rc=runner.execute_run(args,output,cache,manifest,names,[names[0]],plan)
        final=c.read(output/'RESULTS.json')
        audit=[json.loads(line) for line in Path(cfg['audit']).read_text().splitlines()] if Path(cfg['audit']).exists() else []
        self.assertEqual(len(audit),final['launch_count'])
        self.assertTrue(all(row['GPU_executed'] is False and row['executable']==sys.executable for row in audit))
        self.assertFalse(final['GPU_executed']);self.assertFalse(final['supervisor_GPU_access'])
        if final['failures']:
            first=final['failures'][0]
            self.assertEqual(len(audit),first['launch_count_at_stop'],'A child launched AFTER failure')
            self.assertTrue(first['fatal']);self.assertEqual(first['action'],'STOP_RUN_NO_FURTHER_LAUNCH')
            self.assertTrue(final['run_stopped']);self.assertFalse(final['further_launches_permitted'])
            self.assertEqual(final['successful_skips'],{})
        c.write(case/'CPU_ASSERTED_LAUNCH_AUDIT.json',dict(test=self._testMethodName,
            real_CPU_subprocesses=len(audit),new_launches_after_failure=0 if final['failures'] else None,
            GPU_executed=False,Torch_executed=False,production_acceptance=False,
            audit=c.record(Path(cfg['audit'])) if audit else None,result=c.record(output/'RESULTS.json')))
        return rc,final,audit,output

    def test_actual_history_source_failure_stops_after_first_child(self):
        original=c.EVIDENCE_DATA/'attempt-01-full-groups/accepted_baseline/precompile/RESULT.json'
        stack=c.read(c.checked(c.record(original)))['error']
        self.assertIn("HistoryNumericCounter720",stack);self.assertIn("_require_thread",stack)
        rc,result,audit,out=self.run_case(config=dict(bad_phase='precompile',original_stack=stack))
        self.assertEqual(rc,1);self.assertEqual(len(audit),1)
        self.assertEqual(result['failures'][0]['category'],'SOURCE_FAILURE')
        self.assertFalse((out/'candidate_one').exists())
        self.assertTrue((out/'accepted_baseline/precompile/stdout.log').exists())
        self.assertEqual(c.read(out/'accepted_baseline/precompile/RESULT.json')['error'],stack)
        self.assertTrue(result['failures'][0]['preserved_reports'])

    def test_child_exit_zero_failure_and_missing_or_malformed_result_stop(self):
        for extra in ({},{'bad_exit':0},{'bad_exit':0,'completed_with_failure':True},
                      {'missing_result':True},{'malformed_result':True}):
            with self.subTest(extra=extra):
                rc,result,audit,out=self.run_case(config=dict(bad_phase='precompile',**extra))
                self.assertEqual(rc,1);self.assertEqual(len(audit),1)
                self.assertEqual(result['failures'][0]['category'],'CHILD_FAILURE')
                self.assertFalse((out/'candidate_one').exists())
                self.assertTrue((out/'accepted_baseline/precompile/PROCESS.json').exists())

    def test_all_qualification_gates_stop_without_next_arm(self):
        for category,count in (('COMPILE_FAILURE',1),('ADMISSION_FAILURE',1),('ROLE_FAILURE',1),
                               ('CACHE_FAILURE',1),('NUMERIC_FAILURE',2)):
            with self.subTest(category=category):
                rc,result,audit,out=self.run_case(gate=category)
                self.assertEqual(rc,1);self.assertEqual(len(audit),count)
                self.assertEqual(result['failures'][0]['category'],category)
                self.assertFalse((out/'candidate_one').exists())
                self.assertFalse((out/'brackets').exists())
                self.assertFalse((out/'accepted_baseline/CLEANUP.json').exists())

    def test_readonly_child_failure_stops_before_following_arm(self):
        rc,result,audit,out=self.run_case(config=dict(bad_phase='readonly'))
        self.assertEqual(rc,1);self.assertEqual([r['phase'] for r in audit],['precompile','readonly'])
        self.assertFalse((out/'candidate_one').exists());self.assertEqual(result['runs'],{})
        self.assertTrue((out/'accepted_baseline/precompile/RESULT.json').exists())
        self.assertTrue((out/'accepted_baseline/measurement/readonly/RESULT.json').exists())

    def test_numeric_baseline_error_stops_before_next_candidate(self):
        rc,result,audit,out=self.run_case(compare_failure=True)
        self.assertEqual(rc,1);self.assertEqual(len(audit),4)
        self.assertEqual(result['failures'][0]['category'],'NUMERIC_FAILURE')
        self.assertEqual(list(result['runs']),['accepted_baseline'])
        self.assertFalse((out/'candidate_two').exists())
        self.assertFalse((out/'candidate_one/CLEANUP.json').exists())

    def test_repeat_child_failure_stops_remaining_bracket_and_repeat(self):
        rc,result,audit,out=self.run_case(repeats=True,config=dict(bad_phase='readonly',bad_bracket=True))
        self.assertEqual(rc,1);self.assertEqual(len(audit),7)
        self.assertEqual(result['failures'][0]['category'],'CHILD_FAILURE')
        self.assertFalse((out/'brackets/candidate_one/01-candidate_one').exists())
        self.assertFalse((out/'brackets/candidate_two').exists())

    def test_legally_qualified_slower_candidates_continue_without_repeats(self):
        rc,result,audit,out=self.run_case()
        self.assertEqual(rc,0);self.assertEqual(len(audit),6)
        self.assertEqual(list(result['runs']),['accepted_baseline','candidate_one','candidate_two'])
        self.assertTrue(all(row['timing']['baseline_minus_candidate_ms']<0 for row in result['comparisons'].values()))
        self.assertEqual(result['failures'],[]);self.assertFalse((out/'brackets').exists())

    def test_only_explicit_nonfailure_pending_condition_can_skip(self):
        rc,result,audit,out=self.run_case(low_strength=True)
        self.assertEqual(rc,0);self.assertEqual(audit,[]);self.assertEqual(result['failures'],[])
        self.assertEqual(len(result['successful_skips']),3)
        self.assertTrue(all(r['source_failure'] is False and r['GPU_launched'] is False
                            and r['qualification_passed'] is False for r in result['successful_skips'].values()))
        with self.assertRaises(fp.Failure):fp.pending_skip('SOURCE_FAILURE',dict(intensity=.5))
        with self.assertRaises(fp.Failure):fp.pending_skip(next(iter(fp.SKIP_REASONS)),dict(intensity=1.0))
