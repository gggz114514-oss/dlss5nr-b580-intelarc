"""Bounded CPU contract tests; every fixture lives under PHASE2 D: CPU temp."""
from __future__ import annotations
import ast
import builtins
import copy
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import runner_common as c
import evidence as ev
import metrics
import cache_seed
import artifact_cleanup
import phase2_runner as supervisor
import process_policy

def json_file(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')
    return c.record(path)

def npy(path, values, *, dtype='<f4', shape=(1,2,3)):
    path.parent.mkdir(parents=True,exist_ok=True)
    text = repr(dict(descr=dtype,fortran_order=False,shape=shape))
    size = len(text)+1
    text += ' '*((64-(10+size)%64)%64)+'\n'
    raw = struct.pack('<' + {'<f2':'e','<f4':'f','<f8':'d'}[dtype]*len(values), *values)
    path.write_bytes(b'\x93NUMPY\x01\x00'+struct.pack('<H',len(text))+text.encode('latin1')+raw)
    return dict(c.record(path), raw_sha256=hashlib.sha256(raw).hexdigest(),shape=list(shape),dtype=dtype,finite=True)

class Contracts(unittest.TestCase):
    def setUp(self):
        base=c.DATA/'cpu/tests';base.mkdir(parents=True,exist_ok=True)
        self.temp=Path(tempfile.mkdtemp(prefix='case-',dir=base)).resolve()
        self.saved=(c._MANIFEST,c.MANIFEST_PATH,c.MANIFEST_SHA256,c.ARM)
    def tearDown(self):
        c._MANIFEST,c.MANIFEST_PATH,c.MANIFEST_SHA256,c.ARM=self.saved
        c.require(self.temp.is_relative_to((c.DATA/'cpu/tests').resolve()),'Unsafe test cleanup')
        for p in self.temp.rglob('*'):c.no_reparse(p)
        shutil.rmtree(c.no_reparse(self.temp))
    def manifest_fixture(self):
        source=self.temp/'source';game=source/'game';game.mkdir(parents=True)
        fields=['controlled','graph_capture_policy','combo_modes','c512_qkv_library_720','native_k8_720',
                'c128_dual_qkv_720',*c.ACCEPTED_FLAGS]
        (game/'nr_game_fullsize.py').write_text('class FullsizeGameModes:\n    def __init__(self,exact_root,profile_path,profile_sha256,*, '+
            ','.join(f+'=None' for f in fields)+'):\n        pass\n',encoding='utf-8')
        ctor=dict(controlled=True,graph_capture_policy='all',combo_modes=[480,540,720],
                  c512_qkv_library_720=True,native_k8_720=True,c128_dual_qkv_720=False,
                  **dict.fromkeys(c.ACCEPTED_FLAGS,True))
        row=dict(relative_path='game/nr_game_fullsize.py',sha256=c.sha(game/'nr_game_fullsize.py'))
        value=dict(schema=c.SCHEMA,source_root=str(source),source_files=[row],source_manifest_sha256=c.source_rows_digest([row]),
                   arms={'baseline':dict(constructor=ctor,numeric_identity=None,selectors=[],packages=[],baseline=True)},
                   parent_manifest={},cache_seed_root=str(c.SEED_ROOT),revision=1,priority_order=['baseline'],remaining_order=[])
        return value,source
    def test_source_inventory_and_main_digest(self):
        v,s=self.manifest_fixture();self.assertEqual(c.validate_manifest(v,expected_source_root=s)['source_pin_count'],1)
        v['source_manifest_sha256']=c.digest_obj(v['source_files']);c.validate_manifest(v,expected_source_root=s)
    def test_source_tamper_rejected(self):
        v,s=self.manifest_fixture();(s/'game/nr_game_fullsize.py').write_text('changed',encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError,'drift'):c.validate_manifest(v,expected_source_root=s)
    def test_extra_source_rejected(self):
        v,s=self.manifest_fixture();(s/'extra.py').write_text('x=1',encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError,'extra'):c.validate_manifest(v,expected_source_root=s)
    def test_duplicate_source_rejected(self):
        v,s=self.manifest_fixture();v['source_files'].append(dict(v['source_files'][0]))
        with self.assertRaisesRegex(RuntimeError,'Duplicate'):c.validate_manifest(v,expected_source_root=s)
    def test_manifest_digest_rejected(self):
        v,s=self.manifest_fixture();v['source_manifest_sha256']='0'*64
        with self.assertRaisesRegex(RuntimeError,'digest'):c.validate_manifest(v,expected_source_root=s)
    def test_six_flags_cannot_be_disabled(self):
        v,s=self.manifest_fixture();v['arms']['baseline']['constructor']['c64_attention_project_720']=False
        with self.assertRaisesRegex(RuntimeError,'six'):c.validate_manifest(v,expected_source_root=s)
    def test_unknown_constructor_rejected(self):
        v,s=self.manifest_fixture();v['arms']['baseline']['constructor']['scale']=2
        with self.assertRaisesRegex(RuntimeError,'constructor'):c.validate_manifest(v,expected_source_root=s)
    def test_bad_priority_rejected(self):
        v,s=self.manifest_fixture();v['remaining_order']=['baseline']
        with self.assertRaisesRegex(RuntimeError,'order'):c.validate_manifest(v,expected_source_root=s)
    def test_path_traversal_and_output_scope(self):
        for rel in ('../outside','/absolute','a/../b','C:/other','a\\b','a//b','a/./b'):
            with self.assertRaises(RuntimeError):c.relative_name(rel)
        with self.assertRaises(RuntimeError):c.output_path(c.SEED_ROOT/'changed.bin')
        with self.assertRaises(RuntimeError):c.output_path(c.SCRIPT_DIR/'wrong.npy')
    def test_reparse_rejected(self):
        with patch.object(Path,'is_symlink',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'link'):c.no_reparse(self.temp/'file')
    def test_duplicate_json_keys_rejected(self):
        path=self.temp/'dup.json';path.write_text('{"a":1,"a":2}',encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError,'Duplicate'):c.read(path)
    def test_nonfinite_json_rejected(self):
        path=self.temp/'bad.json';path.write_text('{"a":NaN}',encoding='utf-8')
        with self.assertRaises(ValueError):c.read(path)
    def test_full_output_and_private_history_metrics(self):
        ref=npy(self.temp/'r.npy',[0,0,0,1,1,1]);cand=npy(self.temp/'c.npy',[0,.25,0,1,.5,1])
        row=metrics.compare_pair(ref,cand,shape=(1,2,3))
        self.assertAlmostEqual(row['mae'],.75/6);self.assertEqual(row['max_abs_error'],.5)
        self.assertEqual(row['changed_pixels'],2);self.assertEqual(row['changed_values'],2)
        self.assertAlmostEqual(row['psnr_peak_1_db'],-10*math.log10((.25**2+.5**2)/6))
    def test_half_history_and_negative_zero_byte_identity(self):
        a=npy(self.temp/'a.npy',[0]*6,dtype='<f2');b=npy(self.temp/'b.npy',[-0.0]*6,dtype='<f2')
        row=metrics.compare_pair(a,b,shape=(1,2,3))
        self.assertEqual(row['changed_values'],0);self.assertFalse(row['byte_identical']);self.assertTrue(row['psnr_zero_error'])
    def test_saved_nonfinite_rejected(self):
        a=npy(self.temp/'a.npy',[0]*6);b=npy(self.temp/'b.npy',[0,math.inf,0,0,0,0])
        with self.assertRaisesRegex(RuntimeError,'Nonfinite'):metrics.compare_pair(a,b,shape=(1,2,3))
    def test_saved_raw_sha_receipt_drift_rejected(self):
        a=npy(self.temp/'a.npy',[0]*6);b=npy(self.temp/'b.npy',[0]*6);b['raw_sha256']='0'*64
        with self.assertRaisesRegex(RuntimeError,'Raw SHA'):metrics.compare_pair(a,b,shape=(1,2,3))
    def test_mismatched_sequences_rejected(self):
        a=npy(self.temp/'a.npy',[0]*6)
        frame=dict(frame_id=0,reset=True,original_rgb_raw_sha256='r',original_motion_raw_sha256='m',seed_after=1,output=a,history=a)
        other=dict(frame,original_motion_raw_sha256='changed')
        with self.assertRaisesRegex(RuntimeError,'sequences'):metrics.compare_frames([frame],[other],shape=(1,2,3))
    def test_capture_gate_and_status_need_real_proof(self):
        frame=dict(combo_capture_gate=dict.fromkeys(c.CAPTURE_GATES,True),implementation_capture_evidence={},
            implementation_provider_status=dict(selected_modules=['audit_history_host_720_v1'],replacement_gates_passed=True,captured_roles=[],providers=[]))
        ev.check_capture(frame,dict(implementation_hooks_720=[dict(module='audit_history_host_720_v1')]))
        frame['combo_capture_gate']['c64_all_eight']=False
        with self.assertRaises(RuntimeError):ev.check_capture(frame,{})
    def test_replacement_provider_mismatch_rejected(self):
        frame=dict(combo_capture_gate=dict.fromkeys(c.CAPTURE_GATES,True),implementation_capture_evidence={},
            implementation_provider_status=dict(selected_modules=['foreign'],replacement_gates_passed=True,captured_roles=[],providers=[]))
        with self.assertRaises(RuntimeError):ev.check_capture(frame,dict(implementation_hooks_720=[dict(module='audit_history_host_720_v1')]))
    def test_history_front_zero_capture_counters_stay_visible(self):
        history=dict(calls={'fractional.total':12},capture_calls={'fractional.total':0},preflight_complete=True,
            options=dict(history_value='fp32_fractional',history_reciprocal='table'),
            required_specializations=['axes_fp16','axes_fp32','five_tap_debug_false','five_tap_debug_true'],outside_body_graph=True)
        front=dict(calls={'front.history':12},capture_calls={'front.history':0})
        snapshot=dict(identity='id',children={'history':history,'front':front},preflight={'history':{},'front':{}})
        states={key:dict(fields=dict(_live=True),live_field='_live',retired_field=None) for key in snapshot['children']}
        ev.check_numeric(snapshot,states,dict(identity='id',suite_identity='id',children=['history','front']))
        self.assertEqual(snapshot['children']['history']['capture_calls'],{'fractional.total':0})
    def test_zero_actual_history_dispatch_rejected(self):
        with self.assertRaises(RuntimeError):ev.positive_map({'fractional.total':0},'history')
    def test_root_live_fields_and_retirement(self):
        for live,retired in (('live','retired'),('_live',None),('active','_retired')):
            row=dict(fields={live:True,**({retired:False} if retired else {})},live_field=live,retired_field=retired)
            ev.check_state(row)
            with self.assertRaises(RuntimeError):ev.check_state(row,retired=True)
            row['fields'][live]=False
            if retired:row['fields'][retired]=True
            ev.check_state(row,retired=True)
    def test_spill_or_unknown_resource_rejected(self):
        for resources in ({},{'kernel':dict(spills=1,binary_sha256='0'*64)},{'kernel':dict(spills=0)}):
            with self.assertRaises(RuntimeError):ev.check_resources(dict(resources=resources),'test')
        ev.check_resources(dict(resources={'kernel':dict(spills=0,actualbinary_sha256='0'*64)}),'test')
    def test_priority_all_and_explicit_arms(self):
        arms={n:dict(baseline=n=='b',controls={}) for n in ('b','tile','combo','detail')}
        value=dict(arms=arms,priority_order=['b','combo','detail'],remaining_order=['tile'])
        self.assertEqual(supervisor.sequence(value,'all')[0],['b','combo','detail','tile'])
        self.assertEqual(supervisor.sequence(value,['MainPriority'])[0],['b','combo','detail'])
        self.assertEqual(supervisor.sequence(value,['detail,combo'])[0],['b','detail','combo'])
    def test_same_controls_baseline_pairing(self):
        value=dict(arms={'b0':dict(baseline=True), 'b1':dict(baseline=True,controls={'style':1}),
                        'style1':dict(baseline=False,controls={'style':1})})
        self.assertEqual(supervisor.sequence(value,'style1')[0],['b1','style1'])
        value['arms'].pop('b1')
        with self.assertRaises(RuntimeError):supervisor.sequence(value,'style1')
    def test_duplicate_arm_request_rejected(self):
        value=dict(arms={'b':dict(baseline=True),'x':dict(baseline=False)})
        with self.assertRaises(RuntimeError):supervisor.sequence(value,['x,x'])
    def test_process_cpu_exact_script_vs_unknown_and_gpu(self):
        script=self.temp/'cpu.py';script.write_text('pass',encoding='utf-8')
        trusted={str(script).casefold():c.sha(script)}
        row=dict(Name='python.exe',ProcessId=os.getpid()+10000,CommandLine='python.exe -I -B "'+str(script)+'"',
                 CreationTime='stable',ExecutablePath='C:/cpu/python.exe')
        self.assertFalse(process_policy.classify(row,trusted)['blocking'])
        script.write_text('changed',encoding='utf-8');self.assertTrue(process_policy.classify(row,trusted)['blocking'])
        row['CommandLine']='python.exe -I -B "'+str(c.SCRIPT_DIR/'phase2_child.py')+'" --execute-gpu'
        self.assertEqual(process_policy.classify(row,{})['classification'],'known_GPU_job')
    def test_stable_background_authorization_and_pid_reuse(self):
        row=dict(Name='python.exe',ProcessId=os.getpid()+10000,CommandLine='python.exe -c "unknown"',
                 CreationTime='time1',ExecutablePath='C:/cpu/python.exe')
        auth=dict(pid=row['ProcessId'],creation_time='time1',executable_path=row['ExecutablePath'],
                  command_line_sha256=hashlib.sha256(row['CommandLine'].encode()).hexdigest())
        self.assertFalse(process_policy.classify(row,{},[auth])['blocking'])
        row['CreationTime']='time2';self.assertTrue(process_policy.classify(row,{},[auth])['blocking'])
    def seed_fixture(self):
        source=self.temp/'seed';directory=source/'EXACTKEY';directory.mkdir(parents=True)
        src=self.temp/'jit.py';src.write_text('pass',encoding='utf-8')
        binary=directory/'kernel.spv';binary.write_bytes(b'CPU-test fixture, never executable GPU code')
        metadata=directory/'kernel.json';metadata.write_text('{"hash":"test"}',encoding='utf-8')
        group=directory/'__grp__kernel.json';json_file(group,dict(child_paths={p.name:str(p) for p in (binary,metadata)}))
        key='a'*64;row=dict(compiler_key=key,source=c.record(src),target=dict(backend='xpu',arch=dict(name='B580')),
                          src_name='kernel',metadata_files={str(p):c.sha(p) for p in (binary,metadata)},loaded_binary_sha256=c.sha(binary))
        dest=self.temp/'cache';dest.mkdir()
        return source,dest,key,row,{str(src):c.sha(src)}
    def test_exact_cache_copy_relocates_only_group_paths(self):
        source,dest,key,row,allowed=self.seed_fixture();before=c.cache_inventory(source)
        receipt=cache_seed.copy_validated(source,dest,{key:row},allowed)
        self.assertEqual(c.cache_inventory(source),before);self.assertEqual((dest/'EXACTKEY/kernel.spv').read_bytes(),(source/'EXACTKEY/kernel.spv').read_bytes())
        self.assertFalse(receipt['validation_acceptance_inherited'])
        self.assertEqual(c.read(dest/'EXACTKEY/__grp__kernel.json')['child_paths']['kernel.spv'],str(dest/'EXACTKEY/kernel.spv'))
    def test_foreign_seed_source_and_binary_rejected(self):
        source,dest,key,row,allowed=self.seed_fixture()
        with self.assertRaises(RuntimeError):cache_seed.validate_key(key,row,source,{})
        row['loaded_binary_sha256']='0'*64
        with self.assertRaises(RuntimeError):cache_seed.validate_key(key,row,source,allowed)
    def cleanup_fixture(self):
        pre=self.temp/'pre';ro=self.temp/'ro'
        def result(directory):
            output=npy(directory/'output-00.npy',[0]*6);history=npy(directory/'history-00.npy',[0]*6,dtype='<f2')
            return dict(completed=True,cleanup=dict(verified=True),frames=[dict(output=output,history=history)])
        pa=json_file(pre/'RESULT.json',result(pre));ra=json_file(ro/'RESULT.json',result(ro))
        same=json_file(self.temp/'SAME.json',dict(same_arm_initial13_byte_identity=True))
        comp=json_file(self.temp/'COMPARISON.json',dict(baseline_error_comparison_passed=True))
        return pa,ra,same,comp
    def test_cleanup_preserves_candidate_outputs_and_reports(self):
        pa,ra,same,comp=self.cleanup_fixture()
        result=artifact_cleanup.cleanup(self.temp/'clean',pa,ra,qualification=ra,comparison=comp,same_arm_byte_proof=same)
        self.assertEqual(result['status'],'CLEANED');self.assertEqual(len(result['deleted']),3)
        self.assertTrue((self.temp/'ro/output-00.npy').exists());self.assertTrue(Path(ra['path']).exists())
        self.assertFalse((self.temp/'ro/history-00.npy').exists());self.assertGreater(result['bytes_freed'],0)
    def test_cleanup_refuses_failed_owner_without_deletion(self):
        pa,ra,same,comp=self.cleanup_fixture();value=c.read(pa['path']);value['cleanup']['verified']=False;pa=json_file(Path(pa['path']),value)
        result=artifact_cleanup.cleanup(self.temp/'clean',pa,ra,qualification=ra,comparison=comp,same_arm_byte_proof=same)
        self.assertEqual(result['status'],'CLEANUP_FAILED_PRESERVED');self.assertTrue((self.temp/'pre/output-00.npy').exists())
    def test_cleanup_all_paths_validated_before_any_delete(self):
        pa,ra,same,comp=self.cleanup_fixture();value=c.read(ra['path']);value['frames'][0]['history']['path']=str(c.SCRIPT_DIR/'wrong.npy')
        ra=json_file(Path(ra['path']),value)
        result=artifact_cleanup.cleanup(self.temp/'clean',pa,ra,qualification=ra,comparison=comp,same_arm_byte_proof=same)
        self.assertEqual(result['deleted'],[]);self.assertTrue((self.temp/'pre/output-00.npy').exists())
    def test_worker_and_cache_import_never_imports_tensor_runtime(self):
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.split('.')[0] in ('torch','triton','numpy','nr_backend'):raise AssertionError('CPU imported GPU/tensor runtime '+name)
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guarded):
            importlib.import_module('gpu_worker');importlib.import_module('cache_policy')
        self.assertNotIn('torch',sys.modules);self.assertNotIn('triton',sys.modules)
    def test_worker_warmup_cannot_persist_arrays(self):
        tree=ast.parse((c.SCRIPT_DIR/'gpu_worker.py').read_text(encoding='utf-8'))
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='process_one']
        self.assertEqual(len(calls),2)
        self.assertEqual(sorted(next(k.value.value for k in call.keywords if k.arg=='persist') for call in calls),[False,True])

    def real_r4_material(self):
        manifest=c.checked(dict(path=str(c.ROOT/'phase2-complete/PHASE2-r4.json'),
            sha256='ea32381a4bf5ffc31d47fa6f05055411d48fc5c2da7032c5dc288287de456cca'))
        c._MANIFEST=c.read(manifest);c.MANIFEST_PATH=manifest;c.MANIFEST_SHA256=c.sha(manifest);c.ARM='accepted_baseline'
        path=c.EVIDENCE_DATA/'attempt-02-r4-full-groups/accepted_baseline/precompile/RESULT.json'
        result=c.read(c.checked(c.record(path)))
        self.assertEqual(result['manifest_sha256'],c.MANIFEST_SHA256)
        return result

    def test_real_r4_numeric_suite_identity_preserves_child_base(self):
        result=self.real_r4_material();expected=c.effective_numeric();actual=result['evidence']
        self.assertEqual(expected['identity'],result['numeric_options']['identity'])
        self.assertEqual(expected['declared_identity'],expected['identity'])
        self.assertEqual(expected['suite_identity'],actual['numeric']['identity'])
        self.assertNotEqual(expected['identity'],expected['suite_identity'])
        self.assertEqual(set(expected['children']),{'branch_accum','history','front'})
        ev.check_numeric(actual['numeric'],actual['numeric_states'],expected)
        for identity in (expected['identity'],expected['identity']+':forged-suffix'):
            with self.assertRaisesRegex(RuntimeError,'selection/preflight'):
                ev.check_numeric(dict(actual['numeric'],identity=identity),actual['numeric_states'],expected)
        for key in ('children','preflight'):
            changed=dict(actual['numeric']);changed[key]=dict(changed[key]);changed[key].pop('front')
            with self.assertRaisesRegex(RuntimeError,'selection/preflight'):
                ev.check_numeric(changed,actual['numeric_states'],expected)

    def test_suite_identity_uses_selected_specs_not_observed_or_lazy_mode_hash(self):
        result=self.real_r4_material();original=c.effective_numeric()
        c._MANIFEST=copy.deepcopy(c._MANIFEST);ctor=c._MANIFEST['arms'][c.ARM]['constructor']
        ctor['implementation_lazy_assets_720']=True
        self.assertEqual(c.effective_numeric()['suite_identity'],original['suite_identity'])
        ctor['implementation_hooks_720'][0]['kwargs']['CPU_identity_only_probe']=True
        changed=c.effective_numeric()
        self.assertEqual(changed['identity'],original['identity'])
        self.assertNotEqual(changed['suite_identity'],result['evidence']['numeric']['identity'])
        ctor['implementation_hooks_720']=[]
        no_hooks=c.effective_numeric()
        self.assertEqual(no_hooks['suite_identity'],no_hooks['identity'])
        self.assertIsNone(no_hooks['suite_hooks_identity'])

    def test_real_r4_frame_replay_routes_and_raw_graph_counter_are_distinct(self):
        result=self.real_r4_material();workload=result['workload'];timing=result['whole_modes_timing']
        expected=c.expected_graph_counts(workload)
        self.assertEqual(expected,dict(entries=2,capture_routes=2,replay_routes=76,warm_frames=65,graph_replay_counter=78))
        observation=result['failure_live_observation']
        counts=ev.check_graph_routes(result['frames'],timing['warmup_cycles'],timing['warm_cycles'],
            entries=observation['graph_entries'],graph_replays=observation['graph_replays'],workload=workload)
        self.assertEqual(counts['frame_replay_routes'],76);self.assertEqual(counts['replays'],78)
        self.assertEqual(counts['frame_capture_routes'],2)
        for wrong in (76,77,79):
            with self.assertRaises(RuntimeError):
                ev.check_graph_routes(result['frames'],timing['warmup_cycles'],timing['warm_cycles'],
                    entries=2,graph_replays=wrong,workload=workload)
        changed=list(result['frames']);changed[0]=dict(changed[0],route='replay')
        with self.assertRaises(RuntimeError):
            ev.check_graph_routes(changed,timing['warmup_cycles'],timing['warm_cycles'],entries=2,graph_replays=78,workload=workload)
        with self.assertRaises(RuntimeError):
            ev.check_graph_routes(result['frames'],timing['warmup_cycles'][:1],timing['warm_cycles'],
                                 entries=2,graph_replays=78,workload=workload)

def run_tests():
    from cpu_failure_tests import LaunchAudits
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(Contracts),
                             unittest.defaultTestLoader.loadTestsFromTestCase(LaunchAudits)])
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    return dict(tests_run=result.testsRun,passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),
                details=stream.getvalue(),GPU_executed=False,Torch_imported='torch' in sys.modules)

if __name__=='__main__':
    result=run_tests();print(result['details']);print(json.dumps({k:v for k,v in result.items() if k!='details'}))
    raise SystemExit(0 if result['passed'] else 1)
