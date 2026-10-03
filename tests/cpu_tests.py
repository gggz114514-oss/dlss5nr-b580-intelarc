"""Meaningful portable CPU regressions; all Tensor/runtime imports are forbidden."""
from __future__ import annotations
import argparse
import ast
from dataclasses import dataclass
import hashlib
import importlib.abc
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zlib
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))

class NoDeviceImports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','triton','numpy','nr_backend'):
            raise AssertionError('CPU regression imported device/array package: '+fullname)
        return None
sys.meta_path.insert(0,NoDeviceImports())
import release_contracts as c
import portable_runtime as portable
import export_assets as exporter
import run_current as runner
import run_historical as historical
from reproduction_cache import CachePhase,compiler_value
from gpu_reproduction import dispatch_counts,retired_state,game_profiles

SCRATCH=None

def npy(values,shape=(1,4),dtype='<f4',fortran=False):
    text=repr(dict(descr=dtype,fortran_order=fortran,shape=shape))
    header=(text+' '*((-10-len(text)-1)%64)+'\n').encode('latin1')
    return b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+b''.join(struct.pack('<f',x) for x in values)

def put(root,name,data):
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);return p

def fake_bundle(root):
    """CPU binding fixture only; never executable model/kernel or GPU evidence."""
    raw=npy([1,2,3,4]);blob=zlib.compress(raw);name='data/product-v1/calibration/cpu-fixture.npyz'
    meta=dict(path='${ROOT}/'+name,sha256=hashlib.sha256(blob).hexdigest(),stored_bytes=len(blob),format='npy+zlib',
              npy_sha256=hashlib.sha256(raw).hexdigest(),shape=[1,4],dtype='<f4',raw_bytes=16,
              raw_sha256=c.npy_bytes(raw)['raw_sha256'])
    profile=dict(schema=1,profile='nr256-reviewed-v1',vit_hidden_scales=[meta]*8,
                 c512=[dict(name='cpu-fixture-'+str(i),variants={'floor16':{'scales':{'sz':meta}}}) for i in range(16)])
    data={name:blob,'templates/profile-v1.json':(json.dumps(profile)+'\n').encode(),
          'exact/model-assets/sf-v2/WEIGHTS_HT.bin':b'CPU FIXTURE; NOT MODEL WEIGHTS',
          'toolchain/triton/__init__.py':b'# CPU fixture, never imported\n',
          'toolchain/triton/backends/intel/lib/libsycl-spir64-unknown-unknown.bc':b'CPU fixture, not bitcode',
          'host-helpers/manifest.json':b'{}',
          'host-helpers/fast-cache-identity.json':json.dumps(dict(libdevice_key='old-path-token',
             libdevice_sha256=hashlib.sha256(b'CPU fixture, not bitcode').hexdigest())).encode()}
    for prefix in ('exact/model-assets/noise-sm89-v2','exact/model-assets/sigmoid-sm89-v1'):
        data[prefix+'/table.bin']=b'CPU fixture, not a native LUT'
        data[prefix+'/manifest.json']=json.dumps(dict(files={'table.bin':hashlib.sha256(data[prefix+'/table.bin']).hexdigest()})).encode()
    for rel in portable.REQUIRED_ASSETS:
        data.setdefault(rel,b'CPU fixture only')
    rows=[]
    for name,payload in data.items():
        put(root,name,payload);rows.append(dict(relative_path=name,sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload)))
    portable.write_json(root/'assets.json',dict(schema='nr-release-assets-v1',files=rows,profile_template='templates/profile-v1.json'))
    return data,profile

class CPURegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(tempfile.mkdtemp(prefix='nr-release-cpu-',dir=SCRATCH))
        # Evidence is deliberately retained; these tests never delete assets,
        # sources, caches or unknown owners. Caller chooses scratch location.
        print('CPU fixture scratch: '+str(cls.root))
    def fresh(self,name):
        p=self.root/(self._testMethodName+'-'+name);p.mkdir(parents=True);return p

    def test_actual_user_source_commands_positive_and_bad_sha(self):
        cmd=[sys.executable,'-I','-X','utf8','-B',str(c.HERE/'validate_current.py')]
        proc=subprocess.run(cmd,capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stderr+proc.stdout)
        value=json.loads(proc.stdout);self.assertTrue(value['passed']);self.assertFalse(value['GPU_executed'])
        self.assertEqual(value['numeric_identity']['base_identity'],c.NUMERIC_BASE_IDENTITY)
        profiles=c.stdlib_source_module(c.REPO/'current/runtime/game/numeric_game_profiles_720_v1.py','_release_actual_game_profiles_cpu')
        names,options,structural=game_profiles(profiles.PROFILES,profiles.combined_mode_options,c.OPTIONS)
        self.assertTrue(all(options[k] is True for k in c.FLAGS))
        self.assertEqual(portable.report_value(options['numeric_cleanup_720']),c.NUMERIC_BASE)
        self.assertTrue(structural['c128_pairwise_720']);self.assertIn('num_front_both',names)
        manifest=c.read_json(c.REPO/'evidence/2026-10-03/current-source-manifest.json')
        manifest['files'][0]['sha256']='0'*64
        p=self.fresh('manifest')/'bad.json';portable.write_json(p,manifest)
        bad=subprocess.run(cmd+['--manifest',str(p)],capture_output=True,text=True)
        self.assertNotEqual(bad.returncode,0);self.assertIn('SHA mismatch',bad.stdout)

    def test_r18_published_inventory_and_actual_numeric_identity(self):
        root=c.REPO/'experiments/2026-10-03/r18/source';path=c.HERE/'registry/r18.json'
        report,_=c.validate_sources(root,path,exact=True)
        registry=c.read_json(path)
        self.assertEqual(report['files'],len(registry['source_files']))
        self.assertFalse(any(r['relative_path'].endswith('.pyc') for r in registry['source_files']))
        self.assertTrue(registry['omitted_generated_files'])
        baseline=registry['arms']['accepted_baseline']['constructor']
        identity=c.numeric_identity(root,baseline)
        self.assertEqual(identity['base_identity'],c.NUMERIC_BASE_IDENTITY)
        self.assertNotEqual(identity['base_identity'],identity['suite_identity'])
        for alias,arm in runner.ALIASES.items():
            self.assertIn(arm,registry['arms']);self.assertEqual(runner.matching_baseline(registry,arm),'accepted_baseline')

    def test_hash_inventory_missing_extra_alias_and_digest(self):
        root=self.fresh('tree');p=put(root,'one.py',b'x=1\n')
        value=dict(source_files=[dict(relative_path='one.py',sha256=c.sha(p))])
        manifest=self.fresh('pin')/'pin.json';portable.write_json(manifest,value)
        c.validate_sources(root,manifest,exact=True)
        p.write_bytes(b'x=2\n')
        with self.assertRaisesRegex(c.ContractError,'SHA mismatch'):c.validate_sources(root,manifest,exact=True)
        p.write_bytes(b'x=1\n');put(root,'extra.py',b'pass\n')
        with self.assertRaisesRegex(c.ContractError,'extra or missing'):c.validate_sources(root,manifest,exact=True)
        bad=dict(source_files=value['source_files']+[dict(relative_path='ONE.py',sha256=c.sha(p))])
        with self.assertRaisesRegex(c.ContractError,'case-aliased'):c.source_rows(bad)
        with self.assertRaisesRegex(c.ContractError,'digest'):c.source_rows(dict(value,source_manifest_sha256='0'*64))
        value['source_files'][0]['relative_path']='missing.py'
        missing=self.fresh('missing')/'pin.json';portable.write_json(missing,value)
        with self.assertRaisesRegex(c.ContractError,'missing pinned'):c.validate_sources(root,missing)

    def test_npy_payload_shape_finite_order_rgb_contract(self):
        self.assertEqual(c.npy_bytes(npy([0,.1,.5,1]),rgb=True)['shape'],[1,4])
        for data,text in ((npy([0,1,float('nan'),1]),'nonfinite'),(npy([0,1,float('inf'),1]),'nonfinite'),
                          (npy([0,0,0,0])[:-1],'length'),(npy([0,0,0,0],fortran=True),'C order'),
                          (npy([0,0,0,0],dtype='|O'),'float32'),(npy([-1,0,0,0]),'SDR')):
            with self.assertRaisesRegex(c.ContractError,text):c.npy_bytes(data,rgb=True)
        with self.assertRaisesRegex(c.ContractError,'shape'):c.npy_bytes(npy([0,0,0,0]),shape=(720,1280,3))

    def test_path_json_control_and_baseline_omission_rejections(self):
        for name in ('../x','/x','G:/x','a\\b','./x','a//x'):
            with self.assertRaises(c.ContractError):c.relative(name)
        p=self.fresh('json')/'bad.json';p.write_text('{"x":1,"x":2}')
        with self.assertRaisesRegex(c.ContractError,'duplicate'):c.read_json(p)
        for value in ({'style':True},{'auto_mask':1},{'intensity':.5},{'local_tone':float('nan')},{'typo':1}):
            with self.assertRaises(c.ContractError):c.validate_controls(value)
        tree=ast.parse((c.REPO/'current/runtime/game/nr_game_fullsize.py').read_text('utf-8-sig'))
        value=dict(c.OPTIONS);value.pop('numeric_cleanup_720')
        with self.assertRaisesRegex(c.ContractError,'numeric baseline'):c.validate_constructor(value,tree)

    def test_profile_rebinding_preserves_payload_and_rejects_missing_absolute(self):
        root=self.fresh('assets');data,profile=fake_bundle(root)
        rows={r['relative_path']:r for r in c.source_rows(c.read_json(root/'assets.json'))}
        target=self.fresh('view')
        derived,refs=portable._profile(profile,root,rows,destination=target)
        self.assertEqual(derived['vit_hidden_scales'][0]['sha256'],profile['vit_hidden_scales'][0]['sha256'])
        self.assertEqual(derived['vit_hidden_scales'][0]['raw_sha256'],profile['vit_hidden_scales'][0]['raw_sha256'])
        self.assertTrue(Path(derived['vit_hidden_scales'][0]['path']).is_relative_to(target));self.assertEqual(len(refs),24)
        bad=json.loads(json.dumps(profile));bad['vit_hidden_scales'][0]['path']='G:/old/private.npyz'
        with self.assertRaises(c.ContractError):portable._profile(bad,root,rows)
        broken=dict(rows);broken.pop('data/product-v1/calibration/cpu-fixture.npyz')
        with self.assertRaisesRegex(c.ContractError,'not pinned'):portable._profile(profile,root,broken)

    def test_export_collect_from_relative_runtime_without_tensor_import(self):
        root=self.fresh('runtime');data,profile=fake_bundle(root)
        local=json.loads(json.dumps(profile))
        def replace(v):
            if isinstance(v,list):return [replace(x) for x in v]
            if not isinstance(v,dict):return v
            out={k:replace(x) for k,x in v.items()}
            if 'path' in v:out['path']=str(root/v['path'][8:])
            return out
        p=root/'data/product-v1/profile-v1.json';portable.write_json(p,replace(local))
        portable.write_json(root/'data/product-v1/local-runtime-v1.json',dict(profile_sha256=c.sha(p)))
        files,template,receipts,provenance=exporter.collect(root)
        self.assertTrue(template['vit_hidden_scales'][0]['path'].startswith('${ROOT}/'))
        self.assertEqual(provenance['source_profile']['sha256'],c.sha(p))
        self.assertIn('exact/model-assets/noise-sm89-v2/table.bin',files)
        self.assertNotEqual(provenance['derived_template_sha256'],c.sha(p))
        portable.write_json(root/'data/product-v1/wrong.json',dict(profile_sha256='0'*64))
        (root/'data/product-v1/local-runtime-v1.json').write_text('{"profile_sha256":"'+('0'*64)+'"}')
        with self.assertRaisesRegex(c.ContractError,'cfg/profile'):exporter.collect(root)
        with self.assertRaisesRegex(c.ContractError,'external'):exporter.export(root,c.REPO/'must-not-write-assets')

    def test_actual_plan_method_positive_and_missing_asset_cpu_only(self):
        root=self.fresh('bundle');data,_=fake_bundle(root)
        output=Path(tempfile.gettempdir())/('nr-release-plan-'+root.name+'-'+self.root.name)
        args=runner.parser().parse_args(['--plan','--assets',str(root),'--output',str(output)])
        fixture_sha=hashlib.sha256(data['exact/model-assets/sf-v2/WEIGHTS_HT.bin']).hexdigest()
        with mock.patch.object(portable,'WEIGHTS_SHA',fixture_sha):
            plan,selections=runner.prepare_plan(args)
        self.assertFalse(plan['GPU_executed']);self.assertFalse(output.exists());self.assertEqual(plan['arms'],['installed_current'])
        self.assertEqual(selections[0][2]['numeric_identity']['base_identity'],c.NUMERIC_BASE_IDENTITY)
        self.assertEqual([p['phase'] for p in plan['launch_order']],['precompile','readonly'])
        bad=runner.parser().parse_args(['--plan','--assets',str(root/'absent'),'--output',str(output)])
        with self.assertRaises(OSError):runner.prepare_plan(bad)

    def test_fake_child_failure_stops_zero_new_launch(self):
        for kind in ('SOURCE_FAILURE','COMPILE_FAILURE','ADMISSION_FAILURE','CACHE_FAILURE','NUMERIC_FAILURE','CHILD_FAILURE'):
            root=self.fresh(kind);config=root/'CONFIG.json';portable.write_json(config,{})
            launches=[]
            def launch(command,output,tag,**kw):
                launches.append(tag);(output/tag).mkdir()
                portable.write_json(output/tag/'RESULT.json',dict(completed=False,phase=tag,GPU_executed=False,
                    failures=[dict(category=kind)],diagnostic='CPU fake-child audit only'))
                return dict(returncode=0)
            with self.assertRaisesRegex(c.ContractError,'child evidence'):
                runner.gpu_phases(sys.executable,config,root,timeout=1,env={},launch=launch)
            self.assertEqual(launches,['precompile'])
            self.assertTrue((root/'precompile/RESULT.json').is_file())

    def test_success_and_slow_performance_do_not_require_one_ms_cutoff(self):
        root=self.fresh('success');config=root/'CONFIG.json';portable.write_json(config,{})
        launches=[]
        def launch(command,output,tag,**kw):
            launches.append(tag);(output/tag).mkdir()
            portable.write_json(output/tag/'RESULT.json',dict(completed=True,phase=tag,GPU_executed=True,failures=[],
                fake_CPU_fixture=True,observed_ms=500 if tag=='readonly' else 1))
            return dict(returncode=0)
        self.assertEqual(len(runner.gpu_phases(sys.executable,config,root,timeout=1,env={},launch=launch)),2)
        self.assertEqual(launches,['precompile','readonly'])

    def test_cache_receipt_roundtrip_constants_target_and_drift(self):
        @dataclass(frozen=True)
        class Target:backend:str='xpu';arch:str='cpu-fixture'
        row=dict(loaded_binary_sha256='0'*64,metadata_files={'cpu-fixture':'1'*64},
                 signature=compiler_value({0:'*fp16',(1,):'i32'}),target=compiler_value(Target()))
        p=self.fresh('receipt')/'receipt.json';portable.write_json(p,dict(key=row))
        phase=CachePhase(None,readonly=True,prepared=c.read_json(p));phase.accept_receipt('key',row)
        self.assertEqual(phase.hits,1)
        changed=dict(row,loaded_binary_sha256='2'*64)
        with self.assertRaisesRegex(c.ContractError,'drift'):phase.accept_receipt('key',changed)
        with self.assertRaises(c.ContractError):portable.report_value({1:1,'@int:1':2})

    def test_actual_child_lifecycle_fields_and_raw_counter_filter(self):
        for module,name,fields in (
            ('audit_front_decoder_post_720_v1','Scope',dict(active=False,_retired=True)),
            ('audit_vit_c512_scope_720_v1','CompleteCounter',dict(live=False,retired=True)),
            ('audit_history_host_720_v1','HistoryAdmissionCounter',dict(active=False)),
            ('audit_vit_c512_720_v1','NumericStageHook',{})):
            cls=type(name,(),{'__module__':module});child=cls();vars(child).update(fields)
            state=retired_state(child,root=True);self.assertEqual(state['fields'],fields)
            if fields:
                vars(child)[state['live_field']]=True
                with self.assertRaises(c.ContractError):retired_state(child,root=True)
        first={'calls':{'kernel':2},'validation_calls':1};second={'calls':{'kernel':2},'validation_calls':100}
        self.assertEqual(dispatch_counts(first),dispatch_counts(second))
        second['calls']['kernel']=3;self.assertNotEqual(dispatch_counts(first),dispatch_counts(second))

    def test_user_cli_help_and_negative_plan_no_gpu(self):
        for name in ('run_current.py','export_assets.py','run_historical.py'):
            cmd=[sys.executable,'-I','-X','utf8','-B',str(c.HERE/name)]
            self.assertEqual(subprocess.run(cmd+['--help'],capture_output=True).returncode,0)
        bad=subprocess.run([sys.executable,'-I','-X','utf8','-B',str(c.HERE/'run_current.py'),
                            '--plan','--output',str(self.root/'no-launch')],capture_output=True,text=True)
        self.assertNotEqual(bad.returncode,0);self.assertIn('GPU_executed',bad.stderr);self.assertFalse((self.root/'no-launch').exists())
        self.assertFalse(any(name in sys.modules for name in ('torch','triton','numpy','nr_backend')))

def main():
    global SCRATCH
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scratch',type=Path,help='retained CPU fixture directory; default OS temp')
    p.add_argument('--report',type=Path)
    args=p.parse_args()
    SCRATCH=None if args.scratch is None else str(c.plain_path(args.scratch))
    if SCRATCH:Path(SCRATCH).mkdir(parents=True,exist_ok=True)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CPURegressions)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    summary=dict(schema='nr-release-cpu-tests-v1',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),
        GPU_executed=False,Tensor_libraries_imported=False,retained_fixture_directory=str(CPURegressions.root))
    if args.report:portable.write_json(args.report,summary)
    print(json.dumps(summary));return 0 if result.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(main())
