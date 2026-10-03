"""Run with python -I -S; only stdlib and these CPU helper modules are imported."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import unittest
from types import SimpleNamespace as NS
import gc
import hashlib
import threading
import weakref
from contextlib import contextmanager
from types import MethodType

ROOT = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


cpu = load('audit_history_cpu_720_v1')
admission = load('audit_history_host_720_v1')
receipts = load('audit_receipt_720_v1')
launches = load('audit_screened_launch_720_v1')
resources = load('audit_history_resources_720_v1')
summary_module = load('audit_history_input_720_v1')
front_components = load('audit_history_front_input_720_v1')
controls_module = load('audit_control_graph_720_v1')
replay_receipts = load('audit_replay_receipt_720_v1')
source_admission = load('audit_history_source_admission_720_v1')
graph_prepare = load('audit_graph_prepare_720_v1')
METRICS = {}


@contextmanager
def fake_modules(rows):
    saved = {key: sys.modules.get(key) for key in rows}
    sys.modules.update(rows)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def source_method(relative, class_name, method_name):
    tree = ast.parse((ROOT / relative).read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    fn.decorator_list = []
    body = ast.Module(body=[fn], type_ignores=[])
    scope = {}
    exec(compile(ast.fix_missing_locations(body), str(ROOT / relative), 'exec'), scope)
    return scope[method_name]


class FakeTensor:
    def __init__(self, shape=(720, 1280, 3), dtype='float32', device='xpu0', pointer=32, values=()):
        self.shape, self.dtype, self.device, self.pointer = shape, dtype, device, pointer
        self.values = values
        self.value = 0

    def stride(self):
        result, stride = [], 1
        for size in reversed(self.shape):
            result.append(stride); stride *= size
        return tuple(reversed(result))

    def data_ptr(self):
        return self.pointer

    def is_contiguous(self):
        return True

    def item(self):
        return self.value


class FakeChild:
    pass


class FakeCompiled:
    def __init__(self, jit):
        self.src = NS(fn=jit, constants={(2,):720, (3,):False}, attrs={(0,):[['tt.divisibility',16]]})
        self.launches = []
        self.kernel = b'fake-CPU-test-payload'

    def __getitem__(self, grid):
        return lambda *args: self.launches.append((grid, args))


class NumericContract(unittest.TestCase):
    def test_motion_domain_and_whole_frame_near(self):
        for value in (0, -0.0, 65504, -65504, 1, -1, 1 / 256, -1 / 256):
            self.assertEqual(cpu.motion_summary([value]), 0)
        self.assertEqual(cpu.motion_summary([0, 0.004]), 4)
        for value in (math.nan, math.inf, -math.inf):
            self.assertTrue(cpu.motion_summary([value]) & 1)
        self.assertTrue(cpu.motion_summary([65505]) & 2)
        # The predicate is HALF quantized; a float32 just outside can round in.
        self.assertEqual(cpu.motion_summary([1 / 256 + 1e-8]), 0)
        self.assertEqual(cpu.motion_summary([1 / 256 + 4e-6]), 4)

    def test_coordinate_range_and_coefficients(self):
        for x in (-65504, 0, 0.5, 0.999, 1279.5, 65504):
            for y in (-65504, 0, 0.5, 719.5, 65504):
                for direct in (False, True):
                    indices, weights = cpu.coefficients(x, y, direct=direct)
                    self.assertEqual(sum(weights), 256)
                    self.assertGreaterEqual(min(weights), 0)
                    for iy, ix in indices:
                        self.assertTrue(0 <= iy < 720 and 0 <= ix < 1280)

    def test_full_five_tap_rgb_and_edges(self):
        image = lambda y, x, ch: cpu.half(((x * 3 + y * 7 + ch * 13) % 4096) / 4096)
        max_error = 0
        for x, y in ((0, 0), (1279, 719), (640, 360), (1, 1)):
            for mx, my in ((0, 0), (1, -1), (1 / 256, -1 / 256), (0.25, -0.75), (65504, -65504)):
                native, reciprocal, taps = cpu.sample_pixel(image, x, y, mx, my, near=True)
                reference, ref_rcp, ref_taps = cpu.sample_pixel(image, x, y, mx, my, near=True, reference_values=True)
                self.assertEqual(len(taps), 5)
                self.assertEqual(reciprocal, ref_rcp)
                self.assertTrue(all(math.isfinite(v) for v in native))
                max_error = max(max_error, *(abs(a - b) for a, b in zip(native, reference)))
        METRICS['cpu_pixel_numerator_max_abs_vs_reference_values'] = max_error

    def test_value_rounding_records_error(self):
        random.seed(720)
        max_error, different = 0, 0
        for _ in range(2048):
            ax, ay = random.randrange(256), random.randrange(256)
            cross = (ax * ay + 128) >> 8
            weights = (256 - ax - ay + cross, ax - cross, ay - cross, cross)
            pixels = [cpu.half(random.uniform(-1, 1)) for _ in range(4)]
            fast = cpu.texture_native(pixels, weights)
            exact = cpu.texture_reference(pixels, weights)
            different += cpu.half_bits(fast) != cpu.half_bits(exact)
            max_error = max(max_error, abs(fast - exact))
        self.assertGreater(different, 0)
        METRICS.update(cpu_half_value_fixtures=2048, cpu_half_value_different_bits=different,
                       cpu_half_value_max_abs_error=max_error, gpu_numeric_error=None)

    def test_kernel_source_and_all_specializations(self):
        # Compile() parses the Triton source without importing or JIT compiling it.
        for name in ('history_numeric_suite_720_v1_kernel.py', 'history_numeric_suite_720_v1.py',
                     'replay_lifecycle_audit_rest_720_v1.py'):
            source = (ROOT / name).read_text(encoding='utf-8')
            compile(source, str(ROOT / name), 'exec')
        source = (ROOT / 'history_numeric_suite_720_v1_kernel.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_five_tap_rgb')
        self.assertIn('NEAR', [a.arg for a in fn.args.args])
        self.assertIn('WRITE_NORMALIZED', [a.arg for a in fn.args.args])
        static_loop = [n for n in ast.walk(fn) if isinstance(n, ast.For)]
        self.assertEqual(len(static_loop), 1)
        self.assertIn('not NEAR', source)

    def test_all_half_motion_predicates(self):
        import struct
        mismatches = 0
        for bits in range(65536):
            value = struct.unpack('<e', struct.pack('<H', bits))[0]
            expected = cpu.motion_summary([value])
            if math.isfinite(value):
                native_near = abs(value - math.floor(cpu.f32(value + 0.5))) <= 1 / 256
                mismatches += native_near != (expected == 0)
        self.assertEqual(mismatches, 0)
        METRICS['cpu_all_65536_half_motion_predicate_mismatches'] = mismatches

    def test_current_compiled_specialization_and_output_contract(self):
        specializations = source_method('history_numeric_suite_720_v1.py','HistoryNumericCounter720','_specializations')
        contracts = source_method('history_numeric_suite_720_v1.py','HistoryNumericCounter720','output_contracts')
        child = NS(implementation=admission.HistoryImplementation720(input_summary=True),
                   options=NS(history_value='fp32_all_paths',history_reciprocal='native'), _debug=False)
        keys = specializations(child)
        self.assertEqual(len(keys), 12)
        self.assertTrue({'axes_fp16','axes_fp32','five_tap_debug_false','five_tap_debug_true',
                         'near_five_tap_debug_false','near_five_tap_debug_true',
                         'input_flags_color','input_flags_motion'} <= keys)
        self.assertNotIn('NORMALIZED', contracts(child)['_five_tap_rgb'][0])
        child._debug = True
        self.assertIn('NORMALIZED', contracts(child)['_five_tap_rgb'][0])
        self.assertIn('TAPS', contracts(child)['_five_tap_rgb'][0])
        METRICS['selected_history_specializations'] = sorted(keys)


class ReceiptAndDispatchContract(unittest.TestCase):
    def test_100000_frame_history_branch_ring_sequence(self):
        history, branch = receipts.ReceiptRing(64), receipts.ReceiptRing(64)
        counter = NS(frame_routes=history)
        for sequence in range(100000):
            before = receipts.frame_sequence(counter)
            history.append(dict(frame=sequence, history_route='near_integer' if sequence % 3 else 'fractional',
                                selection_gate=True, new_entries=int(sequence < 2),
                                seed_before=sequence, seed_after=sequence + 1))
            detail = receipts.latest_history_frame(counter, before)
            branch.append(dict(frame=sequence, history_detail=detail, passed=True))
        self.assertEqual(history.total, 100000)
        self.assertEqual(len(history),64)
        self.assertEqual(branch[-1]['history_detail']['frame'],99999)
        snap = history.snapshot()
        snap['rows'][-1]['frame'] = -1
        self.assertEqual(history[-1]['frame'],99999)
        with self.assertRaises(RuntimeError): receipts.latest_history_frame(counter,0)
        METRICS['receipt_100000_frames'] = dict(total=history.total, retained=len(history), dropped=history.total-len(history),
                                               branch_retained=len(branch), actual_consumer_sequence=True)

    def test_screened_binary_rejects_abi_changes_before_launch(self):
        jit = NS(arg_names=['IMAGE','OUTPUT','H','DEBUG'])
        kernel = FakeCompiled(jit)
        a, out = FakeTensor(dtype='half'), FakeTensor()
        launch = launches.ScreenedLaunch(kernel,jit,(a,out,720,False),(14400,),FakeTensor)
        self.assertIs(launch(jit,(a,out,720,False),(14400,)),kernel)
        self.assertEqual(len(kernel.launches),1)
        bad = [((FakeTensor(dtype='float32'),out,720,False),(14400,)),
               ((FakeTensor(dtype='half',pointer=36),out,720,False),(14400,)),
               ((a,out,721,False),(14400,)),((a,out,720,True),(14400,)),((a,out,720,False),(14401,))]
        for args, grid in bad:
            with self.assertRaises(RuntimeError): launch(jit,args,grid)
        self.assertEqual(len(kernel.launches),1)

    def test_actual_capture_token_one_replay_and_failure(self):
        graph = NS(entries={'key':object()}, replays=12, last_entry=None)
        child = NS(graph=graph)
        seal = NS(entry=NS(replays=7),current=lambda:True)
        graph.last_entry = seal.entry
        state = NS(frame=object(),graph=graph,pending=None,consumed=('key',seal,6,11),
                   capture_epoch=2,last_capture_receipt=None,require_frame=lambda:None)
        token = replay_receipts.CaptureToken(state,state.frame,2,11,1)
        self.assertEqual(replay_receipts.capture_delta(child,token),(0,1,None))
        graph.replays += 1
        with self.assertRaises(RuntimeError):replay_receipts.capture_delta(child,token)

    def test_actual_dispatch_counts_only_successful_submissions(self):
        method=source_method('history_numeric_suite_720_v1.py','HistoryNumericCounter720','_launch')
        events=[];kernel=NS(hash='cpu-fake-not-a-gpu-hash')
        class Jit:
            def warmup(self,*a,**kwargs):events.append('warmup');return kernel
            def __getitem__(self,grid):
                def run(*a,**kwargs):events.append('jit-launch');return kernel
                return run
        jit=Jit();direct=NS(kernel=kernel)
        class Direct:
            kernel=direct.kernel
            def __call__(self,*args):events.append('screened-launch');return kernel
        child=NS(_compiled={'test':kernel},_screened_launches={'test':Direct()},
                 implementation=NS(direct_compiled_launch=True),triton=NS(cdiv=lambda a,b:(a+b-1)//b),
                 _launch_calls={},actualkernelhash={},resources={'test':{'actualbinary_sha256':'cpu-fake'}},dispatch_cache_queries=0)
        with fake_modules({'replay_lifecycle_audit_rest_720_v1':NS(history_binary=lambda *a,**k:None)}):
            method(child,'test',jit,(),64,64)
            self.assertEqual(events,['screened-launch']);self.assertEqual(child.dispatch_cache_queries,0)
            child.implementation.direct_compiled_launch=False
            method(child,'test',jit,(),64,64)
            self.assertEqual(events,['screened-launch','warmup','jit-launch'])
            self.assertEqual(child._launch_calls,{'test':2})
            child.implementation.direct_compiled_launch=True
            child._screened_launches.clear()
            with self.assertRaises(RuntimeError):method(child,'test',jit,(),64,64)
            self.assertEqual(child._launch_calls,{'test':2})


class InputSummaryContract(unittest.TestCase):
    def make(self, values, dtype='float32'):
        global _IN_FLIGHT
        child = FakeChild()
        child.torch = NS(float32='float32',float16='half',xpu=NS(is_current_stream_capturing=lambda:False))
        child.thread=threading.get_ident();child.model=NS();child.device='xpu0'
        child.in_frame=True;child.active=True;child.retired=False
        child.session=NS(_history_numeric_suite_720=child)
        child.modes=NS(session=child.session)
        child.triton=NS(cdiv=lambda a,b:(a+b-1)//b)
        child._prepare_buffers={'input_partial':FakeTensor((2700,),dtype='int32'),
                                'input_flags':FakeTensor((),dtype='int32')}
        child.labels=[]
        def launch(label,jit,args,count,block):
            child.labels.append(label)
            child._prepare_buffers['input_flags'].value=cpu.motion_summary(values)
        child._launch=launch
        child.jits={'_input_summary':object(),'_summary_flags':object()}
        _IN_FLIGHT=child
        rgb=FakeTensor();motion=FakeTensor((720,1280,2),dtype=dtype,values=values)
        summary=summary_module.InputSummary720(child)
        return child,summary,rgb,motion

    def test_single_readback_reset_and_near_fractional(self):
        for values, expected in (([0,-0.0,1/256],0),([0,.25],4)):
            child,summary,rgb,motion=self.make(values)
            summary.begin(child.model,rgb,motion,False)
            self.assertEqual(summary.flags,expected)
            self.assertEqual(summary.readbacks,1)
            self.assertEqual(child.labels,['input_fp32_motion','input_flags_motion'])
            self.assertTrue(summary.current(child.model,motion))
            self.assertFalse(summary.current(child.model,FakeTensor((720,1280,2))))
            summary.end();self.assertIsNone(summary.motion)

    def test_invalid_does_not_publish_a_summary_or_temporal_commit(self):
        child,summary,rgb,motion=self.make([math.nan])
        child.model._previous=object();child.model._next_seed=17
        previous=child.model._previous
        with self.assertRaises(ValueError):summary.begin(child.model,rgb,motion,True)
        self.assertIsNone(summary.flags)
        self.assertIs(child.model._previous,previous);self.assertEqual(child.model._next_seed,17)
        self.assertEqual(child.labels,['input_fp32_color','input_flags_color'])


class LifetimeAndAdmissionContract(unittest.TestCase):
    def test_close_releases_python_strongrefs_and_preserves_borrowed_owner(self):
        class Allocation: pass
        class Stack: pass
        class Product:
            def __init__(self,stack):self._stack=stack;self._closed=False;self.fail=False
            def close(self):
                if self.fail:raise RuntimeError('fake retirement failure')
                self._stack.graph.closed=True;self._stack.graph.entries.clear()
                self._stack=None;self._closed=True
        class Session:
            def __init__(self,stack):
                self._stack=stack;self._product=Product(stack);self._closed=False
                self._counters=NS(stack=stack)
                self._installed=lambda:stack
            def close(self):
                self._product.close();self._stack=None;self._closed=True
        stack=Stack();allocation=Allocation();stack.graph=NS(closed=False,entries={'a':object()})
        stack.tensor=allocation;borrowed=NS(storage=allocation)
        reference=weakref.ref(stack);alloc_ref=weakref.ref(allocation)
        session=Session(stack);owner=resources.install_session_retirement(session)
        session._product.fail=True
        del stack;del allocation;gc.collect()
        with self.assertRaises(RuntimeError):session.close()
        self.assertIsNotNone(reference());self.assertFalse(owner.closed)
        session._product.fail=False;session.close();gc.collect()
        self.assertIsNone(reference());self.assertIsNotNone(alloc_ref())
        self.assertIsNone(session._counters)
        del borrowed;gc.collect();self.assertIsNone(alloc_ref())
        METRICS['python_strongrefs_released_after_verified_close']=True
        METRICS['gpu_allocator_release_measured']=False

    def test_zero_origin_indices_and_complete_context(self):
        g=NS(source=(720,1280),mode=NS(model=(720,1280),active=(720,1280),internal=(768,1280),inset=(0,0)),
             tables={},rows=object(),columns=object())
        result=resources.release_unused_geometry_indices(g)
        self.assertEqual(result['logical_resident_bytes_removed'],16000)
        self.assertIsNone(g.rows);self.assertIsNone(g.columns)
        self.assertEqual(g.mode.internal,(768,1280))
        g.mode.inset=(1,0)
        with self.assertRaises(ValueError):resources.release_unused_geometry_indices(g)

    def test_hook_delegates_single_child_and_no_thread_ownership(self):
        options=admission.HistoryImplementation720(component_front_input=False, reuse_buffers=False)
        session=NS();modes=NS(session=session,_audit_history_host_options_720=options)
        calls=[]
        child=NS(modes=modes,session=session,implementation=options,active=True,retired=False,thread=threading.get_ident(),
                 _compiled={'key':object()},_specializations=lambda:{'key'},validate=lambda:calls.append('validate') or True,
                 preflight=lambda:calls.append('cached_preflight'),snapshot=lambda:{'actual_child':True},
                 calls={})
        session._history_numeric_suite_720=child
        modes._numeric_cleanup_owner=NS(session=session,children={'history':child})
        counter=admission.HistoryAdmissionCounter(modes,session,options)
        counter.preflight();counter.validate_frame_context()
        self.assertIs(counter.child,child)
        self.assertEqual(calls.count('cached_preflight'),1)
        self.assertFalse(hasattr(counter,'thread'))
        self.assertTrue(counter.snapshot()['single_history_executor'])
        session._history_numeric_suite_720=NS()
        with self.assertRaises(RuntimeError):counter.validate()

    def test_control_lane_semantics(self):
        c=NS(style=2,intensity=.25,local_tone=.5,local_structure=.75,auto_mask=False,skin_structure=1.5)
        self.assertEqual(controls_module.control_lanes(c),(2/128,.5,.75,-1,-1))
        c.auto_mask=True
        self.assertEqual(controls_module.control_lanes(c),(2/128,.5,1,1.5,.75))
        c.skin_structure=None
        self.assertEqual(controls_module.control_lanes(c),(2/128,.5,1,.75,.75))

    def test_serial_metadata_transfer_checks_already_moved_sole_sampler(self):
        module=NS()
        class HookCounter:pass
        module.HookCounter=HookCounter
        lock=threading.RLock();adapter=NS(_process_serial_lock=lock)
        old_mod=sys.modules.get('implementation_hooks_720_v1');old_adapter=sys.modules.get('cyberpunk_nr_adapter')
        sys.modules['implementation_hooks_720_v1']=module;sys.modules['cyberpunk_nr_adapter']=adapter
        try:
            options=admission.HistoryImplementation720();session=NS();modes=NS(session=session)
            child=NS(modes=modes,session=session,implementation=options,active=True,retired=False,thread=threading.get_ident(),
                     _require_serial_idle=lambda:None)
            session._history_numeric_suite_720=child
            modes._numeric_cleanup_owner=NS(session=session,children={'history':child})
            counter=admission.HistoryAdmissionCounter(modes,session,options)
            owner=HookCounter();owner.session=session;owner.modes=modes;owner.children={'audit_history_host_720_v1':counter}
            with lock:
                current=counter._transfer_serial_thread(owner,serial_guard=lock,previous_thread=77)
                self.assertEqual(current,threading.get_ident());self.assertEqual(child.thread,current)
                child.thread=77
                with self.assertRaises(RuntimeError):counter._transfer_serial_thread(owner,serial_guard=lock,previous_thread=77)
                self.assertEqual(child.thread,77)
        finally:
            for name,value in [('implementation_hooks_720_v1',old_mod),('cyberpunk_nr_adapter',old_adapter)]:
                if value is None:sys.modules.pop(name,None)
                else:sys.modules[name]=value

    def test_cold_source_pin_admission_restores_exact_guards(self):
        manifest=json.loads((ROOT/'HISTORY_SOURCE_ADMISSION.json').read_text(encoding='utf-8'))
        roles={row['module']:NS(__file__=str(ROOT.parent/row['relative_path'])) for row in manifest['roles'].values()}
        decoder={'nr_backend.temporal':'old-t','nr_game_controlled_model':'old-m'}
        post={'temporal':('old-t',),'controlled':('old-m',),'graph_v1':('old-g',)}
        history={'temporal':'old-t'}
        guards={'decoder_input_full_k_720_v1':NS(_KNOWN=decoder),
                'post_numeric_suite_720_v1':NS(KNOWN_SHA256=post),
                'history_numeric_suite_720_v1':NS(REFERENCE_SHA256=history)}
        originals=[dict(m) for m in (decoder,post,history)]
        with fake_modules({**roles,**guards}):
            owner=source_admission.SourceAdmission720();receipt=owner.install()
            self.assertEqual(len(receipt),4)
            self.assertEqual(decoder['nr_backend.temporal'],manifest['roles']['temporal']['sha256'])
            self.assertEqual(post['graph_v1'][0],'old-g')
            owner.restore()
            self.assertEqual([decoder,post,history],originals)
            roles['nr_game_controlled_model'].__file__=str(ROOT/'audit_history_resources_720_v1.py')
            with self.assertRaises(RuntimeError):source_admission.SourceAdmission720().install()
            self.assertEqual([decoder,post,history],originals)
        METRICS['cold_source_admission_pins_verified']=4

    def test_actual_main_hook_skips_metadata_thread_and_validates_moved_child(self):
        path=ROOT.parents[1]/'evidence/implementation_hooks_720_v1.py'
        if not path.is_file():
            common=next(p for p in ROOT.parents if (p/'BASELINE.json').is_file())
            path=common/'integrated/source/game/implementation_hooks_720_v1.py'
        text=path.read_text(encoding='utf-8');tree=ast.parse(text)
        # The real dispatcher is stdlib-only; restrict its executed top-level
        # imports as well. No numerical/device child is imported here.
        imports=[n.module if isinstance(n,ast.ImportFrom) else n.names[0].name
                 for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertTrue(set(imports)<={'__future__','contextlib','dataclasses','hashlib','importlib','json','sys','threading','pathlib','types'})
        spec=importlib.util.spec_from_file_location('implementation_hooks_720_v1',path)
        module=importlib.util.module_from_spec(spec)
        with fake_modules({'implementation_hooks_720_v1':module}):
            spec.loader.exec_module(module)
            hooks=module.parse_hooks([{'module':'audit_history_host_720_v1','kwargs':{}}])
            graph=NS(closed=False);stack=NS(graph=graph)
            session=NS(_stack=stack);options=admission.HistoryImplementation720(component_front_input=False)
            modes=NS(session=session,height=720,source=(720,1280),variant='unrounded',implementation_hooks_720=hooks,
                     _audit_history_host_options_720=options)
            child=NS(modes=modes,session=session,implementation=options,active=True,retired=False,thread=77)
            def require_thread():
                if child.thread!=threading.get_ident():raise RuntimeError('actual sole child not migrated')
            child.validate=require_thread
            session._history_numeric_suite_720=child
            modes._numeric_cleanup_owner=NS(session=session,children={'history':child})
            counter=admission.HistoryAdmissionCounter(modes,session,options)
            owner=module.HookCounter(modes,hooks,'before_numeric');owner.register_child('audit_history_host_720_v1',counter)
            self.assertEqual(owner.serial_participants(77),[])
            with self.assertRaises(RuntimeError):owner.validate_frame_context()
            self.assertEqual(child.thread,77)
            child.thread=threading.get_ident();owner.validate_frame_context()
            self.assertFalse(hasattr(counter,'thread'))
            METRICS['actual_main_hook_api_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            METRICS['actual_main_hook_metadata_after_numeric_checked']=True

    def test_prepared_dimension_args_keep_bits_and_logical_sites(self):
        method=source_method('history_numeric_suite_720_v1.py','HistoryNumericCounter720','_dimension_args')
        method.__globals__.update(W=1280,H=720)
        iw,ih=object(),object();hits=[]
        child=NS(prepared_dimensions=(iw,ih),options=NS(history_coord='reference'),_hit=hits.append)
        self.assertIs(method(child,dispatch=True),child.prepared_dimensions)
        self.assertEqual(hits,['dimension.width','dimension.height'])
        child.options.history_coord='direct_pixel';hits.clear()
        self.assertEqual(method(child,dispatch=True),(iw,ih));self.assertEqual(hits,[])


class GraphAndConsumerContract(unittest.TestCase):
    def test_sealed_history_destinations_peek_without_consuming(self):
        d=NS(type='xpu',index=0);graph=NS(closed=False)
        controls=NS(style=0,intensity=1)
        child=NS(implementation=NS(reuse_buffers=True),in_frame=True,graph=graph,
                 model=NS(_controls=controls),device=d,torch=NS(float32='float32',float16='half'),modes=NS(_numeric_cleanup_owner=object()))
        desc=lambda shape,dtype:(shape,dtype,'xpu',0)
        key=(desc((720,1280,3),'float32'),desc((768,1280,16),'half'),
             desc((720,1280,3),'float32'),desc((720,1280),'float32'),False)
        previous=FakeTensor(device=d);rcp=FakeTensor((720,1280),device=d)
        entry=NS(inputs={'previous':previous,'history_reciprocal':rcp})
        state=NS(armed=True,pending=None,graph=graph,sealed={key:NS(entry=entry,current=lambda:True)},
                 entries={key:entry},consumed=None,require_frame=lambda:None)
        with fake_modules({'replay_lifecycle_audit_base_720_v1':NS(state_for=lambda _:state)}):
            self.assertEqual(graph_prepare.history_destinations(child),(previous,rcp))
            self.assertIsNone(state.consumed)
            state.entries[key]=object()
            with self.assertRaises(RuntimeError):graph_prepare.history_destinations(child)
            state.pending=object();self.assertIsNone(graph_prepare.history_destinations(child))
        METRICS['sealed_destination_peek_preserves_actual_graph_consumption']=True

    def test_front_components_keep_full_context_and_authenticate_callable(self):
        class Front:
            def apply_history_components(self,rgb,num,rcp,**options):
                self.received=(rgb,num,rcp,options)
                return FakeTensor((768,1280,16),dtype='half')
            def validate(self):self.validated+=1
        c=Front();c.model=NS(_previous=object());c.stack=NS(model=c.model);c.session=NS(_stack=c.stack)
        c.modes=NS(session=c.session);c.active=True;c.torch=NS(float32='float32',float16='half');c.validated=0
        c.options={'front_noise':'native_both'}
        c.modes._numeric_cleanup_owner=NS(session=c.session,children={'front':c})
        owner=front_components.bind_front_component_consumer(c.modes,c)
        rgb,num,rcp=FakeTensor(),FakeTensor(),FakeTensor((720,1280))
        result=front_components.prepare_history_front(c.model,rgb,num,rcp,{'seed':17},lambda *a,**k:None)
        self.assertEqual(result.shape,(768,1280,16));self.assertIs(c.received[1],num)
        self.assertEqual(owner.calls,1);self.assertEqual(c.validated,1)
        c.apply_history_components=lambda *a,**k:result
        with self.assertRaises(RuntimeError):owner.apply(rgb,num,rcp,{'seed':17})

    def test_actual_graph_controls_raw_capture_eager_and_failure(self):
        method=source_method('nr_game_controlled_model.py','GameLiveControlledNR','graph_controls')
        method.__globals__.update(descriptor=lambda x:None if x is None else tuple(x.shape),
                                  cold_raw_capture_allowed=lambda m,g:getattr(m,'cold',False),
                                  control_lanes=controls_module.control_lanes,store_half_rz=lambda x:('half',x))
        class Lanes(FakeTensor):
            def __init__(self):super().__init__((768,1280,16),dtype='half');self.fills=[]
            def __setitem__(self,key,value):self.fills.append((key[1],value))
        graph=NS(entries={});events=[];output=object()
        def forward(self,*a,**kwargs):events.append(('graph',kwargs));return output
        def eager(*a,**kwargs):events.append(('eager',kwargs));return output
        graph.forward=MethodType(forward,graph);graph.original=eager
        ctrl=NS(style=0,intensity=.25,local_tone=.5,local_structure=.75,auto_mask=True,skin_structure=1.5)
        model=NS(_controls=ctrl,_forward_front=graph.forward,_next_seed=17,_previous=object(),
                 _audit_history_host_modes_720=object(),cold=False)
        native_controls=source_method('nr_game_controlled_model.py','GameLiveControlledNR','_native_controls_present_720')
        native_controls.__globals__['control_lanes']=controls_module.control_lanes
        model._native_controls_present_720=MethodType(native_controls,model)
        rgb,front=FakeTensor(),Lanes();previous,rcp=FakeTensor(),FakeTensor((720,1280))
        call=contextmanager(method)
        with call(model,graph,allow_capture=False):
            with self.assertRaises(RuntimeError):model._forward_front(rgb,front,previous=previous,history_reciprocal=rcp)
        self.assertEqual(events,[]);self.assertEqual(model._next_seed,17)
        model.cold=True
        with call(model,graph,allow_capture=False):
            result=model._forward_front(rgb,front,previous=previous,history_reciprocal=rcp)
        self.assertEqual(result,('half',output));self.assertTrue(events[-1][1]['return_float32'])
        self.assertEqual(front.fills,list(zip(range(10,15),controls_module.control_lanes(ctrl))))
        model.cold=False;del model._audit_history_host_modes_720;front.fills.clear()
        with call(model,graph,allow_capture=False):
            self.assertEqual(model._forward_front(rgb,front,previous=previous,history_reciprocal=rcp),('half',output))
        self.assertEqual(events[-1][0],'eager');self.assertTrue(events[-1][1]['return_float32'])
        self.assertIs(model._raw_private,output)
        METRICS['actual_raw_control_ast_contract']=['missing fails before commit','cold token permits raw after default warmups','legacy eager returns FP32 private then half']

    def test_cold_raw_token_requires_actual_host_and_held_process_guard(self):
        lock=threading.RLock();model=NS();graph=NS(closed=False)
        modes=NS(session=NS(_stack=NS(model=model,graph=graph),_failed=False),height=720,source=(720,1280),variant='unrounded',_numeric_cleanup_owner=object())
        host=NS(_modes=modes,_failed=False);adapter=NS(_process_serial_lock=lock,host=host,_settings_lock=threading.RLock())
        state=NS(require_frame=lambda:None)
        with fake_modules({'cyberpunk_nr_adapter':adapter,'nr_game_pre_xess_host':host,
                           'replay_lifecycle_audit_base_720_v1':NS(state_for=lambda _:state)}):
            token=controls_module.RawGraphWarmup720(modes,adapter,lock)
            with self.assertRaises(RuntimeError):token._require_owner(model,graph)
            with lock:token._require_owner(model,graph)
            host._failed=True
            with lock:
                with self.assertRaises(RuntimeError):token._require_owner(model,graph)
            self.assertNotIn('_audit_raw_graph_warmup_720',modes.__dict__)


class HostCPUContract(unittest.TestCase):
    def test_actual_control_constructor_cache_and_bounded_log(self):
        path=ROOT.parents[1]/'host/nr_host_receipts_720_v1.py'
        spec=importlib.util.spec_from_file_location('nr_host_receipts_720_v1',path)
        host=importlib.util.module_from_spec(spec);spec.loader.exec_module(host)
        source=ROOT.parents[0]/'experimental/fp8_unround_overlay/nr_backend/controlled_temporal.py'
        tree=ast.parse(source.read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NRControls')
        import dataclasses,struct
        scope={'dataclass':dataclasses.dataclass,'math':math,'struct':struct}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(source),'exec'),scope)
        controls_type=scope['NRControls']
        setting=NS(style=0,model_intensity=1.0,local_tone=1.0,local_structure=1.0,auto_mask=False,skin_structure=None)
        cache=host.ControlsCache720();a=cache.get(setting,controls_type)
        self.assertIs(cache.get(setting,controls_type),a)
        setting.style=True
        with self.assertRaises(ValueError):cache.get(setting,controls_type)
        setting.style=1
        self.assertIsNot(cache.get(setting,controls_type),a)
        self.assertEqual(cache.builds,2)
        if len(sys.argv)>1:
            root=Path(r'D:\Codex-NR-Experiments\cyberpunk-opt\b580-full-implementation-v1-20261003\cpu-history-interface').resolve()
            target=(root/'host-cpu-tests/bounded.jsonl').resolve()
            self.assertTrue(target.is_relative_to(root))
            logs=host.BoundedHostLogs(max_bytes=256,recent=4)
            for i in range(100):logs.append(target,{'frame':i,'cold_event':'CPU synthetic'})
            self.assertLessEqual(target.stat().st_size,256)
            self.assertLessEqual(target.with_name(target.name+'.previous').stat().st_size,256)
            self.assertEqual(len(logs.recent),4)
            METRICS['host_control_cache']=cache.snapshot()
            METRICS['host_log_bound_test']=logs.snapshot()


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in
                              (NumericContract,ReceiptAndDispatchContract,InputSummaryContract,
                               LifetimeAndAdmissionContract,GraphAndConsumerContract,HostCPUContract))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(schema='history-host-graph-cpu-stage-v1', tests_run=result.testsRun,
                  passed=result.wasSuccessful(), metrics=METRICS,
                  imported_device_modules=[n for n in sys.modules if n.split('.')[0] in ('torch', 'triton', 'numpy')])
    if len(sys.argv) > 1:
        target = Path(sys.argv[1]); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
