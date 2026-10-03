"""Stdlib CPU routes, real Python callables and fake tensors/native exports.

Run python -I -S -B test_audit_history_stage3_cpu.py D:/.../CPU_STAGE3.json.
No device modules, DLL constructors, drivers, GPU or application calls.
"""
import ast
import ctypes as C
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace as NS, MethodType
import unittest

GAME = Path(__file__).resolve().parent
GROUP = GAME.parents[1]
COMMON = GROUP.parents[1] if GROUP.name == 'history-host-graph' else GROUP.parents[2]
EVIDENCE = GROUP / 'evidence'
METRICS = {}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def modules(rows):
    saved = {n: sys.modules.get(n) for n in rows}
    sys.modules.update(rows)
    try:
        yield
    finally:
        for n, previous in saved.items():
            if previous is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = previous


def method(path, cls, name):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), 'exec'), scope)
    return scope[name]


graph_prepare = load('audit_graph_prepare_720_v1', GAME / 'audit_graph_prepare_720_v1.py')
component = load('audit_history_front_input_720_v1', GAME / 'audit_history_front_input_720_v1.py')
admission = load('audit_history_host_720_v1', GAME / 'audit_history_host_720_v1.py')
raw_control = load('audit_control_graph_720_v1', GAME / 'audit_control_graph_720_v1.py')
registry = load('numeric_game_profiles_720_v1', GAME / 'numeric_game_profiles_720_v1.py')
texture = load('nr_texture_bridge_v1', GROUP / 'host/nr_texture_bridge_v1.py')
hdr = load('nr_hdr720_bridge_v1', GROUP / 'host/nr_hdr720_bridge_v1.py')
host = load('nr_game_pre_xess_host', GROUP / 'host/nr_game_pre_xess_host.py')
receipts = load('nr_host_receipts_720_v1', GROUP / 'host/nr_host_receipts_720_v1.py')
flash = load('periodic_flash_snapshot_v2', GAME / 'periodic_flash_snapshot_v2.py')
web_stub = ModuleType('cyberpunk_nr_web'); web_stub.Controls = hdr.Controls720
protocol_path = EVIDENCE / 'cyberpunk_native_protocol_v1.py'
if not protocol_path.exists():
    protocol_path = COMMON / 'integrated/host/cyberpunk_native_protocol_v1.py'
with modules({'cyberpunk_nr_web': web_stub}):
    protocol = load('cyberpunk_native_protocol_v1', protocol_path)


class Tensor:
    next_pointer = 100
    def __init__(self, shape=(720, 1280, 3), dtype='f32', device=None):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.pointer = Tensor.next_pointer; Tensor.next_pointer += 100
        self.copies = 0
    def is_contiguous(self): return True
    def stride(self):
        result=[]; size=1
        for dimension in reversed(self.shape):
            result.append(size); size*=dimension
        return tuple(reversed(result))
    def data_ptr(self): return self.pointer
    def copy_(self, other): self.copies += 1
    def clone(self): return Tensor(self.shape, self.dtype, self.device)
    def __mul__(self, other): raise AssertionError('Normalized history must not be materialized')


class FusedFront:
    def apply(self, rgb, previous=None, **options):
        raise AssertionError('The selected component route must execute the native front')


def peer_root():
    if (EVIDENCE / 'front/front_noise_native_720_v1.py').exists():
        return EVIDENCE / 'front'
    root = COMMON / 'workers/front-decoder-post'
    return root / ('stage3/source/game' if (root / 'stage3').exists() else 'source/game')


class FrontFixture:
    def __init__(self):
        root = peer_root()
        self.peer_files = {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
                          ('front_noise_native_720_v1.py','front_route_dispatch_720_v1.py',
                           'post_numeric_suite_720_v1.py','audit_front_decoder_post_720_v1.py')}
        front_mod = sys.modules.get('front_noise_native_720_v1') or load('front_noise_native_720_v1', root/'front_noise_native_720_v1.py')
        scope_mod = sys.modules.get('audit_front_decoder_post_720_v1') or load('audit_front_decoder_post_720_v1', root/'audit_front_decoder_post_720_v1.py')
        self.dispatch = front_mod.front_route_dispatch
        self.device = NS(type='xpu', index=0)
        self.torch = NS(Tensor=Tensor, float32='f32', float16='f16', int32='i32',
                        xpu=NS(is_current_stream_capturing=lambda: False),
                        empty=lambda *a, **k: (_ for _ in ()).throw(AssertionError('Unexpected full tensor allocation')))
        self.model = NS(_next_seed=12, _previous=Tensor(dtype='f16', device=self.device), noise=object(),
                        _controls=NS(style=0,intensity=1,local_tone=.5,local_structure=.75,auto_mask=False,skin_structure=None))
        self.graph = NS(model=self.model, closed=False, entries={}, replays=0, last_entry=None)
        forward = method(GAME.parents[0]/'experimental/fp8_unround_overlay/modules/graph_front_v1.py', 'GraphFront', 'forward')
        forward.__globals__.update(descriptor=lambda t: None if t is None else (t.shape,t.dtype,t.device.type,t.device.index),
                                   current_arithmetic_backend=lambda:'triton', record_arithmetic_dispatch=lambda x:None)
        self.graph.forward = MethodType(forward,self.graph)
        self.graph._validate = lambda: None
        self.model.sigmoid=self.model.blend_scale=object()
        self.model._forward_front=self.graph.forward
        self.session=NS(_stack=NS(model=self.model,graph=self.graph),_failed=False)
        self.modes=NS(session=self.session,_numeric_cleanup_owner=None)
        self.owner=FusedFront(); self.owner.noise=self.model.noise; self.owner.calls=0
        self.session._stack.components=[self.owner]
        self.front=object.__new__(front_mod.FrontNoiseCounters)
        self.observer=object.__new__(front_mod.common._Owned720)
        def sampler(): pass
        sampler.session=self.session
        for c in (self.front,self.observer):
            c.modes=self.modes; c.session=self.session; c.stack=self.session._stack; c.model=self.model
            c.graph=self.graph; c.device=self.device; c.torch=self.torch; c.front_owner=self.owner; c.noise=self.model.noise
            c._options=(('front_noise','native_both'),); c._in_frame=c._live=True; c._retired=False; c._front_event=None
            c.calls={}; c.capture_calls={}; c.actualkernelhash={}; c._guard=lambda **k: None; c.validate_frame_context=lambda: True
            c.modules={'fused_front':sys.modules[__name__], 'temporal':NS(warp_history_normalized=sampler)}
            c.known_callees=[(FusedFront,'apply',FusedFront.apply)]
            c.graph_forward=self.graph.forward
            c._history_codes={sampler.__code__:{'path':'CPU-owned-history'}}
            c.sources={'game_history':{'path':'CPU-fused-game'}}
        self.front.radius_native=self.front.trig_native=True
        self.front._history_components=None; self.front._component_calls=0
        self.front.tables=dict.fromkeys(('radius','sine','cosine'),object())
        self.front.triton=NS(cdiv=lambda n,b:(n+b-1)//b)
        kernel=NS(hash='CPU-stub-kernel',n_spills=0)
        fixture=self
        class Jit:
            def warmup(self,*args,**kwargs): return kernel
            def __getitem__(self,grid):
                def submit(*args,**kwargs):
                    fixture.kernel_args=args; fixture.launches+=1; return kernel
                return submit
        self.front.kernels=NS(front=Jit()); self.launches=0
        self.front._compiled={'front.history_components.seed_low':kernel}
        self.front.resources={'front.history_components.seed_low':{'actualbinary_sha256':'0'*64,'actualkernelhash':kernel.hash}}
        self.producer=object.__new__(scope_mod.Scope)
        p=self.producer; p.model=self.model; p.session=self.session; p.graph=self.graph; p.torch=self.torch
        p.post=NS(input_weight=Tensor(device=self.device))
        p.guard=p.validate_frame_context=lambda: True; p.authenticate=lambda m:m
        p.front_broker=None; p._front_broker_binding=None; p.front_record=None
        p.front_producer_calls=p.front_direct_writes=p.front_controls_consumed=0
        self.front.front_producer=p; self.model._audit_fdp_front_producer=p
        self.suite=NS(session=self.session,children={'front':self.front})
        self.modes._numeric_cleanup_owner=self.suite
        self.broker=graph_prepare.GraphDestinationBroker720(self.modes,self.front,p)
        p.bind_front_destination(self.broker)
        self.state=NS(graph=self.graph,armed=True,pending=None,consumed=None,frame=object(),frame_id=1,
                      entries=self.graph.entries,sealed={},require_frame=lambda:None)
        self.rgb=Tensor(device=self.device)
        self.num=Tensor(device=self.device); self.rcp=Tensor((720,1280),device=self.device)
        self.out=Tensor((768,1280,16),'f16',self.device)
        self.key=graph_prepare._key720(self.device,self.torch,temporal=True,raw=False)
        self.entry=NS(inputs={'rgb':Tensor(device=self.device),'front':self.out,'previous':self.num,'history_reciprocal':self.rcp},
                      replays=0,graph=NS(replay=lambda:None),output=Tensor(device=self.device))
        self.graph.entries[self.key]=self.entry
        self.state.sealed[self.key]=NS(entry=self.entry,current=lambda:True)
        self.fake_lifecycle=NS(state_for=lambda s:self.state)
        self.fake_rest=NS(front_reference=lambda c:None,local_after_guard=lambda c:None,launch_owner=lambda *a:False)


class SwitchTests(unittest.TestCase):
    def setUp(self):
        host._failed_mode_switch_720=None
        host._active_experiment='old'; host._active_optimizations=()
    def test_constructor_duplicate_keys_use_selected_override(self):
        calls=[]
        original=(host.DATA,host._mode_options,host.available_experiments_720,host.PROFILES)
        host.DATA=NS(__truediv__=lambda x,y:x)
        class PathData:
            def __truediv__(self,other): return self
            def read_text(self,**kwargs): return json.dumps({'profile_sha256':'CPU-pin'})
        host.DATA=PathData(); host._mode_options={'c128_post_pair':False,'c64_c128_attnproj_fused':False}
        host.available_experiments_720=lambda:('c512_k8',)
        host.PROFILES={'c512_k8':NS(mode_options=lambda:{'c128_post_pair':True,'c64_c128_attnproj_fused':True})}
        try:
            with modules({'nr_game_fullsize':NS(FullsizeGameModes=lambda *a,**k:calls.append(k) or object())}):
                host._new_modes('c512_k8')
            self.assertTrue(calls[0]['c128_post_pair']); self.assertTrue(calls[0]['c64_c128_attnproj_fused'])
        finally:
            host.DATA,host._mode_options,host.available_experiments_720,host.PROFILES=original
    def switch(self, failure=None):
        calls=[]; old=NS(session=object()); replacement=NS(session=None)
        def close():
            self.assertIs(host._modes,old); calls.append('close')
            if failure=='close': raise RuntimeError('close CPU failure')
            old.session=None; old._retired_assets_720=('closed-session','closed-graph')
        def inherit(value):
            self.assertIs(host._modes,old); self.assertIs(value,old); calls.append('inherit')
            if failure=='inherit': raise RuntimeError('inherit CPU failure')
            replacement._retired_assets_720=old._retired_assets_720; old._retired_assets_720=None
        old.close=close; replacement._inherit_retired_assets_720=inherit
        host._modes=old; previous=host._new_modes; host._new_modes=lambda *a:replacement
        try:
            if failure:
                with self.assertRaises(RuntimeError): host._replace_modes_720('next',('flag',))
                self.assertIs(host._modes,old); self.assertEqual(host._active_experiment,'old')
                self.assertEqual(host._failed_mode_switch_720,(old,replacement))
            else:
                host._replace_modes_720('next',('flag',)); self.assertEqual(calls,['close','inherit'])
                self.assertIs(host._modes,replacement); self.assertEqual(replacement._retired_assets_720,('closed-session','closed-graph'))
        finally: host._new_modes=previous
    def test_close_inherit_commit_order(self): self.switch()
    def test_close_failure_preserves_both_anchors(self): self.switch('close')
    def test_inherit_failure_preserves_old_proof(self): self.switch('inherit')


class GraphTests(unittest.TestCase):
    def test_real_sampler_front_dispatch_broker_and_copy_elimination(self):
        f=FrontFixture()
        with modules({'replay_lifecycle_audit_base_720_v1':f.fake_lifecycle,'replay_lifecycle_audit_rest_720_v1':f.fake_rest}):
            sample=method(GAME/'history_numeric_suite_720_v1.py','HistoryNumericCounter720','_sample_buffers')
            sample.__globals__.update(H=720,W=1280)
            child=NS(in_frame=True,graph=f.graph,model=f.model,modes=f.modes,device=f.device,torch=f.torch,
                     implementation=NS(reuse_buffers=True,drop_normalized=True),_buffer=lambda *a,**k:Tensor((720,1280),'i32',f.device))
            pair=sample(child,f.model._previous,False)
            self.assertIs(pair[0],f.num); self.assertIs(pair[1],f.rcp)
            bound=component.bind_front_component_consumer(f.modes,f.front,f.broker)
            with f.dispatch.observe(f.observer),f.dispatch.observe(f.front),f.dispatch.implementation(f.front,f.front._run):
                out=component.prepare_history_front(f.model,f.rgb,*pair[:2],
                    {'padded_size':(768,1280),'seed':12,'noise_source':f.model.noise},lambda *a,**k:self.fail('fallback'))
            self.assertIs(out,f.out); self.assertEqual(f.launches,1); self.assertIs(f.kernel_args[2],f.rcp)
            for c in (f.front,f.observer):
                self.assertEqual(c.calls['frontend.normalized_history'],1)
                self.assertFalse(c._front_event['normalized_history_tensor_materialized'])
            self.assertEqual(f.broker.submissions,1); self.assertEqual(f.front._component_calls,1)
            f.graph.forward(f.rgb,out,previous=f.num,history_reciprocal=f.rcp,sigmoid=f.model.sigmoid,blend_scale=f.model.blend_scale)
            self.assertEqual(f.entry.inputs['rgb'].copies,1)
            self.assertEqual([t.copies for t in (f.num,f.rcp,f.out)],[0,0,0])
            f.state.frame=object(); f.front._front_event={'frontend':'reset','seed':0}
            self.assertIsNone(f.broker.acquire_front_720(f.rgb,None,f.front._front_event))
            self.assertEqual(f.broker.retired_leases,1)
            METRICS['actual_front_peer_sha256']=f.peer_files
            METRICS['functional_route']='actual sampler buffer helper -> consumer -> two real observers -> one fake kernel submit -> actual Scope record -> broker -> GraphFront.forward identity skips three copies'
    def test_component_binding_rejects_second_executor(self):
        f=FrontFixture(); other=NS(model=f.model,session=f.session,options={'front_noise':'native_both'},apply_history_components=lambda *a:None)
        with self.assertRaises(RuntimeError): component.bind_front_component_consumer(f.modes,other)
    def test_broker_requires_replay_before_reuse_and_keeps_failure_lease(self):
        f=FrontFixture(); f.front._front_event={'frontend':'normalized_history','seed':12}
        with modules({'replay_lifecycle_audit_base_720_v1':f.fake_lifecycle}):
            out=f.broker.acquire_front_720(f.rgb,f.num,f.front._front_event)
            with self.assertRaises(RuntimeError): f.broker.acquire_front_720(f.rgb,f.num,f.front._front_event)
            f.front._front_event.update(front_kernel_hash='CPU',front_binary_sha256='0'*64,direct_static_destination=True)
            f.broker.commit_front_720(out,f.front._front_event)
            f.state.frame=object()
            with self.assertRaises(RuntimeError): f.broker.acquire_front_720(f.rgb,f.num,f.front._front_event)
            self.assertIs(f.broker.lease['output'],out); self.assertTrue(f.session._failed)
    def test_broker_rejects_alias_and_consumed_entry(self):
        f=FrontFixture(); event={'frontend':'normalized_history','seed':12}; f.front._front_event=event
        with modules({'replay_lifecycle_audit_base_720_v1':f.fake_lifecycle}):
            f.out.pointer=f.num.pointer
            with self.assertRaises(RuntimeError): f.broker.acquire_front_720(f.rgb,f.num,event)
            f.state.consumed=object()
            with self.assertRaises(RuntimeError): f.broker.acquire_front_720(f.rgb,f.num,event)


def packet(frame=1):
    p=hdr.NativeFrame720(abi_size=280,version=1,sr_input=300,source_tail_state=64,sr_input_state=8)
    f=p.frame; f.abi_size=256; f.device=10; f.queue=20; f.color=100; f.motion=200
    f.render_width=1280; f.render_height=720; f.hdr_input=1; f.frame_id=frame
    f.reset_history=int(frame==1); f.route=1; f.motion_scale_origin=1; f.motion_scale_x=1280; f.motion_scale_y=-720
    f.controls.abi_size=52; f.controls.input_height=720
    return p


def bridge_fixture():
    b=object.__new__(hdr.HDRTextureBridge720)
    b.handle=99; b._gpu_failed=False; b._gpu_exported=False; b._gpu_refs=[]
    b._hdr_packet=b._hdr_pending_packet=b._hdr_consumer=b._hdr_retired_consumer=None
    b._hdr_native_device=10; b._native_queue=20; b.width=1280; b.height=720
    b.gpu_handoff=b._producer_fence_required=True
    b._handoff_supported=b._owner_transfer_supported=b._ever_gpu_handoff=True
    b._hdr_rgb=b._hdr_motion=None; b._hdr_counts=dict(pack_submissions=0,export_submissions=0,retire_registrations=0,retire_polls=0)
    b._ready=lambda:None; b.idle=True; b.error_code=0; b.calls=[]
    b.empty=lambda ch:Tensor((720,1280,ch))
    def call(name,*args):
        b.calls.append((name,args))
        if name=='poll_idle': C.cast(args[0],C.POINTER(C.c_uint32))[0]=int(b.idle)
    def poll(handle,fence,value,idle,error,length):
        C.cast(idle,C.POINTER(C.c_uint32))[0]=int(b.idle or b._hdr_retired_consumer==(fence,value)); return b.error_code
    b._call=call; b.dll=NS(nr_texture_poll_retired_v1=poll)
    return b


def adapter_fixture(b):
    a=ModuleType('cyberpunk_nr_adapter'); a.host=host
    a._process_serial_lock=threading.RLock(); a._settings_lock=threading.RLock(); a._native_control_commit_epoch=0
    a._panel=NS(snapshot=lambda:NS(enabled=True,input_size=720,history_mode='fused',graph_replay=True,backend_variant='unrounded'))
    a.start_controls=lambda:None; a.bridge_protocol_v1=lambda:3
    exec('def process_native_frame_v1(packet):\n with _process_serial_lock:\n  return host.process_native_frame_v1(packet,game_adapter=__import__("sys").modules[__name__],serial_guard=_process_serial_lock)\n'
         'def poll_retired_native_v1(fence,value):\n with _process_serial_lock:\n  return host.poll_retired_native_v1(fence,value,game_adapter=__import__("sys").modules[__name__],serial_guard=_process_serial_lock)\n',a.__dict__)
    host._bridge=b; host._failed=False; host._failure_reason=None; host._error=lambda *a:None
    return a


class NativeTests(unittest.TestCase):
    def test_host_initializes_real_pair_before_arm_then_reuses_adapter_process(self):
        b=bridge_fixture(); a=adapter_fixture(None); events=[]
        previous=host._initialize
        def initialize(d,q,w,h,*,native720):
            self.assertEqual((d,q,w,h,native720),(10,20,1280,720,True)); events.append('initialize'); host._bridge=b
        def process(*args):
            self.assertTrue(a._process_serial_lock._is_owned())
            self.assertIsNotNone(b._hdr_pending_packet); events.append('adapter-process'); return (300,400,5)
        host._initialize=initialize; a.process=process
        try:
            with modules({'cyberpunk_nr_adapter':a}):
                self.assertEqual(a.process_native_frame_v1(packet()),(300,400,5))
            self.assertEqual(events,['initialize','adapter-process']); self.assertEqual(b._hdr_pending_packet.frame.device,10)
        finally: host._initialize=previous
    def test_pending_packet_survives_generic_idle_and_two_exact_consumers(self):
        b=bridge_fixture(); a=adapter_fixture(b)
        with modules({'cyberpunk_nr_adapter':a}):
            for i,pair in ((1,(400,5)),(2,(500,8))):
                b._hdr_pending_packet=packet(i)
                self.assertTrue(b.poll_idle()); self.assertIsNotNone(b._hdr_pending_packet)
                b.prepare(texture.SourceFrame(100,200,1280,720,i,i-1,i==1,producer_fence=600,producer_value=i))
                b._gpu_exported=True; b.retire(*pair)
                if i==2:
                    refs=list(b._gpu_refs)
                    self.assertEqual(a.poll_retired_native_v1(400,5),1)
                    self.assertEqual(b._gpu_refs,refs); self.assertEqual(b._hdr_consumer,pair)
                with self.assertRaises(ValueError): b.retire(*pair)
                b.idle=False; refs=list(b._gpu_refs)
                self.assertEqual(a.poll_retired_native_v1(*pair),0); self.assertEqual(b._gpu_refs,refs)
                self.assertIsNotNone(b._hdr_packet)
                b.idle=True; self.assertEqual(a.poll_retired_native_v1(*pair),1)
                self.assertEqual(a.poll_retired_native_v1(*pair),1)
                self.assertIsNone(b._hdr_consumer); self.assertEqual(b._gpu_refs,[])
            self.assertEqual(b._hdr_counts['retire_registrations'],2)
    def test_actual_owner_transfer_busy_keeps_anchors_success_preserves_pending_packet(self):
        b=bridge_fixture(); a=adapter_fixture(b)
        b.thread=77; b.native_thread=88; b.stream=NS(sycl_queue=900)
        b.torch=NS(xpu=NS(current_stream=lambda:b.stream)); b._hdr_packet=packet()
        b._hdr_pending_packet=packet(2); b._hdr_consumer=(400,5); b._gpu_refs=[object()]; b._gpu_exported=True
        proof=[0]
        def transfer(handle,tid,sycl,queue,adopted,error,length):
            self.assertEqual((handle,tid,sycl,queue),(99,88,900,20))
            C.cast(adopted,C.POINTER(C.c_uint32))[0]=proof[0]; return 0
        b.dll.nr_texture_transfer_owner=transfer
        exec('def process(previous,timeout):\n with _process_serial_lock:\n  host._bridge.transfer_serial_thread(game_adapter=__import__("sys").modules[__name__],serial_guard=_process_serial_lock,previous_thread=previous,timeout=timeout)\n',a.__dict__)
        with modules({'cyberpunk_nr_adapter':a}):
            with self.assertRaises(RuntimeError): a.process(77,.001)
            self.assertEqual(b.thread,77); self.assertEqual(len(b._gpu_refs),1); self.assertIsNotNone(b._hdr_packet)
            proof[0]=1; a.process(77,.01)
            self.assertEqual(b.thread,threading.get_ident()); self.assertEqual(b._gpu_refs,[])
            self.assertEqual(b._hdr_retired_consumer,(400,5)); self.assertIsNone(b._hdr_consumer)
            self.assertIsNone(b._hdr_packet); self.assertEqual(b._hdr_pending_packet.frame.frame_id,2)
    def test_nonblocking_poll_cross_thread_and_error_keeps_anchors(self):
        b=bridge_fixture(); a=adapter_fixture(b); b._hdr_packet=packet(); b._gpu_refs=[object()]
        b._gpu_exported=True; b.retire(400,5); b.idle=False
        with modules({'cyberpunk_nr_adapter':a}):
            out=[]; thread=threading.Thread(target=lambda:out.append(a.poll_retired_native_v1(400,5)))
            thread.start(); thread.join(); self.assertEqual(out,[0])
            b.error_code=1; self.assertEqual(a.poll_retired_native_v1(400,5),2)
            self.assertTrue(b._gpu_failed); self.assertEqual(len(b._gpu_refs),1); self.assertIsNotNone(b._hdr_packet)
    def test_qualified_legacy_rgba_is_rejected_before_initialize(self):
        a=adapter_fixture(None)
        with modules({'cyberpunk_nr_adapter':a,'nr_gpu_handoff_host_v1':NS(invalidate_after_failure=lambda:None)}):
            with self.assertRaisesRegex(RuntimeError,'legacy RGBA'): host.process(10,20,100,200,1,1,1280,720)
        self.assertIsNone(host._bridge)
    def test_packet_shape_motion_and_states_are_preserved(self):
        self.assertEqual([C.sizeof(c) for c in (hdr.Controls720,hdr.Frame720,hdr.NativeFrame720,hdr.SharedLayout720)],[52,256,280,120])
        p=packet(); self.assertEqual(hdr.validate_packet720(p,10,20).motion_scale_y,-720)
        p.frame.motion_scale_y=0
        with self.assertRaises(ValueError): hdr.validate_packet720(p,10,20)


class FlashTests(unittest.TestCase):
    def test_incremental_cursor_failure_loss_bound_and_no_full_v2_read(self):
        sampler=receipts.IncrementalFlashSampler720(); cursors=[]; mode=['ok']
        def delta(ptr):
            row=C.cast(ptr,C.POINTER(protocol.FlashDeltaV3)).contents
            cursors.append((row.after_event,row.after_frame))
            if mode[0]=='busy': return 0
            row.next_event=row.event_head=row.after_event+1; row.next_frame=row.frame_head=row.after_frame+1
            row.events_count=row.frames_count=1
            row.events_lost=2 if mode[0]=='loss' else 0
            e=row.events[0]; e.serial=row.next_event; e.context.sr_sequence=e.serial; e.context.nr_frame_id=e.serial
            f=row.frames[0]; f.serial=row.next_frame; f.context.sr_sequence=f.serial; f.context.nr_frame_id=f.serial
            return 1
        native=NS(NRB_GetPeriodicFlashDeltaV3=delta,
                  NRB_GetPeriodicFlashDiagV2=lambda *a:self.fail('Full V2 ring must not be read'))
        py=dict(frames=[],events=[])
        self.assertEqual(sampler.sample(native,py)['transport_version'],3)
        mode[0]='busy'; self.assertEqual(sampler.sample(native,py)['status'],'unavailable')
        mode[0]='loss'; result=sampler.sample(native,py)
        self.assertEqual(cursors[-2:],[(1,1),(1,1)]); self.assertEqual(result['native_header']['events_lost'],2)
        mode[0]='ok'
        for _ in range(70): result=sampler.sample(native,py)
        self.assertLessEqual(len(result['windows']),8); self.assertEqual(result['retained_native_frames'],64)
        self.assertEqual(result['legacy_full_ring_reads'],0)


class ColdRawTests(unittest.TestCase):
    def test_two_distinct_reset_history_tokens_and_failure_anchor(self):
        lock=threading.RLock(); model=NS(_controls=NS(style=0,intensity=.25)); graph=NS(closed=False,entries={},replays=0,last_entry=None)
        session=NS(_stack=NS(model=model,graph=graph),_failed=False)
        modes=NS(session=session,height=720,source=(720,1280),variant='unrounded',_numeric_cleanup_owner=object())
        h=NS(_modes=modes,_failed=False); a=NS(host=h,_process_serial_lock=lock,_settings_lock=threading.RLock())
        state=NS(frame=None,frame_id=None,pending=None,consumed=None,entries=graph.entries,require_frame=lambda:None)
        with modules({'cyberpunk_nr_adapter':a,'nr_game_pre_xess_host':h,
                      'replay_lifecycle_audit_base_720_v1':NS(state_for=lambda s:state)}),lock:
            with raw_control.cold_raw_control_frame(modes,game_adapter=a,serial_guard=lock) as token:
                for reset in (True,False):
                    state.frame=object(); state.frame_id=int(not reset)+1; state.consumed=None
                    with token.frame(reset=reset):
                        token.require(model,graph)
                        key=(None,None,None if reset else 'history',None,True)
                        entry=NS(replays=1); seal=NS(entry=entry,current=lambda:True)
                        state.entries[key]=entry; graph.last_entry=entry
                        before=graph.replays; graph.replays+=1; state.consumed=(key,seal,0,before)
                with self.assertRaises(RuntimeError):
                    with token.frame(reset=False): pass
            self.assertFalse(session._failed); self.assertFalse(token.active)
            with self.assertRaisesRegex(RuntimeError,'two successful'):
                with raw_control.cold_raw_control_frame(modes,game_adapter=a,serial_guard=lock): pass
            self.assertTrue(session._failed); self.assertIs(modes._audit_raw_graph_failed_anchor_720[1],session)


if __name__ == '__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    imported=[n for n in sys.modules if n.split('.')[0] in ('torch','triton','numpy')]
    report=dict(schema='history-host-graph-stage3-cpu-v1',passed=result.wasSuccessful() and not imported,
                tests_run=result.testsRun,metrics=METRICS,imported_device_modules=imported,
                native_protocol_source_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                GPU_executed=False,DLL_loaded=False,qualified_protocol_flags=0,
                limitations='Fake tensor/native export/kernel completions; actual Python route proof only, no GPU performance or acceptance')
    if len(sys.argv)>1:
        target=Path(sys.argv[1]); target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    sys.exit(0 if report['passed'] else 1)
