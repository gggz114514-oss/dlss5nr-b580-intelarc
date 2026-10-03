"""Owned, cold-installed fixed720 module hooks for the implementation bundle.

Only explicitly selected local scopes execute. Install before numerical child
scopes freeze their parent/consumer interfaces, and retire after those scopes
destroy captured graphs. This registry itself imports no tensor/GPU runtime.
"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import json
import sys
from threading import RLock, get_ident
from pathlib import Path
from types import MappingProxyType

ROOT = Path(__file__).parent
ALLOWED = frozenset({
    'audit_swin_720_v1', 'audit_vit_c512_720_v1',
    'audit_front_decoder_post_720_v1', 'audit_history_host_720_v1',
})

@dataclass(frozen=True)
class HookSpec:
    module: str
    encoded_kwargs: str
    stage: str = 'before_numeric'

    @property
    def kwargs(self):
        return json.loads(self.encoded_kwargs)

    def to_dict(self):
        return {'module':self.module,'kwargs':self.kwargs,'stage':self.stage,'call':'installed'}

def parse_hooks(value=()):
    if not isinstance(value,(list,tuple)):
        raise TypeError('Implementation hooks must be a sequence')
    result=[]
    for item in value:
        if not isinstance(item,dict) or set(item)-{'module','call','kwargs','stage'}:
            raise ValueError('Unknown implementation hook fields')
        if item.get('module') not in ALLOWED or item.get('call','installed')!='installed':
            raise ValueError('Only the fixed local audit scopes can be installed')
        stage=item.get('stage','before_numeric')
        if stage not in ('before_numeric','after_numeric'):
            raise ValueError('Unknown implementation hook stage')
        kwargs=item.get('kwargs',{})
        if not isinstance(kwargs,dict) or any(type(k) is not str for k in kwargs):
            raise TypeError('Hook kwargs must be a JSON object')
        encoded=json.dumps(kwargs,sort_keys=True,separators=(',',':'),allow_nan=False)
        # Roundtrip excludes arbitrary callback/object references from options.
        if json.loads(encoded)!=kwargs:
            raise ValueError('Hook kwargs must have stable JSON semantics')
        result.append(HookSpec(item['module'],encoded,stage))
    if len({h.module for h in result})!=len(result):
        raise ValueError('A scope must be installed once, with its combined options')
    return tuple(result)

def hooks_identity(hooks):
    value=json.dumps([h.to_dict() for h in hooks],sort_keys=True,separators=(',',':'))
    return 'implementation-hooks-720-v1:'+hashlib.sha256(value.encode()).hexdigest()

def _composed_front_pins():
    """Cold, frozen composition of shared dependencies, never owned kernels.

    Main writes this receipt after combining the independent worker sources.
    A missing receipt admits no changed dependency; it does not disable any
    scope guard. The GPU runner pins the entire final source tree separately.
    """
    path=ROOT/'IMPLEMENTATION_COMPOSED_PINS_720.json'
    if not path.exists():
        return {}
    data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema')!='current720-reviewed-composed-pins-v1':
        raise RuntimeError('Unknown composed dependency receipt')
    result={}
    for name,row in data.get('front_dependencies',{}).items():
        if name.startswith('audit_front_decoder_post_'):
            raise RuntimeError('A composed receipt cannot replace Front owned kernels')
        rel=Path(row['relative_path'])
        target=(ROOT.parent/rel).absolute()
        if rel.is_absolute() or '..' in rel.parts:
            raise RuntimeError('Composed dependency path escaped its frozen source')
        target.relative_to(ROOT.parent.absolute())
        if (not target.is_file() or
                hashlib.sha256(target.read_bytes()).hexdigest()!=row['sha256']):
            raise RuntimeError('Reviewed composed dependency changed: '+name)
        result[name]=row['sha256']
    return result

def _load(spec):
    module=importlib.import_module(spec.module)
    path=Path(module.__file__).absolute()
    expected=(ROOT/(spec.module+'.py')).absolute()
    if path!=expected or not path.is_file():
        raise RuntimeError('Implementation scope loaded outside its selected source tree')
    if not callable(getattr(module,'installed',None)):
        raise TypeError('Implementation scope lacks installed()')
    return module, {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

class HookCounter:
    def __init__(self,modes,hooks,stage):
        self.modes=modes
        self.hooks=hooks
        self.stage=stage
        self.session=modes.session
        self.graph=self.session._stack.graph
        self.identity=hooks_identity(hooks)
        self.children={}
        self.sources={}
        self.preflights={}
        self.child_interfaces={}
        self.active=True
        self.retired=False

    def _owner(self):
        if (not self.active or self.retired or self.modes.session is not self.session or
                self.session._stack.graph is not self.graph or self.graph.closed or
                self.modes.implementation_hooks_720!=self.hooks or
                self.modes.height!=720 or self.modes.source!=(720,1280) or
                self.modes.variant!='unrounded'):
            self.session._failed=True
            raise RuntimeError('Fixed720 implementation owner/geometry/options changed')

    def preflight(self):
        self._owner()
        for label,child in self.children.items():
            if label not in self.preflights:
                action=getattr(child,'preflight',None)
                if callable(action):
                    self.preflights[label]=action()
                elif getattr(child,'preflight_complete',False):
                    self.preflights[label]={'scope_performed_preflight':True}
                else:
                    raise TypeError(f'{label} must provide its actual preflight receipt')
        return dict(self.preflights)

    def register_child(self,name,child):
        self.children[name]=child
        self.child_interfaces[name]=(type(child), {
            field:(getattr(getattr(child,field,None),'__func__',getattr(child,field,None)),
                   getattr(getattr(child,field,None),'__code__',None))
            for field in ('validate_frame_context','validate','_transfer_serial_thread','transfer_serial_thread')})
        binder=getattr(child,'bind_serial_owner',None)
        if callable(binder):
            binder(self,transfer_method=self.transfer_serial_thread)

    def serial_participants(self,previous_thread):
        self._owner()
        selected={h.module for h in self.hooks if h.stage==self.stage}
        if set(self.children)!=selected or set(self.child_interfaces)!=selected:
            raise RuntimeError('Implementation serial transaction lost its selected scope')
        participants=[]
        for name,child in self.children.items():
            exact,interfaces=self.child_interfaces[name]
            if type(child) is not exact:
                raise RuntimeError('Implementation counter class changed after admission')
            for field,(function,code) in interfaces.items():
                actual=getattr(child,field,None)
                if (getattr(actual,'__func__',actual) is not function or
                        getattr(actual,'__code__',None) is not code):
                    raise RuntimeError('Implementation serial/validation callable changed')
            if not hasattr(child,'thread'):
                continue
            if (type(child.thread) is not int or child.thread!=previous_thread or
                    child.modes is not self.modes or child.session is not self.session or
                    any(getattr(child,flag,False) for flag in ('in_frame','_in_frame','_in_dispatch'))):
                raise RuntimeError('Implementation scope is not idle on its previous CPU owner')
            direct=(name=='audit_swin_720_v1' and callable(getattr(child,'transfer_serial_thread',None)))
            if not direct and not callable(getattr(child,'_transfer_serial_thread',None)):
                raise RuntimeError('Implementation scope has no explicit serial handoff')
            participants.append((name,child,direct))
        return participants

    def transfer_serial_thread(self,*,numeric_owner,game_adapter,serial_guard,previous_thread):
        """Participant in the existing, atomic numeric process-lock transaction.

        The bridge and numerical children have already moved. A failure rolls
        every numeric and implementation CPU owner back in the outer owner.
        """
        numeric=sys.modules.get('numeric_cleanup_suite_720_v1')
        caller=sys._getframe(1)
        host=getattr(game_adapter,'host',None)
        if (numeric is None or type(numeric_owner) is not numeric.NumericCleanupCounter or
                caller.f_code is not numeric.NumericCleanupCounter.transfer_serial_thread.__code__ or
                caller.f_locals.get('self') is not numeric_owner or
                numeric_owner.modes is not self.modes or numeric_owner.session is not self.session or
                numeric_owner.graph is not self.graph or
                game_adapter is not sys.modules.get('cyberpunk_nr_adapter') or
                host is not sys.modules.get('nr_game_pre_xess_host') or
                getattr(host,'_modes',None) is not self.modes or getattr(host,'_failed',True) or
                type(serial_guard) is not type(RLock()) or
                serial_guard is not getattr(game_adapter,'_process_serial_lock',None) or
                serial_guard is getattr(game_adapter,'_settings_lock',None) or not serial_guard._is_owned()):
            raise RuntimeError('Implementation transfer requires the actual held numeric process transaction')
        for _,child,direct in self.serial_participants(previous_thread):
            if direct:
                # Its explicit interface authenticates the outer numeric code.
                continue
            result=child._transfer_serial_thread(self,serial_guard=serial_guard,previous_thread=previous_thread)
            if isinstance(result,dict):
                result=result.get('current_thread')
            if result is not None and result!=get_ident():
                raise RuntimeError('Implementation child returned another CPU owner')
        return get_ident()

    def validate_frame_context(self):
        self._owner()
        for child in self.children.values():
            action=getattr(child,'validate_frame_context',None)
            if action is None:
                action=getattr(child,'validate',None)
            if not callable(action):
                raise TypeError('An implementation child lacks a replay-owner interface')
            action()

    def snapshot(self):
        return {'identity':self.identity,'stage':self.stage,'active':self.active,
                'retired':self.retired,'sources':dict(self.sources),
                'preflights':dict(self.preflights),'children':{
                    name:child.snapshot() for name,child in self.children.items()}}

@contextmanager
def installed(modes,hooks,*,stage='before_numeric',loader=None,preflight=True):
    hooks=tuple(hooks)
    selected=tuple(h for h in hooks if h.stage==stage)
    session=getattr(modes,'session',None)
    if (session is None or modes.height!=720 or modes.source!=(720,1280) or
            modes.variant!='unrounded' or not modes.c512_qkv_library_720 or
            not modes.native_k8_720 or not modes.controlled or
            session._stack.graph.closed or session._stack.graph.entries):
        raise RuntimeError('Implementation scopes require a fresh current720 C512+K8 session')
    if modes.implementation_hooks_720!=hooks:
        raise RuntimeError('Implementation hook options do not match the session')
    counter=HookCounter(modes,hooks,stage)
    load=_load if loader is None else loader
    with ExitStack() as stack:
        try:
            for spec in selected:
                module,source=load(spec)
                kwargs=spec.kwargs
                if spec.module=='audit_front_decoder_post_720_v1':
                    # Only the main's verified local transaction source is
                    # injected; owned kernel hashes remain immutable.
                    kwargs=dict(kwargs)
                    pins=dict(kwargs.get('source_pins',{}))
                    pins.update(_composed_front_pins())
                    pins[__name__]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                    kwargs['source_pins']=pins
                child=stack.enter_context(module.installed(modes,**kwargs))
                if child is None or not callable(getattr(child,'snapshot',None)):
                    raise TypeError('Implementation scope must return a live execution receipt')
                counter.register_child(spec.module,child)
                counter.sources[spec.module]=source
            if preflight:
                counter.preflight()
            yield counter
        except BaseException:
            session._failed=True
            raise
        finally:
            # Capture/consumer destruction belongs to the outer session. A
            # failed unwind deliberately preserves its anchor until close.
            counter.active=False
            counter.retired=True

def health(modes):
    owners=getattr(modes,'implementation_calls_720',{})
    return {'identity':getattr(modes,'implementation_identity_720',None),
            'selected_scopes':[s.module for s in getattr(modes,'implementation_hooks_720',())],
            'installed_stages':list(owners),
            'active':bool(owners) and all(c.active and not c.retired for c in owners.values()),
            'scope_hits':{name:dict(getattr(child,'calls',{}))
                          for counter in owners.values() for name,child in counter.children.items()}}
