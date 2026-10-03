"""Owned fixed720 entry. Importing this file executes stdlib only.

before_numeric installs a scope without preflight. Main owns the one graph
signature, NumericCleanup, deferred preflight and process-lock transaction.
Numeric-only delegates to the existing real-shape numeric suite. Complete
profiles replace selected instance/callable roles and retain baseline controls.
"""
from __future__ import annotations
from contextlib import contextmanager
from threading import RLock, get_ident
from types import FunctionType
from pathlib import Path
from hashlib import sha256
import sys

NUMERIC_ARGUMENTS = dict(vit_qkv_full_k=True, vit_projection_full_k=True,
    vit_exp_zero_constant=True, vit_denominator="ordered_fused",
    vit_norm_fma=True, vit_exp_fma=True)


def bind_serial_owner(child,owner,transfer_method):
    """Cold bind to the main's exact HookCounter transfer code and source."""
    module=sys.modules.get("implementation_hooks_720_v1")
    owner_type=getattr(module,"HookCounter",None)
    function=getattr(transfer_method,"__func__",None)
    if (type(owner) is not owner_type or function is not owner_type.transfer_serial_thread or
            transfer_method.__self__ is not owner or owner.modes is not child.modes or
            owner.session is not child.session or owner.graph is not child.graph or
            owner.children.get("audit_vit_c512_720_v1") is not child or
            getattr(child,"_serial_owner",None) is not None or
            getattr(child,"_in_frame",False) or child.graph.entries):
        raise RuntimeError("Invalid cold implementation transfer binding")
    path=Path(module.__file__).resolve()
    child._serial_owner=(owner,function,function.__code__,module,path,sha256(path.read_bytes()).hexdigest())


def require_serial_transaction(child, owner, serial_guard, previous_thread):
    """Only the registered HookCounter, nested in Numeric's real transaction."""
    registered=getattr(child,"_serial_owner",None)
    module=sys.modules.get("implementation_hooks_720_v1")
    caller = sys._getframe(2)
    numeric=sys.modules.get("numeric_cleanup_suite_720_v1")
    suite_type=getattr(numeric,"NumericCleanupCounter",None)
    outer=sys._getframe(3)
    numeric_owner=caller.f_locals.get("numeric_owner")
    if (registered is None or registered[0] is not owner or module is not registered[3] or
            type(owner) is not getattr(module,"HookCounter",None) or not owner.active or owner.retired or
            owner_type_action_changed(owner,registered) or
            caller.f_code is not registered[2] or caller.f_globals is not module.__dict__ or
            caller.f_locals.get("self") is not owner or
            caller.f_locals.get("serial_guard") is not serial_guard or
            caller.f_locals.get("previous_thread") != previous_thread or
            type(numeric_owner) is not suite_type or
            outer.f_code is not suite_type.transfer_serial_thread.__code__ or
            outer.f_globals is not numeric.__dict__ or outer.f_locals.get("self") is not numeric_owner or
            sha256(registered[4].read_bytes()).hexdigest()!=registered[5]):
        raise RuntimeError("Hook transfer requires the registered nested process transaction")
    adapter = sys.modules.get("cyberpunk_nr_adapter")
    host = sys.modules.get("nr_game_pre_xess_host")
    owners = getattr(child.modes, "implementation_calls_720", {})
    membership = any(getattr(c, "active", False) and not getattr(c, "retired", True) and
        c.session is child.session and c.graph is child.graph and
        c.children.get("audit_vit_c512_720_v1") is child for c in owners.values())
    if (adapter is None or host is None or getattr(adapter, "host", None) is not host or
            type(serial_guard) is not type(RLock()) or
            serial_guard is not getattr(adapter, "_process_serial_lock", None) or
            serial_guard is getattr(adapter, "_settings_lock", None) or
            not serial_guard._is_owned() or type(previous_thread) is not int or
            previous_thread != child.thread or not membership or numeric_owner.closed or
            getattr(child.modes, "_numeric_cleanup_owner", None) is not numeric_owner or
            owner.modes is not child.modes or owner.session is not child.session or
            owner.graph is not child.graph or numeric_owner.stack is not child.session._stack or
            getattr(host, "_modes", None) is not child.modes or getattr(host, "_failed", False) or
            child.session._closed or getattr(child, "_in_frame", False) or
            getattr(child, "_frame_hooks", None)):
        raise RuntimeError("Hook transfer lost the idle live owner or held process lock")
    if (getattr(numeric_owner, "_game_serial_adapter", None) not in (None, adapter) or
            getattr(numeric_owner, "_game_serial_guard", None) not in (None, serial_guard)):
        raise RuntimeError("Hook process lock/adapter identity changed")
    numeric_owner._validate_suite_owner()
    torch = sys.modules.get("torch")
    bridge = getattr(host, "_bridge", None)
    current = get_ident()
    if (torch is None or bridge is None or torch.xpu.is_current_stream_capturing() or
            host._thread != current or bridge.thread != current or bridge.torch is not torch or
            bridge.device != torch.device("xpu", torch.xpu.current_device()) or
            torch.xpu.current_stream().sycl_queue != bridge.stream.sycl_queue):
        raise RuntimeError("Hook transfer requires the already-bound host/bridge stream")
    return current


def owner_type_action_changed(owner,registered):
    return (type(owner).transfer_serial_thread is not registered[1] or
            registered[1].__code__ is not registered[2])


@contextmanager
def installed(modes, *, profile="numeric", before_numeric=False,
              denominator="ordered_fused", vit_qkv_full_k=True,
              vit_projection_full_k=True, vit_exp_zero_constant=True,
              vit_norm_fma=True, vit_exp_fma=True, **complete_options):
    if (getattr(modes, "height", None) != 720 or
            getattr(modes, "variant", None) != "unrounded" or
            not getattr(modes, "controlled", False) or
            not getattr(modes, "c512_qkv_library_720", False) or
            not getattr(modes, "native_k8_720", False)):
        raise RuntimeError("Select the accepted current720 controlled C512+K8 stack")
    for name,value in (("vit_qkv_full_k",vit_qkv_full_k),
                      ("vit_projection_full_k",vit_projection_full_k),
                      ("vit_exp_zero_constant",vit_exp_zero_constant),
                      ("vit_norm_fma",vit_norm_fma),("vit_exp_fma",vit_exp_fma)):
        if type(value) is not bool:raise TypeError(name)
    arguments = dict(vit_qkv_full_k=vit_qkv_full_k,
        vit_projection_full_k=vit_projection_full_k,
        vit_exp_zero_constant=vit_exp_zero_constant,vit_denominator=denominator,
        vit_norm_fma=vit_norm_fma,vit_exp_fma=vit_exp_fma)
    if profile == "complete":
        global_owned=complete_options.get("global_dataflow",False)
        if (denominator not in ("reference","ordered_fused","fp32_reduction") or
                global_owned and (denominator=="reference" or not vit_exp_zero_constant)):
            raise ValueError("Complete dataflow requires its verified folded denominator")
        from audit_vit_c512_scope_720_v1 import installed as complete_installed
        with complete_installed(modes,denominator=denominator,
                full_qkv=vit_qkv_full_k,full_projection=vit_projection_full_k,
                native_norm=vit_norm_fma,native_exp=vit_exp_fma,**complete_options) as counter:
            yield counter
        return
    if profile != "numeric" or complete_options:
        raise ValueError("Expected numeric or complete with owned, explicit options")
    if denominator not in ("reference","ordered_fused","fp32_reduction"):
        raise ValueError("Unknown numeric denominator")
    if before_numeric:
        hook=NumericStageHook(modes,arguments)
        yield hook
        return
    from vit_numeric_suite_720_v1 import installed as numeric_installed
    with numeric_installed(modes,**arguments) as counter:yield counter


class NumericStageHook:
    """Before-numeric adapter; the actual numeric child remains main-owned."""
    def __init__(self,modes,arguments):
        self.modes=modes;self.session=modes.session
        self.graph=self.session._stack.graph;self.thread=get_ident()
        self.arguments=dict(arguments)
        self.identity=("audit_vit_c512_720_v1","numeric",tuple(self.arguments.items()))
        self._serial_owner=None

    def bind_serial_owner(self,owner,*,transfer_method):
        bind_serial_owner(self,owner,transfer_method)

    def validate(self):
        if self.modes.session is not self.session or get_ident()!=self.thread:
            raise RuntimeError("Numeric hook owner/thread changed outside its transaction")
        self.session._ready()
        child=getattr(self.session,"_vit_numeric_suite_720",None)
        if child is not None:
            if any(getattr(child.mode,k)!=v for k,v in self.arguments.items()):
                raise RuntimeError("Main numeric arguments differ from the selected hook")
            child.validate()

    def validate_frame_context(self):return self.validate()
    def _validate_replay(self):return self.validate()

    def preflight(self):
        self.validate()
        child=getattr(self.session,"_vit_numeric_suite_720",None)
        if child is None:raise RuntimeError("Install NumericCleanup before deferred preflight")
        return child.preflight()

    def snapshot(self):
        child=getattr(self.session,"_vit_numeric_suite_720",None)
        return dict(identity=self.identity,selected=dict(self.arguments),
            owner="main NumericCleanup ViT child",
            child=None if child is None else child.snapshot(),GPU_tested=False)

    def _transfer_serial_thread(self,owner,*,serial_guard,previous_thread):
        current=require_serial_transaction(self,owner,serial_guard,previous_thread)
        self.thread=current
        # The suite expects an integer and rolls back every child's thread on failure.
        return current
