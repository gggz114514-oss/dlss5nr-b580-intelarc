"""Owned independent/combined current720 roles and complete consumer scope.

Main owns NumericCleanup, the unique hooks_identity signature, process transfer
and graph retirement. Existing counted FFN callables remain installed: a narrow
rows-scopes delegate substitutes their actual kernels, preserving real counters.
No class/model forward or shared registry is permanently changed.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass,asdict
from types import MethodType
from threading import get_ident
from pathlib import Path
from hashlib import sha256
import importlib
import json
import sys
import torch
import triton
from quantization_dataflow_v1 import CONTRACTS
from nr_backend.execution import current_arithmetic_backend,record_arithmetic_dispatch
from nr_backend.unround_policy import ENABLED,FAMILIES
from audit_c512_maps_720_v1 import build as build_maps,SHIFTS
from audit_vit_c512_720_v1 import bind_serial_owner,require_serial_transaction
import audit_vit_native_dataflow_720_v1 as global_vit
import audit_c512_query_encoder_720_v1 as local
import audit_int8_ffn_complete_720_v1 as ffn
import c512_int8_ffn_rows_v1 as old_c512
import int8_ffn_segment_rows_v1 as old_vit
import rows_scopes_v1 as rows
from c512_window_projection_v1 import _project as decoder_leaf

SCOPE_MODULE="audit_vit_c512_720_v1"
LAUNCH_OPTIONS=dict(num_warps=4,num_stages=1,enable_fp_fusion=False)
DECODER_OPTIONS=dict(num_warps=4,enable_fp_fusion=False)


@dataclass(frozen=True)
class Options:
    c512_qkv_backend:str="library"
    global_dataflow:bool=False
    local_valid_queries:bool=False
    encoder_direct_projection:bool=False
    encoder_pool_final:bool=False
    use_front_pool_helper:bool=False
    c512_ffn_groups:bool=False
    vit_ffn:bool=False
    denominator:str="ordered_fused"
    full_qkv:bool=True
    full_projection:bool=True
    native_norm:bool=True
    native_exp:bool=True
    vit_fuse_entry:bool=False
    vit_contract_parts:int=1
    diagnostics:bool=False
    bm:int=32
    bn:int=64

    def __post_init__(self):
        for name,value in asdict(self).items():
            if name not in ("denominator","c512_qkv_backend","vit_contract_parts","bm","bn") and type(value) is not bool:
                raise TypeError(name)
        if (self.denominator not in ("reference","ordered_fused","fp32_reduction") or
                self.global_dataflow and self.denominator=="reference"):
            raise ValueError("denominator")
        if self.c512_qkv_backend not in ("library","native_dpas"):raise ValueError("C512 QKV backend")
        if (type(self.bm) is not int or type(self.bn) is not int or
                self.bm not in (16,32) or self.bn not in (32,64)):raise ValueError("tile")
        if type(self.vit_contract_parts) is not int or self.vit_contract_parts not in (1,4):
            raise ValueError("INT32 contract parts")
        if not any((self.c512_qkv_backend=="native_dpas",self.global_dataflow,self.local_valid_queries,self.encoder_direct_projection,
                    self.encoder_pool_final,self.c512_ffn_groups,self.vit_ffn)):
            raise ValueError("Select at least one actual complete720 role")
        if self.vit_fuse_entry and not self.vit_ffn:raise ValueError("Fused entry requires ViT FFN")
        if self.use_front_pool_helper and not self.encoder_pool_final:
            raise ValueError("Front pool helper needs the owned encoder consumer")

    @property
    def encoder_tail(self):
        return self.local_valid_queries or self.encoder_direct_projection or self.encoder_pool_final

    def active_sites(self):
        result=[]
        if self.c512_qkv_backend=="native_dpas":
            result.extend(f"qkv.{family}.{i}" for family in ("encoder512","decoder512") for i in range(8))
        for family,enabled in (("encoder512",self.encoder_tail),
                               ("decoder512",self.local_valid_queries),("vit",self.global_dataflow)):
            if enabled:
                indices=self.encoder_indices if family=="encoder512" else range(8)
                result.extend(f"{family}.{i}" for i in indices)
        if self.c512_ffn_groups:
            result.extend(f"ffn.{family}.{i}" for family in ("encoder512","decoder512") for i in range(8))
        if self.vit_ffn:result.extend(f"ffn.vit.{i}" for i in range(8))
        if self.use_front_pool_helper:result.append("front_pool.down:c512")
        return tuple(result)

    @property
    def encoder_indices(self):
        return tuple(range(8)) if self.local_valid_queries or self.encoder_direct_projection else ((7,) if self.encoder_pool_final else ())


def stamp(t):
    return (id(t),t.data_ptr(),tuple(t.shape),tuple(t.stride()),str(t.dtype),str(t.device),t._version)


def argument_shapes(args):
    return tuple(("tensor",tuple(x.shape),tuple(x.stride()),str(x.dtype),str(x.device),x.storage_offset())
                 if isinstance(x,torch.Tensor) else (type(x).__name__,x) for x in args)


def scope_contains(current,target):
    seen=set()
    for _ in range(64):
        if current is target:return True
        if id(current) in seen:return False
        seen.add(id(current))
        parent=getattr(current,"__nr_numeric_original_installed__",None)
        if parent is None:
            other=getattr(current,"__wrapped__",None)
            if other is not current:parent=other
        if parent is None:
            closure={}
            for name,cell in zip(getattr(getattr(current,"__code__",None),"co_freevars",()),
                                 getattr(current,"__closure__",()) or ()):
                try:closure[name]=cell.cell_contents
                except ValueError:pass
            parent=next((closure[n] for n in ("previous_scope","original_scope","previous_installed",
                                             "original_installed") if n in closure),None)
        if parent is None:return False
        current=parent
    return False


def routes_to_rows(function,name):
    pending=[function];seen=set()
    for _ in range(64):
        if not pending:return False
        fn=pending.pop()
        if id(fn) in seen:continue
        seen.add(id(fn))
        code=getattr(fn,"__code__",None)
        if code is None:continue
        if getattr(fn,"__globals__",None) is rows.__dict__ and name in code.co_names:return True
        for cell in getattr(fn,"__closure__",()) or ():
            try:value=cell.cell_contents
            except ValueError:continue
            if callable(value):pending.append(value)
    return False


class CompleteCounter:
    def __init__(self,modes,options):
        self.modes=modes;self.session=modes.session;self.stack=self.session._stack
        self.graph=self.stack.graph;self.options=options;self._fixed_options=options
        self.thread=get_ident();self.live=True;self.preflight_complete=False
        self.wrapper=None;self._in_frame=False;self._serial_owner=None;self.retired=False
        self.identity=(SCOPE_MODULE,"complete-current720",tuple(asdict(options).items()),
                       "valid_query16/full_KV64/headmajor/minimal_boundary-v2")
        self.calls=dict.fromkeys(options.active_sites(),0)
        self.capture_calls=dict(self.calls);self.eager_calls=dict(self.calls)
        self.kernel_calls={};self.kernel_capture_calls={};self.site_resources={}
        self.resources={};self._prepared={};self._specs={};self._compiled={}
        self._binaries={};self._kernel_bindings={};self.layout_receipts={}
        self.attention_launch_attempts={};self._attention_selection={}
        self.entry_launch_attempts={};self._entry_selection={}
        self.capture_gates=[];self.active_site=None
        self._frame_hooks={};self._lifecycle_block_hooks={};self._frame_routes={}
        self.owned_blocks={family:tuple(getattr(self.stack.model,family))
                           for family in ("encoder512","decoder512","vit")}
        self.owned_weights=[]
        self.ffn_names={}
        for family,blocks in self.owned_blocks.items():
            if len(blocks)!=8 or len({id(b) for b in blocks})!=8:raise RuntimeError("Eight owned blocks required")
            for i,block in enumerate(blocks):
                if family=="vit":
                    if self.stack.int8_vit.modules[id(block)]!=i:raise RuntimeError("Actual ViT index changed")
                    self.owned_weights.extend(getattr(block,n) for n in
                        ("qkv_weight","query_scale","projection","attn_skip"))
                else:
                    if tuple(block.window_shift)!=SHIFTS[i%4]:raise RuntimeError("Actual C512 shift changed")
                    self.ffn_names[self.stack.c512_int8.modules[id(block)]]=f"ffn.{family}.{i}"
                    self.owned_weights.extend((block.attention.qkv,block.attention.scale,block.attention.bias,
                        block.attention.pixel_order,block.attention.pixel_inverse,
                        block.projection.weight,block.projection.skip_scale))
                    if block.final_weight is not None:self.owned_weights.append(block.final_weight)
        self.original_c512_ffn=self.stack.c512_int8.ffn
        self.original_vit_ffn=self.stack.int8_vit.ffn
        self.original_rows=(rows.c512_ffn,rows.vit_ffn)
        self.original_dense=self.stack.provider.dense
        self.qkv_targets={id(b.attention.qkv):(b.attention.qkv,f"qkv.{f}.{i}",tuple(b.window_shift))
            for f in ("encoder512","decoder512") for i,b in enumerate(self.owned_blocks[f])}
        if options.c512_ffn_groups and not routes_to_rows(self.original_c512_ffn,"c512_ffn"):
            raise RuntimeError("Current counted C512 FFN does not route to its rows delegate")
        if options.vit_ffn and not routes_to_rows(self.original_vit_ffn,"vit_ffn"):
            raise RuntimeError("Current counted ViT FFN does not route to its rows delegate")
        self.owned_weights.extend(t for row in self.stack.c512_int8.packed.values() for t in row)
        self.owned_weights.extend(t for row in self.stack.int8_vit.packed for t in row)
        self.weight_stamps=tuple(stamp(t) for t in self.owned_weights)
        self.device=self.owned_weights[0].device
        self.plans={};self.assets=[];cache={}
        if torch.is_inference_mode_enabled():raise RuntimeError("Cold assets must precede inference/capture")
        for family in ("encoder512","decoder512"):
            for i,block in enumerate(self.owned_blocks[family]):
                sy,sx=block.window_shift
                hp=((24+sy+7)//8)*8;wp=((40+sx+7)//8)*8
                if options.local_valid_queries:
                    # Every distinct model order is authenticated; the shared four
                    # uploads are cold resources, never a frame/replay readback.
                    order=tuple(block.attention.pixel_order.detach().cpu().tolist())
                    key=(order,(sy,sx),str(block.attention.qkv.device))
                    if key not in cache:
                        plan=build_maps(order,(sy,sx))
                        for name in ("source","windows","bias_rows","destinations"):
                            t=torch.tensor(plan[name],dtype=torch.int32,device=block.attention.qkv.device)
                            plan[name+"_tensor"]=t;self.assets.append(t)
                        cache[key]=plan
                    self.plans[(family,i)]=cache[key]
                else:
                    self.plans[(family,i)]=dict(hp=hp,wp=wp,window_count=hp*wp//64,shift=(sy,sx))
        self.asset_stamps=tuple(stamp(t) for t in self.assets)
        self.contracts={**global_vit.EXTRA,**local.EXTRA,**ffn.EXTRA}
        # Never pre-register ROWS_EXTRA or the decoder scope's per-frame contract.
        # They remain the exact original owners of their actual borrowed leaves.
        if options.global_dataflow:
            self.contracts["vit_native_fma_720_v1._exp_vit"]=(("Y",),())
            self.contracts["nr_backend.triton_attention_exp._kernel"]=(("Y",),())
        self.added_contracts=[]
        self.borrowed_contracts={}
        if options.c512_ffn_groups:
            self.borrowed_contracts.update({k:v for k,v in old_c512.EXTRA.items() if k.rsplit(".",1)[1] in ("_entry","_linear","_expand")})
        if options.vit_ffn:
            self.borrowed_contracts.update({k:v for k,v in old_vit.EXTRA.items() if k.rsplit(".",1)[1] in ("_entry","_expand","_contract")})
        self.pins=json.loads(Path(__file__).with_name("audit_vit_c512_pins_720_v1.json").read_text(encoding="utf-8"))
        self.sources={};self.source_modules={}
        for name,digest in self.pins["modules"].items():
            module=importlib.import_module(name);path=Path(module.__file__).resolve()
            if sha256(path.read_bytes()).hexdigest()!=digest:raise RuntimeError("Owned/dependency source pin changed: "+name)
            self.sources[name]=dict(path=str(path),sha256=digest);self.source_modules[name]=module
        self.front_binding=None;self.front_pool_receipts=[]
        if options.use_front_pool_helper:self._bind_front_pool()
        self.effective_numeric=dict(vit_qkv_full_k=options.global_dataflow and options.full_qkv,
            vit_projection_full_k=options.global_dataflow and options.full_projection,
            vit_norm_fma=options.global_dataflow and options.native_norm,
            vit_exp_fma=options.global_dataflow and options.native_exp,
            vit_exp_zero_constant=options.global_dataflow,
            vit_denominator=options.denominator if options.global_dataflow else "reference")
        self.owner_schema={
            "global_vit":"instance forward owns QKV/norm/score/raw exponent/value/denominator/head projection; no old numeric ViT child",
            "encoder_c512":"selected instance boundaries; slots3 and last preserved; unused2/4 diagnostic only",
            "decoder_c512":"valid query producer; original FP32 complete decoder leaf and original after-dot residual",
            "ffn":"rows_scopes delegates conditional on this exact stack; counted FFN callable objects retained",
            "library":"default retains16 Torch mm delegates, hardware unknown; native_dpas replaces only16 owned QKV weights with explicit FP16 dot/FP32 accumulators",
            "probability":"P08 owns new valid-query attention math with unrounded weights; actual old hits replaced explicitly",
            "signature":"main unique NumericCounter hooks_identity; no child signature hook",
            "front_pool":"owned final consumer calls down:c512 helper if selected; Front must not own C512 boundaries",
            "model_registry":"model.forward unchanged; selected block instance schema must be admitted by main",
        }

    def _bind_front_pool(self):
        function=getattr(self.stack.model,"_audit_fdp_pool_down",None)
        owner=getattr(function,"__self__",None);fn=getattr(function,"__func__",None)
        module=sys.modules.get("audit_front_decoder_post_720_v1")
        terminal=self.owned_blocks["encoder512"][7]
        if (module is None or type(owner) is not getattr(module,"Scope",None) or
                owner is not getattr(module,"_ACTIVE",None) or fn is not type(owner).pool_down or
                owner.modes is not self.modes or owner.session is not self.session or owner.graph is not self.graph or
                not owner.active or owner.flags.get("pool_c512_transition") is not False or
                owner.down_sites.get("down:c512")!=(terminal,"final_weight",(24,40,512)) or
                any(o is terminal and n in ("forward","forward_boundaries") for o,n,*_ in owner.bindings)):
            raise RuntimeError("Front pool helper requires a live helper-only C512 owner without a boundary hook")
        owner.validate(fresh=True,sources=True)
        self.front_binding=(function,owner,fn,fn.__code__,module)

    def _require_front_pool(self):
        function,owner,fn,code,module=self.front_binding
        current=getattr(self.stack.model,"_audit_fdp_pool_down",None)
        if (sys.modules.get(module.__name__) is not module or getattr(module,"_ACTIVE",None) is not owner or
                getattr(current,"__self__",None) is not owner or getattr(current,"__func__",None) is not fn or
                fn.__code__ is not code or not owner.preflight_complete or
                any(o is self.owned_blocks["encoder512"][7] and n in ("forward","forward_boundaries")
                    for o,n,*_ in owner.bindings)):
            raise RuntimeError("Front compiled helper/owner changed or overlaps P11")
        owner.validate_frame_context()
        return current,owner

    def front_pool(self,full,weight):
        function,owner=self._require_front_pool()
        site=owner.pool_site("down:c512",full)
        if site not in owner.screens:raise RuntimeError("Front actual full stride was not preflighted")
        before=owner.capture_calls.get(site,0);calls=owner.calls.get(site,0)
        out=function("down:c512",full,weight)
        global_vit.require_half(out,(12,20,1024),self.device)
        if owner.calls[site]!=calls+1:raise RuntimeError("Front pool consumer missed its actual launch")
        self.front_pool_receipts.append(dict(site=site,source_site="encoder512.7",shape=list(full.shape),
            stride=list(full.stride()),captured=owner.capture_calls[site]-before,
            compiled_resource=dict(owner.screens[site][4]),raw_pool_materialized=self.options.diagnostics))
        self.hit("front_pool.down:c512")
        return out

    def _guard(self):
        self.session._ready()
        if (not self.live or self.options is not self._fixed_options or get_ident()!=self.thread or
                self.modes.session is not self.session or self.stack.graph is not self.graph or
                self.modes.height!=720 or tuple(self.modes.source)!=(720,1280) or self.modes.variant!="unrounded" or
                not self.modes.c512_qkv_library_720 or not self.modes.native_k8_720 or
                not self.modes.c512_probability_unround_720 or
                self.stack.provider.mode!="fp16_xmx" or ENABLED!=FAMILIES):
            raise RuntimeError("Complete current720 immutable profile/owner/thread changed")
        if getattr(self.session,"_audit_vit_c512_720",None) is not self:raise RuntimeError("Owned marker changed")
        if self.wrapper is not None and not scope_contains(self.session._installed,self.wrapper):
            raise RuntimeError("Owned frame scope lost its actual parent chain")
        if (self.stack.c512_int8.ffn is not self.original_c512_ffn or
                self.stack.int8_vit.ffn is not self.original_vit_ffn):
            raise RuntimeError("Original counted FFN callable identity changed")
        if tuple(stamp(t) for t in self.owned_weights)!=self.weight_stamps:raise RuntimeError("Weights/scales/order changed")
        if tuple(stamp(t) for t in self.assets)!=self.asset_stamps:raise RuntimeError("Cold query maps changed")
        if any(CONTRACTS.get(k)!=v for k,v in self.contracts.items()):raise RuntimeError("Owned contracts changed")
        for jit,fn,code in self._kernel_bindings.values():
            if jit.fn is not fn or fn.__code__ is not code:raise RuntimeError("Prepared JIT function/code changed")
        if self._in_frame:
            if any(getattr(o,n) is not v for o,n,v in self._frame_routes.values()):
                raise RuntimeError("Owned actual rows delegate changed")
            if any(block.__dict__.get(name) is not value for block,name,value in self._frame_hooks.values()):
                raise RuntimeError("Owned instance method changed")
        elif (rows.c512_ffn,rows.vit_ffn)!=self.original_rows:
            raise RuntimeError("Idle rows delegate changed outside its frame scope")
        if not self._in_frame and self.stack.provider.dense!=self.original_dense:
            raise RuntimeError("Idle original joint library/K8 provider binding changed")
        # Existing packed source and ownership guards remain active.
        self.stack.call_guard.validate()

    def validate(self,*,fresh=False,sources=False):
        self._guard()
        if fresh or sources:self.validate_sources()
        return True

    def validate_sources(self):
        for name,row in self.sources.items():
            module=self.source_modules[name];path=Path(row["path"])
            if sys.modules.get(name) is not module or Path(module.__file__).resolve()!=path or sha256(path.read_bytes()).hexdigest()!=row["sha256"]:
                raise RuntimeError("Cold source seal changed: "+name)
        for key,kernel in self._compiled.items():
            if kernel.kernel is not self._binaries[key]:raise RuntimeError("Prepared loaded binary object changed")
            row=self._binary(kernel)
            if any(self.resources[key].get(k)!=v for k,v in row.items()):raise RuntimeError("Prepared binary seal changed")
        return True

    def validate_frame_context(self):return self.validate()
    def _validate_replay(self):return self.validate()

    def bind_serial_owner(self,owner,*,transfer_method):
        bind_serial_owner(self,owner,transfer_method)

    def _transfer_serial_thread(self,owner,*,serial_guard,previous_thread):
        current=require_serial_transaction(self,owner,serial_guard,previous_thread)
        self.thread=current
        self.validate_frame_context()
        return current

    def observe_layout(self,label,tensor):
        key=(label,tuple(tensor.shape),tuple(tensor.stride()),tensor.is_contiguous())
        if key not in self.layout_receipts:
            self.layout_receipts[key]=dict(label=label,shape=list(tensor.shape),stride=list(tensor.stride()),
                contiguous=tensor.is_contiguous(),dtype=str(tensor.dtype),
                observed_phase="actual eager/warmup/capture tensor",
                library_native_task=None,library_binary=None,library_ISA=None)
    
    @staticmethod
    def _spill_metadata(kernel):
        spills=getattr(kernel,"n_spills",None)
        return dict(actualkernelhash=getattr(kernel,"hash",None),
            spills_type=type(spills).__module__+"."+type(spills).__qualname__,
            spills_value=spills if type(spills) in (int,bool,str,type(None)) else repr(spills),
            spills_repr=repr(spills))

    @staticmethod
    def _binary(kernel,*,require_zero=True):
        diagnostic=CompleteCounter._spill_metadata(kernel)
        def fail(reason):
            raise RuntimeError(reason+": "+json.dumps(diagnostic,sort_keys=True,default=repr))
        value=getattr(kernel,"kernel",None)
        if type(value) is not bytes or not value or not isinstance(diagnostic["actualkernelhash"],str) or not kernel.hash:
            fail("Loaded executable bytes/hash missing")
        asm=getattr(kernel,"asm",{});name=getattr(kernel.metadata,"binary_ext",None)
        if name is None:
            names=[n for n in ("spv","zebin") if asm.get(n) is value]
            if len(names)!=1:fail("Actual binary format missing/ambiguous")
            name=names[0]
        if name not in ("spv","zebin") or asm.get(name) is not value:
            fail("Actual Intel binary/asm identity mismatch")
        diagnostic.update(binary_format=name,actualbinary_sha256=sha256(value).hexdigest(),actualbinary_bytes=len(value))
        if type(getattr(kernel,"n_spills",None)) is not int:
            fail("Unknown/invalid spill metadata; zero-spill qualification failed")
        if kernel.n_spills<0:fail("Negative spill metadata; zero-spill qualification failed")
        if require_zero and kernel.n_spills>0:fail("Known positive spills; zero-spill qualification failed")
        return dict(diagnostic,
            actualbinary_bytes=len(value),spills=kernel.n_spills,
            registers=kernel.n_regs,shared_bytes=kernel.metadata.shared)

    @staticmethod
    def _attention_condition(jit,grid,args):
        if jit is not local._attention or len(args)!=11 or tuple(grid)!=(60,16):
            raise RuntimeError("Owned valid-attention JIT/signature/grid changed")
        wph,bm,packed=args[-3:]
        if type(wph) is not int or wph<=0 or type(bm) is not int or bm!=16 or type(packed) is not bool:
            raise RuntimeError("Valid-attention requires its fixed BM16/window/packed mapping")
        return wph,bm,packed

    def _selected_attention(self,condition,jit,requested):
        selected=self._attention_selection.get(condition)
        if selected is None:raise RuntimeError("Valid-attention cold selection missing")
        bound_jit,fn,code,kernel,hash_value,binary,opts,base=selected
        if (jit is not bound_jit or jit.fn is not fn or fn.__code__ is not code or
                requested!=dict(base) or kernel.hash!=hash_value or kernel.kernel is not binary):
            raise RuntimeError("Valid-attention cold compiler/options/binary binding changed")
        return kernel,dict(opts)

    @staticmethod
    def _emit_cold_attempt(role,condition,attempt,kernel=None):
        # Child stdout is already durable under the GPU runner. Emit before a
        # later preflight role can fail; never called by frame/replay launch.
        attempt["compiler_kernel_observed"]=kernel is not None
        attempt["spills_observed"]=kernel is not None and hasattr(kernel,"n_spills")
        binary=getattr(kernel,"kernel",None)
        attempt["actualbinary_sha256"]=sha256(binary).hexdigest() if type(binary) is bytes and binary else None
        attempt["actualbinary_bytes"]=len(binary) if type(binary) is bytes else None
        attempt["binary_receipt_validated"]="resources" in attempt
        for name in ("actualkernelhash","spills_type","spills_value","spills_repr"):
            attempt.setdefault(name,None)
        print(json.dumps(dict(event="CURRENT720_COLD_SELECTION_ATTEMPT",role=role,
            phase="preflight_only",condition=list(condition),
            selected=attempt["status"]=="selected_zero_spill",attempt=attempt),
            sort_keys=True,default=repr),flush=True)

    def _compile_attention(self,jit,grid,args,opts):
        if self.preflight_complete or torch.xpu.is_current_stream_capturing():
            raise RuntimeError("Valid-attention launch selection is cold preflight only")
        condition=self._attention_condition(jit,grid,args)
        if condition in self._attention_selection:
            return self._selected_attention(condition,jit,opts)
        if opts!=LAUNCH_OPTIONS:raise RuntimeError("Valid-attention must screen the unchanged default first")
        wph,bm,packed=condition
        name=f"WPH={wph}/BM={bm}/PACKED_OUTPUT={packed}"
        if name in self.attention_launch_attempts:raise RuntimeError("Failed cold attention selection cannot be retried")
        attempts=self.attention_launch_attempts[name]=[]
        # This pinned Intel compiler spells large GRF '256', not 'large'.
        # It maps to -cl-intel-256-GRF-per-thread; BM/mapping/math stay fixed.
        for candidate in (dict(opts),dict(opts,grf_mode="256")):
            attempt=dict(condition=dict(WPH=wph,BM=bm,PACKED_OUTPUT=packed),
                grid=list(grid),launch_options=dict(candidate),status="compile_begin")
            attempts.append(attempt);kernel=None
            try:
                kernel=jit.warmup(*args,grid=grid,**candidate);kernel._init_handles()
                attempt.update(self._spill_metadata(kernel))
                attempt["compiler_metadata"]={n:getattr(kernel.metadata,n,None)
                    for n in ("build_flags","grf_mode","num_warps","num_stages","shared")}
                row=self._binary(kernel,require_zero=False)
                attempt["resources"]=row
            except Exception as error:
                if kernel is not None:attempt.update(self._spill_metadata(kernel))
                attempt.update(status="failed",error_type=type(error).__name__,error=str(error))
                self._emit_cold_attempt("c512_valid_attention",condition,attempt,kernel)
                raise
            if row["spills"]==0:
                attempt["status"]="selected_zero_spill"
                self._emit_cold_attempt("c512_valid_attention",condition,attempt,kernel)
                self._attention_selection[condition]=(jit,jit.fn,jit.fn.__code__,kernel,
                    kernel.hash,kernel.kernel,tuple(candidate.items()),tuple(opts.items()))
                return kernel,candidate
            attempt["status"]="rejected_known_positive_spills"
            self._emit_cold_attempt("c512_valid_attention",condition,attempt,kernel)
            # Only a valid loaded binary with strict positive-int spills reaches
            # the second option. Unknown/negative/bool/binary/compile errors stop.
        raise RuntimeError("Bounded valid-attention launch options exhausted: "+json.dumps(attempts,sort_keys=True,default=repr))

    @staticmethod
    def _entry_condition(jit,grid,args):
        if jit is not ffn._vit_expand_entry or len(args)!=7:
            raise RuntimeError("Owned fused INT8 entry JIT/signature changed")
        bm,bn=args[-2:]
        if type(bm) is not int or bm!=16 or type(bn) is not int or bn not in (32,64) or tuple(grid)!=(15,4096//bn):
            raise RuntimeError("Fused INT8 entry requested tile/grid changed")
        return bm,bn

    def _selected_vit_entry(self,condition,jit,args,requested):
        selected=self._entry_selection.get(condition)
        if selected is None:raise RuntimeError("Fused INT8 entry cold selection missing")
        bound_jit,fn,code,kernel,hash_value,binary,bm,bn,opts,base=selected
        if (jit is not bound_jit or jit.fn is not fn or fn.__code__ is not code or
                requested!=dict(base) or kernel.hash!=hash_value or kernel.kernel is not binary):
            raise RuntimeError("Fused INT8 entry cold compiler/options/binary binding changed")
        return kernel,tuple(args[:-2])+(bm,bn),(triton.cdiv(240,bm),4096//bn),dict(opts)

    def _compile_vit_entry(self,jit,grid,args,opts):
        if self.preflight_complete or torch.xpu.is_current_stream_capturing():
            raise RuntimeError("Fused INT8 entry selection is cold preflight only")
        condition=self._entry_condition(jit,grid,args)
        if condition in self._entry_selection:
            return self._selected_vit_entry(condition,jit,args,opts)
        if opts!=LAUNCH_OPTIONS:raise RuntimeError("Fused INT8 entry must screen unchanged default options first")
        name=f"requested_BM={condition[0]}/BN={condition[1]}"
        if name in self.entry_launch_attempts:raise RuntimeError("Failed fused INT8 entry selection cannot be retried")
        attempts=self.entry_launch_attempts[name]=[]
        configs=list(dict.fromkeys(((16,condition[1],4),(8,condition[1],4),(4,32,4),(4,32,8))))
        for bm,bn,warps in configs:
            candidate=dict(opts,num_warps=warps)
            selected_args=tuple(args[:-2])+(bm,bn);selected_grid=(triton.cdiv(240,bm),4096//bn)
            attempt=dict(config=dict(BM=bm,BN=bn,num_warps=warps),grid=list(selected_grid),
                launch_options=dict(candidate),strategy="stream_K32_after_fullrow_max_v1",status="compile_begin")
            attempts.append(attempt);kernel=None
            try:
                kernel=jit.warmup(*selected_args,grid=selected_grid,**candidate);kernel._init_handles()
                attempt.update(self._spill_metadata(kernel))
                attempt["compiler_metadata"]={n:getattr(kernel.metadata,n,None)
                    for n in ("build_flags","num_warps","num_stages","threads_per_warp","shared")}
                row=self._binary(kernel,require_zero=False);attempt["resources"]=row
            except Exception as error:
                if kernel is not None:attempt.update(self._spill_metadata(kernel))
                attempt.update(status="failed",error_type=type(error).__name__,error=str(error))
                self._emit_cold_attempt("vit_ffn_entry_expand",condition,attempt,kernel)
                raise
            if row["spills"]==0:
                attempt["status"]="selected_zero_spill"
                self._emit_cold_attempt("vit_ffn_entry_expand",condition,attempt,kernel)
                self._entry_selection[condition]=(jit,jit.fn,jit.fn.__code__,kernel,kernel.hash,kernel.kernel,
                    bm,bn,tuple(candidate.items()),tuple(opts.items()))
                return kernel,selected_args,selected_grid,candidate
            attempt["status"]="rejected_known_positive_spills"
            self._emit_cold_attempt("vit_ffn_entry_expand",condition,attempt,kernel)
        raise RuntimeError("Bounded fused INT8 entry configurations exhausted: "+json.dumps(attempts,sort_keys=True,default=repr))

    def compile_spec(self,label,jit,grid,args,*,launch_options=None):
        opts=dict(LAUNCH_OPTIONS if launch_options is None else launch_options)
        if label not in ("c512_valid_attention","vit_ffn_entry_expand"):
            kernel=jit.warmup(*args,grid=grid,**opts);kernel._init_handles()
            key=label+":"+kernel.hash;row=self._binary(kernel)
        path=Path(jit.fn.__code__.co_filename).resolve()
        source=dict(path=str(path),line=jit.fn.__code__.co_firstlineno,sha256=sha256(path.read_bytes()).hexdigest())
        if source["sha256"] not in {r["sha256"] for r in self.sources.values()}:
            raise RuntimeError("Prepared JIT source outside the owned/dependency seal")
        if label=="c512_valid_attention":
            kernel,opts=self._compile_attention(jit,grid,args,opts)
            key=label+":"+kernel.hash;row=self._binary(kernel)
        elif label=="vit_ffn_entry_expand":
            kernel,args,grid,opts=self._compile_vit_entry(jit,grid,args,opts)
            key=label+":"+kernel.hash;row=self._binary(kernel)
        if key not in self.resources:
            self.resources[key]=dict(row,source_kernel=jit.fn.__module__+"."+jit.fn.__name__,source=source,
                grid=list(grid),launch_options=opts,
                IR_dot_lines={n:[dict(line=i+1,text=l.strip()) for i,l in enumerate(v.splitlines())
                                 if "dot" in l or "dpas" in l]
                              for n,v in kernel.asm.items() if n in ("ttir","ttgir") and isinstance(v,str)},
                proof="actual loaded binary and compiler metadata; no physical counter or timing inference")
            self._compiled[key]=kernel;self._binaries[key]=kernel.kernel
            self._kernel_bindings[key]=(jit,jit.fn,jit.fn.__code__)
            if label=="vit_ffn_entry_expand":
                self.resources[key]["entry_strategy"]="stream_K32_after_fullrow_max_v1"
                self.resources[key]["selected_tile"]={"BM":args[-2],"BN":args[-1]}
        elif self._compiled[key] is not kernel:raise RuntimeError("Compiler object changed for a sealed key")
        self._prepared.setdefault(label,set()).add(kernel.hash)
        specs=self._specs.setdefault(key,[])
        spec=(argument_shapes(args),tuple(grid),opts)
        if spec not in specs:specs.append(spec)
        return kernel

    def launch(self,label,jit,grid,args,*,launch_options=None):
        if not self.preflight_complete:raise RuntimeError("Deferred complete preflight is required")
        opts=dict(LAUNCH_OPTIONS if launch_options is None else launch_options)
        if label=="c512_valid_attention":
            condition=self._attention_condition(jit,grid,args)
            kernel,opts=self._selected_attention(condition,jit,opts)
        elif label=="vit_ffn_entry_expand":
            condition=self._entry_condition(jit,grid,args)
            kernel,args,grid,opts=self._selected_vit_entry(condition,jit,args,opts)
        else:
            # Other roles retain their already-screened warmup resolution.
            kernel=jit.warmup(*args,grid=grid,**opts);kernel._init_handles()
        key=label+":"+kernel.hash
        spec=(argument_shapes(args),tuple(grid),opts)
        if (kernel.hash not in self._prepared.get(label,()) or self._compiled.get(key) is not kernel or
                spec not in self._specs.get(key,()) or kernel.kernel is not self._binaries.get(key)):
            raise RuntimeError("Actual specialization/layout/grid differs from cold preflight")
        if label in ("c512_valid_attention","vit_ffn_entry_expand"):
            if self.resources[key]["launch_options"]!=opts:
                raise RuntimeError("Cold selected resource options changed: "+label)
            # CompiledKernel runner accepts a three-dimensional grid and all
            # original arguments. No JIT/hot selection/fallback takes place.
            kernel[tuple(grid)+(1,)*(3-len(grid))](*args);launched=kernel
        else:launched=jit[grid](*args,**opts)
        if launched is not kernel:raise RuntimeError("Launched compiler object differs from prepared binary")
        self.kernel_calls[label]=self.kernel_calls.get(label,0)+1
        capturing=torch.xpu.is_current_stream_capturing()
        if capturing:self.kernel_capture_calls[label]=self.kernel_capture_calls.get(label,0)+1
        if self.active_site is not None:
            bucket=self.site_resources.setdefault(self.active_site,{})
            counts=bucket.setdefault(key,dict(eager=0,capture=0))
            counts["capture" if capturing else "eager"]+=1
        return kernel

    @contextmanager
    def at_site(self,site):
        previous=self.active_site;self.active_site=site
        try:yield
        finally:self.active_site=previous

    def hit(self,site):
        if site not in self.calls:raise RuntimeError("Unselected owned site executed")
        self.calls[site]+=1
        target=self.capture_calls if torch.xpu.is_current_stream_capturing() else self.eager_calls
        target[site]+=1

    def preflight(self):
        """GPU owner only, deferred after NumericSuite. This worker never ran it."""
        self.validate(fresh=True,sources=True)
        if self.graph.entries or self.graph.replays or torch.xpu.is_current_stream_capturing():
            raise RuntimeError("Cold preflight must precede capture/replay")
        if self.preflight_complete:return self.snapshot()["resources"]
        bm,bn=self.options.bm,self.options.bn;o=self.options
        empty=lambda shape,dtype=torch.float16:torch.empty(shape,device=self.device,dtype=dtype)
        if o.c512_qkv_backend=="native_dpas":
            for family in ("encoder512","decoder512"):
                for i,b in enumerate(self.owned_blocks[family]):
                    p=self.plans[(family,i)];m=p["hp"]*p["wp"]
                    self.compile_spec("c512_qkv_native",local._qkv_native,(triton.cdiv(m,bm),1536//bn),
                        (empty((p["hp"],p["wp"],512)),b.attention.qkv,empty((p["hp"],p["wp"],1536)),m,bm,bn))
        if o.global_dataflow:
            x=empty((240,1024));q=empty((32,240,32));k=empty((32,256,32));v=empty(k.shape)
            e=empty((32,240,256));num=empty(q.shape);out=empty(x.shape)
            for block in self.owned_blocks["vit"]:
                self.compile_spec("vit_qkv",global_vit._qkv,(triton.cdiv(256,bm),32,3),
                    (x,block.qkv_weight,block.query_scale,q,k,v,bm,o.full_qkv,o.native_norm))
                self.compile_spec("vit_projection",global_vit._projection,(triton.cdiv(240,bm),1024//bn),
                    (num,block.projection,x,block.attn_skip,out,bm,bn,o.full_projection))
            jit=global_vit._exp_vit if o.native_exp else global_vit._exp_reference
            args=(e,empty(e.shape),1966080,512) if o.native_exp else (e,empty(e.shape),1966080,True,512)
            self.compile_spec("vit_exp_native" if o.native_exp else "vit_exp_reference",jit,(3840,),args)
            self.compile_spec("vit_denominator_apply",global_vit._denominator_apply,(480,),
                (e,num,empty(num.shape),16,o.denominator=="ordered_fused"))
            zero=torch.zeros((),device=self.device,dtype=torch.float16);zero_out=empty(())
            zero_args=(zero,zero_out,1,512) if o.native_exp else (zero,zero_out,1,True,512)
            kernel=self.compile_spec("vit_zero_setup",jit,(1,),zero_args)
            actual_kernel=jit[(1,)](*zero_args,**LAUNCH_OPTIONS)
            if actual_kernel is not kernel:raise RuntimeError("Native padding setup binary changed")
            bits=int(zero_out.view(torch.int16).item())&0xffff
            if bits!=global_vit.ZERO_BITS:raise RuntimeError("Selected exponent zero differs from folded padding")
            self.resources["vit_zero_setup:"+kernel.hash]["actual_zero_bits"]=bits
            self.resources["vit_zero_setup:"+kernel.hash]["setup_only"]=True
        for family in ("encoder512","decoder512"):
            for i,block in enumerate(self.owned_blocks[family]):
                if not o.local_valid_queries and (family=="decoder512" or i not in o.encoder_indices):
                    continue
                plan=self.plans[(family,i)];hp,wp,wph=plan["hp"],plan["wp"],plan["window_count"]
                sy,sx=plan["shift"];cx=empty((24,40,512))
                if o.local_valid_queries:
                    z=empty((hp,wp,16,3,32));q=empty((16,960,32))
                    k=empty((16,wph,64,32));v=empty(k.shape)
                    packed=family=="decoder512";out=empty(k.shape) if packed else empty(q.shape)
                    self.compile_spec("c512_valid_query",local._prepare_query,(60,16),
                        (z,block.attention.scale,plan["source_tensor"],q,16))
                    if packed:self.compile_spec("c512_full_kv_packed",local._prepare_kv_packed,(wph,16,4),
                        (z,block.attention.pixel_order,k,v,out,wph,wp,16))
                    else:self.compile_spec("c512_full_kv",local._prepare_kv,(wph,16,4),
                        (z,block.attention.pixel_order,k,v,wph,wp,16))
                    self.compile_spec("c512_valid_attention",local._attention,(60,16),
                        (q,k,v,block.attention.bias,plan["windows_tensor"],plan["bias_rows_tensor"],
                         plan["destinations_tensor"],out,wph,16,packed))
                    if packed:
                        packed_out=out.reshape(16,hp//8,wp//8,64,32)
                        self.compile_spec("c512_decoder_projection",decoder_leaf,(60,16),
                            (packed_out,block.projection.weight,cx,block.projection.skip_scale,
                             block.attention.pixel_inverse,empty(cx.shape),hp,wp,24,40,sy,sx,16,32),
                            launch_options=DECODER_OPTIONS)
                if family!="encoder512":continue
                heads=empty((16,960,32)) if o.local_valid_queries else empty((16,hp//8,wp//8,64,32))
                packed=not o.local_valid_queries
                if o.encoder_direct_projection:
                    self.compile_spec("c512_encoder_projection",local._encoder_project,(960//bm,512//bn),
                        (heads,block.projection.weight,cx,block.projection.skip_scale,
                         block.attention.pixel_inverse,empty(cx.shape),hp,wp,sy,sx,packed,bm,bn))
                    if not o.local_valid_queries:
                        z=empty((hp,wp,16,3,32));br=self.stack.window_blocks.layout.rows
                        q,k,v=[empty(heads.shape) for _ in range(3)]
                        self.compile_spec("c512_retained_packed_qkv",local._retained_pack,
                            (triton.cdiv(hp*wp*16,br),3),
                            (z,block.attention.scale,block.attention.pixel_order,q,k,v,hp*wp*16,wp,hp//8,wp//8,br))
                if (o.local_valid_queries and not o.encoder_direct_projection or
                        o.diagnostics and o.encoder_direct_projection):
                    self.compile_spec("c512_encoder_unpack",local._encoder_unpack,(960,),
                        (heads,block.attention.pixel_inverse,empty(cx.shape),hp,wp,sy,sx,packed))
                if i==7 and o.encoder_pool_final:
                    if o.use_front_pool_helper:
                        self._require_front_pool()
                        site=self.front_binding[1].pool_site("down:c512",cx)
                        if site not in self.front_binding[1].screens:raise RuntimeError("Front contiguous C512 pool key absent")
                    elif o.diagnostics:
                        self.compile_spec("c512_encoder_pool_final_debug",local._pool_final_debug,
                            (triton.cdiv(240,bm),1024//bn),
                            (cx,block.final_weight,empty((240,512)),empty((240,1024)),bm,bn))
                    else:self.compile_spec("c512_encoder_pool_final",local._pool_final,
                        (triton.cdiv(240,bm),1024//bn),(cx,block.final_weight,empty((240,1024)),bm,bn))
        if o.c512_ffn_groups or o.vit_ffn:self._preflight_ffn(empty,bm,bn)
        self.preflight_complete=True
        return self.snapshot()["resources"]

    def _preflight_ffn(self,empty,bm,bn):
        if self.options.c512_ffn_groups:
            x=empty((960,512));qx=empty(x.shape,torch.int8);sx=empty((960,),torch.float32)
            qz=empty(qx.shape,torch.int8);qg=empty(qx.shape,torch.int8);out=empty(x.shape)
            self.compile_spec("c512_ffn_entry",old_c512._entry,(960,),(x,qx,sx,960,False))
            for p in self.stack.c512_int8.packed.values():
                w0,s0,sz,we,se,sh,wr,sr,sg,wp,sp,skip=p
                self.compile_spec("c512_ffn_linear",old_c512._linear,(960//bm,512//bn),
                    (qx,sx,w0,s0,sz,qz,x,960,bm,bn,False))
                self.compile_spec("c512_ffn_groups",old_c512._groups,(960//bm,8,64//bn),
                    (qz,we,se,sh,wr,sr,sg,qg,x,960,bm,bn,False))
                self.compile_spec("c512_ffn_project",ffn._c512_project,(960//bm,512//bn),
                    (qg,wp,sp,x,skip,out,960,bm,bn))
                if self.stack.c512_int8.ffn_probe is not None:
                    self.compile_spec("c512_ffn_debug_qh",old_c512._expand,(960//bm,8,256//bn),
                        (qz,we,se,sh,empty((960,2048),torch.int8),x,960,bm,bn,False))
        if not self.options.vit_ffn:return
        x=empty((240,1024));qx=empty(x.shape,torch.int8);sx=empty((240,),torch.float32)
        qh=empty((240,4096),torch.int8);out=empty(x.shape)
        if not self.options.vit_fuse_entry:
            self.compile_spec("vit_ffn_entry",old_vit._entry,(240,),(x,qx,sx,240,1024))
        for we,se,sh,wc,sc,skip in self.stack.int8_vit.packed:
            if self.options.vit_fuse_entry:self.compile_spec("vit_ffn_entry_expand",ffn._vit_expand_entry,
                (15,4096//bn),(x,we,se,sh,qh,16,bn))
            else:self.compile_spec("vit_ffn_expand",old_vit._expand,(triton.cdiv(240,bm),4096//bn),
                (qx,sx,we,se,sh,qh,x,240,1024,4096,bm,bn,False))
            if self.options.vit_contract_parts==1:
                self.compile_spec("vit_ffn_contract_full",ffn._vit_contract_full,(triton.cdiv(240,bm),1024//bn),
                    (qh,wc,sc,x,skip,out,bm,bn))
            else:
                p=empty((4,240,1024),torch.int32)
                self.compile_spec("vit_ffn_contract_p4",old_vit._contract,(triton.cdiv(240,bm),1024//bn,4),
                    (qh,wc,sc,x,skip,p,out,x,240,4096,1024,4,bm,bn,False,False))
                self.compile_spec("vit_ffn_merge",ffn._vit_merge_p4,(480,),(p,sc,x,skip,out))

    def decoder_from_packed(self,block,mlp,packed,plan):
        global_vit.require_half(packed,(16,plan["hp"]//8,plan["wp"]//8,64,32),self.device)
        if CONTRACTS.get("c512_window_projection_v1._project")!=(("OUT",),()):
            raise RuntimeError("Retained decoder frame contract is not installed")
        full=torch.empty_like(mlp)
        self.launch("c512_decoder_projection",decoder_leaf,(60,16),
            (packed,block.projection.weight,mlp,block.projection.skip_scale,block.attention.pixel_inverse,
             full,plan["hp"],plan["wp"],24,40,*plan["shift"],16,32),launch_options=DECODER_OPTIONS)
        # This actual dense leaf ran; logical dispatch is retained honestly.
        self.stack.provider.record("fp16_dense");record_arithmetic_dispatch("dense")
        return full

    def _site_captured_resource(self,site):
        if not self.preflight_complete:return False
        if site.startswith("qkv."):required={"c512_qkv_native"}
        elif site.startswith("encoder512.") and self.options.local_valid_queries:
            required={"c512_valid_query","c512_full_kv","c512_valid_attention"}
        elif site.startswith("decoder512.") and self.options.local_valid_queries:
            required={"c512_valid_query","c512_full_kv_packed","c512_valid_attention","c512_decoder_projection"}
        else:required=set()
        actual={key.split(":",1)[0] for key,counts in self.site_resources.get(site,{}).items()
            if counts["capture"]>0 and key in self.resources and key in self._compiled
            and self._compiled[key].kernel is self._binaries[key]}
        return bool(required) and required<=actual

    def capture_role_replacements(self,before):
        """Only changed old sites. Main combines other owners and verifies remainders."""
        prior=before.get("own",{}).get(SCOPE_MODULE,{})
        if not isinstance(prior,dict):raise TypeError("Own capture snapshot must be a dict")
        result={}
        def add(gate,replaced,old_before,old_after,site_map):
            if old_before is None:return
            if not replaced:raise RuntimeError("Changed old gate has no matching owned site")
            new={site:self.capture_calls[site]-prior.get(site,0) for site in site_map.values()}
            # A replay has zero capture delta; it cannot satisfy this replacement.
            passed=all(delta==1 and self._site_captured_resource(site) for site,delta in new.items())
            result[gate]=dict(passed=passed,replaced_sites=[str(s) for s in replaced],new_sites=new,
                new_counter_keys={site:site for site in new},
                remaining_old_sites={str(k):old_after[k]-v for k,v in old_before.items() if k not in replaced},
                compiled_resources={site:list(self.site_resources.get(site,{})) for site in new},
                proof="actual new capture deltas; retained old counters never incremented by this adapter")
        if self.options.local_valid_queries:
            old=before.get("before_c512_probability")
            counter=getattr(self.modes,"c512_probability_calls",None)
            if old is not None and counter is not None:
                keys=[k for k in old if str(k) in {f"{f}.{i}" for f in ("encoder512","decoder512") for i in range(8)}]
                add("c512_probability_all_sixteen",keys,old,counter.capture_calls,{k:str(k) for k in keys})
            combo=before.get("combo") or {}
            old=combo.get("c512_decoder")
            if old is not None:
                from three_structure_combo_v1 import snapshot_calls
                after=snapshot_calls(self.modes.combo_calls)["c512_decoder"]
                keys=[k for k in old if str(k) in {str(i) for i in range(8)}]
                add("c512_decoder_all_eight",keys,old,after,{k:"decoder512."+str(k) for k in keys})
        if self.options.c512_qkv_backend=="native_dpas":
            old=before.get("before_c512_library")
            if old is not None:
                keys=[k for k in old if str(k) in {f"{f}_{i}" for f in ("encoder512","decoder512") for i in range(8)}]
                add("c512_library_all_sixteen",keys,old,self.modes.c512_library_calls,
                    {k:"qkv."+str(k).rsplit("_",1)[0]+"."+str(k).rsplit("_",1)[1] for k in keys})
        # Default library path retains the original gate; native_dpas reports
        # actual new capture proof and never increments the Torch-mm counter.
        return result

    def snapshot(self):
        return dict(identity=self.identity,options=asdict(self.options),effective_numeric=self.effective_numeric,
            owner_schema=self.owner_schema,calls=dict(self.calls),capture_calls=dict(self.capture_calls),
            eager_calls=dict(self.eager_calls),kernel_calls=dict(self.kernel_calls),
            kernel_capture_calls=dict(self.kernel_capture_calls),site_resources=self.site_resources,
            resources=dict(self.resources),preflight_complete=self.preflight_complete,
            attention_launch_attempts=self.attention_launch_attempts,
            entry_launch_attempts=self.entry_launch_attempts,
            sources=dict(self.sources),capture_gates=list(self.capture_gates),
            layouts=list(self.layout_receipts.values()),front_pool_consumers=list(self.front_pool_receipts),
            cold_assets=[dict(shape=list(t.shape),dtype=str(t.dtype),bytes=t.numel()*t.element_size()) for t in self.assets],
            author_GPU_tested=False,GPU_quality_passed=False,GPU_performance_measured=False)


@contextmanager
def installed(modes,**kwargs):
    options=Options(**kwargs);session=getattr(modes,"session",None)
    if (session is None or modes.height!=720 or tuple(modes.source)!=(720,1280) or modes.variant!="unrounded" or
            not modes.controlled or not modes.c512_qkv_library_720 or not modes.native_k8_720 or
            not modes.c512_probability_unround_720):
        raise RuntimeError("Select the accepted controlled six-flag current720 stack first")
    session._ready();graph=session._stack.graph
    if graph.entries or graph.replays or "_audit_vit_c512_720" in session.__dict__:
        raise RuntimeError("Complete scope requires a fresh owned session")
    if "_vit_numeric_suite_720" in session.__dict__:
        raise RuntimeError("Use selected complete roles before the old ViT numeric child")
    counter=CompleteCounter(modes,options);previous_scope=session._installed

    @contextmanager
    def wrapped_scope():
        counter.validate()
        if not counter.preflight_complete:raise RuntimeError("Deferred owned preflight missing")
        before=dict(counter.calls);cap_before=dict(counter.capture_calls);entries_before=set(graph.entries)
        with previous_scope():
            saved=[];saved_rows=[];hooks={};routes={};added_frame_contracts=[]
            def set_owned(block,name,replacement,site):
                if name in block.__dict__:
                    raise RuntimeError("Two owners attempt the same C512/ViT instance method")
                value=MethodType(replacement,block);setattr(block,name,value)
                saved.append((block,name,value));hooks[(site,name)]=(block,name,value)
            try:
                for name,value in counter.borrowed_contracts.items():
                    if name in CONTRACTS and CONTRACTS[name]!=value:
                        raise RuntimeError("Borrowed frame output contract collision")
                    if name not in CONTRACTS:
                        CONTRACTS[name]=value;added_frame_contracts.append(name)
                if options.c512_qkv_backend=="native_dpas":
                    provider=counter.stack.provider;old_dense=provider.dense
                    def c512_dense(owner,a,w,*,chunk_k,initial=None,**kwargs):
                        target=counter.qkv_targets.get(id(w))
                        if target is None:return old_dense(a,w,chunk_k=chunk_k,initial=initial,**kwargs)
                        weight,site,shift=target
                        sy,sx=shift;hp=((24+sy+7)//8)*8;wp=((40+sx+7)//8)*8
                        if w is not weight or chunk_k!=16 or initial is not None or kwargs:
                            raise RuntimeError("Native owned C512 QKV caller changed")
                        global_vit.require_half(a,(hp,wp,512),w.device)
                        global_vit.require_half(w,(512,1536),a.device)
                        out=torch.empty((hp,wp,1536),device=a.device,dtype=torch.float16)
                        with counter.at_site(site):
                            counter.observe_layout("native C512 QKV input",a)
                            counter.launch("c512_qkv_native",local._qkv_native,
                                (triton.cdiv(hp*wp,options.bm),1536//options.bn),
                                (a,w,out,hp*wp,options.bm,options.bn))
                            owner.record("fp16_dense");counter.hit(site)
                        return out
                    dense=MethodType(c512_dense,provider);provider.dense=dense
                    saved_rows.append((provider,"dense",old_dense,dense))
                    routes["c512_qkv_dense"]=(provider,"dense",dense)
                    # The parent installed scope has already captured old_dense
                    # in the public arithmetic alias. Cover retained attention
                    # consumers too, then restore that exact alias before exit.
                    import nr_backend.triton_math as reference_math
                    old_dot=reference_math.fused_dot
                    if old_dot is not old_dense:
                        raise RuntimeError("Native C512 QKV lost the actual parent arithmetic alias")
                    reference_math.fused_dot=dense
                    saved_rows.append((reference_math,"fused_dot",old_dot,dense))
                    routes["c512_qkv_dot"]=(reference_math,"fused_dot",dense)
                if options.c512_ffn_groups:
                    previous=rows.c512_ffn
                    def c512_rows(stack,name,features,sink=None):
                        if stack is not counter.stack:return previous(stack,name,features,sink)
                        site=counter.ffn_names[name]
                        with counter.at_site(site):
                            output=ffn.c512_forward(stack,name,features,counter,bm=options.bm,bn=options.bn)
                            if sink is not None:sink(name,output)
                            counter.hit(site)
                            return output
                    rows.c512_ffn=c512_rows;saved_rows.append((rows,"c512_ffn",previous,c512_rows))
                    routes["c512_ffn"]=(rows,"c512_ffn",c512_rows)
                if options.vit_ffn:
                    previous_vit=rows.vit_ffn
                    def vit_rows(stack,index,x,sink=None):
                        if stack is not counter.stack:return previous_vit(stack,index,x,sink)
                        site=f"ffn.vit.{index}"
                        with counter.at_site(site):
                            output=ffn.vit_forward(stack,index,x,counter,bm=options.bm,bn=options.bn,
                                fuse_entry=options.vit_fuse_entry,parts=options.vit_contract_parts)
                            if sink is not None:sink(index,output[0])
                            counter.hit(site)
                            return output
                    rows.vit_ffn=vit_rows;saved_rows.append((rows,"vit_ffn",previous_vit,vit_rows))
                    routes["vit_ffn"]=(rows,"vit_ffn",vit_rows)
                if options.encoder_tail:
                    for i,block in enumerate(counter.owned_blocks["encoder512"]):
                        if i not in options.encoder_indices:continue
                        def encode(owner,x,*,_i=i):
                            if current_arithmetic_backend()!="triton":raise RuntimeError("Current arithmetic changed")
                            site=f"encoder512.{_i}";plan=counter.plans[("encoder512",_i)]
                            with counter.at_site(site):
                                name=counter.stack.c512_int8.modules[id(owner)]
                                mlp=counter.stack.c512_int8.ffn(name,x)
                                if options.local_valid_queries:
                                    heads=local.attention(owner.attention,mlp,plan,counter,packed_output=False);layout="headmajor"
                                elif options.encoder_direct_projection:
                                    heads=local.baseline_packed(counter.stack,owner.attention,mlp,plan,counter);layout="packed"
                                else:
                                    sy,sx=plan["shift"]
                                    heads=owner.attention(local.pad(mlp,plan))[sy:sy+24,sx:sx+40];layout="hwc"
                                output=local.encoder(owner,mlp,heads,plan,counter,input_layout=layout,
                                    direct_projection=options.encoder_direct_projection,pool_final=options.encoder_pool_final,
                                    bm=options.bm,bn=options.bn,diagnostics=options.diagnostics)
                                session._counts["c512"]+=1;counter.hit(site)
                                return output
                        def encode_last(owner,x,*,_forward=encode):return _forward(owner,x)[-1]
                        set_owned(block,"forward_boundaries",encode,f"encoder512.{i}")
                        set_owned(block,"forward",encode_last,f"encoder512.{i}")
                if options.local_valid_queries:
                    for i,block in enumerate(counter.owned_blocks["decoder512"]):
                        def decode(owner,x,*,_i=i):
                            if current_arithmetic_backend()!="triton":raise RuntimeError("Current arithmetic changed")
                            site=f"decoder512.{_i}";plan=counter.plans[("decoder512",_i)]
                            with counter.at_site(site):
                                name=counter.stack.c512_int8.modules[id(owner)]
                                mlp=counter.stack.c512_int8.ffn(name,x)
                                packed=local.attention(owner.attention,mlp,plan,counter,packed_output=True)
                                full=counter.decoder_from_packed(owner,mlp,packed,plan)
                                session._counts["c512"]+=1;counter.hit(site)
                                return full
                        set_owned(block,"forward",decode,f"decoder512.{i}")
                if options.global_dataflow:
                    for i,block in enumerate(counter.owned_blocks["vit"]):
                        def global_forward(owner,x,*,_i=i):
                            if current_arithmetic_backend()!="triton":raise RuntimeError("Current arithmetic changed")
                            site=f"vit.{_i}"
                            with counter.at_site(site):
                                mlp=counter.stack.int8_vit.ffn(_i,x)[0]
                                out=global_vit.forward(owner,mlp,counter,full_qkv=options.full_qkv,
                                    full_projection=options.full_projection,native_norm=options.native_norm,
                                    native_exp=options.native_exp,bm=options.bm,bn=options.bn,
                                    denominator=options.denominator)
                                session._counts["vit"]+=1;counter.hit(site)
                                return out
                        set_owned(block,"forward",global_forward,f"vit.{i}")
                counter._frame_hooks=hooks;counter._lifecycle_block_hooks=hooks
                counter._frame_routes=routes;counter._in_frame=True
                yield
                delta={k:counter.calls[k]-v for k,v in before.items()}
                cap={k:counter.capture_calls[k]-v for k,v in cap_before.items()}
                new_entries=len(set(graph.entries)-entries_before)
                if new_entries:
                    passed=all(delta[k]==3*new_entries and cap[k]==new_entries for k in delta)
                    counter.capture_gates.append(dict(passed=passed,new_entries=new_entries,source_delta=delta,
                        capture_delta=cap,count_kind="2warmup+1capture source calls; not replay GPU task frequency"))
                    if not passed:raise RuntimeError("Owned actual site/resource capture gate failed")
            except BaseException:
                session._failed=True
                raise
            finally:
                valid=all(block.__dict__.get(name) is value for block,name,value in saved)
                valid=valid and all(getattr(o,n) is f for o,n,old,f in saved_rows)
                for o,n,old,f in reversed(saved_rows):setattr(o,n,old)
                for block,name,value in reversed(saved):block.__dict__.pop(name,None)
                valid=valid and all(CONTRACTS.get(n)==counter.borrowed_contracts[n] for n in added_frame_contracts)
                for name in added_frame_contracts:CONTRACTS.pop(name,None)
                counter._frame_hooks={};counter._lifecycle_block_hooks={}
                counter._frame_routes={};counter._in_frame=False
                if not valid:
                    session._failed=True
                    raise RuntimeError("Owned method/rows delegate changed during the frame")
    wrapped_scope.__nr_numeric_original_installed__=previous_scope
    counter.wrapper=wrapped_scope
    try:
        if any(name in CONTRACTS and CONTRACTS[name]!=v for name,v in counter.contracts.items()):
            raise RuntimeError("Owned new output contract collision")
        for name,value in counter.contracts.items():
            if name not in CONTRACTS:CONTRACTS[name]=value;counter.added_contracts.append(name)
        session._audit_vit_c512_720=counter;session._installed=wrapped_scope
        yield counter
    finally:
        valid=session._installed is wrapped_scope
        session._installed=previous_scope;session.__dict__.pop("_audit_vit_c512_720",None)
        counter.live=False;counter.retired=True
        if graph.entries and not graph.closed:
            session._failed=True
            session.__dict__.setdefault("_audit_vit_c512_retired",[]).append(counter)
        for name in counter.added_contracts:CONTRACTS.pop(name,None)
        if not valid:
            session._failed=True
            raise RuntimeError("Unwind numeric/higher scopes before the owned complete scope")
