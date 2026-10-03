"""Owned C512 valid queries, encoder projection and pool consumers; fixed720.

QKV retains the sixteen actual Torch-mm library delegates. Valid query groups
keep the complete64 K/V and the original bias rows/columns. Decoder projection
is the accepted existing FP32 implementation. Encoder uses its distinct initial
BEFORE-dot ordering. All new output contracts are half, without FP8 proofs.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl
from audit_vit_native_dataflow_720_v1 import _norm_registers, require_half
from fused_swin_core_native_half_v1 import _exp, _weights_pair
from nr_backend.triton_attention_weights import _add_half
from nr_backend.triton_attention_normalize import _nan_left
from fused_qkv_pack_native_half_v1 import _pack as _current_pack


@triton.jit
def _qkv_native(X,W,OUT,M:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    col=tl.program_id(1)*BN+tl.arange(0,BN);lane=tl.arange(0,32)
    total=tl.full((BM,BN),0,tl.float32)
    for part in range(16):
        kk=part*32+lane
        a=tl.load(X+row[:,None]*512+kk[None,:],row[:,None]<M,other=0)
        w=tl.load(W+kk[:,None]*1536+col[None,:])
        total=tl.dot(a,w,total,out_dtype=tl.float32)
    tl.store(OUT+row[:,None]*1536+col[None,:],total.to(tl.float16),row[:,None]<M)


@triton.jit
def _retained_pack(Z,SCALE,ORDER,Q,K,V,COUNT:tl.constexpr,WP:tl.constexpr,
                   ROWS:tl.constexpr,COLS:tl.constexpr,BR:tl.constexpr):
    _current_pack(Z,SCALE,ORDER,Q,K,V,COUNT,WP,16,ROWS,COLS,BR,False)


@triton.jit
def _prepare_query(Z,SCALE,MAP,Q,BM:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    head=tl.program_id(1);lane=tl.arange(0,32)
    src=tl.load(MAP+row).to(tl.int32)
    z=tl.load(Z+(src[:,None]*16*3+head*3)*32+lane[None,:])
    z=_norm_registers(z,BM,True)
    scale=tl.load(SCALE+head)
    z=_nan_left((z.to(tl.float32)*scale.to(tl.float32)).to(tl.float16),z,scale)
    tl.store(Q+(head*960+row[:,None])*32+lane[None,:],z)


@triton.jit
def _kv_values(Z,ORDER,WPH:tl.constexpr,WP:tl.constexpr,BR:tl.constexpr):
    window=tl.program_id(0);head=tl.program_id(1)
    slot=tl.program_id(2)*BR+tl.arange(0,BR);lane=tl.arange(0,32)
    physical=tl.load(ORDER+slot).to(tl.int32)
    y=(window//(WP//8))*8+physical//8
    x=(window%(WP//8))*8+physical%8
    offset=((y*WP+x)*16*3+head*3)*32
    kval=tl.load(Z+offset[:,None]+32+lane[None,:])
    vval=tl.load(Z+offset[:,None]+64+lane[None,:])
    kval=_norm_registers(kval,BR,True)
    dest=((head*WPH+window)*64+slot[:,None])*32+lane[None,:]
    return kval,vval,dest


@triton.jit
def _prepare_kv(Z,ORDER,K,V,WPH:tl.constexpr,WP:tl.constexpr,BR:tl.constexpr):
    kval,vval,dest=_kv_values(Z,ORDER,WPH,WP,BR)
    tl.store(K+dest,kval);tl.store(V+dest,vval)


@triton.jit
def _prepare_kv_packed(Z,ORDER,K,V,OUT,WPH:tl.constexpr,WP:tl.constexpr,BR:tl.constexpr):
    kval,vval,dest=_kv_values(Z,ORDER,WPH,WP,BR)
    tl.store(K+dest,kval);tl.store(V+dest,vval)
    # Initialize all packed rows in the existing preparation launch. Subsequent
    # attention overwrites exactly the valid rows. This extra full-half store is
    # included in the candidate cost; it defines debug/public invalid padding.
    tl.store(OUT+dest,tl.full((BR,32),0,tl.float16))


@triton.jit
def _attention(Q,K,V,BIAS,MAP_WINDOW,MAP_BIAS,MAP_DST,OUT,
               WPH:tl.constexpr,BM:tl.constexpr,PACKED_OUTPUT:tl.constexpr):
    group=tl.program_id(0);head=tl.program_id(1)
    row=group*BM+tl.arange(0,BM);lane=tl.arange(0,32)
    window=tl.load(MAP_WINDOW+group).to(tl.int32)
    bias_row=tl.load(MAP_BIAS+row).to(tl.int32)
    base=(head*WPH+window)*2048
    q=tl.load(Q+(head*960+row[:,None])*32+lane[None,:])
    k0=tl.load(K+base+lane[None,:]*32+lane[:,None])
    k1=tl.load(K+base+(lane[None,:]+32)*32+lane[:,None])
    score0=tl.dot(q,k0,out_dtype=tl.float32)
    score1=tl.dot(q,k1,out_dtype=tl.float32)
    b0=tl.load(BIAS+head*4096+bias_row[:,None]*64+lane[None,:])
    b1=tl.load(BIAS+head*4096+bias_row[:,None]*64+lane[None,:]+32)
    e0=_exp((score0+b0.to(tl.float32)).to(tl.float16))
    e1=_exp((score1+b1.to(tl.float32)).to(tl.float16))
    p0,p1=_weights_pair(e0,e1,BM,ROUND_WEIGHTS=False)
    v0=tl.load(V+base+lane[:,None]*32+lane[None,:])
    v1=tl.load(V+base+(lane[:,None]+32)*32+lane[None,:])
    result=tl.dot(p0,v0,out_dtype=tl.float32)
    result=tl.dot(p1,v1,result,out_dtype=tl.float32)
    if PACKED_OUTPUT:dst=base+bias_row[:,None]*32+lane[None,:]
    else:
        pixel=tl.load(MAP_DST+row).to(tl.int32)
        dst=(head*960+pixel[:,None])*32+lane[None,:]
    tl.store(OUT+dst,result.to(tl.float16))


@triton.jit
def _encoder_address(row,head,lane,INVERSE,HP:tl.constexpr,WP:tl.constexpr,
                     SY:tl.constexpr,SX:tl.constexpr,PACKED_INPUT:tl.constexpr):
    if PACKED_INPUT:
        y=row//40+SY;x=row%40+SX
        window=(y//8)*(WP//8)+x//8
        slot=tl.load(INVERSE+(y%8)*8+x%8).to(tl.int32)
        return (head*(HP//8)*(WP//8)+window[:,None])*2048+slot[:,None]*32+lane[None,:]
    return (head*960+row[:,None])*32+lane[None,:]


@triton.jit
def _encoder_project(HEADS,W,RESIDUAL,SCALE,INVERSE,OUT,HP:tl.constexpr,
                     WP:tl.constexpr,SY:tl.constexpr,SX:tl.constexpr,
                     PACKED_INPUT:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    col=tl.program_id(1)*BN+tl.arange(0,BN);lane=tl.arange(0,32)
    valid=row[:,None]<960;offset=row[:,None]*512+col[None,:]
    r=tl.load(RESIDUAL+offset,valid,other=0);scale=tl.load(SCALE+col)
    # Distinct from decoder: half residual becomes the FP32 dot INITIAL.
    total=(r.to(tl.float32)*scale[None,:].to(tl.float32)).to(tl.float16).to(tl.float32)
    for head in range(16):
        src=_encoder_address(row,head,lane,INVERSE,HP,WP,SY,SX,PACKED_INPUT)
        a=tl.load(HEADS+src,valid,other=0)
        w=tl.load(W+(head*32+lane[:,None])*512+col[None,:])
        total=tl.dot(a,w,total,out_dtype=tl.float32)
    tl.store(OUT+offset,total.to(tl.float16),valid)


@triton.jit
def _encoder_unpack(HEADS,INVERSE,OUT,HP:tl.constexpr,WP:tl.constexpr,
                    SY:tl.constexpr,SX:tl.constexpr,PACKED_INPUT:tl.constexpr):
    i=tl.program_id(0)*512+tl.arange(0,512)
    row=i//512;head=(i%512)//32;lane=i%32
    if PACKED_INPUT:
        y=row//40+SY;x=row%40+SX
        window=(y//8)*(WP//8)+x//8
        slot=tl.load(INVERSE+(y%8)*8+x%8).to(tl.int32)
        src=(head*(HP//8)*(WP//8)+window)*2048+slot*32+lane
    else:src=(head*960+row)*32+lane
    tl.store(OUT+i,tl.load(HEADS+src,i<491520,other=0),i<491520)


@triton.jit
def _pool_final_impl(FULL,W,POOL,OUT,BM:tl.constexpr,BN:tl.constexpr,WRITE_POOL:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    col=tl.program_id(1)*BN+tl.arange(0,BN);lane=tl.arange(0,32)
    top=(row//20)*80+(row%20)*2;valid=row[:,None]<240
    total=tl.full((BM,BN),0,tl.float32)
    for block in range(16):
        kk=block*32+lane
        a=tl.load(FULL+top[:,None]*512+kk[None,:],valid,other=0)
        b=tl.load(FULL+(top[:,None]+1)*512+kk[None,:],valid,other=0)
        c=tl.load(FULL+(top[:,None]+40)*512+kk[None,:],valid,other=0)
        d=tl.load(FULL+(top[:,None]+41)*512+kk[None,:],valid,other=0)
        pool=(_add_half(_add_half(a,b),_add_half(c,d)).to(tl.float32)*.25).to(tl.float16)
        if WRITE_POOL:
            if tl.program_id(1)==0:tl.store(POOL+row[:,None]*512+kk[None,:],pool,valid)
        w=tl.load(W+kk[:,None]*1024+col[None,:])
        total=tl.dot(pool,w,total,out_dtype=tl.float32)
    tl.store(OUT+row[:,None]*1024+col[None,:],total.to(tl.float16),valid)


@triton.jit
def _pool_final(FULL,W,OUT,BM:tl.constexpr,BN:tl.constexpr):
    _pool_final_impl(FULL,W,OUT,OUT,BM,BN,False)


@triton.jit
def _pool_final_debug(FULL,W,POOL,OUT,BM:tl.constexpr,BN:tl.constexpr):
    _pool_final_impl(FULL,W,POOL,OUT,BM,BN,True)


def pad(mlp,plan):
    sy,sx=plan["shift"];hp,wp=plan["hp"],plan["wp"]
    return torch.nn.functional.pad(mlp,(0,0,sx,wp-40-sx,sy,hp-24-sy))


def attention(module,mlp,plan,counter,*,packed_output):
    import nr_backend.multihead_block as blocks
    from nr_backend.unround_policy import ENABLED,round_multihead
    if module.channels!=512 or "c512" not in ENABLED:
        raise RuntimeError("Only current720 C512 unrounded attention is owned")
    require_half(mlp,(24,40,512),module.qkv.device)
    hp,wp,wph=plan["hp"],plan["wp"],plan["window_count"]
    padded=round_multihead(512,pad(mlp,plan))
    counter.observe_layout("C512 library input",padded)
    z=blocks.sm89_f16_dot(padded,module.qkv,chunk_k=16).reshape(hp,wp,16,3,32)
    q=torch.empty((16,960,32),device=mlp.device,dtype=torch.float16)
    k=torch.empty((16,wph,64,32),device=mlp.device,dtype=torch.float16);v=torch.empty_like(k)
    out=torch.empty_like(k) if packed_output else torch.empty_like(q)
    counter.launch("c512_valid_query",_prepare_query,(60,16),(z,module.scale,plan["source_tensor"],q,16))
    if packed_output:
        counter.launch("c512_full_kv_packed",_prepare_kv_packed,(wph,16,4),
                       (z,module.pixel_order,k,v,out,wph,wp,16))
    else:
        counter.launch("c512_full_kv",_prepare_kv,(wph,16,4),(z,module.pixel_order,k,v,wph,wp,16))
    counter.launch("c512_valid_attention",_attention,(60,16),
        (q,k,v,module.bias,plan["windows_tensor"],plan["bias_rows_tensor"],
         plan["destinations_tensor"],out,wph,16,packed_output))
    return out.reshape(16,hp//8,wp//8,64,32) if packed_output else out


def baseline_packed(stack,module,mlp,plan,counter):
    """Encoder alias, retaining the actual probability-unround child and keys."""
    import nr_backend.multihead_block as blocks
    import native_half_head_layout_v1 as current_layout
    from nr_backend.unround_policy import round_multihead
    padded=round_multihead(512,pad(mlp,plan))
    counter.observe_layout("C512 retained library input",padded)
    hp,wp=plan["hp"],plan["wp"]
    projected=blocks.sm89_f16_dot(padded,module.qkv,chunk_k=16).reshape(hp,wp,16,3,32)
    shape=(16,hp//8,wp//8,64,32)
    q,k,v=[torch.empty(shape,device=mlp.device,dtype=torch.float16) for _ in range(3)]
    br=stack.window_blocks.layout.rows
    counter.launch("c512_retained_packed_qkv",_retained_pack,
        (triton.cdiv(hp*wp*16,br),3),
        (projected,module.scale,module.pixel_order,q,k,v,hp*wp*16,wp,hp//8,wp//8,br))
    # Using decoder_layout.native_heads here would reject an encoder bias.
    # This is the same current encoder alias, so its actual legacy counter fires.
    packed,kernel=current_layout.native_heads(q,k,v,module.bias)
    stack.window_blocks.last_attention_kernel=kernel
    stack.window_blocks.last_attention_selection={"route":"P11 retained encoder probability"}
    return packed


def canonical(module,heads,plan,counter,*,layout):
    if layout=="hwc":return heads
    packed=layout=="packed"
    require_half(heads,(16,plan["hp"]//8,plan["wp"]//8,64,32) if packed else (16,960,32),heads.device)
    out=torch.empty((24,40,512),device=heads.device,dtype=torch.float16)
    counter.launch("c512_encoder_unpack",_encoder_unpack,(960,),
        (heads,module.attention.pixel_inverse,out,plan["hp"],plan["wp"],*plan["shift"],packed))
    return out


def encoder(module,mlp,heads,plan,counter,*,input_layout,direct_projection,
            pool_final,bm=32,bn=64,diagnostics=False):
    if input_layout not in ("headmajor","packed","hwc"):raise ValueError(input_layout)
    attended=None
    if direct_projection:
        if input_layout=="hwc":raise ValueError("Direct encoder requires its packed/head-major producer")
        packed=input_layout=="packed"
        require_half(heads,(16,plan["hp"]//8,plan["wp"]//8,64,32) if packed else (16,960,32),mlp.device)
        full=torch.empty_like(mlp)
        counter.launch("c512_encoder_projection",_encoder_project,
            (triton.cdiv(960,bm),512//bn),
            (heads,module.projection.weight,mlp,module.projection.skip_scale,
             module.attention.pixel_inverse,full,plan["hp"],plan["wp"],*plan["shift"],packed,bm,bn))
        if diagnostics:attended=canonical(module,heads,plan,counter,layout=input_layout)
    else:
        attended=canonical(module,heads,plan,counter,layout=input_layout)
        full=module.projection.forward_unquantized(attended,mlp)
    if module.final_weight is None:return mlp,mlp,attended,full
    if pool_final and counter.options.use_front_pool_helper:
        final=counter.front_pool(full,module.final_weight)
        if diagnostics:
            top=(full[0::2,0::2]+full[0::2,1::2]).half()
            bottom=(full[1::2,0::2]+full[1::2,1::2]).half()
            pool=((top+bottom).half()*.25).half()
        else:pool=None
    elif pool_final:
        final=torch.empty((240,1024),device=mlp.device,dtype=torch.float16)
        if diagnostics:
            pool=torch.empty((240,512),device=mlp.device,dtype=torch.float16)
            counter.launch("c512_encoder_pool_final_debug",_pool_final_debug,
                (triton.cdiv(240,bm),1024//bn),(full,module.final_weight,pool,final,bm,bn))
            pool=pool.reshape(12,20,512)
        else:
            counter.launch("c512_encoder_pool_final",_pool_final,
                (triton.cdiv(240,bm),1024//bn),(full,module.final_weight,final,bm,bn))
            pool=None
        final=final.reshape(12,20,1024)
    else:
        import nr_backend.split_block as split
        top=(full[0::2,0::2]+full[0::2,1::2]).half()
        bottom=(full[1::2,0::2]+full[1::2,1::2]).half()
        pool=split.q(((top+bottom).half()*.25).half())
        # Current12x20 already aligned to4: no canvas semantics are changed.
        final=split.q(split.dot(pool,module.final_weight,chunk_k=16))
    # Only slots3 and last are consumed by current _forward_front. A diagnostic
    # profile materializes the unused canonical HWC slots; main records this
    # explicit boundary schema rather than assuming every returned item is live.
    return mlp,mlp,attended,full,pool,final


EXTRA={
    __name__+"._qkv_native":(("OUT",),()),
    __name__+"._retained_pack":(("Q","K","V"),()),
    __name__+"._prepare_query":(("Q",),()),
    __name__+"._prepare_kv":(("K","V"),()),
    __name__+"._prepare_kv_packed":(("K","V","OUT"),()),
    __name__+"._attention":(("OUT",),()),
    __name__+"._encoder_project":(("OUT",),()),
    __name__+"._encoder_unpack":(("OUT",),()),
    __name__+"._pool_final":(("OUT",),()),
    __name__+"._pool_final_debug":(("POOL","OUT"),()),
}
