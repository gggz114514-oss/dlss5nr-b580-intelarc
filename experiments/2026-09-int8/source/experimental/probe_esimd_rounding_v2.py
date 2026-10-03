"""Localize the ESIMD/random-half mismatch without changing the selected runtime.

The native diagnostic keeps the v3 K16 DPAS loop and stores the FP32 result
before conversion. Triton probes vary only BK16/BK32, saving the accumulator,
post-initial FP32 value, and final half. Added stores are now known to change
the half result, so the uninstrumented baseline's actual loaded native binary
is also saved. This is not a timing or promotion gate.
"""
import ctypes as c
import hashlib,json,os,sys,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;R=ROOT.parent/'nr-b580/reference'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'experimental/esimd-rounding-v2';assert not OUT.exists()
TC=D/'toolchains/triton-xpu-3.8.0-git1e2d42a0'
os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-c32-triton38-v1')
sys.path[:0]=[str(TC/'site'),str(R),str(ROOT/'backend')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main()
original_path=D/'experimental/esimd-dense-v3/validation.json'
assert sha(original_path)=='8b8214b33aea961cab0655913748b9a42b655d4c38da6cbc4b42ba38474cd6eb'
original=js(original_path);assert original['passed'] and original['general_half_equivalence_rejected']
assert all(sha(p)==h for p,h in original['sources'].items())
assert js(original_path.parent.with_suffix('.log.lease.json'))['returncode']==0
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:q=json.load(response)
assert not q['queue_running'] and not q['queue_pending']
import numpy as np
import torch
import triton
import triton.language as tl
assert Path(triton.__file__).is_relative_to(TC/'site')
from esimd_dense_adapter_v2 import Dense
from dense_tiles_v1 import dot
import compressed_arrays_v1 as arrays
from immutable_artifacts_v1 import put


@triton.jit
def _probe(A,W,I,S,F,H,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BK:tl.constexpr):
    rows=tl.program_id(0)*16+tl.arange(0,16)
    cols=tl.program_id(1)*32+tl.arange(0,32)
    kk=tl.arange(0,BK)
    total=tl.full((16,32),0,tl.float32)
    for start in range(0,tl.cdiv(K,BK)):
        k=start*BK+kk
        av=tl.load(A+rows[:,None]*K+k[None,:],(rows[:,None]<M)&(k[None,:]<K),other=0)
        wv=tl.load(W+k[:,None]*N+cols[None,:],(k[:,None]<K)&(cols[None,:]<N),other=0)
        total=tl.dot(av,wv,total,out_dtype=tl.float32)
    offset=rows[:,None]*N+cols[None,:];valid=(rows[:,None]<M)&(cols[None,:]<N)
    value=total+tl.load(I+offset,valid,other=0).to(tl.float32)
    tl.store(S+offset,total,valid);tl.store(F+offset,value,valid)
    tl.store(H+offset,value.to(tl.float16),valid)


dll=D/'experimental/esimd-dense-accum-build-v2/esimd_dense_accum_v2.dll'
sources={**original['sources'],str(original_path):sha(original_path)}
for p in (Path(__file__),HERE/'Run-EsimdRoundingV2.cmd',HERE/'Build-EsimdDenseAccumV2.cmd',HERE/'esimd_dense_accum_v2.cpp',dll,dll.parent/'build.log'):
    sources[str(p)]=sha(p)
OUT.mkdir();report=dict(scope=__doc__,passed=False,complete_migration=False,candidate_promoted=False,sources=sources,exact_gate=gate,cases=[])
save=lambda:(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
raw=lambda t:t.cpu().numpy().tobytes()

def binary_of(engine,kernel):
    get=c.pythonapi.PyCapsule_GetPointer;get.argtypes=[c.py_object,c.c_char_p];get.restype=c.c_void_p
    pointer=get(kernel.module,b'kernel_bundle')
    fn=engine.library.nr_bundle_binary
    fn.argtypes=[c.c_void_p,c.c_void_p,c.c_size_t,c.POINTER(c.c_size_t)];fn.restype=c.c_int
    size=c.c_size_t();engine.check(fn(pointer,None,0,c.byref(size)))
    assert 0<size.value<16*2**20
    buf=c.create_string_buffer(size.value);engine.check(fn(pointer,buf,len(buf),c.byref(size)))
    return put(buf.raw[:size.value],'zebin')

try:
    torch.set_num_threads(2)
    with torch.inference_mode():
        half_engine=Dense(D/'experimental/esimd-dense-build-v4/esimd_dense_v4.dll')
        accum_engine=Dense(dll);report['accum_resources']=accum_engine.resources
        binary=OUT/'native_accum.bin';binary.write_bytes(accum_engine.binary());report['native_binary']=dict(path=str(binary),sha256=sha(binary))
        for old in original['smoke']:
            m,k,n=old['shape'];cpu={name:arrays.load(old[name]) for name in ('a','w','initial')}
            a=torch.from_numpy(cpu['a'].copy()).to('xpu');w=torch.from_numpy(cpu['w'].copy()).to('xpu');initial=torch.from_numpy(cpu['initial'].copy()).to('xpu')
            b=half_engine.pack_weight(cpu['w']);pa=half_engine.pack_activation(a)
            baseline,baseline_kernel=dot(a,w,initial=initial)
            assert raw(baseline)==arrays.load(old['routes']['True']['reference']).tobytes()
            row=dict(shape=[m,k,n],inputs={name:old[name] for name in cpu},native={},triton={});report['cases'].append(row)
            row['baseline']=dict(half=arrays.save(baseline.cpu().numpy()),llir=put(baseline_kernel.asm['llir'].encode(),'llir'),native=binary_of(accum_engine,baseline_kernel),build_flags=baseline_kernel.metadata.build_flags)
            outputs={}
            for packed in (False,True):
                accum_engine.require(packed);av=pa if packed else a
                final=torch.empty((m,n),device='xpu',dtype=torch.float32);summed=torch.empty_like(final)
                for ini,out in ((initial,final),(None,summed)):
                    accum_engine.check(accum_engine.library.nr_esimd_dense(torch.xpu.current_stream().sycl_queue,av.data_ptr(),b.data_ptr(),None if ini is None else ini.data_ptr(),out.data_ptr(),m,k,n,int(packed)))
                h=torch.empty((m,n),device='xpu',dtype=torch.float16)
                half_engine.into(av,b,initial,h,m=m,k=k,n=n,packed=packed)
                fc=final.cpu().numpy();sc=summed.cpu().numpy();hc=h.cpu().numpy()
                outputs['native_'+str(packed)]=(sc,fc,hc)
                row['native'][str(packed)]=dict(sum=arrays.save(sc),post_initial=arrays.save(fc),half=arrays.save(hc),fp32_to_half_matches_original=fc.astype('f2').tobytes()==hc.tobytes())
                assert row['native'][str(packed)]['fp32_to_half_matches_original']
            for bk in (16,32):
                summed=torch.empty((m,n),device='xpu',dtype=torch.float32);final=torch.empty_like(summed);h=torch.empty_like(baseline)
                kernel=_probe[(triton.cdiv(m,16),triton.cdiv(n,32))](a,w,initial,summed,final,h,m,n,k,bk,num_warps=4,enable_fp_fusion=False)
                sc=summed.cpu().numpy();fc=final.cpu().numpy();hc=h.cpu().numpy();outputs['triton_'+str(bk)]=(sc,fc,hc)
                row['triton'][str(bk)]=dict(sum=arrays.save(sc),post_initial=arrays.save(fc),half=arrays.save(hc),llir=put(kernel.asm['llir'].encode(),'llir'),spirv=put(kernel.asm['spv'],'spv'),native=put(kernel.asm['zebin'],'zebin') if 'zebin' in kernel.asm else None,asm_keys=list(kernel.asm),matches_original=hc.tobytes()==raw(baseline))
                row['triton'][str(bk)]['native']=binary_of(accum_engine,kernel)
                # Added stores can change code generation; preserve this observation.
            row['comparisons']={}
            for name,value in outputs.items():
                row['comparisons'][name]={other:[int(np.count_nonzero(x.view('u'+str(x.dtype.itemsize))!=y.view('u'+str(y.dtype.itemsize)))) for x,y in zip(value,against)] for other,against in outputs.items()}
            native_half=outputs['native_True'][2];reference_half=baseline.cpu().numpy()
            bad=np.argwhere(native_half.view('u2')!=reference_half.view('u2'))
            exact=cpu['a'].astype('f8')@cpu['w'].astype('f8')+cpu['initial'].astype('f8')
            row['differing_elements']=[dict(index=[int(i),int(j)],exact_f64=float(exact[i,j]),original_half=float(reference_half[i,j]),values={name:dict(sum=float(v[0][i,j]),post_initial=float(v[1][i,j]),half=float(v[2][i,j])) for name,v in outputs.items()}) for i,j in bad]
            assert all(raw(t)==cpu[name].tobytes() for name,t in (('a',a),('w',w),('initial',initial)))
            row['inputs_unchanged']=True;save();print(json.dumps(dict(shape=row['shape'],comparisons=row['comparisons'],differences=row['differing_elements'])),flush=True)
        report['passed']=True
except Exception as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    try:assert all(sha(p)==h for p,h in sources.items());authenticate_main()
    except Exception as error:report.update(passed=False,finalization_error=repr(error));raise
    finally:save()
