"""CPU feasibility screen for FP8-preserving INT8 execution, not a speed test.

Inspect all 16 C512 and eight ViT blocks from authenticated body-boundary data.
Each K32 activation row and weight column gets the largest exact power-of-two
common divisor. Measure signed INT8 fit and balanced base256 plane counts.
No new rounding, dropped values, GPU execution or model/output modification.
FP32 accumulation order and negative-zero behavior of a future dot still need
separate GPU proof; exact operand representation alone is not output parity.
"""
import hashlib,importlib.util,json,math,os
from pathlib import Path
import sys,time,traceback
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT=D/'experimental/fp8-int8-tiles-v1';assert not OUT.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
checkpoint=D/'experimental/layout-crop-checkpoint-v1/saved-audit-v1.json'
assert sha(checkpoint)=='f56a78d9260e8606e7a0faa280226c8308e770c66651266d72d258218d84fd05'
audit=js(checkpoint);assert audit['passed'] and audit['candidate_promoted']
sources=dict(audit['sources'])
for p,h in audit['new_files'].items():sources[str(ROOT/p)]=h
sources[str(checkpoint)]=sha(checkpoint)
for p in (Path(__file__),HERE/'Run-Fp8Int8TilesV1.cmd'):sources[str(p)]=sha(p)
assert all(sha(p)==h for p,h in sources.items())
pins={
 'vit':('vit-head-layout-body-v1','95faf6bb6818627d30f1aa5cf53fd0b30c01050d0130e6705dce96af206a2c26'),
 'c512':('c512-window-layout-body-v1','d538d84f920fd39b31403d49996b6eceb02f4c0576a33cd670309a5d11b66268'),
 'dense':('current-dense-operands-v2','46a16ba64e9f84425928d0a0fd93862e99574910b031ba098c789f0ca69e7290')}
data={}
for name,(folder,h) in pins.items():
    p=D/'experimental'/folder/'validation.json';assert sha(p)==h
    r=js(p);assert r['passed'];data[name]=r;sources[str(p)]=h
    for p,h in r['sources'].items():
        assert sha(p)==h
        if p in sources:assert sources[p]==h
        sources[p]=h
import numpy as np
import compressed_arrays_v1 as arrays
spec=importlib.util.spec_from_file_location('pinned_weight_reader',ROOT/'backend/nr_backend/weights.py')
reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
weights_path=EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin'
records=reader.load_pinned_records(weights_path)
sources[str(weights_path)]=sha(weights_path)
sources[str(ROOT/'backend/nr_backend/weights.py')]=sha(ROOT/'backend/nr_backend/weights.py')
codes=np.arange(256,dtype=np.int32);exp=(codes>>3)&15;mant=codes&7
finite=(codes&127)!=127
units=np.where(exp==0,mant,(8+mant)<<np.maximum(exp-1,0))
units=np.where(codes&128,-units,units).astype(np.int32)
table=(units.astype(np.float32)/512).astype(np.float16)
table.view(np.uint16)[codes==128]=0x8000
table[~finite]=np.nan
valid_half=np.zeros(65536,dtype=bool);valid_half[table[finite].view(np.uint16)]=True
report=dict(scope=__doc__,passed=False,no_gpu_execution=True,new_quantization=False,
    complete_migration=False,candidate_promoted=False,sources=sources,cases=[],artifacts={},
    representation='FP8 units of 2^-9; per K32 row/column exact binary scale; signed INT8 balanced base256 digits',
    limitations=['One fixed, real temporal body input; not calibration coverage for unseen frames.',
                'Counts estimate integer dot work only, excluding conversion/scales/layout/launch overhead.',
                'No claim of FP16/FP32 accumulator equivalence, speedup or production INT8 support.'])
OUT.mkdir();save=lambda:(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


def get(meta):
    value=arrays.load(meta)
    assert value.dtype==np.dtype('f2') and np.isfinite(value).all()
    report['artifacts'][meta['path']]=meta['sha256']
    return value


def unit_values(value):
    assert value.dtype==np.dtype('f2') and valid_half[value.view(np.uint16)].all(),'Non-FP8 input at selected boundary'
    scaled=value.astype(np.float32)*512
    out=scaled.astype(np.int32)
    assert np.array_equal(out.astype(np.float32),scaled)
    return out


def decode(payload,k,n):
    assert len(payload)==k*n
    physical=np.frombuffer(payload,dtype=np.uint8).reshape(k//32,n//16,8,4,2,2,2,2)
    code=physical.transpose(0,5,6,3,7,1,4,2).reshape(k,n)
    assert finite[code].all()
    return units[code],table[code]


def grouped(value,axis):
    # OR gives the smallest trailing-zero count of all nonzero integer entries.
    ored=np.bitwise_or.reduce(np.abs(value),axis=axis)
    low=ored & -ored
    shifts=np.zeros(ored.shape,dtype=np.int32)
    np.copyto(shifts,np.log2(np.maximum(low,1)).astype(np.int32),where=ored!=0)
    lo=np.min(value,axis=axis)>>shifts;hi=np.max(value,axis=axis)>>shifts
    direct=(lo>=-128)&(hi<=127)
    current_lo,current_hi=lo.copy(),hi.copy()
    planes=np.ones(lo.shape,dtype=np.int32)
    for _ in range(3):
        more=(current_lo<-128)|(current_hi>127)
        if not more.any():break
        current_lo=np.where(more,(current_lo+128)//256,current_lo)
        current_hi=np.where(more,(current_hi+128)//256,current_hi)
        planes+=more
    assert ((current_lo>=-128)&(current_hi<=127)).all()
    return dict(shifts=shifts,direct=direct,planes=planes,all_zero=ored==0)


def stats(result):
    return dict(groups=int(result['direct'].size),direct_int8_groups=int(result['direct'].sum()),
        direct_int8_fraction=float(result['direct'].mean()),all_zero_groups=int(result['all_zero'].sum()),
        plane_histogram={str(i):int((result['planes']==i).sum()) for i in range(1,4)},
        shift_histogram={str(int(i)):int((result['shifts']==i).sum()) for i in np.unique(result['shifts'])})


def verify_reconstruction(value,result,axis):
    divisor=np.expand_dims(result['shifts'],axis)
    q=value>>divisor
    assert np.array_equal(q<<divisor,value),'Binary scale lost a value'
    remainder=q.copy();recovered=np.zeros(q.shape,dtype=np.int64)
    for plane in range(int(result['planes'].max())):
        digit=((remainder+128)%256-128).astype(np.int8)
        recovered+=digit.astype(np.int64)*(256**plane)
        remainder=(remainder-digit.astype(np.int32))//256
    assert not remainder.any() and np.array_equal(recovered<<divisor,value)


old_weight_hashes={r['operands']['w']['raw_sha256'] for r in data['dense']['cases'].values()}
weight_matches=set()


def case(name,a,payload,k,n):
    w,decoded=decode(payload,k,n)
    for value in (decoded,decoded[:k//2]):
        h=hashlib.sha256(value.tobytes()).hexdigest()
        if h in old_weight_hashes:weight_matches.add(h)
    aa=unit_values(a).reshape(-1,k);m=aa.shape[0]
    assert k%32==0 and m%16==0 and n%32==0
    ag=grouped(aa.reshape(m,k//32,32),2)
    wg=grouped(w.reshape(k//32,32,n),1)
    verify_reconstruction(aa.reshape(m,k//32,32),ag,2)
    verify_reconstruction(w.reshape(k//32,32,n),wg,1)
    # A complete BM16/BN32/K32 dot can use one integer instruction group only
    # if every participating row and column has one exact signed INT8 plane.
    af=ag['direct'].reshape(m//16,16,k//32).all(axis=1)
    wf=wg['direct'].reshape(k//32,n//32,32).all(axis=2)
    az=ag['all_zero'].reshape(m//16,16,k//32).all(axis=1)
    wz=wg['all_zero'].reshape(k//32,n//32,32).all(axis=2)
    ap=ag['planes'].reshape(m//16,16,k//32).max(axis=1)
    wp=wg['planes'].reshape(k//32,n//32,32).max(axis=2)
    total=(m//16)*(k//32)*(n//32)
    active=int(np.sum((~az).sum(axis=0,dtype=np.int64)*(~wz).sum(axis=1,dtype=np.int64)))
    covered=int(np.sum((af&~az).sum(axis=0,dtype=np.int64)*(wf&~wz).sum(axis=1,dtype=np.int64)))
    plane_work=int(np.sum(np.where(az,0,ap).sum(axis=0,dtype=np.int64)*np.where(wz,0,wp).sum(axis=1,dtype=np.int64)))
    scalar_total=int(np.sum((~ag['all_zero']).sum(axis=0,dtype=np.int64)*(~wg['all_zero']).sum(axis=1,dtype=np.int64)))
    scalar_covered=int(np.sum((ag['direct']&~ag['all_zero']).sum(axis=0,dtype=np.int64)*(wg['direct']&~wg['all_zero']).sum(axis=1,dtype=np.int64)))
    row=dict(name=name,shape=[m,k,n],multiply_add_pairs=m*k*n,activation=stats(ag),weight=stats(wg),
        int8_fit_scalar_k32_outputs=scalar_covered,scalar_k32_outputs=scalar_total,
        int8_fit_scalar_fraction=scalar_covered/scalar_total if scalar_total else None,
        full_tiles=total,active_tiles=active,zero_product_tiles=total-active,
        direct_int8_tiles=covered,direct_tile_fraction=covered/active if active else 0.0,
        balanced_plane_dot_tiles=plane_work,plane_dot_work_factor=plane_work/active if active else 0.0,
        negative_zero_activations=int((a.view(np.uint16)==0x8000).sum()),
        negative_zero_weights=int((decoded.view(np.uint16)==0x8000).sum()),
        exact_integer_reconstruction=True)
    report['cases'].append(row)
    print(json.dumps(dict(case=name,direct_tile_fraction=row['direct_tile_fraction'],
        plane_dot_work_factor=row['plane_dot_work_factor'])),flush=True)


try:
    started=time.perf_counter()
    # Exhaust all finite E4M3 encodings. A zero-sign sidecar is required to
    # reconstruct operand bytes; it is not evidence about dot signed zeros.
    u=units[finite].reshape(1,-1)
    g=grouped(u,1);verify_reconstruction(u,g,1)
    restored=(u.astype(np.float32)/512).astype(np.float16).reshape(-1)
    restored.view(np.uint16)[(codes[finite]&127)==0]|=((codes[finite][(codes[finite]&127)==0]&128)<<8).astype(np.uint16)
    assert restored.tobytes()==table[finite].tobytes()
    report['finite_fp8_encodings_reconstructed']=int(finite.sum())
    report['signed_zero_requires_side_metadata']=True
    for row in data['c512']['boundaries']:
        group,index=row['block'].split('.');index=int(index)
        block=(23 if group=='encoder512' else 40)+index
        record=lambda layer:records[f'block{block}.layer{layer}.layer']
        a=get(row['input']);ffwd=get(row['reference'][0]);mlp=get(row['reference'][1]);attended=get(row['reference'][2])
        case(row['block']+'.ffwd_linear',a,record(0)[:262144],512,512)
        case(row['block']+'.ffwd_projection',ffwd,record(1)[:262144],512,512)
        sy,sx=row['shift'];h,w=mlp.shape[:2]
        padded=np.pad(mlp,((sy,(-h-sy)%8),(sx,(-w-sx)%8),(0,0)))
        case(row['block']+'.qkv',padded,record(2)[:786432],512,1536)
        case(row['block']+'.attention_projection',attended,record(3)[:262144],512,512)
    for row in data['vit']['boundaries']:
        index=int(row['block']);block=31+index
        record=lambda layer:records[f'block{block}.layer{layer}.layer']
        a=get(row['input']);hidden=get(row['reference'][0]);mlp=get(row['reference'][1]);attended=get(row['reference'][5])
        case(f'vit.{index}.expand',a,record(0)[:4194304],1024,4096)
        case(f'vit.{index}.contract',hidden,record(1)[:4194304],4096,1024)
        case(f'vit.{index}.qkv',mlp,record(2)[128:],1024,3072)
        case(f'vit.{index}.attention_projection',attended,record(4)[:1048576],1024,1024)
    assert len(report['cases'])==96
    assert len(weight_matches)>=4,'Decoded layout not cross-checked against enough captured weight buffers'
    report['decoded_weight_hashes_matching_prior_capture']=sorted(weight_matches)
    report['families']={}
    for family in ('c512','vit'):
        rows=[r for r in report['cases'] if r['name'].startswith('vit.')==(family=='vit')]
        total=sum(r['active_tiles'] for r in rows)
        report['families'][family]=dict(cases=len(rows),direct_tile_fraction=sum(r['direct_int8_tiles'] for r in rows)/total,
            plane_dot_work_factor=sum(r['balanced_plane_dot_tiles'] for r in rows)/total)
    report['best_direct_tile_cases']=sorted([dict(name=r['name'],fraction=r['direct_tile_fraction']) for r in report['cases']],key=lambda r:r['fraction'],reverse=True)[:12]
    report['elapsed_cpu_seconds']=time.perf_counter()-started
    assert all(sha(p)==h for p,h in sources.items())
    report['passed']=True
except BaseException:
    report['error']=traceback.format_exc();raise
finally:save()
print(json.dumps(dict(passed=report['passed'],families=report.get('families'),no_gpu_execution=True)),flush=True)
