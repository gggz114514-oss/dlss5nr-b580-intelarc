"""Bounded CPU NumPy comparison with original sequential float64 reduction.

Import is stdlib-only; NumPy is imported lazily in an ordinary CPU process.
API/result fields match the frozen streaming reference. No full-array load,
Torch, runtime bootstrap, GPU/device API or model code is used.
"""
from __future__ import annotations
import ast
import hashlib
import importlib
import math
import struct
import sys
from pathlib import Path
import metrics_common as c

BLOCK_SCALARS=262144

def cpu_numpy():
    c.require('torch' not in sys.modules and Path(sys.executable).drive.upper()!='G:',
              'CPU metrics cannot run inside the XPU runtime/Torch worker')
    np=importlib.import_module('numpy')
    c.require(Path(np.__file__).drive.upper()!='G:','No NumPy import from isolated XPU runtime')
    return np

def exact_read(stream,n,message):
    block=stream.read(n);c.require(len(block)==n,message);return block

def header(stream):
    c.require(exact_read(stream,6,'Truncated NPY magic')==b'\x93NUMPY','Not NPY')
    version=tuple(exact_read(stream,2,'Truncated NPY version'))
    c.require(version in ((1,0),(2,0),(3,0)),'Unsupported NPY version')
    width=2 if version==(1,0) else 4
    count=struct.unpack('<H' if width==2 else '<I',exact_read(stream,width,'Truncated NPY header size'))[0]
    c.require(count<=4096,'Oversized NPY header')
    row=ast.literal_eval(exact_read(stream,count,'Truncated NPY header').decode('utf-8' if version==(3,0) else 'latin1'))
    c.require(isinstance(row,dict) and set(row)=={'descr','fortran_order','shape'} and
              row['fortran_order'] is False and row['descr'] in ('<f2','<f4','<f8'),
              'Only contiguous little-endian float output accepted')
    c.require(type(row['shape']) is tuple and row['shape'] and all(type(x) is int and x>0 for x in row['shape']),
              'Positive nonempty integer NPY shape required')
    return row

def _sequential_sum(np,array,carry):
    # Preserve every IEEE64 addition from the streaming Python loop across
    # chunk boundaries. np.sum would use a different pairwise reduction tree.
    array[0]=float(carry)+float(array[0])
    np.cumsum(array,dtype=np.float64,out=array)
    return float(array[-1])

def compare_pair(reference,candidate,*,shape=c.FRAME_SHAPE,block_scalars=BLOCK_SCALARS):
    np=cpu_numpy()
    c.require(type(block_scalars) is int and 1<=block_scalars<=1048576,'Bounded scalar chunk required')
    c.require(shape and all(type(x) is int and x>0 for x in shape),'Positive comparison shape required')
    channels=shape[-1];c.require(channels<=block_scalars,'Channel width exceeds bounded chunk')
    ref_path,path=c.checked(reference),c.checked(candidate)
    sums=squares=maximum=0.0;changed_values=changed_pixels=count=0
    hashes=(hashlib.sha256(),hashlib.sha256())
    with ref_path.open('rb') as left,path.open('rb') as right:
        ha,hb=header(left),header(right)
        c.require(ha['shape']==hb['shape']==tuple(shape),'Comparison shape mismatch')
        c.require(ha['descr']==reference['dtype'] and hb['descr']==candidate['dtype'],'Saved dtype receipt differs')
        remaining=math.prod(shape);limit=block_scalars-block_scalars%channels
        da,db=np.dtype(ha['descr']),np.dtype(hb['descr'])
        while remaining:
            size=min(limit,remaining)
            ba=exact_read(left,size*da.itemsize,'Truncated NPY');bb=exact_read(right,size*db.itemsize,'Truncated NPY')
            hashes[0].update(ba);hashes[1].update(bb)
            va=np.frombuffer(ba,dtype=da);vb=np.frombuffer(bb,dtype=db)
            c.require(bool(np.isfinite(va).all()) and bool(np.isfinite(vb).all()),'Nonfinite saved output/private history')
            # Validation still reads/hashes every payload byte, even when an
            # identical block needs no numerical work. Signed zeros continue
            # to use floating comparison, not a raw-byte changed-pixel count.
            changed=np.not_equal(va,vb)
            differences=int(np.count_nonzero(changed))
            changed_values+=differences
            changed_pixels+=int(np.count_nonzero(changed.reshape(-1,channels).any(axis=1)))
            if differences:
                a=va.astype(np.float64,copy=True);delta=vb.astype(np.float64,copy=True)
                with np.errstate(over='ignore',invalid='ignore'):
                    np.subtract(delta,a,out=delta);np.abs(delta,out=delta)
                    maximum=max(maximum,float(np.max(delta)))
                    np.multiply(delta,delta,out=a)
                    sums=_sequential_sum(np,delta,sums);squares=_sequential_sum(np,a,squares)
                del a,delta
            count+=size;remaining-=size
        c.require(not left.read(1) and not right.read(1),'Unexpected trailing NPY bytes')
    for row,digest in zip((reference,candidate),hashes):
        c.require(row['raw_sha256']==digest.hexdigest(),'Raw SHA receipt differs from saved bytes')
    mse=squares/count
    identical=ha['descr']==hb['descr'] and hashes[0].digest()==hashes[1].digest()
    return dict(finite=True,nan_count=0,inf_count=0,byte_identical=identical,mae=sums/count,
        max_abs_error=maximum,mse=mse,rmse=math.sqrt(mse),psnr_peak_1_db=None if mse==0 else -10*math.log10(mse),
        psnr_zero_error=mse==0,changed_values=changed_values,changed_pixels=changed_pixels,scalar_count=count,
        pixel_count=count//shape[-1],reference_raw_sha256=reference['raw_sha256'],candidate_raw_sha256=candidate['raw_sha256'])

def compare_frames(reference,candidate,*,shape=c.FRAME_SHAPE,block_scalars=BLOCK_SCALARS):
    c.require(len(reference)==len(candidate) and reference,'Unequal input sequence lengths')
    rows=[]
    for ref,cand in zip(reference,candidate):
        keys=('frame_id','reset','original_rgb_raw_sha256','original_motion_raw_sha256','seed_after')
        c.require(all(ref[k]==cand[k] for k in keys),'Comparison uses different source/reset/control/history sequences')
        rows.append(dict(frame_id=ref['frame_id'],output=compare_pair(ref['output'],cand['output'],shape=shape,block_scalars=block_scalars),
                         history=compare_pair(ref['history'],cand['history'],shape=shape,block_scalars=block_scalars)))
    aggregate={}
    for role in ('output','history'):
        reports=[r[role] for r in rows];scalar_count=sum(r['scalar_count'] for r in reports)
        mse=sum(r['mse']*r['scalar_count'] for r in reports)/scalar_count
        aggregate[role]=dict(finite=True,all_byte_identical=all(r['byte_identical'] for r in reports),
            mae=sum(r['mae']*r['scalar_count'] for r in reports)/scalar_count,
            max_abs_error=max(r['max_abs_error'] for r in reports),mse=mse,
            psnr_peak_1_db=None if mse==0 else -10*math.log10(mse),psnr_zero_error=mse==0,
            changed_pixels=sum(r['changed_pixels'] for r in reports),changed_values=sum(r['changed_values'] for r in reports))
    return dict(frame_count=len(rows),aggregate=aggregate,per_frame=rows)
