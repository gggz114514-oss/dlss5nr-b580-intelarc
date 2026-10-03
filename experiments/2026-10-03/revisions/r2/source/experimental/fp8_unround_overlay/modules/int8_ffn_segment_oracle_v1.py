"""CPU numerical contract for a new approximate continuous INT8 FFN segment.

Dynamic row scale only at entry, static per-channel hidden scale, and the hidden
scale folded into contraction weights before their per-column INT8 quantization.
The intended GPU producer writes INT8 hidden values directly for the consumer.
This CPU oracle is not an executable GPU implementation or a latency estimate.
"""
import numpy as np

_code=np.arange(127,dtype=np.int32)
_exp=_code>>3;_mant=_code&7
_positive=(np.where(_exp==0,_mant,(8+_mant)<<np.maximum(_exp-1,0)).astype(np.float32)/512)


def fp8_boundary(value):
    half=np.asarray(value,dtype=np.float16)
    assert np.isfinite(half).all()
    magnitude=np.minimum(np.abs(half.astype(np.float32)),448)
    upper=np.searchsorted(_positive,magnitude,side='left')
    lower=np.maximum(upper-1,0)
    du=_positive[upper]-magnitude;dl=magnitude-_positive[lower]
    index=np.where((du<dl)|((du==dl)&((upper&1)==0)),upper,lower)
    out=_positive[index].astype(np.float16)
    out.view(np.uint16)[...] |= half.view(np.uint16)&0x8000
    return out


def decode_weights(payload,k,n):
    assert len(payload)==k*n and k%32==n%16==0
    physical=np.frombuffer(payload,dtype=np.uint8).reshape(k//32,n//16,8,4,2,2,2,2)
    codes=physical.transpose(0,5,6,3,7,1,4,2).reshape(k,n)
    assert ((codes&127)!=127).all()
    out=_positive[codes&127].astype(np.float16)
    out.view(np.uint16)[...] |= ((codes.astype(np.uint16)&128)<<8)
    return out


def quantize(value,scale):
    value=np.asarray(value,dtype=np.float32);scale=np.asarray(scale,dtype=np.float32)
    assert np.isfinite(value).all() and np.isfinite(scale).all() and (scale>0).all()
    unit=value/scale
    # Symmetric signed INT8, nearest with ties away from zero; zero point is zero.
    rounded=np.floor(np.abs(unit)+np.float32(.5))*np.where(unit<0,np.float32(-1),np.float32(1))
    return np.clip(rounded,-127,127).astype(np.int8)


def quantize_axis(value,axis):
    value=np.asarray(value,dtype=np.float32)
    maximum=np.max(np.abs(value),axis=axis,keepdims=True)
    scale=np.where(maximum>0,maximum/np.float32(127),np.float32(1))
    return quantize(value,scale),scale


def integer_dot(a,w):
    assert a.dtype==w.dtype==np.dtype('i1') and a.ndim==w.ndim==2 and a.shape[1]==w.shape[0]
    assert a.shape[1]<=4096
    # Every INT8 product and sum is exact in FP64 at these bounded dimensions.
    # The result is then cast to the INT32 accumulator used by the future GPU.
    value=a.astype(np.float64)@w.astype(np.float64)
    assert np.isfinite(value).all() and np.abs(value).max()<2**31
    assert (value==np.rint(value)).all()
    return value.astype(np.int32)


def cubic_half(value):
    x=np.asarray(value,dtype=np.float16)
    assert np.isfinite(x).all()
    t=np.clip(x,-4,4).astype(np.float16)
    p=(-np.abs(t.astype(np.float64))*.055908203125+.447265625).astype(np.float16)
    v=(t.astype(np.float64)*p.astype(np.float64)+.89453125).astype(np.float16)
    return (x.astype(np.float32)*v.astype(np.float32)).astype(np.float16)


def expansion(x,weight):
    qx,sx=quantize_axis(x,1);qw,sw=quantize_axis(weight,0)
    z=(integer_dot(qx,qw).astype(np.float32)*sx)*sw
    hidden=cubic_half(z.astype(np.float16))
    assert np.isfinite(hidden).all()
    return hidden,dict(qx=qx,sx=sx,qw=qw,sw=sw)


def hidden_scale(reference_calibration,margin):
    value=np.asarray(reference_calibration,dtype=np.float32)
    maximum=np.max(np.abs(value),axis=0,keepdims=True)
    # A bounded floor also covers channels absent from this small calibration set.
    floor=np.float32(max(float(maximum.max())/4096,2**-14))
    return np.maximum(maximum*np.float32(margin),floor)/np.float32(127)


def contraction(hidden,scale,weight,initial):
    qh=quantize(hidden,scale)
    folded=weight.astype(np.float32)*scale.reshape(-1,1)
    qw,sw=quantize_axis(folded,0)
    result=integer_dot(qh,qw).astype(np.float32)*sw
    # One complete INT32 sum: intentionally replaces the old four half merges.
    out=(result+initial.astype(np.float32)).astype(np.float16)
    assert np.isfinite(out).all()
    return out,fp8_boundary(out),dict(qh=qh,qw=qw,sw=sw,folded=folded)


def error_metrics(actual,reference):
    a=np.asarray(actual);r=np.asarray(reference)
    assert a.shape==r.shape and np.isfinite(a).all() and np.isfinite(r).all()
    af=a.astype(np.float64);rf=r.astype(np.float64);delta=af-rf
    mse=float(np.mean(delta*delta));energy=float(np.mean(rf*rf))
    return dict(rmse=float(np.sqrt(mse)),relative_rmse=float(np.sqrt(mse/max(energy,1e-30))),
        mean_absolute_error=float(np.mean(np.abs(delta))),max_absolute_error=float(np.abs(delta).max()),
        p95_absolute_error=float(np.percentile(np.abs(delta),95)))
