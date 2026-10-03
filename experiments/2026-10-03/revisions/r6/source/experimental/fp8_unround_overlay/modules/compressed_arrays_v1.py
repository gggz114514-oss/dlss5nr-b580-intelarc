"""Complete lossless array storage for long sequences, sharing identical contents."""
import hashlib,io,zlib
from pathlib import Path
import numpy as np
from immutable_artifacts_v1 import put

def save(value):
    value=np.asarray(value)
    if value.dtype.hasobject:raise ValueError('Object arrays are not supported')
    stream=io.BytesIO()
    np.save(stream,value,allow_pickle=False)
    serialized=stream.getvalue()
    meta=put(zlib.compress(serialized,level=1),'npyz')
    meta.update(format='npy+zlib',compression_level=1,npy_sha256=hashlib.sha256(serialized).hexdigest(),shape=list(value.shape),dtype=value.dtype.str,raw_bytes=value.nbytes,raw_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
    return meta

def load(meta):
    data=Path(meta['path']).read_bytes()
    if len(data)!=meta['stored_bytes'] or hashlib.sha256(data).hexdigest()!=meta['sha256']:
        raise ValueError('Compressed artifact changed')
    serialized=zlib.decompress(data)
    if hashlib.sha256(serialized).hexdigest()!=meta['npy_sha256']:
        raise ValueError('Uncompressed NPY changed')
    value=np.load(io.BytesIO(serialized),allow_pickle=False)
    if list(value.shape)!=meta['shape'] or value.dtype.str!=meta['dtype'] or value.nbytes!=meta['raw_bytes'] or hashlib.sha256(value.tobytes(order='C')).hexdigest()!=meta['raw_sha256']:
        raise ValueError('Array data mismatch')
    return value
