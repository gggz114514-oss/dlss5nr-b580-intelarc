"""Keep complete experiment bytes once per content digest; never overwrite evidence."""
import hashlib
import io
from pathlib import Path
import re
import numpy as np

ROOT = Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental/immutable-artifacts-v1')

def put(data, kind):
    if not re.fullmatch('[a-z0-9]+', kind):
        raise ValueError('Invalid artifact suffix')
    digest = hashlib.sha256(data).hexdigest()
    folder = ROOT / kind
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (digest + '.' + kind)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'Immutable artifact changed: {path}')
    else:
        with path.open('xb') as f:
            f.write(data)
    return dict(path=str(path), sha256=digest, stored_bytes=len(data))

def array(value):
    value = np.asarray(value)
    if value.dtype.hasobject:
        raise ValueError('Object arrays are not supported')
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    meta = put(stream.getvalue(), 'npy')
    meta.update(shape=list(value.shape), dtype=value.dtype.str, raw_bytes=value.nbytes,
        raw_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
    return meta

def text(value, kind):
    return put(value.encode('utf-8'), kind)
