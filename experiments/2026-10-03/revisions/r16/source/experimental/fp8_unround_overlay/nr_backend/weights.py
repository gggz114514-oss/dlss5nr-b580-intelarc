"""Bounded reader for the pinned SF-v2 serialized record envelope."""
import hashlib,struct
from pathlib import Path

WEIGHTS_SHA256='836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4'


def load_pinned_records(path: str|Path) -> dict[str,bytes]:
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=WEIGHTS_SHA256:raise ValueError('Backend requires the validated SF-v2 weights')
    position=0
    def take(size):
        nonlocal position
        if size<0 or position+size>len(data):raise ValueError('Truncated weight record')
        result=data[position:position+size];position+=size;return result
    if struct.unpack('<Q',take(8))[0]!=len(data):raise ValueError('Serialized size mismatch')
    records={}
    while position<len(data):
        length=struct.unpack('<Q',take(8))[0];name=take(length).decode('ascii')
        outer,inner,size,present=struct.unpack('<QQQI',take(28));payload=take(size)
        device,rank,dtype,count=struct.unpack('<QIII',take(20))
        if name in records or outer!=inner or outer!=size+40 or size!=2*count:raise ValueError('Malformed weight envelope')
        records[name]=payload
    if len(records)!=153:raise ValueError('Unexpected pinned record count')
    return records
