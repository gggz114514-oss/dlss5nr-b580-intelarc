"""Complete native EX2 exposure domain, represented by lossless constant runs."""
from array import array
from bisect import bisect_right
import hashlib,struct,sys
from pathlib import Path

def float32(value):return struct.unpack('<f',struct.pack('<f',value))[0]

class NativeExposureTable:
    FILE='exposure.runs.u32x2.bin'
    SIZE=6710888
    SHA256='11183662a4b164ad499da0b07854f332899d40a39e2e86255174a61313e42c82'
    def __init__(self,path):
        data=Path(path).read_bytes()
        if len(data)!=self.SIZE or hashlib.sha256(data).hexdigest()!=self.SHA256:raise ValueError('Native exposure scalar asset mismatch')
        values=array('I');values.frombytes(data)
        if sys.byteorder!='little':values.byteswap()
        self._starts=values[::2];self._outputs=values[1::2]

    def __call__(self,exposure):
        # The coefficient is a host scalar parameter, never a frame transfer.
        exposure=float32(exposure)
        if not -float32(.1)<=exposure<=0:raise ValueError('Exposure coefficient outside complete native domain')
        bits=struct.unpack('<I',struct.pack('<f',exposure))[0]|0x80000000
        index=bisect_right(self._starts,bits)-1
        return struct.unpack('<f',struct.pack('<I',self._outputs[index]))[0]
