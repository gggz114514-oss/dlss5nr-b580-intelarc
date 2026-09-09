"""Asset-free demonstration of the recovered half-FMA midpoint counterexample.

Finite binary16 only. The integer oracle keeps products/addition in units of
2**-48 and performs one ties-to-even rounding. It is independent of the backend.
"""
import json
import struct

def units(bits):
    exponent=(bits>>10)&31
    if exponent==31:
        raise ValueError('This oracle accepts finite binary16 operands only')
    mantissa=bits&1023
    return (-1 if bits&0x8000 else 1)*(mantissa if exponent==0 else (mantissa+1024)<<(exponent-1))

def oracle(a,b,c):
    value=units(a)*units(b)+(units(c)<<24)
    magnitude=abs(value)
    if not magnitude:
        return 0x8000 if ((a^b)&c&0x8000) else 0
    shift=max(magnitude.bit_length()-11,24)
    quotient,remainder=divmod(magnitude,1<<shift)
    midpoint=1<<(shift-1)
    quotient+=int(remainder>midpoint or (remainder==midpoint and quotient&1))
    bits=min(0x7c00,(shift-24)*1024+quotient)
    return bits|(0x8000 if value<0 else 0)

def half_bits(value):
    return struct.unpack('<H',struct.pack('<e',value))[0]

def half_value(bits):
    return struct.unpack('<e',struct.pack('<H',bits))[0]

def witness():
    a,b,c=-6.125,-6.125,2**-20
    exact=oracle(half_bits(a),half_bits(b),half_bits(c))
    intermediate=struct.unpack('<f',struct.pack('<f',a*b+c))[0]
    fp32_then_half=half_value(half_bits(intermediate))
    fused=half_value(exact)
    assert fused==37.53125 and fp32_then_half==37.5
    assert oracle(half_bits(6.125),half_bits(-6.125),half_bits(-2**-20))==half_bits(-37.53125)
    assert oracle(0x8000,0x3c00,0x8000)==0x8000
    assert oracle(0,0x3c00,0x8000)==0
    return dict(passed=True,operands=[a,b,c],true_half_fma=fused,
                fp32_then_half=fp32_then_half,scope='Scalar counterexample; no NR model executed')

if __name__=='__main__':
    print(json.dumps(witness(),indent=2))
