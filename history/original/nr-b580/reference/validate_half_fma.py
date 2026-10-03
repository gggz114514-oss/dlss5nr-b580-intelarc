"""Compare portable half FMA with an independent exact-integer rounding oracle."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parent;root=R.parent;sys.path.insert(0,str(root/'backend'))
from nr_backend.tensor_math import half_fma
def units(bits):
 sign=-1 if bits&0x8000 else 1;e=(bits>>10)&31;m=bits&1023
 return sign*(m if e==0 else (m+1024)<<(e-1))
def oracle(a,b,c):
 value=units(a)*units(b)+(units(c)<<24);v=abs(value)
 if not v:return 0x8000 if ((a^b)&c&0x8000) else 0
 shift=max(v.bit_length()-11,24);q,rem=divmod(v,1<<shift);mid=1<<(shift-1)
 q+=int(rem>mid or(rem==mid and q&1));bits=min(0x7c00,(shift-24)*1024+q)
 return bits|(0x8000 if value<0 else 0)
rng=np.random.default_rng(156)
cases=rng.integers(0,0x7c00,size=(24000,3),dtype=np.uint16)
cases[:12000]|=(rng.integers(0,2,size=(12000,3),dtype=np.uint16)<<15)
cases[12000:,1]=cases[12000:,0]
# Native failure: 6.125**2 + half(2**-20), just above an FP16 midpoint.
witness=np.array([[6.125,6.125,2**-20],[-6.125,6.125,-2**-20],[256,256,-16],[256,256,-16.015625],[0,0,-0.0],[-0.0,1,-0.0],[2**-12,2**-13,2**-24]],dtype='<f2').view('u2')
cases=np.vstack((cases,witness));expected=np.array([oracle(*map(int,row))for row in cases],dtype='u2')
report=dict(cases=len(cases),oracle='Python arbitrary-precision integer products/addition at 2**-48; one ties-to-even half rounding',devices={})
for device in('cpu','xpu'):
 x=torch.from_numpy(cases.copy()).view(torch.float16).to(device)
 with torch.inference_mode():actual=half_fma(x[:,0],x[:,1],x[:,2]).cpu().numpy().view('u2')
 bad=np.flatnonzero(actual!=expected);row=dict(mismatches=int(len(bad)),first_failures=[dict(inputs=cases[i].tolist(),expected=int(expected[i]),actual=int(actual[i]))for i in bad[:10]])
 report['devices'][device]=row;print(device,row,flush=True)
 # Cross the FMA workspace boundary with scalar broadcasting and a noncontiguous
 # first operand; compare with the same integer oracle, not the implementation.
 big=torch.from_numpy(cases.copy()).view(torch.float16).repeat(45,1).to(device)
 with torch.inference_mode():chunked=half_fma(big[:,0],big[:,1],big[:,2]).cpu().numpy().view('u2')
 row['chunked_cases']=len(chunked);row['chunked_mismatches']=int(np.count_nonzero(chunked!=np.tile(expected,45)))
 assert row['chunked_mismatches']==0,row
report['source_sha256']=hashlib.sha256((root/'backend/nr_backend/tensor_math.py').read_bytes()).hexdigest()
(R/'half-fma-validation.json').write_text(json.dumps(report,indent=2));assert all(v['mismatches']==0 for v in report['devices'].values()),report
