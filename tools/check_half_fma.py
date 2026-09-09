"""Compare the released backend to a finite-domain integer oracle; no model assets."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from fma_witness import oracle

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','xpu'),default='cpu')
    args=parser.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
    from nr_backend.tensor_math import half_fma
    rng=np.random.default_rng(156)
    cases=rng.integers(0,0x7c00,size=(24000,3),dtype=np.uint16)
    cases[:12000]|=rng.integers(0,2,size=(12000,3),dtype=np.uint16)<<15
    cases[12000:,1]=cases[12000:,0]
    witnesses=np.array([[6.125,6.125,2**-20],[-6.125,6.125,-2**-20],
        [256,256,-16],[256,256,-16.015625],[0,0,-0.0],[-0.0,1,-0.0],
        [2**-12,2**-13,2**-24]],dtype='<f2').view('u2')
    cases=np.vstack((cases,witnesses))
    expected=np.array([oracle(*map(int,row)) for row in cases],dtype='u2')
    values=torch.from_numpy(cases.copy()).view(torch.float16).to(args.device)
    with torch.inference_mode():
        actual=half_fma(values[:,0],values[:,1],values[:,2]).cpu().numpy().view('u2')
    mismatches=int(np.count_nonzero(actual!=expected))
    print(json.dumps(dict(device=args.device,cases=len(cases),mismatches=mismatches,
        passed=mismatches==0,scope='Finite scalar operands; not an exhaustive domain or a complete NR run'),indent=2))
    if mismatches:raise SystemExit(1)

if __name__=='__main__':main()
