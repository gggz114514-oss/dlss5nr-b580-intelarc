"""Research entry for one reset frame using privately prepared pinned assets.

Input NPY is floating HWC RGB, already in the same colour/quantization contract
as the reference. This script neither decodes nor rescales images.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','xpu'),default='xpu')
    p.add_argument('--arithmetic',choices=('reference','triton'),default='reference')
    p.add_argument('--seed',type=lambda x:int(x,0),default=0)
    args=p.parse_args()
    if args.output.exists():p.error('Output directory must be new')
    if not 0<=args.seed<2**32:p.error('Seed must fit uint32')
    import numpy as np
    import torch
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
    from nr_backend import ResetNR,use_arithmetic_backend
    pixels=np.load(args.input,allow_pickle=False)
    if pixels.ndim!=3 or pixels.shape[-1]!=3 or pixels.dtype.kind!='f' or not np.isfinite(pixels).all():
        p.error('Expected finite floating HWC3 NPY; no implicit resizing or colour conversion')
    model=ResetNR.from_assets(args.assets/'sf-v2/WEIGHTS_HT.bin',args.assets/'noise-sm89-v2').to(args.device).eval()
    rgb=torch.from_numpy(pixels.copy()).to(args.device)
    if args.device=='xpu':torch.xpu.synchronize()
    started=time.perf_counter()
    with torch.inference_mode(),use_arithmetic_backend(args.arithmetic):
        output=model(rgb,seed=args.seed)
    if args.device=='xpu':torch.xpu.synchronize()
    elapsed=time.perf_counter()-started
    value=output.cpu().numpy()
    args.output.mkdir(parents=True,exist_ok=False)
    np.save(args.output/'rgb.npy',value,allow_pickle=False)
    report=dict(scope='One reset frame; elapsed time includes cold initialization of kernels, not a benchmark',
        seed=args.seed,shape=list(value.shape),dtype=value.dtype.str,seconds=elapsed,
        input_file_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        raw_output_sha256=hashlib.sha256(value.tobytes()).hexdigest(),native_comparison_performed=False)
    (args.output/'run.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
