"""Identify remaining matrix views on one full reset frame; no timing claim."""
import hashlib,json,os,sys,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
EXACT=HERE.parent.parent/'nr-b580'
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT=DREF/'experimental/static-weight-cache-v1/misses.json'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-sm89-v1')
sys.path.insert(0,str(HERE.parent/'backend'))
import torch
import numpy as np
from PIL import Image
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from static_weight_cache_v1 import CachedFastMatrices
torch.set_num_threads(2)
model=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
parents={w.untyped_storage().data_ptr():name for name,w in model.named_buffers()}
exp=CachedFastMatrices()
misses={}
with torch.inference_mode(),exp.installed():
    exp.static_cache.prepack(model)
    old=exp.static_cache.lookup
    def observe(w):
        value=old(w)
        if value is None:
            parent=parents.get(w.untyped_storage().data_ptr(),'not_a_registered_buffer')
            signature=(parent,tuple(w.shape),tuple(w.stride()),w.storage_offset())
            if signature not in misses:
                misses[signature]=dict(parent=parent,shape=list(w.shape),strides=list(w.stride()),storage_offset=w.storage_offset(),calls=0,stack=[dict(file=f.filename,line=f.lineno,name=f.name) for f in traceback.extract_stack()[-9:-1]])
            misses[signature]['calls']+=1
        return value
    exp.static_cache.lookup=observe
    exp.select('int8_cached')
    pixels=np.asarray(Image.open(EXACT/'reference/inputs/flow-full-864x480-v2/frame00.png').convert('RGB'),dtype='f4')/255
    rgb=torch.from_numpy(pixels).to('xpu');motion=torch.zeros((480,864,2),device='xpu')
    with use_arithmetic_backend('triton'):value=model(rgb,motion,reset=True)
    actual=value.cpu().float().numpy()
    prior=json.loads((DREF/'results/fast-precision-864x480-v1/validation.json').read_text())['runs']['int8_dense'][0]
    assert hashlib.sha256(actual.tobytes()).hexdigest()==prior['rgb_sha256']
    report=dict(sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'static_weight_cache_v1.py']},calls=exp.calls,misses=list(misses.values()),byte_equal_prior=True,diagnostic_only=True)
    OUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(calls=exp.calls,misses=[{k:v for k,v in row.items() if k!='stack'} for row in misses.values()])),flush=True)
