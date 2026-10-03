"""CPU-only probe: a bundled parent must still delegate to a separate child."""
from pathlib import Path
import json
import os
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).absolute().parent))
import metrics
import metrics_subprocess as rpc

def main():
    raw=sys.stdin.buffer.read(rpc.REQUEST_LIMIT+1)
    if len(raw)>rpc.REQUEST_LIMIT:raise RuntimeError('Oversized CPU probe')
    row=rpc.unique_json(raw)
    result=metrics.compare_pair(row['reference'],row['candidate'],shape=tuple(row['shape']))
    if any(name in sys.modules for name in ('numpy','torch','triton')):
        raise RuntimeError('CPU probe parent imported a tensor library')
    print(json.dumps(dict(result=result,call=rpc.LAST_CALL,parent_pid=os.getpid(),parent_python=sys.executable,
        parent_numpy_imported='numpy' in sys.modules,parent_torch_imported='torch' in sys.modules),allow_nan=False))

if __name__=='__main__':main()
