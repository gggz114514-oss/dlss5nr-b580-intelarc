"""Isolated CPU JSON comparison child; never uses the XPU Python/runtime."""
from pathlib import Path
import hashlib
import json
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).absolute().parent
sys.path.insert(0,str(HERE))
import metrics_common as c

def verify_runtime():
    row=c.read(HERE/'CPU_RUNTIME.json')
    c.require(Path(sys.executable).absolute()==Path(row['python']['path']).absolute() and
              c.sha(sys.executable)==row['python']['sha256'],'Wrong CPU interpreter')
    c.require(sys.flags.isolated==1 and sys.flags.utf8_mode==1 and sys.dont_write_bytecode,'CPU child requires -I -X utf8 -B')
    for name,pin in row['worker_files'].items():c.require(c.sha(HERE/name)==pin,'CPU worker bytes changed: '+name)
    import metrics_vectorized as kernel
    np=kernel.cpu_numpy()
    c.require(Path(np.__file__).absolute()==Path(row['numpy']['path']).absolute() and
              c.sha(np.__file__)==row['numpy']['sha256'] and np.__version__==row['numpy']['version'],'Wrong CPU NumPy')
    return kernel,row

def main():
    try:
        kernel,runtime=verify_runtime()
        raw=sys.stdin.buffer.read(2*1024**2+1);c.require(len(raw)<=2*1024**2,'Oversized CPU comparison request')
        request=json.loads(raw,parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
        c.require(request.get('schema')=='metrics-vectorized-request-v1','Wrong CPU request schema')
        fn={'pair':kernel.compare_pair,'frames':kernel.compare_frames}.get(request['operation'])
        c.require(fn is not None,'Unknown CPU operation')
        result=fn(request['reference'],request['candidate'],shape=tuple(request['shape']))
        response={'schema':'metrics-vectorized-response-v1','ok':True,'result':result,
                  'CPU_python':runtime['python'],'NumPy':runtime['numpy'],
                  'Torch_imported':'torch' in sys.modules,'GPU_runtime_imported':False}
        print(json.dumps(response,allow_nan=False,separators=(',',':')));return 0
    except BaseException as exc:
        print(json.dumps({'schema':'metrics-vectorized-response-v1','ok':False,
                          'error':str(exc),'exception':type(exc).__name__},allow_nan=False));return 1

if __name__=='__main__':raise SystemExit(main())
