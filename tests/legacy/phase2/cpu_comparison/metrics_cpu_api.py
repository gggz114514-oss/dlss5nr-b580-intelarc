"""Stdlib-only supervisor API: direct pinned CPU NumPy or a CPU subprocess.

Same compare_pair/compare_frames call/result schema as the current metrics.
Main may route its frozen comparison entry here after independent review.
No existing runner, source, GPU environment or installation is changed.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import metrics_common as c

def _runtime():
    c.require('torch' not in sys.modules and Path(sys.executable).drive.upper()!='G:',
              'CPU metrics supervisor cannot be an XPU/Torch process')
    row=c.read(c.HERE/'CPU_RUNTIME.json')
    py=Path(row['python']['path']).absolute()
    c.require(py.drive.upper()=='C:' and py.name.lower()=='python.exe' and c.sha(py)==row['python']['sha256'],
              'Pinned existing CPU interpreter required; no isolated XPU runtime')
    for name,pin in row['worker_files'].items():c.require(c.sha(c.HERE/name)==pin,'CPU comparison source changed: '+name)
    return row

def _call(operation,reference,candidate,shape):
    runtime=_runtime()
    if Path(sys.executable).absolute()==Path(runtime['python']['path']).absolute():
        import metrics_vectorized as kernel
        return getattr(kernel,'compare_'+operation)(reference,candidate,shape=shape)
    request={'schema':'metrics-vectorized-request-v1','operation':operation,
             'reference':reference,'candidate':candidate,'shape':list(shape)}
    payload=json.dumps(request,allow_nan=False,separators=(',',':')).encode('utf-8')
    c.require(len(payload)<=2*1024**2,'Oversized CPU comparison request')
    env=dict(os.environ)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    command=[runtime['python']['path'],'-I','-X','utf8','-B',str(c.HERE/'metrics_cpu_worker.py')]
    process=subprocess.run(command,input=payload,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,
        timeout=180,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
    c.require(len(process.stdout)<=2*1024**2 and len(process.stderr)<=65536,'Oversized CPU worker response')
    response=json.loads(process.stdout)
    c.require(response.get('schema')=='metrics-vectorized-response-v1','Wrong CPU worker response')
    c.require(process.returncode==0 and response.get('ok') is True,'CPU comparison failed: '+str(response.get('error',process.stderr.decode('utf-8',errors='replace'))))
    c.require(response.get('Torch_imported') is False and response.get('GPU_runtime_imported') is False,'Unexpected tensor/device runtime in CPU worker')
    return response['result']

def compare_pair(reference,candidate,*,shape=c.FRAME_SHAPE):return _call('pair',reference,candidate,shape)
def compare_frames(reference,candidate,*,shape=c.FRAME_SHAPE):return _call('frames',reference,candidate,shape)
