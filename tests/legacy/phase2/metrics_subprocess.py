"""Always execute the pinned comparison in a separate existing CPU bundle.

Even a parent launched with the same bundled interpreter uses a subprocess.
This module imports stdlib only. The GPU child never imports this module.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import runner_common as c

CPU_PYTHON = Path('C:/Users/REFERENCE_USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
CPU_PYTHON_SHA256 = '372c2eae555b344520bf147be0096e009069aeca4e7f78d6aecea6d53158056a'
RUNTIME_SHA256 = '52bf530a7ad6ecc78cc473a69f6bc6912af17a2aa33fe943ebaa5fa3cc4e9358'
CPU_DIR = c.SCRIPT_DIR/'cpu_comparison'
WORKER = CPU_DIR/'metrics_cpu_worker.py'
REQUEST_LIMIT = RESPONSE_LIMIT = 2*1024**2
STDERR_LIMIT = 65536
TIMEOUT_SECONDS = 180
LAST_CALL = None

def runtime_contract():
    row=c.read(c.checked(dict(path=str(CPU_DIR/'CPU_RUNTIME.json'),sha256=RUNTIME_SHA256)))
    c.require(Path(row['python']['path'])==CPU_PYTHON and row['python']['sha256']==CPU_PYTHON_SHA256,
              'Only the exact bundled CPU interpreter is admitted')
    c.require(c.sha(c.no_reparse(CPU_PYTHON))==CPU_PYTHON_SHA256,'Bundled CPU interpreter bytes changed')
    expected={'metrics_common.py','metrics_vectorized.py','metrics_cpu_api.py','metrics_cpu_worker.py'}
    c.require(set(row['worker_files'])==expected,'Incomplete frozen comparison module set')
    for name,pin in row['worker_files'].items():
        c.checked(dict(path=str(CPU_DIR/name),sha256=pin))
    return row

def worker_command():
    runtime_contract()
    return [str(CPU_PYTHON),'-I','-X','utf8','-B',str(WORKER)]

def worker_environment():
    env=dict(os.environ)
    for name in ('PYTHONPATH','PYTHONHOME','NR_PHASE1_ARM','NR_PHASE1_SEED_RECEIPT_PATH'):
        env.pop(name,None)
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name]='1'
    env.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
    return env

def unique_json(raw):
    def unique(pairs):
        row={}
        for key,value in pairs:c.require(key not in row,'Duplicate CPU response key: '+key);row[key]=value
        return row
    return json.loads(raw,object_pairs_hook=unique,
        parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))

def _compare(operation,reference,candidate,shape):
    global LAST_CALL
    c.require(all(name not in sys.modules for name in ('torch','triton','numpy')),
              'Runner comparison parent must remain stdlib-only; never use the tensor/XPU process')
    runtime=runtime_contract();command=worker_command();worker_receipt=c.record(WORKER)
    request=dict(schema='metrics-vectorized-request-v1',operation=operation,
                 reference=reference,candidate=candidate,shape=list(shape))
    payload=json.dumps(request,allow_nan=False,separators=(',',':')).encode('utf-8')
    c.require(len(payload)<=REQUEST_LIMIT,'Oversized CPU comparison request')
    started=time.perf_counter()
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        env=worker_environment(),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|
        getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
    LAST_CALL=dict(command=command,interpreter=runtime['python'],worker=worker_receipt,
        pid=process.pid,operation=operation,request_bytes=len(payload),parent_python=sys.executable,
        parent_numpy_imported='numpy' in sys.modules,parent_torch_imported='torch' in sys.modules,
        GPU_executed=False)
    stdout=stderr=b''
    try:
        try:
            stdout,stderr=process.communicate(payload,timeout=TIMEOUT_SECONDS)
        except BaseException:
            # This exact Popen owns a pinned CPU-only child; never kill foreign
            # processes. Wait for its exit before propagating any failure.
            if process.poll() is None:process.kill()
            stdout,stderr=process.communicate(timeout=10)
            raise
    finally:
        LAST_CALL.update(returncode=process.poll(),wall_s=time.perf_counter()-started,
            child_exited=process.poll() is not None,stdout_bytes=len(stdout),stderr_bytes=len(stderr),
            stdout_sha256=hashlib.sha256(stdout).hexdigest(),stderr_sha256=hashlib.sha256(stderr).hexdigest())
    c.require(len(stdout)<=RESPONSE_LIMIT and len(stderr)<=STDERR_LIMIT,'Oversized CPU comparison response')
    try:response=unique_json(stdout)
    except Exception as exc:
        raise RuntimeError('CPU comparison returned invalid JSON; pinned command='+repr(command)+
            '; stderr='+stderr.decode('utf-8',errors='replace')) from exc
    c.require(response.get('schema')=='metrics-vectorized-response-v1','Wrong CPU comparison response schema')
    if process.returncode!=0 or response.get('ok') is not True:
        LAST_CALL.update(child_error=response.get('error'),child_exception=response.get('exception'),
                         original_CPU_response=response,stderr=stderr.decode('utf-8',errors='replace'))
        raise RuntimeError('CPU comparison failed: '+str(response.get('error'))+
            '; exception='+str(response.get('exception'))+'; exit='+str(process.returncode))
    c.require(response.get('CPU_python')==runtime['python'] and response.get('NumPy')==runtime['numpy'],
              'CPU comparison response uses a different pinned interpreter/package')
    c.require(response.get('Torch_imported') is False and response.get('GPU_runtime_imported') is False,
              'Unexpected tensor/device runtime in CPU comparison')
    LAST_CALL.update(child_CPU_python=response['CPU_python'],child_numpy=response['NumPy'],
        child_torch_imported=response['Torch_imported'],child_GPU_runtime_imported=response['GPU_runtime_imported'])
    return response['result']

def compare_pair(reference,candidate,*,shape=c.FRAME_SHAPE):return _compare('pair',reference,candidate,shape)
def compare_frames(reference,candidate,*,shape=c.FRAME_SHAPE):return _compare('frames',reference,candidate,shape)
