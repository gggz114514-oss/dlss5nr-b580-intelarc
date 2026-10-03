"""Copy exact runner3 and the SHA sealed CPU comparator into this owned review."""
from pathlib import Path
import hashlib
import json
import stat
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).absolute().parent
IMP=Path('E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003')
BASE=IMP/'luna/phase2'
FROZEN=IMP/'reviews/metrics-vectorized/stage1'

def check_path(path):
    p=Path(path).absolute()
    for parent in (p,*p.parents):
        assert not parent.is_symlink() and not parent.is_junction(),str(parent)
        if parent.exists():assert not getattr(parent.lstat(),'st_file_attributes',0)&stat.FILE_ATTRIBUTE_REPARSE_POINT,str(parent)
    return p.resolve()

def sha(path):
    with check_path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def record(path):return {'path':str(check_path(path)),'sha256':sha(path),'bytes':Path(path).stat().st_size}

def copy(origin,target,pin):
    origin,target=check_path(origin),check_path(target)
    assert target.is_relative_to(HERE) and sha(origin)==pin
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:stream.write(origin.read_bytes())
    assert sha(target)==pin

def main():
    assert check_path(HERE)==check_path(IMP/'reviews/metrics-vectorized/runner4-candidate')
    ready=json.loads((BASE/'READY.json').read_text());ready_pin=record(BASE/'READY.json')
    assert ready['runner_revision']==3 and ready['target_manifest']['sha256']=='ea32381a4bf5ffc31d47fa6f05055411d48fc5c2da7032c5dc288287de456cca'
    for name,pin in ready['files'].items():
        assert Path(name).as_posix()==name and '..' not in Path(name).parts and not Path(name).is_absolute()
        copy(BASE/name,HERE/name,pin)
    copy(BASE/'READY.json',HERE/'BASE_RUNNER3_READY.json',ready_pin['sha256'])
    comparison_ready=IMP/'reviews/metrics-vectorized/READY.json'
    assert sha(comparison_ready)=='a1e8343d0d610f4c00bea5b758da9f14ccacf480876431b0ca786b154a7892f8'
    comparison=json.loads(comparison_ready.read_text());snapshot=json.loads((FROZEN/'SNAPSHOT.json').read_text())
    assert sha(FROZEN/'SNAPSHOT.json')==comparison['snapshot_manifest']['sha256']
    assets=['metrics_common.py','metrics_vectorized.py','metrics_cpu_api.py','metrics_cpu_worker.py','CPU_RUNTIME.json']
    for name in assets:
        row=next(r for r in snapshot['files'] if r['relative_path']==name)
        copy(FROZEN/name,HERE/'cpu_comparison'/name,row['sha256'])
    copy(BASE/'metrics.py',HERE/'metrics_streaming_reference.py',ready['files']['metrics.py'])
    assert sha(BASE/'READY.json')==ready_pin['sha256']
    receipt={'schema':'runner4-private-copy-provenance-v1','base_runner3_READY':ready_pin,
        'runner3_files':{name:record(BASE/name) for name in ready['files']},
        'comparison_READY':record(comparison_ready),'comparison_snapshot':record(FROZEN/'SNAPSHOT.json'),
        'comparison_asset_copies':{name:record(HERE/'cpu_comparison'/name) for name in assets},
        'target_manifest':ready['target_manifest'],'GPU_executed':False,'writes':'Only runner4-candidate source'}
    with (HERE/'BASE_PROVENANCE.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps({'copied_runner3_files':len(ready['files']),'base_READY':ready_pin,
        'CPU_RUNTIME':record(HERE/'cpu_comparison/CPU_RUNTIME.json'),'GPU_executed':False}))

if __name__=='__main__':main()
