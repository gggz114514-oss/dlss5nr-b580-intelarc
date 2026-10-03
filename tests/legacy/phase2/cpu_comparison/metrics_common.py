"""Small CPU file contracts. E/D only; no runtime, tensor or GPU imports."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import stat

HERE=Path(__file__).absolute().parent
IMP=HERE.parent.parent
DATA=Path('D:/Codex-NR-Experiments/cyberpunk-opt/b580-full-implementation-v1-20261003/reviews/metrics-vectorized')
FRAME_SHAPE=(720,1280,3)

def require(ok,message):
    if not ok:raise RuntimeError(message)

def no_reparse(path):
    p=Path(path).absolute()
    require(p.drive.upper() in ('E:','D:'),'CPU metrics input/output stays E/D; never G/runtime')
    for part in (p,*p.parents):
        require(not part.is_symlink() and not (hasattr(part,'is_junction') and part.is_junction()),'Path crosses a link/junction: '+str(part))
        if part.exists():require(not getattr(part.lstat(),'st_file_attributes',0)&getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',1024),'Reparse path')
    return p.resolve()

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def checked(row):
    require(isinstance(row,dict) and {'path','sha256'}<=set(row),'Hashed file reference required')
    p=no_reparse(row['path']).resolve(strict=True)
    require(sha(p)==row['sha256'],'File drift: '+str(p))
    return p

def record(path):
    p=no_reparse(path);return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}

def read(path):
    def unique(pairs):
        row={}
        for key,value in pairs:require(key not in row,'Duplicate JSON key: '+key);row[key]=value
        return row
    return json.loads(no_reparse(path).read_text(encoding='utf-8-sig'),object_pairs_hook=unique,
                      parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))

def write(path,row):
    p=no_reparse(path);require(p.is_relative_to(DATA) and p!=DATA,'Writes stay in owned D metrics root')
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(row,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    return record(p)
