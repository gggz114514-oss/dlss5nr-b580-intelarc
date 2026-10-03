"""Verify public identity/syntax/privacy without executing model code."""
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '__pycache__', '_cpu_work', '_scratch'}
FORBIDDEN = {'.bin', '.dll', '.pyd', '.exe', '.so', '.lib', '.obj', '.spv', '.npy', '.npyz',
             '.npz', '.mp4', '.mkv', '.avi', '.f32', '.zip', '.7z', '.pyc', '.cubin',
             '.hsaco', '.pt', '.pth', '.safetensors', '.dylib', '.a', '.o'}
SHA = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def files():
    for p in ROOT.rglob('*'):
        relative = p.relative_to(ROOT)
        if p.is_file() and not set(relative.parts) & EXCLUDED and relative.as_posix() != 'evidence/file-manifest.json':
            assert not p.is_symlink(), str(relative)
            yield p

def check(row):
    p = ROOT / row['path']
    assert p.resolve().is_relative_to(ROOT) and p.is_file(), row['path']
    data = p.read_bytes()
    assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], row['path']
    assert p.suffix.lower() not in FORBIDDEN, row['path']
    if p.suffix.lower() == '.py':
        ast.parse(data.decode('utf-8-sig'), filename=row['path'])
    if p.suffix.lower() in {'.py', '.ps1', '.md', '.json', '.jsonl', '.txt'}:
        text = data.decode('utf-8-sig')
        assert not re.search(r'-----BEGIN (?:OPENSSH|RSA) PRIVATE KEY-----\r?\n[A-Za-z0-9+/=]{20}', text), row['path']
        assert not re.search(r'gh[pousr]_[A-Za-z0-9]{25,}', text), row['path']
        assert not re.search(r'192\.168\.2\.147|HBHGGGZ[-]LAPTOP|C:[/\\]Users[/\\]gggz', text, flags=re.I), row['path']
    return p.suffix.lower() == '.py'

def main():
    manifest = json.loads((ROOT / 'evidence/file-manifest.json').read_text(encoding='utf-8'))
    rows = manifest['files']
    assert len({r['path'] for r in rows}) == len(rows)
    assert {r['path'] for r in rows} == {p.relative_to(ROOT).as_posix() for p in files()}, 'Incomplete inventory'
    with ThreadPoolExecutor(max_workers=16) as pool:
        python_files = sum(pool.map(check, rows))
    # Historical originals retain references to unpublished data/old source trees.
    navigation = [ROOT / 'README.md', ROOT / 'current/README.md', ROOT / 'current/BUILD.md',
                  ROOT / 'tests/README.md', ROOT / 'tests/PREPARATION.md']
    links = 0
    for p in navigation:
        if not p.exists():
            continue
        for target in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)', p.read_text(encoding='utf-8')):
            target = target.strip('<>').split('#', 1)[0]
            if not target or '://' in target or target.startswith('mailto:'):
                continue
            assert (p.parent / unquote(target)).exists(), f'{p.relative_to(ROOT)}: {target}'
            links += 1
    historical = json.loads((ROOT / 'evidence/source-manifest.json').read_text(encoding='utf-8'))
    for row in historical['files']:
        assert SHA(ROOT / row['path']) == row['sha256'], row['path']
    print(json.dumps(dict(passed=True, files_verified=len(rows), python_syntax_checked=python_files,
                         current_navigation_links_checked=links, GPU_executed=False,
                         scope='Published source identity, syntax, privacy and navigation; no model execution'), indent=2))

if __name__ == '__main__':
    main()
