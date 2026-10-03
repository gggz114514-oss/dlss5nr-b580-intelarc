"""Copy the frozen bridge to an external build tree with one declared build fix.

The installed source snapshot remains byte-identical. The fix supplies a
scope variable when the optional identity/tail diagnostics are both disabled;
the historically accepted live configuration preprocesses the original branch.
No DLL is built, loaded, installed, or executed by this stdlib-only command.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'current/bridge/'
BEFORE = b'''#ifdef NRB_XESS_IDENTITY_TEST
    bool identity_source_state_proven=false;
    uint64_t identity_source_generation=0;
#endif
#endif
    nrb::flashdiag::Context flash{};'''
AFTER = BEFORE.replace(b'#endif\n#endif\n    nrb::flashdiag',
    b'#endif\n#else\n    const bool diagnostic_dlss_scope=route==NRB_ROUTE_DLSS;\n#endif\n    nrb::flashdiag')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def plain(path):
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        if part.exists():
            st = part.lstat()
            if part.is_symlink() or getattr(st, 'st_file_attributes', 0) & 0x400:
                raise ValueError('Reparse path refused: '+str(part))
    return path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True,
                   help='fresh source directory outside this repository')
    args = p.parse_args()
    root, output = plain(ROOT), plain(args.output)
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Choose a fresh directory outside the repository')
    manifest_path = root/'evidence/2026-10-03/current-source-manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    rows = [r for r in json.loads(manifest_bytes)['files'] if r['path'].startswith(PREFIX)]
    if len(rows) != 162:
        raise ValueError('Unexpected frozen bridge inventory')
    source = root/'current/bridge'
    expected = {r['path'][len(PREFIX):] for r in rows}
    actual = {f.relative_to(source).as_posix() for f in source.rglob('*') if f.is_file()}
    if actual != expected:
        raise ValueError('Frozen bridge inventory differs')
    payloads, changes = [], []
    for row in rows:
        relative = row['path'][len(PREFIX):]
        rel = Path(relative)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Invalid manifest path')
        original = plain(source/rel).read_bytes()
        if len(original) != row['bytes'] or sha(original) != row['sha256']:
            raise ValueError('Frozen source identity mismatch: '+relative)
        prepared = original
        if relative == 'src/asi.cpp':
            newline = b'\r\n' if b'\r\n' in original else b'\n'
            before, after = BEFORE.replace(b'\n', newline), AFTER.replace(b'\n', newline)
            if original.count(before) != 1:
                raise ValueError('Build-fix anchor differs; review before changing source')
            prepared = original.replace(before, after, 1)
            changes.append(dict(path=relative, original_sha256=sha(original),
                                prepared_sha256=sha(prepared), reason='default-build conditional scope declaration'))
        payloads.append((rel, prepared))
    output.mkdir(parents=True)
    for relative, data in payloads:
        target = output/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
    receipt = dict(schema='nr-bridge-build-preparation-v1', GPU_executed=False,
                   original_source_modified=False, source_files=len(rows),
                   current_manifest_sha256=sha(manifest_bytes), changes=changes)
    (output/'BUILD_PREPARATION.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))

if __name__ == '__main__':
    main()
