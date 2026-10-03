"""Add missing installed cache groups to the isolated offline preparation seed.

CPU/file operations only. Existing D files and all G files remain unchanged.
New group child paths point into the D copy, not back into the live runtime.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SOURCE = Path('G:/SteamLibrary/steamapps/common/Resident Evil Village BIOHAZARD VILLAGE/nr-runtime/data/fast-cache')
ROOT = Path('D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/cache-prepare-slow-v1-20261001')
TARGET = ROOT / 'cache'
RECEIPT = ROOT / 'installed-baseline-seed-receipt-v2.json'
BASELINE_MARKERS = {'__grp___entry_kernel.json', '__grp___entry_mlp_kernel.json'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def physical(path):
    for p in (path, *path.parents):
        if p.is_symlink() or p.is_junction():
            raise RuntimeError('refuse redirected cache path: ' + str(p))


def complete_group(group, marker_name):
    """A failed lookup creates a directory; only a usable group is complete."""
    marker = group / marker_name
    if not marker.is_file():
        return False
    physical(marker)
    children = json.loads(marker.read_bytes()).get('child_paths')
    if not isinstance(children, dict) or not children:
        return False
    for name, path in children.items():
        child = Path(path)
        if Path(name).name != name or child.name != name:
            raise RuntimeError('malformed existing cache child: ' + str(marker))
        physical(child)
        if not child.is_file():
            return False
    return True


def main():
    physical(SOURCE)
    physical(TARGET)
    if RECEIPT.exists():
        raise FileExistsError('preserving completed baseline seed receipt')
    plans, omitted, existing = [], [], 0
    for group in sorted(SOURCE.iterdir()):
        physical(group)
        if not group.is_dir():
            continue
        target_group = TARGET / group.name
        markers = list(group.glob('__grp__*.json'))
        if len(markers) != 1:
            omitted.append({'group': group.name, 'reason': 'not_one_complete_group'})
            continue
        marker = markers[0]
        if marker.name not in BASELINE_MARKERS:
            continue
        physical(target_group)
        if target_group.exists() and not target_group.is_dir():
            raise RuntimeError('cache group is not a directory: ' + str(target_group))
        if complete_group(target_group, marker.name):
            existing += 1
            continue
        raw = marker.read_bytes()
        value = json.loads(raw)
        children = value.get('child_paths')
        if not isinstance(children, dict) or not children:
            raise RuntimeError('malformed installed group marker: ' + str(marker))
        files, updated, unusable = [], {}, None
        for name, original in children.items():
            prior = Path(original)
            source = group / name
            if Path(name).name != name or prior.name != name or prior.parent.name != group.name:
                raise RuntimeError('installed group child has a different cache identity: ' + str(prior))
            physical(source)
            data = source.read_bytes()
            before = digest(data)
            if prior != source:
                # Some installed markers retain an earlier task-cache path.
                # Accept the local G mirror only with exact prior-byte proof.
                physical(prior)
                if prior.drive.casefold() != 'd:' or not prior.is_file() or digest(prior.read_bytes()) != before:
                    unusable = 'cross_root_child_without_matching_local_mirror'
                    break
            if source.suffix == '.json':
                metadata = json.loads(data)
                if 'cache_dir' in metadata:
                    metadata['cache_dir'] = str(target_group)
                data = (json.dumps(metadata, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
            files.append((source, target_group / name, before, data))
            updated[name] = str(target_group / name)
        if unusable:
            omitted.append({'group': group.name, 'reason': unusable})
            continue
        value['child_paths'] = updated
        payload = (json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
        files.append((marker, target_group / marker.name, digest(raw), payload))
        missing = []
        for source, target, before, data in files:
            physical(target)
            if target.exists():
                if not target.is_file() or digest(target.read_bytes()) != digest(data):
                    raise RuntimeError('refusing to replace a different existing artifact: ' + str(target))
            else:
                missing.append((source, target, before, data))
        plans.append((target_group, marker.name, missing))
    total = sum(len(data) for _, _, files in plans for _, _, _, data in files)
    if total > 512 * 1024 * 1024:
        raise RuntimeError('baseline seed exceeds the bounded 512 MiB copy budget')
    rows = []
    for target_group, marker_name, files in plans:
        physical(target_group)
        target_group.mkdir(exist_ok=True)
        # The final file is the group marker; publish it after all child files.
        for source, target, before, data in files:
            if digest(source.read_bytes()) != before:
                raise RuntimeError('installed cache changed during copy: ' + str(source))
            with target.open('xb') as stream:
                stream.write(data)
            if digest(target.read_bytes()) != digest(data):
                raise RuntimeError('copied artifact SHA mismatch: ' + str(target))
            rows.append({'source': str(source), 'target': str(target),
                         'source_sha256': before, 'target_sha256': digest(data)})
        if not complete_group(target_group, marker_name):
            raise RuntimeError('copied group is not readable: ' + str(target_group))
    report = {'status': 'completed', 'added_groups': len(plans), 'added_files': len(rows),
              'added_bytes': total, 'existing_groups_preserved': existing, 'omitted': omitted,
              'G_written': False, 'GPU_executed': False, 'files': rows}
    RECEIPT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({**{key: val for key, val in report.items() if key not in ('files', 'omitted')},
                      'omitted_count': len(omitted), 'receipt': str(RECEIPT)}))


if __name__ == '__main__':
    main()
