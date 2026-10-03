"""Merge a frozen lifecycle payload into the E source tree, never the G runtime.

Existing canonical sources must still match the original V1 backup modulo line
endings. Refuse unrelated edits, redirects, or reusing a backup directory.
"""
from pathlib import Path
import argparse
import hashlib
import json

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
SHARED = WORKSPACE / '.codex-worktrees/re8-fp8-unround-fast-20260928/game'
PERF = Path('D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/live-web-perf-20261001')
ORIGINALS = PERF / 'replay-lifecycle-installed-v1-20261001'
FILES = {
    'game/branch_accum_native_720_v1.py', 'game/decoder_input_full_k_720_v1.py',
    'game/front_noise_native_720_v1.py', 'game/front_route_dispatch_720_v1.py',
    'game/history_numeric_suite_720_v1.py', 'game/nr_game_fullsize.py',
    'game/numeric_cleanup_suite_720_v1.py', 'game/numeric_frame_validation_720_v1.py',
    'game/post_numeric_suite_720_v1.py', 'game/vit_numeric_suite_720_v1.py',
    'game/replay_lifecycle_audit_base_720_v1.py',
    'game/replay_lifecycle_audit_rest_720_v1.py',
    'game/replay_lifecycle_graph_constants_720_v1.py',
    'plugins/cyberpunk_nr_adapter.py', 'plugins/nr_numeric_cost_meter_v1.py', 'game/nr_game_controls.py',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def physical(path):
    for part in (path, *path.parents):
        if part.is_symlink() or part.is_junction():
            raise RuntimeError('redirected path: ' + str(part))


def normalized(data):
    return data.decode('utf-8-sig').replace('\r\n', '\n')


def inspect(manifest_path, previous_manifest=None):
    physical(manifest_path)
    manifest_path = manifest_path.resolve(strict=True)
    if not manifest_path.is_relative_to(PROJECT / 'artifacts'):
        raise RuntimeError('manifest outside this project')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = sha(json.dumps({r['label']: r['after_sha256'] for r in manifest['files']},
                             sort_keys=True, separators=(',', ':')).encode())
    if expected != manifest['bundle_sha256']:
        raise RuntimeError('bundle checksum mismatch')
    previous = {}
    if previous_manifest is not None:
        physical(previous_manifest)
        previous_manifest = previous_manifest.resolve(strict=True)
        if not previous_manifest.is_relative_to(PROJECT / 'artifacts'):
            raise RuntimeError('previous manifest outside this project')
        admitted = json.loads(previous_manifest.read_text(encoding='utf-8'))
        previous = {r['label']: r['after_sha256'] for r in admitted['files']}
        if sha(json.dumps(previous, sort_keys=True, separators=(',', ':')).encode()) != admitted['bundle_sha256']:
            raise RuntimeError('previous bundle checksum mismatch')
    changes, labels = [], set()
    for row in manifest['files']:
        label = row['label']
        if label not in FILES or label in labels:
            raise RuntimeError('unexpected/duplicate canonical label: ' + label)
        labels.add(label)
        source = Path(row['source'])
        physical(source)
        if not source.is_relative_to(manifest_path.parent / 'payload'):
            raise RuntimeError('source outside manifest payload')
        data = source.read_bytes()
        if sha(data) != row['after_sha256']:
            raise RuntimeError('payload changed: ' + str(source))
        target = (PROJECT / 'game' / Path(label).name if label.startswith('plugins/')
                  else SHARED / Path(label).name)
        physical(target)
        before = target.read_bytes() if target.exists() else None
        if before == data:
            continue
        original = ORIGINALS / label
        if before is not None:
            original_matches = original.is_file() and normalized(before) == normalized(original.read_bytes())
            if (not original_matches and sha(before) != previous.get(label) and
                    sha(before) != row['before_sha256']):
                raise RuntimeError('canonical has unrelated edits: ' + str(target))
        elif not Path(label).name.startswith('replay_lifecycle_'):
            raise RuntimeError('unexpected missing canonical file: ' + str(target))
        compile(data, str(target), 'exec', dont_inherit=True)
        changes.append((label, target, before, data))
    return manifest, changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('check', 'promote'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--previous-manifest', type=Path)
    parser.add_argument('--backup', type=Path)
    args = parser.parse_args()
    manifest, changes = inspect(args.manifest, args.previous_manifest)
    if args.command == 'check':
        print(json.dumps({'status': 'ready', 'writes': False, 'changes': len(changes),
                          'bundle_sha256': manifest['bundle_sha256']}))
        return
    if args.backup is None:
        raise RuntimeError('missing backup path')
    physical(args.backup)
    backup = args.backup.resolve()
    if not backup.is_relative_to(PERF) or backup.exists():
        raise RuntimeError('backup must be a fresh task-owned PERF directory')
    backup.mkdir(parents=True)
    rows = []
    for label, target, before, data in changes:
        if before is not None:
            copy = backup / label
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(before)
        rows.append(dict(label=label, target=str(target),
                         before_sha256=sha(before) if before is not None else None,
                         after_sha256=sha(data)))
    receipt = dict(status='backed_up', manifest=str(args.manifest),
                   bundle_sha256=manifest['bundle_sha256'], changes=rows)
    (backup / 'before.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    written = []
    try:
        for label, target, before, data in changes:
            current = target.read_bytes() if target.exists() else None
            if current != before:
                raise RuntimeError('canonical changed after backup: ' + str(target))
            written.append((target, before))
            target.write_bytes(data)
            if target.read_bytes() != data:
                raise RuntimeError('canonical write mismatch')
    except BaseException:
        for target, before in reversed(written):
            if before is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(before)
        receipt['status'] = 'rolled_back'
        (backup / 'rollback.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
        raise
    receipt['status'] = 'source_merged'
    (backup / 'promoted.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': receipt['status'], 'changes': len(changes),
                      'receipt': str(backup / 'promoted.json')}))


if __name__ == '__main__':
    main()
