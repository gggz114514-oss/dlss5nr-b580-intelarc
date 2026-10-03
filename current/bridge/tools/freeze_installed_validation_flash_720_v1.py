"""Pin the reviewed installed CPU fixes for offline cache preparation only.

Preserves the old manifests and never writes runtime source or cache files.
Uses the completed installation receipts, rather than accepting arbitrary drift.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent / '.codex-worktrees/re8-fp8-unround-fast-20260928'
TASK = Path('D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930')
PERF = TASK / 'live-web-perf-20261001'
RUNTIME = Path('G:/SteamLibrary/steamapps/common/Resident Evil Village BIOHAZARD VILLAGE/nr-runtime')
PLUGINS = Path('G:/epic/Cyberpunk2077/bin/x64/plugins')
OLD_FREEZE = TASK / 'source-freeze-native-rtz-host-v7-20261001.json'
OLD_INSTALL = TASK / 'backups/20260930T162535.484538Z/manifest.json'
NEW_FREEZE = TASK / 'source-freeze-validation-flash-v8-20261001.json'
NEW_INSTALL = TASK / 'source-installed-validation-flash-v8-20261001.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(source, target, expected):
    for path in (source, target):
        for part in (path, *path.parents):
            if part.is_symlink() or part.is_junction():
                raise RuntimeError('redirected source/target: ' + str(part))
        if sha(path) != expected:
            raise RuntimeError('reviewed installed SHA mismatch: ' + str(path))
    return {'source': str(source), 'target': str(target), 'source_hash': expected}


def main():
    batch_path = PERF / 'validation-batch-install-v1-20261001/installed-receipt.json'
    dispatch_path = PERF / 'validation-batch-dispatch-fix-v1-20261001/installed-receipt.json'
    flash_path = PERF / 'periodic-flash-v2-installed-20261001/installed.json'
    batch, dispatch, flash = (read(p) for p in (batch_path, dispatch_path, flash_path))
    if any(r.get('status') != 'installed' for r in (batch, dispatch, flash)):
        raise RuntimeError('only completed installation receipts may update the freeze')
    reviewed = {}
    for row in batch['files']:
        reviewed[row['source']] = (row['target'], row['after_sha256'])
    reviewed[dispatch['source']] = (dispatch['target'], dispatch['after_sha256'])
    for row in flash['files']:
        reviewed[row['source']] = (row['target'], row['installed_sha256'])
    old = read(OLD_FREEZE)
    new = dict(old)
    frozen = {}
    old_host = ROOT / 'game/nr_game_pre_xess_host.py'
    for raw, expected in old['source_sha256'].items():
        source = Path(raw)
        if source == old_host:
            # The installed v2 host is pinned to its exact E frozen stage below.
            continue
        if raw in reviewed:
            target, expected = reviewed[raw]
            pin(source, Path(target), expected)
        elif sha(source) != expected:
            raise RuntimeError('unreviewed source drift: ' + raw)
        frozen[raw] = expected
    installed = []
    for raw, expected in frozen.items():
        source = Path(raw)
        target = (PLUGINS if source.parent == PROJECT / 'game' else RUNTIME / 'game') / source.name
        installed.append(pin(source, target, expected))
    for raw, (target, expected) in reviewed.items():
        source = Path(raw)
        installed = [r for r in installed if Path(r['target']) != Path(target)]
        installed.append(pin(source, Path(target), expected))
        if source.drive.casefold() == 'e:':
            frozen[raw] = expected
    # The installed product contract did not change in this repair.
    for row in read(OLD_INSTALL):
        if Path(row['target']).suffix == '.json':
            installed.append(pin(Path(row['source']), Path(row['target']), row['source_hash']))
    new['source_sha256'] = frozen
    new['derivation'] = {str(p): sha(p) for p in (OLD_FREEZE, OLD_INSTALL, batch_path, dispatch_path, flash_path)}
    new['scope'] = 'reviewed installed validation and passive flash diagnostics; kernel mathematics unchanged'
    for path, value in ((NEW_FREEZE, new), (NEW_INSTALL, installed)):
        if path.exists() and read(path) != value:
            raise FileExistsError('preserving different prior manifest: ' + str(path))
        if not path.exists():
            path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    spec = importlib.util.spec_from_file_location('_freeze_cpu_smoke', ROOT / 'tools/smoke_numeric_game_all_720_v1.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hashes = module._source_freeze(NEW_FREEZE)
    manifests = module._load_runtime_manifests(
        NEW_FREEZE, NEW_INSTALL, TASK / 'rtz-cache-installed-exact-v7-20261001.json',
        TASK / 'cache-prepare-slow-v1-20261001/cache-manifest.json',
        TASK / 'cache-prepare-slow-v1-20261001/cache', hashes, prepare_cache=True)
    if 'torch' in sys.modules or 'triton' in sys.modules:
        raise RuntimeError('CPU manifest check imported a GPU runtime')
    print(json.dumps({'status': 'passed', 'freeze': str(NEW_FREEZE), 'install': str(NEW_INSTALL),
                      'source_count': len(hashes), 'installed_rows': len(installed),
                      'seed_files_checked': manifests['checked_staged_cache_files'], 'GPU_executed': False}))


if __name__ == '__main__':
    main()
