"""Replace four reviewed bridge files; never replace loaded files or math."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical(path):
    for part in (path, *path.parents):
        if part.is_junction() or part.is_symlink():
            raise ValueError('Redirected task path: ' + str(part))


def dump(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('check', 'install'))
    parser.add_argument('--current-installation', type=Path, required=True)
    parser.add_argument('--CPU-build', type=Path, required=True)
    parser.add_argument('--runtime-review', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    installed, build = read(args.current_installation), read(args.CPU_build)
    if installed.get('status') != 'installed_trial_default_off' or len(installed['files']) != 9:
        raise ValueError('Expected the actual-overlay nine-file trial')
    if build.get('status') != 'native_CPU_compiled_GPU_UNTESTED':
        raise ValueError('Expected the isolated CPU build receipt')
    for path, digest in {**build['source_pins'], **build['input_pins']}.items():
        physical(Path(path))
        if sha(Path(path)) != digest:
            raise ValueError('Build source/input changed: ' + path)
    for row in installed['files']:
        physical(Path(row['target']))
        if sha(Path(row['target'])) != row['new_sha256']:
            raise ValueError('Current installed file changed: ' + row['target'])
    native = Path(build['candidate_DLL'])
    physical(native)
    if sha(native) != build['candidate_sha256']:
        raise ValueError('Compiled native DLL changed')
    stage = Path(__file__).resolve().parents[1] / 'artifacts/gpu-handoff-owner-transfer-v1-20261002'
    replacements = {2: stage / 'payload/plugins/cyberpunk_nr_adapter.py',
                    6: stage / 'payload/game/nr_texture_bridge_v1.py',
                    7: native, 8: stage / 'payload/game/nr_texture_bridge_v1.py'}
    plan = []
    for index, source in replacements.items():
        physical(source)
        row = installed['files'][index]
        if not (source == native or build['source_pins'].get(str(source)) == sha(source)):
            raise ValueError('Replacement is not in the frozen build inputs')
        plan.append({'index': index, 'source': str(source), 'target': row['target'],
                     'before_sha256': row['new_sha256'], 'after_sha256': sha(source)})
    out = args.output.resolve()
    physical(out)
    if out.drive.casefold() != 'd:':
        raise ValueError('Keep backups and receipts on D')
    out.mkdir(parents=True, exist_ok=False)
    detail = {'status': 'owner_transfer_install_plan_checked', 'files': plan,
              'parent_installation': str(args.current_installation),
              'parent_installation_sha256': sha(args.current_installation),
              'CPU_build_receipt': str(args.CPU_build),
              'CPU_build_receipt_sha256': sha(args.CPU_build),
              'math_files_changed': False, 'GPU_or_API_executed': False}
    dump(out / 'PLAN.json', detail)
    if args.mode == 'check':
        print(json.dumps({'status': detail['status'], 'files': len(plan), 'G_writes': False}))
        return
    if args.runtime_review is None:
        raise ValueError('Installation requires the main review of the real two-thread GPU run')
    review = read(args.runtime_review)
    if (review.get('status') != 'runtime_owner_transfer_reviewed' or
            review.get('candidate_DLL_sha256') != build['candidate_sha256'] or
            review.get('CPU_build_receipt_sha256') != sha(args.CPU_build) or
            any(review.get(k) is not True for k in ('source_hashes_unchanged', 'byte_checks_passed',
                    'two_actual_threads_observed', 'normal_exit'))):
        raise ValueError('The exact candidate has not passed the main runtime review')
    process = subprocess.run(['tasklist', '/FO', 'CSV', '/NH'], capture_output=True,
                             text=True, timeout=5, check=True)
    if any(row.lower().startswith(('"cyberpunk2077.exe"', '"re8.exe"'))
           for row in process.stdout.splitlines()):
        raise ValueError('Exit both games before replacing a loaded bridge')
    originals = {}
    for item in plan:
        target = Path(item['target'])
        originals[target] = target.read_bytes()
        if sha(target) != item['before_sha256']:
            raise ValueError('Target changed before backup')
        backup = out / ('old-%02d-%s' % (item['index'], target.name))
        backup.write_bytes(originals[target])
        item['backup'] = str(backup)
    written = []
    try:
        for item in plan:
            target, source = Path(item['target']), Path(item['source'])
            if sha(target) != item['before_sha256'] or sha(source) != item['after_sha256']:
                raise ValueError('Source or target changed before write')
            written.append(target)
            target.write_bytes(source.read_bytes())
            if sha(target) != item['after_sha256']:
                raise ValueError('Written file does not match frozen candidate')
    except BaseException:
        for target in reversed(written):
            target.write_bytes(originals[target])
        raise
    for item in plan:
        installed['files'][item['index']] = {'source': item['source'], 'target': item['target'],
            'old_sha256': item['before_sha256'], 'new_sha256': item['after_sha256'],
            'backup': item['backup']}
    installed['owner_transfer_installation'] = detail
    installed['owner_transfer_installed_utc'] = datetime.now(timezone.utc).isoformat()
    installed['owner_transfer_runtime_review'] = str(args.runtime_review)
    installed['owner_transfer_runtime_review_sha256'] = sha(args.runtime_review)
    dump(out / 'installed.json', installed)
    print(json.dumps({'status': 'owner_transfer_trial_installed', 'G_files_changed': len(plan),
                      'receipt': str(out / 'installed.json')}))


if __name__ == '__main__':
    main()
