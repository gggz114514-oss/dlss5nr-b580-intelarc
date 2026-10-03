"""Install three reviewed serial-protocol files; model/native math stay pinned."""
import argparse
import copy
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


def dump(path, data):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('check', 'install'))
    parser.add_argument('--current-installation', type=Path, required=True)
    parser.add_argument('--CPU-review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    installed, review = read(args.current_installation), read(args.CPU_review)
    if installed.get('status') != 'installed_trial_default_off' or len(installed['files']) != 9:
        raise ValueError('Expected the actual-overlay owner-transfer nine-file trial')
    if (review.get('status') != 'serial_gpu_handoff_CPU_reviewed' or
            any(review.get(key) is not True for key in (
                'old_strict_arm_passed', 'new_serial_CPU_passed', 'current_native_partner_reviewed',
                'math_shaders_deferred_consumer_unchanged', 'source_hashes_unchanged'))):
        raise ValueError('Exact serial contract has not passed main CPU review')
    build_path = Path(review['CPU_build_receipt'])
    if sha(build_path) != review['CPU_build_receipt_sha256']:
        raise ValueError('CPU build receipt changed')
    build = read(build_path)
    if build.get('status') != 'serial_handoff_native_CPU_build_passed_GPU_UNTESTED':
        raise ValueError('Wrong serial CPU build')
    all_pins = {**build['source_pins'], **build['input_pins'], **review['review_pins']}
    for path, digest in all_pins.items():
        physical(Path(path))
        if sha(Path(path)) != digest:
            raise ValueError('Reviewed source/input changed: ' + path)
    for row in installed['files']:
        target = Path(row['target'])
        physical(target)
        if sha(target) != row['new_sha256']:
            raise ValueError('Current installed target changed: ' + str(target))
    binary = Path(build['candidate_ASI'])
    physical(binary)
    if sha(binary) != build['candidate_sha256']:
        raise ValueError('Compiled candidate ASI changed')
    stage = Path(__file__).resolve().parents[1] / 'artifacts/gpu-handoff-serial-owner-v2-20261002'
    replacements = {0: binary, 1: stage / 'payload/game/cyberpunk_nr_web.py',
                    5: stage / 'payload/game/nr_gpu_handoff_host_v1.py'}
    plan = []
    for index, source in replacements.items():
        physical(source)
        row = installed['files'][index]
        if source != binary and build['source_pins'].get(str(source)) != sha(source):
            raise ValueError('Replacement is outside frozen CPU build')
        plan.append({'index': index, 'source': str(source), 'target': row['target'],
                     'before_sha256': row['new_sha256'], 'after_sha256': sha(source)})
    output = args.output.resolve()
    physical(output)
    if output.drive.casefold() != 'd:':
        raise ValueError('Keep backups and receipts on physical D')
    output.mkdir(parents=True, exist_ok=False)
    detail = {'status': 'serial_handoff_three_file_plan_checked', 'files': plan,
              'parent_installation': str(args.current_installation),
              'parent_installation_sha256': sha(args.current_installation),
              'CPU_review': str(args.CPU_review), 'CPU_review_sha256': sha(args.CPU_review),
              'math_files_changed': False, 'GPU_or_API_executed': False,
              'live_acceptance': False, 'unchanged_native_partner_sha256': installed['files'][7]['new_sha256']}
    dump(output / 'PLAN.json', detail)
    if args.mode == 'check':
        print(json.dumps({'status': detail['status'], 'files': 3, 'G_writes': False}))
        return
    process = subprocess.run(['tasklist', '/FO', 'CSV', '/NH'], capture_output=True,
                             text=True, timeout=5, check=True)
    if any(line.lower().startswith(('"cyberpunk2077.exe"', '"re8.exe"'))
           for line in process.stdout.splitlines()):
        raise ValueError('Exit both games before replacing loaded bridge files')
    originals = {}
    for item in plan:
        target = Path(item['target'])
        originals[target] = target.read_bytes()
        if sha(target) != item['before_sha256']:
            raise ValueError('Target changed before backup')
        backup = output / ('old-%02d-%s' % (item['index'], target.name))
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
                raise ValueError('Written file differs from candidate')
    except BaseException:
        for target in reversed(written):
            target.write_bytes(originals[target])
        raise
    updated = copy.deepcopy(installed)
    for item in plan:
        updated['files'][item['index']] = {'source': item['source'], 'target': item['target'],
            'old_sha256': item['before_sha256'], 'new_sha256': item['after_sha256'], 'backup': item['backup']}
    for row in updated['files']:
        if sha(Path(row['target'])) != row['new_sha256']:
            raise ValueError('Nine-file installation readback failed')
    updated['serial_gpu_handoff_installation'] = detail
    updated['serial_gpu_handoff_installed_utc'] = datetime.now(timezone.utc).isoformat()
    dump(output / 'installed.json', updated)
    print(json.dumps({'status': 'serial_handoff_trial_installed', 'G_files_changed': 3,
                      'receipt': str(output / 'installed.json')}))


if __name__ == '__main__':
    main()
