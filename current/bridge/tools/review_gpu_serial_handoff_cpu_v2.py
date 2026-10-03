"""Review frozen CPU gate/host results plus the unchanged proved native partner."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--install-receipt', type=Path, required=True)
    parser.add_argument('--CPU-build', type=Path, required=True)
    parser.add_argument('--CPU-sidecar', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    install, build, tests = map(read, (args.install_receipt, args.CPU_build, args.CPU_sidecar))
    if build['status'] != 'serial_handoff_native_CPU_build_passed_GPU_UNTESTED':
        raise ValueError('Wrong CPU candidate')
    if (tests['status'] != 'serial_sidecar_CPP_and_host_CPU_passed' or
            tests['candidate_ASI_sha256'] != build['candidate_sha256'] or
            tests['CPU_build_receipt_sha256'] != sha(args.CPU_build) or
            tests['real_gate_two_CPU_threads_passed'] is not True or
            tests['old_strict_arm_passed'] is not True or
            tests['host_summary']['tests_run'] != 23):
        raise ValueError('Incomplete exact serial contract tests')
    pins = {**build['source_pins'], **build['input_pins'], **tests['source_pins'],
            **tests['host_summary']['source_pins']}
    binary = Path(build['candidate_ASI'])
    pins[str(binary)] = build['candidate_sha256']
    cpp = Path(tests['new_CPP_executable'])
    pins[str(cpp)] = tests['new_CPP_sha256']
    for path, digest in pins.items():
        if sha(Path(path)) != digest:
            raise ValueError('CPU result source/input changed: ' + path)
    if install['status'] != 'installed_trial_default_off' or len(install['files']) != 9:
        raise ValueError('Wrong current nine-file installation')
    for row in install['files']:
        if sha(Path(row['target'])) != row['new_sha256']:
            raise ValueError('Current game file changed: ' + row['target'])
    partner_path = Path(install['owner_transfer_runtime_review'])
    if sha(partner_path) != install['owner_transfer_runtime_review_sha256']:
        raise ValueError('Native GPU review changed')
    partner = read(partner_path)
    if (partner['status'] != 'runtime_owner_transfer_reviewed' or
            partner['candidate_DLL_sha256'] != install['files'][7]['new_sha256'] or
            any(partner.get(k) is not True for k in ('byte_checks_passed',
                'two_actual_threads_observed', 'normal_exit', 'source_hashes_unchanged'))):
        raise ValueError('Current native partner is not the proved owner-transfer binary')
    # Historical pre-install G baseline hashes are not current-source pins.
    # Check the exact four installed proven replacements and actual SYCL instead.
    for index in (2, 6, 7, 8):
        row = install['files'][index]
        if partner['verified_hashes'].get(row['source']) != row['new_sha256']:
            raise ValueError('Installed adapter/native partner differs from GPU proof')
        pins[row['source']] = row['new_sha256']
    for key in ('runtime_receipt', 'child_result', 'CPU_build_receipt'):
        path = Path(partner[key])
        if sha(path) != partner[key + '_sha256']:
            raise ValueError('Original native runtime evidence changed')
        pins[str(path)] = sha(path)
    runtime = Path(install['files'][5]['target']).parents[1]
    sycl = runtime / 'python/Library/bin/sycl9.dll'
    if partner['verified_hashes'].get(str(sycl)) != sha(sycl):
        raise ValueError('Actual proved SYCL library changed')
    pins[str(sycl)] = sha(sycl)
    ctest = args.CPU_sidecar.parent / 'ctest.log'
    text = ctest.read_text(encoding='utf-8')
    cpp_rows = [json.loads(match.group(1)) for line in text.splitlines()
                if (match := re.match(r'^1:\s*(\{.*\})\s*$', line))]
    if (len(cpp_rows) != 1 or cpp_rows[0].get('status') != 'CPU_passed' or
            cpp_rows[0].get('checks') != 540 or cpp_rows[0].get('persistent_host_threads') != 2 or
            cpp_rows[0].get('GPU_executed') is not False or cpp_rows[0].get('ASI_loaded') is not False or
            '100% tests passed, 0 tests failed out of 4' not in text):
        raise ValueError('Missing actual C++/host/legacy CTest completion')
    for path in (args.install_receipt, args.CPU_build, args.CPU_sidecar,
                 partner_path, ctest, Path(__file__).resolve()):
        pins[str(path)] = sha(path)
    result = {'status': 'serial_gpu_handoff_CPU_reviewed',
              'CPU_build_receipt': str(args.CPU_build),
              'CPU_build_receipt_sha256': sha(args.CPU_build),
              'CPU_sidecar_receipt': str(args.CPU_sidecar),
              'CPU_sidecar_receipt_sha256': sha(args.CPU_sidecar),
              'candidate_ASI_sha256': build['candidate_sha256'],
              'CPP_completion': cpp_rows[0], 'host_tests': 23,
              'old_strict_arm_passed': True, 'new_serial_CPU_passed': True,
              'current_native_partner_reviewed': True,
              'native_partner_GPU_review': str(partner_path),
              'math_shaders_deferred_consumer_unchanged': build['math_shaders_deferred_consumer_unchanged'],
              'source_hashes_unchanged': True, 'review_pins': pins,
              'GPU_or_API_executed_by_main': False, 'v2_game_acceptance': False}
    output = args.output.resolve()
    if output.drive.casefold() != 'd:':
        raise ValueError('Put review output on D')
    for part in (output, *output.parents):
        if part.is_junction() or part.is_symlink():
            raise ValueError('Redirected result path')
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('review_pins',)}))


if __name__ == '__main__':
    main()
