"""CPU process classification: exact pinned scripts, games, stable authorizations."""
from __future__ import annotations
from pathlib import Path
import re
import shlex
import hashlib
import runner_common as c

MAIN_CPU = ('cpu_checks.py','cpu_registry_checks.py','cpu_capture_role_checks.py',
            'cpu_native_protocol_checks.py','cpu_session_protocol_checks.py',
            'freeze_complete_cpu.py','freeze_numeric_cpu.py','prepare_cpu.py','merge_cpu.py',
            'build_checkbox_payload_cpu.py','apply_branch_patch_cpu.py')
GPU_FILES = (c.ROOT/'luna/phase1/phase1_child.py', c.ROOT/'luna/phase1/phase1_runner.py',
             c.ROOT/'luna/baseline_gpu_worker.py', c.ROOT/'luna/baseline_runner.py',
             c.SCRIPT_DIR/'phase2_child.py', c.SCRIPT_DIR/'gpu_worker.py',
             c.V4/'gpu_worker.py', c.V4/'runner.py')
GAME_NAMES = frozenset({'cyberpunk2077.exe','re8.exe','residentevil.exe','zenlesszonezero.exe',
                        'zenless.exe','ollama.exe','llama-server.exe','comfyui.exe'})
AUTHORIZATIONS = None

def script_path(command):
    if not command: return None
    try: tokens = shlex.split(command, posix=False)
    except ValueError: return None
    tokens = [token.strip('"\'') for token in tokens]
    # -c/-m can execute arbitrary Python; they require a stable Main receipt.
    if '-c' in tokens or '-m' in tokens: return None
    for token in tokens[1:]:
        if token.lower().endswith('.py') and Path(token).is_absolute():
            return c.no_reparse(token)
    return None

def load_authorizations(path):
    global AUTHORIZATIONS
    if path is None:
        AUTHORIZATIONS = None; return None
    path = c.no_reparse(path)
    c.require(path.is_relative_to(c.ROOT) or path.is_relative_to(c.DATA), 'Main process receipt must be in project/task-owned data')
    row = c.record(path)
    value = c.read(path)
    c.require(value.get('schema') == 'b580-phase2-main-cpu-process-authorizations-v1', 'Unknown process authorization receipt')
    for item in value['processes']:
        c.require(type(item['pid']) is int and item['GPU_access'] is False
                  and item['authorization'] == 'Main explicit stable CPU background authorization'
                  and all(item.get(k) for k in ('creation_time','executable_path','command_line_sha256')), 'Incomplete stable process authorization')
    AUTHORIZATIONS = row
    return row

def trusted_cpu_scripts():
    ready = c.read(c.SCRIPT_DIR/'READY.json')
    result = {str((c.ROOT/name).resolve()).casefold(): digest for name,digest in ready['main_cpu_script_pins'].items()}
    for name in ('cpu_checks.py','cpu_tests.py','freeze_ready_cpu.py'):
        if name in ready['files']:
            result[str((c.SCRIPT_DIR/name).resolve()).casefold()] = ready['files'][name]
    # This one new worker is admitted by script SHA, bundled interpreter SHA
    # and its full isolated CLI. It never enters the generic script exception.
    name='cpu_comparison/metrics_cpu_worker.py'
    if name in ready['files']:
        result[str((c.SCRIPT_DIR/name).resolve()).casefold()]=ready['files'][name]
    return result

def classify(row, trusted, authorized=()):
    if row['ProcessId'] == __import__('os').getpid():
        return dict(classification='self_cpu_supervisor', blocking=False)
    name = row.get('Name','').casefold()
    if name in GAME_NAMES:
        return dict(classification='known_game_or_GPU_service', blocking=True)
    is_python = bool(re.fullmatch(r'(?:python|pythonw|python[0-9.]*)(?:\.exe)?', name))
    if not is_python:
        return dict(classification='other_OS_process', blocking=False)
    command = row.get('CommandLine') or ''
    path = script_path(command)
    if path is not None:
        expected = trusted.get(str(path).casefold())
        if path == (c.SCRIPT_DIR/'cpu_comparison/metrics_cpu_worker.py').resolve():
            try:
                import metrics_subprocess as comparison
                runtime=comparison.runtime_contract()
                tokens=[v.strip('"\'') for v in shlex.split(command,posix=False)]
                wanted=comparison.worker_command()
                exact=(len(tokens)==len(wanted) and
                    Path(tokens[0])==comparison.CPU_PYTHON and tokens[1:-1]==wanted[1:-1] and
                    Path(tokens[-1])==comparison.WORKER and
                    Path(row.get('ExecutablePath') or '')==comparison.CPU_PYTHON and name=='python.exe' and
                    expected==runtime['worker_files']['metrics_cpu_worker.py'] and c.sha(path)==expected)
                if exact:
                    return dict(classification='exact_bundled_CPU_comparison',blocking=False,
                        script=c.record(path),interpreter=runtime['python'],isolated_command=tokens)
                reason='Exact CPU interpreter/script/hash/isolated flags required'
            except Exception as exc:
                reason=str(exc)
            return dict(classification='CPU_comparison_fail_closed',blocking=True,reason=reason,
                        script=str(path),executable_path=row.get('ExecutablePath'))
        if expected is not None and path.is_file() and c.sha(path) == expected:
            return dict(classification='exact_pinned_CPU_script', blocking=False, script=c.record(path))
        if path in tuple(p.resolve() for p in GPU_FILES) or path == (c.SCRIPT_DIR/'phase2_runner.py').resolve():
            # The same known supervisor is CPU-only when its real CLI requests
            # preflight; GPU opt-in can never be authorized as a CPU exception.
            tokens = [v.strip('"\'') for v in shlex.split(command,posix=False)]
            if path.name in ('phase1_runner.py','phase2_runner.py') and '--preflight-only' in tokens and '--execute-gpu' not in tokens:
                return dict(classification='known_supervisor_CPU_preflight',blocking=False,script=str(path))
            return dict(classification='known_GPU_job', blocking=True, script=str(path))
    digest = hashlib.sha256(command.encode('utf-8')).hexdigest()
    for item in authorized:
        if (item['pid'] == row['ProcessId'] and item['creation_time'] == row.get('CreationTime')
                and Path(item['executable_path']) == Path(row.get('ExecutablePath') or '')
                and item['command_line_sha256'] == digest):
            return dict(classification='Main_authorized_stable_CPU_background', blocking=False)
    return dict(classification='unknown_python_fail_closed', blocking=True,
                authorization_required=dict(pid=row['ProcessId'], creation_time=row.get('CreationTime'),
                    executable_path=row.get('ExecutablePath'), command_line_sha256=digest))

def classify_inventory(rows):
    trusted = trusted_cpu_scripts()
    authorized = [] if AUTHORIZATIONS is None else c.read(c.checked(AUTHORIZATIONS))['processes']
    return [dict(row, **classify(row,trusted,authorized)) for row in rows]
