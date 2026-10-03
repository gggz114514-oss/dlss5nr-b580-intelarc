"""Run one isolated logged-on 4060 sequence, then retrieve the exact report.

Transport is observed through this process; never start another native run to
work around an observation timeout. Large native buffers stay under the task E
directory until copied to the local D experiment directory.
"""
import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path, PureWindowsPath

HERE = Path(__file__).resolve().parent
D = Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental/small-geometry-v1')
parser = argparse.ArgumentParser()
parser.add_argument('--dimension', required=True, choices=['128x128', '192x192', '256x144'])
parser.add_argument('--kind', required=True, choices=['sequence', 'trace'])
args = parser.parse_args()
out = D / (args.dimension + '-' + args.kind)
assert not out.exists()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest = json.loads((D / 'manifest.json').read_text())
assert all(sha(p) == h for p, h in manifest['sources'].items())
for name, value in manifest['files'].items():
    assert sha(D / 'stage' / name) == value['sha256']
out.mkdir()
options = ['-i', __import__('os').environ['NR_REFERENCE_SSH_KEY'],
           '-o', 'UserKnownHostsFile='+__import__('os').environ['NR_REFERENCE_KNOWN_HOSTS'],
           '-o', 'StrictHostKeyChecking=yes', '-o', 'IdentitiesOnly=yes',
           '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8']
stage_sha = sha(D / 'stage/stage-manifest.json')
remote = r"""
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$nrRoot='E:\Codex-NR-Reference\experiments\small-geometry-v1'
$nrManifest=Join-Path $nrRoot 'stage-manifest.json'
if ((Get-FileHash -LiteralPath $nrManifest).Hash -ne 'STAGE_SHA') {throw 'Changed stage manifest'}
$nrFiles=Get-Content -LiteralPath $nrManifest -Raw | ConvertFrom-Json
foreach ($nrItem in $nrFiles.PSObject.Properties) {
    $nrPath=Join-Path $nrRoot $nrItem.Name
    if ((Get-FileHash -LiteralPath $nrPath).Hash -ne $nrItem.Value.sha256) {throw ('Changed staged file '+$nrItem.Name)}
}
& (Join-Path $nrRoot 'Invoke-NrSmallGeometryV1.ps1') -Dimension DIMENSION -Mode fine TRACE_FLAG
""".replace('STAGE_SHA', stage_sha).replace('DIMENSION', args.dimension).replace('TRACE_FLAG', '-Trace' if args.kind == 'trace' else '')
encoded = base64.b64encode(remote.encode('utf-16le')).decode('ascii')
command = ['ssh.exe', *options, __import__('os').environ['NR_REFERENCE_SSH_TARGET'], 'powershell.exe', '-NoProfile',
           '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', encoded]
print(f'Native {args.dimension} {args.kind} starting', flush=True)
with (out / 'ssh-stdout.json').open('xb') as stdout, (out / 'ssh-stderr.log').open('xb') as stderr:
    process = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=330)
receipt = dict(returncode=process.returncode, runner_sha256=sha(Path(__file__)),
               stage_manifest_sha256=stage_sha, dimension=args.dimension, kind=args.kind)
(out / 'transport.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
if process.returncode:
    print((out / 'ssh-stderr.log').read_bytes().decode('utf-8', errors='replace')[-6000:], flush=True)
    raise RuntimeError('Native call failed; preserve records and inspect before any retry')
console_report = json.loads((out / 'ssh-stdout.json').read_bytes().decode('utf-8-sig', errors='replace'))
assert console_report['completedAllInputs'] and console_report['exitCode'] == 0 and not console_report['timedOut']
run = PureWindowsPath(console_report['runDirectory'])
assert run.parent == PureWindowsPath(r'E:\Codex-NR-Reference\runs')
assert run.name.startswith('small-geometry-v1-' + args.dimension + '-')
assert sum(v['bytes'] for k in ('outputs', 'traceFiles') for v in console_report[k]) < 8 * 1024**3
subprocess.run(['scp.exe', '-q', '-r', *options, __import__('os').environ['NR_REFERENCE_SSH_TARGET'] + ':' + run.as_posix(), str(out / 'native')], check=True, timeout=300)
report = json.loads((out / 'native/run.json').read_text(encoding='utf-8-sig'))
for key in ('runDirectory', 'outputs', 'traceFiles', 'arguments', 'exitCode', 'completedAllInputs'):
    assert report[key] == console_report[key]
print(json.dumps(dict(dimension=args.dimension, kind=args.kind, complete=True,
                      run_directory=report['runDirectory'], report_sha256=sha(out / 'native/run.json'),
                      bytes=sum(v['bytes'] for k in ('outputs', 'traceFiles') for v in report[k])), indent=2), flush=True)
