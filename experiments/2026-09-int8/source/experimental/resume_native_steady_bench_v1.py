"""Resume the staged native benchmark after the Windows command-line length failure.

The rejected command did not run: remote directory held stage.zip only. Upload
the exact previously saved script and invoke its path, avoiding shell length limits.
"""
import hashlib,json,subprocess
from pathlib import Path
D=Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental/native-steady-bench-v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=json.loads((D/'sources.json').read_text())
assert all(sha(p)==h for p,h in source['sources'].items())
assert json.loads((D/'run-transport.json').read_text())['returncode']==1
assert '命令行太长' in (D/'run-stderr.log').read_bytes().decode('gb18030')
script=D/'remote-command.ps1'
assert hashlib.sha256(script.read_text().encode()).hexdigest()==json.loads((D/'run-transport.json').read_text())['script_sha256']
opts=['-i',__import__('os').environ['NR_REFERENCE_SSH_KEY'],
      '-o','UserKnownHostsFile='+__import__('os').environ['NR_REFERENCE_KNOWN_HOSTS'],
      '-o','StrictHostKeyChecking=yes','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=8']
remote='E:/Codex-NR-Reference/experiments/native-steady-bench-v1/'
subprocess.run(['scp.exe','-q',*opts,str(script),__import__('os').environ['NR_REFERENCE_SSH_TARGET'] + ':'+remote+'remote-command.ps1'],check=True,timeout=30)
with (D/'resume-stdout.log').open('xb') as out,(D/'resume-stderr.log').open('xb') as err:
    p=subprocess.run(['ssh.exe',*opts,__import__('os').environ['NR_REFERENCE_SSH_TARGET'],'powershell.exe','-NoProfile','-NonInteractive',
        '-ExecutionPolicy','Bypass','-File',remote+'remote-command.ps1'],stdout=out,stderr=err,timeout=500)
(D/'resume-transport.json').write_text(json.dumps(dict(returncode=p.returncode,runner_sha256=sha(__file__),script_sha256=sha(script))))
if p.returncode:
    print((D/'resume-stdout.log').read_bytes().decode('utf-8-sig',errors='replace')[-4500:])
    print((D/'resume-stderr.log').read_bytes().decode('utf-8-sig',errors='replace')[-4500:])
    raise RuntimeError('Native benchmark failure: inspect staged files before retry')
subprocess.run(['scp.exe','-q','-r',*opts,__import__('os').environ['NR_REFERENCE_SSH_TARGET'] + ':'+remote+'results',str(D/'native')],check=True,timeout=120)
assert all(sha(p)==h for p,h in source['sources'].items())
print(json.dumps(dict(passed=True,native_results=str(D/'native'))),flush=True)
