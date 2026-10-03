"""Build and run one isolated native4060 timing experiment, all remote files on E.

Only the timing harness changes. Repeated resident fixed image/zero motion calls
measure GPU timestamps and serial host completion separately, with no capture.
"""
import base64,hashlib,json,subprocess,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE.parent.parent/'nr-b580/reference'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental/native-steady-bench-v1')
assert not D.exists();D.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
archive=R/'video2dlssnr-real-flow-v2.zip'
with zipfile.ZipFile(archive) as z:files={n:z.read(n) for n in z.namelist()}
nr=files['src/nr.cpp'].decode('utf-8-sig')
marker='        const bool sequenceReset = !NrSequenceEnabled() || ok == 0 || ok == 12;'
assert nr.count(marker)==1
nr=nr.replace(marker,'#include "native_steady_bench_v1.inc"\n'+marker)
marker='        LoadNrProbeMotion(probeMotion, imgPath, motionProbeReset);'
assert nr.count(marker)==1
nr=nr.replace(marker,'        if (!GetEnvironmentVariableA("CODEX_NR_STEADY_BENCH", nullptr, 0))\n    '+marker)
files['src/nr.cpp']=nr.encode()
files['src/native_steady_bench_v1.inc']=(HERE/'native_steady_bench_v1.inc').read_bytes()
assert "v[0] == '1'" in files['src/nr_capture.h'].decode() or "value[0] == '1'" in files['src/nr_capture.h'].decode()
stage={'source/'+name:data for name,data in files.items()}
stage['Run-NativeSteadyBenchV1.ps1']=(HERE/'Run-NativeSteadyBenchV1.ps1').read_bytes()
sources={str(p):sha(p) for p in (archive,Path(__file__),HERE/'native_steady_bench_v1.inc',HERE/'Run-NativeSteadyBenchV1.ps1')}
for dimension,version in [('256x256',3),('864x480',2),('1920x1080',3)]:
    folder=R/f'inputs/flow-full-{dimension}-v{version}'
    manifest=folder/'manifest.json';m=js(manifest);p=folder/m['frames'][0]['file']
    stage[f'inputs/{dimension}.png']=p.read_bytes();sources[str(p)]=sha(p);sources[str(manifest)]=sha(manifest)
members={name:hashlib.sha256(data).hexdigest() for name,data in stage.items()}
stage['stage-manifest.json']=(json.dumps(members,indent=2)+'\n').encode()
bundle=D/'stage.zip'
with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
    for name,data in stage.items():z.writestr(name,data)
(D/'sources.json').write_text(json.dumps(dict(sources=sources,stage_files=members,archive_sha256=sha(bundle),scope=__doc__),indent=2)+'\n')
options=['-i',__import__('os').environ['NR_REFERENCE_SSH_KEY'],
         '-o','UserKnownHostsFile='+__import__('os').environ['NR_REFERENCE_KNOWN_HOSTS'],
         '-o','StrictHostKeyChecking=yes','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=8']

def ssh(script,name,timeout):
    encoded=base64.b64encode(script.encode('utf-16le')).decode()
    with (D/(name+'-stdout.log')).open('xb') as out,(D/(name+'-stderr.log')).open('xb') as err:
        p=subprocess.run(['ssh.exe',*options,__import__('os').environ['NR_REFERENCE_SSH_TARGET'],'powershell.exe','-NoProfile','-NonInteractive',
                          '-ExecutionPolicy','Bypass','-EncodedCommand',encoded],stdout=out,stderr=err,timeout=timeout)
    (D/(name+'-transport.json')).write_text(json.dumps(dict(returncode=p.returncode,script_sha256=hashlib.sha256(script.encode()).hexdigest())))
    if p.returncode:
        print((D/(name+'-stdout.log')).read_text(errors='replace')[-3000:])
        print((D/(name+'-stderr.log')).read_text(errors='replace')[-3000:])
        raise RuntimeError('Native transport/build/run failed; inspect existing result before retrying')

ssh(r"""$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$nrRoot='E:\Codex-NR-Reference\experiments\native-steady-bench-v1'
if (Test-Path -LiteralPath $nrRoot) {throw 'Native benchmark destination already exists'}
if ((Get-PSDrive E).Free -lt 2GB) {throw 'Need2GiB free on task E'}
if (Get-Process -Name video2dlssnr -ErrorAction SilentlyContinue) {throw 'Another NR host is running'}
[IO.Directory]::CreateDirectory($nrRoot) | Out-Null
""",'prepare',30)
subprocess.run(['scp.exe','-q',*options,str(bundle),__import__('os').environ['NR_REFERENCE_SSH_TARGET'] + ':E:/Codex-NR-Reference/experiments/native-steady-bench-v1/stage.zip'],check=True,timeout=120)
remote=r"""
$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$nrRoot='E:\Codex-NR-Reference\experiments\native-steady-bench-v1'
$nrBase='E:\Codex-NR-Reference'
$nrArchive=Join-Path $nrRoot 'stage.zip'
if ((Get-FileHash -LiteralPath $nrArchive).Hash -ne 'ARCHIVE_SHA') {throw 'Changed benchmark archive'}
Expand-Archive -LiteralPath $nrArchive -DestinationPath $nrRoot
$nrManifest=Get-Content -Raw -LiteralPath (Join-Path $nrRoot 'stage-manifest.json') | ConvertFrom-Json
foreach ($nrEntry in $nrManifest.PSObject.Properties) {
    if ((Get-FileHash -LiteralPath (Join-Path $nrRoot $nrEntry.Name)).Hash -ne $nrEntry.Value) {throw ('Changed staged file '+$nrEntry.Name)}
}
$env:TEMP="$nrBase\tmp";$env:TMP=$env:TEMP;$env:LOCALAPPDATA="$nrBase\cache\local";$env:CUDA_CACHE_PATH="$nrBase\cache\cuda"
Push-Location (Join-Path $nrRoot 'source')
try { & .\build.bat release *> (Join-Path $nrRoot 'build.log'); if ($LASTEXITCODE -ne 0) {Get-Content -LiteralPath (Join-Path $nrRoot 'build.log') -Tail 35;throw 'Native timing build failed'} } finally {Pop-Location}
$nrRuntime="$nrBase\video2dlssnr-55a4ceb5\out\nvngx_dlssnr.dll"
if ((Get-FileHash -LiteralPath $nrRuntime).Hash -ne '6EB209E764F39872625DEBD6ABAF45E2BB6322F6F270F781F70C059AE30B3927') {throw 'Changed native model DLL'}
Copy-Item -LiteralPath $nrRuntime -Destination (Join-Path $nrRoot 'source\out\nvngx_dlssnr.dll')
$nrDesktop=(Get-CimInstance Win32_ComputerSystem).UserName
if ($nrDesktop -ne $env:NR_REFERENCE_SESSION_USER) {throw 'Expected configured interactive reference session'}
$nrTask='Codex-NR-SteadyBench-'+[guid]::NewGuid().ToString('N')
$nrScript=Join-Path $nrRoot 'Run-NativeSteadyBenchV1.ps1'
$nrAction=New-ScheduledTaskAction -Execute 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -Argument ('-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File '+$nrScript) -WorkingDirectory $nrRoot
$nrPrincipal=New-ScheduledTaskPrincipal -UserId $nrDesktop -LogonType Interactive -RunLevel Limited
$nrSettings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 6) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $nrTask -Action $nrAction -Principal $nrPrincipal -Settings $nrSettings | Out-Null
$nrStarted=Get-Date
try {
    Start-ScheduledTask -TaskName $nrTask
    $nrDeadline=(Get-Date).AddSeconds(340)
    do {Start-Sleep -Seconds 1;$nrState=(Get-ScheduledTask -TaskName $nrTask).State;$nrInfo=Get-ScheduledTaskInfo -TaskName $nrTask}
    while (($nrState -eq 'Running' -or $nrInfo.LastRunTime -lt $nrStarted.AddSeconds(-1)) -and (Get-Date) -lt $nrDeadline)
    if ($nrState -eq 'Running') {throw 'Native experiment observation timeout'}
    if ($nrInfo.LastTaskResult -ne 0) {throw ('Native task failure '+$nrInfo.LastTaskResult)}
    $nrReport=Get-Content -Raw -LiteralPath (Join-Path $nrRoot 'results\run.json') | ConvertFrom-Json
    if (!$nrReport.passed) {throw 'Failed native report'}
    [ordered]@{passed=$true;taskResult=$nrInfo.LastTaskResult;removedTemporaryTask=$nrTask;cases=$nrReport.cases} | ConvertTo-Json -Depth 5
} finally {
    if ((Get-ScheduledTask -TaskName $nrTask).State -eq 'Running') {Stop-ScheduledTask -TaskName $nrTask}
    Unregister-ScheduledTask -TaskName $nrTask -Confirm:$false
}
""".replace('ARCHIVE_SHA',sha(bundle))
(D/'remote-command.ps1').write_text(remote,encoding='utf-8')
print('Native4060 release build and bounded timestamp benchmark starting',flush=True)
ssh(remote,'run',500)
subprocess.run(['scp.exe','-q','-r',*options,__import__('os').environ['NR_REFERENCE_SSH_TARGET'] + ':E:/Codex-NR-Reference/experiments/native-steady-bench-v1/results',str(D/'native')],check=True,timeout=120)
assert all(sha(p)==h for p,h in sources.items())
print(json.dumps(dict(passed=True,native_results=str(D/'native'),stage_bytes=bundle.stat().st_size)),flush=True)
