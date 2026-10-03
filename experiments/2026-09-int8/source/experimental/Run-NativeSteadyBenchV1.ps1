$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$nrRoot='E:\Codex-NR-Reference\experiments\native-steady-bench-v1'
$nrBase='E:\Codex-NR-Reference'
$nrBin=Join-Path $nrRoot 'source\out'
$nrRun=Join-Path $nrRoot 'results'
if (Test-Path -LiteralPath $nrRun) {throw 'Preserve existing native benchmark'}
if (Get-Process -Name video2dlssnr -ErrorAction SilentlyContinue) {throw 'Another native NR host is running'}
[IO.Directory]::CreateDirectory($nrRun) | Out-Null
$env:LOCALAPPDATA="$nrBase\cache\local";$env:CUDA_CACHE_PATH="$nrBase\cache\cuda"
$env:TEMP="$nrBase\tmp";$env:TMP=$env:TEMP
$env:CODEX_NR_CAPTURE='0';$env:CODEX_NR_TRACE_DLL='';$env:CODEX_NR_SEQUENCE='0';$env:CODEX_NR_STEADY_BENCH='1'
$nrRecords=@()
try {
    foreach ($nrDimension in @('256x256','864x480','1920x1080')) {
        $nrInput=Join-Path $nrRoot ('inputs\'+$nrDimension+'.png')
        $nrOutput=Join-Path $nrRun $nrDimension
        [IO.Directory]::CreateDirectory($nrOutput) | Out-Null
        $nrArgs=@('--nr-run','--in',$nrInput,'--out',$nrOutput,'--dll-dir',$nrBin,'--adapter','0',
            '--nr-scale','1','--nr-preset','0','--nr-style','0','--nr-intensity','1',
            '--nr-local-structure','1','--nr-local-tone','1','--nr-ui-correction','0','--nr-detail','1','--nr-color','1')
        $nrBefore=@(& nvidia-smi.exe --query-gpu=name,driver_version,power.draw,temperature.gpu,clocks.sm,utilization.gpu --format=csv,noheader)
        $nrProcess=Start-Process -FilePath (Join-Path $nrBin 'video2dlssnr.exe') -ArgumentList $nrArgs -WorkingDirectory $nrOutput -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $nrOutput 'stdout.log') -RedirectStandardError (Join-Path $nrOutput 'stderr.log')
        $null=$nrProcess.Handle
        $nrDone=$nrProcess.WaitForExit(120000)
        if (!$nrDone) {$nrProcess.Kill();$nrProcess.WaitForExit();throw 'Native benchmark timed out'}
        if ($nrProcess.ExitCode -ne 0) {throw ('Native benchmark failed: '+$nrDimension)}
        $nrReport=Get-Content -Raw -LiteralPath (Join-Path $nrOutput 'native-timing.json') | ConvertFrom-Json
        if (!$nrReport.passed -or $nrReport.samples.Count -ne 360) {throw 'Incomplete native timings'}
        $nrRecords += [ordered]@{dimension=$nrDimension;exitCode=$nrProcess.ExitCode;before=$nrBefore;after=@(& nvidia-smi.exe --query-gpu=name,driver_version,power.draw,temperature.gpu,clocks.sm,utilization.gpu --format=csv,noheader);arguments=$nrArgs;inputSha256=(Get-FileHash -LiteralPath $nrInput).Hash}
    }
    [ordered]@{passed=$true;sessionId=(Get-Process -Id $PID).SessionId;cases=$nrRecords;runtimeSha256=(Get-FileHash -LiteralPath (Join-Path $nrBin 'nvngx_dlssnr.dll')).Hash;binarySha256=(Get-FileHash -LiteralPath (Join-Path $nrBin 'video2dlssnr.exe')).Hash;files=@(Get-ChildItem -LiteralPath $nrRun -File -Recurse | ForEach-Object {[ordered]@{relative=$_.FullName.Substring($nrRun.Length+1);bytes=$_.Length;sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})} | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $nrRun 'run.json') -Encoding UTF8
} catch {
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $nrRun 'failure.log') -Encoding UTF8
    throw
}
