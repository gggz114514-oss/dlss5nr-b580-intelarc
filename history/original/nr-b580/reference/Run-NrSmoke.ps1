param([switch]$Capture, [switch]$Batch, [switch]$Trace, [ValidatePattern('^v[0-9]+$')][string]$TraceVersion = 'v3', [ValidateRange(13,164)][int]$TraceFirst=13, [ValidateRange(13,164)][int]$TraceLast=16)
$ErrorActionPreference = 'Stop'
$root = 'E:\Codex-NR-Reference'
$bin = Join-Path $root 'video2dlssnr-55a4ceb5\out'
if ($Capture) { $bin = Join-Path $root 'video2dlssnr-capture-v1\out' }
if ($Trace) { $Capture = $true; $bin = Join-Path $root ('video2dlssnr-trace-' + $TraceVersion + '\out') }
$runtime = Join-Path $bin 'nvngx_dlssnr.dll'
if ((Get-FileHash -LiteralPath $runtime).Hash -ne '6EB209E764F39872625DEBD6ABAF45E2BB6322F6F270F781F70C059AE30B3927') {
    throw 'Unexpected NR runtime'
}
$prefix = if ($Capture) { 'capture-256-' } else { 'smoke-256-' }
if ($Trace) { $prefix = 'nvapi-trace-256-' }
$run = Join-Path $root ('runs\' + $prefix + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$output = Join-Path $run 'output'
foreach ($directory in @($run, $output, "$root\cache\local", "$root\cache\cuda")) {
    [IO.Directory]::CreateDirectory($directory) | Out-Null
}
# These environment changes apply only to this experiment and its children.
$env:LOCALAPPDATA = "$root\cache\local"
$env:CUDA_CACHE_PATH = "$root\cache\cuda"
$env:TEMP = "$root\tmp"
$env:TMP = $env:TEMP
$env:CODEX_NR_CAPTURE = if ($Capture) { '1' } else { '0' }
$env:CODEX_NR_TRACE_DLL = if ($Trace) { Join-Path $bin 'nr_nvapi_trace.dll' } else { '' }
$env:CODEX_NR_TRACE_FIRST = $TraceFirst.ToString()
$env:CODEX_NR_TRACE_LAST = $TraceLast.ToString()
$inputPath = if ($Batch) { "$root\inputs\smoke-256" } else { "$root\inputs\smoke-256\gradient.png" }
$arguments = @('--nr-run','--in',$inputPath,'--out',$output,
    '--dll-dir',$bin,'--adapter','0','--nr-scale','1','--nr-preset','0','--nr-style','0',
    '--nr-intensity','1','--nr-local-structure','1','--nr-local-tone','1',
    '--nr-ui-correction','0','--nr-detail','1','--nr-color','1','--nr-orig','--nr-diff','--verbose')
$stdout = Join-Path $run 'stdout.log'
$stderr = Join-Path $run 'stderr.log'
$process = Start-Process -FilePath (Join-Path $bin 'video2dlssnr.exe') -ArgumentList $arguments `
    -WorkingDirectory $run -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$null = $process.Handle
$finished = $process.WaitForExit(120000)
if (!$finished) {
    $process.Kill()
    $process.WaitForExit()
}
$exitCode = $process.ExitCode
$expectedImages = if ($Batch) { 3 } else { 1 }
$resultCount = @(Get-ChildItem -LiteralPath $output -File -Filter '*_nr.png').Count
$captureCount = @(Get-ChildItem -LiteralPath $output -File -Filter '*_capture.json').Count
$complete = $finished -and $null -ne $exitCode -and $exitCode -eq 0 -and $resultCount -eq $expectedImages
if ($Capture -and $captureCount -ne $expectedImages) { $complete = $false }
$traceDirectory = Join-Path $output 'nvapi-trace'
$traceLaunches = 0
$traceBuffers = 0
if ($Trace) {
    $eventPath = Join-Path $traceDirectory 'events.jsonl'
    if (Test-Path -LiteralPath $eventPath) {
        $events = @(Get-Content -LiteralPath $eventPath | ForEach-Object { ConvertFrom-Json $_ })
        $traceLaunches = @($events | Where-Object { $_.event -eq 'launch' -and $_.written }).Count
        $traceBuffers = @($events | Where-Object { $_.event -eq 'buffer_written' -and $_.written }).Count
        if ($traceLaunches -eq 0 -or @($events | Where-Object { $_.event -eq 'shutdown' }).Count -ne 1) { $complete = $false }
        if ($traceBuffers -lt 2) { $complete = $false }
    } else { $complete = $false }
}
$report = [ordered]@{
    runDirectory=$run
    parentSessionId=(Get-Process -Id $PID).SessionId
    tensorCapture=[bool]$Capture
    nativeKernelTrace=[bool]$Trace
    capturedKernelLaunches=$traceLaunches
    capturedArenaBuffers=$traceBuffers
    traceDllSha256=$(if ($Trace) { (Get-FileHash -LiteralPath (Join-Path $bin 'nr_nvapi_trace.dll')).Hash } else { $null })
    traceFiles=@(if ($Trace -and (Test-Path -LiteralPath $traceDirectory)) { Get-ChildItem -LiteralPath $traceDirectory -File | ForEach-Object { [ordered]@{name=$_.Name; bytes=$_.Length; sha256=(Get-FileHash -LiteralPath $_.FullName).Hash} } })
    expectedImages=$expectedImages
    completedAllInputs=$complete
    executable=(Join-Path $bin 'video2dlssnr.exe')
    executableSha256=(Get-FileHash -LiteralPath (Join-Path $bin 'video2dlssnr.exe')).Hash
    runtimeSha256=(Get-FileHash -LiteralPath $runtime).Hash
    driver=@(& nvidia-smi.exe --query-gpu=name,driver_version --format=csv,noheader | ForEach-Object { $_.ToString() })
    arguments=$arguments
    reset=1
    srScale=1
    modelInputColorSpace='sRGB'
    timedOut=(!$finished)
    exitCode=$exitCode
    outputs=@(Get-ChildItem -LiteralPath $output -File | ForEach-Object { [ordered]@{name=$_.Name; bytes=$_.Length; sha256=(Get-FileHash -LiteralPath $_.FullName).Hash} })
    stdoutTail=@(Get-Content -LiteralPath $stdout -Tail 35 | ForEach-Object { $_.ToString() })
    stderrTail=@(Get-Content -LiteralPath $stderr -Tail 15 | ForEach-Object { $_.ToString() })
}
$report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $run 'run.json') -Encoding UTF8
$report | ConvertTo-Json -Depth 5
if (!$complete) { throw 'NR smoke test failed or incomplete; see captured run logs' }
