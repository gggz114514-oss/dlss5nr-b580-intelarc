param([ValidateSet('zero','integer','fractional','edge','field','fine')][string]$Mode='fine',[switch]$Trace,[switch]$ForceReset,[ValidateSet('none','mask-zero','mask-one','mask-r','mask-g','mask-b','mask-a','mask-field','mask-field-auto','ui-zero','ui-one','ui-field','ui-one-off','ui-field-off','backbuffer')][string]$AuxCase='none',[ValidateSet('balanced')][string]$UiControl='balanced',[ValidateSet('128x128','192x192','256x144')][string]$Dimension='128x128')
$ErrorActionPreference = 'Stop'
$ProgressPreference='SilentlyContinue'
$nrUiControls=@{
    'balanced'=@{style='0';intensity='1';tone='1';structure='1'}
    'i05'=@{style='0';intensity='0.5';tone='1';structure='1'}
    'i2'=@{style='0';intensity='2';tone='1';structure='1'}
    't05s15'=@{style='0';intensity='1';tone='0.5';structure='1.5'}
    'i2t05s15'=@{style='0';intensity='2';tone='0.5';structure='1.5'}
    'style1i05t05'=@{style='1';intensity='0.5';tone='0.5';structure='1'}
    'style2i2t05'=@{style='2';intensity='2';tone='0.5';structure='1'}
}
$nrSettings=$nrUiControls[$UiControl]

$root = 'E:\Codex-NR-Reference'
if ((Get-PSDrive -Name E).Free -lt 16GB) {throw 'Need 16GiB free on reference E'}
if (Get-Process -Name video2dlssnr -ErrorAction SilentlyContinue) {throw 'NR host already running'}
$bin = Join-Path $root 'video2dlssnr-aux-dimensions-v1\out'
$Capture=$true; $Batch=$true; $TraceVersion='v5'; $TraceFirst=13; $TraceLast=13; $Case='small-geometry-v1-'+$Dimension
$runtime = Join-Path $bin 'nvngx_dlssnr.dll'
if ((Get-FileHash -LiteralPath (Join-Path $bin 'video2dlssnr.exe')).Hash -ne 'AD69B6C840CCAA3ADB977D10CCDDA6DF12BB68CCE11C03904A313BC6EDD89B53') {throw 'Unexpected reference executable'}
if ((Get-FileHash -LiteralPath (Join-Path $bin 'nr_nvapi_trace.dll')).Hash -ne '1D1A002C01195D20DD40E950C4C5333827A619CEFF1919751EB52223D6DF7620') {throw 'Unexpected trace DLL'}
if ((Get-FileHash -LiteralPath $runtime).Hash -ne '6EB209E764F39872625DEBD6ABAF45E2BB6322F6F270F781F70C059AE30B3927') {
    throw 'Unexpected NR runtime'
}
$prefix = 'small-geometry-v1-' + $Dimension + '-' + $UiControl + '-' + $AuxCase + '-' + $Mode + $(if ($ForceReset) {'-reset-'} elseif ($Trace) {'-trace-'} else {'-sequence-'})
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
$env:CODEX_NR_CAPTURE = '1'
$env:CODEX_NR_DEPTH_MODE='zero'
$env:CODEX_NR_AUX_MODE=$AuxCase
$env:CODEX_NR_MOTION_MODE = $Mode
$env:CODEX_NR_SEQUENCE = if ($ForceReset) {'0'} else {'1'}
$env:CODEX_NR_TRACE_DLL = if ($Trace) { Join-Path $bin 'nr_nvapi_trace.dll' } else { '' }
$env:CODEX_NR_TRACE_FIRST = $TraceFirst.ToString()
$env:CODEX_NR_TRACE_LAST = $TraceLast.ToString()
$inputPath = "$root\experiments\small-geometry-v1\inputs\$Dimension"
$fixtureManifest=Join-Path $inputPath "manifest.json"
$fixture=Get-Content -LiteralPath $fixtureManifest -Raw | ConvertFrom-Json
if ($fixture.dimension -ne $Dimension -or $fixture.frames.Count -ne 4) {throw "Bad fixture"}
foreach ($frame in $fixture.frames) {
    if ($frame.file -notmatch "^frame0[0-3][.]png$") {throw "Unexpected frame name"}
    if ((Get-FileHash -LiteralPath (Join-Path $inputPath $frame.file)).Hash -ne $frame.sha256) {throw "Changed fixture frame"}
}
$arguments = @('--nr-run','--in',$inputPath,'--out',$output,
    '--dll-dir',$bin,'--adapter','0','--nr-scale','1','--nr-preset','0','--nr-style','0',
    '--nr-intensity','1','--nr-local-structure','1','--nr-local-tone','1',
    '--nr-ui-correction','0','--nr-detail','1','--nr-color','1','--nr-orig','--nr-diff','--verbose')
foreach ($nrPair in @(@('--nr-style','style'),@('--nr-intensity','intensity'),@('--nr-local-tone','tone'),@('--nr-local-structure','structure'))) {$arguments[[Array]::IndexOf($arguments,$nrPair[0])+1]=$nrSettings[$nrPair[1]]}
if ($AuxCase -eq 'mask-field-auto') {$arguments += '--nr-auto-mask'}
if ($AuxCase -in @('ui-zero','ui-one','ui-field')) {$arguments[[Array]::IndexOf($arguments,'--nr-ui-correction')+1]='1'}
$stdout = Join-Path $run 'stdout.log'
$stderr = Join-Path $run 'stderr.log'
$process = Start-Process -FilePath (Join-Path $bin 'video2dlssnr.exe') -ArgumentList $arguments `
    -WorkingDirectory $run -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$null = $process.Handle
$nrDeadline=[DateTime]::UtcNow.AddSeconds(240)
do {$finished=$process.WaitForExit(1000)} while (!$finished -and [DateTime]::UtcNow -lt $nrDeadline)
if (!$finished) {
    $process.Kill()
    $process.WaitForExit()
}
$exitCode = $process.ExitCode
$expectedImages = 4
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
        if ($TraceVersion -ne 'v2' -and $traceBuffers -lt 2) { $complete = $false }
    } else { $complete = $false }
}
$report = [ordered]@{
    runnerSha256=(Get-FileHash -LiteralPath $PSCommandPath).Hash
    fixtureManifestSha256=(Get-FileHash -LiteralPath $fixtureManifest).Hash
    dimension=$Dimension
    uiControlCase=$UiControl
    uiControlSettings=$nrSettings
    auxiliaryTextureCase=$AuxCase
    depthProbeCase='zero'
    depthInverted=$false
    depthBound=$true
    dimensionCase=$Case
    motionProbeMode=$Mode
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
    resetPattern=$(if ($ForceReset) {@(1,1,1,1)} else {@(1,0,0,1)})
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
