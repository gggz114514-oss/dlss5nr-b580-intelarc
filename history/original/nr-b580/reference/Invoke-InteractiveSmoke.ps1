# One-time logged-on-session comparison. No password is stored; task removed afterwards.
param([switch]$Capture, [switch]$Batch, [switch]$Trace, [ValidatePattern('^v[0-9]+$')][string]$TraceVersion = 'v3', [ValidateRange(13,164)][int]$TraceFirst=13, [ValidateRange(13,164)][int]$TraceLast=16)
$ErrorActionPreference='Stop'
$root='E:\Codex-NR-Reference'
$desktopUser=(Get-CimInstance Win32_ComputerSystem).UserName
if ($desktopUser -ne $env:NR_REFERENCE_SESSION_USER) { throw 'Expected configured reference desktop session is not logged on' }
$taskName='Codex-NR-Smoke-' + [guid]::NewGuid().ToString('N')
$script=Join-Path $root 'Run-NrSmoke.ps1'
if (!(Test-Path -LiteralPath $script)) { throw 'Smoke script missing' }
$started=Get-Date
$scriptArguments = '-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File ' + $script
if ($Capture) { $scriptArguments += ' -Capture' }
if ($Batch) { $scriptArguments += ' -Batch' }
if ($Trace) { $scriptArguments += ' -Trace -TraceVersion ' + $TraceVersion + ' -TraceFirst ' + $TraceFirst + ' -TraceLast ' + $TraceLast }
$action=New-ScheduledTaskAction -Execute 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' `
    -Argument $scriptArguments -WorkingDirectory $root
$principal=New-ScheduledTaskPrincipal -UserId $desktopUser -LogonType Interactive -RunLevel Limited
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 3) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings | Out-Null
try {
    Start-ScheduledTask -TaskName $taskName
    $deadline=(Get-Date).AddSeconds(150)
    do {
        Start-Sleep -Seconds 1
        $state=(Get-ScheduledTask -TaskName $taskName).State
        $info=Get-ScheduledTaskInfo -TaskName $taskName
    } while (($state -eq 'Running' -or $info.LastRunTime -lt $started.AddSeconds(-1)) -and (Get-Date) -lt $deadline)
    if ($state -eq 'Running') { throw 'Interactive comparison did not finish in time' }
    $run=Get-ChildItem -LiteralPath (Join-Path $root 'runs') -Directory |
        Where-Object { $_.CreationTime -ge $started.AddSeconds(-1) } | Sort-Object CreationTime -Descending | Select-Object -First 1
    if (!$run -or !(Test-Path -LiteralPath (Join-Path $run.FullName 'run.json'))) {
        throw ('Interactive task returned ' + $info.LastTaskResult + ' without a report')
    }
    $reportText = Get-Content -LiteralPath (Join-Path $run.FullName 'run.json') -Raw
    $reportText
    $report = ConvertFrom-Json $reportText
    if ($null -eq $report.exitCode -or $report.exitCode -ne 0 -or $report.timedOut -or $report.completedAllInputs -eq $false) {
        throw 'Logged-on-session NR experiment failed or incomplete'
    }
} finally {
    if ((Get-ScheduledTask -TaskName $taskName).State -eq 'Running') { Stop-ScheduledTask -TaskName $taskName }
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}
