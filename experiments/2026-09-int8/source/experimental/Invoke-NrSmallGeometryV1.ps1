# One-time logged-on-session comparison. No password is stored; task removed afterwards.
param([ValidateSet('zero','integer','fractional','edge','field','fine')][string]$Mode='fine',[switch]$Trace,[switch]$ForceReset,[ValidateSet('none','mask-zero','mask-one','mask-r','mask-g','mask-b','mask-a','mask-field','mask-field-auto','ui-zero','ui-one','ui-field','ui-one-off','ui-field-off','backbuffer')][string]$AuxCase='none',[ValidateSet('balanced')][string]$UiControl='balanced',[ValidateSet('128x128','192x192','256x144')][string]$Dimension='128x128')
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$root='E:\Codex-NR-Reference'
$desktopUser=(Get-CimInstance Win32_ComputerSystem).UserName
if ($desktopUser -ne $env:NR_REFERENCE_SESSION_USER) { throw 'Expected configured reference desktop session is not logged on' }
$taskName='Codex-NR-SmallGeometry-' + [guid]::NewGuid().ToString('N')
$script=Join-Path $root 'experiments\small-geometry-v1\Run-NrSmallGeometryV1.ps1'
if (!(Test-Path -LiteralPath $script)) { throw 'Smoke script missing' }
$started=Get-Date
$scriptArguments = '-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File ' + $script
$scriptArguments += ' -Mode ' + $Mode + ' -AuxCase ' + $AuxCase + ' -UiControl ' + $UiControl + ' -Dimension ' + $Dimension
if ($Trace) { $scriptArguments += ' -Trace' }
if ($ForceReset) { $scriptArguments += ' -ForceReset' }
$action=New-ScheduledTaskAction -Execute 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' `
    -Argument $scriptArguments -WorkingDirectory $root
$principal=New-ScheduledTaskPrincipal -UserId $desktopUser -LogonType Interactive -RunLevel Limited
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings | Out-Null
try {
    Start-ScheduledTask -TaskName $taskName
    $deadline=(Get-Date).AddSeconds(270)
    do {
        Start-Sleep -Seconds 1
        $state=(Get-ScheduledTask -TaskName $taskName).State
        $info=Get-ScheduledTaskInfo -TaskName $taskName
    } while (($state -eq 'Running' -or $info.LastRunTime -lt $started.AddSeconds(-1)) -and (Get-Date) -lt $deadline)
    if ($state -eq 'Running') { throw 'Interactive comparison did not finish in time' }
    $run=Get-ChildItem -LiteralPath (Join-Path $root 'runs') -Directory -Filter ('small-geometry-v1-'+$Dimension+'-*') |
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
