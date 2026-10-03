param(
    [Parameter(Mandatory=$true)][string]$InstallReceipt,
    [switch]$Launch
)
$ErrorActionPreference = 'Stop'
$taskInstalled = Get-Content -LiteralPath $InstallReceipt -Raw | ConvertFrom-Json
if ($taskInstalled.status -ne 'installed_trial_default_off' -or $taskInstalled.files.Count -notin @(8,9)) {
    throw 'A completed reviewed trial installation is required.'
}
foreach ($taskFile in $taskInstalled.files) {
    $taskDigest = (Get-FileHash -LiteralPath $taskFile.target -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($taskDigest -ne $taskFile.new_sha256) { throw ('Installed file changed: ' + $taskFile.target) }
}
$taskExe = 'G:\epic\Cyberpunk2077\bin\x64\Cyberpunk2077.exe'
if (-not (Test-Path -LiteralPath $taskExe -PathType Leaf)) { throw 'Game executable missing.' }
if (Get-Process -Name Cyberpunk2077,re8 -ErrorAction SilentlyContinue) { throw 'A game is already running.' }
if (-not $Launch) {
    [ordered]@{status='verified_launch_plan';installed_files=$taskInstalled.files.Count;exe=$taskExe;GPU_test_executed=$false} | ConvertTo-Json
    exit 0
}
$taskStart = [System.Diagnostics.ProcessStartInfo]::new()
$taskStart.FileName = $taskExe
$taskStart.Arguments = '--launcher-skip'
$taskStart.WorkingDirectory = Split-Path -Parent $taskExe
$taskStart.UseShellExecute = $false
$taskStart.EnvironmentVariables.Remove('NRB_DLSS_COLOR_STATE_PROOF_SCOPE')
$taskStart.EnvironmentVariables['CYBERPUNK_NR_COST_METER'] = '1'
$taskStart.EnvironmentVariables['NR_DIAG_PERIODIC_FLASH_V2'] = '1'
# The executable's reviewed default selects its source-state proof. No global
# environment is changed, and no temporary source-scope override is needed.
$taskProcess = [System.Diagnostics.Process]::Start($taskStart)
$taskNow = [DateTimeOffset]::UtcNow
$taskOutRoot = 'D:\Codex-NR-Experiments\cyberpunk-opt\gpu-handoff-product-trial-v1-20261002'
$taskLaunchReceipt = Join-Path $taskOutRoot ('launch-' + $taskNow.ToString('yyyyMMddTHHmmssfffffffZ') + '.json')
$taskState = [ordered]@{
    status='launched';pid=$taskProcess.Id;exe=$taskExe;arguments='--launcher-skip'
    started_utc=$taskNow.ToString('o');installation_receipt=$InstallReceipt
    installation_receipt_sha256=(Get-FileHash -LiteralPath $InstallReceipt -Algorithm SHA256).Hash.ToLowerInvariant()
    child_environment=@{CYBERPUNK_NR_COST_METER='1';NR_DIAG_PERIODIC_FLASH_V2='1'}
    child_source_scope_override_present=$false
    GPU_handoff_initially_requested=$false;resource_pool_initially_enabled=$false
    game_or_performance_accepted=$false
}
$taskState | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $taskLaunchReceipt -Encoding utf8
[ordered]@{status='launched';pid=$taskProcess.Id;receipt=$taskLaunchReceipt} | ConvertTo-Json
