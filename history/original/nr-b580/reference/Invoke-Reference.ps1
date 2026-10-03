param(
    [Parameter(Mandatory=$true)][string]$ScriptPath,
    [string]$OutputPath,
    [hashtable]$ScriptParameters = @{},
    [Parameter(Mandatory=$true)][string]$ReferenceHost,
    [Parameter(Mandatory=$true)][string]$ReferenceUser,
    [Parameter(Mandatory=$true)][string]$IdentityFile,
    [Parameter(Mandatory=$true)][string]$KnownHostsFile
)
$ErrorActionPreference = 'Stop'
$key = (Resolve-Path -LiteralPath $IdentityFile).Path
$knownHosts = (Resolve-Path -LiteralPath $KnownHostsFile).Path
$sshOptions = @('-i', $key, '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes', '-o', ('UserKnownHostsFile=' + $knownHosts), '-o', 'ConnectTimeout=8')
if ($ReferenceHost -notmatch '^[A-Za-z0-9][A-Za-z0-9.-]*$' -or $ReferenceUser -notmatch '^[A-Za-z0-9_][A-Za-z0-9_.-]*$') { throw 'Invalid explicit reference host or user' }
$target = $ReferenceUser + '@' + $ReferenceHost
$name = 'codex-nr-' + [guid]::NewGuid().ToString('N') + '.ps1'
$localTemporaryFile = Join-Path $env:TEMP $name
$remoteTemporaryFile = 'E:/Codex-NR-Reference/tmp/' + $name
$scriptText = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $ScriptPath).Path)
$parameterJson = ConvertTo-Json -InputObject $ScriptParameters -Compress -Depth 10
$parameterEncoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($parameterJson))
[IO.File]::WriteAllText($localTemporaryFile, $scriptText, [Text.UTF8Encoding]::new($true))
try {
    # All task-controlled temporary files and child-process working directories
    # on the laptop stay on E:, as requested by the user.
    $bootstrap = '$ErrorActionPreference="Stop"; $ProgressPreference="SilentlyContinue"; if (!(Test-Path -LiteralPath "E:\")) { throw "Reference E drive unavailable" }; foreach ($p in @("E:\Codex-NR-Reference", "E:\Codex-NR-Reference\tmp")) { if (Test-Path -LiteralPath $p) { if ((Get-Item -LiteralPath $p).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reference directory is a reparse point" } } else { [IO.Directory]::CreateDirectory($p) | Out-Null } }'
    $bootstrapEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($bootstrap))
    & ssh.exe @sshOptions -T $target powershell.exe -NoLogo -NoProfile -NonInteractive -EncodedCommand $bootstrapEncoded
    if ($LASTEXITCODE -ne 0) { throw 'Reference E-drive bootstrap failed' }
    & scp.exe @sshOptions $localTemporaryFile ($target + ':' + $remoteTemporaryFile)
    if ($LASTEXITCODE -ne 0) { throw "Reference script upload failed: $LASTEXITCODE" }
    $command = '$ErrorActionPreference="Stop"; $ProgressPreference="SilentlyContinue"; [Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); $env:TEMP="E:\Codex-NR-Reference\tmp"; $env:TMP=$env:TEMP; Set-Location -LiteralPath "E:\Codex-NR-Reference"; $parameters=@{}; $json=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String("' + $parameterEncoded + '")); (ConvertFrom-Json $json).PSObject.Properties | ForEach-Object { $parameters[$_.Name]=$_.Value }; try { & "' + $remoteTemporaryFile + '" @parameters } catch { [Console]::Error.WriteLine($_.Exception.Message); exit 1 } finally { Remove-Item -LiteralPath "' + $remoteTemporaryFile + '" -ErrorAction SilentlyContinue }'
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
    $output = @(& ssh.exe @sshOptions -T -o ServerAliveInterval=15 -o ServerAliveCountMax=3 $target powershell.exe -NoLogo -NoProfile -NonInteractive -EncodedCommand $encoded)
    $remoteExitCode = $LASTEXITCODE
    $text = $output -join "`n"
    if ($OutputPath) {
        $fullOutputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputPath)
        [IO.File]::WriteAllText($fullOutputPath, $text, [Text.UTF8Encoding]::new($false))
    }
    $text
    if ($remoteExitCode -ne 0) { throw "Reference script failed: $remoteExitCode" }
} finally {
    Remove-Item -LiteralPath $localTemporaryFile -ErrorAction SilentlyContinue
}
