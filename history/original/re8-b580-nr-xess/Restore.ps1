param([string]$GameDirectory = '', [string]$BackupDirectory = '')
$ErrorActionPreference = 'Stop'
if (-not $GameDirectory) {
    if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 're8.exe')) { $GameDirectory = $PSScriptRoot }
    else { $GameDirectory = Read-Host 'Paste the Resident Evil Village game folder (containing re8.exe)' }
}
$game = (Resolve-Path -LiteralPath $GameDirectory.Trim().Trim('"')).Path.TrimEnd('\')
if (-not (Test-Path -LiteralPath (Join-Path $game 're8.exe'))) { throw 'Choose the folder containing re8.exe.' }
$gameExe = [IO.Path]::GetFullPath((Join-Path $game 're8.exe'))
$matchingGame = Get-Process re8 -ErrorAction SilentlyContinue | Where-Object {
    try { [IO.Path]::GetFullPath($_.Path) -eq $gameExe } catch { $false }
}
if ($matchingGame) { throw 'Close this Resident Evil Village instance before restoring.' }
$backups = Join-Path $game '.nr-backup'
if (-not $BackupDirectory) {
    $candidate = Get-ChildItem -LiteralPath $backups -Directory | Sort-Object Name -Descending | Select-Object -First 1
    if (-not $candidate) { throw 'No NR backup found.' }
    $BackupDirectory = $candidate.FullName
}
$backup = (Resolve-Path -LiteralPath $BackupDirectory).Path.TrimEnd('\')
if ([IO.Path]::GetDirectoryName($backup) -ne [IO.Path]::GetFullPath($backups).TrimEnd('\')) {
    throw 'Backup must be a direct child of the game .nr-backup directory.'
}
$manifest = Get-Content -LiteralPath (Join-Path $backup 'manifest.json') -Raw | ConvertFrom-Json
if ($manifest.game -ne $game) { throw 'Backup belongs to a different game folder.' }
$current = Join-Path $game 'dxgi.dll'
if (-not (Test-Path -LiteralPath $current)) { throw 'Current dxgi.dll is absent; manual inspection needed.' }
$currentSha = (Get-FileHash -LiteralPath $current -Algorithm SHA256).Hash.ToLowerInvariant()
if ($currentSha -ne $manifest.installedDxgiSha256) { throw 'dxgi.dll changed after NR install; refusing to overwrite it.' }
foreach ($name in @('dxgi.dll', 'nr-game-enable.txt', 'nr-pre-xess-nr-enable.txt')) {
    $saved = Join-Path $backup $name
    $target = Join-Path $game $name
    if (Test-Path -LiteralPath $saved) {
        $expected = $manifest.original.$name
        if (-not $expected -or (Get-FileHash -LiteralPath $saved -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
            throw ('Backup checksum mismatch: ' + $name)
        }
        Copy-Item -LiteralPath $saved -Destination $target -Force
    } elseif ($name -eq 'dxgi.dll') {
        Remove-Item -LiteralPath $target
    } elseif (Test-Path -LiteralPath $target) {
        $marker = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($target)).Trim()
        if ($marker -eq '1') { Remove-Item -LiteralPath $target }
        else { Write-Warning ('Preserving changed marker: ' + $name) }
    }
}
$savedIni = Join-Path $backup 'OptiScaler.ini'
if (Test-Path -LiteralPath $savedIni) {
    $expectedIni = $manifest.original.'OptiScaler.ini'
    if (-not $expectedIni -or (Get-FileHash -LiteralPath $savedIni -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedIni) {
        throw 'OptiScaler.ini backup checksum mismatch.'
    }
    $currentIni = Join-Path $game 'OptiScaler.ini'
    if (-not (Test-Path -LiteralPath $currentIni)) {
        Write-Warning 'OptiScaler.ini is missing; preserving its backup for manual recovery.'
    } elseif (-not $manifest.installedOptiScalerIniSha256 -or
              (Get-FileHash -LiteralPath $currentIni -Algorithm SHA256).Hash.ToLowerInvariant() -ne
              $manifest.installedOptiScalerIniSha256) {
        Write-Warning 'OptiScaler.ini changed after installation; preserving your edits. The original remains in .nr-backup.'
    } else {
        Copy-Item -LiteralPath $savedIni -Destination $currentIni -Force
    }
}
Write-Host 'Previous game files restored. The nr-runtime directory remains available for inspection.'
