<# Stage or restore the verified C128 direct-write module set; preview by default. #>
param([switch]$Apply, [switch]$Rollback)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($Apply -and $Rollback) { throw 'Choose -Apply or -Rollback, not both' }

$source = [System.IO.Path]::GetFullPath('E:\ComfyUI-aki-v3-IntelArc_20260722\re8-b580-nr-xess\game').TrimEnd('\')
$runtime = [System.IO.Path]::GetFullPath('G:\SteamLibrary\steamapps\common\Resident Evil Village BIOHAZARD VILLAGE\nr-runtime').TrimEnd('\')
$target = Join-Path $runtime 'game'
$backup = [System.IO.Path]::GetFullPath('D:\Codex-NR-Experiments\nr-b580\game-c128-direct-backup-20260927').TrimEnd('\')
$combo = 'three_structure_combo_v1.py'
$oldHash = 'EE1FC6A3625BC933578D85B797FE2E7893A3AC0002CF56B46DA216C389939E6A'
$newFiles = @('c128_qkv_direct_pack_one_v1.py', 'c128_qkv_direct_pack_all_v1.py')
$names = @($combo) + $newFiles

if (-not (Test-Path -LiteralPath $source -PathType Container) -or
    -not (Test-Path -LiteralPath $target -PathType Container) -or
    -not (Test-Path -LiteralPath (Join-Path $runtime 'data\fast-cache') -PathType Container)) {
    throw 'Expected source, game target, or installed cache is missing'
}
if (Get-Process -Name re8 -ErrorAction SilentlyContinue) {
    throw 'Close Resident Evil Village before changing its NR runtime'
}

if ($Rollback) {
    $manifestPath = Join-Path $backup 'manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw 'Backup manifest missing' }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $saved = Join-Path $backup $combo
    $current = Join-Path $target $combo
    if (-not (Test-Path -LiteralPath $saved -PathType Leaf) -or
        (Get-FileHash -LiteralPath $saved -Algorithm SHA256).Hash -ne $oldHash) {
        throw 'Saved combo differs from original'
    }
    $currentHash = (Get-FileHash -LiteralPath $current -Algorithm SHA256).Hash
    if ($currentHash -ne $oldHash -and $currentHash -ne $manifest.new_hashes.PSObject.Properties[$combo].Value) {
        throw 'Installed combo changed since staging'
    }
    Copy-Item -LiteralPath $saved -Destination $current -Force
    if ((Get-FileHash -LiteralPath $current -Algorithm SHA256).Hash -ne $oldHash) {
        throw 'Rollback hash mismatch'
    }
    Write-Host 'Previous game combo restored; unused candidate modules remain on disk.'
    return
}

foreach ($name in $names) {
    if (-not (Test-Path -LiteralPath (Join-Path $source $name) -PathType Leaf)) {
        throw "Candidate source missing: $name"
    }
}
$installedCombo = Join-Path $target $combo
if ((Get-FileHash -LiteralPath $installedCombo -Algorithm SHA256).Hash -ne $oldHash) {
    throw 'Installed combo differs from verified starting version'
}
foreach ($name in $newFiles) {
    if (Test-Path -LiteralPath (Join-Path $target $name)) {
        throw "New candidate already exists in game directory: $name"
    }
}
if (Test-Path -LiteralPath $backup) { throw 'Backup directory already exists' }

[pscustomobject]@{
    Action = $(if ($Apply) { 'STAGE' } else { 'PREVIEW' })
    Game = $target
    ExistingFilesToBackUp = 1
    NewFiles = $newFiles.Count
    Backup = $backup
    CacheChanged = $false
}
if (-not $Apply) { return }

New-Item -ItemType Directory -Path $backup -ErrorAction Stop | Out-Null
Copy-Item -LiteralPath $installedCombo -Destination (Join-Path $backup $combo) -ErrorAction Stop
if ((Get-FileHash -LiteralPath (Join-Path $backup $combo) -Algorithm SHA256).Hash -ne $oldHash) {
    throw 'Backup hash mismatch'
}
$newHashes = @{}
foreach ($name in $names) {
    $newHashes[$name] = (Get-FileHash -LiteralPath (Join-Path $source $name) -Algorithm SHA256).Hash
}
@{ staged_at = (Get-Date).ToString('o'); runtime = $runtime;
   original_hash = $oldHash; new_hashes = $newHashes } |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $backup 'manifest.json') -Encoding UTF8
foreach ($name in $names) {
    $to = Join-Path $target $name
    Copy-Item -LiteralPath (Join-Path $source $name) -Destination $to -Force -ErrorAction Stop
    if ((Get-FileHash -LiteralPath $to -Algorithm SHA256).Hash -ne $newHashes[$name]) {
        throw "Stage hash mismatch: $name"
    }
}
Write-Host 'C128 source set staged with verified combo backup; precompile installed cache before launching game.'
