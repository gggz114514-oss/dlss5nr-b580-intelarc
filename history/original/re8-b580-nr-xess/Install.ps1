param(
    [string]$GameDirectory = '',
    [string]$AssetDirectory = '',
    [string]$OverlayArchive = '',
    [string]$ModelDll = ''
)
$ErrorActionPreference = 'Stop'
$env:PSModulePath = Join-Path $PSHOME 'Modules'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Get-Sha256([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose(); $stream.Dispose() }
}
function Get-Sha256Bytes([byte[]]$Bytes) {
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($algorithm.ComputeHash($Bytes)).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose() }
}
function Assert-DirectChild([string]$Parent, [string]$Child, [string]$Name) {
    $parentPath = [IO.Path]::GetFullPath($Parent).TrimEnd('\')
    $childPath = [IO.Path]::GetFullPath($Child).TrimEnd('\')
    if ([IO.Path]::GetDirectoryName($childPath) -ne $parentPath -or [IO.Path]::GetFileName($childPath) -ne $Name) {
        throw ('Unsafe target path: ' + $childPath)
    }
}
function Copy-VerifiedPart([object]$Part, [string]$Destination) {
    $uri = [uri]$Part.url
    if ($uri.Scheme -ne 'https' -or $uri.Host -ne 'github.com' -or
        [IO.Path]::GetFileName($Part.name) -ne $Part.name) { throw 'Invalid asset URL or filename' }
    if ($AssetDirectory) {
        Copy-Item -LiteralPath (Join-Path $AssetDirectory $Part.name) -Destination $Destination
    } else {
        Write-Host ('Downloading ' + $Part.name)
        Invoke-WebRequest -Uri $Part.url -OutFile $Destination -UseBasicParsing
    }
    if ((Get-Sha256 $Destination) -ne $Part.sha256) { throw ('Checksum mismatch: ' + $Part.name) }
}

if (-not $GameDirectory) {
    if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 're8.exe')) { $GameDirectory = $PSScriptRoot }
    else { $GameDirectory = Read-Host 'Paste the Resident Evil Village game folder (containing re8.exe)' }
}
$game = (Resolve-Path -LiteralPath $GameDirectory.Trim().Trim('"')).Path.TrimEnd('\')
if (-not (Test-Path -LiteralPath (Join-Path $game 're8.exe'))) { throw 'Choose the Resident Evil Village folder containing re8.exe.' }
$gameExe = [IO.Path]::GetFullPath((Join-Path $game 're8.exe'))
$matchingGame = Get-Process re8 -ErrorAction SilentlyContinue | Where-Object {
    try { [IO.Path]::GetFullPath($_.Path) -eq $gameExe } catch { $false }
}
if ($matchingGame) { throw 'Close this Resident Evil Village instance before installing.' }
foreach ($name in @('dinput8.dll', 'PDPerfPlugin.dll', 'OptiScaler.dll', 'libxess.dll')) {
    if (-not (Test-Path -LiteralPath (Join-Path $game $name))) {
        throw ('Missing ' + $name + '. Install REFramework TemporalUpscaler / UpscalerBasePlugin 1.1.2 / OptiScaler from their original project pages first; see README.')
    }
}
$optiIni = Join-Path $game 'OptiScaler.ini'
if (-not (Test-Path -LiteralPath $optiIni -PathType Leaf)) {
    throw 'Missing OptiScaler.ini. Install the original OptiScaler configuration before NR.'
}
$utf8 = [Text.UTF8Encoding]::new($false, $true)
$optiBytes = [IO.File]::ReadAllBytes($optiIni)
$optiText = $utf8.GetString($optiBytes)
if ((Get-Sha256Bytes $optiBytes) -ne (Get-Sha256Bytes ($utf8.GetBytes($optiText)))) {
    throw 'OptiScaler.ini encoding is not safely editable as UTF-8.'
}
$manualInputPattern = '(?m)^([ \t]*ManualInputPolling[ \t]*=[ \t]*)(auto|false|true)([ \t]*)(?=\r?$)'
$manualInputMatches = [regex]::Matches($optiText, $manualInputPattern)
if ($manualInputMatches.Count -ne 1) {
    throw 'OptiScaler.ini must contain exactly one ManualInputPolling = auto/false/true setting.'
}
$manualInputMatch = $manualInputMatches[0]
$optiPatchedText = $optiText.Substring(0, $manualInputMatch.Index) +
    $manualInputMatch.Groups[1].Value + 'true' + $manualInputMatch.Groups[3].Value +
    $optiText.Substring($manualInputMatch.Index + $manualInputMatch.Length)
$optiPatchedBytes = $utf8.GetBytes($optiPatchedText)
$installedIniSha = Get-Sha256Bytes $optiPatchedBytes
$runtime = Join-Path $game 'nr-runtime'
Assert-DirectChild $game $runtime 'nr-runtime'
if (Test-Path -LiteralPath $runtime) { throw 'nr-runtime already exists. Preserve it; use a clean game folder or the update instructions.' }
$stageName = '.nr-runtime-staging-' + [guid]::NewGuid().ToString('N')
$stage = Join-Path $game $stageName
Assert-DirectChild $game $stage $stageName
New-Item -ItemType Directory -Path $stage | Out-Null

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$runtimeManifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'runtime-assets.json') -Raw | ConvertFrom-Json
foreach ($asset in $runtimeManifest.assets) {
    if ([IO.Path]::GetFileName($asset.name) -ne $asset.name) { throw 'Invalid runtime archive name' }
    $archive = Join-Path $stage $asset.name
    $joined = [IO.File]::Create($archive)
    try {
        $pieces = @($asset)
        if ($asset.parts) { $pieces = @($asset.parts) }
        foreach ($piece in $pieces) {
            $partPath = Join-Path $stage ($piece.name + '.download')
            Copy-VerifiedPart $piece $partPath
            $partStream = [IO.File]::OpenRead($partPath)
            try { $partStream.CopyTo($joined) } finally { $partStream.Dispose() }
            Remove-Item -LiteralPath $partPath
        }
    } finally { $joined.Dispose() }
    if ((Get-Sha256 $archive) -ne $asset.sha256) { throw ('Archive checksum mismatch: ' + $asset.name) }
    [IO.Compression.ZipFile]::ExtractToDirectory($archive, $stage)
    Remove-Item -LiteralPath $archive
}

$python = Join-Path $stage 'python/python.exe'
if ($ModelDll) { & $python -B -X utf8 (Join-Path $stage 'setup_runtime.py') --dll $ModelDll }
else { & $python -B -X utf8 (Join-Path $stage 'setup_runtime.py') }
if ($LASTEXITCODE -ne 0) { throw 'Model preparation failed. See upstream model link in README. Staging files retained.' }

$gameManifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'game-assets.json') -Raw | ConvertFrom-Json
$gameAsset = $gameManifest.overlay
if ($OverlayArchive) {
    $overlay = (Resolve-Path -LiteralPath $OverlayArchive).Path
} else {
    $overlay = Join-Path $stage $gameAsset.name
    Copy-VerifiedPart $gameAsset $overlay
}
if ((Get-Sha256 $overlay) -ne $gameAsset.sha256) { throw 'Game overlay checksum mismatch' }
& $python -B -X utf8 (Join-Path $PSScriptRoot 'tools/install_overlay.py') --archive $overlay --runtime $stage
if ($LASTEXITCODE -ne 0) { throw 'Game overlay install failed' }
if (-not $OverlayArchive) { Remove-Item -LiteralPath $overlay }

$backupParent = Join-Path $game '.nr-backup'
Assert-DirectChild $game $backupParent '.nr-backup'
if (-not (Test-Path -LiteralPath $backupParent)) { New-Item -ItemType Directory -Path $backupParent | Out-Null }
$backupName = Get-Date -Format 'yyyyMMdd-HHmmss'
$backup = Join-Path $backupParent $backupName
Assert-DirectChild $backupParent $backup $backupName
New-Item -ItemType Directory -Path $backup | Out-Null
$prior = @{}
foreach ($name in @('dxgi.dll', 'nr-game-enable.txt', 'nr-pre-xess-nr-enable.txt', 'OptiScaler.ini')) {
    $source = Join-Path $game $name
    if (Test-Path -LiteralPath $source) {
        Copy-Item -LiteralPath $source -Destination (Join-Path $backup $name)
        $prior[$name] = Get-Sha256 $source
    }
}
$installedDll = Get-Sha256 (Join-Path $stage 'native/OptiScaler-nr.dll')
@{ game = $game; original = $prior; installedDxgiSha256 = $installedDll;
   installedOptiScalerIniSha256 = $installedIniSha } |
    ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $backup 'manifest.json') -Encoding UTF8

Move-Item -LiteralPath $stage -Destination $runtime
& (Join-Path $runtime 'python/python.exe') -B -X utf8 (Join-Path $runtime 'relocate.py')
if ($LASTEXITCODE -ne 0) { throw 'Final runtime relocation failed; original game DLL remains backed up.' }
& (Join-Path $runtime 'python/python.exe') -B -X utf8 (Join-Path $PSScriptRoot 'tools/relocate_game_cache.py') --runtime $runtime --prune-incomplete
if ($LASTEXITCODE -ne 0) { throw 'Final game cache relocation failed; original game DLL remains backed up.' }
[IO.File]::WriteAllBytes($optiIni, $optiPatchedBytes)
Copy-Item -LiteralPath (Join-Path $runtime 'native/OptiScaler-nr.dll') -Destination (Join-Path $game 'dxgi.dll') -Force
Set-Content -LiteralPath (Join-Path $game 'nr-game-enable.txt') -Value '1' -Encoding Ascii
Set-Content -LiteralPath (Join-Path $game 'nr-pre-xess-nr-enable.txt') -Value '1' -Encoding Ascii
Write-Host ('Installed. Backup: ' + $backup)
Write-Host 'Launch the game, then open http://127.0.0.1:8765/ for NR controls.'
