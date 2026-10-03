# Run on the reference PC through Invoke-Reference.ps1. Does not load any DLL.
$ErrorActionPreference = 'Stop'
$root = 'E:\Codex-NR-Reference'
$runtimeDirectory = Join-Path $root 'video2dlssnr-55a4ceb5\out'
# SF-v2: cross-checked against Wan2GP's DLSS5 manifest and the research
# repository taowen/dlss5-as-inpainting's Git LFS object identifier.
$expectedNrHash = '6EB209E764F39872625DEBD6ABAF45E2BB6322F6F270F781F70C059AE30B3927'
$driverRows = @(& nvidia-smi.exe --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits)
if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi failed' }
$gpus = @($driverRows | ForEach-Object {
    $fields = $_ -split ',\s*'
    [ordered]@{ name=$fields[0].Trim(); driver=$fields[1].Trim(); memoryMiB=[int]$fields[2] }
})
$referenceGpu = @($gpus | Where-Object { $_.name -match 'RTX 4060' })
$missing = [Collections.Generic.List[string]]::new()
if ($referenceGpu.Count -ne 1) {
    $missing.Add('Expected exactly one RTX 4060 reference GPU')
} elseif ([version]$referenceGpu[0].driver -lt [version]'616.56') {
    $missing.Add('NVIDIA driver 616.56 or newer')
}
$artifacts = @('video2dlssnr.exe', 'nvngx.dll_dlssnr.dll', 'nvngx_dlssnr.dll') | ForEach-Object {
    $path = Join-Path $runtimeDirectory $_
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) {
        $missing.Add($_)
        [ordered]@{ name=$_; exists=$false }
    } else {
        $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
        if ($_ -eq 'nvngx_dlssnr.dll' -and $hash -ne $expectedNrHash) {
            $missing.Add('NR runtime differs from the pinned RTX 40 reference; investigate its origin and version')
        }
        [ordered]@{ name=$_; exists=$true; bytes=(Get-Item -LiteralPath $path).Length; sha256=$hash }
    }
}
[ordered]@{
    computer=$env:COMPUTERNAME
    gpus=$gpus
    runtimeDirectory=$runtimeDirectory
    expectedNrSha256=$expectedNrHash
    artifacts=@($artifacts)
    missing=@($missing.ToArray())
    prerequisitesPresent=($missing.Count -eq 0)
    nrInferenceExecutedByThisCheck=$false
} | ConvertTo-Json -Depth 5
