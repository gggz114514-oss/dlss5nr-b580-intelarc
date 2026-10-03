# Read-only readiness for future bounded native small-dimension captures.
# Does not launch NR or alter the laptop's configuration.
$ErrorActionPreference = 'Stop'
$root = 'E:\Codex-NR-Reference'
$names = @('video2dlssnr-55a4ceb5', 'video2dlssnr-capture-v1', 'video2dlssnr-trace-v3', 'video2dlssnr-aux-dimensions-v1')
$artifacts = foreach ($name in $names) {
    $bin = Join-Path $root ($name + '\out')
    foreach ($leaf in @('video2dlssnr.exe', 'nvngx_dlssnr.dll', 'nr_nvapi_trace.dll')) {
        $path = Join-Path $bin $leaf
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            [ordered]@{ path=$path; bytes=(Get-Item -LiteralPath $path).Length; sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash }
        }
    }
}
$gpu = @(& nvidia-smi.exe --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader,nounits)
$gpuExit = $LASTEXITCODE
[ordered]@{
    utc=[DateTime]::UtcNow.ToString('o')
    rootPresent=(Test-Path -LiteralPath $root -PathType Container)
    eFreeBytes=(Get-PSDrive -Name E).Free
    gpuQueryExit=$gpuExit
    gpu=$gpu
    artifacts=@($artifacts)
    nrProcesses=@(Get-Process -Name video2dlssnr -ErrorAction SilentlyContinue | Select-Object Id,SessionId)
    nrInferenceExecuted=$false
} | ConvertTo-Json -Depth 5
