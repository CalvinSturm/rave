param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CargoArgs
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir

$ffmpegBin = Join-Path (Split-Path -Parent $repoRoot) "ffmpeg\bin"
$cudaBin = if ($env:CUDA_PATH) { Join-Path ${env:CUDA_PATH} "bin" } else { $null }

$pathParts = @()
if (Test-Path $ffmpegBin) { $pathParts += $ffmpegBin }
if ($cudaBin -and (Test-Path $cudaBin)) { $pathParts += $cudaBin }
if ($pathParts.Count -gt 0) {
    $env:PATH = ($pathParts -join ";") + ";" + $env:PATH
}

Push-Location $repoRoot
try {
    if ($CargoArgs.Count -eq 0) {
        cargo build --workspace
    } else {
        cargo build @CargoArgs
    }
}
finally {
    Pop-Location
}
