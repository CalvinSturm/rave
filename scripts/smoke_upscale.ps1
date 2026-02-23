param(
    [string]$InputPath = "",
    [string]$ModelPath = "tests/assets/models/resize2x_rgb.onnx"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$thirdPartyRoot = Split-Path -Parent $repoRoot

$ffmpegBin = Join-Path $thirdPartyRoot "ffmpeg\bin"
$cudaBin = if ($env:CUDA_PATH) { Join-Path ${env:CUDA_PATH} "bin" } else { $null }
$pathParts = @()
if (Test-Path $ffmpegBin) { $pathParts += $ffmpegBin }
if ($cudaBin -and (Test-Path $cudaBin)) { $pathParts += $cudaBin }
if ($pathParts.Count -gt 0) {
    $env:PATH = ($pathParts -join ";") + ";" + $env:PATH
}

$modelAbs = if ([System.IO.Path]::IsPathRooted($ModelPath)) {
    $ModelPath
} else {
    Join-Path $repoRoot $ModelPath
}
if (-not (Test-Path $modelAbs)) {
    throw "Model not found: $modelAbs"
}

$inputAbs = $InputPath
if (-not $inputAbs) {
    $smokeDir = Join-Path $repoRoot "target\smoke"
    New-Item -ItemType Directory -Force -Path $smokeDir | Out-Null
    $inputAbs = Join-Path $smokeDir "smoke_input.mp4"
    $ffmpegExe = Join-Path $ffmpegBin "ffmpeg.exe"
    if (-not (Test-Path $ffmpegExe)) {
        $ffmpegExe = "ffmpeg"
    }

    & $ffmpegExe -hide_banner -loglevel error -y -f lavfi -i "testsrc=size=320x240:rate=24" -t 1 -pix_fmt yuv420p $inputAbs 2> $null | Out-Null
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $inputAbs)) {
        throw "Failed to generate smoke input clip via ffmpeg."
    }
}

$outputAbs = Join-Path $repoRoot "target\smoke\smoke_out.mp4"
$raveExe = Join-Path $repoRoot "target\release\rave.exe"
if (-not (Test-Path $raveExe)) {
    throw "Missing release binary: $raveExe (run .\\scripts\\build.ps1 -- -p rave-cli --bin rave --release)"
}

$stderrPath = Join-Path $repoRoot "target\smoke\smoke_stderr.log"
$jsonLines = & $raveExe upscale --input $inputAbs --output $outputAbs --model $modelAbs --dry-run --json --progress off 2> $stderrPath
if ($LASTEXITCODE -ne 0) {
    $stderrTail = ""
    if (Test-Path $stderrPath) {
        $stderrTail = (Get-Content $stderrPath -Raw)
    }
    throw "rave upscale smoke failed (exit=$LASTEXITCODE). stderr=$stderrTail"
}

$jsonText = (($jsonLines | ForEach-Object { "$_" }) -join "`n").Trim()
if (-not $jsonText) {
    throw "Smoke command produced empty stdout."
}

try {
    $obj = $jsonText | ConvertFrom-Json
} catch {
    throw "Smoke command stdout was not valid JSON: $jsonText"
}

if (-not $obj.ok) {
    throw "Smoke JSON returned ok=false: $jsonText"
}

Write-Host "RAVE upscale smoke ok=true input=$inputAbs model=$modelAbs"
