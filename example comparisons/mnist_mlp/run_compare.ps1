# Compare wall-clock MNIST MLP training: PyTorch vs RusTorch (naive + fast).
#
# Order: rust fast, rust naive, python (rust first so a cold/slow Python start
# does not inflate "speedup"). Optional -Trials N reports median walls.
param(
    [int]$Epochs = 25,
    [int]$BatchSize = 128,
    [int]$Seed = 42,
    [int]$Trials = 1
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $Root "..\..")
$DataDir = Join-Path $Root "data"
$PyTrain = Join-Path $Root "python\train_mnist.py"
$RustToml = Join-Path $Root "rust\Cargo.toml"
$Download = Join-Path $Root "download_mnist.py"

$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
if (-not $env:CARGO_TARGET_DIR) {
    $env:CARGO_TARGET_DIR = Join-Path $RepoRoot "target\example_comparisons"
}
$RustBin = Join-Path $env:CARGO_TARGET_DIR "release\mnist_mlp_rustorch.exe"

function Invoke-Capture {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList
    )
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $lines = & $FilePath @ArgumentList 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    foreach ($line in $lines) {
        Write-Host $line
    }
    return @{ ExitCode = $code; Lines = $lines }
}

function Parse-Wall([string]$line) {
    if ($line -match "wall_sec=([0-9.]+)") { return [double]$Matches[1] }
    return $null
}

function Get-Median([double[]]$xs) {
    $s = $xs | Sort-Object
    $n = $s.Count
    if ($n -eq 0) { return $null }
    if ($n % 2 -eq 1) { return $s[[int]($n / 2)] }
    return ($s[$n / 2 - 1] + $s[$n / 2]) / 2.0
}

Write-Host "== ensure MNIST data =="
$dl = Invoke-Capture -FilePath "python" -ArgumentList @($Download)
if ($dl.ExitCode -ne 0) { exit $dl.ExitCode }

Write-Host "== build rust (release) =="
$build = Invoke-Capture -FilePath "cargo" -ArgumentList @(
    "build", "--release", "--manifest-path", $RustToml
)
if ($build.ExitCode -ne 0) { exit $build.ExitCode }
if (-not (Test-Path $RustBin)) {
    Write-Error "missing binary: $RustBin"
    exit 1
}

$common = @(
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--seed", "$Seed",
    "--data-dir", $DataDir
)

$pyWalls = @()
$naiveWalls = @()
$fastWalls = @()
$pyResult = $null
$naiveResult = $null
$fastResult = $null

for ($t = 1; $t -le $Trials; $t++) {
    if ($Trials -gt 1) {
        Write-Host ""
        Write-Host "==== trial $t / $Trials ===="
    }

    # Warm / measure rust before python so a cold MKL/torch start does not dominate.
    Write-Host "== rust fast (fused train helpers) =="
    $rsFast = Invoke-Capture -FilePath $RustBin -ArgumentList ($common + @("--mode", "fast"))
    if ($rsFast.ExitCode -ne 0) { exit $rsFast.ExitCode }
    $fastResult = ($rsFast.Lines | Where-Object { $_ -match "^RESULT " } | Select-Object -Last 1)
    $fastWalls += ,(Parse-Wall "$fastResult")

    Write-Host "== rust naive (1:1 PyTorch-shaped API) =="
    $rsNaive = Invoke-Capture -FilePath $RustBin -ArgumentList ($common + @("--mode", "naive"))
    if ($rsNaive.ExitCode -ne 0) { exit $rsNaive.ExitCode }
    $naiveResult = ($rsNaive.Lines | Where-Object { $_ -match "^RESULT " } | Select-Object -Last 1)
    $naiveWalls += ,(Parse-Wall "$naiveResult")

    Write-Host "== python (PyTorch) =="
    $py = Invoke-Capture -FilePath "python" -ArgumentList (@($PyTrain) + $common)
    if ($py.ExitCode -ne 0) { exit $py.ExitCode }
    $pyResult = ($py.Lines | Where-Object { $_ -match "^RESULT " } | Select-Object -Last 1)
    $pyWalls += ,(Parse-Wall "$pyResult")
}

$pyWall = Get-Median $pyWalls
$naiveWall = Get-Median $naiveWalls
$fastWall = Get-Median $fastWalls

Write-Host ""
Write-Host "======== COMPARISON ========"
if ($Trials -gt 1) {
    Write-Host ("trials={0} (reporting median wall_sec)" -f $Trials)
    Write-Host ("python walls:      {0}" -f (($pyWalls | ForEach-Object { "{0:N4}" -f $_ }) -join ", "))
    Write-Host ("rust naive walls:  {0}" -f (($naiveWalls | ForEach-Object { "{0:N4}" -f $_ }) -join ", "))
    Write-Host ("rust fast walls:   {0}" -f (($fastWalls | ForEach-Object { "{0:N4}" -f $_ }) -join ", "))
}
Write-Host "python:      $pyResult"
Write-Host "rust naive:  $naiveResult"
Write-Host "rust fast:   $fastResult"
if ($null -ne $pyWall -and $null -ne $naiveWall -and $naiveWall -gt 0) {
    Write-Host ("speedup naive (py / rust_naive) = {0:N3}x" -f ($pyWall / $naiveWall))
}
if ($null -ne $pyWall -and $null -ne $fastWall -and $fastWall -gt 0) {
    Write-Host ("speedup fast  (py / rust_fast)  = {0:N3}x" -f ($pyWall / $fastWall))
}
if ($null -ne $naiveWall -and $null -ne $fastWall -and $fastWall -gt 0) {
    Write-Host ("fast vs naive (naive / fast)    = {0:N3}x" -f ($naiveWall / $fastWall))
}
Write-Host "============================"
Write-Host "Note: single-run speedups swing with PyTorch cold start (your earlier"
Write-Host "26s vs 19s python walls). Prefer -Trials 3 for a stable median."
