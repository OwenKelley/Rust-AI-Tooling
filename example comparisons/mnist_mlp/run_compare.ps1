# Compare wall-clock MNIST MLP training: PyTorch vs rtorch.
param(
    [int]$Epochs = 25,
    [int]$BatchSize = 128,
    [int]$Seed = 42
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
$RustBin = Join-Path $env:CARGO_TARGET_DIR "release\mnist_mlp_rtorch.exe"

function Invoke-Capture {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList
    )
    # Do not merge stderr with 2>&1 under Stop: native stderr becomes terminating ErrorRecords.
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

Write-Host "== python (PyTorch) =="
$py = Invoke-Capture -FilePath "python" -ArgumentList (@($PyTrain) + $common)
if ($py.ExitCode -ne 0) { exit $py.ExitCode }
$pyResult = ($py.Lines | Where-Object { $_ -match "^RESULT " } | Select-Object -Last 1)

Write-Host "== rust (rtorch) =="
$rs = Invoke-Capture -FilePath $RustBin -ArgumentList $common
if ($rs.ExitCode -ne 0) { exit $rs.ExitCode }
$rsResult = ($rs.Lines | Where-Object { $_ -match "^RESULT " } | Select-Object -Last 1)

function Parse-Wall([string]$line) {
    if ($line -match "wall_sec=([0-9.]+)") { return [double]$Matches[1] }
    return $null
}

$pyWall = Parse-Wall "$pyResult"
$rsWall = Parse-Wall "$rsResult"

Write-Host ""
Write-Host "======== COMPARISON ========"
Write-Host "python: $pyResult"
Write-Host "rust:   $rsResult"
if ($null -ne $pyWall -and $null -ne $rsWall -and $rsWall -gt 0) {
    $speedup = $pyWall / $rsWall
    Write-Host ("speedup (py_wall / rs_wall) = {0:N3}x" -f $speedup)
}
Write-Host "============================"
