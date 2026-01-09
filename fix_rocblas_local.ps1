<#
.SYNOPSIS
    Fix rocBLAS by copying Tensile libraries to local directory
.DESCRIPTION
    Copies rocBLAS Tensile kernel libraries from AMD HIP SDK to local folder
    to avoid path issues and ensure proper initialization.
#>

$sourcePath = "C:\Program Files\AMD\ROCm\5.7\bin\rocblas\library"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$localPath = Join-Path $scriptDir "rocblas_library"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  AMD AI Toolkit - rocBLAS Library Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $sourcePath)) {
    Write-Host "ERROR: AMD HIP SDK not found at expected location" -ForegroundColor Red
    Write-Host "Expected: $sourcePath" -ForegroundColor Yellow
    Write-Host "Install AMD HIP SDK 5.7 first" -ForegroundColor Yellow
    exit 1
}

Write-Host "Source: $sourcePath" -ForegroundColor White
Write-Host "Target: $localPath" -ForegroundColor White
Write-Host ""

# Create local directory
if (-not (Test-Path $localPath)) {
    New-Item -ItemType Directory -Path $localPath | Out-Null
}

# Copy all Tensile files
Write-Host "Copying Tensile libraries (this may take a moment)..."
Copy-Item "$sourcePath\*" $localPath -Recurse -Force

# Create TensileLibrary.dat from lazy loader
$lazyFile = Join-Path $localPath "TensileLibrary_lazy_gfx1030.dat"
$targetFile = Join-Path $localPath "TensileLibrary.dat"

if (Test-Path $lazyFile) {
    Copy-Item $lazyFile $targetFile -Force
    Write-Host "Created TensileLibrary.dat" -ForegroundColor Green
}

Write-Host ""
Write-Host "SUCCESS: Local rocBLAS library ready" -ForegroundColor Green
Write-Host ""
Write-Host "The run_hip_model.ps1 script will automatically use this library." -ForegroundColor White
