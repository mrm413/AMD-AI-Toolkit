<#
.SYNOPSIS
    Run Python with AMD HIP GPU environment configured
.DESCRIPTION
    Sets up the AMD HIP environment for llama-cpp-python with HIP support.
    CRITICAL: Sets HIP_VISIBLE_DEVICES=0 to use only discrete GPU.
.EXAMPLE
    .\run_hip_model.ps1 hip_example.py
    .\run_hip_model.ps1 xena.py "your security question"
#>

param(
    [Parameter(Position=0, ValueFromRemainingArguments=$true)]
    [string[]]$PythonArgs
)

# Get script directory for relative paths
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# GPU configuration
$TargetGPU = "gfx1031"
$GfxVersion = "10.3.0"

# Configure HIP environment
$env:HIP_PATH = "C:\Program Files\AMD\ROCm\5.7"
$env:ROCM_PATH = $env:HIP_PATH
$env:HSA_OVERRIDE_GFX_VERSION = $GfxVersion
$env:PATH = "$env:HIP_PATH\bin;" + $env:PATH

# CRITICAL: Only use discrete GPU (device 0)
$env:HIP_VISIBLE_DEVICES = "0"

# Point to local rocBLAS library
$env:ROCBLAS_TENSILE_LIBPATH = Join-Path $scriptDir "rocblas_library"

# Force MMQ for larger models
$env:GGML_CUDA_FORCE_MMQ = "1"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  AMD HIP Environment Configured" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  GPU Target: $TargetGPU" -ForegroundColor White
Write-Host "  GFX Version: $GfxVersion" -ForegroundColor White
Write-Host "  HIP_VISIBLE_DEVICES: 0 (discrete GPU only)" -ForegroundColor Yellow
Write-Host "  FORCE_MMQ: enabled (for 7B+ models)" -ForegroundColor White
Write-Host ""

if ($PythonArgs.Count -eq 0) {
    Write-Host "Usage: .\run_hip_model.ps1 <script.py> [args...]" -ForegroundColor Yellow
    python
} else {
    python @PythonArgs
}
