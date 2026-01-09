# AMD AI Toolkit - HIP Build targeting gfx1031
# For RX 6700 XT (gfx1031) - native target with HSA_OVERRIDE for rocBLAS

Write-Host "============================================================"
Write-Host "  AMD AI Toolkit - HIP Build (gfx1031 native)"
Write-Host "  For RX 6700 XT with HSA_OVERRIDE for rocBLAS compatibility"
Write-Host "============================================================"
Write-Host ""

# Set HIP environment
$env:HIP_PATH = "C:\Program Files\AMD\ROCm\5.7"
$env:ROCM_PATH = $env:HIP_PATH
$env:HIP_PLATFORM = "amd"
$env:HSA_OVERRIDE_GFX_VERSION = "10.3.0"

# Add HIP to PATH
$env:PATH = "$env:HIP_PATH\bin;$env:PATH"

# Set compilers to HIP SDK clang
$env:CC = "$env:HIP_PATH\bin\clang.exe"
$env:CXX = "$env:HIP_PATH\bin\clang++.exe"

# CMAKE_ARGS - targeting gfx1031 natively
$env:CMAKE_ARGS = "-G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1031 -DGPU_TARGETS=gfx1031 -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DGGML_CUDA_FORCE_MMQ=ON -DGGML_CUDA_FA=OFF -DCMAKE_CXX_FLAGS=`"-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -Wl,/FORCE:MULTIPLE`" -DCMAKE_C_FLAGS=`"-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`""

Write-Host "Environment:"
Write-Host "  HIP_PATH: $env:HIP_PATH"
Write-Host "  CC: $env:CC"
Write-Host "  CXX: $env:CXX"
Write-Host "  TARGET: gfx1031 (native)"
Write-Host "  HSA_OVERRIDE: 10.3.0 (for rocBLAS gfx1030 kernels)"
Write-Host ""

# Verify clang exists
if (-not (Test-Path $env:CC)) {
    Write-Host "ERROR: clang.exe not found at $env:CC"
    exit 1
}

Write-Host "Clang version:"
& $env:CXX --version | Select-Object -First 1
Write-Host ""

# Clean old build
Write-Host "Cleaning old build..."
pip uninstall llama-cpp-python -y 2>$null
Remove-Item -Recurse -Force "F:\Desktop\Projects\llama_cpp_python_test\build" -ErrorAction SilentlyContinue

# Build
Write-Host ""
Write-Host "Building llama-cpp-python with HIP (gfx1031)..."
Write-Host "This may take 10-20 minutes..."
Write-Host ""

Set-Location "F:\Desktop\Projects\llama_cpp_python_test"
pip install . --no-cache-dir --verbose

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Build failed!"
    exit 1
}

# Verify
Write-Host ""
Write-Host "Verifying GPU offload..."
$env:HSA_OVERRIDE_GFX_VERSION = "10.3.0"
python -c "from llama_cpp import llama_cpp; print('GPU offload:', llama_cpp.llama_supports_gpu_offload())"

Write-Host ""
Write-Host "Build complete! Now test with run_with_hip.ps1"
