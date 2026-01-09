# AMD-AI-Toolkit
One-click AMD GPU AI toolkit for Windows. Native HIP/ROCm support for RX 5000/6000/7000/9000 series. Auto-detects, auto-configures, runs any model size. Includes quantization tools.

# AMD AI Toolkit - HIP GPU Acceleration for Windows

Run LLMs on your AMD GPU with full hardware acceleration. No more CPU-only inference!

## Why This Exists

AMD left millions of consumer GPU owners behind with no 
Windows AI support. This toolkit fixes that.

**Tested on AMD Radeon RX 6700 XT (gfx1031) + Ryzen 9 7900X**

## What This Does

This toolkit lets you run AI models (like Llama, Mistral, Qwen) on your AMD graphics card instead of your CPU. Result: **3-5x faster inference**.

## Supported GPUs

| GPU | Architecture | GFX Version |
|-----|-------------|-------------|
| RX 6700 XT | RDNA2 | gfx1031 |
| RX 6800/6900 | RDNA2 | gfx1030 |
| RX 7900 XT/XTX | RDNA3 | gfx1100 |
| Other RDNA2/3 | Check your GPU | See AMD docs |

## Prerequisites

You need these installed BEFORE starting:

### 1. Visual Studio Build Tools
Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Run installer
- Select "Desktop development with C++"
- Install

### 2. CMake
Download: https://cmake.org/download/
- Get "Windows x64 Installer"
- During install, select "Add CMake to PATH"

### 3. Ninja Build System
```powershell
# Option A: Using winget
winget install Ninja-build.Ninja

# Option B: Using pip
pip install ninja
```

### 4. AMD HIP SDK 5.7
Download: https://www.amd.com/en/developer/resources/rocm-hub.html
- Look for "HIP SDK for Windows"
- Download version 5.7.x
- Install to default location: `C:\Program Files\AMD\ROCm\5.7`
- **Restart your PC after installation**

### 5. Python 3.10+
Download: https://www.python.org/downloads/
- During install, check "Add Python to PATH"

## Quick Start (Step by Step)

### Step 1: Open PowerShell
- Press `Win + X`
- Click "Windows Terminal" or "PowerShell"
- Navigate to this folder:
```powershell
cd "C:\path\to\AMD-AI-Toolkit"
```

### Step 2: Fix rocBLAS Library
```powershell
.\fix_rocblas_local.ps1
```
This copies required GPU kernel files. Only need to do this once.

### Step 3: Build llama-cpp-python with HIP
```powershell
.\rebuild_hip_gfx1031.ps1
```
This takes 5-15 minutes. You'll see lots of compiler output - that's normal.

**If you get errors:** Make sure Visual Studio Build Tools is installed and restart your terminal.

### Step 4: Download a Test Model
You need a GGUF model file. Download a small one to test:

```powershell
# Create models folder
mkdir models

# Download TinyLlama (600MB) - good for testing
Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" -OutFile "models\tinyllama.gguf"
```

Or manually download any GGUF model from HuggingFace and put it in the `models` folder.

### Step 5: Test It!
```powershell
.\run_hip_model.ps1 hip_example.py
```

You should see:
- "AMD HIP Environment Configured"
- Model loading with "ROCm" device
- Generated text output
- Speed in tokens/sec

**If it works, your AMD GPU is now accelerating AI inference!**

## Performance

| Model | Size | GPU Speed | CPU Speed |
|-------|------|-----------|-----------|
| TinyLlama 1.1B | 600MB | ~3.0 tok/s | ~1.0 tok/s |
| Qwen 7B | 4.4GB | ~0.6 tok/s | ~0.2 tok/s |

## Running Your Own Models

### Option A: Any GGUF Model
1. Download a `.gguf` file from HuggingFace
2. Put it in the `models` folder
3. Run:
```powershell
.\run_hip_model.ps1 hip_autoconfig.py models\your-model.gguf
```

### Option B: Extract from Ollama
If you have models in Ollama, you can extract them:

```powershell
# 1. Find your model's info (replace YOUR_MODEL with model name like "llama2" or "mistral")
Get-Content "$env:USERPROFILE\.ollama\models\manifests\registry.ollama.ai\library\YOUR_MODEL\latest"

# 2. Look for the "digest" of the largest layer (the model weights)
#    It looks like: sha256:abc123def456...

# 3. Copy that file (replace THE_DIGEST with actual digest):
Copy-Item "$env:USERPROFILE\.ollama\models\blobs\sha256-THE_DIGEST" "models\my-model.gguf"

# 4. Run it:
.\run_hip_model.ps1 hip_autoconfig.py models\my-model.gguf
```

## Included Tools

| File | What It Does |
|------|--------------|
| `run_hip_model.ps1` | Sets up GPU environment and runs Python scripts |
| `rebuild_hip_gfx1031.ps1` | Builds llama-cpp-python with AMD GPU support |
| `fix_rocblas_local.ps1` | Fixes AMD library paths (run once) |
| `hip_example.py` | Basic test - loads model and generates text |
| `hip_autoconfig.py` | Smart loader - auto-detects best GPU settings |
| `xena.py` | Security-focused AI assistant example |
| `tools\llama-quantize.exe` | Compress models to smaller sizes |

## Troubleshooting

### "No GGUF model found"
Download a model and put it in the `models` folder. See Step 4 above.

### Build fails with compiler errors
- Make sure Visual Studio Build Tools is installed
- Restart your terminal after installing
- Run from a fresh PowerShell window

### Model loads but hangs during generation
The `run_hip_model.ps1` script already handles this, but if running manually:
```python
import os
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["GGML_CUDA_FORCE_MMQ"] = "1"
```

### "rocBLAS TensileLibrary.dat not found"
Run `.\fix_rocblas_local.ps1` again.

### Very slow or using CPU instead of GPU
- Check that AMD HIP SDK 5.7 is installed
- Restart PC after HIP SDK installation
- Rebuild with `.\rebuild_hip_gfx1031.ps1`

### Different GPU (not RX 6700 XT)
Edit `run_hip_model.ps1` and `rebuild_hip_gfx1031.ps1`:
- RX 6800/6900: Change `gfx1031` to `gfx1030`
- RX 7900: Change `gfx1031` to `gfx1100` and `10.3.0` to `11.0.0`

## Model Quantization

Make models smaller (uses less VRAM, slightly lower quality):

```powershell
.\tools\llama-quantize.exe models\big-model.gguf models\smaller-model.gguf q4_k_m
```

Quantization types (from largest to smallest):
- `f16` - Full precision (largest)
- `q8_0` - 8-bit (good quality)
- `q5_k_m` - 5-bit (balanced)
- `q4_k_m` - 4-bit (recommended)
- `q4_0` - 4-bit (smallest)

**Note:** You can only go DOWN in size, not up. Can't turn a q4 model into q8.

## Credits

- brknsoul/ROCmLibs - Tensile libraries
- likelovewant - Multi-arch support  
- llama.cpp community

## License

MIT License © 2026 Torsova LLC

## Contact
  Torsova LLC - R@D

- GitHub Issues: Bug reports & features
- Email: 413mrm@gmail.com

---

*Don't go around the mountain. Don't go over it. Go through it.*
```

---
