#!/usr/bin/env python3
"""
AMD AI Toolkit - Smart Auto-Configuration
Automatically detects optimal GPU settings for any model size.
Works with custom/injected models that may have different sizes.

Usage:
    .\run_hip_model.ps1 hip_autoconfig.py models\your-model.gguf
    .\run_hip_model.ps1 hip_autoconfig.py models\your-model.gguf "your prompt"
"""

import os
import sys
import subprocess
import re
import time

# Force line buffering
sys.stdout.reconfigure(line_buffering=True)


def get_gpu_vram_mb():
    """Get GPU VRAM in MB. Returns 12288 (12GB) as fallback for RX 6700 XT."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            match = re.search(r"Total Memory.*?(\d+)", result.stdout)
            if match:
                return int(match.group(1)) // (1024 * 1024)
    except Exception:
        pass
    return 12288  # Default for RX 6700 XT


def estimate_model_vram(model_path):
    """Estimate VRAM needed based on model file size."""
    try:
        file_size_gb = os.path.getsize(model_path) / (1024**3)
        # Q4 models use ~1.5x file size in VRAM + KV cache overhead
        vram_needed_mb = int(file_size_gb * 1.5 * 1024) + 512
        return vram_needed_mb, file_size_gb
    except Exception:
        return None, None


def get_optimal_config(model_path, gpu_vram_mb=12288):
    """Calculate optimal configuration for a model."""
    vram_needed, file_size_gb = estimate_model_vram(model_path)

    if vram_needed is None:
        return {
            "n_gpu_layers": 0,
            "n_ctx": 512,
            "n_batch": 512,
            "strategy": "FALLBACK",
            "reason": "Could not read model file"
        }

    available_vram = gpu_vram_mb - 1024  # Reserve 1GB for system

    config = {
        "model_size_gb": file_size_gb,
        "vram_estimate_mb": vram_needed,
        "available_vram_mb": available_vram,
    }

    if vram_needed < available_vram * 0.7:
        # Model fits easily
        config.update({
            "n_gpu_layers": -1,
            "n_ctx": 4096,
            "n_batch": 512,
            "strategy": "FULL_GPU",
            "reason": f"Model ({file_size_gb:.1f}GB) fits in VRAM"
        })
    elif vram_needed < available_vram:
        # Model fits with conservative settings
        config.update({
            "n_gpu_layers": -1,
            "n_ctx": 2048,
            "n_batch": 256,
            "strategy": "FULL_GPU_CONSERVATIVE",
            "reason": f"Model ({file_size_gb:.1f}GB) fits with conservative settings"
        })
    else:
        # Partial offload
        vram_ratio = available_vram / vram_needed
        estimated_layers = max(1, int(32 * vram_ratio * 0.8))
        config.update({
            "n_gpu_layers": estimated_layers,
            "n_ctx": 1024,
            "n_batch": 128,
            "strategy": "PARTIAL_GPU",
            "reason": f"Model ({file_size_gb:.1f}GB) exceeds VRAM, using {estimated_layers} layers"
        })

    return config


class SmartLlamaLoader:
    """Auto-configuring model loader."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.gpu_vram = get_gpu_vram_mb()

    def load(self, model_path, **override_kwargs):
        """Load model with automatic optimal configuration."""
        from llama_cpp import Llama

        config = get_optimal_config(model_path, self.gpu_vram)

        if self.verbose:
            print("=" * 60, flush=True)
            print("  AMD AI Toolkit - Smart Auto-Configuration", flush=True)
            print("=" * 60, flush=True)
            print(f"  Model: {os.path.basename(model_path)}", flush=True)
            print(f"  Size: {config.get('model_size_gb', 0):.2f} GB", flush=True)
            print(f"  GPU VRAM: {self.gpu_vram} MB", flush=True)
            print(f"  Strategy: {config.get('strategy', 'UNKNOWN')}", flush=True)
            print(f"  Reason: {config.get('reason', 'N/A')}", flush=True)
            print("=" * 60, flush=True)
            print(flush=True)

        load_kwargs = {
            "model_path": model_path,
            "n_gpu_layers": config["n_gpu_layers"],
            "n_ctx": config["n_ctx"],
            "n_batch": config["n_batch"],
            "main_gpu": 0,
            "tensor_split": [1.0],
            "verbose": self.verbose,
        }
        load_kwargs.update(override_kwargs)

        try:
            if self.verbose:
                print(f"Loading with n_gpu_layers={load_kwargs['n_gpu_layers']}, "
                      f"n_ctx={load_kwargs['n_ctx']}...", flush=True)
            return Llama(**load_kwargs)
        except Exception as e:
            if self.verbose:
                print(f"GPU config failed: {str(e)[:80]}...", flush=True)
                print("Falling back to CPU...", flush=True)

            if load_kwargs["n_gpu_layers"] != 0:
                load_kwargs["n_gpu_layers"] = 0
                load_kwargs["n_ctx"] = 512
                load_kwargs["n_threads"] = 12
                return Llama(**load_kwargs)
            raise


def main():
    if len(sys.argv) < 2:
        print("AMD AI Toolkit - Smart Auto-Configuration", flush=True)
        print(flush=True)
        print("Usage: hip_autoconfig.py <model.gguf> [prompt]", flush=True)
        print(flush=True)
        print("Examples:", flush=True)
        print("  hip_autoconfig.py models/tinyllama.gguf", flush=True)
        print("  hip_autoconfig.py models/qwen-7b.gguf \"What is XSS?\"", flush=True)
        sys.exit(1)

    model_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "What is SQL injection?"

    loader = SmartLlamaLoader(verbose=True)

    print(flush=True)
    start = time.time()
    model = loader.load(model_path)
    load_time = time.time() - start
    print(flush=True)
    print(f"Model loaded in {load_time:.2f}s", flush=True)
    print(flush=True)

    print(f"Prompt: {prompt}", flush=True)
    print(flush=True)
    print("Generating...", flush=True)

    start = time.time()
    output = model(prompt, max_tokens=100, temperature=0.7)
    gen_time = time.time() - start

    response = output["choices"][0]["text"]
    tokens = output["usage"]["completion_tokens"]

    print(flush=True)
    print(f"Response: {response}", flush=True)
    print(flush=True)
    print("=" * 60, flush=True)
    print(f"  Generated {tokens} tokens in {gen_time:.2f}s", flush=True)
    print(f"  Speed: {tokens/gen_time:.2f} tokens/sec", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
