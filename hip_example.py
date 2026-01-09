#!/usr/bin/env python3
"""
AMD AI Toolkit - HIP GPU Example
Working configuration for AMD RX 6700 XT (gfx1031)

Usage:
    .\run_hip_model.ps1 hip_example.py

Requirements:
    - AMD HIP SDK 5.7 installed
    - llama-cpp-python built with HIP support
    - A GGUF model in the models/ directory
"""

import os
import sys
import time

# Force line buffering
sys.stdout.reconfigure(line_buffering=True)

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Find a model in the models directory
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
MODEL_PATH = None

if os.path.exists(MODELS_DIR):
    for f in os.listdir(MODELS_DIR):
        if f.endswith('.gguf'):
            MODEL_PATH = os.path.join(MODELS_DIR, f)
            break

if MODEL_PATH is None:
    print("No GGUF model found in models/ directory")
    print("Download a model and place it in:", MODELS_DIR)
    print()
    print("Example:")
    print("  wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    sys.exit(1)

print("=" * 60, flush=True)
print("  AMD AI Toolkit - HIP GPU Example", flush=True)
print("  AMD Radeon RX 6700 XT (gfx1031)", flush=True)
print("=" * 60, flush=True)
print(flush=True)

from llama_cpp import Llama

print(f"Model: {os.path.basename(MODEL_PATH)}", flush=True)
print("Loading with full GPU offload...", flush=True)
start = time.time()

model = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=512,
    main_gpu=0,
    tensor_split=[1.0],
    verbose=True
)

load_time = time.time() - start
print(flush=True)
print(f"Model loaded in {load_time:.2f}s", flush=True)
print(flush=True)

prompt = "The AMD Radeon RX 6700 XT graphics card is"
print(f"Prompt: {prompt}", flush=True)
print(flush=True)

print("Generating...", flush=True)
start = time.time()
output = model(prompt, max_tokens=50, temperature=0.7)
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
print(flush=True)
print("HIP GPU acceleration is WORKING!", flush=True)
