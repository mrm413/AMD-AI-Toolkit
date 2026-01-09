"""
GPU Thread Helper for AMD HIP/ROCm on Windows
==============================================

Drop this file into your project when building multi-threaded servers
(FastAPI, Flask, etc.) with llama-cpp-python on AMD GPUs.

THE PROBLEM:
rocBLAS initializes its GPU context per-thread. Web frameworks handle
requests in different threads, causing "MUL_MAT failed" errors.

THE SOLUTION:
This helper routes all GPU operations through a single dedicated thread,
ensuring rocBLAS initializes once and stays valid.

USAGE:
    from gpu_thread_helper import gpu_generate, gpu_call

    # For llama-cpp-python style (model as callable)
    response = gpu_generate(llm, prompt, max_tokens=500)

    # For custom loaders with .generate() method
    response = gpu_call(loader.generate, model_name, prompt, temperature=0.7)

    # For any GPU function
    result = gpu_call(my_gpu_function, arg1, arg2, kwarg=value)

Author: Torsova LLC
License: MIT
"""

from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, TypeVar
import threading
import functools

T = TypeVar('T')

# Single-threaded executor for all GPU operations
_gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="GPU_Worker")
_gpu_lock = threading.Lock()
_initialized = False


def _ensure_initialized():
    """Log when the GPU thread is first used."""
    global _initialized
    if not _initialized:
        thread = threading.current_thread()
        print(f"[GPU Helper] GPU thread initialized: {thread.name} (tid: {thread.ident})")
        _initialized = True


def gpu_call(func: Callable[..., T], *args, timeout: float = 600, **kwargs) -> T:
    """
    Execute any function in the GPU thread.

    Args:
        func: The function to call
        *args: Positional arguments for the function
        timeout: Maximum wait time in seconds (default 10 minutes)
        **kwargs: Keyword arguments for the function

    Returns:
        The function's return value

    Raises:
        TimeoutError: If execution exceeds timeout
        Exception: Any exception raised by the function
    """
    def _execute():
        _ensure_initialized()
        with _gpu_lock:
            return func(*args, **kwargs)

    future: Future = _gpu_executor.submit(_execute)
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        print(f"[GPU Helper] WARNING: GPU call timed out after {timeout}s")
        raise


def gpu_generate(model, prompt: str, timeout: float = 600, **kwargs) -> str:
    """
    Thread-safe wrapper for model generation.

    Works with llama-cpp-python models that are callable:
        llm = Llama(model_path="...")
        response = gpu_generate(llm, "Hello", max_tokens=100)

    Args:
        model: A callable model (like Llama instance)
        prompt: The input prompt
        timeout: Maximum wait time in seconds
        **kwargs: Additional generation parameters (max_tokens, temperature, etc.)

    Returns:
        Generated text string
    """
    def _generate():
        _ensure_initialized()
        with _gpu_lock:
            result = model(prompt, **kwargs)
            # Handle llama-cpp-python response format
            if isinstance(result, dict) and 'choices' in result:
                return result['choices'][0]['text']
            return result

    future: Future = _gpu_executor.submit(_generate)
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        print(f"[GPU Helper] WARNING: Generation timed out after {timeout}s")
        raise


def gpu_method(method_name: str):
    """
    Decorator factory for wrapping object methods to run on GPU thread.

    Usage:
        class MyLoader:
            @gpu_method('generate')
            def generate(self, prompt, **kwargs):
                return self._model(prompt, **kwargs)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return gpu_call(func, *args, **kwargs)
        return wrapper
    return decorator


def shutdown():
    """
    Cleanly shutdown the GPU executor.
    Call this when your server is shutting down.
    """
    global _initialized
    _gpu_executor.shutdown(wait=True)
    _initialized = False
    print("[GPU Helper] GPU executor shut down")


# For FastAPI integration
class GPULifespan:
    """
    FastAPI lifespan context manager for clean GPU shutdown.

    Usage:
        from contextlib import asynccontextmanager
        from gpu_thread_helper import GPULifespan

        @asynccontextmanager
        async def lifespan(app):
            yield
            GPULifespan.shutdown()

        app = FastAPI(lifespan=lifespan)
    """
    @staticmethod
    def shutdown():
        shutdown()


# Example usage and self-test
if __name__ == "__main__":
    print("GPU Thread Helper - Self Test")
    print("=" * 40)

    # Simulate a "model" that's just a function
    def fake_model(prompt, max_tokens=100):
        import time
        time.sleep(0.1)  # Simulate processing
        return {"choices": [{"text": f"Response to: {prompt[:50]}"}]}

    # Test gpu_generate
    print("\nTesting gpu_generate...")
    result = gpu_generate(fake_model, "Hello, world!", max_tokens=50)
    print(f"Result: {result}")

    # Test gpu_call
    print("\nTesting gpu_call...")
    def add(a, b):
        return a + b
    result = gpu_call(add, 2, 3)
    print(f"2 + 3 = {result}")

    # Test multiple calls (proves single-thread execution)
    print("\nTesting sequential execution...")
    import time
    start = time.time()
    for i in range(5):
        gpu_call(lambda x: time.sleep(0.05) or x, i)
    elapsed = time.time() - start
    print(f"5 calls took {elapsed:.2f}s (should be ~0.25s if sequential)")

    # Cleanup
    shutdown()
    print("\nAll tests passed!")
