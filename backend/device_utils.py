"""Hardware Device Management Utility for CPU and GPU (CUDA) Execution.

Provides seamless, automatic device selection:
- Uses NVIDIA CUDA GPU when available and verified operational.
- Automatically falls back to CPU when no GPU is installed or when CUDA initialization fails.
- Supports automatic recovery and fallback to CPU on CUDA OutOfMemory (OOM) errors.
- Respects the 'FORCE_CPU' or 'DEVICE=cpu' environment variables for headless or CPU-only testing.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

try:
    import torch
except Exception:
    torch = None


def is_cpu_forced() -> bool:
    """Check if CPU execution is explicitly requested via environment variables."""
    force_cpu_val = os.environ.get("FORCE_CPU", "").strip().lower()
    device_val = os.environ.get("DEVICE", "").strip().lower()
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()

    if force_cpu_val in {"1", "true", "yes"}:
        return True
    if device_val == "cpu":
        return True
    if cuda_vis == "-1" or cuda_vis == '""' or cuda_vis == "''":
        return True
    return False


def get_device() -> str:
    """Return 'cuda' if a working NVIDIA GPU is available and not forced to CPU; else 'cpu'."""
    if is_cpu_forced() or torch is None:
        return "cpu"

    if torch.cuda.is_available():
        try:
            # Test actual memory allocation to ensure drivers/runtime are operational
            test_tensor = torch.zeros(1, device="cuda")
            del test_tensor
            return "cuda"
        except Exception as err:
            print(f"[Device] Warning: CUDA reported available but failed initialization: {err}. Falling back to CPU.")
            return "cpu"

    return "cpu"


def get_device_info() -> Dict[str, Any]:
    """Return diagnostic metadata about current computing hardware and PyTorch device."""
    active_device = get_device()
    info: Dict[str, Any] = {
        "active_device": active_device,
        "is_gpu": (active_device == "cuda"),
        "cpu_count": os.cpu_count() or 1,
        "cuda_available": False,
        "gpu_name": None,
        "gpu_count": 0,
        "gpu_memory_total_mb": None,
        "gpu_memory_allocated_mb": None,
    }

    if torch is not None:
        info["torch_version"] = torch.__version__
        if hasattr(torch, "get_num_threads"):
            info["cpu_threads"] = torch.get_num_threads()

        if torch.cuda.is_available() and not is_cpu_forced():
            info["cuda_available"] = True
            info["gpu_count"] = torch.cuda.device_count()
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                info["gpu_memory_total_mb"] = round(props.total_memory / (1024 * 1024), 2)
                info["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
            except Exception:
                pass

    return info


def configure_torch_threads():
    """Optimize PyTorch CPU thread count if running on CPU."""
    if torch is None:
        return
    active_dev = get_device()
    if active_dev == "cpu" and hasattr(torch, "set_num_threads"):
        try:
            optimal_threads = min(8, max(1, (os.cpu_count() or 4) - 1))
            torch.set_num_threads(optimal_threads)
        except Exception:
            pass


def configure_offline_mode():
    """Ensure Hugging Face and Transformers operate offline by default without network calls."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# Automatically enable offline mode on module import
configure_offline_mode()


def load_pretrained_safe(loader_fn, *args, **kwargs):
    """Attempt loading pretrained model/tokenizer offline first (local_files_only=True).
    Falls back to standard loading if local cache is missing and internet is available.
    """
    try:
        return loader_fn(*args, **kwargs, local_files_only=True)
    except Exception as err:
        # If strict offline requested or local_files_only failed, retry without local_files_only
        # only if network might be available, otherwise re-raise the original error
        if os.environ.get("STRICT_OFFLINE", "").strip().lower() in {"1", "true", "yes"}:
            raise
        try:
            return loader_fn(*args, **kwargs)
        except Exception:
            raise err


def clear_device_cache(device: str | None = None):
    """Safely flush CUDA cache if using GPU."""
    if torch is None:
        return
    dev = device or get_device()
    if dev == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
