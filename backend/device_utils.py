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


def is_strict_offline() -> bool:
    """Check if strict offline execution is explicitly requested via environment variables."""
    for var in ("STRICT_OFFLINE", "OFFLINE_MODE", "AIR_GAPPED"):
        if os.environ.get(var, "").strip().lower() in {"1", "true", "yes"}:
            return True
    return False


def configure_offline_mode(force: bool = False):
    """Configure Hugging Face offline environment if explicitly requested or forced.
    By default, allows dynamic online download when models are missing from local cache
    (e.g., Streamlit Community Cloud and fresh container deployments).
    """
    if force or is_strict_offline():
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        # Do not force offline mode by default
        pass


# Only activate offline mode if explicitly requested by environment
if is_strict_offline():
    configure_offline_mode()


def load_pretrained_safe(loader_fn, *args, **kwargs):
    """Attempt loading pretrained model/tokenizer offline from local cache first.
    If files are not cached locally (such as on Streamlit Community Cloud), dynamically
    downloads them from Hugging Face Hub unless strict offline mode is explicitly enabled.
    """
    # 1. Attempt loading from local cache first (instant, zero network latency)
    try:
        kwargs_cached = dict(kwargs)
        kwargs_cached["local_files_only"] = True
        return loader_fn(*args, **kwargs_cached)
    except TypeError:
        # loader_fn does not accept local_files_only parameter
        pass
    except Exception as cache_err:
        if is_strict_offline():
            print(f"[ModelLoader] Offline loading failed and strict offline mode is active: {cache_err}")
            raise cache_err
        print(f"[ModelLoader] Model components not in local cache. Downloading from Hugging Face Hub...")

    # 2. Local cache miss: dynamically download from Hugging Face Hub
    # Temporarily remove any offline environment flags so HF Hub client can connect
    saved_env = {}
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if var in os.environ:
            saved_env[var] = os.environ.pop(var)

    try:
        kwargs_online = dict(kwargs)
        kwargs_online["local_files_only"] = False
        return loader_fn(*args, **kwargs_online)
    except TypeError:
        # If loader_fn rejects local_files_only=False, call without it
        try:
            return loader_fn(*args, **kwargs)
        except Exception as retry_err:
            raise retry_err
    except Exception as dl_err:
        print(f"[ModelLoader] Failed to download model from Hugging Face Hub: {dl_err}")
        raise dl_err
    finally:
        # Restore environment variables
        for var, val in saved_env.items():
            os.environ[var] = val


def clear_device_cache(device: str | None = None):
    """Safely flush CUDA and CPU memory caches."""
    import gc
    gc.collect()
    if torch is None:
        return
    dev = device or get_device()
    if dev == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        except Exception:
            pass
