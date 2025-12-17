from __future__ import annotations

from typing import Optional, Tuple

import torch


def _cuda_driver_context_error() -> Optional[str]:
    """Return a driver-level CUDA context creation error if any.

    This uses the CUDA Driver API (libcuda) via ctypes, so it can detect failures
    even when `torch.cuda.is_available()` returns True.
    """
    try:
        import ctypes
    except Exception as e:  # pragma: no cover
        return f"ctypes import failed: {e}"

    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        return f"libcuda.so.1 not found: {e}"

    CUresult = ctypes.c_int
    CUdevice = ctypes.c_int
    CUcontext = ctypes.c_void_p

    def _check(fn_name: str, code: int) -> Optional[str]:
        if code == 0:
            return None
        try:
            name_p = ctypes.c_char_p()
            desc_p = ctypes.c_char_p()
            libcuda.cuGetErrorName.argtypes = [CUresult, ctypes.POINTER(ctypes.c_char_p)]
            libcuda.cuGetErrorName.restype = CUresult
            libcuda.cuGetErrorString.argtypes = [CUresult, ctypes.POINTER(ctypes.c_char_p)]
            libcuda.cuGetErrorString.restype = CUresult
            libcuda.cuGetErrorName(CUresult(code), ctypes.byref(name_p))
            libcuda.cuGetErrorString(CUresult(code), ctypes.byref(desc_p))
            name = name_p.value.decode("utf-8") if name_p.value else str(code)
            desc = desc_p.value.decode("utf-8") if desc_p.value else ""
            return f"{fn_name} failed: {name}{f' ({desc})' if desc else ''}"
        except Exception:
            return f"{fn_name} failed with code {code}"

    libcuda.cuInit.argtypes = [ctypes.c_uint]
    libcuda.cuInit.restype = CUresult
    code = int(libcuda.cuInit(0))
    err = _check("cuInit", code)
    if err:
        return err

    libcuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    libcuda.cuDeviceGetCount.restype = CUresult
    count = ctypes.c_int()
    code = int(libcuda.cuDeviceGetCount(ctypes.byref(count)))
    err = _check("cuDeviceGetCount", code)
    if err:
        return err
    if count.value <= 0:
        return "No CUDA devices detected by the driver"

    libcuda.cuDeviceGet.argtypes = [ctypes.POINTER(CUdevice), ctypes.c_int]
    libcuda.cuDeviceGet.restype = CUresult
    dev = CUdevice()
    code = int(libcuda.cuDeviceGet(ctypes.byref(dev), 0))
    err = _check("cuDeviceGet", code)
    if err:
        return err

    libcuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(CUcontext), ctypes.c_uint, CUdevice]
    libcuda.cuCtxCreate_v2.restype = CUresult
    libcuda.cuCtxDestroy_v2.argtypes = [CUcontext]
    libcuda.cuCtxDestroy_v2.restype = CUresult

    ctx = CUcontext()
    code = int(libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev))
    err = _check("cuCtxCreate", code)
    if err:
        return err

    # Best-effort cleanup.
    try:
        libcuda.cuCtxDestroy_v2(ctx)
    except Exception:
        pass

    return None


def cuda_smoke_test() -> Tuple[bool, str]:
    """Return (ok, details) for CUDA usability, not just availability."""
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"

    try:
        torch.empty(1, device="cuda")
        return True, ""
    except Exception as e:
        driver_err = _cuda_driver_context_error()
        details = f"{type(e).__name__}: {e}"
        if driver_err:
            details = f"{details}\nDriver check: {driver_err}"
        return False, details


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a torch.device from 'auto'/'cpu'/'cuda' with a CUDA health check.

    - 'auto': use CUDA if available + usable, else CPU.
    - 'cuda': require CUDA to be usable (raises on failure).
    - anything else: forwarded to torch.device().
    """
    req = (requested or "auto").lower().strip()
    if req == "auto":
        ok, details = cuda_smoke_test()
        if ok:
            return torch.device("cuda")
        if torch.cuda.is_available():
            print("WARNING: CUDA detected but unusable; falling back to CPU.\n" + details)
        return torch.device("cpu")

    dev = torch.device(req)
    if dev.type != "cuda":
        return dev

    ok, details = cuda_smoke_test()
    if ok:
        return dev

    raise RuntimeError(
        "CUDA was requested but a basic CUDA smoke test failed.\n"
        f"{details}\n\n"
        "This is usually a host NVIDIA driver/GPU state issue (Docker/venvs won't fix it).\n"
        "Try: reboot; or `sudo nvidia-smi --gpu-reset -i 0` (if supported); or restart NVIDIA services/modules."
    )

