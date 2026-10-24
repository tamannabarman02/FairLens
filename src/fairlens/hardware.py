"""Hardware detection and profile selection."""
from __future__ import annotations

import os
import platform
import shutil
import socket
from dataclasses import asdict, dataclass

import psutil


@dataclass
class HardwareInfo:
    python: str
    os: str
    cpu: str
    cpu_threads: int
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float
    cuda: bool
    mps: bool
    gpu_name: str | None
    gpu_vram_gb: float | None
    internet: bool
    torch_version: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _internet(host: str = "huggingface.co", port: int = 443, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def detect(check_internet: bool = True, disk_path: str | None = None) -> HardwareInfo:
    from .config import PROJECT_ROOT

    vm = psutil.virtual_memory()
    cuda, mps, gpu_name, vram, torch_version = False, False, None, None, None
    try:
        import torch

        torch_version = torch.__version__
        cuda = torch.cuda.is_available()
        if cuda:
            gpu_name = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        mps = _mps_available(torch)
        if mps and not cuda:
            gpu_name = "Apple MPS"
    except ImportError:
        pass
    return HardwareInfo(
        python=platform.python_version(),
        os=f"{platform.system()} {platform.release()} ({platform.version()})",
        cpu=_cpu_name(),
        cpu_threads=os.cpu_count() or 1,
        ram_total_gb=round(vm.total / 2**30, 1),
        ram_available_gb=round(vm.available / 2**30, 1),
        disk_free_gb=round(shutil.disk_usage(disk_path or PROJECT_ROOT).free / 2**30, 1),
        cuda=cuda,
        mps=mps,
        gpu_name=gpu_name,
        gpu_vram_gb=vram,
        internet=_internet() if check_internet else False,
        torch_version=torch_version,
    )


def select_profile(info: HardwareInfo | None = None) -> str:
    """Return 'gpu', 'standard' or 'lite'. FAIRLENS_PROFILE env var overrides."""
    forced = os.environ.get("FAIRLENS_PROFILE")
    if forced in {"gpu", "standard", "lite"}:
        return forced
    info = info or detect(check_internet=False)
    if info.cuda and (info.gpu_vram_gb or 0) >= 4:
        return "gpu"
    if info.ram_total_gb > 8.5:
        return "standard"
    return "lite"


def _mps_available(torch_module) -> bool:
    backend = getattr(torch_module.backends, "mps", None)
    return bool(backend and backend.is_available() and backend.is_built())


def device() -> str:
    """Return 'cuda', 'mps' or 'cpu' (in that order of preference). FAIRLENS_DEVICE env var overrides."""
    forced = (os.environ.get("FAIRLENS_DEVICE") or "").strip().lower()
    try:
        import torch
    except ImportError:
        return "cpu"
    available = {"cpu": True, "cuda": torch.cuda.is_available(), "mps": _mps_available(torch)}
    if forced in available:
        if available[forced]:
            return forced
        import warnings

        warnings.warn(f"FAIRLENS_DEVICE={forced} is not available on this machine; falling back to CPU.")
        return "cpu"
    if available["cuda"]:
        return "cuda"
    if available["mps"]:
        return "mps"
    return "cpu"


def device_label() -> str:
    """Short human-readable name of the selected compute device for the UI."""
    return {"cuda": "CUDA GPU", "mps": "Apple GPU (MPS)", "cpu": "CPU"}[device()]
