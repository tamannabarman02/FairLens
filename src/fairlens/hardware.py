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
    gpu_name: str | None
    gpu_vram_gb: float | None
    mps: bool
    device: str
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
    cuda, gpu_name, vram, torch_version = False, None, None, None
    try:
        import torch

        torch_version = torch.__version__
        cuda = torch.cuda.is_available()
        if cuda:
            gpu_name = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
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
        gpu_name=gpu_name,
        gpu_vram_gb=vram,
        mps=_mps_available(),
        device=device(),
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


def _mps_available() -> bool:
    try:
        import torch

        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except ImportError:
        return False


_SELFCHECK: dict[str, dict] = {}


def accelerator_selfcheck(dev: str, tol: float = 1e-3) -> dict:
    """Run a small fixed transformer encoder layer on CPU and on `dev` and compare outputs.

    On a GitHub macOS (virtualised) runner, MPS reported available but every encoder/decoder model except T5
    produced different outputs from CPU (outputs/audit/ci_device_families/). A device that fails this check is
    not used automatically. FAIRLENS_SKIP_SELFCHECK=1 disables the check."""
    if dev in _SELFCHECK:
        return _SELFCHECK[dev]
    if os.environ.get("FAIRLENS_SKIP_SELFCHECK") == "1":
        _SELFCHECK[dev] = {"ok": True, "skipped": True}
        return _SELFCHECK[dev]
    import torch

    g = torch.Generator().manual_seed(0)
    layer = torch.nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.0, activation="gelu",
                                             batch_first=True).eval()
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)
        x = torch.randn(2, 12, 64, generator=g)
        ref = layer(x)
        try:
            out = layer.to(dev)(x.to(dev)).cpu()
            diff = float((out - ref).abs().max())
            res = {"ok": diff <= tol, "max_abs_diff": diff}
        except Exception as exc:  # noqa: BLE001
            res = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    _SELFCHECK[dev] = res
    if not res["ok"]:
        import warnings

        warnings.warn(f"{dev} failed the FairLens numerical self-check ({res}); falling back to the next device", stacklevel=2)
    return res


def available_devices(check: bool = True) -> list[str]:
    """Usable devices in preference order. With check=True an accelerator must pass accelerator_selfcheck."""
    out = []
    try:
        import torch

        if torch.cuda.is_available() and (not check or accelerator_selfcheck("cuda")["ok"]):
            out.append("cuda")
    except ImportError:
        pass
    if _mps_available() and (not check or accelerator_selfcheck("mps")["ok"]):
        out.append("mps")
    return out + ["cpu"]


def device() -> str:
    """Device order CUDA -> MPS -> CPU. FAIRLENS_DEVICE=cuda|mps|cpu forces one; an unavailable forced device
    falls back to the automatic choice (with a warning) instead of crashing."""
    avail = available_devices()
    forced = os.environ.get("FAIRLENS_DEVICE", "").strip().lower()
    if forced:
        if forced in avail:
            return forced
        import warnings

        warnings.warn(f"FAIRLENS_DEVICE={forced} is not available here; using {avail[0]}", stacklevel=2)
    return avail[0]


def place(model):
    """Move a torch model to the selected device (no-op for CPU)."""
    dev = device()
    return model.to(dev) if dev != "cpu" else model


def to_model(batch, model):
    """Move a tokenizer/processor output (BatchEncoding/BatchFeature) to the model's device."""
    dev = getattr(model, "device", None)
    if dev is None or str(dev) == "cpu" or not hasattr(batch, "to"):
        return batch
    return batch.to(dev)
