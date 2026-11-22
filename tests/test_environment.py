import ast
import os
import re
from pathlib import Path

from fairlens import config, hardware


def test_config_loads_and_declares_all_slots():
    cfg = config.load_config()
    assert {"rule_based", "bert", "deberta", "bart_mnli"} <= set(cfg["detectors"])
    assert {"rule_based", "flan_t5", "qwen"} <= set(cfg["mitigation"]["rewriters"])
    assert {"blip", "florence"} <= set(cfg["captioning"]["captioners"])
    assert 0 < cfg["gate"]["min_semantic_similarity"] <= 1


def test_hf_cache_is_configured():
    assert os.environ.get("HF_HOME")


def test_hardware_detection_and_profile(monkeypatch):
    info = hardware.detect(check_internet=False)
    assert info.ram_total_gb > 0 and info.cpu_threads >= 1
    assert hardware.select_profile(info) in {"gpu", "standard", "lite"}
    monkeypatch.setenv("FAIRLENS_PROFILE", "lite")
    assert hardware.select_profile(info) == "lite"


ROOT = Path(__file__).resolve().parents[1]
PENDING_OWNERS: set[str] = set()


def _python_files():
    yield from ROOT.glob("src/**/*.py")
    yield from ROOT.glob("scripts/*.py")
    yield from ROOT.glob("tests/*.py")
    yield ROOT / "app.py"


def _text_io_without_encoding(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call) or any(k.arg == "encoding" for k in node.keywords):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        if name in ("read_text", "write_text"):
            found.append(f"{name}:{node.lineno}")
        elif name == "open" and isinstance(node.func, ast.Name):
            mode = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else "r"
            if "b" not in str(mode):
                found.append(f"open:{node.lineno}")
    return found


def test_text_io_always_names_an_encoding():
    bad = {}
    for p in _python_files():
        rel = p.relative_to(ROOT).as_posix()
        if rel not in PENDING_OWNERS and (hits := _text_io_without_encoding(p)):
            bad[rel] = hits
    assert bad == {}


def test_no_drive_letters_or_backslash_paths_in_code():
    pattern = re.compile(r"""["'][A-Za-z]:[\\/]|D:/FairLens""")
    code = [p for p in _python_files() if p.parent.name != "tests" and p.relative_to(ROOT).as_posix() not in PENDING_OWNERS]
    bad = [f"{p.relative_to(ROOT).as_posix()}:{i}" for p in code
           for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if pattern.search(line)]
    assert bad == []


def test_device_selection_and_override(monkeypatch):
    assert hardware.device() in {"cpu", "cuda", "mps"}
    monkeypatch.setenv("FAIRLENS_DEVICE", "cpu")
    assert hardware.device() == "cpu" and hardware.device_label() == "CPU"


def test_default_cache_dir_is_per_user():
    assert config.default_cache_dir().is_relative_to(Path.home()) or os.environ.get("LOCALAPPDATA")
