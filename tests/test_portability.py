"""V4 1.2b: no operating-system assumptions in code or shared config.

Fails on: text file I/O without an explicit encoding (open/read_text/write_text), Windows drive letters, and
hard-coded backslash path separators in source, scripts and shared configs.
"""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = [p for d in ["src", "scripts", "tests"] for p in (ROOT / d).rglob("*.py")] + [ROOT / "app.py"]
CONFIGS = [ROOT / "configs" / "models.yaml", ROOT / "pyproject.toml", ROOT / "requirements.txt", ROOT / "install.sh"]


def encodingless_calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kws = {k.arg for k in node.keywords}
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        if name in {"read_text", "write_text"} and "encoding" not in kws:
            bad.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno} {name}")
        if name == "open" and isinstance(node.func, ast.Name) and "encoding" not in kws:
            mode = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else \
                next((k.value.value for k in node.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)), "r")
            if "b" not in str(mode):
                bad.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno} open")
    return bad


def test_no_encodingless_text_io():
    bad = [b for p in CODE for b in encodingless_calls(p)]
    assert not bad, "text I/O without encoding=: " + ", ".join(bad)


DRIVE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/](?![\\/])")


def test_no_drive_letters_or_backslash_paths():
    bad = []
    for p in CODE + CONFIGS:
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if p.name == "test_portability.py" or "http" in line:
                continue
            code = line.split("#", 1)[0] if p.suffix in {".py", ".yaml", ".txt", ".sh", ".toml"} else line
            if DRIVE.search(code):
                bad.append(f"{p.relative_to(ROOT).as_posix()}:{i}")
    assert not bad, "drive letters in: " + ", ".join(bad)


def test_gitattributes_normalises_line_endings():
    ga = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "* text=auto" in ga and "*.sh text eol=lf" in ga
