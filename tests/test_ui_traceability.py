"""V4 U.6: the UI never renders a metric typed into its code; research pages show run provenance."""
import ast
import re
from pathlib import Path

PAGES = Path(__file__).resolve().parents[1] / "src" / "fairlens" / "ui" / "pages"
NUMBERISH = re.compile(r"\d+(\.\d+)?\s*%?|\d+(\.\d+)?x")
METRIC_CALLS = {"metric_tile", "score_bar", "bar", "grouped_bar", "donut", "hbar", "lines", "radar", "html_table"}


def _literal_numbers(node) -> list[str]:
    """Numeric constants (or number-looking strings) inside a call's data arguments. Constants used as slice
    bounds, `.get()` defaults, comparison operands or arithmetic guards are not displayed values and are ignored."""
    skip = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Slice):
            skip |= {id(x) for x in (sub.lower, sub.upper, sub.step) if x is not None}
        elif isinstance(sub, ast.Call) and getattr(sub.func, "attr", "") == "get":
            skip |= {id(x) for x in sub.args[1:]}
        elif isinstance(sub, (ast.Compare, ast.BinOp, ast.IfExp)):
            skip |= {id(x) for x in ast.walk(sub) if isinstance(x, ast.Constant)}
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and id(sub) not in skip:
            v = sub.value
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(repr(v))
            elif isinstance(v, str) and NUMBERISH.fullmatch(v.strip()):
                out.append(repr(v))
    return out


def test_no_hardcoded_metric_values():
    bad = []
    for p in PAGES.glob("*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name not in METRIC_CALLS:
                continue
            # value-carrying arguments: positional args after the label (layout kwargs like height/yrange are allowed)
            data_args = node.args[1:2] if name in {"metric_tile", "score_bar"} else node.args[:2]
            for a in data_args:
                nums = _literal_numbers(a)
                if nums:
                    bad.append(f"{p.name}:{node.lineno} {name}({', '.join(nums)})")
    assert not bad, "hard-coded metric values in UI: " + "; ".join(bad)


def test_research_pages_show_provenance_for_every_summary():
    for page, summaries in {"benchmarks.py": ["det", "mit", "gate", "img"], "model_comparison.py": ["s"]}.items():
        src = (PAGES / page).read_text(encoding="utf-8")
        for var in summaries:
            assert re.search(rf"C\.provenance\({var},", src), f"{page}: results from `{var}` rendered without provenance"
        assert "latest_with_manifest" in src and "latest_summary(" not in src
