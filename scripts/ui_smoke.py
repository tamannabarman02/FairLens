"""Browser smoke test of the dashboard with headless Chromium (Playwright). Runs on Windows, macOS and Linux.

Starts Streamlit (unless --url is given), opens every page at the given widths in light and dark theme by clicking the
sidebar (like a user), records Streamlit exceptions / console errors / horizontal overflow and saves full-page
screenshots to outputs/ui_screenshots/.

  python scripts/ui_smoke.py [--widths 1366,1920] [--themes light,dark] [--pages Home,Benchmarks,...] [--flow] [--no-pages]

--flow then restarts the app with FAIRLENS_UI_STUBS=1 (light stand-ins for the neural models, see
src/fairlens/ui/smoke_stubs.py) and drives: navigation from every Home entry card, the 6-step FairLens Review with a
real PDF and a real DOCX (plus a scanned PDF that must be refused), text upload of TXT and CSV, image upload of JPG, PNG
and GIF, the Gate decision display and every export button. Browsers: `python -m playwright install chromium`
(set PLAYWRIGHT_BROWSERS_PATH yourself if you keep them somewhere else).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ui_screenshots"
FIXTURES = ROOT / "tests" / "fixtures" / "ingestion"
PORT = 8599
PAGES = ["Home", "FairLens Review", "Text Analysis", "Mitigation", "Reliability Gate", "Image Analysis", "Benchmarks",
         "Model Comparison", "Error Analysis", "Experiments & Exports", "Use Cases", "About"]
SLUG = {"Home": "home", "FairLens Review": "review", "Text Analysis": "text", "Mitigation": "mitigation",
        "Reliability Gate": "gate", "Image Analysis": "image", "Benchmarks": "benchmarks", "Model Comparison": "models",
        "Error Analysis": "errors", "Experiments & Exports": "experiments", "Use Cases": "use-cases", "About": "about"}
HOME_ENTRIES = ["FairLens Review", "Text Analysis", "Image Analysis", "Mitigation", "Reliability Gate", "Benchmarks", "Use Cases"]


STUBS = True


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def wait_up(url: str, timeout: int = 90) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url + "/_stcore/health", timeout=3)
            return
        except Exception:  # noqa: BLE001
            time.sleep(1)
    raise RuntimeError("streamlit did not start")


def start_server(stubs: bool, log_name: str) -> subprocess.Popen:
    env = {**os.environ, "PYTHONUTF8": "1"}
    if stubs:
        env["FAIRLENS_UI_STUBS"] = "1"
    else:
        env.pop("FAIRLENS_UI_STUBS", None)
    log = open(OUT.parent / log_name, "w", encoding="utf-8")
    return subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
                             "--server.port", str(PORT)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def settle(page, timeout_ms: int = 600_000) -> None:
    """Wait until Streamlit finished running the script (no running indicator)."""
    page.wait_for_load_state("networkidle")
    try:
        page.wait_for_selector('.fl-crumb, [data-testid="stException"]', timeout=timeout_ms)
    except Exception:  # noqa: BLE001
        pass
    time.sleep(2.5)
    t0 = time.time()
    while (time.time() - t0) * 1000 < timeout_ms:
        if not _busy(page):
            time.sleep(0.8)
            if not _busy(page):
                return
        time.sleep(0.5)


RUNNING_LABELS = ("Running FairLens pipeline", "Generating rewrites and checking reliability", "flagged sentence(s) with the rewriters",
                  "Loading captioners and analysing", "Running the full pipeline")


def _busy(page) -> bool:
    """True while Streamlit reruns or one of the app's long-running status panels is still working."""
    if page.locator('[data-testid="stStatusWidget"]').count():
        return True
    return any(page.get_by_text(label).count() for label in RUNNING_LABELS)


def errors_on(page) -> list[str]:
    errs = [e.inner_text()[:400] for e in page.locator('[data-testid="stException"]').all()]
    errs += [e.inner_text()[:300] for e in page.locator('[data-testid="stAlertContentError"]').all()]
    return errs


def nav(page, title: str) -> None:
    page.locator('[data-testid="stSidebar"]').get_by_role("link", name=title, exact=True).click()
    settle(page)


def crumb(page) -> str:
    return page.locator(".fl-crumb").first.inner_text().strip()


def full_screenshot(page, path: Path, width: int) -> None:
    """Streamlit scrolls inside its own container, so grow the viewport to the content height first."""
    h = page.evaluate("""() => { const m = document.querySelector('[data-testid="stMain"]') ||
                                   document.querySelector('[data-testid="stAppViewContainer"]');
                                 return Math.max(m ? m.scrollHeight : 0, document.body.scrollHeight); }""")
    page.set_viewport_size({"width": width, "height": min(max(900, int(h) + 40), 9000)})
    time.sleep(0.8)
    page.screenshot(path=str(path), full_page=True)
    page.set_viewport_size({"width": width, "height": 900})


def set_theme(page, want_dark: bool) -> None:
    is_dark = page.evaluate("getComputedStyle(document.querySelector('.stApp')).backgroundColor") == "rgb(15, 16, 32)"
    tog = page.locator(".st-key-theme_toggle button")
    if tog.count() and is_dark != want_dark:
        tog.first.click()
        settle(page)
    page.mouse.move(2, 2)
    time.sleep(0.4)


def screenshots(br, url: str, widths: list[int], themes: list[str], pages: list[str], report: dict) -> None:
    for width in widths:
        ctx = br.new_context(viewport={"width": width, "height": 900})
        page = ctx.new_page()
        console: list[str] = []
        page.on("console", lambda m: console.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: console.append(str(e)))
        page.goto(url)
        settle(page)
        for theme in themes:
            for title in pages:
                console.clear()
                nav(page, title)
                set_theme(page, theme == "dark")
                shot = OUT / f"{SLUG[title]}_{theme}_{width}.png"
                full_screenshot(page, shot, width)
                errs = errors_on(page)
                overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 2")
                key = f"{SLUG[title]}/{theme}/{width}"
                report["pages"][key] = {"errors": errs, "console": console[:5], "horizontal_overflow": overflow,
                                        "crumb": crumb(page), "screenshot": rel(shot)}
                ok = not errs and not overflow and not console and crumb(page).endswith(title)
                print(f"{'OK ' if ok else 'ERR'} {key} errors={len(errs)} overflow={overflow} console={len(console)}", flush=True)
                for e in errs + console[:3]:
                    print("    ", e.replace("\n", " ")[:300])
        ctx.close()


def ensure_docs(tmp: Path) -> dict[str, Path]:
    """Real files for the flow: the ingestion fixtures when present, else generated with reportlab/python-docx/PIL."""
    want = {"pdf": "job_description_2_pages.pdf", "docx": "header_paragraphs_table.docx", "scanned": "scanned_no_text_layer.pdf",
            "txt": "utf8.txt", "csv": "multi_column.csv", "jpg": "photo.jpg", "png": "transparent_alpha.png", "gif": "static.gif"}
    out = {k: FIXTURES / v for k, v in want.items() if (FIXTURES / v).exists()}
    jd = ("We are looking for a young, aggressive salesman to join our London team. The ideal candidate is a digital native "
          "with at least 5 years of experience in B2B sales. Applications close on 12 March 2026.")
    if "pdf" not in out:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        out["pdf"] = tmp / "job.pdf"
        c = canvas.Canvas(str(out["pdf"]), pagesize=A4)
        c.drawString(60, 780, jd[:95])
        c.drawString(60, 760, jd[95:])
        c.save()
    if "docx" not in out:
        from docx import Document

        out["docx"] = tmp / "job.docx"
        d = Document()
        d.add_paragraph(jd)
        d.save(out["docx"])
    if "txt" not in out:
        out["txt"] = tmp / "job.txt"
        out["txt"].write_text(jd, encoding="utf-8")
    if "csv" not in out:
        out["csv"] = tmp / "jobs.csv"
        out["csv"].write_text("id,text\n1," + jd.replace(",", "") + "\n", encoding="utf-8")
    from PIL import Image

    for k, fmt in (("jpg", "JPEG"), ("png", "PNG"), ("gif", "GIF")):
        if k not in out:
            out[k] = tmp / f"img.{k}"
            Image.new("RGB", (64, 64), (120, 90, 200)).save(out[k], fmt)
    return out


class Checks:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.items.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"  {'PASS' if ok else 'FAIL'} {name}{(' - ' + detail) if detail else ''}", flush=True)
        return ok


def download(page, label: str, dest: Path) -> Path | None:
    btn = page.get_by_role("button", name=label, exact=True)
    if not btn.count():
        return None
    with page.expect_download(timeout=60_000) as d:
        btn.first.click()
    p = dest / d.value.suggested_filename
    d.value.save_as(p)
    settle(page)
    return p


def upload(page, path: Path, index: int = 0) -> None:
    page.locator('[data-testid="stFileUploaderDropzoneInput"]').nth(index).set_input_files(str(path))
    settle(page)


def review_run(page, chk: Checks, doc: Path, tag: str, dl: Path) -> None:
    nav(page, "FairLens Review")
    if page.get_by_role("button", name="Start a new review").count():
        page.get_by_role("button", name="Start a new review").click()
        settle(page)
    chk.add(f"review[{tag}] disclaimer shown", page.get_by_text("Review assistant, not a legal or compliance certification tool.").count() > 0)
    page.get_by_role("tab", name="Upload file").click()
    upload(page, doc)
    chk.add(f"review[{tag}] text extracted", page.get_by_text("Extracted").count() > 0, "; ".join(errors_on(page)))
    page.get_by_role("button", name="Use this document").click()
    settle(page)
    page.get_by_role("button", name="Run review").click()
    settle(page, 900_000)
    chk.add(f"review[{tag}] step 2 flagged phrases", page.get_by_text("2 · Flagged phrases").count() > 0, "; ".join(errors_on(page)))
    chk.add(f"review[{tag}] evidence cards", page.locator(".fl-evidence").count() > 0 or page.get_by_text("No phrases flagged").count() > 0,
            f"{page.locator('.fl-evidence').count()} phrase cards")
    full_screenshot(page, OUT / f"flow_review_{tag}_2_flags.png", 1366)
    page.get_by_role("button", name="Next: Rewrites").click()
    settle(page)
    chk.add(f"review[{tag}] step 3 diff view", page.locator(".fl-diff").count() > 0 or page.get_by_text("No rewrite proposed").count() > 0)
    full_screenshot(page, OUT / f"flow_review_{tag}_3_rewrites.png", 1366)
    page.get_by_role("button", name="Next: Fact checks").click()
    settle(page)
    names = page.locator(".fl-check .n").all_inner_texts()
    chk.add(f"review[{tag}] step 4 fact checks", all(n in names for n in ["Names / entities", "Numbers", "Dates", "Roles", "Negation", "Quotations"]),
            ", ".join(names))
    page.get_by_role("button", name="Next: Gate verdict").click()
    settle(page)
    verdict = page.locator(".fl-banner .ttl").all_inner_texts()
    chk.add(f"review[{tag}] step 5 gate verdict", bool(verdict), f"verdict={verdict}")
    full_screenshot(page, OUT / f"flow_review_{tag}_5_gate.png", 1366)
    page.get_by_role("button", name="Next: Export").click()
    settle(page)
    pdf = download(page, "PDF report", dl)
    chk.add(f"review[{tag}] PDF export", bool(pdf and pdf.read_bytes()[:4] == b"%PDF"), f"{pdf.stat().st_size if pdf else 0} bytes")
    js = download(page, "JSON", dl)
    ok = False
    if js:
        data = json.loads(js.read_text(encoding="utf-8"))
        ok = "not a legal or compliance certification tool" in data.get("disclaimer", "") and "sentences" in data
    chk.add(f"review[{tag}] JSON export", ok, f"{js.stat().st_size if js else 0} bytes")
    html = download(page, "HTML report", dl)
    chk.add(f"review[{tag}] HTML export", bool(html and b"FairLens Review" in html.read_bytes()))


def flow(br, url: str, docs: dict[str, Path]) -> dict:
    chk = Checks()
    dl = OUT / "flow_downloads"
    dl.mkdir(parents=True, exist_ok=True)
    ctx = br.new_context(viewport={"width": 1366, "height": 900}, accept_downloads=True)
    page = ctx.new_page()
    console: list[str] = []
    page.on("console", lambda m: console.append(m.text) if m.type == "error" else None)
    page.goto(url)
    settle(page)
    nav(page, "Text Analysis")
    chk.add("stand-in banner shown only with stand-in models", (page.locator(".fl-stub").count() > 0) == STUBS)

    print("navigation:", flush=True)
    for title in HOME_ENTRIES:
        nav(page, "Home")
        btns = page.locator(f'.st-key-home_go_{title.lower().replace(" ", "_")} button')
        if not btns.count():
            chk.add(f"home card -> {title}", False, "button not found")
            continue
        btns.first.click()
        settle(page)
        chk.add(f"home card -> {title}", crumb(page).endswith(title), crumb(page))
    for title in PAGES:
        nav(page, title)
        chk.add(f"sidebar -> {title}", crumb(page).endswith(title) and not errors_on(page), crumb(page))

    print("review workflow:", flush=True)
    review_run(page, chk, docs["pdf"], "pdf", dl)
    review_run(page, chk, docs["docx"], "docx", dl)
    if "scanned" in docs:
        nav(page, "FairLens Review")
        page.get_by_role("button", name="Start a new review").click()
        settle(page)
        page.get_by_role("tab", name="Upload file").click()
        upload(page, docs["scanned"])
        err = " ".join(errors_on(page))
        chk.add("review refuses scanned PDF with a plain message", bool(err) and "Traceback" not in err, err[:160])

    print("text upload + gate:", flush=True)
    for key in ("txt", "csv"):
        nav(page, "Text Analysis")
        page.get_by_role("tab", name="Upload File").click()
        upload(page, docs[key])
        extracted = page.get_by_text("Extracted").count() > 0
        picker = page.get_by_text("Text column").count() > 0 if key == "csv" else True
        chk.add(f"text upload {key.upper()}", extracted and picker, "; ".join(errors_on(page)) or ("column picker shown" if key == "csv" else ""))
    page.get_by_role("button", name="Use extracted text").click()
    settle(page)
    page.get_by_role("button", name="Analyze & mitigate").click()
    settle(page, 900_000)
    chk.add("analyze & mitigate -> Mitigation page", crumb(page).endswith("Mitigation") and page.locator(".fl-card h4").count() >= 2,
            crumb(page))
    nav(page, "Reliability Gate")
    banner = page.locator(".fl-banner .ttl").all_inner_texts()
    chk.add("gate decision banner", bool(banner), f"{banner}")
    page.get_by_role("tab", name="Detailed Analysis").click()
    time.sleep(1)
    chk.add("gate preservation grid", page.locator(".fl-check").count() >= 6)
    full_screenshot(page, OUT / "flow_gate.png", 1366)
    nav(page, "Text Analysis")
    for label, magic in (("JSON", b"{"), ("HTML report", b"<!doctype"), ("PDF report", b"%PDF")):
        p = download(page, label, dl)
        chk.add(f"text analysis export {label}", bool(p and p.read_bytes()[:len(magic)].lower() == magic.lower()))

    print("image upload:", flush=True)
    for key in ("jpg", "png", "gif"):
        nav(page, "Image Analysis")
        upload(page, docs[key])
        gen = page.get_by_role("button", name="Generate captions & analyse")
        if not gen.count():
            chk.add(f"image upload {key.upper()}", False, "; ".join(errors_on(page)))
            continue
        gen.click()
        settle(page, 600_000)
        chk.add(f"image upload {key.upper()} -> captions", page.locator(".fl-card h4").count() >= 2 and not errors_on(page),
                "; ".join(errors_on(page)))
    full_screenshot(page, OUT / "flow_image.png", 1366)

    print("research pages:", flush=True)
    nav(page, "Experiments & Exports")
    labels = page.locator(".fl-status").all_inner_texts()
    chk.add("stopped DeBERTa runs shown as Failed run", labels.count("Failed run") >= 2 if labels else False, f"{labels}")
    nav(page, "Benchmarks")
    status = page.locator(".fl-status").all_inner_texts()
    tiles = page.locator(".fl-tile").count()
    consistent = tiles > 0 if "Completed run" in status else tiles == 0
    chk.add("benchmarks: status labels shown, numbers only with completed runs", bool(status) and consistent,
            f"{status}, {tiles} tiles")
    ctx.close()
    passed = sum(c["ok"] for c in chk.items)
    return {"passed": passed, "total": len(chk.items), "checks": chk.items, "console_errors": console[:10]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--widths", default="1366,1920")
    ap.add_argument("--themes", default="light,dark")
    ap.add_argument("--pages", default=",".join(PAGES))
    ap.add_argument("--flow", action="store_true")
    ap.add_argument("--no-pages", action="store_true")
    ap.add_argument("--real", action="store_true", help="run --flow with the real models instead of the stand-ins")
    a = ap.parse_args()
    global STUBS
    STUBS = not a.real
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    url = a.url or f"http://localhost:{PORT}"
    report = {"url": url, "pages": {}, "flow": None}
    with sync_playwright() as pw:
        br = pw.chromium.launch()
        if not a.no_pages:
            proc = None if a.url else start_server(False, "ui_smoke_server.log")
            try:
                wait_up(url)
                screenshots(br, url, [int(w) for w in a.widths.split(",")], a.themes.split(","), a.pages.split(","), report)
            finally:
                stop_server(proc)
        if a.flow:
            proc = None if a.url else start_server(not a.real, "ui_smoke_flow_server.log")
            try:
                wait_up(url)
                with tempfile.TemporaryDirectory() as tmp:
                    report["flow"] = flow(br, url, ensure_docs(Path(tmp)))
            finally:
                stop_server(proc)
        br.close()
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    bad = [k for k, v in report["pages"].items() if v["errors"] or v["horizontal_overflow"] or v["console"]]
    print(f"pages checked: {len(report['pages'])}, with problems: {len(bad)}")
    if report["flow"]:
        print(f"flow checks passed: {report['flow']['passed']}/{report['flow']['total']}")
    sys.exit(1 if bad or (report["flow"] and report["flow"]["passed"] < report["flow"]["total"]) else 0)


if __name__ == "__main__":
    main()
