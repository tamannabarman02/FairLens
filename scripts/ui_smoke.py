"""Browser smoke test of the dashboard with headless Chromium (Playwright).

Starts Streamlit (unless --url is given), opens every page at the given widths in light and dark theme,
records Streamlit exceptions / console errors and saves screenshots to outputs/ui_screenshots/.

  python scripts/ui_smoke.py [--widths 1366,1920] [--themes light,dark] [--pages home,text,...] [--flow]
--flow additionally drives the main workflow: example text -> Analyze -> Mitigation -> Gate, and exports.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ui_screenshots"
PAGES = ["", "text", "mitigation", "gate", "image", "models", "benchmarks", "about", "errors", "experiments", "use-cases"]


def wait_up(url: str, timeout: int = 90) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url + "/_stcore/health", timeout=3)
            return
        except Exception:  # noqa: BLE001
            time.sleep(1)
    raise RuntimeError("streamlit did not start")


def settle(page, timeout_ms: int = 120_000) -> None:
    """Wait until Streamlit finished running the script (no running indicator)."""
    page.wait_for_load_state("networkidle")
    try:  # the app renders its breadcrumb first; an exception block also counts as "rendered"
        page.wait_for_selector('.fl-crumb, [data-testid="stException"]', timeout=timeout_ms)
    except Exception:  # noqa: BLE001
        pass
    time.sleep(4)  # Streamlit keeps rendering widgets after the first element appears
    t0 = time.time()
    while (time.time() - t0) * 1000 < timeout_ms:
        if page.locator('[data-testid="stStatusWidget"]').count() == 0:
            time.sleep(0.8)
            if page.locator('[data-testid="stStatusWidget"]').count() == 0:
                return
        time.sleep(0.5)


def errors_on(page) -> list[str]:
    errs = [e.inner_text()[:400] for e in page.locator('[data-testid="stException"]').all()]
    errs += [e.inner_text()[:300] for e in page.locator('[data-testid="stAlertContentError"]').all()]
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--widths", default="1366,1920")
    ap.add_argument("--themes", default="light,dark")
    ap.add_argument("--pages", default=",".join(PAGES))
    ap.add_argument("--flow", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from fairlens.config import load_config

    pw_path = load_config()["paths"].get("playwright_browsers")  # optional, per machine (configs/local.yaml)
    if pw_path:
        os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(pw_path))
    from playwright.sync_api import sync_playwright

    proc = None
    url = a.url
    if not url:
        url = "http://localhost:8599"
        proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
                                 "--server.port", "8599"], cwd=ROOT, stdout=open(OUT.parent / "ui_smoke_server.log", "w", encoding="utf-8"),
                                stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUTF8": "1"})
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"url": url, "pages": {}, "flow": None}
    try:
        wait_up(url)
        with sync_playwright() as pw:
            br = pw.chromium.launch()
            for width in [int(w) for w in a.widths.split(",")]:
                ctx = br.new_context(viewport={"width": width, "height": 900})
                page = ctx.new_page()
                console = []
                page.on("console", lambda m: console.append(m.text) if m.type == "error" else None)
                page.on("pageerror", lambda e: console.append(str(e)))
                for theme in a.themes.split(","):
                    for p in a.pages.split(","):
                        console.clear()
                        page.goto(f"{url}/{p}")
                        settle(page)
                        cur_dark = page.locator('button[title="Switch to light mode"], [data-testid="stTooltipHoverTarget"] button').count() > 0
                        want_dark = theme == "dark"
                        btn = page.get_by_role("button", name="").filter(has=page.locator('[data-testid="stIconMaterial"]'))
                        # the theme toggle is the first icon-only button in the main area
                        tog = page.locator('.st-key-theme_toggle button')
                        if tog.count():
                            is_dark = page.evaluate("getComputedStyle(document.querySelector('.stApp')).backgroundColor") == "rgb(15, 16, 32)"
                            if is_dark != want_dark:
                                tog.first.click()
                                settle(page)
                        shot = OUT / f"{p or 'home'}_{theme}_{width}.png"
                        page.screenshot(path=str(shot), full_page=True)
                        errs = errors_on(page)
                        overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 2")
                        key = f"{p}/{theme}/{width}"
                        # Streamlit's frontend probes <deep-link>/_stcore/{health,host-config} before falling back to the
                        # root when a page URL is opened directly; these 404s come from Streamlit, not from FairLens.
                        known = [c for c in console if "404" in c]
                        app_console = [c for c in console if "404" not in c and "favicon" not in c]
                        report["pages"][key] = {"errors": errs, "console": app_console[:5], "streamlit_deeplink_404s": len(known),
                                                "horizontal_overflow": overflow, "screenshot": str(shot.relative_to(ROOT))}
                        flag = "OK " if not errs and not overflow else "ERR"
                        print(f"{flag} {key} errors={len(errs)} overflow={overflow} console={len(app_console)} deeplink404={len(known)}", flush=True)
                        for e in errs:
                            print("    ", e.replace("\n", " ")[:300])
                ctx.close()
            if a.flow:
                report["flow"] = flow(br, url)
            br.close()
    finally:
        if proc:
            proc.terminate()
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    bad = [k for k, v in report["pages"].items() if v["errors"] or v["horizontal_overflow"]]
    print(f"pages checked: {len(report['pages'])}, with problems: {len(bad)}")


def flow(br, url) -> dict:
    """Drive the main workflow end to end with real models."""
    res = {}
    ctx = br.new_context(viewport={"width": 1366, "height": 900}, accept_downloads=True)
    page = ctx.new_page()
    page.goto(f"{url}/text")
    settle(page)
    page.get_by_role("button", name="Job description").click()
    settle(page)
    page.get_by_role("button", name="Analyze & mitigate").click()
    settle(page, 900_000)
    res["after_mitigate_url"] = page.url
    page.screenshot(path=str(OUT / "flow_1_mitigation.png"), full_page=True)
    res["mitigation_errors"] = errors_on(page)
    res["cards"] = page.locator(".fl-card h4").all_inner_texts()[:8]
    btns = page.get_by_role("button", name="Use this version")
    res["use_buttons"] = btns.count()
    if btns.count() > 1:
        btns.nth(btns.count() - 1).click()
        settle(page)
    res["gate_url"] = page.url
    res["banner"] = page.locator(".fl-banner .ttl").all_inner_texts()
    page.screenshot(path=str(OUT / "flow_2_gate.png"), full_page=True)
    page.get_by_text("Compare with lowest-bias-score selection").click()
    settle(page)
    page.screenshot(path=str(OUT / "flow_3_gate_compare.png"), full_page=True)
    page.get_by_role("tab", name="Detailed Analysis").click()
    time.sleep(1)
    res["checks"] = page.locator(".fl-check .n").all_inner_texts()
    page.screenshot(path=str(OUT / "flow_4_gate_details.png"), full_page=True)
    res["gate_errors"] = errors_on(page)
    page.goto(f"{url}/text")
    settle(page)
    downloads = {}
    for label in ["JSON", "HTML report", "PDF report"]:
        b = page.get_by_role("button", name=label)
        if b.count():
            with page.expect_download() as d:
                b.first.click()
            p = OUT / f"export_{d.value.suggested_filename}"
            d.value.save_as(p)
            downloads[label] = p.stat().st_size
    res["downloads"] = downloads
    print("flow:", json.dumps(res, indent=1)[:2000], flush=True)
    ctx.close()
    return res


if __name__ == "__main__":
    main()
