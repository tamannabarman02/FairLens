"""Autopilot for a demo video: drives the running FairLens app in Chrome through text and image analysis.

Start the app first (VS Code terminal, project folder):
    .venv\\Scripts\\streamlit run app.py
Then, in a second terminal (needs the `websocket-client` package, e.g. Anaconda's Python):
    python scripts/demo_autopilot.py                 # visible Chrome window: record it with Win + Shift + R
    python scripts/demo_autopilot.py --headless out  # check run: screenshots of every step saved to ./out
    python scripts/demo_autopilot.py --record demo.mp4   # records the run itself and writes an MP4 (needs PowerPoint)

Each step shows a caption bar at the bottom of the page, so the video explains itself.
"""
import argparse
import atexit
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket

DEMO_TEXT = ("We are looking for a young, energetic waitress to join our Bengaluru cafe. "
             "She must be available 6 days a week and have 2 years of experience.")

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8501/")
ap.add_argument("--headless", metavar="OUTDIR", help="no window; save a screenshot after each step to OUTDIR")
ap.add_argument("--image", default="474344", help="COCO image id for the image step (default: man with bat, woman)")
ap.add_argument("--speed", type=float, default=1.0, help="pause multiplier (2 = slower video)")
ap.add_argument("--record", metavar="MP4", help="record the (headless) run and save it as an MP4 video via PowerPoint")
ap.add_argument("--max-wait", type=float, default=6.0, help="in the video, model-loading waits are shortened to this many s")
args = ap.parse_args()
if args.record and not args.headless:
    args.headless = tempfile.mkdtemp(prefix="fairlens_demo_shots_")

CHROME = os.environ.get("CHROME_PATH") or os.path.join(os.environ.get("PROGRAMFILES", ""), "Google", "Chrome",
                                                        "Application", "chrome.exe")
PORT = 9334
W, H = 1440, 900
OUT = Path(args.headless).resolve() if args.headless else None
if OUT:
    OUT.mkdir(parents=True, exist_ok=True)

REC = bool(args.record)
FRAME_DIR = Path(tempfile.mkdtemp(prefix="fairlens_demo_frames_")) if REC else None
FRAMES: list[tuple[float, Path]] = []  # (capture time, jpeg)
WAITS: list[tuple[float, float]] = []  # model-loading waits, shortened in the video
ws = None


def on_event(m):
    """Store a screencast frame and acknowledge it so Chrome keeps sending frames."""
    if m.get("method") != "Page.screencastFrame":
        return
    p = m["params"]
    path = FRAME_DIR / f"{len(FRAMES):05d}.jpg"
    path.write_bytes(base64.b64decode(p["data"]))
    FRAMES.append((p["metadata"]["timestamp"], path))
    _id[0] += 1
    ws.send(json.dumps({"id": _id[0], "method": "Page.screencastFrameAck", "params": {"sessionId": p["sessionId"]}}))


def sleep(seconds):
    """time.sleep that keeps reading screencast frames while recording."""
    if not (REC and ws):
        time.sleep(seconds)
        return
    end = time.time() + seconds
    while (left := end - time.time()) > 0:
        ws.settimeout(left)
        try:
            on_event(json.loads(ws.recv()))
        except websocket.WebSocketTimeoutException:
            break
    ws.settimeout(1800)


flags = [CHROME, f"--remote-debugging-port={PORT}", f"--user-data-dir={tempfile.mkdtemp(prefix='fairlens_demo_')}",
         "--no-first-run", "--no-default-browser-check", f"--window-size={W},{H}"]
flags += ["--headless=new", "--hide-scrollbars"] if OUT else ["--start-maximized", f"--app={args.url}"]
proc = subprocess.Popen(flags + ([] if not OUT else ["about:blank"]), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
if OUT:
    atexit.register(proc.kill)  # never leave a headless Chrome behind, even after an error
for _ in range(100):
    try:
        tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
        page = [t for t in tabs if t["type"] == "page"][0]
        break
    except Exception:  # noqa: BLE001
        time.sleep(0.3)
ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=1800)
_id = [0]


def cdp(method, **params):
    _id[0] += 1
    my_id = _id[0]
    ws.send(json.dumps({"id": my_id, "method": method, "params": params}))
    while True:
        m = json.loads(ws.recv())
        if m.get("id") != my_id:  # events (screencast frames) and replies to frame acks
            on_event(m)
            continue
        if "error" in m:
            raise RuntimeError(m["error"])
        return m.get("result", {})


def js(expr):
    return cdp("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True).get("result", {}).get("value")


def pause(s):
    sleep(s * args.speed)


def text():
    return js("document.body.innerText") or ""


def idle(extra=1.5):
    for _ in range(3600):
        if not js("!!document.querySelector('[data-testid=\"stStatusWidget\"]')"):
            break
        sleep(0.5)
    sleep(extra)


def wait_for(substr, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if substr in text():
            idle()
            return
        sleep(1)
    raise TimeoutError(f"'{substr}' did not appear within {timeout} s")


def caption(msg):
    """Bottom caption bar that stays on screen across Streamlit reruns and page changes."""
    js(f"""(() => {{ let d=document.getElementById('fl-demo-cap');
        if(!d){{ d=document.createElement('div'); d.id='fl-demo-cap'; document.body.appendChild(d);
          Object.assign(d.style,{{position:'fixed',left:'50%',bottom:'28px',transform:'translateX(-50%)',zIndex:999999,
            background:'rgba(17,24,39,0.92)',color:'#fff',padding:'14px 26px',borderRadius:'12px',maxWidth:'80vw',
            font:'600 20px/1.4 Segoe UI, Arial, sans-serif',boxShadow:'0 6px 24px rgba(0,0,0,.3)',textAlign:'center'}}); }}
        d.textContent={json.dumps(msg)}; }})()""")


def click(selector, label, exact=False):
    ok = js(f"""(() => {{ const els=[...document.querySelectorAll({json.dumps(selector)})];
        const e=els.find(x => {'x.innerText.trim()===' if exact else 'x.innerText.includes('}{json.dumps(label)}{'' if exact else ')'});
        if(!e) return false; e.scrollIntoView({{behavior:'smooth',block:'center'}}); setTimeout(()=>e.click(),600); return true; }})()""")
    if not ok:
        raise RuntimeError(f"not found on page: {label}")
    sleep(1.2)


def nav(title):
    js("""(() => { const sb=document.querySelector('[data-testid="stSidebar"]');
        if (sb && sb.getBoundingClientRect().width > 100) return;
        const c=document.querySelector('[data-testid="stExpandSidebarButton"]'); const b=c && (c.querySelector('button')||c);
        if (b) b.click(); })()""")
    sleep(0.8)
    click('[data-testid="stSidebar"] a', title)
    idle(2)
    scroll_top()


def scroller():
    return """(() => { const c=[document.querySelector('[data-testid="stMain"]'), document.querySelector('section.main'),
        document.scrollingElement]; return c.find(e => e && e.scrollHeight > e.clientHeight + 5) || document.scrollingElement; })()"""


def scroll_top():
    js(f"{scroller()}.scrollTo({{top:0,behavior:'smooth'}})")
    sleep(1)


def scroll_through(seconds=12, stop_at=None):
    """Scroll smoothly from the current position to the bottom (or to the element containing `stop_at`)."""
    total = js(f"(() => {{ const s={scroller()}; return s.scrollHeight - s.clientHeight - s.scrollTop; }})()") or 0
    if stop_at:
        y = js(f"""(() => {{ const s={scroller()}; const w=document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT); let n;
            while((n=w.nextNode())) if(n.nodeValue.includes({json.dumps(stop_at)}))
              return n.parentElement.getBoundingClientRect().top - 120; return null; }})()""")
        if y is not None:
            total = min(total, max(0, y))
    steps = max(1, int(seconds * args.speed * 10))
    for _ in range(steps):
        js(f"{scroller()}.scrollBy(0, {total / steps})")
        sleep(0.1)


def type_text(value):
    js("(() => { const ta=document.querySelector('textarea'); ta.scrollIntoView({block:'center'}); ta.focus(); ta.select(); })()")
    sleep(0.5)
    words = value.split(" ")
    for i, w in enumerate(words):
        cdp("Input.insertText", text=w + (" " if i < len(words) - 1 else ""))
        sleep(0.12 * args.speed)
    for t in ("keyDown", "keyUp"):  # Ctrl+Enter commits the text area
        cdp("Input.dispatchKeyEvent", type=t, key="Enter", code="Enter", modifiers=2, windowsVirtualKeyCode=13)
    idle(1)


SHOT = [0]


def shot(name):
    if not OUT:
        return
    SHOT[0] += 1
    data = cdp("Page.captureScreenshot", format="png")["data"]
    (OUT / f"{SHOT[0]:02d}_{name}.png").write_bytes(base64.b64decode(data))
    (OUT / f"{SHOT[0]:02d}_{name}.txt").write_text(text(), encoding="utf-8")
    print("step", SHOT[0], name, flush=True)


# --------------------------------------------------------------------------- the demo
cdp("Page.enable")
cdp("Runtime.enable")
if OUT:
    cdp("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
    cdp("Page.navigate", url=args.url)
wait_for("FairLens", 180)
idle(3)
if REC:  # start once the page is loaded (not while it is navigating)
    cdp("Page.startScreencast", format="jpeg", quality=88, maxWidth=W, maxHeight=H, everyNthFrame=1)
caption("FairLens: bias detection, mitigation and a Reliability Gate (running locally on this laptop)")
shot("home")
pause(5)

# 1. text analysis
nav("Text Analysis")
wait_for("Text to analyse", 180)
caption("Step 1 · Text analysis: we paste a job advert that contains biased wording")
pause(2)
type_text(DEMO_TEXT)
shot("text_input")
pause(2)
caption("4 detectors score the text, 3 rewriters propose neutral versions, and the Reliability Gate checks each one")
click("button", "Analyze & mitigate")
if REC:
    caption("Running 4 detectors, 3 rewriters and the Gate on the laptop CPU (waiting time shortened in this video)")
w0 = time.time()
wait_for("Gate recommendation", 1800)
WAITS.append((w0, time.time()))
nav("Text Analysis")
wait_for("Overall bias score", 300)
caption("Detection: overall bias score, flagged phrases (\"young\", \"waitress\") and each detector's score")
shot("text_result")
pause(3)
scroll_through(14)
shot("text_result_scrolled")
pause(3)

# 2. mitigation and the Gate's verdicts
nav("Mitigation")
caption("Step 2 · Mitigation: the original beside the rule-based, FLAN-T5 and Qwen2.5 rewrites, each with the Gate's verdict")
shot("mitigation")
pause(4)
scroll_through(14)
shot("mitigation_scrolled")
pause(3)

# 3. Reliability Gate
nav("Reliability Gate")
caption("Step 3 · Reliability Gate: the accepted rewrite keeps every fact (6 days, 2 years, Bengaluru) and removes the bias")
shot("gate")
pause(4)
scroll_through(12)
shot("gate_scrolled")
pause(3)

# 4. image analysis
nav("Image Analysis")
wait_for("Image source", 180)
caption("Step 4 · Image analysis: two captioners describe a photo; the captions go through the same bias detectors")
pause(2)
click('[data-testid="stRadio"] label', "COCO sample")
idle(2)
js("""(() => { const s=document.querySelector('[data-testid="stSelectbox"] input');
    s.scrollIntoView({block:'center'}); s.focus(); s.select(); })()""")
sleep(1.2)
cdp("Input.insertText", text=args.image)  # the sample list is long: type the image id to filter it
sleep(1.5)
picked = js(f"""(() => {{ const o=[...document.querySelectorAll('[role="option"]')].find(x => x.innerText.includes({json.dumps(args.image)}));
    if (!o) return false; o.click(); return true; }})()""")
if not picked:
    for t in ("keyDown", "keyUp"):
        cdp("Input.dispatchKeyEvent", type=t, key="Enter", code="Enter", windowsVirtualKeyCode=13)
idle(2)
if args.image not in (js("""document.querySelector('[data-testid="stSelectbox"] input').value""") or ""):
    raise RuntimeError(f"COCO sample {args.image} was not selected")
shot("image_selected")
pause(2)
caption("Generating captions with BLIP and Florence-2, checking image-caption match with CLIP, then detecting bias")
click("button", "Generate captions & analyse")
if REC:
    caption("BLIP, Florence-2 and CLIP running on the laptop CPU (waiting time shortened in this video)")
w0 = time.time()
wait_for("Captions ready", 1800)
WAITS.append((w0, time.time()))
idle(2)
caption("Each caption with its CLIP match score, gendered terms and bias scores")
shot("image_result")
pause(3)
scroll_through(14)
shot("image_result_scrolled")
pause(3)

scroll_top()
caption("FairLens: every rewrite is checked before it is accepted. Thank you!")
shot("end")
pause(5)
js("document.getElementById('fl-demo-cap')?.remove()")
if REC:
    cdp("Page.stopScreencast")
ws.close()
if OUT:
    proc.terminate()
print("demo finished", flush=True)


def build_video(frames, out: Path, fps=15):
    """Turn timed screencast frames into an MP4 with PowerPoint: one slide per frame, advancing on its duration."""
    import win32com.client

    times = [t for t, _ in frames] + [frames[-1][0] + 3.0]
    durs = [b - a for a, b in zip(times, times[1:])]
    for w0, w1 in WAITS:  # shorten long model-loading waits
        idx = [i for i, (t, _) in enumerate(frames) if w0 <= t < w1]
        total = sum(durs[i] for i in idx)
        if total > args.max_wait:
            for i in idx:
                durs[i] *= args.max_wait / total
    slides, acc = [], 0.0  # merge frames shorter than one video frame
    for (t, p), d in zip(frames, durs):
        acc += d
        if acc >= 1 / fps:
            slides.append((p, acc))
            acc = 0.0
    print(f"building video: {len(slides)} frames, {sum(d for _, d in slides):.0f} s", flush=True)
    app = win32com.client.DispatchEx("PowerPoint.Application")
    pres = app.Presentations.Add(False)
    pres.PageSetup.SlideWidth, pres.PageSetup.SlideHeight = W * 0.75, H * 0.75
    for i, (p, d) in enumerate(slides, 1):
        sl = pres.Slides.Add(i, 12)  # blank layout
        sl.Shapes.AddPicture(str(p), False, True, 0, 0, W * 0.75, H * 0.75)
        sl.SlideShowTransition.AdvanceOnTime = True
        sl.SlideShowTransition.AdvanceTime = round(d, 3)
    pres.CreateVideo(str(out), True, 3, H, fps, 90)
    while pres.CreateVideoStatus in (1, 2):  # queued / in progress
        time.sleep(2)
    ok = pres.CreateVideoStatus == 3
    pres.Close()
    app.Quit()
    if not ok:
        raise RuntimeError("PowerPoint could not create the video")
    print("saved video", out, flush=True)


if REC:
    build_video(FRAMES, Path(args.record).resolve())
