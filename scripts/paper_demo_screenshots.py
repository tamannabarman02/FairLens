"""Drive headless Chrome over CDP against the running FairLens app and take full-page screenshots.
usage: python drive.py <outdir> <step...>   steps: benchmarks | demo
"""
import base64, json, subprocess, sys, time, urllib.request, os
import websocket

URL = "http://127.0.0.1:8599/"
OUT = os.path.abspath(sys.argv[1])
os.makedirs(OUT, exist_ok=True)
PORT = 9333
CHROME = os.environ.get("CHROME_PATH") or os.path.join(
    os.environ.get("PROGRAMFILES", ""), "Google", "Chrome", "Application", "chrome.exe")
W, H = 1440, 900

prof = os.path.join(OUT, "_profile")
proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={prof}",
                         f"--window-size={W},{H}", "--hide-scrollbars", "--force-device-scale-factor=1.5", "about:blank"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(120):
    try:
        tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
        page = [t for t in tabs if t["type"] == "page"][0]
        break
    except Exception:
        time.sleep(0.3)
ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=600)
_id = [0]


def cdp(method, **params):
    _id[0] += 1
    ws.send(json.dumps({"id": _id[0], "method": method, "params": params}))
    while True:
        m = json.loads(ws.recv())
        if m.get("id") == _id[0]:
            if "error" in m:
                raise RuntimeError(m["error"])
            return m.get("result", {})


def js(expr):
    r = cdp("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
    return r.get("result", {}).get("value")


def text():
    return js("document.body.innerText") or ""


def wait_for(substr, timeout=900, gone=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        t = text()
        if substr in t and (gone is None or gone not in t) and "Running FairLens pipeline" not in t:
            time.sleep(2)
            return True
        time.sleep(1.5)
    raise TimeoutError(substr)


def idle(extra=2.5):
    # wait while Streamlit shows its running indicator
    for _ in range(600):
        if not js("!!document.querySelector('[data-testid=\"stStatusWidget\"]')"):
            break
        time.sleep(0.5)
    time.sleep(extra)


def click_text(selector, label, exact=True):
    ok = js(f"""(() => {{
      const els=[...document.querySelectorAll({json.dumps(selector)})];
      const e=els.find(x => {'x.innerText.trim()===' if exact else 'x.innerText.includes('}{json.dumps(label)}{'' if exact else ')'});
      if(!e) return false; e.scrollIntoView(); e.click(); return true; }})()""")
    if not ok:
        raise RuntimeError(f"not found: {selector} {label}")


def nav(page_title):
    click_text('[data-testid="stSidebar"] a', page_title, exact=False)
    time.sleep(1.5)
    idle()


def set_textarea(value):
    js("(() => { const ta=document.querySelector('textarea'); ta.scrollIntoView(); ta.focus(); ta.select(); })()")
    time.sleep(0.5)
    cdp("Input.insertText", text=value)
    time.sleep(0.5)
    for t in ("keyDown", "keyUp"):
        cdp("Input.dispatchKeyEvent", type=t, key="Enter", code="Enter", modifiers=2, windowsVirtualKeyCode=13)
    time.sleep(2)
    idle()
    n = js("document.querySelector('textarea').value.length")
    print("textarea chars:", n, flush=True)
    if not n:
        raise RuntimeError("text was not entered")


def prep():
    js("""(() => { const sb=document.querySelector('[data-testid="stSidebar"]');
              if (sb && sb.getBoundingClientRect().width > 100) return;
              const c=document.querySelector('[data-testid="stExpandSidebarButton"]');
              const b=c && (c.tagName==='BUTTON' ? c : c.querySelector('button')) || c; if (b) b.click(); })()""")
    time.sleep(1.5)
    js("""(() => { const w=document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT); let n;
              while((n=w.nextNode())) if(n.nodeValue.includes('/Users/')) n.nodeValue=n.nodeValue.replace(/\/Users\/[^\/]+\/Downloads\/FairLens-main\//g,''); })()""")
    time.sleep(0.5)


def shot(name, sidebar=True):
    prep()
    # expand viewport to the full height of Streamlit's inner scroll container
    h = js("""(() => { const m=document.querySelector('[data-testid="stMain"]')||document.querySelector('section.main');
              const b=document.querySelector('[data-testid="stMainBlockContainer"]')||m;
              return Math.ceil(Math.max(m.scrollHeight, b.scrollHeight)) + 40; })()""")
    h = max(H, int(h))
    cdp("Emulation.setDeviceMetricsOverride", width=W, height=h, deviceScaleFactor=1.5, mobile=False)
    time.sleep(2.5)
    data = cdp("Page.captureScreenshot", format="png", captureBeyondViewport=True)["data"]
    with open(os.path.join(OUT, name + ".png"), "wb") as f:
        f.write(base64.b64decode(data))
    cdp("Emulation.clearDeviceMetricsOverride")
    time.sleep(1)
    print("saved", name, "h=", h, flush=True)


cdp("Page.enable"); cdp("Runtime.enable")
cdp("Page.navigate", url=URL)
wait_for("Workflow".upper() if False else "FairLens", 120)
idle(4)

steps = sys.argv[2:]
if "benchmarks" in steps:
    nav("Benchmarks")
    wait_for("Benchmarks & Results", 120); idle(5)
    shot("benchmarks_full")

if "demo" in steps:
    DEMO = os.environ.get("DEMO_TEXT")
    nav("Text Analysis")
    wait_for("Text to analyse", 120)
    set_textarea(DEMO)
    click_text("button", "Analyze & mitigate", exact=False)
    wait_for("Gate recommendation", 1200)
    idle(4)
    shot("demo_mitigation")
    with open(os.path.join(OUT, "demo_mitigation.txt"), "w", encoding="utf-8") as f:
        f.write(text())
    nav("Text Analysis")
    wait_for("Overall bias score", 120); idle(3)
    shot("demo_text_analysis")
    with open(os.path.join(OUT, "demo_text_analysis.txt"), "w", encoding="utf-8") as f:
        f.write(text())
    nav("Reliability Gate")
    idle(4)
    shot("demo_gate")
    with open(os.path.join(OUT, "demo_gate.txt"), "w", encoding="utf-8") as f:
        f.write(text())

ws.close(); proc.terminate()
