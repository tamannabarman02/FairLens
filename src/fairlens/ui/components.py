"""Reusable HTML components for the Streamlit dashboard (card, tile, badge, stepper, score bar, banner...).

All functions return HTML strings; `render(html)` writes them. CSS lives in assets/styles.css.
"""
from __future__ import annotations

import difflib
import html as _h
from pathlib import Path

import streamlit as st

from ..config import PROJECT_ROOT

ICONS = {  # lucide-style outline icons (24x24 viewBox)
    "home": '<path d="M3 10.5 12 3l9 7.5V21a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1z"/>',
    "text": '<path d="M4 6h16M4 12h16M4 18h10"/>',
    "image": '<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="9" cy="9" r="2"/><path d="m21 15-5-5L5 21"/>',
    "layers": '<path d="m12 2 10 6-10 6L2 8z"/><path d="m2 16 10 6 10-6"/>',
    "wand": '<path d="M15 4V2M15 16v-2M8 9h2M20 9h2M17.8 11.8 19 13M17.8 6.2 19 5M3 21l9-9M12.2 6.2 11 5"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>', "x": '<path d="M18 6 6 18M6 6l12 12"/>',
    "refresh": '<path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 3v5h-5"/>',
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21c1.5-4 5-6 8-6s6.5 2 8 6"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M7 15l4-4 3 3 6-6"/>', "flask": '<path d="M9 3h6M10 3v6L4 19a2 2 0 0 0 2 3h12a2 2 0 0 0 2-3l-6-10V3"/>',
    "lens": '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>',
    "eye": '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
    "alert": '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/>',
    "inbox": '<path d="M22 12h-6l-2 3h-4l-2-3H2"/><path d="M5.5 5.1 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.5-6.9A2 2 0 0 0 16.8 4H7.2a2 2 0 0 0-1.7 1.1z"/>',
}


def icon(name: str, size: int = 20, stroke: str = "currentColor", width: float = 2) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{stroke}" stroke-width="{width}" '
            f'stroke-linecap="round" stroke-linejoin="round">{ICONS.get(name, ICONS["info"])}</svg>')


def esc(x) -> str:
    return _h.escape(str(x))


def render(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


def inject_css(theme: str) -> None:
    css = (PROJECT_ROOT / "assets" / "styles.css").read_text(encoding="utf-8")
    # The theme toggle re-binds tokens by placing them on :root when dark.
    dark_block = css.split('[data-fl-theme="dark"] {', 1)[1].split("}", 1)[0]
    extra = f":root {{{dark_block}}}" if theme == "dark" else ""
    render(f"<style>{css}\n{extra}</style>")


# --- primitives -------------------------------------------------------------------------------
def tone_for_score(score: float | None) -> str:
    if score is None:
        return "grey"
    return "red" if score >= 0.66 else "amber" if score >= 0.4 else "green"


def level_for_score(score: float | None) -> str:
    if score is None:
        return "n/a"
    return "High" if score >= 0.66 else "Medium" if score >= 0.4 else "Low"


DECISION_TONE = {"ACCEPT": "green", "REJECT": "red", "RETRY": "amber", "HUMAN_REVIEW": "blue"}
DECISION_LABEL = {"ACCEPT": "Accept", "REJECT": "Reject", "RETRY": "Retry", "HUMAN_REVIEW": "Human review"}


def badge(text: str, tone: str = "accent") -> str:
    return f'<span class="fl-badge {tone}">{esc(text)}</span>'


def decision_badge(decision: str) -> str:
    return badge(DECISION_LABEL.get(decision, decision), DECISION_TONE.get(decision, "grey"))


def help_dot(text: str) -> str:
    return f'<span class="fl-help" title="{esc(text)}">?</span>'


def page_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    render(f'<div class="fl-header">{f"<div class=fl-eyebrow>{esc(eyebrow)}</div>" if eyebrow else ""}'
           f'<div class="fl-title">{esc(title)}</div>{f"<div class=fl-sub>{subtitle}</div>" if subtitle else ""}</div>')


def card(title: str, body: str, tone: str = "", icon_name: str | None = None, extra: str = "") -> str:
    ic = f'<div class="fl-icon">{icon(icon_name)}</div>' if icon_name else ""
    return f'<div class="fl-card {f"tone-{tone}" if tone else ""}">{ic}<h4>{esc(title)}</h4><p>{body}</p>{extra}</div>'


def metric_tile(label: str, value: str, sub: str = "", tone: str = "", tip: str = "") -> str:
    return (f'<div class="fl-tile {f"tone-{tone}" if tone else ""}"><div class="lbl">{esc(label)} {help_dot(tip) if tip else ""}</div>'
            f'<div class="val">{esc(value)}</div>{f"<div class=sub>{sub}</div>" if sub else ""}</div>')


def stepper(steps: list[str], active: int) -> str:
    """active = index of the current step; steps before it are done; len(steps) = all done."""
    parts = []
    for i, s in enumerate(steps):
        cls = "done" if i < active else "active" if i == active else ""
        dot = "&#10003;" if i < active else str(i + 1)
        parts.append(f'<div class="fl-step {cls}"><div class="dot">{dot}</div><div class="name">{esc(s)}</div></div>')
        if i < len(steps) - 1:
            parts.append(f'<div class="fl-step-line {"done" if i < active else ""}"></div>')
    return f'<div class="fl-stepper">{"".join(parts)}</div>'


def score_bar(label: str, score: float | None, right: str = "", tone: str | None = None, sub: str = "") -> str:
    tone = tone or tone_for_score(score)
    pct = 0 if score is None else max(0.0, min(1.0, score)) * 100
    val = "n/a" if score is None else f"{score:.2f}"
    return (f'<div class="fl-bar"><div class="row"><span>{esc(label)}</span><span class="meta">{right}<b>{val}</b></span></div>'
            f'<div class="track"><div class="fill {tone}" style="width:{pct:.1f}%"></div></div>'
            f'{f"<div class=fl-sub style=font-size:12px;margin-top:4px>{sub}</div>" if sub else ""}</div>')


def decision_banner(decision: str, reason: str, sub: str = "") -> str:
    ic = {"ACCEPT": "check", "REJECT": "x", "RETRY": "refresh", "HUMAN_REVIEW": "user"}.get(decision, "info")
    return (f'<div class="fl-banner {decision}"><div class="icon">{icon(ic, 24, "#fff", 2.5)}</div><div>'
            f'<div class="ttl">{esc(DECISION_LABEL.get(decision, decision))}</div><div class="why">{esc(reason)}</div>'
            f'{f"<div class=fl-sub style=font-size:12.5px;margin-top:4px>{sub}</div>" if sub else ""}</div></div>')


def pipeline_cards(steps: list[tuple[str, str, str]]) -> str:
    """steps: (icon, title, description)."""
    out = []
    for i, (ic, t, d) in enumerate(steps):
        out.append(f'<div class="fl-pipe"><div class="fl-icon" style="margin:0">{icon(ic)}</div><div class="t">{esc(t)}</div><div class="d">{esc(d)}</div></div>')
        if i < len(steps) - 1:
            out.append('<div class="fl-arrow">&rarr;</div>')
    return f'<div class="fl-pipeline">{"".join(out)}</div>'


def empty_state(title: str, text: str, icon_name: str = "inbox") -> str:
    return (f'<div class="fl-empty"><div class="fl-icon" style="margin:0 auto">{icon(icon_name)}</div>'
            f'<div class="ttl">{esc(title)}</div><div class="txt">{text}</div></div>')


def skeleton(lines: int = 4) -> str:
    return "".join(f'<div class="fl-skel" style="width:{90 - 12 * (i % 3)}%"></div>' for i in range(lines))


def highlighted_text(text: str, spans: list[dict]) -> str:
    """Wrap span offsets in <mark> with tooltip explanations; overlapping spans are merged left-to-right."""
    spans = sorted((s for s in spans if 0 <= s["start"] < s["end"] <= len(text)), key=lambda s: (s["start"], -s["end"]))
    out, pos = [], 0
    for s in spans:
        if s["start"] < pos:
            continue
        out.append(esc(text[pos:s["start"]]))
        w = s.get("weight", 0.5)
        cls = "h-high" if w >= 0.5 else "h-med" if w >= 0.25 else "h-low"
        tip = f'{s.get("category", "").replace("_", " ").title()}: {s.get("explanation", "")}'
        out.append(f'<mark class="{cls}" title="{esc(tip)}">{esc(text[s["start"]:s["end"]])}</mark>')
        pos = s["end"]
    out.append(esc(text[pos:]))
    return f'<div class="fl-text">{"".join(out)}</div>'


def diff_html(a: str, b: str) -> str:
    aw, bw = a.split(), b.split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=aw, b=bw, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(esc(" ".join(aw[i1:i2])))
        if op in ("delete", "replace"):
            out.append(f"<del>{esc(' '.join(aw[i1:i2]))}</del>")
        if op in ("insert", "replace"):
            out.append(f"<ins>{esc(' '.join(bw[j1:j2]))}</ins>")
    return f'<div class="fl-text fl-diff">{" ".join(out)}</div>'


def check_grid(items: list[tuple[str, bool | None, str]]) -> str:
    cells = []
    for name, ok, detail in items:
        cls = "na" if ok is None else "pass" if ok else "fail"
        sym = "&ndash;" if ok is None else "&#10003;" if ok else "&#10007;"
        cells.append(f'<div class="fl-check {cls}"><div class="ic">{sym}</div><div><div class="n">{esc(name)}</div>'
                     f'<div class="x">{esc(detail)}</div></div></div>')
    return f'<div class="fl-checks">{"".join(cells)}</div>'


def html_table(rows: list[dict], columns: list[str] | None = None, best: dict[str, str] | None = None,
               fmt: dict[str, str] | None = None, raw: set[str] | None = None) -> str:
    """best: {column: 'max'|'min'} highlights the best value; raw: columns already containing HTML."""
    if not rows:
        return ""
    columns = columns or list(rows[0])
    fmt, raw, best = fmt or {}, raw or set(), best or {}
    bests = {}
    for c, how in best.items():
        vals = [r.get(c) for r in rows if isinstance(r.get(c), (int, float))]
        if vals:
            bests[c] = max(vals) if how == "max" else min(vals)
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = []
    for r in rows:
        tds = []
        for c in columns:
            v = r.get(c)
            if c in raw:
                s = v if v is not None else ""
            elif isinstance(v, float):
                s = esc(format(v, fmt.get(c, ".3f")))
            else:
                s = esc("" if v is None else v)
            cls = ' class="best"' if c in bests and v == bests[c] and len(rows) > 1 else ""
            tds.append(f"<td{cls}>{s}</td>")
        body.append(f"<tr>{''.join(tds)}</tr>")
    return f'<div class="fl-table-wrap"><table class="fl-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def note(text: str, kind: str = "note") -> str:
    return f'<div class="fl-{kind}">{text}</div>'


def kv(pairs: list[tuple[str, str]]) -> str:
    return '<div class="fl-kv">' + "".join(f'<div class="k">{esc(k)}</div><div class="v">{esc(v)}</div>' for k, v in pairs) + "</div>"


RESULT_LABELS = {"completed": ("Completed run", "green"), "pending": ("Pending", "grey"), "failed": ("Failed run", "red"),
                 "illustrative": ("Illustrative example", "amber")}


def result_label(kind: str) -> str:
    """One of the four V4 result labels: completed, pending, failed, illustrative."""
    text, tone = RESULT_LABELS[kind]
    return badge(text, tone)


def provenance(summary: dict, manifest: dict | None = None) -> str:
    """'Completed run' label plus run ID, models (with revisions), dataset hashes and device, from the run files."""
    parts = [result_label("completed"), f'<span class="fl-mono">run {esc(summary.get("run_id", "?"))}</span>']
    if manifest:
        models = manifest.get("models") or {}
        if models:
            parts.append("models: " + esc(", ".join(f"{k}={v}" for k, v in list(models.items())[:6])))
        data = manifest.get("datasets") or {}
        if data:
            parts.append("data: " + esc(", ".join(f"{k.rsplit('/', 1)[-1]}#{v[:8]}" for k, v in data.items())))
        dev = manifest.get("device") or (manifest.get("hardware") or {}).get("device")
        if dev:
            parts.append(f"device: {esc(dev)}")
        if manifest.get("git_commit"):
            parts.append(f"git {esc(manifest['git_commit'])}")
    return '<div class="fl-sub" style="font-size:12px;margin:2px 0 10px;display:flex;flex-wrap:wrap;gap:8px;align-items:center">' + \
        " · ".join(parts) + "</div>"
