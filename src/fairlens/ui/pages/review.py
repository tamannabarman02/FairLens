"""FairLens Review: a guided 6-step job-description workflow built on the existing pipeline, Gate and exports."""
import time

import streamlit as st

from ...ingestion.loaders import MAX_CHARS
from ...reliability_gate.rules import REASONS
from ...reporting import export
from .. import components as C
from ..common import examples, run_analysis, sentence_cap_note, show_error, stub_banner, text_uploader
from ..review_logic import entailment_signal, flag_evidence, preservation_items, reason_rows
from ..state import gate_config, run_mitigation
from .gate import TIPS
from .mitigation import name_of

STEPS = ["Input", "Flagged phrases", "Rewrites", "Fact checks", "Gate verdict", "Export"]
DISCLAIMER = "Review assistant, not a legal or compliance certification tool."


def _ready() -> bool:
    a, units = st.session_state.analysis, st.session_state.get("review_units")
    return bool(a and units is not None and st.session_state.review_source == a["text"])


def review_units(a: dict, threshold: float) -> list[dict]:
    """Sentences the Gate should check: those with a rule-flagged phrase or a model mean score at or above the threshold."""
    out = []
    for s in a["sentences"]:
        scores = [v for v in (s.get("scores") or {}).values() if v is not None]
        mean = sum(scores) / len(scores) if scores else None
        flagged = any(s["start"] <= sp["start"] < s["end"] for sp in a["spans"])
        if flagged or (mean is not None and mean >= threshold):
            out.append({"text": " ".join(s["text"].split()), "score": mean, "rule_flagged": flagged})
    return out


def _run_units(units: list[dict]) -> bool:
    """Run the rewriters and the Gate on each review sentence, the unit the Gate was evaluated on."""
    with st.status(f"Checking {len(units)} flagged sentence(s) with the rewriters and the Gate...", expanded=True) as status:
        try:
            for i, u in enumerate(units, 1):
                status.write(f"Sentence {i} of {len(units)}: {u['text'][:80]}")
                u["mitigation"] = run_mitigation(u["text"])
            status.update(label="Review complete", state="complete", expanded=False)
            return True
        except Exception as exc:  # noqa: BLE001
            status.update(label="Review failed", state="error", expanded=True)
            show_error(exc)
            return False


def _unit_label(i: int, u: dict) -> str:
    d = u["mitigation"]["decision"]
    return f"Sentence {i + 1} · {C.DECISION_LABEL.get(d['decision'], d['decision'])} · {u['text'][:70]}"


def _pick_unit() -> dict | None:
    """Sentence selector shared by steps 3–5; returns the chosen unit."""
    units = st.session_state.review_units
    if not units:
        C.render(C.empty_state("Nothing for the Gate to check", "No sentence had a rule-flagged phrase or a model score at or "
                               "above the threshold, so no rewrite was attempted.", "check"))
        return None
    idx = min(st.session_state.get("review_unit", 0), len(units) - 1)
    if len(units) > 1:
        idx = st.selectbox("Sentence to review", range(len(units)), index=idx, format_func=lambda i: _unit_label(i, units[i]),
                           key="rv_unit_widget")
    if idx != st.session_state.get("review_unit"):
        st.session_state.review_choice = None
    st.session_state.review_unit = idx
    return units[idx]


def _goto(step: int) -> None:
    st.session_state.review_step = step


def _nav(step: int) -> None:
    C.gap()
    a, b, _ = st.columns([1, 1.2, 4])
    if step > 0:
        a.button("Back", icon=":material/arrow_back:", key=f"rv_back_{step}", on_click=_goto, args=(step - 1,),
                 use_container_width=True)
    if step < len(STEPS) - 1:
        b.button(f"Next: {STEPS[step + 1]}", type="primary", key=f"rv_next_{step}", on_click=_goto, args=(step + 1,),
                 use_container_width=True)


def _set_text(text: str) -> None:
    st.session_state.rv_text = text


def _keep_choice() -> None:
    st.session_state.review_choice = st.session_state.get("rv_choice_widget")


def _selected(m: dict) -> str:
    cands = {c["candidate"]: c for c in m["candidates"]}
    sel = st.session_state.get("review_choice")
    if sel in cands:
        return sel
    d = m["decision"]
    return d.get("chosen") or next((k for k, c in cands.items() if c["changed"]), "original")


def step_input() -> None:
    C.section("1 · Add the job description")
    st.session_state.setdefault("rv_text", st.session_state.get("review_source") or "")
    t_paste, t_upload = st.tabs(["Paste text", "Upload file"])
    with t_paste:
        st.button("Use the sample job description", key="rv_sample", icon=":material/description:",
                  on_click=_set_text, args=(examples().get("Job description", ""),))
        text = st.text_area("Job description text", key="rv_text", height=180, max_chars=MAX_CHARS,
                            placeholder="Paste the full job advert here...")
        C.render(C.caption(f"{len(text):,} / {MAX_CHARS:,} characters"))
    with t_upload:
        doc = text_uploader("rv_upload", "Upload the job description")
        if doc is not None:
            st.button("Use this document", key="rv_use_doc", type="primary", on_click=_set_text, args=(doc.text,))
    text = st.session_state.get("rv_text", "")
    if text.strip():
        sentence_cap_note(text)
    C.gap()
    lite = " in Lite mode" if st.session_state.lite else ""
    if st.button("Run review", type="primary", icon=":material/play_arrow:", key="rv_run", disabled=not text.strip(),
                 help=f"Runs the detectors, the rewriters and the Reliability Gate{lite}. The first run loads models and can take a minute on CPU."):
        st.session_state.text_input = text
        st.session_state.review_choice = None
        st.session_state.review_unit = 0
        st.session_state.review_units = None
        if run_analysis(text, with_mitigation=False):
            units = review_units(st.session_state.analysis, gate_config().biased_input_threshold)
            if _run_units(units):
                st.session_state.review_units = units
                st.session_state.review_source = text
                st.session_state.review_step = 1
                st.rerun()
    if not text.strip():
        C.render(C.caption("Paste text, upload a document or use the sample to start."))
    elif _ready():
        st.success("A review for this text is ready. Use Next to continue, or run again after editing the text.")
        _nav(0)


def step_flags(a: dict) -> None:
    C.section("2 · Flagged phrases and supporting evidence",
              "Each phrase lists the rule that matched it and every detector score for its sentence (flag = score ≥ 0.5).")
    s = a["ensemble_score"]
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        C.render(C.highlighted_text(a["text"], a["spans"]))
        C.render(C.caption("Hover a highlight for the reason it was flagged."))
    with right:
        C.render(C.score_bar("Overall bias score", s, right=C.badge(C.level_for_score(s), C.tone_for_score(s)),
                             sub="Mean of the available detectors; each scores the most biased sentence."))
        for k, r in a["detector_results"].items():
            if r["score"] is None:
                C.render(C.note(f"<b>{C.esc(r['display'])}</b> could not run: {C.esc(r['error'])}", "warn"))
                continue
            C.render(C.score_bar(r["display"], r["score"], right=f'<span>{r["runtime_ms"]:.0f} ms</span>'))
        if a.get("missing_detectors"):
            C.render(C.note("Not available on this machine: " + C.esc(", ".join(a["missing_detectors"])), "warn"))
    ev = flag_evidence(a)
    units = st.session_state.review_units or []
    model_flagged = [u for u in units if u["score"] is not None and u["score"] >= gate_config().biased_input_threshold]
    if ev and not model_flagged:
        C.render(C.note("<b>The rules and the models disagree.</b> The rule lexicon flagged the phrases below, but no sentence "
                        f"reached the models' mean score of {gate_config().biased_input_threshold:g}, so the Gate treats the text "
                        "as not biased and keeps it unchanged. Fine-tuned detectors were trained on news and stereotype "
                        "sentences, not job adverts; a person should judge these phrases.", "warn"))
    C.gap()
    if not ev:
        C.render(C.empty_state("No phrases flagged", "The rule lexicon found no trigger phrases. Model scores above may still "
                               "rate the text; continue to see whether any rewrite was proposed.", "check"))
    for e in ev:
        dets = " ".join(C.badge(f"{d['display']} {d['score']:.2f}" + (" · flags" if d["flags"] else ""),
                                "red" if d["flags"] else "grey") for d in e["detectors"] if d["score"] is not None)
        weight = C.badge(f"rule weight {e['weight']:.2f}", C.tone_for_score((e["weight"] or 0) + 0.2)) if e.get("weight") is not None else ""
        C.render(f'<div class="fl-card compact fl-evidence"><div class="fl-row">{C.badge(e["category_label"], "accent")}{weight}</div>'
                 f'<div class="phrase">“{C.esc(e["phrase"])}”</div><div class="why">{C.esc(e["explanation"])}</div>'
                 f'<div class="fl-caption">Sentence: {C.esc(e["sentence"][:220])}</div><div class="fl-row mt">{dets}</div></div>')
    _nav(1)


def step_rewrites() -> None:
    C.section("3 · Original vs proposed rewrites", "Each flagged sentence is checked on its own. Deleted words are struck "
                                                   "through in red, inserted words are green.")
    u = _pick_unit()
    if u is None:
        _nav(2)
        return
    m = u["mitigation"]
    cands = {c["candidate"]: c for c in m["candidates"]}
    changed = [k for k, c in cands.items() if c["changed"]]
    if not changed:
        C.render(C.empty_state("No rewrite proposed", "No rewriter changed the text. The Gate verdict (step 5) explains why."))
    else:
        sel = _selected(m)
        pick = st.segmented_control("Version to review", changed, format_func=name_of, key="rv_choice_widget",
                                    default=sel if sel in changed else changed[0], on_change=_keep_choice)
        st.session_state.review_choice = pick or changed[0]
        pick = pick or changed[0]
        c = cands[pick]
        C.render(f'<div class="fl-row">{C.decision_badge(c["verdict"])}'
                 f'{C.badge("bias " + format(c["bias_before"], ".2f") + " → " + format(c["bias_after"], ".2f"), "grey")}'
                 f'{C.badge("facts kept " + format(c["fact_retention"], ".0%"), "green" if c["fact_retention"] >= 1 else "red")}</div>')
        C.gap()
        C.render(C.diff_html(m["original"], c["text"]))
        rec = m["decision"].get("chosen")
        C.render(C.caption(f"Gate recommendation: {name_of(rec) if rec else 'no automatic choice'}. "
                           "The selected version is used for the fact checks in step 4."))
    _nav(2)


def step_facts() -> None:
    u = _pick_unit()
    if u is None:
        _nav(3)
        return
    m = u["mitigation"]
    sel = _selected(m)
    cands = {c["candidate"]: c for c in m["candidates"]}
    c = cands[sel]
    C.section(f"4 · Critical-fact preservation · {name_of(sel)}",
              "From the fact checker: every name, number, date, role, quotation and negation in the original must survive.")
    cfg = gate_config()
    C.render(C.check_grid(preservation_items(c, cfg.min_semantic_similarity, cfg.max_contradiction)))
    C.render(C.caption(f"Fact retention {c['fact_retention']:.0%} (Gate threshold {cfg.min_fact_retention:g}). "
                       "'–' means the original contains nothing of that type."))
    _nav(3)


def step_gate() -> None:
    C.section("5 · Reliability Gate verdict", "The Gate turns preservation and detector signals into a decision with reason codes.")
    units = st.session_state.review_units or []
    if len(units) > 1:
        rows = [{"Sentence": f"{i + 1}. {x['text'][:90]}", "Verdict": C.decision_badge(x["mitigation"]["decision"]["decision"]),
                 "Reason": x["mitigation"]["decision"]["reason_code"]} for i, x in enumerate(units)]
        C.render(C.html_table(rows, raw={"Verdict"}))
        C.gap()
    u = _pick_unit()
    if u is None:
        _nav(4)
        return
    m = u["mitigation"]
    d = m["decision"]
    C.render(C.decision_banner(d["decision"], d["reason"], f"Reason code: {d['reason_code']}"
                               + (f" · chosen version: {name_of(d['chosen'])}" if d.get("chosen") else "")
                               + (f" · {d['attempts']} strict retr{'y' if d['attempts'] == 1 else 'ies'}" if d.get("attempts") else "")))
    if d.get("chosen_text"):
        C.render(f'<div class="fl-quote">{C.esc(d["chosen_text"])}</div>')
    sel = _selected(m)
    c = {x["candidate"]: x for x in m["candidates"]}[sel]
    cfg = gate_config()
    C.section(f"Evidence for {name_of(sel)}")
    cols = st.columns(4)
    vals = [("Semantic similarity", c["semantic_similarity"], cfg.min_semantic_similarity),
            (*entailment_signal(c, cfg.entailment_mode), cfg.min_entailment_prob),
            ("Fact retention", c["fact_retention"], cfg.min_fact_retention), ("Fluency", c["fluency"], cfg.min_fluency)]
    for col, (lab, v, thr) in zip(cols, vals):
        col.markdown(C.metric_tile(lab, f"{v:.2f}", f"threshold {thr:g}", "green" if v >= thr else "red", TIPS[lab]),
                     unsafe_allow_html=True)
    C.section("Reason codes per version")
    rows = reason_rows(m, REASONS, name_of)
    for r in rows:
        r["Verdict"] = C.decision_badge(r["Verdict"])
    C.render(C.html_table(rows, raw={"Verdict"}))
    _nav(4)


def step_export(a: dict) -> None:
    C.section("6 · Export the review")
    cfg = gate_config()
    ev = flag_evidence(a)
    units = st.session_state.review_units or []
    secs = export.sections_for_review(a, None, ev, [], None, REASONS)
    unit_payload = []
    for i, u in enumerate(units):
        m = u["mitigation"]
        sel = _selected(m) if i == st.session_state.get("review_unit", 0) else (m["decision"].get("chosen") or "original")
        c = {x["candidate"]: x for x in m["candidates"]}[sel]
        checks = preservation_items(c, cfg.min_semantic_similarity, cfg.max_contradiction)
        for title, body in export.sections_for_review(a, m, ev, checks, name_of(sel), REASONS)[3:]:
            secs.append((f"Sentence {i + 1} · {title}", [f"Sentence: {u['text']}", *body]))
        unit_payload.append({"sentence": u["text"], "model_mean_score": u["score"], "rule_flagged": u["rule_flagged"],
                             "selected_version": sel, "mitigation": m,
                             "preservation_checks": [{"check": n, "pass": ok, "detail": dt} for n, ok, dt in checks]})
    decisions = sorted({u["mitigation"]["decision"]["decision"] for u in units}) or ["NO SENTENCE CHECKED"]
    meta = {"generated": time.strftime("%Y-%m-%d %H:%M"), "decisions": ", ".join(decisions), "sentences_checked": len(units),
            "lite_mode": st.session_state.lite}
    payload = {"disclaimer": export.REVIEW_DISCLAIMER, "meta": meta, "analysis": a, "flag_evidence": ev, "sentences": unit_payload}
    C.render(C.caption("The report contains the input, flagged phrases with evidence, every proposed rewrite, the fact checks "
                       "and the Gate verdict with reason codes."))
    x, y, z, _ = st.columns([1, 1, 1, 2])
    x.download_button("PDF report", export.to_pdf("FairLens Review", secs, meta), "fairlens_review.pdf", "application/pdf",
                      icon=":material/picture_as_pdf:", on_click="ignore", use_container_width=True, key="rv_pdf")
    y.download_button("JSON", export.to_json(payload), "fairlens_review.json", "application/json",
                      icon=":material/data_object:", on_click="ignore", use_container_width=True, key="rv_json")
    z.download_button("HTML report", export.to_html("FairLens Review", secs, meta), "fairlens_review.html", "text/html",
                      icon=":material/html:", on_click="ignore", use_container_width=True, key="rv_html")
    C.gap()
    st.button("Start a new review", icon=":material/restart_alt:", key="rv_new", on_click=_reset)
    _nav(5)


def _reset() -> None:
    st.session_state.review_step = 0
    st.session_state.review_source = None
    st.session_state.review_choice = None
    st.session_state.review_units = None
    st.session_state.review_unit = 0
    st.session_state.rv_text = ""


def render():
    C.page_header("FairLens Review", "A guided check of a job description: find biased wording, see the evidence, compare "
                                     "proposed rewrites, confirm that facts survive and get the Reliability Gate's verdict.",
                  "Guided workflow")
    C.render(C.note(f"<b>{DISCLAIMER}</b> Detectors make mistakes and bias labels depend on context; a person makes the final call."))
    C.gap()
    stub_banner()
    ready = _ready()
    step = st.session_state.review_step if ready else 0
    st.session_state.review_step = step
    C.render(C.stepper(STEPS, step if step < len(STEPS) - 1 or not ready else len(STEPS)))
    if step == 0:
        step_input()
        return
    a = st.session_state.analysis
    {1: lambda: step_flags(a), 2: step_rewrites, 3: step_facts, 4: step_gate, 5: lambda: step_export(a)}[step]()
