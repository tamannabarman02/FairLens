import streamlit as st

from .. import components as C
from ..common import STEPS, current_step, fmt, fmt_pct, go, require_analysis, run_analysis, stub_banner

CARD_TONE = {"original": "grey", "rule_based": "amber", "flan_t5": "blue", "qwen": "accent"}
CARD_NAME = {"original": "Original", "rule_based": "Rule-based", "flan_t5": "FLAN-T5", "qwen": "Qwen"}


def name_of(c: str) -> str:
    base, _, retry = c.partition("_retry")
    return CARD_NAME.get(base, base) + (f" (strict retry {retry})" if retry else "")


def render():
    C.page_header("Mitigation", "Four versions side by side: the original, a rule-based rewrite and two LLM rewrites. "
                                "Each carries its Reliability Gate verdict and preservation metrics.", "Step 2 · Mitigation")
    stub_banner()
    C.render(C.stepper(STEPS, current_step()))
    if not require_analysis():
        return
    m = st.session_state.mitigation
    if not m or m["original"] != st.session_state.analysis["text"]:
        C.render(C.empty_state("No rewrites generated yet", "Generate rewrites for the analysed text. The first run loads FLAN-T5 and Qwen "
                               "(about a minute on CPU; Lite mode uses FLAN-T5-small and is faster)."))
        C.gap()
        if st.button("Generate rewrites", type="primary", icon=":material/auto_fix_high:"):
            if run_analysis(st.session_state.analysis["text"], with_mitigation=True):
                st.rerun()
        return

    cands = {c["candidate"]: c for c in m["candidates"]}
    first = [k for k in ["original", "rule_based", "flan_t5", "qwen"] if k in cands]
    missing = [CARD_NAME[k] for k in ["rule_based", "flan_t5", "qwen"] if k not in cands]
    if missing:
        errs = [r for r in m["rewrite_log"] if r.get("error")]
        C.render(C.note("Unavailable: " + ", ".join(missing) + (". Errors: " + "; ".join(e["error"][:120] for e in errs) if errs else ""), "warn"))
    cols = st.columns(len(first))
    for col, k in zip(cols, first):
        c = cands[k]
        with col:
            delta = c["bias_before"] - c["bias_after"]
            badges = (f'{C.decision_badge(c["verdict"])} {C.badge(f"bias Δ {delta:+.2f}", "green" if delta > 0.05 else "grey")} '
                      f'{C.badge("sim " + fmt(c["semantic_similarity"]), "blue")} {C.badge("facts " + fmt_pct(c["fact_retention"]), "green" if c["fact_retention"] >= 1 else "red")}')
            C.render(f'<div class="fl-card tall tone-{CARD_TONE.get(k, "grey")}"><h4>{name_of(k)}</h4>'
                     f'<div class="fl-card-text">{C.esc(c["text"])}</div><div class="fl-row">{badges}</div></div>')
            if st.button("Use this version", key=f"use_{k}", use_container_width=True,
                         type="primary" if m["decision"].get("chosen") == k else "secondary"):
                st.session_state.selected_candidate = k
                go("Reliability Gate")
    d = m["decision"]
    C.gap()
    C.render(C.note(f"<b>Gate recommendation:</b> {C.DECISION_LABEL.get(d['decision'], d['decision'])}"
                    + (f" → {name_of(d['chosen'])}" if d.get("chosen") else "") + f". {C.esc(d['reason'])}"
                    + (f" ({d['attempts']} strict retr{'y' if d['attempts'] == 1 else 'ies'} used)" if d.get("attempts") else "")))
    C.section("Comparison", "Green = best value in each column.")
    rows = []
    for k, c in cands.items():
        rows.append({"Version": name_of(k), "Verdict": C.decision_badge(c["verdict"]), "Bias score": c["bias_after"],
                     "Semantic similarity": c["semantic_similarity"], "Bidirectional entailment": c["entailment_both_ways"],
                     "Fact retention": c["fact_retention"], "Fluency": c["fluency"], "Length change": c["length_ratio"] - 1})
    C.render(C.html_table(rows, raw={"Verdict"}, best={"Bias score": "min", "Semantic similarity": "max", "Fact retention": "max",
                                                       "Fluency": "max", "Bidirectional entailment": "max"},
                          fmt={"Length change": "+.0%"}))
    C.render(C.caption("Green = best value in the column. The original trivially has perfect similarity; the Gate decides "
                       "whether lower bias was bought by losing meaning or facts."))
    C.section("Diff view", "Deleted words in red, inserted words in green.")
    changed = [k for k, c in cands.items() if c["changed"]]
    if changed:
        pick = st.segmented_control("Compare original with", changed, format_func=name_of, default=changed[0], key="diff_pick")
        if pick:
            C.render(C.diff_html(m["original"], cands[pick]["text"]))
    else:
        st.caption("No rewriter changed the text.")
    with st.expander("Prompts and generation log"):
        for r in m["rewrite_log"]:
            st.markdown(f"**{r['rewriter']}** · attempt {r['attempt']} · strict={r['strict']} · {r['runtime_ms']:.0f} ms")
            if r.get("prompt"):
                st.code(r["prompt"][-900:], language=None)
            st.write(r.get("text") or f"error: {r.get('error')}")
