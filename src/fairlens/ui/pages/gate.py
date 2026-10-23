import streamlit as st

from ...reliability_gate.rules import REASONS
from ...reporting import charts
from .. import components as C
from ..common import STEPS, current_step, fmt, fmt_pct, go, require_analysis, theme
from ..state import gate_config
from .mitigation import name_of

TIPS = {
    "Semantic similarity": "Cosine similarity of MiniLM sentence embeddings between original and rewrite (1 = same meaning).",
    "Bidirectional entailment": "min(P(original ⇒ rewrite), P(rewrite ⇒ original)) from an NLI cross-encoder.",
    "Fact retention": "Share of entities, numbers, dates, roles, quotes and negation polarity that survive the rewrite.",
    "Fluency": "Heuristic sanity score: repetition, prompt leakage, garbage characters (not a language-model score).",
}


def render():
    C.page_header("Reliability Gate", "The Gate turns preservation and detector signals into a decision with a reason code: "
                                      "accept, retry, reject or send to a human.", "Step 3 · Reliability check")
    C.render(C.stepper(STEPS, current_step()))
    if not require_analysis():
        return
    m = st.session_state.mitigation
    if not m:
        C.render(C.empty_state("No rewrites to check", "Generate rewrites on the Mitigation page first."))
        if st.button("Go to Mitigation", type="primary"):
            go("Mitigation")
        return
    cands = {c["candidate"]: c for c in m["candidates"]}
    d = m["decision"]
    default = st.session_state.selected_candidate or d.get("chosen") or next((k for k in cands if cands[k]["changed"]), "original")
    keys = list(cands)
    sel = st.selectbox("Selected rewrite", keys, index=keys.index(default) if default in keys else 0, format_func=name_of)
    c = cands[sel]
    C.render(f'<div class="fl-quote">{C.esc(c["text"])}</div>')
    reasons = c["reasons"]
    C.render(C.decision_banner(c["verdict"], REASONS.get(reasons[0], reasons[0]) if reasons else "",
                               "Reason codes: " + ", ".join(reasons)))
    if d.get("chosen") != sel:
        st.caption(f"Gate's overall recommendation for this input: {C.DECISION_LABEL.get(d['decision'], d['decision'])}"
                   + (f" → {name_of(d['chosen'])}" if d.get("chosen") else "") + f" ({d['reason_code']}).")

    t1, t2, t3 = st.tabs(["Evaluation Metrics", "Detailed Analysis", "Explanation"])
    cfg = gate_config()
    with t1:
        cols = st.columns(4)
        vals = [("Semantic similarity", c["semantic_similarity"], cfg.min_semantic_similarity),
                ("Bidirectional entailment", c["entailment_both_ways"], cfg.min_entailment_prob),
                ("Fact retention", c["fact_retention"], cfg.min_fact_retention),
                ("Fluency", c["fluency"], cfg.min_fluency)]
        for col, (lab, v, thr) in zip(cols, vals):
            col.markdown(C.metric_tile(lab, fmt(v), f"threshold {thr:g}", "green" if v >= thr else "red", TIPS[lab]), unsafe_allow_html=True)
        st.write("")
        names = ["Bias reduction", "Similarity", "Entail →", "Entail ←", "Facts", "Agreement", "Confidence", "Fluency"]
        def vec(x):
            return [max(0.0, min(1.0, x["bias_reduction"] / max(1e-6, x["bias_before"]))) if x["bias_before"] > 0 else 0.0,
                    x["semantic_similarity"], x["entail_forward"], x["entail_backward"], x["fact_retention"],
                    1 - min(1.0, x["detector_std"] / 0.5), x["confidence_margin"], x["fluency"]]
        lb = m.get("lowest_bias_choice")
        compare = st.toggle("Compare with lowest-bias-score selection", key="cmp_lowest",
                            help="The naive alternative: always pick the candidate with the lowest ensemble bias score.")
        fig = charts.radar(names, vec(c), theme(), label=name_of(sel),
                           compare=(f"Lowest-bias pick: {name_of(lb)}", vec(cands[lb])) if compare and lb in cands else None)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        if compare and lb in cands:
            o = cands[lb]
            a, b = st.columns(2)
            a.markdown(C.card(f"FairLens Gate → {name_of(d['chosen']) if d.get('chosen') else 'no automatic choice'}",
                              f"{C.esc(d.get('chosen_text') or 'Escalated: ' + d['reason'])}", "accent",
                              extra=f'<div style="margin-top:8px">{C.decision_badge(d["decision"])}</div>'), unsafe_allow_html=True)
            b.markdown(C.card(f"Lowest bias score → {name_of(lb)}", C.esc(o["text"]), "grey",
                              extra=f'<div style="margin-top:8px">{C.decision_badge(o["verdict"])} {C.badge("bias " + fmt(o["bias_after"]), "grey")} '
                                    f'{C.badge("facts " + fmt_pct(o["fact_retention"]), "green" if o["fact_retention"] >= 1 else "red")}</div>'),
                       unsafe_allow_html=True)
            st.caption("The lowest-bias pick ignores whether meaning and facts survived; the Gate does not.")
    with t2:
        fc = c["fact_checks"]
        def det(k):
            x = fc.get(k, {})
            if k == "negation":
                return f"{x.get('original', 0)} cue(s) in original, {x.get('rewrite', 0)} in rewrite"
            if not x.get("kept") and not x.get("missing"):
                return "none in original"
            return (f"kept: {', '.join(x.get('kept', []))}" if x.get("kept") else "") + \
                   (f" · missing: {', '.join(x['missing'])}" if x.get("missing") else "")
        items = [("Named entities", fc.get("entities", {}).get("pass"), det("entities")),
                 ("Numbers", fc.get("numbers", {}).get("pass"), det("numbers")),
                 ("Dates", fc.get("dates", {}).get("pass"), det("dates")),
                 ("Roles", fc.get("roles", {}).get("pass"), det("roles")),
                 ("Negation", fc.get("negation", {}).get("pass"), det("negation")),
                 ("Overall meaning", c["semantic_similarity"] >= cfg.min_semantic_similarity and c["contradiction"] <= cfg.max_contradiction,
                  f"similarity {c['semantic_similarity']:.2f}, contradiction {c['contradiction']:.2f}")]
        if fc.get("quotes", {}).get("kept") or fc.get("quotes", {}).get("missing"):
            items.append(("Quotations", fc["quotes"]["pass"], det("quotes")))
        st.markdown("##### Preservation checks")
        C.render(C.check_grid(items))
        st.write("")
        st.markdown("##### Detector scores before → after")
        rows = [{"Detector": k, "Original": c["detector_scores_before"][k], "Rewrite": c["detector_scores_after"][k],
                 "Change": c["detector_scores_after"][k] - c["detector_scores_before"][k]} for k in c["detector_scores_after"]]
        C.render(C.html_table(rows, fmt={"Change": "+.2f"}))
        st.caption(f"Ensemble {c['bias_before']:.2f} → {c['bias_after']:.2f}; detector std {c['detector_std']:.2f}; "
                   f"confidence margin {c['confidence_margin']:.2f}; length ratio {c['length_ratio']:.2f}.")
        st.markdown("##### Checks applied")
        chk = [{"Check": k, "Passed": "✓" if v else "✗", "Meaning": REASONS.get(k, "")} for k, v in c["checks"].items()]
        C.render(C.html_table(chk))
    with t3:
        C.render(f"""<div class="fl-card"><h4>How this decision was made</h4><p>
        1. The input was judged <b>{'biased' if d['input_biased'] else 'not biased'}</b> (ensemble ≥ {cfg.biased_input_threshold}).
        For a non-biased input the correct action is NO CHANGE, so any rewrite is rejected as unnecessary.<br><br>
        2. Each rewrite must keep meaning (similarity ≥ {cfg.min_semantic_similarity}, entailment [{cfg.entailment_mode}] ≥ {cfg.min_entailment_prob},
        contradiction ≤ {cfg.max_contradiction}), keep every extracted fact (retention ≥ {cfg.min_fact_retention}), stay fluent and within
        {cfg.length_ratio_bounds[0]}–{cfg.length_ratio_bounds[1]}× the length. Failing any of these → <b>Reject</b>.<br><br>
        3. If meaning is kept but the bias score drops by less than {cfg.min_bias_reduction} or stays ≥ {cfg.max_residual_bias} → <b>Retry</b>
        with a stricter preserve-facts prompt (up to {cfg.max_retries} times), then <b>Human review</b>.<br><br>
        4. If detectors disagree (std &gt; {cfg.max_detector_std}) or the ensemble is unsure (margin &lt; {cfg.min_confidence_margin}) → <b>Human review</b>.<br><br>
        5. Otherwise → <b>Accept</b>; among accepted rewrites the one with the best bias reduction + similarity + fact retention is chosen.
        </p></div>""")
        st.write("")
        C.render(C.note("Thresholds are transparent and documented in configs/models.yaml; where a gate benchmark exists, the "
                        "values tuned on its validation split are used. The Gate cannot certify that a text is unbiased."))
