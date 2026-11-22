import json

import streamlit as st

from .. import components as C
from ..common import result_header
from ..results import latest_frame, latest_summary
from .model_comparison import run_button


def render():
    C.page_header("Error Analysis", "Browse where detectors and the Gate go wrong: false positives and negatives, preservation "
                                    "failures in real rewrites, and unsafe accepts on the controlled set.", "More")
    t1, t2, t3 = st.tabs(["Detector errors", "Preservation failures", "Gate errors"])
    with t1:
        df = latest_frame("benchmark_detection", "detection_scores.parquet")
        if not result_header("Detection benchmark", "benchmark_detection", df is not None):
            run_button("detection", "Run detection benchmark", "ea_det")
        else:
            dets = [c for c in df.columns if c not in {"text", "label", "source", "bias_type"}]
            a, b, c = st.columns(3)
            d = a.selectbox("Detector", dets)
            kind = b.radio("Error type", ["False positives", "False negatives"], horizontal=True)
            src = c.selectbox("Source", ["all"] + sorted(df.source.unique()))
            sub = df if src == "all" else df[df.source == src]
            err = sub[(sub[d] >= 0.5) & (sub.label == 0)] if kind == "False positives" else sub[(sub[d] < 0.5) & (sub.label == 1)]
            err = err.assign(conf=(err[d] - 0.5).abs()).sort_values("conf", ascending=False)
            st.caption(f"{len(err)} of {len(sub)} sentences ({len(err) / max(1, len(sub)):.1%}); most confident errors first.")
            for r in err.head(12).itertuples():
                others = " ".join(C.badge(f"{o} {getattr(r, o):.2f}", C.tone_for_score(getattr(r, o))) for o in dets if o != d)
                C.render(f'<div class="fl-card compact">{C.badge(r.source, "grey")} '
                         f'{C.badge(f"{d} {getattr(r, d):.2f}", "red" if kind == "False positives" else "amber")} '
                         f'{C.badge("gold: " + ("biased" if r.label else "neutral"), "blue")} {C.badge(r.bias_type, "grey")}'
                         f'<div class="fl-card-text">{C.esc(r.text)}</div><div class="fl-row">{others}</div></div>')
    with t2:
        cand = latest_frame("benchmark_mitigation", "mitigation_candidates.parquet")
        if not result_header("Mitigation benchmark", "benchmark_mitigation", cand is not None):
            run_button("mitigation", "Run mitigation benchmark", "ea_mit")
        else:
            fails = cand[(cand.changed) & ((cand.fact_retention < 1) | (cand.contradiction > 0.5))]
            rw = st.selectbox("Rewriter", ["all"] + sorted(fails.candidate.str.replace(r"_retry\d", "", regex=True).unique()))
            if rw != "all":
                fails = fails[fails.candidate.str.startswith(rw)]
            st.caption(f"{len(fails)} rewrites lost a fact or contradicted the original ({len(fails) / max(1, cand.changed.sum()):.1%} of changed candidates).")
            for r in fails.head(12).itertuples():
                fc = json.loads(r.fact_checks)
                lost = [f"{k}: {', '.join(v.get('missing', []))}" for k, v in fc.items() if not v.get("pass") and v.get("missing")]
                if not fc["negation"]["pass"]:
                    lost.append("negation changed")
                C.render(f'<div class="fl-card compact">{C.badge(r.candidate, "accent")} '
                         f'{C.decision_badge(r.verdict)} {C.badge("contradiction " + format(r.contradiction, ".2f"), "red" if r.contradiction > .5 else "grey")}'
                         f'{C.diff_html(r.input_text, r.text)}<div class="fl-caption">Lost: {C.esc("; ".join(lost) or "-")}</div></div>')
    with t3:
        g = latest_frame("benchmark_gate", "gate_signals.parquet")
        if not result_header("Gate benchmark", "benchmark_gate", g is not None):
            run_button("gate", "Run gate benchmark", "ea_gate")
        else:
            test = g[g.split == "test"]
            unsafe = test[(test.pred_tuned == "ACCEPT") & (test.gold != "ACCEPT")]
            missed = test[(test.pred_tuned != "ACCEPT") & (test.gold == "ACCEPT")]
            which = st.radio("Show", [f"Unsafe accepts ({len(unsafe)})", f"Missed good rewrites ({len(missed)})"], horizontal=True)
            rows = unsafe if which.startswith("Unsafe") else missed
            st.caption("Unsafe accept = the Gate accepted a candidate whose gold label is REJECT/RETRY. Missed = a gold-acceptable "
                       "candidate was not accepted (shows the cost of strictness).")
            for r in rows.head(12).itertuples():
                C.render(f'<div class="fl-card compact">{C.badge(r.perturbation.replace("_", " "), "amber")} '
                         f'{C.badge("gold " + r.gold, "blue")} {C.decision_badge(r.pred_tuned)} {C.badge("seed " + str(r.seed), "grey")}'
                         f'{C.diff_html(r.input_text, r.candidate_text)}<div class="fl-caption">'
                         f'sim {r.semantic_similarity:.2f} · entail→ {r.entail_forward:.2f} · entail← {r.entail_backward:.2f} · facts {r.fact_retention:.2f} · '
                         f'bias {r.bias_before:.2f}→{r.bias_after:.2f} · reasons {C.esc(r.reasons_tuned)}</div></div>')
