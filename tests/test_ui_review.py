"""FairLens Review helpers and report exports, driven by the real pipeline with fake models (no downloads)."""
import json

import pytest
from conftest import FakeDetector, FakeRewriter, FakeSuite

from fairlens.detection.rule_based import RuleBasedDetector
from fairlens.pipeline import analyze_text, mitigate_text
from fairlens.reliability_gate.rules import REASONS, GateConfig
from fairlens.reporting import export
from fairlens.ui.review_logic import FACT_TYPES, flag_evidence, preservation_items, reason_rows

SRC = "We need a young, aggressive salesman in Leeds with 5 years of experience. Applications close on 12 March 2026."
GOOD = "We need a proactive salesperson in Leeds with 5 years of experience. Applications close on 12 March 2026."
BAD = "We need a proactive salesperson with 2 years of experience."


@pytest.fixture(scope="module")
def run():
    t = {"salesman": 0.9, "young": 0.8, "aggressive": 0.7}
    dets = {"a": FakeDetector("a", t), "rule_based": RuleBasedDetector()}
    a = analyze_text(SRC, dets, explain=False).to_dict()
    rw = {"good": FakeRewriter("good", {SRC: GOOD}), "bad": FakeRewriter("bad", {SRC: BAD})}
    cfg = GateConfig.from_config(min_semantic_similarity=0.6, min_entailment_prob=0.4, max_detector_std=0.5)
    m = mitigate_text(SRC, dets, rw, FakeSuite(), cfg).to_dict()
    return a, m, cfg


def test_flagged_phrases_carry_rule_explanation_and_detector_scores(run):
    a, _, _ = run
    ev = flag_evidence(a)
    phrases = {e["phrase"].lower() for e in ev}
    assert {"young", "salesman"} <= phrases
    young = next(e for e in ev if e["phrase"].lower() == "young")
    assert young["explanation"] and young["category_label"]
    assert {d["detector"] for d in young["detectors"]} == {"a", "rule_based"}
    assert "A" in young["flagged_by"]
    assert young["sentence"].startswith("We need a young")


def test_evidence_scores_are_copied_from_the_analysis_not_recomputed(run):
    a, _, _ = run
    sent_scores = a["sentences"][0]["scores"]
    for e in flag_evidence(a):
        if e["start"] < a["sentences"][0]["end"]:
            assert {d["detector"]: d["score"] for d in e["detectors"]} == sent_scores


def test_fact_checks_cover_all_critical_fact_types(run):
    _, m, cfg = run
    bad = next(c for c in m["candidates"] if c["candidate"] == "bad")
    items = preservation_items(bad, cfg.min_semantic_similarity, cfg.max_contradiction)
    names = [n for n, _, _ in items]
    assert names[:len(FACT_TYPES)] == [label for _, label in FACT_TYPES] and names[-1] == "Overall meaning"
    by = {n: (ok, d) for n, ok, d in items}
    assert by["Numbers"][0] is False and "missing" in by["Numbers"][1]
    assert by["Quotations"][0] is None and by["Quotations"][1] == "none in original"


def test_reason_rows_explain_every_code(run):
    _, m, _ = run
    rows = reason_rows(m, REASONS)
    assert len(rows) == len(m["candidates"])
    assert all(r["Reason codes"] and r["What it means"] for r in rows)


def test_review_report_exports_pdf_html_json(run):
    a, m, cfg = run
    good = next(c for c in m["candidates"] if c["candidate"] == "good")
    secs = export.sections_for_review(a, m, flag_evidence(a), preservation_items(good, 0.7, 0.5), "good", REASONS)
    heads = [h for h, _ in secs]
    assert heads[0] == "Scope" and any(h.startswith("5. Reliability Gate verdict") for h in heads)
    assert export.REVIEW_DISCLAIMER in secs[0][1][0]
    assert export.to_pdf("FairLens Review", secs).startswith(b"%PDF")
    html = export.to_html("FairLens Review", secs).decode("utf-8")
    assert "not a legal or compliance certification tool" in html and "young" in html
    assert json.loads(export.to_json({"disclaimer": export.REVIEW_DISCLAIMER}))["disclaimer"].startswith("Review assistant")
