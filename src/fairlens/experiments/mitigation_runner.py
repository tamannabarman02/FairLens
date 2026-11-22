"""Batched mitigation experiment: rewriters x mitigation_eval set, Gate (+retry) vs. baselines.

Stages run one model at a time (generation per rewriter, then each detector over all texts, then the
preservation models) so the experiment fits a CPU laptop. Generations and detector scores are cached.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT
from ..detection.registry import build_detectors
from ..evaluation import facts as fact_checker
from ..evaluation import metrics as M
from ..evaluation.preservation import PreservationSuite
from ..mitigation.llm_adapters import build_rewriters
from ..models import manager
from ..reliability_gate import baselines as B
from ..reliability_gate.rules import ACCEPT, HUMAN_REVIEW, RETRY, GateConfig, decide, judge
from ..reliability_gate.signals import compute_signals, normalise
from .cache import ScoreCache, detector_identity, generator_identity, score_texts
from .tracking import RunTracker, rel_path

PROC = PROJECT_ROOT / "datasets" / "processed"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class PrecomputedSuite:
    """Preservation suite backed by a dict of precomputed (original, rewrite) evaluations."""

    def __init__(self, table: dict[tuple[str, str], dict]):
        self.table = table

    def evaluate(self, originals, rewrites):
        return [self.table[(o, r)] for o, r in zip(originals, rewrites)]


def generate(rewriters: dict, items: list[tuple[str, str]], strict: bool, attempt: int, cache: ScoreCache,
             log_rows: list[dict]) -> dict[tuple[str, str], str]:
    """items: (input_id, text). Returns {(input_id, rewriter_name): text}. One model loaded at a time."""
    out = {}
    for name, rw in rewriters.items():
        if strict and name == "rule_based":
            continue
        manager().clear()
        ident = generator_identity(rw)
        log(f"  generating with {rw.display} ({ident}) for {len(items)} inputs, strict={strict}")
        t0 = time.perf_counter()
        for k, (iid, text) in enumerate(items):
            hints = fact_checker.extract(text).as_hints() if strict else []
            pk = json.dumps([strict, text, hints])
            payload = cache.get_gen(ident, pk) if name != "rule_based" else None
            if payload is None:
                r = rw.rewrite(text, strict=strict, facts=hints, attempt=attempt)
                payload = r.to_dict()
                if name != "rule_based" and r.text is not None:
                    cache.put_gen(ident, pk, payload)
            payload = {**payload, "attempt": attempt, "input_id": iid, "rewriter": name, "model": ident}
            log_rows.append(payload)
            if payload.get("text"):
                out[(iid, name if attempt == 0 else f"{name}_retry{attempt}")] = payload["text"]
            if k % 50 == 49:
                log(f"    {k + 1}/{len(items)} ({(time.perf_counter() - t0) / (k + 1):.2f}s/input)")
    return out


def score_all(detectors: dict, texts: list[str], cache: ScoreCache) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for name, det in detectors.items():
        manager().clear()
        log(f"  scoring {len(set(texts))} texts with {det.display}")
        sc, _ = score_texts(det, texts, cache)
        for t, s in sc.items():
            table.setdefault(t, {})[name] = s
    return table


def preserve_all(pairs: list[tuple[str, str]], suite: PreservationSuite) -> dict[tuple[str, str], dict]:
    pairs = [p for p in dict.fromkeys(pairs) if normalise(p[0]) != normalise(p[1])]
    if not pairs:
        return {}
    manager().clear()
    log(f"  preservation checks for {len(pairs)} pairs")
    res = suite.evaluate([p[0] for p in pairs], [p[1] for p in pairs])
    return dict(zip(pairs, res))


def run_mitigation(limit: int | None = None, lite: bool = False, rewriter_keys=None, detector_keys=None,
                   cfg: GateConfig | None = None) -> str:
    df = pd.read_parquet(PROC / "mitigation_eval.parquet")
    if limit:
        df = pd.concat([d.head(max(1, limit * len(d) // len(df))) for _, d in df.groupby("label")])
    if cfg is None:
        from ..ui.state import tuned_gate_overrides  # thresholds tuned on the gate *val* split, if a gate run exists

        cfg = GateConfig.from_config(**tuned_gate_overrides())
    rewriters = build_rewriters(rewriter_keys, lite=lite)
    detectors = build_detectors(detector_keys, lite=lite)
    cache = ScoreCache()
    suite = PreservationSuite()
    tracker = RunTracker("benchmark_mitigation", config={"n_inputs": len(df), "lite": lite, "gate": cfg.__dict__ | {"disabled": sorted(cfg.disabled)},
                                                         "rewriters": {k: getattr(v, "model_id", k) for k, v in rewriters.items()},
                                                         "detectors": list(detectors)},
                         seed=13, dataset_files=[PROC / "mitigation_eval.parquet"],
                         models={k: getattr(v, "model_id", k) for k, v in rewriters.items()} |
                                {k: detector_identity(d) for k, d in detectors.items()})
    texts = dict(zip(df.id, df.text))
    gen_log: list[dict] = []
    log(f"mitigation run {tracker.run_id}: {len(df)} inputs, rewriters={list(rewriters)}, detectors={list(detectors)}")

    cands = generate(rewriters, list(texts.items()), strict=False, attempt=0, cache=cache, log_rows=gen_log)
    per_input: dict[str, dict[str, str]] = {iid: {"original": t} for iid, t in texts.items()}
    for (iid, name), t in cands.items():
        per_input[iid][name] = t

    det_scores = score_all(detectors, list(texts.values()) + list(cands.values()), cache)
    pres = preserve_all([(texts[i], t) for (i, _), t in cands.items()], suite)

    def signals_for(iid):
        return compute_signals(texts[iid], per_input[iid], detectors, PrecomputedSuite(pres), det_scores)

    sigs = {iid: signals_for(iid) for iid in texts}
    static = {iid: decide(s, cfg, allow_retry=False) for iid, s in sigs.items()}
    dec = {iid: decide(s, cfg, allow_retry=True) for iid, s in sigs.items()}
    llm = {k: v for k, v in rewriters.items() if k != "rule_based"}
    for attempt in range(1, cfg.max_retries + 1):
        todo = [iid for iid, d in dec.items() if d.decision == RETRY]
        if not todo or not llm:
            break
        log(f"retry {attempt}: {len(todo)} inputs")
        new = generate(llm, [(i, texts[i]) for i in todo], strict=True, attempt=attempt, cache=cache, log_rows=gen_log)
        for (iid, name), t in new.items():
            per_input[iid][name] = t
        for t, v in score_all(detectors, list(new.values()), cache).items():
            det_scores.setdefault(t, {}).update(v)
        pres.update(preserve_all([(texts[i], t) for (i, _), t in new.items()], suite))
        for iid in todo:
            sigs[iid] = signals_for(iid)
            dec[iid] = decide(sigs[iid], cfg, allow_retry=True, attempts=attempt)
    for iid, d in dec.items():
        if d.decision == RETRY:
            final = decide(sigs[iid], cfg, allow_retry=False, attempts=d.attempts)
            if final.decision != ACCEPT:
                final.reason_code, final.reason = "RETRIES_EXHAUSTED", "No candidate passed after strict retries; escalated to human review."
            dec[iid] = final

    # reference similarity (authored subset) for every candidate
    refs = dict(zip(df.id, df.reference))
    ref_pairs = [(t, refs[i]) for i, c in per_input.items() for t in c.values() if isinstance(refs.get(i), str)]
    ref_sim = {}
    if ref_pairs:
        manager().clear()
        u = list(dict.fromkeys(ref_pairs))
        ref_sim = dict(zip(u, suite.sim.similarity([a for a, _ in u], [b for _, b in u])))

    rows = []
    label = dict(zip(df.id, df.label))
    source = dict(zip(df.id, df.source))
    for iid, ss in sigs.items():
        biased = static[iid].input_biased
        for s in ss:
            v = judge(s, cfg, biased)
            rows.append({"input_id": iid, "source": source[iid], "gold_label": int(label[iid]), "input_text": texts[iid],
                         **{k: val for k, val in s.to_dict().items() if k not in {"fact_checks", "detector_scores_before", "detector_scores_after"}},
                         "fact_checks": json.dumps(s.fact_checks), "detector_scores_after": json.dumps(s.detector_scores_after),
                         "detector_scores_before": json.dumps(s.detector_scores_before),
                         "verdict": v.decision, "reasons": ",".join(v.reasons), "input_biased_pred": biased,
                         "ref_similarity": ref_sim.get((s.text, refs[iid])) if isinstance(refs.get(iid), str) else None,
                         "final_decision": dec[iid].decision, "final_choice": dec[iid].chosen})
    cand_df = pd.DataFrame(rows)

    summary = summarise(cand_df, sigs, dec, static, label, cfg, rewriters)
    summary["run_id"] = tracker.run_id
    tracker.save_artifact("mitigation_candidates.parquet", cand_df)
    (tracker.dir / "prompts_log.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in gen_log), encoding="utf-8")
    tracker.manifest.setdefault("artifacts", {})["prompts_log.jsonl"] = rel_path(tracker.dir / "prompts_log.jsonl")
    tracker.save_artifact("mitigation_summary.json", summary)
    tracker.finish(metrics={"gate_final": summary["gate_final_distribution"],
                            "policies": {k: {m: v.get(m) for m in ["accepted_fact_failure_rate", "unnecessary_change_rate"]}
                                         for k, v in summary["policies"].items()}})
    log(f"done: {tracker.run_id}")
    return tracker.run_id


STORED_SIGNAL_FIELDS = ["changed", "bias_before", "bias_after", "bias_reduction", "detector_std", "confidence_margin",
                        "uncertainty", "semantic_similarity", "entail_forward", "entail_backward", "entailment_both_ways",
                        "contradiction", "fact_retention", "fluency", "length_ratio"]


def run_mitigation_from_stored(source_run: str, gate_run: str | None = None) -> str:
    """Re-decide a stored mitigation run with (new) Gate thresholds, reusing its rewrites and signals.

    Rewrites, retry rewrites, detector scores and preservation signals come from `source_run`'s
    mitigation_candidates.parquet; only the Gate decisions, the retry schedule and the policy metrics are
    recomputed. Thresholds come from `gate_run`'s gate_summary.json (default: the latest gate run). If the new
    thresholds would retry an input more often than the source run did, there is no stored rewrite to use and the
    run stops, listing those inputs, rather than mixing in freshly generated rewrites.
    """
    from ..reliability_gate.signals import Signals

    src = PROJECT_ROOT / "outputs" / "runs" / source_run
    stored = pd.read_parquet(src / "mitigation_candidates.parquet")
    src_manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    if gate_run:
        gate_summary = json.loads((PROJECT_ROOT / "outputs" / "runs" / gate_run / "gate_summary.json").read_text(encoding="utf-8"))
        overrides = dict(gate_summary["tuned_rules"])
    else:
        from ..ui.state import tuned_gate_overrides

        overrides = tuned_gate_overrides()
    cfg = GateConfig.from_config(**overrides)
    df = pd.read_parquet(PROC / "mitigation_eval.parquet")
    if set(df.id) != set(stored.input_id):
        raise SystemExit("stored candidates do not cover mitigation_eval.parquet")
    texts, label, source = dict(zip(df.id, df.text)), dict(zip(df.id, df.label)), dict(zip(df.id, df.source))
    if any(texts[i] != t for i, t in zip(stored.input_id, stored.input_text)):
        raise SystemExit("stored input texts differ from mitigation_eval.parquet")

    def to_sig(r) -> Signals:
        s = Signals(candidate=r.candidate, text=r.text, detector_scores_before=json.loads(r.detector_scores_before),
                    detector_scores_after=json.loads(r.detector_scores_after), fact_checks=json.loads(r.fact_checks),
                    **{f: getattr(r, f) for f in STORED_SIGNAL_FIELDS})
        s.changed = bool(s.changed)
        return s

    by_input = {iid: list(g.itertuples()) for iid, g in stored.groupby("input_id", sort=False)}
    all_sigs = {iid: [to_sig(r) for r in rows] for iid, rows in by_input.items()}
    sigs = {iid: [s for s in ss if "retry" not in s.candidate] for iid, ss in all_sigs.items()}
    tracker = RunTracker("benchmark_mitigation", config={"n_inputs": len(df), "gate": cfg.__dict__ | {"disabled": sorted(cfg.disabled)},
                                                         "reused_candidates_from": source_run,
                                                         "reused_candidates_device": src_manifest.get("device"),
                                                         "thresholds_from_gate_run": gate_run,
                                                         "rewriters": src_manifest.get("config", {}).get("rewriters"),
                                                         "detectors": src_manifest.get("config", {}).get("detectors")},
                         seed=13, dataset_files=[PROC / "mitigation_eval.parquet", src / "mitigation_candidates.parquet"],
                         models=src_manifest.get("models", {}))
    log(f"mitigation re-decision {tracker.run_id}: {len(df)} inputs from {source_run}, thresholds {overrides}")
    static = {iid: decide(s, cfg, allow_retry=False) for iid, s in sigs.items()}
    dec = {iid: decide(s, cfg, allow_retry=True) for iid, s in sigs.items()}
    missing = []
    for attempt in range(1, cfg.max_retries + 1):
        todo = [iid for iid, d in dec.items() if d.decision == RETRY]
        if not todo:
            break
        log(f"retry {attempt}: {len(todo)} inputs (stored rewrites)")
        for iid in todo:
            new = [s for s in all_sigs[iid] if s.candidate.endswith(f"_retry{attempt}")]
            if not new:
                missing.append((iid, attempt))
                continue
            sigs[iid] = sigs[iid] + new
            dec[iid] = decide(sigs[iid], cfg, allow_retry=True, attempts=attempt)
    if missing:
        tracker.finish(status="failed", metrics={"missing_stored_retries": len(missing)})
        raise SystemExit(f"{len(missing)} retries needed that the source run never generated, e.g. {missing[:10]}")
    for iid, d in dec.items():
        if d.decision == RETRY:
            final = decide(sigs[iid], cfg, allow_retry=False, attempts=d.attempts)
            if final.decision != ACCEPT:
                final.reason_code, final.reason = "RETRIES_EXHAUSTED", "No candidate passed after strict retries; escalated to human review."
            dec[iid] = final

    stored_rows = {(r.input_id, r.candidate): r for r in stored.itertuples()}
    rows = []
    for iid, ss in sigs.items():
        biased = static[iid].input_biased
        for s in ss:
            v = judge(s, cfg, biased)
            r = stored_rows[(iid, s.candidate)]
            rows.append({"input_id": iid, "source": source[iid], "gold_label": int(label[iid]), "input_text": texts[iid],
                         **{k: val for k, val in s.to_dict().items() if k not in {"fact_checks", "detector_scores_before", "detector_scores_after"}},
                         "fact_checks": r.fact_checks, "detector_scores_after": r.detector_scores_after,
                         "detector_scores_before": r.detector_scores_before,
                         "verdict": v.decision, "reasons": ",".join(v.reasons), "input_biased_pred": biased,
                         "ref_similarity": r.ref_similarity, "final_decision": dec[iid].decision, "final_choice": dec[iid].chosen})
    cand_df = pd.DataFrame(rows)
    rewriters = {k: None for k in (src_manifest.get("config", {}).get("rewriters") or {})}
    summary = summarise(cand_df, sigs, dec, static, label, cfg, rewriters)
    summary["run_id"] = tracker.run_id
    summary["reused_candidates_from"] = source_run
    tracker.save_artifact("mitigation_candidates.parquet", cand_df)
    tracker.save_artifact("mitigation_summary.json", summary)
    tracker.finish(metrics={"gate_final": summary["gate_final_distribution"],
                            "policies": {k: {m: v.get(m) for m in ["accepted_fact_failure_rate", "unnecessary_change_rate"]}
                                         for k, v in summary["policies"].items()}})
    log(f"done: {tracker.run_id}")
    return tracker.run_id


def summarise(cand: pd.DataFrame, sigs, dec, static, label, cfg, rewriters) -> dict:
    out: dict = {"n_inputs": len(sigs), "n_biased_gold": int(sum(label.values())),
                 "n_neutral_gold": int(len(label) - sum(label.values()))}
    first = cand[~cand.candidate.str.contains("retry")]
    per_rw = {}
    for name, g in first.groupby("candidate"):
        b = g[g.gold_label == 1]
        n = g[g.gold_label == 0]
        per_rw[name] = {
            "n": len(g), "mean_bias_after_biased": float(b.bias_after.mean()) if len(b) else None,
            "mean_bias_reduction_biased": float(b.bias_reduction.mean()) if len(b) else None,
            "bias_reduction_ci": M.bootstrap_ci(b.bias_reduction.to_numpy()) if len(b) else None,
            "semantic_similarity": float(g[g.changed].semantic_similarity.mean()) if g.changed.any() else 1.0,
            "entailment_both": float(g[g.changed].entailment_both_ways.mean()) if g.changed.any() else 1.0,
            "fact_retention": float(g.fact_retention.mean()), "fact_failure_rate": float((g.fact_retention < 1).mean()),
            "fluency": float(g.fluency.mean()), "length_change": float((g.length_ratio - 1).mean()),
            "changed_rate": float(g.changed.mean()),
            "unnecessary_change_rate_neutral": float(n.changed.mean()) if len(n) else None,
            "ref_similarity": float(g.ref_similarity.dropna().mean()) if g.ref_similarity.notna().any() else None,
            "verdicts": dict(Counter(g.verdict)),
        }
    out["rewriters"] = per_rw
    out["candidate_verdict_distribution"] = dict(Counter(first[first.candidate != "original"].verdict))
    out["gate_final_distribution"] = dict(Counter(d.decision for d in dec.values()))
    out["gate_final_reasons"] = dict(Counter(d.reason_code for d in dec.values()))
    out["gate_static_distribution"] = dict(Counter(d.decision for d in static.values()))
    out["retry"] = {"inputs_retried": int(sum(1 for d in dec.values() if d.attempts > 0)),
                    "accepted_after_retry": int(sum(1 for d in dec.values() if d.attempts > 0 and d.decision == ACCEPT)),
                    "chosen_retry_candidate": int(sum(1 for d in dec.values() if d.chosen and "retry" in d.chosen))}
    out["input_detection"] = {"pred_biased_rate_on_gold_biased": float(np.mean([static[i].input_biased for i in sigs if label[i] == 1])),
                              "pred_biased_rate_on_gold_neutral": float(np.mean([static[i].input_biased for i in sigs if label[i] == 0]))
                              if any(label[i] == 0 for i in sigs) else None}

    first_llm = next((k for k in rewriters if k != "rule_based"), "rule_based")
    policies = {"fairlens_gate": None, "fairlens_gate_retry": None, "lowest_bias": B.lowest_bias,
                "highest_similarity": B.highest_similarity, f"always_{first_llm}": B.always_first(first_llm)}
    policies.update({f"random_s{s}": B.random_policy(s) for s in range(5)})
    by_input = {iid: {r.candidate: r for r in g.itertuples()} for iid, g in cand.groupby("input_id")}
    pol_out = {}
    for pname, pol in policies.items():
        choices = {}
        for iid, ss in sigs.items():
            base = [s for s in ss if "retry" not in s.candidate]
            if pname == "fairlens_gate":
                d = static[iid]
                choices[iid] = d.chosen if d.decision == ACCEPT else None
            elif pname == "fairlens_gate_retry":
                d = dec[iid]
                choices[iid] = d.chosen if d.decision == ACCEPT else None
            else:
                choices[iid] = pol(base)
        pol_out[pname] = policy_metrics(choices, by_input, label)
    rnd = [k for k in pol_out if k.startswith("random_s")]
    pol_out["random_mean"] = {m: float(np.mean([pol_out[k][m] for k in rnd if pol_out[k][m] is not None]))
                              for m in pol_out[rnd[0]] if isinstance(pol_out[rnd[0]][m], (int, float))}
    for k in rnd:
        pol_out.pop(k)
    out["policies"] = pol_out
    return out


def policy_metrics(choices: dict, by_input: dict, label: dict) -> dict:
    biased = [i for i in choices if label[i] == 1]
    neutral = [i for i in choices if label[i] == 0]
    chosen = {i: by_input[i][c] for i, c in choices.items() if c is not None}
    def rate(ids, pred):
        ids = [i for i in ids if i in chosen]
        return float(np.mean([pred(chosen[i]) for i in ids])) if ids else None
    acc_b = [i for i in biased if i in chosen]
    return {
        "defer_rate": float(np.mean([c is None for c in choices.values()])),
        "accepted_fact_failure_rate": rate(list(choices), lambda r: r.fact_retention < 1),
        "accepted_contradiction_rate": rate(list(choices), lambda r: r.contradiction > 0.5),
        "unnecessary_change_rate": float(np.mean([i in chosen and bool(chosen[i].changed) for i in neutral])) if neutral else None,
        "mean_bias_after_biased": rate(biased, lambda r: r.bias_after),
        "mean_bias_reduction_biased": rate(biased, lambda r: r.bias_reduction),
        "biased_left_unchanged_rate": float(np.mean([i in chosen and not chosen[i].changed for i in biased])) if biased else None,
        "mean_semantic_similarity": rate(list(choices), lambda r: r.semantic_similarity),
        "mean_ref_similarity": float(np.nanmean([chosen[i].ref_similarity for i in acc_b if chosen[i].ref_similarity is not None]))
        if any(chosen[i].ref_similarity is not None for i in acc_b) else None,
        "n_chosen": len(chosen),
    }
