"""V4 A.3 blinded human evaluation: export annotation sheets and score returned labels.

  python scripts/human_eval.py --export [--n 120]       # from the latest benchmark_mitigation run
  python scripts/human_eval.py --score outputs/human_eval/<dir>   # after annotators return labels_*.csv

Export: stratified sample of real changed rewrites (per rewriter and Gate verdict), shuffled, blinded.
Score: inter-annotator agreement (Cohen/Fleiss kappa), majority labels, Gate vs LLM-judge vs human table.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fairlens.experiments.tracking import RunTracker, latest_run  # noqa: E402

QUESTIONS = ["facts_kept", "bias_reduced", "acceptable"]


def export(n: int) -> Path:
    mr = latest_run("benchmark_mitigation")
    if not mr:
        sys.exit("no completed benchmark_mitigation run")
    cand = pd.read_parquet(Path(mr["manifest_path"]).parent / "mitigation_candidates.parquet")
    cand = cand[cand.changed].copy()
    cand["rewriter"] = cand.candidate.str.replace(r"_retry\d", "", regex=True)
    jr = latest_run("judge_validation")
    if jr:
        jp = Path(jr["manifest_path"]).parent / "judge_scores_mitigation.parquet"
        if jp.exists():
            js = pd.read_parquet(jp)
            cand = cand.merge(js[["input_text", "text", "p_yes"]], on=["input_text", "text"], how="left")
    per = max(1, n // max(1, cand.groupby(["rewriter", "verdict"]).ngroups))
    s = cand.groupby(["rewriter", "verdict"], group_keys=False).apply(lambda g: g.sample(min(len(g), per), random_state=7))
    s = s.sample(frac=1, random_state=7).reset_index(drop=True)
    s["row_id"] = [f"r{i:03d}" for i in range(len(s))]
    out = ROOT / "outputs" / "human_eval" / datetime.now().strftime("%Y%m%d")
    out.mkdir(parents=True, exist_ok=True)
    blind = s[["row_id", "input_text", "text"]].rename(columns={"input_text": "original", "text": "rewrite"})
    for q in QUESTIONS:
        blind[q] = ""
    blind["comment"] = ""
    blind.head(20).to_csv(out / "pilot.csv", index=False, encoding="utf-8")
    blind.iloc[20:].to_csv(out / "main.csv", index=False, encoding="utf-8")
    key = s[["row_id", "input_id", "rewriter", "candidate", "verdict", "reasons"] + (["p_yes"] if "p_yes" in s else [])]
    key.to_csv(out / "key.csv", index=False, encoding="utf-8")
    (out / "README.txt").write_text("Annotators: read docs/HUMAN_EVAL_GUIDELINE.md. Do NOT open key.csv.\n"
                                    f"Source mitigation run: {mr['run_id']}\n", encoding="utf-8")
    print(f"exported {len(s)} rows to {out}")
    return out


def score(folder: Path) -> None:
    from sklearn.metrics import cohen_kappa_score

    files = sorted(folder.glob("labels_*.csv"))
    if len(files) < 2:
        sys.exit("need at least two labels_<initials>.csv files")
    key = pd.read_csv(folder / "key.csv")
    labs = {f.stem.split("_", 1)[1]: pd.read_csv(f).set_index("row_id") for f in files}
    ids = sorted(set.intersection(*(set(d.index) for d in labs.values())) & set(key.row_id))
    res = {"annotators": list(labs), "n_rows": len(ids), "kappa": {}}
    names = list(labs)
    for q in QUESTIONS:
        if len(names) == 2:
            a, b = (labs[n].loc[ids, q].astype(str).str.strip().str.lower() for n in names)
            res["kappa"][q] = float(cohen_kappa_score(a, b))
        else:
            cats = sorted(set(np.concatenate([labs[n].loc[ids, q].astype(str).str.lower().values for n in names])))
            tab = np.array([[sum(labs[n].loc[i, q].lower() == c for n in names) for c in cats] for i in ids])
            P = ((tab ** 2).sum(1) - len(names)) / (len(names) * (len(names) - 1))
            pj = tab.sum(0) / tab.sum()
            res["kappa"][q] = float((P.mean() - (pj ** 2).sum()) / (1 - (pj ** 2).sum()))
    maj = pd.DataFrame({q: [pd.Series([labs[n].loc[i, q].lower() for n in names]).mode().sort_values().iloc[-1 if q == "acceptable" else 0]
                            for i in ids] for q in QUESTIONS}, index=ids)
    k = key.set_index("row_id").loc[ids]
    human_acc = (maj.acceptable == "accept").astype(int)
    gate_acc = (k.verdict == "ACCEPT").astype(int)
    rows = [{"method": "FairLens Gate", "agreement_with_humans": float((gate_acc == human_acc).mean()),
             "kappa": float(cohen_kappa_score(gate_acc, human_acc))}]
    if "p_yes" in k:
        jr = latest_run("judge_validation")
        thr = json.loads((Path(jr["manifest_path"]).parent / "judge_validation.json").read_text(encoding="utf-8"))["threshold"]
        judge_acc = (k.p_yes >= thr).astype(int)
        rows.append({"method": "LLM judge (facts kept)", "agreement_with_humans": float((judge_acc == human_acc).mean()),
                     "kappa": float(cohen_kappa_score(judge_acc, human_acc))})
    res["method_vs_human"] = rows
    t = RunTracker("human_eval", config={"folder": str(folder.relative_to(ROOT).as_posix()), "annotators": names}, seed=0,
                   dataset_files=[folder / "key.csv"] + files)
    t.save_artifact("human_eval_summary.json", res)
    t.save_artifact("majority_labels.csv", maj.reset_index().rename(columns={"index": "row_id"}))
    t.finish(metrics=res)
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--score", type=Path)
    a = ap.parse_args()
    if a.export:
        export(a.n)
    if a.score:
        score(a.score)


if __name__ == "__main__":
    main()
