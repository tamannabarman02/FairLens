"""Classification, calibration and bootstrap statistics."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def classification(y_true: Sequence[int], scores: Sequence[float], threshold: float = 0.5) -> dict:
    from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)

    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    p = (s >= threshold).astype(int)
    out = {
        "n": int(len(y)), "accuracy": float(accuracy_score(y, p)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "confusion": confusion_matrix(y, p, labels=[0, 1]).tolist(),   # [[TN, FP], [FN, TP]]
        "fpr": float(((p == 1) & (y == 0)).sum() / max(1, (y == 0).sum())),
    }
    if len(set(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, s))
        out["pr_auc"] = float(average_precision_score(y, s))
    out.update(calibration(y, s))
    return out


def curves(y_true, scores, n_points: int = 101) -> dict:
    from sklearn.metrics import precision_recall_curve, roc_curve

    y, s = np.asarray(y_true), np.asarray(scores, dtype=float)
    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)

    def thin(a, b):
        idx = np.unique(np.linspace(0, len(a) - 1, min(n_points, len(a))).astype(int))
        return a[idx].round(4).tolist(), b[idx].round(4).tolist()

    rf, rt = thin(fpr, tpr)
    pr, pp = thin(rec[::-1], prec[::-1])
    return {"roc": {"fpr": rf, "tpr": rt}, "pr": {"recall": pr, "precision": pp}}


def calibration(y_true, scores, n_bins: int = 10) -> dict:
    """Expected calibration error with equal-width bins, plus the reliability-diagram table."""
    y, s = np.asarray(y_true, dtype=float), np.clip(np.asarray(scores, dtype=float), 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(s, bins[1:-1], right=False), 0, n_bins - 1)
    ece, table = 0.0, []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            table.append({"bin": b, "lo": float(bins[b]), "hi": float(bins[b + 1]), "n": 0, "conf": None, "acc": None})
            continue
        conf, acc = float(s[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(conf - acc)
        table.append({"bin": b, "lo": float(bins[b]), "hi": float(bins[b + 1]), "n": int(m.sum()), "conf": conf, "acc": acc})
    brier = float(np.mean((s - y) ** 2))
    return {"ece": float(ece), "brier": brier, "reliability": table}


def bootstrap_ci(values: Sequence[float] | np.ndarray, stat: Callable = np.mean, n: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> dict:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    boots = np.array([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n)])
    return {"mean": float(stat(v)), "lo": float(np.quantile(boots, alpha / 2)),
            "hi": float(np.quantile(boots, 1 - alpha / 2)), "n": int(len(v))}


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], n: int = 1000, seed: int = 0) -> dict:
    """CI of mean(a) - mean(b) over paired items; p = share of resamples where the sign flips."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = a - b
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    p = float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))
    return {"diff": float(d.mean()), "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975)), "p_boot": p}


def f1_from_scores(threshold: float = 0.5):
    def f(y, s):
        from sklearn.metrics import f1_score

        return f1_score(y, (np.asarray(s) >= threshold).astype(int), zero_division=0)
    return f


def bootstrap_metric(y, s, metric: Callable, n: int = 1000, seed: int = 0) -> dict:
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(metric(y[i], s[i]))
    return {"value": float(metric(y, s)), "lo": float(np.quantile(vals, 0.025)), "hi": float(np.quantile(vals, 0.975))}
