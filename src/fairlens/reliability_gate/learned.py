"""Learned gate variant: logistic regression or a shallow decision tree over gate signals,
fit on a validation split of the controlled gate set and compared with the rule gate on test."""
from __future__ import annotations

import numpy as np

from .rules import ACCEPT, HUMAN_REVIEW, REJECT, RETRY
from .signals import FEATURE_GROUPS, FEATURE_NAMES

CLASSES = [ACCEPT, REJECT, RETRY]


class LearnedGate:
    def __init__(self, kind: str = "logreg", abstain_below: float = 0.6, drop_groups: tuple[str, ...] = (), seed: int = 0):
        self.kind, self.abstain_below, self.seed = kind, abstain_below, seed
        dropped = {f for g in drop_groups for f in FEATURE_GROUPS[g]}
        self.cols = [i for i, f in enumerate(FEATURE_NAMES) if f not in dropped]
        self.model = None

    def _make(self):
        if self.kind == "logreg":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0,
                                                                      random_state=self.seed))
        from sklearn.tree import DecisionTreeClassifier

        return DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, class_weight="balanced", random_state=self.seed)

    def fit(self, X: np.ndarray, y: list[str]) -> "LearnedGate":
        self.model = self._make().fit(np.asarray(X)[:, self.cols], y)
        return self

    def predict(self, X: np.ndarray) -> list[str]:
        P = self.model.predict_proba(np.asarray(X)[:, self.cols])
        classes = list(self.model.classes_)
        out = []
        for row in P:
            j = int(np.argmax(row))
            out.append(HUMAN_REVIEW if row[j] < self.abstain_below else classes[j])
        return out

    def describe(self) -> dict:
        if self.kind == "logreg":
            lr = self.model[-1]
            names = [FEATURE_NAMES[i] for i in self.cols]
            return {"classes": list(lr.classes_), "coef": {c: dict(zip(names, map(float, w))) for c, w in zip(lr.classes_, lr.coef_)}}
        from sklearn.tree import export_text

        return {"tree": export_text(self.model, feature_names=[FEATURE_NAMES[i] for i in self.cols])}
