"""Build the mitigation evaluation set and the controlled Reliability-Gate set.

mitigation_eval.parquet   ~270 inputs: authored hard cases (+references) and benchmark *test-split* sentences
                          (BABE biased/non-biased, StereoSet stereotypes, CrowS-Pairs stereotypes).
gate_controlled.parquet   (input, candidate, gold) triples. Gold is known by construction:
    biased input : reference rewrite -> ACCEPT; unchanged original -> RETRY; seeded perturbations of the
                   reference (number / negation / entity / role / antonym / addition / truncation / quote) -> REJECT
    neutral input: unchanged -> ACCEPT; needless paraphrase or any perturbation -> REJECT
    Each source sentence is assigned once to val (for learned gate + threshold tuning) or test, stratified by
    label with a fixed seed, so all perturbation seeds of a sentence stay in the same half (68 / 68 sentences).
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fairlens.evaluation.facts import NEGATION_CUES, ROLE_WORDS, _nlp  # noqa: E402

OUT = ROOT / "datasets" / "processed"
AUTHORED = ROOT / "datasets" / "curated" / "authored_cases.jsonl"
SEEDS = [11, 22, 33]
SPLIT_SEED = 42

ANTONYMS = {
    "approved": "rejected", "rejected": "approved", "increase": "decrease", "rose": "fell", "fell": "rose",
    "good": "bad", "better": "worse", "worse": "better", "finished": "started", "refused": "accepted",
    "blocked": "approved", "cancelled": "held", "closed": "open", "open": "closed", "criticized": "praised",
    "criticised": "praised", "praised": "criticized", "won": "lost", "lost": "won", "passed": "failed",
    "hired": "fired", "joined": "left", "promoted": "demoted", "strong": "weak", "caring": "careless",
    "fluent": "poor", "valid": "expired", "able": "unable", "accessible": "inaccessible", "only": "also",
    "more": "fewer", "fewer": "more", "before": "after", "after": "before", "against": "for",
    "accepts": "rejects", "offers": "withdraws", "leads": "follows", "manages": "assists", "completed": "abandoned",
    "injured": "unharmed", "deleted": "published", "decisive": "indecisive", "motivated": "unmotivated",
}
SYNONYMS = {
    "approved": "passed", "checked": "reviewed", "closed": "shut", "hired": "recruited", "tested": "trialled",
    "finished": "completed", "manages": "runs", "leads": "heads", "hosts": "holds", "rose": "climbed",
    "fixed": "repaired", "served": "provided", "presented": "showed", "reviewed": "checked", "processed": "handled",
    "joined": "signed up for", "works": "is employed", "explained": "described", "announced": "advertised",
}
ADDITIONS = [
    " The company was later fined for discrimination.", " This was confirmed by 3 independent witnesses.",
    " Critics called the decision illegal.", " The policy was reversed a week later.",
    " Most staff disagreed with this.",
]
FAKE_NAMES = ["Olivia Brandt", "Kwame Mensah", "Lucia Ferreira", "Hiro Tanaka", "Nadia Rahman", "Pavel Novak"]
FAKE_ORGS = ["Crestline Group", "Orbis Labs", "Maple Street Council", "Vantage Bank"]
FAKE_PLACES = ["Glasgow", "Toronto", "Nairobi", "Lyon", "Osaka"]


def load_authored() -> pd.DataFrame:
    return pd.DataFrame([json.loads(l) for l in AUTHORED.read_text(encoding="utf-8").splitlines() if l.strip()])


# --- perturbations (each returns new text or None if not applicable) ------------------------------
def p_number(t, rng):
    nums = list(re.finditer(r"\b\d[\d,]*(?:\.\d+)?\b", t))
    if not nums:
        return None
    m = rng.choice(nums)
    v = m.group(0).replace(",", "")
    new = str(int(float(v)) + rng.randint(1, 9)) if "." not in v else str(round(float(v) + rng.randint(1, 9) / 2, 1))
    return t[:m.start()] + new + t[m.end():]


def p_negation(t, rng):
    if re.search(NEGATION_CUES, t, re.I):
        out = re.sub(r"\b(does|do|did|is|are|was|were|will|can|should|must|has|have) not\b", r"\1", t, count=1, flags=re.I)
        out = re.sub(r"\bn't\b", "", out, count=1) if out == t else out
        out = re.sub(r"\bnever\b", "always", out, count=1, flags=re.I) if out == t else out
        out = re.sub(r"\bno one\b", "everyone", out, count=1, flags=re.I) if out == t else out
        out = re.sub(r"\bno\b", "some", out, count=1, flags=re.I) if out == t else out
        out = re.sub(r"\bcannot\b", "can", out, count=1, flags=re.I) if out == t else out
        return out if out != t else None
    m = re.search(r"\b(is|are|was|were|will|can|must|should)\b", t)
    if m:
        return t[:m.end()] + " not" + t[m.end():]
    return None


def p_entity(t, rng):
    nlp = _nlp()
    ents = [e for e in nlp(t).ents if e.label_ in {"PERSON", "ORG", "GPE"}] if nlp else []
    if not ents:
        return None
    e = rng.choice(ents)
    pool = {"PERSON": FAKE_NAMES, "ORG": FAKE_ORGS, "GPE": FAKE_PLACES}[e.label_]
    repl = rng.choice([p for p in pool if p.lower() != e.text.lower()])
    return t[:e.start_char] + repl + t[e.end_char:]


def p_role(t, rng):
    words = [m for m in re.finditer(r"\b[a-z]+\b", t) if m.group(0) in ROLE_WORDS]
    if not words:
        return None
    m = rng.choice(words)
    repl = rng.choice(sorted(r for r in ROLE_WORDS if " " not in r and r != m.group(0)))
    return t[:m.start()] + repl + t[m.end():]


def p_antonym(t, rng):
    hits = [m for m in re.finditer(r"\b[A-Za-z]+\b", t) if m.group(0).lower() in ANTONYMS]
    if not hits:
        return None
    m = rng.choice(hits)
    rep = ANTONYMS[m.group(0).lower()]
    rep = rep[0].upper() + rep[1:] if m.group(0)[0].isupper() else rep
    out = t[:m.start()] + rep + t[m.end():]
    out = re.sub(r"\b([Aa])n (?=[^aeiouAEIOU\W])", r"\1 ", out)
    return re.sub(r"\b([Aa]) (?=[aeiouAEIOU])", r"\1n ", out)


def p_addition(t, rng):
    return t.rstrip() + rng.choice(ADDITIONS)


def p_truncate(t, rng):
    w = t.split()
    if len(w) < 6:
        return None
    cut = max(2, len(w) // 2 - rng.randint(0, 1))
    return " ".join(w[:cut]).rstrip(",;") + "."


def p_quote(t, rng):
    m = re.search(r"\"([^\"]+)\"", t)
    if not m:
        return None
    inner = m.group(1).split()
    if len(inner) < 2:
        return None
    inner[rng.randrange(len(inner))] = rng.choice(["sometimes", "rarely", "often"])
    return t[:m.start(1)] + " ".join(inner) + t[m.end(1):]


def p_paraphrase(t, rng):
    hits = [m for m in re.finditer(r"\b[a-z]+\b", t) if m.group(0) in SYNONYMS]
    if hits:
        m = rng.choice(hits)
        return t[:m.start()] + SYNONYMS[m.group(0)] + t[m.end():]
    return "Notably, " + t[0].lower() + t[1:]


PERTURB = {"number_changed": p_number, "negation_flipped": p_negation, "entity_swapped": p_entity,
           "role_swapped": p_role, "antonym": p_antonym, "hallucinated_addition": p_addition,
           "truncated": p_truncate, "quote_altered": p_quote}


def sentence_split(auth: pd.DataFrame) -> dict[str, str]:
    """One val/test assignment per source sentence, stratified by label, shared by every perturbation seed.

    Odd-sized label groups are balanced in opposite directions (biased: extra sentence to val; neutral: extra
    sentence to test) so that val and test each hold half of the sentences.
    """
    rng = random.Random(SPLIT_SEED)
    split = {}
    for label, extra_to_val in ((1, True), (0, False)):
        ids = sorted(auth.loc[auth.label == label, "id"])
        rng.shuffle(ids)
        n_val = (len(ids) + 1) // 2 if extra_to_val else len(ids) // 2
        split.update({i: ("val" if k < n_val else "test") for k, i in enumerate(ids)})
    return split


def check_no_leakage(gate: pd.DataFrame) -> None:
    """Fail if any source sentence (or its reference rewrite) appears in both val and test."""
    halves = gate.groupby("input_id").split.nunique()
    leaked = sorted(halves[halves > 1].index)
    texts = gate.groupby("input_text").split.nunique()
    if leaked or (texts > 1).any():
        raise SystemExit(f"gate_controlled leakage: {len(leaked)} sentences in both val and test, e.g. {leaked[:5]}")


def controlled(auth: pd.DataFrame, seed: int, split: dict[str, str]) -> pd.DataFrame:
    rng = random.Random(seed)
    ids = sorted(auth["id"])
    # Kept only to advance the generator exactly as before, so every perturbation text is unchanged;
    # the val/test assignment now comes from sentence_split().
    rng.shuffle(ids)
    rows = []
    for r in auth.itertuples():
        base = {"input_id": r.id, "input_text": r.text, "input_label": r.label, "bias_type": r.bias_type,
                "split": split[r.id], "seed": seed}
        if r.label == 1:
            rows.append({**base, "candidate": "reference", "candidate_text": r.reference, "gold": "ACCEPT", "perturbation": "none"})
            rows.append({**base, "candidate": "original", "candidate_text": r.text, "gold": "RETRY", "perturbation": "unchanged"})
            src = r.reference
        else:
            rows.append({**base, "candidate": "original", "candidate_text": r.text, "gold": "ACCEPT", "perturbation": "unchanged"})
            rows.append({**base, "candidate": "paraphrase", "candidate_text": p_paraphrase(r.text, rng), "gold": "REJECT",
                         "perturbation": "needless_paraphrase"})
            src = r.text
        for name, fn in PERTURB.items():
            out = fn(src, rng)
            if out and out.strip() != src.strip():
                rows.append({**base, "candidate": name, "candidate_text": out, "gold": "REJECT", "perturbation": name})
    return pd.DataFrame(rows)


def mitigation_eval(auth: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    det = pd.read_parquet(OUT / "detection.parquet")
    test = det[det.split == "test"]
    parts = [auth.assign(reference=auth.reference.where(auth.label == 1, auth.text))]

    def pick(df, n, tag):
        df = df[df.text.str.split().str.len().between(5, 45)]
        s = df.sample(min(n, len(df)), random_state=seed)
        return pd.DataFrame({"id": [f"{tag}-{i}" for i in range(len(s))], "text": s.text.values, "label": s.label.values,
                             "bias_type": s.bias_type.values, "tags": [[] for _ in range(len(s))], "reference": None,
                             "source": s.source.values})

    parts.append(pick(test[(test.source == "babe") & (test.label == 1)], 50, "babe-b"))
    parts.append(pick(test[(test.source == "babe") & (test.label == 0)], 20, "babe-n"))
    parts.append(pick(test[test.source.str.startswith("stereoset") & (test.label == 1)], 40, "ss-b"))
    parts.append(pick(test[(test.source == "crows_pairs") & (test.label == 1)], 30, "cp-b"))
    return pd.concat(parts, ignore_index=True)


def main():
    auth = load_authored()
    mit = mitigation_eval(auth)
    mit.to_parquet(OUT / "mitigation_eval.parquet", index=False)
    split = sentence_split(auth)
    gate = pd.concat([controlled(auth, s, split) for s in SEEDS], ignore_index=True)
    check_no_leakage(gate)
    gate.to_parquet(OUT / "gate_controlled.parquet", index=False)
    print("mitigation_eval:", len(mit), mit.groupby(["source", "label"]).size().to_dict())
    g = gate[gate.seed == SEEDS[0]]
    print("gate_controlled (seed 11):", len(g), g.groupby(["gold"]).size().to_dict())
    print(g.groupby("perturbation").size().to_dict())
    for p in ["number_changed", "negation_flipped", "entity_swapped", "antonym", "truncated", "quote_altered"]:
        ex = g[g.perturbation == p].iloc[0]
        print(f"  {p}: {ex.input_text!r}\n     -> {ex.candidate_text!r}")


def resplit_only() -> None:
    """Re-assign val/test in the existing gate_controlled.parquet without regenerating candidates.

    Needs no spaCy. Candidate texts are untouched; the authored file must match the parquet's inputs.
    """
    path = OUT / "gate_controlled.parquet"
    gate = pd.read_parquet(path)
    auth = load_authored()
    stored = gate.drop_duplicates("input_id").set_index("input_id")
    if set(stored.index) != set(auth["id"]) or any(
            stored.loc[r.id, "input_label"] != r.label or stored.loc[r.id, "input_text"] != r.text for r in auth.itertuples()):
        raise SystemExit("authored_cases.jsonl does not match gate_controlled.parquet; run the full build instead")
    before = gate.groupby("input_id").split.nunique().gt(1).sum()
    split = sentence_split(auth)
    gate["split"] = gate.input_id.map(split)
    check_no_leakage(gate)
    gate.to_parquet(path, index=False)
    u = gate.drop_duplicates(["input_id", "seed"])
    print(f"resplit: sentences in both halves before {before}, after 0")
    print("sentences per split:", gate.drop_duplicates("input_id").groupby(["split", "input_label"]).size().to_dict())
    print("inputs (sentence x seed) per split:", u.groupby(["split", "input_label"]).size().to_dict())
    print("candidates per split:", gate.groupby("split").size().to_dict())


if __name__ == "__main__":
    if "--resplit-only" in sys.argv:
        resplit_only()
    else:
        main()
