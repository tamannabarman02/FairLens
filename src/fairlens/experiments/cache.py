"""On-disk cache of detector scores and rewrites, keyed by (model identity, text).

A model identity includes the model ID, its pinned Hub revision and the compute device, so scores computed
with a different checkpoint or on different hardware (CPU / CUDA / MPS numerics differ slightly) are never mixed.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from ..config import outputs_dir, pinned_revision
from ..hardware import device


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class ScoreCache:
    def __init__(self, path: Path | None = None):
        self.path = path or outputs_dir("cache") / "scores.sqlite"
        with closing(sqlite3.connect(self.path)) as con, con:
            con.execute("CREATE TABLE IF NOT EXISTS scores (model TEXT, k TEXT, score REAL, PRIMARY KEY (model, k))")
            con.execute("CREATE TABLE IF NOT EXISTS gens (model TEXT, k TEXT, payload TEXT, PRIMARY KEY (model, k))")

    def get_many(self, model: str, texts: list[str]) -> dict[str, float]:
        keys = {_key(t): t for t in texts}
        out = {}
        with closing(sqlite3.connect(self.path)) as con, con:
            ks = list(keys)
            for i in range(0, len(ks), 900):
                chunk = ks[i:i + 900]
                q = f"SELECT k, score FROM scores WHERE model=? AND k IN ({','.join('?' * len(chunk))})"
                for k, s in con.execute(q, [model, *chunk]):
                    out[keys[k]] = s
        return out

    def put_many(self, model: str, scores: dict[str, float]) -> None:
        with closing(sqlite3.connect(self.path)) as con, con:
            con.executemany("INSERT OR REPLACE INTO scores VALUES (?,?,?)", [(model, _key(t), float(s)) for t, s in scores.items()])

    def get_gen(self, model: str, prompt_key: str) -> dict | None:
        with closing(sqlite3.connect(self.path)) as con, con:
            r = con.execute("SELECT payload FROM gens WHERE model=? AND k=?", (model, _key(prompt_key))).fetchone()
        return json.loads(r[0]) if r else None

    def put_gen(self, model: str, prompt_key: str, payload: dict) -> None:
        with closing(sqlite3.connect(self.path)) as con, con:
            con.execute("INSERT OR REPLACE INTO gens VALUES (?,?,?)", (model, _key(prompt_key), json.dumps(payload, ensure_ascii=False)))


def detector_identity(det) -> str:
    """Cache key for a detector: name|model|revision + training timestamp, NLI hypotheses and device when relevant."""
    info = det.info()
    model = info.get("model") or ""
    ident = f"{info['name']}|{model}|{info.get('revision') or pinned_revision(model) or ''}"
    tr = info.get("training") or {}
    if tr:
        ident += f"|{tr.get('trained_at', '')}"
    if hasattr(det, "hyp_biased"):
        ident += f"|{det.hyp_biased}|{det.hyp_neutral}"
    if model:
        ident += f"|device={device()}"
    return ident


def generator_identity(rw) -> str:
    """Cache key for a rewriter's generations: model|revision|device (rule-based rewriters: their name)."""
    model = getattr(rw, "model_id", None)
    if not model:
        return getattr(rw, "name", str(rw))
    return f"{model}|{getattr(rw, 'revision', None) or pinned_revision(model) or ''}|device={device()}"


def score_texts(det, texts: list[str], cache: ScoreCache | None = None, batch_size: int = 16,
                progress=None) -> tuple[dict[str, float], float]:
    """Score unique texts with a detector using the cache. Returns ({text: score}, ms/sample for new texts)."""
    cache = cache or ScoreCache()
    ident = detector_identity(det)
    uniq = list(dict.fromkeys(texts))
    have = cache.get_many(ident, uniq)
    todo = [t for t in uniq if t not in have]
    t0 = time.perf_counter()
    for i in range(0, len(todo), batch_size * 8):
        chunk = todo[i:i + batch_size * 8]
        res = det.predict_batch(chunk, batch_size=batch_size)
        new = {t: r.score for t, r in zip(chunk, res)}
        cache.put_many(ident, new)
        have.update(new)
        if progress:
            progress(min(i + len(chunk), len(todo)), len(todo))
    ms = (time.perf_counter() - t0) * 1000 / len(todo) if todo else float("nan")
    return have, ms
