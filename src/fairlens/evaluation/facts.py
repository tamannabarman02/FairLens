"""Fact-retention checker: entities, numbers, dates, roles, negation and quotations.

The checker extracts "facts" from the original text and tests whether each survives in the
rewrite. Gendered role words are normalised to a neutral role family first, so
"salesman" -> "salesperson" counts as the role being *kept* (that change is the point of the
rewrite), while "nurse" -> "doctor" counts as the role being *changed*.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from ..detection.lexicon import GENDERED_TERMS

NEGATION_CUES = r"\b(?:not|no|never|none|nobody|nothing|neither|nor|without|cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|shouldn't|mustn't|hasn't|haven't|n't)\b"
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "hundred": 100, "thousand": 1000, "million": 10**6, "billion": 10**9,
}
MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
WEEKDAYS = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"

ROLE_WORDS = {
    "nurse", "doctor", "physician", "surgeon", "engineer", "developer", "programmer", "manager", "director",
    "ceo", "secretary", "assistant", "teacher", "professor", "lecturer", "student", "lawyer", "attorney",
    "judge", "pilot", "driver", "chef", "cook", "cleaner", "receptionist", "accountant", "analyst",
    "scientist", "researcher", "designer", "architect", "mechanic", "electrician", "plumber", "carpenter",
    "farmer", "soldier", "officer", "detective", "journalist", "reporter", "editor", "writer", "author",
    "artist", "musician", "singer", "actor", "dancer", "athlete", "coach", "player", "captain", "president",
    "senator", "minister", "governor", "mayor", "candidate", "employee", "employer", "worker", "intern",
    "supervisor", "cashier", "clerk", "librarian", "pharmacist", "dentist", "therapist", "consultant",
    "salesperson", "firefighter", "chairperson", "spokesperson", "businessperson", "police officer",
    "flight attendant", "server", "homemaker", "babysitter", "nanny", "janitor", "guard", "tailor",
    "baker", "banker", "economist", "auditor", "paramedic", "midwife", "caregiver", "veterinarian",
}


def _role_family(word: str) -> str:
    w = word.lower()
    w = GENDERED_TERMS.get(w, w)
    return {"salespeople": "salesperson", "firefighters": "firefighter"}.get(w, w.rstrip("s") if w.endswith("s") and w[:-1] in ROLE_WORDS else w)


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    try:
        return spacy.load("en_core_web_sm", disable=["lemmatizer"])
    except OSError:
        return None


@dataclass
class FactReport:
    entities: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    quotes: list[str] = field(default_factory=list)
    negations: int = 0

    def to_dict(self):
        return asdict(self)

    def as_hints(self) -> list[str]:
        return list(dict.fromkeys(self.entities + self.numbers + self.dates + self.quotes))


def _norm_number(tok: str) -> str:
    t = tok.lower().replace(",", "")
    if t in NUMBER_WORDS:
        return str(NUMBER_WORDS[t])
    return t.rstrip("%").rstrip(".")


def extract(text: str) -> FactReport:
    rep = FactReport()
    nlp = _nlp()
    if nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW"}:
                rep.entities.append(ent.text.strip())
            elif ent.label_ == "DATE":
                rep.dates.append(ent.text.strip())
    for m in re.finditer(rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS}|{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?|{WEEKDAYS}|\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}/\d{{2,4}})\b", text, re.I):
        if not any(m.group(0).lower() in d.lower() for d in rep.dates):
            rep.dates.append(m.group(0))
    nums = re.findall(r"\b\d[\d,]*(?:\.\d+)?%?|\b(?:" + "|".join(NUMBER_WORDS) + r")\b", text, re.I)
    rep.numbers = list(dict.fromkeys(nums))
    rep.roles = list(dict.fromkeys(_role_family(m.group(0)) for m in re.finditer(r"\b[\w-]+(?: officer| attendant)?\b", text)
                                    if _role_family(m.group(0)) in ROLE_WORDS))
    rep.quotes = [q.strip() for q in re.findall(r"[\"“]([^\"”]{3,})[\"”]", text)]
    rep.negations = len(re.findall(NEGATION_CUES, text, re.I))
    rep.entities = list(dict.fromkeys(rep.entities))
    rep.dates = list(dict.fromkeys(rep.dates))
    return rep


def check(original: str, rewrite: str) -> dict:
    """Compare facts of `original` against `rewrite`.

    Returns per-type kept/missing lists, pass flags and an overall fact_retention score in [0, 1]
    (share of extracted facts that survive; negation polarity counts as one fact).
    """
    src = extract(original)
    low = rewrite.lower()
    rw_nums = {_norm_number(n) for n in re.findall(r"\b\d[\d,]*(?:\.\d+)?%?|\b(?:" + "|".join(NUMBER_WORDS) + r")\b", rewrite, re.I)}
    rw_roles = {_role_family(m.group(0)) for m in re.finditer(r"\b[\w-]+(?: officer| attendant)?\b", rewrite)}

    def kept(items, pred):
        k = [i for i in items if pred(i)]
        return k, [i for i in items if i not in k]

    ent_k, ent_m = kept(src.entities, lambda e: e.lower() in low)
    num_k, num_m = kept(src.numbers, lambda n: _norm_number(n) in rw_nums)
    date_k, date_m = kept(src.dates, lambda d: d.lower() in low or all(_norm_number(x) in rw_nums for x in re.findall(r"\d+", d)) and bool(re.findall(r"\d+", d)))
    role_k, role_m = kept(src.roles, lambda r: r in rw_roles)
    quote_k, quote_m = kept(src.quotes, lambda q: q.lower() in low)
    neg_rw = len(re.findall(NEGATION_CUES, rewrite, re.I))
    negation_ok = (src.negations > 0) == (neg_rw > 0) and (src.negations % 2) == (neg_rw % 2)

    total = len(src.entities) + len(src.numbers) + len(src.dates) + len(src.roles) + len(src.quotes) + 1
    kept_n = len(ent_k) + len(num_k) + len(date_k) + len(role_k) + len(quote_k) + int(negation_ok)
    return {
        "fact_retention": kept_n / total,
        "checks": {
            "entities": {"pass": not ent_m, "kept": ent_k, "missing": ent_m},
            "numbers": {"pass": not num_m, "kept": num_k, "missing": num_m},
            "dates": {"pass": not date_m, "kept": date_k, "missing": date_m},
            "roles": {"pass": not role_m, "kept": role_k, "missing": role_m},
            "quotes": {"pass": not quote_m, "kept": quote_k, "missing": quote_m},
            "negation": {"pass": negation_ok, "original": src.negations, "rewrite": neg_rw},
        },
        "n_facts": total,
    }
