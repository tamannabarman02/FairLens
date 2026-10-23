"""Sentence segmentation that keeps character offsets and does not split inside quotes or abbreviations."""
from __future__ import annotations

import re
from dataclasses import dataclass

ABBREV = r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|Inc|Ltd|Co|No|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"


@dataclass
class Sentence:
    text: str
    start: int
    end: int


def split_sentences(text: str, max_sentences: int = 200) -> list[Sentence]:
    out: list[Sentence] = []
    in_quote = False
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "\"“”":
            was = in_quote
            in_quote = not in_quote if ch == '"' else (ch == "“")
            if was and not in_quote and i > 0 and text[i - 1] in ".!?":
                nxt = text[i + 1:i + 3]
                if i + 1 >= n or (nxt[:1].isspace() and len(nxt) > 1 and nxt[1].isupper()):
                    _push(out, text, start, i + 1)
                    start = i + 1
            i += 1
            continue
        if ch == "\n" and i + 1 < n and text[i + 1] == "\n":
            _push(out, text, start, i)
            start = i + 2
        elif ch in ".!?" and not in_quote:
            j = i + 1
            while j < n and text[j] in ".!?\"”')":
                j += 1
            nxt = text[j:j + 2]
            before = text[max(0, i - 6):i + 1]
            if (j >= n or (nxt[:1].isspace() and (len(nxt) < 2 or nxt[1].isupper() or nxt[1] in "\"“0123456789"))) \
                    and not re.search(rf"\b{ABBREV}\.$", before) and not re.search(r"\b[A-Z]\.$", before):
                _push(out, text, start, j)
                start = j
                i = j
                continue
        i += 1
    _push(out, text, start, n)
    return out[:max_sentences]


def _push(out: list[Sentence], text: str, s: int, e: int) -> None:
    seg = text[s:e]
    lstrip = len(seg) - len(seg.lstrip())
    seg = seg.strip()
    if seg:
        out.append(Sentence(seg, s + lstrip, s + lstrip + len(seg)))
