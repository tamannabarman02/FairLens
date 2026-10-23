"""One prompt-template family shared by all LLM rewriters (fairness: same instructions and, when enabled,
the same demonstrations for every model; only the model-specific wrapping differs)."""
from __future__ import annotations

DEFAULT_INSTRUCTION = BASE_INSTRUCTION = (
    "Rewrite the text so that it does not contain stereotypes, biased or exclusionary language. "
    "Keep the original meaning, all facts, names, numbers, dates, roles and negations. "
    "If the text is already neutral, return it unchanged. Return only the rewritten text."
)

STRICT_INSTRUCTION = (
    "Rewrite the text so that it does not contain stereotypes, biased or exclusionary language. "
    "Change ONLY the biased words. Do NOT change, add or remove any names, numbers, dates, job "
    "requirements, roles, quotations or negations (words like 'not', 'never', 'no'). "
    "Return only the rewritten text."
)

# Demonstrations written for the prompt (not taken from any benchmark or the evaluation set).
DEMOS = [
    ("We need a strong fireman who can lift 30 kg.", "We need a strong firefighter who can lift 30 kg."),
    ("Girls are bad at physics, so Mia skipped the 2 labs.", "Mia skipped the 2 labs."),
    ("Immigrants are lazy and live on benefits in our town.", "Some people in our town live on benefits."),
]
USE_DEMOS = False


def _load_selected() -> None:
    """Use the variant chosen on the development sample by scripts/select_prompts.py, if present."""
    global BASE_INSTRUCTION, USE_DEMOS
    import json
    from pathlib import Path

    p = Path(__file__).resolve().parents[3] / "configs" / "prompt_selection.json"
    if p.exists():
        sel = json.loads(p.read_text(encoding="utf-8"))
        BASE_INSTRUCTION = sel["instruction"]
        USE_DEMOS = bool(sel.get("use_demos", False))


_load_selected()


def instruction(strict: bool) -> str:
    return STRICT_INSTRUCTION if strict else BASE_INSTRUCTION


def facts_hint(facts: list[str]) -> str:
    return ("These exact details must appear unchanged: " + "; ".join(facts) + ".") if facts else ""


def seq2seq_prompt(text: str, strict: bool = False, facts: list[str] | None = None) -> str:
    hint = facts_hint(facts or []) if strict else ""
    demos = "".join(f"Text: {a}\nRewritten text: {b}\n\n" for a, b in DEMOS) if USE_DEMOS else ""
    return f"{instruction(strict)} {hint}\n\n{demos}Text: {text}\nRewritten text:".replace("  ", " ")


def chat_messages(text: str, strict: bool = False, facts: list[str] | None = None) -> list[dict]:
    hint = facts_hint(facts or []) if strict else ""
    msgs = [{"role": "system", "content": "You are a careful editor who removes bias from text without changing its facts. "
                                          + instruction(strict)}]
    if USE_DEMOS:
        for a, b in DEMOS:
            msgs += [{"role": "user", "content": f"Text: {a}"}, {"role": "assistant", "content": b}]
    msgs.append({"role": "user", "content": f"{hint}\n\nText: {text}".strip() if hint else f"Text: {text}"})
    return msgs
