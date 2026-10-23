"""One prompt-template family shared by all LLM rewriters (fairness: same instructions for every model)."""
from __future__ import annotations

BASE_INSTRUCTION = (
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


def instruction(strict: bool) -> str:
    return STRICT_INSTRUCTION if strict else BASE_INSTRUCTION


def facts_hint(facts: list[str]) -> str:
    return ("These exact details must appear unchanged: " + "; ".join(facts) + ".") if facts else ""


def seq2seq_prompt(text: str, strict: bool = False, facts: list[str] | None = None) -> str:
    hint = facts_hint(facts or []) if strict else ""
    return f"{instruction(strict)} {hint}\n\nText: {text}\n\nRewritten text:".replace("  ", " ")


def chat_messages(text: str, strict: bool = False, facts: list[str] | None = None) -> list[dict]:
    hint = facts_hint(facts or []) if strict else ""
    return [
        {"role": "system", "content": "You are a careful editor who removes bias from text without changing its facts."},
        {"role": "user", "content": f"{instruction(strict)} {hint}\n\nText: {text}".replace("  ", " ")},
    ]
