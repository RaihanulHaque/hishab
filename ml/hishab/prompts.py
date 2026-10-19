"""System prompts. Kept short on purpose: on the phone every prompt token costs time."""
from .schema import CATEGORIES

_BASE = (
    "You are Hishab, a Bangladeshi personal-finance assistant. The user writes in English, "
    "Bangla or Banglish (Bangla in Latin letters), often mixed. Extract every financial "
    "transaction they mention as a JSON array of objects with keys description, amount, "
    "category, type. Amounts are BDT numbers (expand shorthand like 1.5k or 2 lakh). "
    f"category is one of: {', '.join(CATEGORIES)}. type is income or expense. "
    "If there is no transaction, the array is empty."
)

_PIPE = (
    "You are Hishab, a Bangladeshi personal-finance assistant. The user writes in English, "
    "Bangla or Banglish (Bangla in Latin letters), often mixed. Extract every financial "
    "transaction they mention, one per line as description|amount|category. amount is a BDT "
    "number (expand shorthand like 1.5k or 2 lakh) written with - for an expense and + for income. "
    f"category is one of: {', '.join(CATEGORIES)}. "
    "If there is no transaction, reply NONE. Reply with the lines only."
)

SYSTEM_PROMPTS = {
    "pipe_signed": _PIPE,
    "json": _BASE + " Reply with the JSON array only.",
    "two_phase": _BASE
    + " Reply in two parts: a line starting with 'TRANSLATION: ' giving a clean English "
    "translation of the message, then a line 'JSON:' followed by the array. Nothing else.",
}


def system_prompt(fmt: str) -> str:
    return SYSTEM_PROMPTS[fmt]


def build_messages(fmt: str, instruction: str, target: str | None = None) -> list[dict]:
    msgs = [
        {"role": "system", "content": system_prompt(fmt)},
        {"role": "user", "content": instruction},
    ]
    if target is not None:
        msgs.append({"role": "assistant", "content": target})
    return msgs
