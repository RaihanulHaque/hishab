"""Turn LLM templates (generate_llm_templates.py) into training rows by filling the amount
placeholders from ml/hishab/numwords.py. Labels are exact by construction: the same integer
that is rendered into the text is written into the target JSON.

  python ml/data/fill_templates.py --templates ml/datasets/llm_templates_v1.jsonl \
      --out ml/datasets/synthetic_llm_v1.jsonl --per-template 2
  python ml/data/fill_templates.py ... --digit-free --per-template 1   # for an eval set

Output rows use the extra-train format build_splits.py reads (messages + metadata), with the
English translation in metadata.translation for the two_phase format.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hishab.numwords import BN_DIGITS, parse, to_words  # noqa: E402
from hishab.schema import dumps_transactions  # noqa: E402

SYSTEM = ("You are a financial assistant who understands English, Bengali and Banglish(Bengali written in english). "
          "Extract transactions from text into a JSON list.")

# (low, high, round-to) per slot size; most amounts people mention are roundish
SIZE_RANGES = {
    "tiny": (10, 100, 5), "small": (100, 1000, 10), "medium": (1000, 10000, 50),
    "large": (10000, 100000, 500), "huge": (100000, 1000000, 5000),
}


def sample_amount(size: str, rng: random.Random, words: bool) -> int:
    """An amount of this size. Spoken amounts are round ("sare tin hajar", "bish hajar"): when
    the amount will be written in words only round values are drawn; digits may be anything."""
    lo, hi, step = SIZE_RANGES[size]
    r = rng.random()
    if words or r < 0.55:
        if hi >= 1000 and rng.random() < 0.3:      # half units so der / arai / sare appear
            base = max(1000, 10 ** (len(str(lo)) - 1))
            return min(hi, max(lo, (rng.randint(1, 9) * 2 + 1) * base // 2))
        coarse = step * (10 if rng.random() < 0.5 else 2)  # 20,000 / 45,000 rather than 23,500
        v = round(rng.randint(lo, hi) / coarse) * coarse
        return int(min(hi, max(lo, v)))
    return rng.randint(lo, hi)


def pick_style(lang: str, rng: random.Random, digit_free: bool) -> str:
    """words | mix | digits, per language."""
    if digit_free:
        return "words"
    r = rng.random()
    if lang == "english":
        return "words" if r < 0.2 else "mix" if r < 0.4 else "digits"
    return "words" if r < 0.5 else "mix" if r < 0.72 else "digits"


def render(a: int, lang: str, style: str, rng: random.Random) -> str:
    """How the amount appears in the message."""
    words_lang = "bangla" if lang == "bangla" else "banglish"
    if style == "words":
        return to_words(a, words_lang, rng)
    if style == "mix" and a >= 1000:
        return to_words(a, words_lang, rng, mix_digits=True)
    if lang == "bangla" or (lang == "mixed" and rng.random() < 0.5):
        return str(a).translate(BN_DIGITS)
    if a >= 1000 and a % 100 == 0 and rng.random() < 0.3:
        return f"{a / 1000:g}k"
    return f"{a:,}" if rng.random() < 0.2 else str(a)


_PH = re.compile(r"\{(A\d+)\}")

TOTAL_SUFFIX = {
    "banglish": [" mot {T} tk", ", total {T}", " sob miliye {T} taka gelo", ". ajke mot {T}", " = {T} tk total"],
    "mixed": [" মোট {T} টাকা", " total {T} tk", " সব মিলিয়ে {T}", ", mot {T}"],
    "bangla": [" মোট {T} টাকা", " সব মিলিয়ে {T} টাকা গেল", ", মোট {T}"],
    "english": [", total {T}", " that's {T} in total", " ({T} all together)"],
}


def with_total(text: str, lang: str, amounts: list[int], rng: random.Random) -> str:
    """Append a stated total of the expenses; the label must not include it (guideline)."""
    total = sum(amounts)
    shown = render(total, lang, "digits" if rng.random() < 0.7 else "words", rng)
    return text.rstrip(" .।") + rng.choice(TOTAL_SUFFIX[lang]).replace("{T}", shown)


_EMOJI = ["😩", "🙂", "😅", "🥲", "👍", "🤦", "😭", "...", "!!", " :("]


def noise(text: str, protect: list[str], rng: random.Random) -> str:
    """Phone typing: lowercase, a dropped letter, a doubled letter, a missing space, an emoji.
    Never touches the amount strings in `protect`."""
    spans = [(m.start(), m.end()) for p in protect for m in re.finditer(re.escape(p), text)]

    def safe(i: int) -> bool:
        return not any(a <= i < b for a, b in spans)

    chars = list(text)
    if rng.random() < 0.5:
        chars = [c.lower() for c in chars]
    letters = [i for i, c in enumerate(chars) if c.isalpha() and safe(i)]
    if letters and rng.random() < 0.4:   # drop a letter
        chars[rng.choice(letters)] = ""
    if letters and rng.random() < 0.25:  # double a letter
        i = rng.choice(letters)
        chars[i] = chars[i] * 2
    spaces = [i for i, c in enumerate(chars) if c == " " and safe(i) and safe(i + 1)]
    if spaces and rng.random() < 0.35:   # glue two words
        chars[rng.choice(spaces)] = ""
    out = "".join(chars)
    if rng.random() < 0.3:
        out = out.rstrip() + rng.choice(_EMOJI)
    return out


def fill(t: dict, rng: random.Random, digit_free: bool, totals_rate: float = 0.0, noise_rate: float = 0.0) -> dict | None:
    lang = t["lang"]
    styles = {s["id"]: pick_style(lang, rng, digit_free) for s in t["slots"]}
    amounts = {s["id"]: sample_amount(s["size"], rng, words=styles[s["id"]] != "digits") for s in t["slots"]}
    shown = {sid: render(a, lang, styles[sid], rng) for sid, a in amounts.items()}
    for sid, s in shown.items():       # the rendering must read back as the amount we labelled
        if parse(s) != amounts[sid]:
            return None
    text = _PH.sub(lambda m: shown[m.group(1)], t["text"])
    translation = _PH.sub(lambda m: f"{amounts[m.group(1)]:,}", t["translation"])
    order = _PH.findall(t["text"])
    by_id = {s["id"]: s for s in t["slots"]}
    expense_ids = [i for i in order if by_id[i]["type"] == "expense"]
    if len(expense_ids) >= 2 and rng.random() < totals_rate:
        text = with_total(text, lang, [amounts[i] for i in expense_ids], rng)
    if rng.random() < noise_rate:
        text = noise(text, list(shown.values()), rng)
    txs = [{"description": by_id[i]["description"], "amount": float(amounts[i]), "category": by_id[i]["category"],
            "type": by_id[i]["type"]} for i in order]
    n = len(txs)
    complexity = "simple" if n == 1 else "medium" if n == 2 else "complex"
    rid = "txn_" + hashlib.sha1(text.encode()).hexdigest()[:8]
    return {
        "id": rid,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
            {"role": "assistant", "content": dumps_transactions(txs)},
        ],
        "metadata": {
            "lang": lang, "complexity": complexity, "generated_by": t.get("generated_by", "codex-template-v1"),
            "persona": t.get("persona", ""), "translation": translation, "digit_free": digit_free,
            "timestamp": datetime.now(timezone.utc).isoformat(), "validated": True,
            "validation_notes": "amounts rendered by numwords and parsed back; labels exact by construction",
        },
        "raw_transactions": txs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-template", type=int, default=2)
    ap.add_argument("--digit-free", action="store_true", help="amounts in words only (eval sets)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generated-by", default="codex-template-v1")
    ap.add_argument("--totals-rate", type=float, default=0.0, help="share of multi-expense rows that also state a total")
    ap.add_argument("--noise-rate", type=float, default=0.0, help="share of rows with phone-typing noise")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    templates = [json.loads(l) for l in args.templates.read_text().splitlines() if l.strip()]
    seen: set[str] = set()
    rows = []
    for t in templates:
        t["generated_by"] = args.generated_by
        for _ in range(args.per_template):
            for _attempt in range(5):
                r = fill(t, rng, args.digit_free, args.totals_rate, args.noise_rate)
                if r and r["messages"][1]["content"] not in seen:
                    seen.add(r["messages"][1]["content"])
                    rows.append(r)
                    break
    rng.shuffle(rows)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    langs: dict[str, int] = {}
    for r in rows:
        langs[r["metadata"]["lang"]] = langs.get(r["metadata"]["lang"], 0) + 1
    print(f"{len(rows)} rows from {len(templates)} templates -> {args.out}  {langs}")


if __name__ == "__main__":
    main()
