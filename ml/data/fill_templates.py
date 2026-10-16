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


def sample_amount(size: str, rng: random.Random) -> int:
    lo, hi, step = SIZE_RANGES[size]
    r = rng.random()
    if r < 0.55:                       # round: 1500, 20000, 45000
        return max(lo, round(rng.randint(lo, hi) / step) * step)
    if r < 0.8 and hi >= 1000:         # half units: 1500, 3500, 150000 (so der/arai/sare appear)
        base = max(1000, 10 ** (len(str(lo)) - 1))
        return min(hi, max(lo, (rng.randint(1, 9) * 2 + 1) * base // 2))
    return rng.randint(lo, hi)         # arbitrary: 1234, 15750


def render(a: int, lang: str, rng: random.Random, digit_free: bool) -> str:
    """How the amount appears in the message."""
    r = rng.random()
    if lang == "english":
        if digit_free or r < 0.25:
            return to_words(a, "banglish", rng, mix_digits=not digit_free and rng.random() < 0.5)
        if r < 0.5 and a >= 1000 and a % 100 == 0:
            return f"{a / 1000:g}k"
        return f"{a:,}" if r < 0.7 else str(a)
    words_lang = "bangla" if lang == "bangla" else "banglish"
    if digit_free:
        return to_words(a, words_lang, rng)
    if r < 0.5:
        return to_words(a, words_lang, rng)
    if r < 0.75 and a >= 1000:
        return to_words(a, words_lang, rng, mix_digits=True)
    if lang == "bangla" or (lang == "mixed" and rng.random() < 0.5):
        return str(a).translate(BN_DIGITS)
    if a >= 1000 and a % 100 == 0 and rng.random() < 0.3:
        return f"{a / 1000:g}k"
    return f"{a:,}" if rng.random() < 0.2 else str(a)


_PH = re.compile(r"\{(A\d+)\}")


def fill(t: dict, rng: random.Random, digit_free: bool) -> dict | None:
    lang = t["lang"]
    amounts = {s["id"]: sample_amount(s["size"], rng) for s in t["slots"]}
    shown = {sid: render(a, lang, rng, digit_free) for sid, a in amounts.items()}
    for sid, s in shown.items():       # the rendering must read back as the amount we labelled
        if parse(s) != amounts[sid]:
            return None
    text = _PH.sub(lambda m: shown[m.group(1)], t["text"])
    translation = _PH.sub(lambda m: f"{amounts[m.group(1)]:,}", t["translation"])
    order = _PH.findall(t["text"])
    by_id = {s["id"]: s for s in t["slots"]}
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
    args = ap.parse_args()

    rng = random.Random(args.seed)
    templates = [json.loads(l) for l in args.templates.read_text().splitlines() if l.strip()]
    seen: set[str] = set()
    rows = []
    for t in templates:
        t["generated_by"] = args.generated_by
        for _ in range(args.per_template):
            for _attempt in range(5):
                r = fill(t, rng, args.digit_free)
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
