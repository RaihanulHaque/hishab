"""Rewrite digit amounts in existing training rows as words, keeping the labels.

"lunch 250, rickshaw 60" -> "lunch duisho ponchash, rickshaw shat". Only rows where every
labelled amount appears exactly once as a plain integer in the text are used, so the rewrite
is unambiguous; the rendering is parsed back to check it still reads as the same number.

  python ml/data/augment_numbers.py --train ml/datasets/splits/train.jsonl \
      --out ml/datasets/synthetic_numwords_aug_v1.jsonl --max 2000
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
from hishab.numwords import parse, to_words  # noqa: E402
from hishab.schema import dumps_transactions  # noqa: E402

SYSTEM = ("You are a financial assistant who understands English, Bengali and Banglish(Bengali written in english). "
          "Extract transactions from text into a JSON list.")


def rewrite(text: str, amounts: list[float], lang: str, rng: random.Random) -> str | None:
    out = text
    for a in amounts:
        if a != int(a) or a < 10:
            return None
        n = int(a)
        # a currency sign glued to the digits goes too: "৳1500" -> "der hajar", not "৳der hajar"
        pat = re.compile(rf"(?:৳\s?)?(?<![0-9,.])({n}|{n:,})(?![0-9,.])")
        hits = pat.findall(out)
        if len(hits) != 1:
            return None
        words_lang = "bangla" if lang == "bangla" else "banglish"
        mix = n >= 1000 and rng.random() < 0.3
        w = to_words(n, words_lang, rng, mix_digits=mix)
        if parse(w) != n:
            return None
        out = pat.sub(w, out, count=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", nargs="*", default=["base", "hard", "handwritten-v1", "handwritten-v2"])
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in args.train.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["source"] in args.sources and r["language"] != "english" and r["transactions"]]
    rng.shuffle(rows)
    out = []
    seen: set[str] = set()
    for r in rows:
        if len(out) >= args.max:
            break
        text = rewrite(r["instruction"], [t["amount"] for t in r["transactions"]], r["language"], rng)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append({
            "id": "txn_" + hashlib.sha1(text.encode()).hexdigest()[:8],
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": text},
                {"role": "assistant", "content": dumps_transactions(r["transactions"])},
            ],
            "metadata": {
                "lang": r["language"], "complexity": r["complexity"], "generated_by": "numwords-aug-v1",
                "augmented_from": r["id"], "translation": r.get("translation", ""),
                "timestamp": datetime.now(timezone.utc).isoformat(), "validated": True,
                "validation_notes": "digits rewritten as words by numwords; labels unchanged",
            },
            "raw_transactions": r["transactions"],
        })
    with args.out.open("w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    langs: dict[str, int] = {}
    for r in out:
        langs[r["metadata"]["lang"]] = langs.get(r["metadata"]["lang"], 0) + 1
    print(f"{len(out)} augmented rows -> {args.out}  {langs}")


if __name__ == "__main__":
    main()
