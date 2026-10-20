"""Small-amount (1 to 9 taka) rows: held-out eval split + extra training rows for v12.

v11 read "5 taka rickshaw bhara" as 500: no training row had an amount under 10, and a lone "5"
almost always meant "5 hajar" or "5 sho". The templates come from generate_llm_templates.py
--focus micro (every message has at least one 1-9 taka slot, labelled by the guideline); the
amounts are filled by numwords, so the labels are exact by construction.

Templates are split before filling, so no message pattern is in both train and eval:

  python ml/data/build_micro.py --templates ml/datasets/micro_gen/templates_*.jsonl

writes ml/datasets/splits/small_amounts.jsonl (eval) and ml/datasets/splits/train_v12.jsonl
(train_v10 + the small-amount training rows).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_templates import fill  # noqa: E402

SPLITS = Path(__file__).resolve().parents[1] / "datasets" / "splits"


def to_split_row(r: dict, source: str) -> dict:
    txs = r["raw_transactions"]
    n = len(txs)
    return {
        "id": r["id"],
        "language": r["metadata"]["lang"],
        "complexity": "simple" if n == 1 else "medium" if n == 2 else "complex",
        "source": source,
        "instruction": r["messages"][1]["content"],
        "translation": r["metadata"]["translation"],
        "transactions": txs,
    }


def fill_rows(templates: list[dict], per: int, rng: random.Random, noise_rate: float, seen: set[str], source: str) -> list[dict]:
    rows = []
    for t in templates:
        got = 0
        for _ in range(per * 6):
            if got == per:
                break
            r = fill(t, rng, digit_free=False, totals_rate=0.0, noise_rate=noise_rate)
            if not r or r["messages"][1]["content"] in seen:
                continue
            if not any(tx["amount"] < 10 for tx in r["raw_transactions"]):
                continue
            seen.add(r["messages"][1]["content"])
            rows.append(to_split_row(r, source))
            got += 1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", type=Path, nargs="+", required=True)
    ap.add_argument("--eval-share", type=float, default=0.2)
    ap.add_argument("--train-per-template", type=int, default=3)
    ap.add_argument("--noise-rate", type=float, default=0.15)
    ap.add_argument("--base-train", type=Path, default=SPLITS / "train_v10.jsonl")
    ap.add_argument("--seed", type=int, default=12)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    templates, keys = [], set()
    for p in args.templates:
        for line in p.read_text().splitlines():
            if line.strip():
                t = json.loads(line)
                k = t["text"].lower()
                if k not in keys:
                    keys.add(k)
                    templates.append(t)
    rng.shuffle(templates)
    n_eval = round(len(templates) * args.eval_share)
    eval_t, train_t = templates[:n_eval], templates[n_eval:]

    base = [json.loads(l) for l in args.base_train.read_text().splitlines() if l.strip()]
    seen = {r["instruction"] for r in base}
    for name in ("validation", "test", "test_clean", "numbers", "numbers_clean"):
        p = SPLITS / f"{name}.jsonl"
        if p.exists():
            seen |= {json.loads(l)["instruction"] for l in p.read_text().splitlines() if l.strip()}

    eval_rows = fill_rows(eval_t, 1, rng, 0.0, seen, "micro-sonnet-eval")
    train_rows = fill_rows(train_t, args.train_per_template, rng, args.noise_rate, seen, "micro-sonnet")

    (SPLITS / "small_amounts.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in eval_rows))
    merged = base + train_rows
    random.Random(args.seed + 1).shuffle(merged)
    (SPLITS / "train_v12.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged))

    def langs(rows):
        d: dict[str, int] = {}
        for r in rows:
            d[r["language"]] = d.get(r["language"], 0) + 1
        return d

    print(f"templates: {len(templates)} ({len(train_t)} train, {len(eval_t)} eval)")
    print(f"small_amounts.jsonl: {len(eval_rows)} rows {langs(eval_rows)}")
    print(f"train_v12.jsonl: {len(base)} base + {len(train_rows)} small-amount = {len(merged)} rows {langs(train_rows)}")


if __name__ == "__main__":
    main()
