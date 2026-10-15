"""Build a clean, de-duplicated, stratified train / validation / test split.

Sources
  ml/datasets/synthetic_dataset.jsonl   local rows with metadata (lang, complexity, generator)
  HF ryuma007/hishab-synthetic-v2        same inputs + an English TRANSLATION line (train parquet)

Why this exists: the published v2 "validation" split is a subset of its train split, so every
number measured on it is a training-set number. This script makes a real held-out test set.

Output: ml/datasets/splits/{train,validation,test}.jsonl with rows
  {id, language, complexity, source, instruction, translation, transactions}
plus stats.json. Run locally:

  conda run -n ML python ml/data/build_splits.py [--test-frac 0.10 --val-frac 0.05 --seed 42]

Needs pyarrow to read the HF parquet (pass --translations <parquet> or let it download).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "ml/datasets/synthetic_dataset.jsonl"
OUT = ROOT / "ml/datasets/splits"
HF_PARQUET = (
    "https://huggingface.co/datasets/ryuma007/hishab-synthetic-v2/resolve/main/"
    "data/train-00000-of-00001.parquet"
)

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def norm_key(text: str) -> str:
    """Dedup key: NFKC, lowercase, punctuation stripped, whitespace collapsed."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def load_translations(parquet: Path) -> dict[str, str]:
    import pyarrow.parquet as pq  # local dev dependency only

    out: dict[str, str] = {}
    for r in pq.read_table(parquet).to_pylist():
        resp = r["response"]
        if "TRANSLATION:" in resp:
            head = re.split(r"JSON:\s*", resp, maxsplit=1)[0]
            out[r["instruction"]] = head.replace("TRANSLATION:", "", 1).strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--translations", type=Path, default=None, help="local copy of the v2 train parquet")
    args = ap.parse_args()

    parquet = args.translations or (OUT / "_v2_train.parquet")
    if not parquet.exists():
        print(f"downloading {HF_PARQUET}")
        OUT.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HF_PARQUET, parquet)
    translations = load_translations(parquet)

    rows, seen, dropped = [], set(), Counter()
    for line in LOCAL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        user = next((m["content"] for m in r["messages"] if m["role"] == "user"), None)
        txs = r.get("raw_transactions")
        if not user or txs is None:
            dropped["missing_fields"] += 1
            continue
        key = norm_key(user)
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        meta = r.get("metadata", {})
        source = "hard" if "hard" in str(meta.get("generated_by", "")) else "base"
        rows.append(
            {
                "id": r["id"],
                "language": meta.get("lang", "unknown"),
                "complexity": meta.get("complexity", "unknown"),
                "source": source,
                "instruction": user,
                "translation": translations.get(user, ""),
                "transactions": [
                    {
                        "description": t["description"],
                        "amount": float(t["amount"]),
                        "category": t["category"],
                        "type": t["type"],
                    }
                    for t in txs
                ],
            }
        )

    missing_tr = sum(1 for r in rows if not r["translation"])
    rng = random.Random(args.seed)
    strata: dict[tuple, list] = defaultdict(list)
    for r in rows:
        strata[(r["language"], r["source"], len(r["transactions"]) == 0)].append(r)

    splits = {"train": [], "validation": [], "test": []}
    for key in sorted(strata):
        bucket = strata[key]
        rng.shuffle(bucket)
        n = len(bucket)
        n_test = round(n * args.test_frac)
        n_val = round(n * args.val_frac)
        splits["test"] += bucket[:n_test]
        splits["validation"] += bucket[n_test : n_test + n_val]
        splits["train"] += bucket[n_test + n_val :]
    for s in splits.values():
        rng.shuffle(s)

    OUT.mkdir(parents=True, exist_ok=True)
    stats = {"seed": args.seed, "dropped": dict(dropped), "missing_translation": missing_tr, "splits": {}}
    for name, s in splits.items():
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in s:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats["splits"][name] = {
            "n": len(s),
            "language": dict(Counter(r["language"] for r in s)),
            "source": dict(Counter(r["source"] for r in s)),
            "n_tx": dict(sorted(Counter(len(r["transactions"]) for r in s).items())),
        }
    # Leakage guard: no normalised instruction may appear in two splits.
    keys = {n: {norm_key(r["instruction"]) for r in s} for n, s in splits.items()}
    assert not (keys["train"] & keys["test"]) and not (keys["train"] & keys["validation"]), "split leakage"
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
