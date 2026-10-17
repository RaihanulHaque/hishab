"""Build a guideline-consistent training split from the two-labeller relabel of the Gemini rows.

  python ml/data/apply_relabel.py --labels ml/results/relabel/gemini_train.jsonl ml/results/relabel/sonnet_train.jsonl \
      --out ml/datasets/splits/train_v10.jsonl

For every train row whose `source` is one of --sources (the Gemini-written rows: base, hard):
  * both labellers agree            -> use their label (keeps the old descriptions when the tuple set is unchanged)
  * labellers disagree              -> drop the row (its convention is ambiguous; keeping either side teaches noise)
  * a labeller has no answer yet    -> drop the row and count it
Every other source (Codex templates, programmatic lexicon, hand-written, number-word augmentation,
negatives) already follows docs/LABELLING_GUIDELINE.md and is kept as is.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "ml/datasets/splits"


def key(txs) -> tuple:
    return tuple(sorted((round(float(t["amount"]), 2), t["category"], t["type"]) for t in txs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, nargs="+", required=True)
    ap.add_argument("--sources", nargs="+", default=["base", "hard"])
    ap.add_argument("--train", type=Path, default=SPLITS / "train.jsonl")
    ap.add_argument("--out", type=Path, default=SPLITS / "train_v10.jsonl")
    args = ap.parse_args()

    labels: dict[str, dict[str, list]] = {}
    for p in args.labels:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                labels.setdefault(r["labeller"], {})[r["id"]] = r["labels"]
    names = sorted(labels)
    print("labellers:", ", ".join(f"{n} ({len(labels[n])})" for n in names))

    stat = Counter()
    out = []
    for line in args.train.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("source") not in args.sources:
            out.append(r)
            stat["kept_other_source"] += 1
            continue
        got = [labels[n].get(r["id"]) for n in names]
        if any(g is None for g in got):
            stat["dropped_missing"] += 1
            continue
        keys = {key(g) for g in got}
        if len(keys) != 1:
            stat["dropped_disagree"] += 1
            continue
        if keys.pop() == key(r["transactions"]):
            stat["kept_unchanged"] += 1
            out.append({**r, "label_source": "gold"})
        else:
            stat["relabelled"] += 1
            out.append({**r, "transactions": got[0], "label_source": names[0]})
    with args.out.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(dict(stat), "->", len(out), "rows in", args.out)


if __name__ == "__main__":
    main()
