"""Merge independent relabellings into a trustworthy test set.

  python ml/data/relabel_merge.py --labels ml/results/relabel/gemini.jsonl ml/results/relabel/sonnet.jsonl \
      --out-dir ml/datasets/splits_clean

For every row of the held-out splits (test, numbers) compare the original gold and each
labeller on the (amount, category, type) multiset.

  * agree      : all labellers agree with each other -> their label is the clean gold
                 (whether or not the original gold agreed; reports how often it did)
  * disagree   : labellers disagree -> written to <out-dir>/disagreements.jsonl for a judge
  * missing    : some labeller has no answer for the row

Writes <out-dir>/{test,numbers}.jsonl with the clean rows (same schema as ml/datasets/splits,
plus "label_source"), a disagreements file, and prints an agreement report.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "ml/datasets/splits"


def key(txs) -> tuple:
    return tuple(sorted((round(float(t["amount"]), 2), t["category"], t["type"]) for t in txs))


def load_labels(paths: list[Path]) -> dict[str, dict[str, list[dict]]]:
    """labeller -> id -> labels"""
    out: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["labeller"]][r["id"]] = r["labels"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=["test", "numbers"])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "ml/datasets/splits_clean")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    names = sorted(labels)
    print("labellers:", ", ".join(f"{n} ({len(labels[n])} rows)" for n in names))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dis_f = (args.out_dir / "disagreements.jsonl").open("w", encoding="utf-8")
    report = []
    for split in args.splits:
        rows = [json.loads(l) for l in (SPLITS / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        stat = Counter()
        gold_vs = Counter()
        field = Counter()
        clean = []
        for r in rows:
            got = {n: labels[n].get(r["id"]) for n in names}
            if any(v is None for v in got.values()):
                stat["missing"] += 1
                continue
            keys = {n: key(v) for n, v in got.items()}
            gk = key(r["transactions"])
            for n in names:
                gold_vs[n] += keys[n] == gk
            if len(set(keys.values())) == 1:
                stat["agree"] += 1
                stat["agree_and_gold_agrees"] += next(iter(keys.values())) == gk
                # keep the original gold's descriptions when the tuple set matches, else the first labeller's
                src = "gold" if next(iter(keys.values())) == gk else names[0]
                txs = r["transactions"] if src == "gold" else got[names[0]]
                clean.append({**r, "transactions": txs, "label_source": src})
            else:
                stat["disagree"] += 1
                a, b = (keys[n] for n in names[:2])
                if len(a) != len(b):
                    field["count"] += 1
                elif {t[0] for t in a} != {t[0] for t in b}:
                    field["amount"] += 1
                elif {t[1] for t in a} != {t[1] for t in b}:
                    field["category"] += 1
                else:
                    field["type"] += 1
                dis_f.write(json.dumps({"id": r["id"], "split": split, "language": r["language"], "instruction": r["instruction"],
                                        "gold": r["transactions"], **{n: got[n] for n in names}}, ensure_ascii=False) + "\n")
        with (args.out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for c in clean:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        n = len(rows)
        report.append(f"{split}: {n} rows | agree {stat['agree']} ({100 * stat['agree'] / n:.1f}%) of which original gold "
                      f"also agrees {stat['agree_and_gold_agrees']} | disagree {stat['disagree']} "
                      f"(count {field['count']}, amount {field['amount']}, category {field['category']}, type {field['type']}) "
                      f"| missing {stat['missing']}")
        report.append("   each labeller vs original gold: " + ", ".join(f"{k} {100 * v / n:.1f}%" for k, v in gold_vs.items()))
    dis_f.close()
    print("\n".join(report))
    print(f"clean splits -> {args.out_dir}")


if __name__ == "__main__":
    main()
