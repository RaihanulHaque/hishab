"""Compare the guideline-v2 Opus labels with the current training labels and summarise the changes.

  python ml/data/relabel_v2_diff.py [--examples 3]

Current label = train_v13.jsonl where the row is there (v1 two-labeller fixes applied), else
train_v12.jsonl. A change is only a category/type change when Opus's amounts equal the current
amounts; rows where Opus reads different amounts are counted separately and never applied.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "ml/datasets/splits"
R = ROOT / "ml/results/relabel_v2"

# old (category, type) -> new (category, type) pairs that the v2 rule changes explain
def rule_of(old: tuple, new: tuple, text: str) -> str | None:
    (oc, ot), (nc, nt) = old, new
    if ot != nt:
        return None
    if oc == "gift" and nc not in ("gift",):
        return "family money by purpose"
    if nc == "salary" and nt == "expense":
        return "wages I pay"
    if nc == "gift" and oc in ("grocery", "tuition", "medical", "other", "rent", "food"):
        return "family money with no purpose"
    if oc == "rent" and nc == "transport":
        return "vehicle joma"
    return None


def load(p: Path) -> dict[str, dict]:
    return {json.loads(l)["id"]: json.loads(l) for l in p.read_text().splitlines() if l.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()
    v12, v13 = load(S / "train_v12.jsonl"), load(S / "train_v13.jsonl")
    opus = {i: r["labels"] for i, r in load(R / "opus_train.jsonl").items()}

    stat, by_rule, other_pairs, ex = Counter(), Counter(), Counter(), defaultdict(list)
    src_change = Counter()
    for i, lab in opus.items():
        cur = v13.get(i)
        if cur is None:
            stat["row dropped in v13 (labellers disagreed)"] += 1
            continue
        txs = cur["transactions"]
        if sorted(round(float(t["amount"]), 2) for t in txs) != sorted(round(float(t["amount"]), 2) for t in lab):
            stat["opus amounts differ (not applied)"] += 1
            continue
        pool = [dict(t) for t in lab]
        changed = False
        for t in txs:
            j = next(k for k, g in enumerate(pool) if round(float(g["amount"]), 2) == round(float(t["amount"]), 2))
            g = pool.pop(j)
            old, new = (t["category"], t["type"]), (g["category"], g["type"])
            if old != new:
                changed = True
                rule = rule_of(old, new, cur["instruction"])
                key = f"{old[0]}/{old[1][:3]} -> {new[0]}/{new[1][:3]}"
                if rule:
                    by_rule[rule] += 1
                else:
                    other_pairs[key] += 1
                if len(ex[rule or key]) < args.examples:
                    ex[rule or key].append((cur["instruction"][:90], t["amount"], key))
        stat["changed" if changed else "unchanged"] += 1
        if changed:
            src_change[cur["source"]] += 1

    print("rows:", dict(stat))
    print("changed rows by source:", dict(src_change.most_common()))
    print("\nentry changes explained by the v2 rules:")
    for k, n in by_rule.most_common():
        print(f"  {n:5d}  {k}")
        for t, a, key in ex[k]:
            print(f"           {key}: {a:g}  {t}")
    print(f"\nother entry changes ({sum(other_pairs.values())}), top 25:")
    for k, n in other_pairs.most_common(25):
        print(f"  {n:5d}  {k}")
        for t, a, _ in ex[k][: args.examples]:
            print(f"           {a:g}  {t}")


if __name__ == "__main__":
    main()
