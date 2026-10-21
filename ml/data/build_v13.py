"""Build the guideline-v2 data for v13 from the relabel runs.

  python ml/data/build_v13.py

Inputs (ml/results/relabel_v2): opus_train.jsonl (Opus, guideline v2, every non-negative train row),
sonnet_changed.jsonl (Sonnet, v2, the rows where Opus changed something), opus-g2_eval.jsonl and
sonnet-g2_eval.jsonl (both models on validation / test / numbers / small_amounts).

Outputs (ml/datasets/splits):
  train_v13.jsonl       train_v13_v1.jsonl (the template relabel under guideline v1: Sonnet + Codex/Opus,
                        amount-locked) plus every v2 change on which Opus and Sonnet agree, with amounts
                        equal to the current label's (category and type move, descriptions stay)
  validation_v2.jsonl   agreed v2 label where Opus and Sonnet agree on the same amounts, else the old label
  test_clean_v2.jsonl   test rows where Opus and Sonnet agree (their label is the gold)
  numbers_clean_v2.jsonl  the same for numbers; amounts must also equal the template amounts
  small_amounts_v2.jsonl  the same for small_amounts (amounts exact by construction)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "ml/datasets/splits"
R = ROOT / "ml/results/relabel_v2"


def load(p: Path) -> dict[str, dict]:
    return {json.loads(l)["id"]: json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def key(txs) -> tuple:
    return tuple(sorted((round(float(t["amount"]), 2), t["category"], t["type"]) for t in txs))


def amounts(txs) -> list:
    return sorted(round(float(t["amount"]), 2) for t in txs)


def move_labels(txs: list[dict], new: list[dict]) -> list[dict]:
    """Category/type from `new` onto `txs`, matched by amount; descriptions and amounts stay."""
    pool = [dict(t) for t in new]
    out = []
    for t in txs:
        j = next(i for i, g in enumerate(pool) if round(float(g["amount"]), 2) == round(float(t["amount"]), 2))
        g = pool.pop(j)
        out.append({**t, "category": g["category"], "type": g["type"]})
    return out


def write(name: str, rows: list[dict]) -> None:
    (S / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main() -> None:
    base = S / "train_v13_v1.jsonl"
    if not base.exists():  # first run: keep the v1 template relabel as the base
        (S / "train_v13.jsonl").rename(base)
    train = [json.loads(l) for l in base.read_text(encoding="utf-8").splitlines() if l.strip()]
    opus = {i: r["labels"] for i, r in load(R / "opus_train.jsonl").items()}
    sonnet = {i: r["labels"] for i, r in load(R / "sonnet_changed.jsonl").items()}

    st = Counter()
    out = []
    for r in train:
        o, s = opus.get(r["id"]), sonnet.get(r["id"])
        if o is None or amounts(o) != amounts(r["transactions"]) or key(o) == key(r["transactions"]):
            st["unchanged"] += 1
            out.append(r)
        elif s is not None and key(s) == key(o):
            st["v2 change, Opus and Sonnet agree"] += 1
            out.append({**r, "transactions": move_labels(r["transactions"], o), "label_source": "opus+sonnet-g2"})
        else:
            st["v2 change, Sonnet disagrees: kept old label"] += 1
            out.append(r)
    write("train_v13", out)
    print(f"train_v13: {len(out)} rows {dict(st)}")

    eo = load(R / "opus-g2_eval.jsonl")
    es = load(R / "sonnet-g2_eval.jsonl")
    for split, name, lock in (("validation", "validation_v2", False), ("test", "test_clean_v2", False),
                              ("numbers", "numbers_clean_v2", True), ("small_amounts", "small_amounts_v2", True)):
        rows = [json.loads(l) for l in (S / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        res, c = [], Counter()
        for r in rows:
            o, s = eo.get(r["id"]), es.get(r["id"])
            agree = o is not None and s is not None and key(o["labels"]) == key(s["labels"])
            if agree and lock and amounts(o["labels"]) != amounts(r["transactions"]):
                agree = False
            if agree:
                lab = o["labels"]
                txs = move_labels(r["transactions"], lab) if amounts(lab) == amounts(r["transactions"]) else lab
                c["changed" if key(txs) != key(r["transactions"]) else "same as old gold"] += 1
                res.append({**r, "transactions": txs, "label_source": "opus+sonnet-g2"})
            elif split == "validation":
                c["kept old (no agreement)"] += 1
                res.append(r)
            else:
                c["dropped (labellers disagree)"] += 1
        write(name, res)
        print(f"{name}: {len(res)} of {len(rows)} {dict(c)}")


if __name__ == "__main__":
    main()
