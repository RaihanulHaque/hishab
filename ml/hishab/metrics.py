"""Scoring for transaction extraction.

Per-example flags (all booleans unless noted):
  valid        a JSON array was parsed at all
  strict       valid AND every item well-formed with an allowed category/type
  count_ok     same number of transactions as gold
  amounts_ok   same multiset of amounts
  exact        same multiset of (amount, category, type)   <- the headline metric
  empty_ok     only for gold == []: prediction is also []
  cat_hits / type_hits / aligned   per-field accuracy on amount-aligned pairs
  tp / n_pred / n_gold             for micro precision / recall / F1 over (amount, category, type)
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from .schema import parse_output


def _key(t: dict[str, Any]) -> tuple:
    return (round(float(t["amount"]), 2), t["category"], t["type"])


def score_example(pred_text: str, gold_txs: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = parse_output(pred_text)
    gold = [{**g, "amount": round(float(g["amount"]), 2)} for g in gold_txs]
    pred = parsed["txs"] or []
    valid = parsed["txs"] is not None

    gk, pk = Counter(_key(t) for t in gold), Counter(_key(t) for t in pred)
    tp = sum((gk & pk).values())
    amounts_ok = Counter(t["amount"] for t in gold) == Counter(t["amount"] for t in pred)

    # Align by amount (greedy, in order) to measure category / type accuracy independently.
    cat_hits = type_hits = aligned = 0
    pool = list(pred)
    for g in gold:
        for i, p in enumerate(pool):
            if p["amount"] == g["amount"]:
                aligned += 1
                cat_hits += p["category"] == g["category"]
                type_hits += p["type"] == g["type"]
                pool.pop(i)
                break

    return {
        "valid": valid,
        "strict": parsed["strict"],
        "count_ok": valid and len(pred) == len(gold),
        "amounts_ok": valid and amounts_ok,
        "exact": valid and gk == pk,
        "is_empty_gold": len(gold) == 0,
        "empty_ok": len(gold) == 0 and valid and len(pred) == 0,
        "cat_hits": cat_hits,
        "type_hits": type_hits,
        "aligned": aligned,
        "tp": tp,
        "n_pred": len(pred),
        "n_gold": len(gold),
        "error": parsed["error"],
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """rows: each has 'score' (from score_example) and 'language'."""
    def summarise(sub: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(sub)
        if n == 0:
            return {"n": 0}
        s = [r["score"] for r in sub]
        tp = sum(x["tp"] for x in s)
        n_pred = sum(x["n_pred"] for x in s)
        n_gold = sum(x["n_gold"] for x in s)
        aligned = sum(x["aligned"] for x in s)
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_gold if n_gold else 0.0
        empties = [x for x in s if x["is_empty_gold"]]
        return {
            "n": n,
            "valid_json": sum(x["valid"] for x in s) / n,
            "strict_schema": sum(x["strict"] for x in s) / n,
            "exact_match": sum(x["exact"] for x in s) / n,
            "count_match": sum(x["count_ok"] for x in s) / n,
            "amounts_match": sum(x["amounts_ok"] for x in s) / n,
            "category_acc": sum(x["cat_hits"] for x in s) / aligned if aligned else 0.0,
            "type_acc": sum(x["type_hits"] for x in s) / aligned if aligned else 0.0,
            "tx_precision": prec,
            "tx_recall": rec,
            "tx_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "empty_n": len(empties),
            "empty_acc": (sum(x["empty_ok"] for x in empties) / len(empties)) if empties else None,
            "errors": dict(Counter(x["error"] for x in s if x["error"])),
        }

    out = {"all": summarise(rows), "by_language": {}}
    for lang in sorted({r["language"] for r in rows}):
        out["by_language"][lang] = summarise([r for r in rows if r["language"] == lang])
    return out


def evaluate(examples: list[dict[str, Any]], predictions: list[str]) -> dict[str, Any]:
    """examples: split rows (need 'language' and 'transactions'); predictions: raw model text."""
    assert len(examples) == len(predictions)
    rows = [
        {"language": ex["language"], "score": score_example(p, ex["transactions"])}
        for ex, p in zip(examples, predictions)
    ]
    return aggregate(rows)
