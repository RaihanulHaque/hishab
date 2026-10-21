"""Print every EXPERIMENTS.md row for one model version, in each table's column order.

  conda run -n ML python ml/results/experiment_rows.py --label v12 \
      --run qwen3.5-0.8b__pipe_signed__full__v12 --gguf-run qwen3.5-0.8b__pipe_signed__full__v12__trim38k \
      --train-split train_v12

--run is the trained run (bf16 eval_hf.json, train_log.json, preds_*.jsonl); --gguf-run is the run whose
eval_gguf/ holds the phone-file scores (the same run, or its __trim38k / __L20 child). Files are pulled
from the Modal volume `hishab-runs` (MODAL_PROFILE=cortex-technologies). Missing pieces print as blank
cells, never as guesses. Paste each row next to the previous version's row in the table named above it;
EXPERIMENTS.md is only complete when every table that has a row for the previous version has one for
this version too.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
from hishab.metrics import score_example  # noqa: E402
from hishab.schema import parse_output  # noqa: E402

LANGS = ["banglish", "bangla", "english", "mixed"]
TEST_QUANTS = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M-imat", "Q4_K_M", "Q4_0-imat", "IQ4_XS-imat", "Q3_K_M-imat", "IQ3_M-imat"]
NUM_QUANTS = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M-imat"]
MODAL = os.environ.get("MODAL_BIN", str(Path.home() / "miniconda3/envs/ML/bin/modal"))


def pull(run: str, rel: str, tmp: Path) -> Path | None:
    out = tmp / run / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MODAL_PROFILE": os.environ.get("MODAL_PROFILE", "cortex-technologies")}
    r = subprocess.run([MODAL, "volume", "get", "--force", "hishab-runs", f"/{run}/{rel}", str(out)],
                       capture_output=True, text=True, env=env)
    return out if r.returncode == 0 and out.exists() and out.is_file() else None


def pct(x) -> str:
    return "" if x is None else f"{100 * x:.1f}"


def split_row(label: str, m: dict) -> str:
    a, lang = m["all"], m["by_language"]
    cells = [pct(a["exact_match"]), pct(a["tx_f1"]), pct(a["count_match"]), pct(a["amounts_match"]),
             pct(a["category_acc"]), pct(a["type_acc"])] + [pct(lang[l]["exact_match"]) if l in lang else "" for l in LANGS]
    return f"| {label} | " + " | ".join(cells) + " |"


def wrong_row(label: str, preds: Path | None) -> str:
    if not preds:
        return f"| {label} | | | | | | |"
    k, swaps = Counter(), Counter()
    for line in preds.read_text().splitlines():
        r = json.loads(line)
        s = score_example(r["pred"], r["gold"])
        if s["exact"]:
            continue
        k["count" if not s["count_ok"] else "amount" if not s["amounts_ok"] else "type" if s["type_hits"] < s["aligned"] else "category"] += 1
        pool = list(parse_output(r["pred"])["txs"] or [])
        for g in r["gold"]:
            for i, p in enumerate(pool):
                if round(p["amount"], 2) == round(float(g["amount"]), 2):
                    if p["category"] != g["category"]:
                        swaps[(g["category"], p["category"])] += 1
                    pool.pop(i)
                    break
    top = ", ".join(f"{a} to {b} {c}" for (a, b), c in swaps.most_common(3))
    return f"| {label} | {sum(k.values())} | {k['count']} | {k['amount']} | {k['type']} | {k['category']} | {top} |"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="row label, e.g. v12")
    ap.add_argument("--run", required=True, help="trained run (bf16 eval)")
    ap.add_argument("--gguf-run", help="run with the phone-file evals (defaults to --run)")
    ap.add_argument("--train-split", help="local split file name for the data tables, e.g. train_v12")
    args = ap.parse_args()
    grun = args.gguf_run or args.run

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ev_p = pull(args.run, "eval_hf.json", tmp)
        log_p = pull(args.run, "train_log.json", tmp)
        ev = json.loads(ev_p.read_text()) if ev_p else {}
        sp = ev.get("splits", {})
        L = args.label

        print("## Training setup")
        if log_p:
            log = json.loads(log_p.read_text())
            vals = [h["eval_loss"] for h in log if "eval_loss" in h]
            losses = [h["loss"] for h in log if "loss" in h]
            best = min(range(len(vals)), key=lambda i: vals[i]) + 1 if vals else ""
            vl = " / ".join(f"{v:.4f}".lstrip("0") for v in vals)
            t = ev.get("train_time_s")
            print(f"| {L} | {len(vals)} | {best} | {min(vals):.4f} | {vl} | {losses[-1]:.3f} | {round(t / 60) if t else ''} min |")
        print()

        if args.train_split:
            rows = [json.loads(l) for l in (ROOT / "ml/datasets/splits" / f"{args.train_split}.jsonl").read_text().splitlines() if l.strip()]
            lang = Counter(r["language"] for r in rows)
            per = Counter(min(len(r["transactions"]), 4) for r in rows)
            print(f"## Training data: {len(rows):,} rows")
            print(f"| {L} | {lang['banglish']:,} | {lang['bangla']:,} | {lang['mixed']:,} | {lang['english']:,} |   <- train rows by language")
            print(f"| {L} | {per[0]:,} | {per[1]:,} | {per[2]:,} | {per[3]:,} | {per[4]:,} |   <- transactions per message")
            print("sources:", dict(Counter(r["source"] for r in rows).most_common()))
            print()

        print("## Test set, bf16")
        for s, lab in (("test", L), ("test_clean", f"{L} on `test_clean` (476)")):
            if s in sp:
                print(split_row(lab, sp[s]))
        print("wrong messages:")
        print(wrong_row(L, pull(args.run, "preds_test.jsonl", tmp)))
        print(wrong_row(f"{L} on `test_clean` (of 476)", pull(args.run, "preds_test_clean.jsonl", tmp)))
        print()

        print("## Numbers set, bf16")
        for s, lab in (("numbers", L), ("numbers_clean", f"{L} on `numbers_clean` (317)")):
            if s in sp:
                print(split_row(lab, sp[s]))
        print("wrong messages:")
        print(wrong_row(L, pull(args.run, "preds_numbers.jsonl", tmp)))
        print(wrong_row(f"{L} on `numbers_clean` (of 317)", pull(args.run, "preds_numbers_clean.jsonl", tmp)))
        print()

        extra = [s for s in sp if s not in ("validation", "test", "test_clean", "numbers", "numbers_clean")]
        for s in extra:
            print(f"## {s}, bf16")
            print(split_row(L, sp[s]))
            print(wrong_row(L, pull(args.run, f"preds_{s}.jsonl", tmp)))
            print()

        def gscore(quant: str, split: str) -> dict | None:
            p = pull(grun, f"eval_gguf/{quant}.{split}.json", tmp)
            return json.loads(p.read_text())["metrics"] if p else None

        print(f"## Phone files ({grun})")
        for split, quants, hdr in (("test", TEST_QUANTS, "test"), ("test_clean", TEST_QUANTS, "test_clean")):
            cells = [pct(m["all"]["exact_match"]) if (m := gscore(q, split)) else "" for q in quants]
            print(f"| {L}{'' if split == 'test' else ' on `test_clean`'} | " + " | ".join(cells) + f" |   <- {hdr}: {', '.join(quants)}")
        for split in ("numbers", "numbers_clean"):
            bf = pct(sp[split]["all"]["exact_match"]) + " (HF)" if split in sp else ""
            cells = [pct(m["all"]["exact_match"]) if (m := gscore(q, split)) else "" for q in NUM_QUANTS]
            print(f"| {L}{'' if split == 'numbers' else ' on `numbers_clean`'} | {bf} | " + " | ".join(cells) + f" |   <- {split}: BF16, {', '.join(NUM_QUANTS)}")
        for s in extra:
            cells = [f"{q} {pct(m['all']['exact_match'])}" for q in TEST_QUANTS if (m := gscore(q, s))]
            print(f"{s}: " + ", ".join(cells))
        print()

        print("## Phone path vs full precision, numbers set by language")
        if "numbers" in sp:
            lang = sp["numbers"]["by_language"]
            print(f"| {L} | bf16 on HF | " + " | ".join(pct(lang[l]["exact_match"]) if l in lang else "" for l in LANGS) + " |")
        for q in ("Q6_K", "Q4_K_M-imat"):
            if m := gscore(q, "numbers"):
                lang = m["by_language"]
                print(f"| {L} | {q} GGUF | " + " | ".join(pct(lang[l]["exact_match"]) if l in lang else "" for l in LANGS) + " |")


if __name__ == "__main__":
    main()
