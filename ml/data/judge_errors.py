"""Codex as judge: for rows the model got wrong, decide per docs/LABELLING_GUIDELINE.md whether
the gold label, the model, or neither is right, and give the correct transactions.

  python ml/data/judge_errors.py --preds preds_test_v6.jsonl preds_numbers_v6.jsonl \
      --out ml/results/judge_v6.jsonl

Only disagreements are sent (cheap). Output rows: {"id", "split", "instruction", "gold", "pred",
"verdict": gold|pred|neither, "correct": [...], "note"}.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hishab.schema import CATEGORIES, parse_output  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GUIDE = (ROOT / "docs/LABELLING_GUIDELINE.md").read_text()


def key(t):
    return (round(float(t["amount"]), 2), t["category"], t["type"])


def disagreements(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        pred = parse_output(r["pred"])["txs"] or []
        if sorted(key(t) for t in r["gold"]) != sorted(key(t) for t in pred):
            out.append({"id": r["id"], "split": path.stem.replace("preds_", ""), "instruction": r["instruction"],
                        "gold": r["gold"], "pred": pred})
    return out


def prompt_for(rows: list[dict]) -> str:
    items = [{"id": r["id"], "message": r["instruction"], "gold": r["gold"], "model": r["pred"]} for r in rows]
    return f"""You are the final judge for a Bangladeshi personal-finance dataset. Messages are in English, Bangla
or Banglish (Bangla in Latin letters). Two labellings of each message disagree: "gold" (from the dataset) and
"model". Decide, strictly by the guideline below, which is right.

GUIDELINE
{GUIDE}

For each item output: "id"; "verdict": "gold" if the gold labelling is fully correct, "model" if the model's
is fully correct, "neither" otherwise; "correct": the fully correct list of transactions per the guideline
(objects with description, amount, category, type; description short, in the message's language); "note":
one short sentence naming the rule that decided it. Categories must be from: {", ".join(CATEGORIES)}.
Amounts are numbers. Keep descriptions from gold or model when they are fine.

Output ONLY a JSON array with one object per item, same order, no prose, no code fence.

ITEMS
{json.dumps(items, ensure_ascii=False, indent=0)}
"""


def run_codex(prompt: str, timeout: int = 900) -> list[dict]:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "out.txt"
        subprocess.run(["codex", "exec", "--ephemeral", "--skip-git-repo-check", "-s", "read-only", "-o", str(out), prompt],
                       cwd=d, capture_output=True, text=True, timeout=timeout)
        text = out.read_text() if out.exists() else ""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return arr if isinstance(arr, list) else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    rows = [r for p in args.preds for r in disagreements(p)]
    print(f"{len(rows)} disagreements to judge")
    by_id = {r["id"]: r for r in rows}
    judged = 0
    with args.out.open("w") as f:
        for i in range(0, len(rows), args.batch):
            batch = rows[i:i + args.batch]
            for v in run_codex(prompt_for(batch)):
                r = by_id.get(v.get("id"))
                if not r or v.get("verdict") not in ("gold", "model", "neither") or not isinstance(v.get("correct"), list):
                    continue
                ok = all(isinstance(t, dict) and t.get("category") in CATEGORIES and t.get("type") in ("income", "expense")
                         and isinstance(t.get("amount"), (int, float)) for t in v["correct"])
                if not ok:
                    continue
                f.write(json.dumps({**r, "verdict": v["verdict"], "correct": v["correct"], "note": v.get("note", "")},
                                   ensure_ascii=False) + "\n")
                judged += 1
            print(f"batch {i // args.batch + 1}: judged so far {judged}", flush=True)
    print(f"{judged}/{len(rows)} judged -> {args.out}")


if __name__ == "__main__":
    main()
