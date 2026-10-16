"""Smoke-test a GGUF on short probes with kiln-cli, the way the phone runs it.

  python ml/scripts/probe_gguf.py --model ../kiln/models/hishab-qwen3.5-0.8b-json-v5-Q6_K.gguf \
      --probes ml/datasets/probes_numbers.jsonl

Prints one line per probe and the amount / category / exact accuracy. Each probe is one
transaction: {"instruction", "amount", "category", "type"}.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hishab.prompts import system_prompt  # noqa: E402
from hishab.schema import parse_output  # noqa: E402

TEMPLATE = "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probes", type=Path, default=Path("ml/datasets/probes_numbers.jsonl"))
    ap.add_argument("--kiln", default=str(Path(__file__).resolve().parents[3] / "kiln/build/host/kiln-cli"))
    ap.add_argument("--fmt", default="json")
    args = ap.parse_args()

    probes = [json.loads(l) for l in args.probes.read_text().splitlines() if l.strip()]
    n_amount = n_cat = n_exact = 0
    for p in probes:
        prompt = TEMPLATE.format(sys=system_prompt(args.fmt), user=unicodedata.normalize("NFC", p["instruction"]))
        out = subprocess.run([args.kiln, "-m", args.model, "--raw", "-p", prompt, "-n", "200"],
                             capture_output=True, text=True).stdout
        out = re.sub(r"\x1b\[[0-9;]*m", "", out)
        txs = parse_output(out)["txs"] or []
        got = txs[0] if len(txs) == 1 else None
        ok_a = got is not None and abs(got["amount"] - p["amount"]) < 0.5
        ok_c = got is not None and got["category"] == p["category"] and got["type"] == p["type"]
        n_amount += ok_a
        n_cat += ok_c
        n_exact += ok_a and ok_c
        shown = f"{got['amount']:g} {got['category']}/{got['type'][:3]}" if got else f"{len(txs)} tx"
        print(f"{'✓' if ok_a and ok_c else '✗'} {p['instruction']:<40} → {shown:<28} want {p['amount']:g} {p['category']}")
    n = len(probes)
    print(f"\namount {n_amount}/{n}  category+type {n_cat}/{n}  exact {n_exact}/{n}")


if __name__ == "__main__":
    main()
