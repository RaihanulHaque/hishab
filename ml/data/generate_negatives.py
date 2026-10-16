"""Messages with no transaction, labelled []: questions, plans, chit-chat, and numbers that are
not money. The model must learn to say nothing rather than invent an entry.

  python ml/data/generate_negatives.py --n 500 --out ml/datasets/synthetic_negatives_v1.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

SYSTEM = ("You are a financial assistant who understands English, Bengali and Banglish(Bengali written in english). "
          "Extract transactions from text into a JSON list.")

# (text, lang). {N} is a non-money number, {M} a money-looking number that is NOT a transaction
# (a question or a plan), filled below.
TEMPLATES = [
    ("ei mashe koto khoroch holo?", "banglish"), ("amar balance koto?", "banglish"), ("salary kobe dibe?", "banglish"),
    ("kal bazar korte jabo", "banglish"), ("rent ta next week e dibo", "banglish"), ("ajke kono khoroch kori nai", "banglish"),
    ("bkash e taka ashe nai ekhono", "banglish"), ("abbu ke phone dite hobe", "banglish"), ("exam {N} tay shuru", "banglish"),
    ("class e {N} jon chilo ajke", "banglish"), ("{N} kg chal lagbe kal", "banglish"), ("bus number {N} e uthsi", "banglish"),
    ("amar number {P}", "banglish"), ("ajke onek gorom", "banglish"), ("kal ki plan?", "banglish"), ("hishab app ta valo lagse", "banglish"),
    ("ami ki {M} taka save korte parbo ei mashe?", "banglish"), ("{M} taka te phone pawa jabe?", "banglish"),
    ("gotomashe koto gelo dekhao", "banglish"), ("last month er report dao", "banglish"), ("tuition fee koto jani na", "banglish"),
    ("rickshaw vara barse ajkal", "banglish"), ("bazar e agun, dam onek", "banglish"), ("loan er kisti kobe?", "banglish"),
    ("এই মাসে কত খরচ হলো?", "bangla"), ("বেতন কবে দিবে?", "bangla"), ("কাল বাজার করতে যাবো", "bangla"), ("আজ কোনো খরচ করিনি", "bangla"),
    ("ক্লাসে {N} জন ছিল", "bangla"), ("পরীক্ষা {N} টায়", "bangla"), ("{N} কেজি চাল লাগবে", "bangla"), ("আমার ব্যালেন্স কত?", "bangla"),
    ("{M} টাকায় কি ফোন পাওয়া যাবে?", "bangla"), ("গত মাসের হিসাব দেখাও", "bangla"), ("ভাড়া বেড়ে গেছে", "bangla"), ("আজ খুব গরম", "bangla"),
    ("How much did I spend this month?", "english"), ("When is salary coming?", "english"), ("Going to the bazar tomorrow", "english"),
    ("No expenses today", "english"), ("{N} people came over", "english"), ("Need {N} kg of rice", "english"), ("Can I save {M} this month?", "english"),
    ("Show last month's report", "english"), ("Rent is due next week", "english"), ("Bus {N} was late again", "english"),
    ("ajke কোনো খরচ হয়নি", "mixed"), ("salary টা কবে আসবে?", "mixed"), ("কাল bazar করবো", "mixed"), ("{N} জন friend আসছিল", "mixed"),
    ("{M} taka তে কি laptop হবে?", "mixed"), ("গত মাসের hishab দেখাও", "mixed"), ("exam {N} টায় শুরু", "mixed"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows, seen = [], set()
    tries = 0
    while len(rows) < args.n and tries < args.n * 20:
        tries += 1
        text, lang = rng.choice(TEMPLATES)
        text = (text.replace("{N}", str(rng.randint(2, 12)))
                    .replace("{M}", rng.choice([str(rng.randint(5, 80) * 1000), f"{rng.randint(10, 60)}k", f"{rng.randint(1, 5)} lakh"]))
                    .replace("{P}", f"01{rng.randint(300000000, 999999999)}"))
        if rng.random() < 0.3:
            text = text[:1].upper() + text[1:]
        if rng.random() < 0.2:
            text = text.rstrip("?.") + rng.choice(["", "...", " 🙂", "!!"])
        if text in seen:
            continue
        seen.add(text)
        rows.append({
            "id": "txn_" + hashlib.sha1(text.encode()).hexdigest()[:8],
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text},
                         {"role": "assistant", "content": "[]"}],
            "metadata": {"lang": lang, "complexity": "simple", "generated_by": "negatives-v1",
                         "timestamp": datetime.now(timezone.utc).isoformat(), "validated": True,
                         "validation_notes": "no transaction by construction"},
            "raw_transactions": [],
        })
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} negative rows -> {args.out}")


if __name__ == "__main__":
    main()
