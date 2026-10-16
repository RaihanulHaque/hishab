"""Ask Codex for natural messages with amount placeholders, never with numbers.

The model writes the words; the amounts are filled in later by ml/hishab/numwords.py
(fill_templates.py), so every label is right by construction and the LLM is never asked to
do arithmetic. Each template:

  {"text": "ajke lunch korlam {A1} takar, arpor wifi bill {A2} dite hoyechilo",
   "translation": "Had lunch for {A1} taka today, then had to pay the wifi bill of {A2}",
   "lang": "banglish", "persona": "student",
   "slots": [{"id": "A1", "description": "lunch", "category": "food", "type": "expense", "size": "small"},
             {"id": "A2", "description": "wifi bill", "category": "utilities", "type": "expense", "size": "medium"}]}

Usage:
  python ml/data/generate_llm_templates.py --lang banglish --batches 20 --per-batch 40 \
      --out ml/datasets/llm_templates_v1.jsonl
Appends; safe to re-run. Needs `codex` on PATH (codex exec, read-only sandbox, ephemeral).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hishab.schema import CATEGORIES  # noqa: E402

SIZES = ("tiny", "small", "medium", "large", "huge")

PERSONAS = [
    "university student in Dhaka living in a mess", "garment worker in Gazipur", "housewife in Chattogram managing the family budget",
    "rickshaw puller", "small shopkeeper (mudi dokan) in a mofussil town", "private-job holder in Banani with a car",
    "school teacher in Rajshahi", "freelancer who earns in dollars", "rural farmer who sells at the haat",
    "bKash agent", "new parent", "retired government officer", "Uber/Pathao driver", "medical student",
    "someone who just got their first salary", "bachelor who eats out every day", "expat sending money home",
    "tuition teacher with four batches", "small online clothing business owner", "hostel warden",
]

TOPICS = [
    "daily food and tea", "rent and bills (electricity, gas, wifi, water)", "mobile recharge and data packs",
    "transport (rickshaw, CNG, bus, metro, Pathao)", "salary, tuition income, freelance payment", "gifts, salami, money from family",
    "loans given and taken, repayments", "shopping (clothes, phone, shoes)", "doctor, medicine, tests", "cinema, concert, outing",
    "grocery (bazar, rice, oil, fish, chicken)", "education fees, books, coaching", "a mix of several things in one day",
    "money sent or received via bKash/Nagad", "Eid or wedding expenses", "a bad day where everything cost extra",
]

STYLE_NOTES = {
    "banglish": (
        "Banglish: Bangla written in Latin letters the way people type on a phone. Vary spelling (korlam/korsi/korci, "
        "taka/tk/tka), use fillers (ajke, arpor, abar, r, o, tarpor, ar), run-on sentences, no capital letters half the time, "
        "occasional typos and missing spaces. Some messages may contain one or two Bangla-script words."
    ),
    "bangla": (
        "Bangla script, informal spoken register (চলিত), like a WhatsApp message to a friend or a note to self. Some messages "
        "may contain an English word or two in Latin letters (wifi, salary, Pathao)."
    ),
    "english": (
        "English as Bangladeshis write it: short, informal, local words kept (bazar, rickshaw, salami, tuition, bKash). "
        "Occasional small grammar slips are fine."
    ),
    "mixed": (
        "Code-switching: Bangla script and English/Banglish in the same message, switching mid-sentence the way people "
        "actually type. Roughly half the words in Bangla script."
    ),
}


def prompt_for(lang: str, persona: str, topic: str, n: int, seed: int) -> str:
    cats = ", ".join(CATEGORIES)
    return f"""You generate training data for a Bangladeshi personal-finance app. Write {n} DIFFERENT messages that a person
might type into the app to record money they spent or received. The person: {persona}. Theme for this batch: {topic}
(but vary it; not every message must fit the theme). Batch seed: {seed}.

LANGUAGE: {STYLE_NOTES[lang]}

RULES (strict):
1. NEVER write a number for an amount. Put a placeholder like {{A1}}, {{A2}}, {{A3}} where the amount goes. The placeholder
   replaces the whole amount; words like taka/tk/টাকা may follow it or not, as people write. Do not write currency symbols.
2. Each message has 1 to 4 transactions, most have 1 or 2. Every transaction has exactly one placeholder.
3. Non-amount numbers are fine and must stay literal (dates, "2 kg chal", "3 jon", phone numbers, "5 ta bike").
4. Do not state a total of the transactions; if you mention a total, it is a separate placeholder and you must NOT list it as a slot.
   Simpler: avoid totals.
5. For each placeholder give a slot: id, a short "description" (what the entry should be called in the ledger, 1-4 words, in
   the message's own language), "category" from exactly this list: {cats}, "type" income or expense, and "size" from
   tiny (10-100), small (100-1000), medium (1000-10000), large (10000-100000), huge (100000-1000000) in taka.
   Use sizes realistically (tea is tiny, rent is medium/large, salary large, a car huge).
6. Also give an English "translation" of the message with the same placeholders in the same positions.
7. Keep the message 4 to 40 words. Make them sound like real people, with real item names (not "item", "something").
8. Loan direction matters: money I borrowed or someone repaid me is income; money I lent or repaid is expense.
9. Output ONLY a JSON array of {n} objects with keys: text, translation, slots. slots is a list of
   {{"id","description","category","type","size"}}. No prose, no code fence.
"""


_ARR = re.compile(r"\[.*\]", re.S)


def run_codex(prompt: str, model: str | None, timeout: int) -> list[dict]:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "out.txt"
        cmd = ["codex", "exec", "--ephemeral", "--skip-git-repo-check", "-s", "read-only", "-o", str(out)]
        if model:
            cmd += ["-m", model]
        cmd.append(prompt)
        subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=timeout)
        text = out.read_text() if out.exists() else ""
    m = _ARR.search(text)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return arr if isinstance(arr, list) else []


_PH = re.compile(r"\{(A\d+)\}")


def validate(t: dict, lang: str) -> dict | None:
    text = str(t.get("text", "")).strip()
    tr = str(t.get("translation", "")).strip()
    slots = t.get("slots")
    if not text or not tr or not isinstance(slots, list) or not slots:
        return None
    ids = _PH.findall(text)
    if not ids or len(set(ids)) != len(ids) or sorted(ids) != sorted(_PH.findall(tr)):
        return None
    if re.search(r"[0-9০-৯]+\s*(taka|tk|টাকা|৳)", text, re.I):  # a literal amount slipped through
        return None
    clean = []
    for s in slots:
        if not isinstance(s, dict):
            return None
        sid, cat, typ, size = s.get("id"), str(s.get("category", "")).lower(), str(s.get("type", "")).lower(), str(s.get("size", "")).lower()
        desc = str(s.get("description", "")).strip()
        if sid not in ids or cat not in CATEGORIES or typ not in ("income", "expense") or size not in SIZES or not desc:
            return None
        clean.append({"id": sid, "description": desc, "category": cat, "type": typ, "size": size})
    if sorted(s["id"] for s in clean) != sorted(ids):
        return None
    return {"text": text, "translation": tr, "lang": lang, "slots": clean}


def norm(text: str) -> str:
    return re.sub(r"[^a-zঀ-৿]+", "", text.lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=list(STYLE_NOTES))
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--per-batch", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen: set[str] = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                seen.add(norm(json.loads(line)["text"]))
    jobs = [(rng.choice(PERSONAS), rng.choice(TOPICS), rng.randrange(10**6)) for _ in range(args.batches)]

    def one(job):
        persona, topic, seed = job
        raw = run_codex(prompt_for(args.lang, persona, topic, args.per_batch, seed), args.model, args.timeout)
        rows = []
        for t in raw:
            v = validate(t, args.lang)
            if v:
                v["persona"] = persona
                rows.append(v)
        return len(raw), rows

    kept = 0
    with args.out.open("a") as f, ThreadPoolExecutor(args.workers) as ex:
        for n_raw, rows in ex.map(one, jobs):
            new = 0
            for r in rows:
                k = norm(r["text"])
                if k in seen:
                    continue
                seen.add(k)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                new += 1
            kept += new
            print(f"batch: {n_raw} returned, {len(rows)} valid, {new} new", flush=True)
    print(f"kept {kept} templates -> {args.out}")


if __name__ == "__main__":
    main()
