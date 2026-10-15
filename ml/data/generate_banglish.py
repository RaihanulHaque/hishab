"""Programmatic Banglish / Bangla / mixed training data with exact labels.

Combines the hand-authored lexicon in ml/data/lexicon/banglish.py (items with categories and
realistic amount ranges, spelling variants, verbs, time words, connectors, number slang) into
sentences whose transactions are correct by construction. No LLM, no API key, no label noise.

Weighted toward what the model gets wrong: rent, loan, gift, medical, utilities in Banglish,
mobile_recharge / other in Bangla, and income verbs (pelam, paisi, dhuklo, ashlo).

  conda run -n ML python ml/data/generate_banglish.py --n 5000 --seed 7
  -> ml/datasets/synthetic_programmatic_v1.jsonl   (same row format as synthetic_dataset.jsonl)

Rows are de-duplicated against each other and against the existing dataset. Use them for
training only: build_splits.py keeps the held-out test set unchanged.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "data" / "lexicon"))
sys.path.insert(0, str(ROOT / "ml" / "data"))
import banglish as L  # noqa: E402
from build_splits import norm_key  # noqa: E402

SYSTEM_MESSAGE = (
    "You are a financial assistant who understands English, Bengali and "
    "Banglish(Bengali written in english). Extract transactions from text into a JSON list."
)
BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")

# category weights for expense items, per language (boost the weak spots)
CAT_W = {
    "banglish": {"rent": 4.0, "loan": 3.0, "medical": 3.0, "utilities": 2.5, "gift": 2.0, "grocery": 1.5,
                 "entertainment": 1.5, "other": 1.2},
    "bangla": {"mobile_recharge": 4.0, "other": 3.0, "entertainment": 2.5, "tuition": 2.0, "rent": 2.0, "loan": 1.5},
    "mixed": {"rent": 2.0, "loan": 2.0, "gift": 1.5, "medical": 1.5},
}


def nice_amount(rng: random.Random, lo: int, hi: int) -> int:
    a = rng.randint(lo, hi)
    step = 5 if a < 100 else 10 if a < 500 else 50 if a < 2000 else 100 if a < 10000 else 500 if a < 50000 else 1000
    a = max(lo, round(a / step) * step)
    return int(a)


def spelled_amount(rng: random.Random, a: int, lang: str) -> list[str]:
    """'ponero hajar' / 'পনেরো হাজার' / 'der lakh' renderings of a, or [] if a has no short form."""
    out: list[str] = []
    for unit, roman_u, bn_u in ((100000, ["lakh", "lac", "lokkho"], "লাখ"), (1000, ["hajar", "hazar"], "হাজার")):
        if a >= unit and a % unit == 0 and (a // unit) in L.NUM_WORDS_ROMAN:
            n = a // unit
            out += [f"{w} {u}" for w in L.NUM_WORDS_ROMAN[n] for u in roman_u]
            if lang in ("bangla", "mixed"):
                out += [f"{L.NUM_WORDS_BN[n]} {bn_u}"]
            break
        frac = a / unit
        if a >= unit and frac in L.FRACTION_WORDS:
            roman_f, bn_f = L.FRACTION_WORDS[frac]
            out += [f"{w} {u}" for w in roman_f for u in roman_u]
            if lang in ("bangla", "mixed"):
                out += [f"{bn_f} {bn_u}"]
            break
    if not out and a in L.NUM_WORDS_ROMAN and a >= 100:
        out += list(L.NUM_WORDS_ROMAN[a])
        if lang in ("bangla", "mixed"):
            out += [L.NUM_WORDS_BN[a]]
    if lang != "bangla":
        out = [f"{w} taka" if rng.random() < 0.5 else w for w in out]
    return out


def render_amount(rng: random.Random, a: int, lang: str) -> str:
    """Many ways to write the same number; all map back to float(a)."""
    forms = [str(a), str(a)]
    spelled = spelled_amount(rng, a, lang)
    if spelled and rng.random() < 0.25:
        return rng.choice(spelled)
    if lang != "bangla":
        forms += [f"{a}{rng.choice(L.CURRENCY)}"]
    if a >= 1000 and a % 100 == 0:
        k = a / 1000
        ks = f"{k:g}"
        forms += [f"{ks}k", f"{ks}K", f"{ks} k", f"{ks}k tk"]
        if a % 1000 == 0 and a <= 90000:
            forms += [f"{a // 1000} hajar", f"{a // 1000} hazar", f"{a // 1000} hajar taka"]
            if lang in ("bangla", "mixed"):
                forms += [f"{a // 1000} হাজার", f"{str(a // 1000).translate(BN_DIGITS)} হাজার"]
    if a >= 100000 and a % 10000 == 0:
        lk = a / 100000
        forms += [f"{lk:g} lakh", f"{lk:g} lac", f"{lk:g} lakh taka"]
        if lang in ("bangla", "mixed"):
            forms += [f"{lk:g} লাখ", f"{str(lk).rstrip('0').rstrip('.').translate(BN_DIGITS)} লাখ"]
    if a >= 1000 and rng.random() < 0.3:
        forms += [f"{a:,}"]
    if lang in ("bangla", "mixed"):
        bn = str(a).translate(BN_DIGITS)
        forms += [bn, f"{bn}{rng.choice(L.CURRENCY_BN)}", f"৳{a}", f"৳{bn}"]
    return rng.choice(forms)


def pick_item(rng: random.Random, lang: str, income: bool):
    pool = L.INCOME if income else L.ITEMS
    w = CAT_W.get(lang, {})
    weights = [w.get(it[2], 1.0) for it in pool]
    return rng.choices(pool, weights=weights, k=1)[0]


def surface(rng: random.Random, item, lang: str) -> str:
    ban, bn = item[0], item[1]
    if lang == "bangla":
        return rng.choice(bn) if bn else rng.choice(ban)
    if lang == "mixed":
        return rng.choice(bn) if (bn and rng.random() < 0.5) else rng.choice(ban)
    return rng.choice(ban)


def maybe_caps(rng: random.Random, s: str) -> str:
    r = rng.random()
    if r < 0.35:
        return s[:1].upper() + s[1:]
    if r < 0.40:
        return s.upper() if len(s) < 25 else s
    return s


VERB_TOKENS = {"pelam", "paisi", "pathalo", "pathailo", "pathalam", "dilo", "dilam", "nilam", "becha", "bikri",
               "jitlam", "dekhalam", "gelam", "katlam", "kinlam", "ferot", "shodh", "khawa", "theke",
               "পেলাম", "পাঠালো", "দিলো", "দিলাম", "নিলাম", "বিক্রি", "জিতলাম", "দেখালাম", "গেলাম", "কাটলাম", "থেকে", "ফেরত"}
EAT_VERBS = {"khaisi", "kheyechi", "খেলাম"}


def has_verb(name: str) -> bool:
    return any(tok in VERB_TOKENS for tok in name.replace("'", " ").split())


def clause(rng: random.Random, lang: str, income: bool) -> tuple[str, dict]:
    item = pick_item(rng, lang, income)
    cat = item[2]
    amount = nice_amount(rng, item[3], item[4])
    name = surface(rng, item, lang)
    amt = render_amount(rng, amount, lang)
    script_bn = lang == "bangla" or (lang == "mixed" and any("\u0980" <= ch <= "\u09ff" for ch in name))
    embedded = has_verb(name)
    if income:
        verbs = L.INCOME_VERBS_BN if script_bn else L.INCOME_VERBS
        verb = rng.choice(verbs)
        if embedded:  # "ammu pathalo 5k", "5k dhar ferot pelam"
            pats = [f"{name} {amt}", f"{amt} {name}", f"{name}: {amt}", f"{amt} taka {name}" if not script_bn else f"{amt} টাকা {name}"]
        elif script_bn:
            pats = [f"{name} {amt} {verb}", f"{name} থেকে {amt} {verb}", f"{amt} {name} {verb}", f"{name} {verb} {amt}",
                    f"{name} এর {amt} {verb}"]
        else:
            pats = [f"{name} {amt} {verb}", f"{name} theke {amt} {verb}", f"{amt} {name} {verb}", f"{name} {verb} {amt}",
                    f"{name} er {amt} {verb}", f"{name} {amt} tk {verb}"]
    else:
        verbs = [v for v in (L.EXPENSE_VERBS_BN if script_bn else L.EXPENSE_VERBS) if cat == "food" or v not in EAT_VERBS]
        verb = rng.choice(verbs)
        if embedded:  # "doctor dekhalam 800", "ammu ke pathalam 5k"
            pats = [f"{name} {amt}", f"{amt} {name}", f"{name}: {amt}", f"{name} - {amt}",
                    f"{amt} taka {name}" if not script_bn else f"{amt} টাকা {name}"]
        elif script_bn:
            pats = [f"{name} {amt} {verb}", f"{name} {amt}", f"{name} এ {amt} {verb}", f"{name} এর জন্য {amt} {verb}",
                    f"{amt} দিয়ে {name} {verb}" if verb not in EAT_VERBS else f"{amt} টাকার {name} {verb}",
                    f"{name}: {amt}", f"{amt} {name}", f"{name} {verb} {amt}"]
        else:
            pats = [f"{name} {amt} {verb}", f"{name} {amt}", f"{amt} {name}", f"{name} e {amt} {verb}",
                    f"{name} er jonno {amt} {verb}", f"{amt} diye {name}" if cat != "food" else f"{amt} takar {name} {verb}",
                    f"{name} {verb} {amt}", f"{name}: {amt}", f"{name} - {amt}", f"{name} te {amt} gelo"]
        if cat == "loan" and not embedded and rng.random() < 0.7:  # bare "dhar 5k" is ambiguous; keep a verb
            pats = [p for p in pats if verb in p] or pats
    text = rng.choice(pats)
    tx = {"description": name, "amount": float(amount), "category": cat,
          "type": "income" if income else "expense"}
    return text, tx


def sentence(rng: random.Random, lang: str) -> tuple[str, list[dict]]:
    r = rng.random()
    n = 1 if r < 0.40 else 2 if r < 0.75 else 3 if r < 0.93 else rng.choice([4, 5])
    script_bn = lang == "bangla"
    n_income = 0
    if rng.random() < 0.45:
        n_income = 1 if n > 1 or rng.random() < 0.6 else 0
        if n >= 3 and rng.random() < 0.2:
            n_income = 2
    kinds = [True] * n_income + [False] * (n - n_income)
    rng.shuffle(kinds)
    parts, txs = [], []
    for inc in kinds:
        t, tx = clause(rng, lang, inc)
        parts.append(t)
        txs.append(tx)
    time = rng.choice(L.TIME_BN if script_bn else L.TIME)
    conns = L.CONNECTORS_BN if script_bn else L.CONNECTORS
    out = parts[0]
    for i in range(1, n):
        # contrast word between an income and an expense reads naturally
        if kinds[i] != kinds[i - 1] and rng.random() < 0.5:
            c = rng.choice(L.CONTRAST_BN if script_bn else L.CONTRAST)
        else:
            c = rng.choice(conns)
        out += c + parts[i]
    if time:
        out = f"{time} {out}" if rng.random() < 0.8 else f"{out} {time}"
    if not script_bn and rng.random() < 0.3:
        out = rng.choice(L.FILLERS) + out
    out = maybe_caps(rng, out.strip()) + rng.choice(L.TAIL if not script_bn else ["", "", "।", "।"])
    out = " ".join(out.split())
    return out, txs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mix", default="banglish=0.60,bangla=0.24,mixed=0.16")
    ap.add_argument("--empty-frac", type=float, default=0.012)
    ap.add_argument("--out", type=Path, default=ROOT / "ml/datasets/synthetic_programmatic_v1.jsonl")
    ap.add_argument("--existing", type=Path, default=ROOT / "ml/datasets/synthetic_dataset.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    mix = {k: float(v) for k, v in (p.split("=") for p in args.mix.split(","))}
    seen = set()
    if args.existing.exists():
        for line in args.existing.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                u = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
                seen.add(norm_key(u))

    rows, stats = [], Counter()
    n_empty = round(args.n * args.empty_frac)
    empties = L.NO_TX[:]
    rng.shuffle(empties)
    for s in empties[:n_empty]:
        lang = "bangla" if any("ঀ" <= ch <= "৿" for ch in s) else ("english" if s[0].isupper() and " ami " not in s and "ajke" not in s.lower() and "kal" not in s.lower() else "banglish")
        rows.append((s, [], lang))
    attempts = 0
    while len(rows) < args.n and attempts < args.n * 20:
        attempts += 1
        lang = rng.choices(list(mix), weights=list(mix.values()), k=1)[0]
        text, txs = sentence(rng, lang)
        k = norm_key(text)
        if k in seen or len(text) < 6:
            continue
        seen.add(k)
        rows.append((text, txs, lang))
    rng.shuffle(rows)

    with args.out.open("w", encoding="utf-8") as f:
        for text, txs, lang in rows:
            complexity = "simple" if len(txs) <= 1 else "medium" if len(txs) <= 3 else "complex"
            row = {
                "id": f"txn_{uuid.uuid4().hex[:8]}",
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": json.dumps(txs, ensure_ascii=False, separators=(",", ":"))},
                ],
                "metadata": {"lang": lang, "complexity": complexity, "generated_by": "programmatic-lexicon-v1",
                             "timestamp": datetime.now(timezone.utc).isoformat(), "validated": True,
                             "validation_notes": "labels exact by construction", "tokens_input": None, "tokens_output": None},
                "raw_transactions": txs,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["lang:" + lang] += 1
            stats["ntx:" + str(len(txs))] += 1
            for t in txs:
                stats[f"cat:{t['category']}"] += 1
                stats["type:" + t["type"]] += 1
    print(f"wrote {len(rows)} rows to {args.out}")
    for k in sorted(stats):
        print(f"  {k:24s} {stats[k]}")


if __name__ == "__main__":
    main()
