"""Numbers the way Bangladeshis write them: Banglish words with their common spellings, Bangla
script, Indian grouping (hajar / lakh / koti), fractions (der, arai, sare), and digit-word
mixes ("15 hajar 500"). `to_words` renders, `parse` reads back; every generated row is checked
round-trip so an amount in the data is right by construction.

    >>> to_words(15500)                      # one of several
    'ponero hajar pachsho'
    >>> parse('sare tin hajar')
    3500
    >>> parse('২ লাখ ২০ হাজার')
    220000
"""
from __future__ import annotations

import random
import re

# ---- 0..99, Roman: first spelling is the most common, the rest are what people also type
ONES_ROMAN: dict[int, list[str]] = {
    0: ["shunno", "zero"], 1: ["ek"], 2: ["dui"], 3: ["tin"], 4: ["char", "chaar"], 5: ["pach", "panch", "paach"],
    6: ["choy", "chhoy", "chhoi"], 7: ["sat", "saat"], 8: ["at", "aat"], 9: ["noy", "noi"], 10: ["dosh", "dos"],
    11: ["egaro", "egaro"], 12: ["baro", "baro"], 13: ["tero"], 14: ["chouddo", "chodda"], 15: ["ponero", "ponro", "pnero"],
    16: ["sholo", "shol"], 17: ["shotero", "sotero"], 18: ["atharo", "aatharo"], 19: ["unish"], 20: ["bish", "bis"],
    21: ["ekush"], 22: ["baish"], 23: ["teish"], 24: ["chobbish"], 25: ["pochish", "pochis"], 26: ["chabbish"],
    27: ["sataish"], 28: ["athash", "athaish"], 29: ["unotrish"], 30: ["trish", "tirish"], 31: ["ektrish"],
    32: ["botrish"], 33: ["tetrish"], 34: ["choutrish"], 35: ["poytrish", "paytrish"], 36: ["chhotrish", "chotrish"],
    37: ["saitrish"], 38: ["attrish"], 39: ["unochollish"], 40: ["chollish", "cholish"], 41: ["ekchollish"],
    42: ["biyallish"], 43: ["tetallish"], 44: ["chuallish"], 45: ["poytallish", "paytallish"], 46: ["chechollish"],
    47: ["satchollish"], 48: ["atchollish"], 49: ["unoponchash"], 50: ["ponchash", "pochash"], 51: ["ekanno"],
    52: ["bawanno"], 53: ["tippanno"], 54: ["chuanno"], 55: ["pochanno"], 56: ["chhappanno", "chappanno"],
    57: ["satanno"], 58: ["atanno"], 59: ["unoshat"], 60: ["shat", "shaat"], 61: ["ekshotti"], 62: ["bashotti"],
    63: ["teshotti"], 64: ["choushotti"], 65: ["poyshotti"], 66: ["chheshotti"], 67: ["satshotti"], 68: ["atshotti"],
    69: ["unoshottor"], 70: ["shottor", "sottor"], 71: ["ekattor"], 72: ["bahattor"], 73: ["tiattor"], 74: ["chuattor"],
    75: ["pochattor"], 76: ["chhiattor"], 77: ["satattor"], 78: ["atattor"], 79: ["unoashi"], 80: ["ashi"],
    81: ["ekashi"], 82: ["birashi"], 83: ["tirashi"], 84: ["chourashi"], 85: ["pochashi"], 86: ["chhiyashi"],
    87: ["satashi"], 88: ["atashi"], 89: ["unonobboi"], 90: ["nobboi", "nobbui"], 91: ["ekanobboi"], 92: ["biranobboi"],
    93: ["tiranobboi"], 94: ["chouranobboi"], 95: ["pochanobboi"], 96: ["chhianobboi"], 97: ["satanobboi"],
    98: ["atanobboi"], 99: ["niranobboi"],
}

ONES_BN: dict[int, str] = dict(enumerate(
    "শূন্য এক দুই তিন চার পাঁচ ছয় সাত আট নয় দশ এগারো বারো তেরো চৌদ্দ পনেরো ষোলো সতেরো আঠারো উনিশ "
    "বিশ একুশ বাইশ তেইশ চব্বিশ পঁচিশ ছাব্বিশ সাতাশ আটাশ ঊনত্রিশ ত্রিশ একত্রিশ বত্রিশ তেত্রিশ চৌত্রিশ পঁয়ত্রিশ "
    "ছত্রিশ সাঁইত্রিশ আটত্রিশ ঊনচল্লিশ চল্লিশ একচল্লিশ বিয়াল্লিশ তেতাল্লিশ চুয়াল্লিশ পঁয়তাল্লিশ ছেচল্লিশ "
    "সাতচল্লিশ আটচল্লিশ ঊনপঞ্চাশ পঞ্চাশ একান্ন বায়ান্ন তিপ্পান্ন চুয়ান্ন পঞ্চান্ন ছাপ্পান্ন সাতান্ন আটান্ন "
    "ঊনষাট ষাট একষট্টি বাষট্টি তেষট্টি চৌষট্টি পঁয়ষট্টি ছেষট্টি সাতষট্টি আটষট্টি ঊনসত্তর সত্তর একাত্তর বাহাত্তর "
    "তিয়াত্তর চুয়াত্তর পঁচাত্তর ছিয়াত্তর সাতাত্তর আটাত্তর ঊনআশি আশি একাশি বিরাশি তিরাশি চুরাশি পঁচাশি "
    "ছিয়াশি সাতাশি আটাশি ঊননব্বই নব্বই একানব্বই বিরানব্বই তিরানব্বই চুরানব্বই পঁচানব্বই ছিয়ানব্বই সাতানব্বই "
    "আটানব্বই নিরানব্বই".split()
))
assert len(ONES_BN) == 100

HUNDRED_ROMAN = ["sho", "shoto", "so"]
HUNDRED_BN = ["শো", "শ"]
THOUSAND_ROMAN = ["hajar", "hazar", "hajaar"]
THOUSAND_BN = ["হাজার"]
LAKH_ROMAN = ["lakh", "lac", "lokkho", "lakkho"]
LAKH_BN = ["লাখ", "লক্ষ"]
CRORE_ROMAN = ["koti", "crore", "kuti"]
CRORE_BN = ["কোটি"]

# x.5 of a unit: "der hajar" (1500), "arai lakh" (250000), "sare tin hajar" (3500)
HALF_ROMAN = {1.5: ["der", "dedh"], 2.5: ["arai", "aarai", "arhai"]}
HALF_BN = {1.5: "দেড়", 2.5: "আড়াই"}
SARE_ROMAN = ["sare", "saare", "share"]
SARE_BN = "সাড়ে"

BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
UNITS = [(10_000_000, CRORE_ROMAN, CRORE_BN), (100_000, LAKH_ROMAN, LAKH_BN), (1000, THOUSAND_ROMAN, THOUSAND_BN)]


def _ones(n: int, bn: bool, rng: random.Random) -> str:
    return ONES_BN[n] if bn else rng.choice(ONES_ROMAN[n])


def _hundreds(n: int, bn: bool, rng: random.Random, allow_half: bool) -> str:
    """0 < n < 1000 as words, e.g. 350 -> 'tinsho ponchash' or 'sare tinsho'."""
    assert 0 < n < 1000
    h, r = divmod(n, 100)
    if h == 0:
        return _ones(r, bn, rng)
    if bn:
        hundred = HUNDRED_BN[0]
        if allow_half and r == 50 and h + 0.5 in HALF_BN:
            return f"{HALF_BN[h + 0.5]}{hundred}"  # দেড়শো, আড়াইশো
        if allow_half and r == 50 and h >= 3:
            return f"{SARE_BN} {ONES_BN[h]}{hundred}"
        head = f"{ONES_BN[h]}{hundred}"
        return head if r == 0 else f"{head} {ONES_BN[r]}"
    hundred = rng.choice(HUNDRED_ROMAN)
    joined = rng.random() < 0.6  # "pachsho" vs "pach sho"
    if allow_half and r == 50 and h + 0.5 in HALF_ROMAN:
        half = rng.choice(HALF_ROMAN[h + 0.5])
        return f"{half}{hundred}" if joined else f"{half} {hundred}"
    if allow_half and r == 50 and h >= 3 and rng.random() < 0.5:
        return f"{rng.choice(SARE_ROMAN)} {_ones(h, False, rng)}{hundred if joined else ' ' + hundred}"
    head = f"{_ones(h, False, rng)}{hundred}" if joined else f"{_ones(h, False, rng)} {hundred}"
    return head if r == 0 else f"{head} {_ones(r, False, rng)}"


def to_words(n: int, lang: str = "banglish", rng: random.Random | None = None, *, mix_digits: bool = False) -> str:
    """Render integer taka amount n > 0 in Banglish (`lang` banglish/mixed/english) or Bangla
    script (`lang` bangla). With mix_digits the big part is digits and the unit is a word:
    "15 hajar 500", "২ লাখ ২০ হাজার", "1.5 lakh"."""
    rng = rng or random.Random()
    if n <= 0:
        raise ValueError("amount must be positive")
    bn = lang == "bangla"
    parts: list[str] = []
    rest = n
    for unit, roman_u, bn_u in UNITS:
        q, rest = divmod(rest, unit)
        if q == 0:
            continue
        u = bn_u[0] if bn else rng.choice(roman_u)
        # x.5 units: 150000 -> der lakh, 3500 -> sare tin hajar; only when the remainder is exactly half
        if rest * 2 == unit and q <= 9 and rng.random() < 0.6:
            if q + 0.5 in HALF_ROMAN:
                half = HALF_BN[q + 0.5] if bn else rng.choice(HALF_ROMAN[q + 0.5])
                parts.append(f"{half} {u}")
            else:
                parts.append(f"{SARE_BN if bn else rng.choice(SARE_ROMAN)} {_ones(q, bn, rng)} {u}")
            rest = 0
            continue
        if mix_digits:
            qs = str(q).translate(BN_DIGITS) if bn else str(q)
            if rest * 2 == unit and q <= 9:  # 1.5 lakh
                qs = f"{q}.5".translate(BN_DIGITS) if bn else f"{q}.5"
                rest = 0
            parts.append(f"{qs} {u}")
        else:
            parts.append(f"{_hundreds(q, bn, rng, allow_half=False) if q < 1000 else str(q)} {u}")
    if rest:
        if mix_digits and rest >= 100 and rng.random() < 0.5:
            parts.append(str(rest).translate(BN_DIGITS) if bn else str(rest))
        else:
            parts.append(_hundreds(rest, bn, rng, allow_half=not parts))
    return " ".join(parts)


# ---- parsing: words (either script, any spelling above, digits of either script) -> int

_WORD_VALUE: dict[str, int] = {}
for _n, _ws in ONES_ROMAN.items():
    for _w in _ws:
        _WORD_VALUE[_w] = _n
for _n, _w in ONES_BN.items():
    _WORD_VALUE[_w] = _n
_UNIT_VALUE: dict[str, int] = {}
for _ws, _v in ((HUNDRED_ROMAN + HUNDRED_BN, 100), (THOUSAND_ROMAN + THOUSAND_BN, 1000),
                (LAKH_ROMAN + LAKH_BN, 100_000), (CRORE_ROMAN + CRORE_BN, 10_000_000)):
    for _w in _ws:
        _UNIT_VALUE[_w] = _v
_HALF_VALUE: dict[str, float] = {w: v for v, ws in HALF_ROMAN.items() for w in ws}
_HALF_VALUE.update({w: v for v, w in HALF_BN.items()})
_SARE = set(SARE_ROMAN) | {SARE_BN}
_UNIT_SUFFIX = re.compile(r"^(.+?)(sho|shoto|so|শো|শ)$")
_NOISE = {"taka", "tk", "টাকা", "tak", "takar", "টাকার", "only"}


def _split_glued(tok: str) -> list[str]:
    """'pachsho' -> ['pach', 'sho'], 'dersho' -> ['der', 'sho'], 'পাঁচশো' -> ['পাঁচ', 'শো']."""
    m = _UNIT_SUFFIX.match(tok)
    if m and (m.group(1) in _WORD_VALUE or m.group(1) in _HALF_VALUE):
        return [m.group(1), m.group(2)]
    return [tok]


def parse(text: str) -> int | None:
    """Read an amount written in words, digits or a mix. None when nothing numeric is found or
    the phrase does not parse cleanly (e.g. a word that is not a number)."""
    t = text.lower().replace(",", "").replace("৳", " ")
    toks: list[str] = []
    for raw in re.findall(r"[0-9০-৯]+(?:\.[0-9০-৯]+)?k?|[a-z]+|[ঀ-৿]+", t):
        if raw in _NOISE:
            continue
        if raw.endswith("k") and raw[:-1].replace(".", "").isdigit():  # 80k, 1.5k
            toks += [raw[:-1], "hajar"]
            continue
        toks += _split_glued(raw)
    if not toks:
        return None
    total = 0.0
    group = 0.0       # value accumulated since the last big unit
    pending_half: float | None = None  # der / arai / sare X waiting for a unit
    sare = False
    for tok in toks:
        if re.fullmatch(r"[0-9০-৯]+(?:\.[0-9০-৯]+)?", tok):
            group += float(tok.translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")))
        elif tok in _SARE:
            sare = True
        elif tok in _HALF_VALUE:
            pending_half = _HALF_VALUE[tok]
        elif tok in _WORD_VALUE:
            v = _WORD_VALUE[tok]
            if sare:
                pending_half = v + 0.5
                sare = False
            else:
                group += v
        elif tok in _UNIT_VALUE:
            u = _UNIT_VALUE[tok]
            if pending_half is not None:
                group = pending_half
                pending_half = None
            if group == 0:
                group = 1  # "hajar taka" = 1000
            if u == 100:
                group *= 100
            else:
                total += group * u
                group = 0
        else:
            return None
    if pending_half is not None or sare:
        return None
    total += group
    return int(round(total)) if total > 0 else None


def all_forms(n: int, lang: str, rng: random.Random, k: int = 6) -> list[str]:
    """A handful of distinct renderings of n for data generation, words and digit mixes."""
    out: set[str] = set()
    for _ in range(k * 3):
        out.add(to_words(n, lang, rng))
        if n >= 1000:
            out.add(to_words(n, lang, rng, mix_digits=True))
        if len(out) >= k:
            break
    return sorted(out)
