"""Round-trip and spot checks for ml/hishab/numwords.py. Run: python -m pytest ml/tests -q"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hishab.numwords import parse, to_words  # noqa: E402


def test_roundtrip_many():
    rng = random.Random(11)
    amounts = [rng.choice([rng.randint(1, 999), rng.randint(1000, 99999), rng.randint(100000, 9999999)]) for _ in range(300)]
    amounts += [40, 100, 150, 250, 350, 1500, 2500, 3500, 15500, 150000, 220000, 12500000]
    for a in amounts:
        for lang in ("banglish", "bangla"):
            for mix in (False, True):
                w = to_words(a, lang, rng, mix_digits=mix)
                assert parse(w) == a, (a, lang, mix, w)


def test_spot_phrases():
    cases = {
        "ponero hajar taka": 15000, "sare tin hajar": 3500, "ek hajar pachsho": 1500, "dui lakh bish hajar": 220000,
        "der lakh": 150000, "chollish taka": 40, "tin sho ponchash": 350, "paytallish hajar": 45000, "eksho bish": 120,
        "15 hajar 500": 15500, "pach hajar 2 sho": 5200, "পনেরো হাজার": 15000, "আড়াই হাজার": 2500, "১৫ হাজার ৫০০": 15500,
        "সাড়ে তিন হাজার": 3500, "dersho taka": 150, "hajar taka": 1000, "2.5 lakh": 250000, "80k": 80000, "1.5k": 1500,
    }
    for s, v in cases.items():
        assert parse(s) == v, (s, parse(s), v)


def test_not_a_number():
    assert parse("bazar") is None
    assert parse("sare") is None
    assert parse("") is None
