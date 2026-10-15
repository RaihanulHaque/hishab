"""Transaction schema, target rendering and lenient output parsing."""
from __future__ import annotations

import json
import re
from typing import Any

CATEGORIES = [
    "food", "grocery", "rent", "utilities", "mobile_recharge", "transport", "tuition",
    "salary", "gift", "loan", "shopping", "medical", "entertainment", "other",
]
TYPES = ["income", "expense"]
LANGUAGES = ["banglish", "bangla", "english", "mixed"]
FORMATS = ["json", "two_phase"]

REQUIRED_KEYS = ("description", "amount", "category", "type")


def dumps_transactions(txs: list[dict[str, Any]]) -> str:
    """Minified JSON array, non-ASCII kept as is (Bangla descriptions stay readable)."""
    clean = [
        {
            "description": str(t["description"]),
            "amount": float(t["amount"]),
            "category": str(t["category"]),
            "type": str(t["type"]),
        }
        for t in txs
    ]
    return json.dumps(clean, ensure_ascii=False, separators=(",", ":"))


def render_target(fmt: str, translation: str | None, txs: list[dict[str, Any]]) -> str:
    """The assistant text the model is trained to produce."""
    body = dumps_transactions(txs)
    if fmt == "json":
        return body
    if fmt == "two_phase":
        return f"TRANSLATION: {(translation or '').strip()}\nJSON:\n{body}"
    raise ValueError(f"unknown format {fmt!r}")


_JSON_MARK = re.compile(r"JSON:\s*", re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)


def _coerce_amount(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.replace(",", "").replace("৳", "").strip()
        try:
            return float(s)
        except ValueError:
            return None
    return None


def normalize_tx(t: Any) -> dict[str, Any] | None:
    """Return a canonical transaction dict, or None if the object is unusable."""
    if not isinstance(t, dict) or not all(k in t for k in REQUIRED_KEYS):
        return None
    amount = _coerce_amount(t["amount"])
    if amount is None:
        return None
    cat = str(t["category"]).strip().lower()
    typ = str(t["type"]).strip().lower()
    return {
        "description": str(t["description"]).strip(),
        "amount": round(amount, 2),
        "category": cat,
        "type": typ,
    }


def parse_output(text: str) -> dict[str, Any]:
    """Parse a model reply of either format.

    Returns a dict with:
      translation: str (empty when absent)
      txs: list[dict] | None  (None = no parseable JSON array)
      strict: bool  (array parsed AND every item is well-formed with valid category/type)
      error: str | None
    """
    out: dict[str, Any] = {"translation": "", "txs": None, "strict": False, "error": None}
    if not text:
        out["error"] = "empty"
        return out
    body = text
    if "TRANSLATION:" in text or _JSON_MARK.search(text):
        parts = _JSON_MARK.split(text, maxsplit=1)
        out["translation"] = parts[0].replace("TRANSLATION:", "", 1).strip()
        body = parts[1] if len(parts) > 1 else ""
    body = _FENCE.sub("", body).strip()
    start, end = body.find("["), body.rfind("]")
    if start == -1 or end == -1 or end < start:
        out["error"] = "no_array"
        return out
    try:
        arr = json.loads(body[start : end + 1])
    except json.JSONDecodeError as e:
        out["error"] = f"json:{e.msg}"
        return out
    if not isinstance(arr, list):
        out["error"] = "not_list"
        return out
    txs = [normalize_tx(t) for t in arr]
    strict = all(t is not None and t["category"] in CATEGORIES and t["type"] in TYPES for t in txs)
    out["txs"] = [t for t in txs if t is not None]
    out["strict"] = strict
    if not strict:
        out["error"] = "malformed_item"
    return out
