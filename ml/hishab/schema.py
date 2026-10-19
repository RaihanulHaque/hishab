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
FORMATS = ["json", "two_phase", "pipe_signed"]
PIPE_NONE = "NONE"  # pipe_signed reply when there is no transaction

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


def _fmt_amount(v: Any) -> str:
    a = float(v)
    return str(int(a)) if a.is_integer() else f"{a:.2f}".rstrip("0").rstrip(".")


def dumps_pipe_signed(txs: list[dict[str, Any]]) -> str:
    """One `description|±amount|category` line per transaction: - for expense, + for income.
    About half the tokens of the JSON array (no keys, no quotes, no type field, no `.0`)."""
    if not txs:
        return PIPE_NONE
    lines = []
    for t in txs:
        desc = " ".join(str(t["description"]).replace("|", "/").split())
        sign = "+" if str(t["type"]) == "income" else "-"
        lines.append(f"{desc}|{sign}{_fmt_amount(t['amount'])}|{t['category']}")
    return "\n".join(lines)


def render_target(fmt: str, translation: str | None, txs: list[dict[str, Any]]) -> str:
    """The assistant text the model is trained to produce."""
    if fmt == "pipe_signed":
        return dumps_pipe_signed(txs)
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


def _parse_pipe(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"translation": "", "txs": None, "strict": False, "error": None}
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if lines == [PIPE_NONE]:
        out.update(txs=[], strict=True)
        return out
    txs, strict = [], True
    for line in lines:
        parts = line.split("|")
        if len(parts) < 3:
            strict = False
            continue
        desc, amt, cat = "|".join(parts[:-2]).strip(), parts[-2].strip(), parts[-1].strip().lower()
        sign = amt[:1]
        amount = _coerce_amount(amt[1:] if sign in "+-−" and sign else amt)
        if amount is None:
            strict = False
            continue
        if sign not in ("+", "-", "−"):
            strict = False  # unsigned: keep the row as an expense but flag it
        typ = "income" if sign == "+" else "expense"
        strict = strict and cat in CATEGORIES
        txs.append({"description": desc, "amount": round(abs(amount), 2), "category": cat, "type": typ})
    if not txs and lines:
        out["error"] = "no_rows"
        return out
    out["txs"] = txs
    out["strict"] = strict
    if not strict:
        out["error"] = "malformed_item"
    return out


def _looks_pipe(text: str) -> bool:
    t = text.strip()
    if t == PIPE_NONE:
        return True
    return "|" in t and "[{" not in t and not t.startswith("[") and "JSON:" not in t


def parse_output(text: str) -> dict[str, Any]:
    """Parse a model reply of any format (json, two_phase or pipe_signed; detected from the text).

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
    if _looks_pipe(text):
        return _parse_pipe(text)
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
