"""Vocabulary trimming for byte-level BPE tokenizers (Qwen3.5): keep the tokens Hishab can use.

Qwen3.5-0.8B ties its 248k-row embedding to the output layer, so that one table is ~40% of the
phone file and is read in full for every generated token. Hishab only needs English, Bangla and
Banglish. `select_tokens` keeps:
  - every token seen when encoding the training / eval texts (messages and targets),
  - the added (special) tokens and every single-character token (the byte alphabet, so any
    text still encodes, byte by byte if need be),
  - the lowest-id (most frequent) tokens that are plain ASCII or Bangla script, up to `target`,
  - the merge ancestors of every kept token (BPE builds a token from two parents).
`trim_tokenizer_json` then renumbers the kept base tokens 0..N-1 in their original order, puts the
added tokens right after them, and drops every merge whose parts or result were removed. Any text
whose tokens were all kept encodes to exactly the same pieces as before; other text falls back to
smaller kept pieces. Pure python (json + regex) so it runs locally and in Modal images alike.
"""
from __future__ import annotations

import re
from typing import Iterable

# ASCII, the Bengali block, and the punctuation Bangla text uses (danda, ZWJ/ZWNJ, curly quotes, dashes)
_SCRIPT_OK = re.compile(r"^[\x00-\x7Fঀ-৿‌‍–—‘’“”…।॥৳]*$")


def _byte_decoder() -> dict[str, int]:
    """GPT-2 byte-level alphabet: the printable character standing for each byte."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def _merges(tj: dict) -> list[tuple[str, str]]:
    return [tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m) for m in tj["model"]["merges"]]


def select_tokens(tj: dict, used_ids: Iterable[int], target: int) -> list[int]:
    """Old ids (base vocab and added tokens) to keep, sorted."""
    vocab: dict[str, int] = tj["model"]["vocab"]
    inv = {i: s for s, i in vocab.items()}
    added = {t["id"] for t in tj["added_tokens"]}
    keep = set(used_ids) | added
    keep |= {i for s, i in vocab.items() if len(s) == 1}

    parents = {a + b: (a, b) for a, b in _merges(tj)}

    def close(ids: set[int]) -> set[int]:
        stack = [inv[i] for i in ids if i in inv]
        while stack:
            s = stack.pop()
            for p in parents.get(s, ()):
                pid = vocab[p]
                if pid not in ids:
                    ids.add(pid)
                    stack.append(p)
        return ids

    keep = close(keep)
    dec = _byte_decoder()
    for i in sorted(inv):
        if len(keep) >= target:
            break
        if i in keep:
            continue
        try:
            text = bytes(dec[c] for c in inv[i]).decode("utf-8")
        except (UnicodeDecodeError, KeyError):
            continue  # a fragment of a multi-byte character: only kept as an ancestor
        if _SCRIPT_OK.match(text):
            keep.add(i)
            keep = close(keep)
    return sorted(keep)


def trim_tokenizer_json(tj: dict, keep: list[int]) -> tuple[dict, list[int]]:
    """Return (new tokenizer.json, old_ids) where old_ids[new_id] is the row to copy from the
    old embedding. Base tokens first (original order), then the added tokens."""
    import copy

    vocab: dict[str, int] = tj["model"]["vocab"]
    kept = set(keep)
    added_ids = sorted(t["id"] for t in tj["added_tokens"])
    assert all(a in kept for a in added_ids), "added tokens must be kept"
    base = [(i, s) for s, i in sorted(vocab.items(), key=lambda kv: kv[1]) if i in kept]
    new_vocab = {s: n for n, (_, s) in enumerate(base)}
    old_ids = [i for i, _ in base]
    remap = {old: n for n, old in enumerate(old_ids)}
    for a in added_ids:
        remap[a] = len(old_ids)
        old_ids.append(a)

    out = copy.deepcopy(tj)
    out["model"]["vocab"] = new_vocab
    merges = [(a, b) for a, b in _merges(tj) if a in new_vocab and b in new_vocab and a + b in new_vocab]
    out["model"]["merges"] = [f"{a} {b}" for a, b in merges] if isinstance(tj["model"]["merges"][0], str) else [list(m) for m in merges]
    for t in out["added_tokens"]:
        t["id"] = remap[t["id"]]
    # post_processor / decoder hold no ids for Qwen; refuse silently wrong output for other models
    assert "ids" not in str(out.get("post_processor") or ""), "post_processor references token ids"
    return out, old_ids
