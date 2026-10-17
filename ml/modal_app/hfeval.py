"""HF-side evaluation helpers shared by train.py and train_grpo.py (no Modal app registration here)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from modal_app.common import DATA_DIR

MAX_NEW_TOKENS = {"json": 192, "two_phase": 320}

# Probe used to verify the tokenizer honours tokenizer.json. transformers 5.x's Qwen2Tokenizer
# rebuilds the pre-tokenizer with the old Qwen2 regex (\p{L}+), which splits Bengali words at
# every combining mark (nukta, vowel signs). tokenizer.json for Qwen3.5 says [\p{L}\p{M}]+, and
# llama.cpp follows tokenizer.json, so a model trained with the transformers tokenizer sees
# different Bangla tokens on the phone than it saw in training (v3..v8 all have this defect).
_BANGLA_PROBE = "\u098f\u09df\u09be\u09b0\u09aa\u09cb\u09b0\u09cd\u099f\u09c7\u09b0"  # এয়ারপোর্টের


def load_tokenizer(path_or_id: str):
    """AutoTokenizer whose Rust backend is loaded straight from tokenizer.json, so that
    tokenization matches the `tokenizers` library and llama.cpp exactly."""
    import unicodedata
    from pathlib import Path as _P

    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path_or_id)
    if _P(path_or_id).is_dir():
        tj = _P(path_or_id) / "tokenizer.json"
    else:
        from huggingface_hub import hf_hub_download

        tj = _P(hf_hub_download(path_or_id, "tokenizer.json"))
    raw = Tokenizer.from_file(str(tj))
    probe = unicodedata.normalize("NFC", _BANGLA_PROBE)
    before = tok(probe, add_special_tokens=False)["input_ids"]
    want = raw.encode(probe, add_special_tokens=False).ids
    if before != want:
        tok._tokenizer = raw  # swap the backend; chat template and special tokens stay
        after = tok(probe, add_special_tokens=False)["input_ids"]
        print(f"tokenizer backend replaced from tokenizer.json: probe {len(before)} -> {len(after)} tokens (faithful: {after == want})")
        if after != want:
            raise RuntimeError("tokenizer still differs from tokenizer.json after backend swap")
    else:
        print("tokenizer already faithful to tokenizer.json")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok



def load_split(name: str) -> list[dict]:
    p = Path(DATA_DIR) / "splits" / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def eval_splits(eval_limit: int = 0) -> dict[str, list[dict]]:
    """validation + test, plus any extra held-out slices present in /splits: numbers.jsonl (digit-free
    spelled amounts, never trained on) and test_clean / numbers_clean (rows where two frontier labellers
    agree with each other under docs/LABELLING_GUIDELINE.md; see ml/data/relabel_merge.py)."""
    out = {"validation": load_split("validation"), "test": load_split("test")}
    for extra in ("numbers", "test_clean", "numbers_clean"):
        if (Path(DATA_DIR) / "splits" / f"{extra}.jsonl").exists():
            out[extra] = load_split(extra)
    if eval_limit:
        out = {k: v[:eval_limit] for k, v in out.items()}
    return out


@torch.inference_mode()
def generate_all(model, tokenizer, rows: list[dict], fmt: str, batch_size: int = 32) -> tuple[list[str], float]:
    from hishab.prompts import build_messages

    tokenizer.padding_side = "left"
    preds: list[str] = []
    n_tokens = 0
    t0 = time.time()
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        texts = [
            tokenizer.apply_chat_template(build_messages(fmt, r["instruction"]), tokenize=False, add_generation_prompt=True)
            for r in batch
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS[fmt],
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1] :]
        for g in gen:
            n_tokens += int((g != tokenizer.pad_token_id).sum())
        preds += [t.strip() for t in tokenizer.batch_decode(gen, skip_special_tokens=True)]
        print(f"  generated {min(i + batch_size, len(rows))}/{len(rows)}", flush=True)
    return preds, n_tokens / max(len(rows), 1)


def run_eval(model, tokenizer, name, fmt, out_dir, results, splits, batch_size):
    from hishab.metrics import evaluate

    for split, rows in splits.items():
        print(f"--- generating on {split} ({len(rows)} rows, batch {batch_size})")
        preds, avg_tok = generate_all(model, tokenizer, rows, fmt, batch_size=batch_size)
        metrics = evaluate(rows, preds)
        metrics["avg_completion_tokens"] = round(avg_tok, 1)
        results["splits"][split] = metrics
        with (out_dir / f"preds_{split}.jsonl").open("w", encoding="utf-8") as f:
            for r, p in zip(rows, preds):
                f.write(json.dumps({"id": r["id"], "language": r["language"], "instruction": r["instruction"],
                                    "gold": r["transactions"], "pred": p}, ensure_ascii=False) + "\n")
        print(f"{name} {split}: exact={metrics['all']['exact_match']:.3f} f1={metrics['all']['tx_f1']:.3f} "
              f"valid={metrics['all']['valid_json']:.3f} avg_tok={avg_tok:.0f}")
    (out_dir / "eval_hf.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
