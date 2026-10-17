"""Publish the merged bf16 weights of a run (plus tokenizer, eval json and a short card) to the HF Hub.

  conda run -n ML modal run ml/modal_app/push_model.py --run qwen3.5-0.8b__json__lora__v5

Repo name: ryuma007/hishab-<run with __ and _ turned into ->  (e.g. hishab-qwen3.5-0.8b-json-lora-v5).
The GGUF repos published by gguf.py carry a -GGUF suffix next to it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modal_app.common import RUNS_DIR, VOLUMES, app, gguf_image, hf_secret  # noqa: E402

CARD = """---
license: apache-2.0
base_model: {hf_id}
language: [bn, en]
tags: [finance, information-extraction, banglish, bangla, qwen3.5, hishab]
datasets: [{dataset}]
---

# {repo_name}

bf16 weights of `{hf_id}` fine-tuned ({training}) to turn a Banglish / Bangla /
English message about money into a JSON array of transactions:

```json
{example}
```

Categories: food, grocery, rent, utilities, mobile_recharge, transport, tuition, salary, gift,
loan, shopping, medical, entertainment, other. Type is `expense` or `income`.

System prompt used in training (use it verbatim):

```
{system_prompt}
```

Qwen3.5 is trained and served with thinking disabled (empty `<think>` block in the assistant turn).

## Held-out results (bf16, greedy; none of these rows were seen in training)

Splits of [`{dataset}`](https://huggingface.co/datasets/{dataset}). *exact* = the set of
(amount, category, type) tuples equals the gold set; F1 is micro F1 over those tuples.

| split | rows | exact | tuple F1 | valid JSON | Banglish / Bangla / English / mixed exact |
|---|---|---|---|---|---|
{rows}
{splits_note}
## Use

```python
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{repo}")
tok._tokenizer = Tokenizer.from_file(hf_hub_download("{repo}", "tokenizer.json"))  # see below
model = AutoModelForCausalLM.from_pretrained("{repo}", torch_dtype="bfloat16")
msgs = [{{"role": "system", "content": SYSTEM_PROMPT}},  # the prompt above, verbatim
        {{"role": "user", "content": "Salary paisi 35k but basha vara dilam 12k"}}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
out = model.generate(ids, max_new_tokens=384, do_sample=False)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

The `tok._tokenizer` line matters for Bangla script: transformers 5.10 rebuilds a `Qwen2Tokenizer`
that splits Bengali words at their combining marks, while this model was trained (and llama.cpp
runs) with the tokenization in `tokenizer.json`. English and Banglish are unaffected.

## Limitations

- Trained and evaluated on synthetic messages only; real users type messier Banglish.
- Category conventions follow the Hishab labelling guideline
  (https://github.com/RaihanulHaque/hishab/blob/main/docs/LABELLING_GUIDELINE.md); other apps may
  want e.g. tutoring income as `tuition` rather than `salary`.
- Amounts in spelled-out Banglish are the weakest area (see the numbers rows above).

Quantized GGUF builds for llama.cpp live in [`{repo}-GGUF`](https://huggingface.co/{repo}-GGUF).
Training pipeline: https://github.com/RaihanulHaque/hishab (`ml/`). Full results: `eval_hf.json` in this repo.
"""

SPLITS = {
    "test_clean": "**test_clean**",
    "numbers_clean": "**numbers_clean** (amounts in words)",
    "test": "test (original labels)",
    "numbers": "numbers (original labels, amounts in words)",
}

CLEAN_NOTE = """
`test_clean` and `numbers_clean` are the rows where two independent labellers (Gemini 3.8 Flash and
Claude Sonnet 5.5, following the labelling guideline) agree, with their label as the gold; they are
the headline numbers. The original `test` labels break the guideline on about 14% of rows (plans and
stated totals recorded as entries, tutoring income as `tuition`), so this model scores lower there by
design.
"""


@app.function(image=gguf_image, cpu=4, timeout=60 * 60, volumes=VOLUMES, secrets=[hf_secret])
def push_model(run: str, repo_prefix: str = "ryuma007/hishab-", dataset: str = "ryuma007/hishab-synthetic-v4") -> str:
    from huggingface_hub import HfApi

    from hishab.prompts import system_prompt

    rdir = Path(RUNS_DIR) / run
    merged = rdir / "merged"
    if not (merged / "config.json").exists():
        raise RuntimeError(f"{merged} has no merged model")
    meta = json.loads((rdir / "eval_hf.json").read_text())
    pct = lambda d, k="exact_match": 100 * d[k]  # noqa: E731
    rows = []
    for name, doc in SPLITS.items():
        s = meta["splits"].get(name)
        if not s:
            continue
        a, lang = s["all"], s["by_language"]
        by_lang = " / ".join(f"{pct(lang[k]):.1f}" if k in lang else "-" for k in ("banglish", "bangla", "english", "mixed"))
        rows.append(f"| {doc} | {a['n']} | {pct(a):.1f} | {100 * a['tx_f1']:.1f} | {pct(a, 'valid_json'):.1f} | {by_lang} |")
    splits_note = CLEAN_NOTE if "test_clean" in meta["splits"] else ""
    repo = f"{repo_prefix}{run.replace('__', '-').replace('_', '-')}"
    example = json.dumps([{"description": "tuition fee", "amount": 5000.0, "category": "tuition", "type": "expense"},
                          {"description": "rickshaw", "amount": 80.0, "category": "transport", "type": "expense"}],
                         indent=1)
    training = ("full fine-tune, all weights" if meta.get("tune") == "full"
                else f"LoRA r={meta.get('lora_r', 16)}, merged")
    training += f", {meta['epochs']:g} epochs, lr {meta['lr']:g}"
    card = CARD.format(hf_id=meta["hf_id"], repo=repo, dataset=dataset, training=training, repo_name=repo.split("/")[-1],
                       example=example, system_prompt=system_prompt(meta["fmt"]), rows="\n".join(rows),
                       splits_note=splits_note)
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(merged), repo_id=repo, ignore_patterns=["*.gguf"])
    api.upload_file(path_or_fileobj=(rdir / "eval_hf.json").read_bytes(), path_in_repo="eval_hf.json", repo_id=repo)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo)
    return f"https://huggingface.co/{repo}"


@app.local_entrypoint()
def main(run: str = "qwen3.5-0.8b__json__lora__v5", dataset: str = "ryuma007/hishab-synthetic-v4"):
    print("pushed:", push_model.remote(run, dataset=dataset))
