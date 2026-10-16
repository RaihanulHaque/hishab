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

## Held-out test (518 rows, never seen in training)

| metric | value |
|---|---|
| exact transaction-set match | {exact:.1f}% |
| tuple F1 | {f1:.1f}% |
| valid JSON | {valid:.1f}% |
| Banglish / Bangla / English / mixed exact | {bl:.1f} / {bn:.1f} / {en:.1f} / {mx:.1f} |
| numbers split exact (357 rows, amounts spelled out in words) | {num} |

Quantized GGUF builds for llama.cpp live in [`{repo}-GGUF`](https://huggingface.co/{repo}-GGUF).
Training pipeline: https://github.com/RaihanulHaque/hishab (`ml/`). Full results: `eval_hf.json` in this repo.
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
    test = meta["splits"]["test"]
    lang = test["by_language"]
    pct = lambda d, k="exact_match": 100 * d[k]  # noqa: E731
    repo = f"{repo_prefix}{run.replace('__', '-').replace('_', '-')}"
    example = json.dumps([{"description": "tuition fee", "amount": 5000.0, "category": "tuition", "type": "expense"},
                          {"description": "rickshaw", "amount": 80.0, "category": "transport", "type": "expense"}],
                         indent=1)
    num = f"{100 * meta['splits']['numbers']['all']['exact_match']:.1f}%" if "numbers" in meta["splits"] else "-"
    training = ("full fine-tune, all weights" if meta.get("tune") == "full"
                else f"LoRA r={meta.get('lora_r', 16)}, merged")
    training += f", {meta['epochs']:g} epochs, lr {meta['lr']:g}"
    card = CARD.format(hf_id=meta["hf_id"], repo=repo, dataset=dataset, training=training, num=num, repo_name=repo.split("/")[-1], example=example,
                       system_prompt=system_prompt(meta["fmt"]), exact=pct(test["all"]), f1=100 * test["all"]["tx_f1"],
                       valid=pct(test["all"], "valid_json"), bl=pct(lang["banglish"]), bn=pct(lang["bangla"]),
                       en=pct(lang["english"]), mx=pct(lang["mixed"]))
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(merged), repo_id=repo, ignore_patterns=["*.gguf"])
    api.upload_file(path_or_fileobj=(rdir / "eval_hf.json").read_bytes(), path_in_repo="eval_hf.json", repo_id=repo)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo)
    return f"https://huggingface.co/{repo}"


@app.local_entrypoint()
def main(run: str = "qwen3.5-0.8b__json__lora__v5"):
    print("pushed:", push_model.remote(run))
