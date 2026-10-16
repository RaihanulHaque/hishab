"""Publish the clean splits as a Hugging Face dataset (runs in Modal, where the HF token lives).

  conda run -n ML modal run ml/modal_app/push_dataset.py [--repo ryuma007/hishab-synthetic-v4]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modal_app.common import DATA_DIR, VOLUMES, app, hf_secret, train_image  # noqa: E402

CARD = """---
language: [bn, en]
license: mit
task_categories: [text-generation]
tags: [finance, banglish, bangla, information-extraction, hishab]
size_categories: [10K<n<100K]
---

# {repo_name}

Bangladeshi personal-finance messages (English, Bangla, Banglish, mixed) paired with the
structured transactions they describe. Successor of `ryuma007/hishab-synthetic-v3`: same
validation and test splits, a much larger train split, and a new `numbers` split.

| split | rows | what it is |
|---|---|---|
| train | {train} | everything below |
| validation | {validation} | held out, for picking the best epoch |
| test | {test} | held out, Gemini-generated, same as v3 |
| numbers | {numbers} | held out, every amount spelled out in words ("der hajar", "পনেরো হাজার"); never train on it |

Train rows by source:

| source | rows | how it was made |
|---|---|---|
{sources}

Labels follow a written guideline (loans by direction, money gifts vs bought gifts, stated totals
are never an entry, questions and plans are `[]`):
https://github.com/RaihanulHaque/hishab/blob/main/docs/LABELLING_GUIDELINE.md

Columns: `id`, `language`, `complexity`, `source`, `instruction`, `translation` (English),
`transactions` (JSON string: list of `{{description, amount, category, type}}`).

Categories: food, grocery, rent, utilities, mobile_recharge, transport, tuition, salary, gift,
loan, shopping, medical, entertainment, other. Type: income | expense. Amounts in BDT.

All rows are synthetic. Models trained on it: `ryuma007/hishab-qwen3.5-0.8b-json-full-v8`
(and its `-GGUF` repo).
"""

SOURCES = {
    "base": "Gemini-generated messages, deduplicated",
    "hard": "Gemini-generated hard cases",
    "programmatic-lexicon-v1": "lexicon-driven generator with exact labels",
    "codex-template-v2": "Codex-written templates; amounts filled by a number-word engine, some with stated totals and typing noise",
    "codex-template-v2-bangla-words": "Codex templates in Bangla script with amounts in words",
    "numwords-aug-v1": "existing rows with digits rewritten as number words",
    "negatives-v1": "messages with no transaction, labelled []",
    "handwritten-v1": "hand-written",
    "handwritten-v2": "hand-written, targeted fixes",
}



@app.function(image=train_image, volumes=VOLUMES, secrets=[hf_secret], timeout=1800)
def push(repo: str) -> str:
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi

    splits = {}
    for name in ("train", "validation", "test", "numbers"):
        p = Path(DATA_DIR) / "splits" / f"{name}.jsonl"
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:  # keep transactions as a JSON string: stable schema for Arrow
            r["transactions"] = json.dumps(r["transactions"], ensure_ascii=False)
        splits[name] = Dataset.from_list(rows)
    from collections import Counter
    counts = Counter(splits["train"]["source"])
    sources = "\n".join(f"| {k} | {n:,} | {SOURCES.get(k, '')} |" for k, n in counts.most_common())
    ds = DatasetDict(splits)
    ds.push_to_hub(repo, private=False)
    HfApi().upload_file(
        path_or_fileobj=CARD.format(repo_name=repo.split("/")[-1], sources=sources,
                                    **{k: len(v) for k, v in splits.items()}).encode(),
        path_in_repo="README.md", repo_id=repo, repo_type="dataset",
    )
    return f"https://huggingface.co/datasets/{repo}"


@app.local_entrypoint()
def main(repo: str = "ryuma007/hishab-synthetic-v4"):
    print(push.remote(repo))
