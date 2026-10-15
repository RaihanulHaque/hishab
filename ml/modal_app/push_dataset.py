"""Publish the clean splits as a Hugging Face dataset (runs in Modal, where the HF token lives).

  conda run -n ML modal run ml/modal_app/push_dataset.py [--repo ryuma007/hishab-synthetic-v3]
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
size_categories: [1K<n<10K]
---

# hishab-synthetic-v3

Synthetic Bangladeshi personal-finance messages (English, Bangla, Banglish, mixed) paired with
the structured transactions they describe. Successor of `ryuma007/hishab-synthetic-v2`, whose
validation split was a subset of its train split; v3 is de-duplicated (NFKC + case + punctuation
normalised inputs) and split with stratification on language, source (base / hard) and
empty-output rows. No normalised instruction appears in more than one split.

| split | rows |
|---|---|
| train | {train} |
| validation | {validation} |
| test | {test} |

Columns: `id`, `language`, `complexity`, `source`, `instruction`, `translation` (English),
`transactions` (list of `{{description, amount, category, type}}`).

Categories: food, grocery, rent, utilities, mobile_recharge, transport, tuition, salary, gift,
loan, shopping, medical, entertainment, other. Type: income | expense. Amounts in BDT.
"""


@app.function(image=train_image, volumes=VOLUMES, secrets=[hf_secret], timeout=1800)
def push(repo: str) -> str:
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi

    splits = {}
    for name in ("train", "validation", "test"):
        p = Path(DATA_DIR) / "splits" / f"{name}.jsonl"
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:  # keep transactions as a JSON string: stable schema for Arrow
            r["transactions"] = json.dumps(r["transactions"], ensure_ascii=False)
        splits[name] = Dataset.from_list(rows)
    ds = DatasetDict(splits)
    ds.push_to_hub(repo, private=False)
    HfApi().upload_file(
        path_or_fileobj=CARD.format(**{k: len(v) for k, v in splits.items()}).encode(),
        path_in_repo="README.md", repo_id=repo, repo_type="dataset",
    )
    return f"https://huggingface.co/datasets/{repo}"


@app.local_entrypoint()
def main(repo: str = "ryuma007/hishab-synthetic-v3"):
    print(push.remote(repo))
