"""Publish the clean splits as a Hugging Face dataset (runs in Modal, where the HF token lives).

  conda run -n ML modal run ml/modal_app/push_dataset.py [--repo ryuma007/hishab-synthetic-v5] [--train-split train_v10]

Dataset versions: v3 = the 4,409-row Gemini train set (model v3), v4 = the 20,333-row set
(models v7, v8, v9), v5 = v4 with the Gemini rows relabelled by two labellers plus the
`test_clean` / `numbers_clean` splits (model v10), v6 = v5 + 1,041 rows with 1 to 9 taka amounts and the
`small_amounts` split (model v12).
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
structured transactions they describe.

{history}

| split | rows | what it is |
|---|---|---|
{split_rows}

Train rows by source:

| source | rows | how it was made |
|---|---|---|
{sources}

Labels follow a written guideline (loans by direction, money gifts vs bought gifts, stated totals
are never an entry, questions and plans are `[]`):
https://github.com/RaihanulHaque/hishab/blob/main/docs/LABELLING_GUIDELINE.md

Columns: `id`, `language`, `complexity`, `source`, `instruction`, `translation` (English),
`transactions` (JSON string: list of `{{description, amount, category, type}}`), `label_source`
(`original`: the row's own label; `gold`: two independent labellers confirmed the original;
`gemini-3.8-flash`: both labellers agreed on a different label, which replaced it).

Categories: food, grocery, rent, utilities, mobile_recharge, transport, tuition, salary, gift,
loan, shopping, medical, entertainment, other. Type: income | expense. Amounts in BDT.

All rows are synthetic. Models trained on it: {models}.
"""

SPLIT_DOC = {
    "train": "everything below",
    "validation": "held out, for picking the best epoch (original labels)",
    "test": "held out, Gemini-generated, original labels (same rows since v3)",
    "test_clean": "the `test` rows where two independent labellers agree; their label is the gold. Use this one",
    "numbers": "held out, every amount spelled out in words (\"der hajar\", \"পনেরো হাজার\"); never train on it",
    "numbers_clean": "the `numbers` rows where the two labellers agree",
    "small_amounts": "held out, every message has a 1 to 9 taka item (\"5 taka rickshaw bhara\"); templates disjoint from train",
}

HISTORY = {
    "ryuma007/hishab-synthetic-v4": "Successor of `ryuma007/hishab-synthetic-v3`: same validation and test splits, a much "
                                    "larger train split, and a new `numbers` split.",
    "ryuma007/hishab-synthetic-v5": "Successor of `ryuma007/hishab-synthetic-v4`. The 4,409 Gemini-written train rows "
                                    "(`source` base / hard) were relabelled from scratch by Gemini 3.8 Flash and Claude "
                                    "Sonnet 5.5 against the guideline: 3,285 confirmed, 848 replaced by the agreed label, "
                                    "276 dropped where the two disagree. The test and numbers splits were relabelled the "
                                    "same way into `test_clean` (476 rows) and `numbers_clean` (317); the original splits "
                                    "are kept for comparison with older models.",
    "ryuma007/hishab-synthetic-v6": "Successor of `ryuma007/hishab-synthetic-v5`. v5's model read \"5 taka rickshaw bhara\" "
                                    "as 500: no train row had an amount under 10. v6 adds 1,041 train rows where every "
                                    "message has a 1 to 9 taka item, from 347 Claude Sonnet templates labelled with the "
                                    "guideline, amounts filled by a number-word engine (digits, Bangla digits, words), and "
                                    "a new `small_amounts` split (87 rows) from 87 held-out templates. All v5 splits are "
                                    "unchanged.",
}

MODELS = {
    "ryuma007/hishab-synthetic-v4": "`ryuma007/hishab-qwen3.5-0.8b-json-full-v8` and `-v9` (and their `-GGUF` repos)",
    "ryuma007/hishab-synthetic-v5": "`ryuma007/hishab-qwen3.5-0.8b-json-full-v10`, `...-pipe-signed-full-v11` and "
                                    "`...-v11-trim38k` (and their `-GGUF` repos)",
    "ryuma007/hishab-synthetic-v6": "`ryuma007/hishab-qwen3.5-0.8b-pipe-signed-full-v12-trim38k` (and its `-GGUF` repo)",
}

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
    "micro-sonnet": "Claude Sonnet templates with a 1 to 9 taka item, labelled by the guideline; amounts filled by the number-word engine",
}



@app.function(image=train_image, volumes=VOLUMES, secrets=[hf_secret], timeout=1800)
def push(repo: str, train_split: str = "train") -> str:
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi

    splits = {}
    for name in ("train", "validation", "test", "test_clean", "numbers", "numbers_clean", "small_amounts"):
        p = Path(DATA_DIR) / "splits" / f"{train_split if name == 'train' else name}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:  # keep transactions as a JSON string: stable schema for Arrow
            r["transactions"] = json.dumps(r["transactions"], ensure_ascii=False)
            r["label_source"] = r.get("label_source") or "original"
        splits[name] = Dataset.from_list(rows)
    from collections import Counter
    counts = Counter(splits["train"]["source"])
    sources = "\n".join(f"| {k} | {n:,} | {SOURCES.get(k, '')} |" for k, n in counts.most_common())
    ds = DatasetDict(splits)
    ds.push_to_hub(repo, private=False)
    HfApi().upload_file(
        path_or_fileobj=CARD.format(repo_name=repo.split("/")[-1], sources=sources, history=HISTORY.get(repo, ""),
                                    models=MODELS.get(repo, "see the Hishab repo"),
                                    split_rows="\n".join(f"| {k} | {len(v):,} | {SPLIT_DOC[k]} |" for k, v in splits.items())).encode(),
        path_in_repo="README.md", repo_id=repo, repo_type="dataset",
    )
    return f"https://huggingface.co/datasets/{repo}"


@app.local_entrypoint()
def main(repo: str = "ryuma007/hishab-synthetic-v5", train_split: str = "train_v10"):
    print(push.remote(repo, train_split))
