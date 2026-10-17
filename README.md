# Hishab

Hishab is a Bangladeshi personal-finance app whose core trick is turning a messy chat message
into structured transactions, on the phone, with no server:

```
"Ajke tution fee 5k disi, riksha 80, cha 20"
->
[{"description":"tuition fee","amount":5000.0,"category":"tuition","type":"expense"},
 {"description":"rickshaw","amount":80.0,"category":"transport","type":"expense"},
 {"description":"tea","amount":20.0,"category":"food","type":"expense"}]
```

Inputs come in English, Bangla script, Banglish (Bangla in Latin letters) and mixtures of the
three. The extractor is a small open model fine-tuned on synthetic data, quantized to GGUF and
run on device through [kiln](https://github.com/RaihanulHaque/kiln) (llama.cpp on the phone CPU).

## Repository

| path | purpose |
|---|---|
| `ml/` | the fine-tuning, evaluation and quantization pipeline (runs on Modal). Start with [`ml/README.md`](ml/README.md). |
| `ml/results/RESULTS.md` | current leaderboard and the shipping recommendation |
| `backend/` | FastAPI prototype backend: Gemini extraction + Supabase storage (independent of `ml/`) |
| `experiments/` | notebooks and scripts that preceded the pipeline, see [`experiments/README.md`](experiments/README.md) |
| `docs/` | historical design notes |
| `CLAUDE.md` | working notes for coding agents: environment, invariants, commands |

## Current state (2026-10-17)

Shipping line: `Qwen/Qwen3.5-0.8B`, JSON-only output, full fine-tune on 20,333 rows (v9).
Held-out scores (518-row `test`, 357-row digit-free `numbers` split; exact = the set of
(amount, category, type) matches gold):

| build | MB | test | numbers | notes |
|---|---|---|---|---|
| v9 bf16 (HF) | | 89.2 | 83.2 | `ryuma007/hishab-qwen3.5-0.8b-json-full-v9` |
| v9 Q6_K | 630 | 89.2 | 83.8 | `...-v9-GGUF`, recommended phone build |
| v9 Q4_K_M-imat | 529 | 89.6 | 84.0 | smallest; now as good as Q6_K |

v9 is v8 with one fix: training now tokenizes with `tokenizer.json`'s own backend
(`ml/modal_app/hfeval.load_tokenizer`). transformers 5 was splitting Bengali words at their
combining marks, so v3..v8 trained on tokens the phone never produces; their GGUFs lost up to
16 points on Bangla-script numbers. The full history, data mix and lessons per version are in
[`ml/results/EXPERIMENTS.md`](ml/results/EXPERIMENTS.md); the remaining test errors are mostly
label-convention noise in the Gemini gold (`ml/results/ERROR_ANALYSIS.md`,
`docs/LABELLING_GUIDELINE.md`), and a relabelled test set is in progress (`ml/data/relabel.py`).

## Setup

```bash
conda activate ML                      # has modal; heavy ML libs live in the Modal images
cp .env.example .env                   # fill in keys (never commit .env)
modal token new                        # once; Modal secret `huggingface-secret` must expose HF_TOKEN
```

Backend prototype:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Data

`ml/datasets/synthetic_dataset.jsonl` holds 5200 Gemini-generated examples (40% Banglish,
25% Bangla, 20% English, 15% mixed; 14 categories). `ml/data/build_splits.py` turns them into
the train / validation / test files in `ml/datasets/splits/`. Published copies: dataset
`ryuma007/hishab-synthetic-v2` (old, leaky validation split) and the models
`ryuma007/qwen3.5-0.8b-hishab-synthetic-v2` and `ryuma007/lfm2.5-350M-hishab-synthetic-v2x`.

## Security notes

`.env` was tracked in git until 2026-10-15 and the notebooks once contained a Hugging Face
token in plain text. Both are gone from the working tree but remain in history: rotate every
key that was in them (Gemini, Supabase service role, SMTP, Hugging Face).
