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

## Current state (2026-10-15)

Eight fine-tunes (Qwen3.5-0.8B, LFM2.5-350M, LFM2.5-1.2B, Gemma-3-270M; two output formats)
were trained on a clean, de-duplicated split and evaluated on 518 held-out rows, then each
candidate was quantized up to eleven ways and re-evaluated with llama.cpp at kiln's pinned
commit. The two best configurations were then retrained on 9.5k rows (v3 + 5k programmatic
Banglish/Bangla rows + 155 hand-written rows; see `ml/README.md`) as `v4` / `v5`.

Best full-precision model: **Qwen3.5-0.8B, JSON-only output, v5**: 88.2% exact transaction
match (Banglish 84.0%). On the phone, pick a quant by size budget:

| GGUF | MB | exact | notes |
|---|---|---|---|
| `hishab-qwen3.5-0.8b-json-lora-v5-GGUF` Q6_K | 630 | 86.5 | best quality |
| `hishab-qwen3.5-0.8b-json-lora-v5-GGUF` Q4_K_M-imat | 529 | 84.6 | default phone build |
| `hishab-lfm2.5-350m-two-phase-lora-v4-GGUF` Q5_K_M | 260 | 84.7 | fast tier, 3x faster |

The 518-row test set has a label-noise ceiling near 90%: most remaining misses are
convention disagreements inside the Gemini-generated gold (`ml/results/ERROR_ANALYSIS.md`).
Per-language and per-quant tables are in [`ml/results/RESULTS.md`](ml/results/RESULTS.md).

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
