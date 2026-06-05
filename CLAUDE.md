# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Fine-tune `Qwen/Qwen3.5-0.8B` to extract structured financial transactions from user input written in English, Bangla script, Banglish (Bengali in Latin characters), and mixed code-switching. Target output is strict JSON:

```json
[{"description": "...", "amount": 1000.0, "category": "food", "type": "expense"}]
```

Valid categories: `food`, `grocery`, `rent`, `utilities`, `mobile_recharge`, `transport`, `tuition`, `salary`, `gift`, `loan`, `shopping`, `medical`, `entertainment`, `other`

## Repository Layout

```
hishab/
├── backend/          # FastAPI app (Gemini-powered, Supabase persistence)
├── ml/
│   ├── datasets/     # Training data and generation scripts (model-agnostic)
│   ├── scripts/      # Data utilities (token analysis, corpus building)
│   └── models/
│       └── qwen3_0_8b/          # Model-specific work — add gemma_3_1b/ etc. here
│           ├── tokenizer/       # Tokenizer analysis, expansion, and artifacts
│           └── training/        # Pretrain + SFT scripts and model checkpoints
└── experiments/      # Past experiments and scratch notebooks
```

## Environment

All ML work runs in the `ML` conda environment:

```bash
conda run -n ML python <script.py>
```

Training scripts require a CUDA GPU — they deliberately raise if `torch.cuda.is_available()` is False. The current development machine has no GPU; run training on a separate GPU machine/runtime.

Key packages: `torch 2.9.1+cu130`, `transformers 5.5.0`, `datasets 4.3.0`, `peft`, `trl`, `bitsandbytes`, `accelerate`, `unsloth`, `google-genai`, `tokenizers`

API keys are in `.env` (Hugging Face, `GOOGLE_API_KEY` for Gemini, Supabase, etc.).

## Training Pipeline

The pipeline has 5 phases, run in order:

### Phase 1 — Dataset Generation (COMPLETE: 5000 rows)

```bash
conda run -n ML python ml/scripts/append_synthetic_dataset.py --target-total 5000 --balanced-5000
```

Uses Gemini (`gemini-2.5-flash-lite`) to generate synthetic examples. Writes to both `ml/datasets/synthetic_dataset.jsonl` (with metadata) and `ml/datasets/synthetic_dataset_chatml.jsonl` (pure ChatML). Language target: 40% banglish, 25% bangla, 20% english, 15% mixed.

### Phase 2 — Tokenizer Analysis & Expansion (COMPLETE)

```bash
# Analyze base tokenizer efficiency on the dataset
conda run -n ML python ml/models/qwen3_0_8b/tokenizer/tokenizer_analysis.py

# Expand tokenizer with 405 Banglish candidate tokens → 1620 entries (4 variants each)
conda run -n ML python ml/models/qwen3_0_8b/tokenizer/expand_tokenizer.py
```

Output: `ml/models/qwen3_0_8b/tokenizer/banglish_tokenizer/` (249,697 vocab, up from 248,077). Banglish token efficiency improved ~18.4%. Each source token gets 4 variants: lowercase/capitalized × bare/leading-space, because Qwen is space-aware and phone autocorrect capitalizes sentence-start words.

### Phase 3 — Embedding Warmup / Continued Pretraining (COMPLETE: 500 steps)

```bash
# Dry-run to validate dataset blocks (no GPU needed)
conda run -n ML python ml/models/qwen3_0_8b/training/continue_pretrain.py --dataset-only

# Quick smoke test on GPU
conda run -n ML python ml/models/qwen3_0_8b/training/continue_pretrain.py \
  --output-dir ml/models/qwen3_0_8b/training/warmup_test \
  --block-size 256 --batch-size 1 --grad-accum 8 --max-steps 20 --save-every 20

# Full warmup run (uses defaults pointing to correct paths)
conda run -n ML python ml/models/qwen3_0_8b/training/continue_pretrain.py
```

Freezes all base model weights; trains only the embedding/LM-head rows for added tokens. Output at `ml/models/qwen3_0_8b/training/banglish_embedding_warmup/` (checkpoints at 100/200/300/400/500 steps, ~1.5 GB).

### Phase 4 — SFT with LoRA (NEXT: script ready, not yet run)

```bash
conda run -n ML python ml/models/qwen3_0_8b/training/sft.py
```

Loads from `banglish_embedding_warmup/`, applies LoRA (r=16, alpha=32) to all attention+MLP projections, trains 3 epochs on `synthetic_dataset_chatml.jsonl`, saves adapter to `sft_lora_final/`.

### Phase 5 — Evaluation & Quantization (TODO)

No evaluation script exists yet. Metrics: valid JSON rate, exact transaction count match, amount accuracy, category accuracy per language, Banglish-only accuracy. After eval, quantize to GGUF or llama.cpp for mobile.

## Critical Invariants

**The expanded tokenizer must travel with the model.** Any script that loads the warmed model or SFT adapter must use `ml/models/qwen3_0_8b/tokenizer/banglish_tokenizer/` (or a model dir that contains it).

**`resize_token_embeddings` is mandatory.** A freshly downloaded `Qwen/Qwen3.5-0.8B` cannot use the expanded tokenizer without `model.resize_token_embeddings(len(expanded_tokenizer))`. The warmup script handles this; any new script loading the base model must too.

**New tokens need training.** The expanded tokenizer alone reduces token count but added embedding rows have no semantic meaning until after embedding warmup.

**GGUF export compatibility.** Before quantizing, verify that added tokens survive conversion — llama.cpp and mobile runtimes may not handle `AddedToken` entries identically.

## Key Artifacts

| Path | Description |
|------|-------------|
| `ml/datasets/synthetic_dataset.jsonl` | 5000-row SFT dataset with metadata (language, complexity, categories) |
| `ml/datasets/synthetic_dataset_chatml.jsonl` | Same 5000 rows in pure ChatML format — used directly for SFT |
| `ml/datasets/gemma_pruning_corpus_50mb.txt` | ~51 MB mixed Banglish/Bangla/English corpus for embedding warmup |
| `ml/models/qwen3_0_8b/tokenizer/added_tokens.txt` | 405 source Banglish candidate tokens |
| `ml/models/qwen3_0_8b/tokenizer/banglish_tokenizer/` | Expanded tokenizer (249,697 vocab) |
| `ml/models/qwen3_0_8b/training/banglish_embedding_warmup/` | Warmed model after 500 steps (~1.5 GB, gitignored) |
| `ml/models/qwen3_0_8b/training/sft_lora/` | SFT LoRA output — empty until Phase 4 runs (gitignored) |
| `HANDOFF.md` | Detailed session history and design decisions |

## Backend App

Run from the project root:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The app uses Gemini (`gemini-2.0-flash`) for transaction extraction and Supabase for persistence. It is independent from the ML training pipeline.
