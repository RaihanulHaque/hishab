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
├── backend/            # FastAPI app (Gemini-powered, Supabase persistence)
├── ml/
│   ├── hishab/         # pure-python shared package: schema, prompts, metrics, model registry
│   ├── data/           # build_splits.py -> ml/datasets/splits/{train,validation,test}.jsonl
│   ├── modal_app/      # Modal apps: train.py, gguf.py, report.py, push_dataset.py (see ml/README.md)
│   ├── datasets/       # raw synthetic data + splits + pretraining corpus
│   ├── scripts/        # Gemini dataset-generation utilities
│   └── results/        # RESULTS.md / results.json written by report.py
├── experiments/        # notebooks (the TRL notebooks produced the published v2 HF models)
├── docs/               # HANDOFF_2026-01 (old tokenizer-expansion plan, code not in this checkout)
└── README.md
```

## Environment

Use the `ML` conda env for every command: `conda run -n ML ...`.

**This machine has no GPU and the `ML` env has no transformers/trl/peft.** It only has `modal`,
CPU torch and fastapi. All training, evaluation and GGUF work runs on Modal
(`conda run -n ML modal run ml/modal_app/<app>.py ...`). Images pin `transformers==5.10.2`,
`trl==1.5.1`, `peft==0.19.1`. The GGUF image builds llama.cpp at the commit kiln
(`../kiln/third_party/llama.cpp`) is pinned to, so phone and eval numerics agree.

Secrets: the Modal secret `huggingface-secret` exposes `HF_TOKEN` (the `HF_TOKEN2` in `.env`
is dead). `.env` is tracked in git and the TRL notebooks contain a hardcoded HF token: rotate
before publishing anything.

## Pipeline (current)

Full details and commands in `ml/README.md`.

1. `ml/data/build_splits.py` dedupes the 5200 synthetic rows (14 near-duplicates) and makes a
   stratified 4409 / 259 / 518 train / validation / test split. The published
   `ryuma007/hishab-synthetic-v2` validation split is a subset of its train split; never
   report numbers on it.
2. `ml/modal_app/train.py` runs TRL SFT (LoRA, completion-only loss, prompt-completion
   format) for each (model, output format) in parallel, merges the adapter, and evaluates
   bf16 greedy generation on validation + test. Output formats: `json` (array only) and
   `two_phase` (`TRANSLATION:` line then `JSON:`).
3. `ml/modal_app/gguf.py` converts a run to F16 GGUF, builds an in-domain imatrix, writes
   Q8_0 .. IQ3 quants and evaluates each with `llama-server`.
4. `ml/modal_app/report.py` aggregates everything into `ml/results/RESULTS.md`.

Candidate models (`ml/hishab/registry.py`): Qwen3.5-0.8B, LFM2.5-350M, LFM2.5-1.2B,
Gemma-3-270M. All are verified to run in kiln.

## Critical Invariants

**Chat-template parity.** `train.py` asserts the generation prompt is a prefix of the fully
rendered training example. Qwen3.5 is trained and served with thinking off
(`<think>\n\n</think>\n\n` in the assistant turn); llama.cpp's `--jinja` default matches.

**Never evaluate on training rows.** Use `ml/datasets/splits/test.jsonl` only.

**Transformers 5 tokenizer files.** `tokenizer_config.json` written by transformers 5 says
`TokenizersBackend`; the GGUF image uses transformers 5 too, with a fallback that rewrites it
to `PreTrainedTokenizerFast` for the converter.

## Key Artifacts

| Path | Description |
|------|-------------|
| `ml/datasets/synthetic_dataset.jsonl` | 5200 raw rows with metadata (lang, complexity, generator) |
| `ml/datasets/splits/` | clean train / validation / test (+ stats.json) |
| Modal volume `hishab-data` | `/splits` mirror of the above |
| Modal volume `hishab-runs` | `/runs/<model>__<fmt>__<tune>/{adapter,merged,gguf,eval_hf.json,eval_gguf/}` |
| HF `ryuma007/hishab-synthetic-v2` | old dataset (leaky validation split) |
| HF `ryuma007/qwen3.5-0.8b-hishab-synthetic-v2[-Q4_K_M-GGUF]` | first published model (notebook pipeline, vanilla tokenizer) |

## Backend App

Run from the project root:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The app uses Gemini (`gemini-2.0-flash`) for transaction extraction and Supabase for persistence. It is independent from the ML training pipeline.
