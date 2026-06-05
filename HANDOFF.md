# Hishab LLM Fine-Tuning Handoff

This document summarizes the objective, decisions, artifacts, and next steps so another coding agent can continue from the current workspace state.

## High-Level Objective

Build a small local language model for a Bangladeshi personal-finance app that can read messy user transaction notes in:

- English
- Bangla script
- Banglish, meaning Bengali written in Latin characters
- mixed Bangla-English-Banglish code-switching

The model should output a strict JSON list of transactions:

```json
[
  {
    "description": "Short item description",
    "amount": 1000.0,
    "category": "food",
    "type": "expense"
  }
]
```

Target base model:

```text
Qwen/Qwen3.5-0.8B
```

The user wants this because previous SFT attempts on small models worked better for English and Bangla, but performed poorly on Banglish. The chosen approach is tokenizer expansion plus continued pretraining, then SFT.

## Current Strategy

The intended training pipeline is:

```text
1. Expand / improve synthetic SFT dataset
2. Analyze Qwen tokenizer on English, Bangla, Banglish, mixed data
3. Expand tokenizer with high-value Banglish finance tokens
4. Continued pretraining / embedding warmup so new tokens learn meaning
5. SFT on ChatML JSON extraction dataset
6. Evaluate valid JSON, amount accuracy, category accuracy, type accuracy, Banglish accuracy
7. Quantize / export for phone deployment
```

Important distinction:

This is not full pretraining from scratch. The next phase should be continued pretraining / embedding warmup for the expanded tokenizer.

## Dataset State

Primary dataset files:

```text
synthetic_dataset.jsonl
synthetic_dataset_chatml.jsonl
```

Both currently contain:

```text
5000 rows
```

Language distribution in `synthetic_dataset.jsonl`:

```text
banglish: 2003
bangla:   1252
english: 997
mixed:   748
```

Complexity distribution:

```text
simple:  1694
medium:  3196
complex: 110
```

Transaction type distribution:

```text
expense: 7683
income:  1353
```

Category distribution:

```text
food:            1101
transport:       909
tuition:         887
shopping:        871
utilities:       761
grocery:         742
salary:          667
mobile_recharge: 565
other:           533
gift:            525
medical:         469
loan:            411
rent:            319
entertainment:   276
```

Known issue:

```text
duplicate user inputs: 9
```

Those duplicates were already present; the new append script avoids adding new duplicate user messages.

## Dataset Generation Work Completed

Added:

```text
append_synthetic_dataset.py
```

Purpose:

- Uses `GOOGLE_API_KEY` from `.env`
- Calls Gemini to generate synthetic examples
- Appends to existing files without deleting old rows
- Writes both production format and ChatML format
- Balances toward 5000 rows using this target mix:

```text
40% banglish
25% bangla
20% english
15% mixed
```

Command used to reach 5000:

```bash
conda run -n ML python append_synthetic_dataset.py --target-total 5000 --balanced-5000
```

The script includes retry/backoff for temporary Gemini `503 high demand` errors.

## Corpus State

Continued-pretraining corpus:

```text
gemma_pruning_corpus_50mb.txt
```

Size:

```text
201985 lines
about 51 MB
```

Origin:

Built by `vocab_prunning_dataset.py` from:

- `Ayon128/Banglish-English`
- `csebuetnlp/BanglaNMT`

The corpus is general mixed Banglish/Bangla/English, not finance-only. For the current phase, it is acceptable because the goal is language/token adaptation. SFT will teach JSON transaction extraction.

When packed with the expanded tokenizer at `block_size=512`, it produced:

```text
Loaded text lines: 206985
Training blocks: 21518 x 512 tokens
```

The `206985` count includes corpus lines plus SFT user messages when `synthetic_dataset.jsonl` is included.

## Tokenizer Analysis Completed

Added:

```text
qwen_tokenizer_analysis.py
```

Generated:

```text
qwen_tokenizer_analysis.json
qwen_expanded_tokenizer_analysis.json
qwen_added_tokens.txt
qwen_expanded_remaining_candidates.txt
```

Base tokenizer:

```text
Qwen/Qwen3.5-0.8B
```

Base tokenizer vocab size:

```text
248077
```

Base tokenizer efficiency:

```text
bangla   rows=1252 avg_tokens=40.58 median=38.0 chars/token=1.63 tokens/word=3.60
english  rows= 997 avg_tokens=21.31 median=21.0 chars/token=3.09 tokens/word=1.91
banglish rows=2003 avg_tokens=28.05 median=27.0 chars/token=2.57 tokens/word=2.13
mixed    rows= 748 avg_tokens=40.68 median=40.0 chars/token=1.87 tokens/word=3.24
all      rows=5000 avg_tokens=31.73 median=30.0 chars/token=2.33 tokens/word=2.62
```

Important findings:

- Banglish was not catastrophically tokenized, but many high-value words split.
- Bangla script tokenization is much less efficient than English, but the user’s observed issue was specifically Banglish behavior.
- Adding bare tokens like `taka` alone made tokenization worse because Qwen has space-aware tokenization.
- Best expansion strategy was:

```text
AddedToken(token, single_word=True) + " token-with-leading-space"
```

Then capitalized variants were added too, because phone/autocorrect often capitalizes sentence-start words like `Ajke`.

Frequently split Banglish words included:

```text
taka, ajke, diye, jonno, kinlam, kichu, lagse, korlam,
kheyechi, fees, disi, dilam, theke, bkash, khoroch,
khabar, paisi, laglo, rickshaw, photocopy, tution, bazar,
cng, riksha
```

Candidate source token file:

```text
qwen_added_tokens.txt
```

It contains:

```text
405 Banglish candidate tokens
```

## Expanded Tokenizer Completed

Added:

```text
expand_qwen_tokenizer.py
```

Generated expanded tokenizer folder:

```text
qwen3_0_8b_banglish_tokenizer/
```

Contents:

```text
banglish_added_tokens_manifest.json
chat_template.jinja
tokenizer.json
tokenizer_config.json
```

Expansion summary:

```text
Base tokenizer:       Qwen/Qwen3.5-0.8B
Original vocab size:  248077
Source tokens:        405
Actual added entries: 1620
New vocab size:       249603
```

Why 1620 entries?

For each of the 405 source tokens, the script adds:

- lowercase single-word token
- lowercase leading-space token
- capitalized single-word token
- capitalized leading-space token

Example:

```text
ajke
 Ajke
 ajke
Ajke
```

Actual behavior is handled through `AddedToken` and string tokens in the script.

Expanded tokenizer efficiency:

```text
bangla   rows=1252 avg_tokens=40.45 median=38.0 chars/token=1.65 tokens/word=3.58
english  rows= 997 avg_tokens=21.23 median=21.0 chars/token=3.10 tokens/word=1.90
banglish rows=2003 avg_tokens=22.88 median=22.0 chars/token=3.13 tokens/word=1.76
mixed    rows= 748 avg_tokens=40.24 median=40.0 chars/token=1.92 tokens/word=3.21
all      rows=5000 avg_tokens=29.55 median=27.0 chars/token=2.57 tokens/word=2.46
```

Banglish improvement:

```text
Before: 28.05 avg tokens
After:  22.88 avg tokens
Improvement: about 18.4% fewer Banglish tokens
```

Probe example:

```text
Ajke tution fees 1k disi r riksha vara 80
Before: 16 tokens
After:  13 tokens
```

Important:

Only the tokenizer has been saved. The model weights have not yet been resized/trained successfully in this workspace.

## Continued Pretraining Script Prepared

Added:

```text
qwen_continue_pretrain.py
```

Purpose:

This is an embedding-warmup continued-pretraining script. It is intentionally conservative:

1. Loads base model `Qwen/Qwen3.5-0.8B`
2. Loads expanded tokenizer from `qwen3_0_8b_banglish_tokenizer`
3. Resizes model embeddings to match the tokenizer
4. Initializes new token rows from averages of old base-tokenizer subtoken embeddings
5. Freezes the base model
6. Trains only the embedding / LM-head rows corresponding to added tokens
7. Saves the adapted model

Why this phase:

New tokenizer entries initially have no semantic meaning. SFT alone may not be enough. The warmup phase teaches the added Banglish tokens to map into useful embedding space before JSON SFT.

Validation already performed:

```bash
conda run -n ML python qwen_continue_pretrain.py --dataset-only --block-size 512
```

Output:

```text
Loaded text lines: 206985
Training blocks: 21518 x 512 tokens
```

## Environment Notes

The active conda env is:

```text
ML
```

Verified packages:

```text
torch        2.9.1+cu130
transformers 5.5.0
datasets     4.3.0
peft         installed
trl          installed
bitsandbytes installed
accelerate   installed
unsloth      installed
```

However, the current machine/workspace cannot use GPU:

```text
torch.cuda.is_available() = False
nvidia-smi cannot communicate with NVIDIA driver
```

Because of this, the actual 0.8B model continued-pretraining was not run here. The script deliberately raises if CUDA is unavailable.

## Recommended Next Step

Run embedding warmup on a GPU machine/runtime:

```bash
conda run -n ML python qwen_continue_pretrain.py \
  --base-model Qwen/Qwen3.5-0.8B \
  --expanded-tokenizer qwen3_0_8b_banglish_tokenizer \
  --corpus gemma_pruning_corpus_50mb.txt \
  --sft-dataset synthetic_dataset.jsonl \
  --output-dir qwen3_0_8b_banglish_embedding_warmup \
  --block-size 512 \
  --batch-size 1 \
  --grad-accum 16 \
  --max-steps 500
```

Suggested first GPU test:

```bash
conda run -n ML python qwen_continue_pretrain.py \
  --base-model Qwen/Qwen3.5-0.8B \
  --expanded-tokenizer qwen3_0_8b_banglish_tokenizer \
  --corpus gemma_pruning_corpus_50mb.txt \
  --sft-dataset synthetic_dataset.jsonl \
  --output-dir qwen3_0_8b_banglish_embedding_warmup_test \
  --block-size 256 \
  --batch-size 1 \
  --grad-accum 8 \
  --max-steps 20 \
  --save-every 20
```

If the small test works, run the longer 500-step warmup. If GPU memory is limited, reduce:

```text
block-size: 512 -> 256
grad-accum: 16 -> 8
batch-size: keep 1
```

## After Embedding Warmup

Next phase should be SFT on:

```text
synthetic_dataset_chatml.jsonl
```

SFT should use the warmed model output directory:

```text
qwen3_0_8b_banglish_embedding_warmup
```

Likely SFT options:

- Use Unsloth if it supports this exact Qwen checkpoint cleanly
- Otherwise use Hugging Face `trl` `SFTTrainer` plus PEFT LoRA/QLoRA
- Make sure tokenizer is loaded from the warmup model directory or `qwen3_0_8b_banglish_tokenizer`
- Make sure model embeddings are already resized before LoRA training

Suggested SFT training style:

```text
LoRA/QLoRA
max_seq_length: 512 or 1024
epochs: 2-4
learning_rate: 1e-4 to 2e-4 for LoRA
train on ChatML messages
evaluate strict JSON validity
```

## Evaluation Plan

Create a held-out evaluation file before SFT if possible. Do not evaluate only on training examples.

Metrics to compute:

- valid JSON rate
- exact transaction count match
- amount accuracy
- category accuracy
- income/expense type accuracy
- Banglish-only accuracy
- mixed-script accuracy
- no-transaction `[]` accuracy

Use test examples like:

```text
Ajke tution fee 5k disi, riksha 80, cha 20
Bkash e ammu 2k pathalam, cashout charge 35
আজ market থেকে grocery কিনলাম ২k আর rickshaw ৮০
Salary paisi 35k but basha vara dilam 12k
Gotokal kichu khoroch kori nai
```

## Important Caveats

1. The expanded tokenizer must travel with the model.

   Any later model loading, SFT, inference, or export must use the expanded tokenizer or a model directory that contains it.

2. Resizing embeddings is mandatory.

   A base `Qwen/Qwen3.5-0.8B` model cannot directly use the expanded tokenizer unless:

   ```python
   model.resize_token_embeddings(len(expanded_tokenizer))
   ```

3. New tokens are not meaningful until trained.

   The expanded tokenizer improves token count, but the added embedding rows need continued pretraining / warmup.

4. Phone deployment needs tokenizer compatibility checks.

   If exporting to GGUF / llama.cpp / mobile runtime, verify that added tokens survive conversion.

5. Existing dataset has some unrealistic examples from older generation.

   Earlier tails showed examples like very large food or Bkash amounts. New generation was biased toward realistic amounts, but before final SFT it may be worth adding a cleaning script to flag outlier amounts per category.

## Files Added In This Session

```text
HANDOFF.md
append_synthetic_dataset.py
qwen_tokenizer_analysis.py
expand_qwen_tokenizer.py
qwen_continue_pretrain.py
qwen_added_tokens.txt
qwen_tokenizer_analysis.json
qwen_expanded_tokenizer_analysis.json
qwen_expanded_remaining_candidates.txt
qwen3_0_8b_banglish_tokenizer/
```

## Files Modified / Expanded In This Session

```text
synthetic_dataset.jsonl
synthetic_dataset_chatml.jsonl
```

Both were expanded from 1402 rows to 5000 rows.

## Suggested Immediate Task For Next Agent

1. Run the 20-step embedding warmup test on a GPU runtime.
2. Inspect loss and confirm output directory saves correctly.
3. Run 500-step embedding warmup.
4. Create an SFT script using the warmed model and `synthetic_dataset_chatml.jsonl`.
5. Create an evaluation script before/after SFT.

