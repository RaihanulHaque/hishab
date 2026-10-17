# Hishab ML pipeline

Fine-tune small open models to turn messy Bangla / Banglish / English finance messages into
strict JSON transactions, pick the best one, quantize it to GGUF, and ship it in the phone app
through [kiln](../../kiln) (llama.cpp on the device CPU).

Everything heavy runs on **Modal**; the laptop only builds the dataset splits and reads reports.
Use the `ML` conda env for every command.

```
ml/
├── hishab/            pure-python package shared by laptop and Modal containers
│   ├── schema.py      categories, target rendering (json | two_phase), lenient output parser
│   ├── prompts.py     short system prompts per output format
│   ├── metrics.py     scoring: valid JSON, exact match, tuple F1, per-field acc, per language
│   └── registry.py    candidate base models + LoRA targets
├── data/build_splits.py   dedupe + stratified train/validation/test  ->  ml/datasets/splits/
├── modal_app/
│   ├── common.py      Modal app, volumes (hishab-data, hishab-runs, hishab-hf-cache), images
│   ├── train.py       TRL SFT (LoRA or full) per (model, format), then bf16 eval on val + test
│   ├── gguf.py        HF -> F16 GGUF -> imatrix -> many quants -> llama-server eval of each
│   ├── push_dataset.py  publish splits to HF (v5: --train-split train_v10)
│   ├── push_model.py    publish a run's merged bf16 weights + card as ryuma007/hishab-<run>
│   └── report.py      aggregate every eval json into ml/results/RESULTS.md
├── datasets/          raw synthetic data, splits, pretraining corpus
├── scripts/           dataset generation utilities (Gemini)
└── results/           RESULTS.md + results.json written by report.py
```

## Output formats

* `json` – the assistant replies with the JSON array only. Fewest output tokens, fastest on
  the phone.
* `two_phase` – `TRANSLATION: <english>` line then `JSON:` and the array (what the v2 models
  were trained on). The translation is a cheap chain of thought; it costs ~2x output tokens.

Both are trained and compared. Loss is on the completion only (TRL prompt-completion format).

## Commands

```bash
# 1. splits (laptop; needs pyarrow for the v2 parquet with translations)
conda run -n ML python ml/data/generate_banglish.py --n 5000 --seed 7      # programmatic Banglish/Bangla rows, no API
conda run -n ML python ml/data/build_splits.py \
    --extra-train ml/datasets/synthetic_programmatic_v1.jsonl ml/datasets/synthetic_handwritten_v1.jsonl
conda run -n ML modal volume put hishab-data ml/datasets/splits /splits --force
conda run -n ML modal run ml/modal_app/push_dataset.py

# 2. train + evaluate (one A100 container per run, all in parallel)
conda run -n ML modal run ml/modal_app/train.py --models qwen3.5-0.8b,lfm2.5-350m,lfm2.5-1.2b,gemma-3-270m --fmts json,two_phase
#    smoke test: add --max-steps 30 --eval-limit 48 --train-limit 400

# 3. GGUF sweep for the winners (CPU containers, llama.cpp at kiln's commit)
conda run -n ML modal run ml/modal_app/gguf.py --runs qwen3.5-0.8b__json__lora,lfm2.5-350m__json__lora --push

# 4. report
conda run -n ML modal run ml/modal_app/report.py
```

GGUF files are reused if present; pass `--force` to rebuild (done automatically when the merged weights are newer than the F16 file).

Run artefacts live in the `hishab-runs` volume: `/runs/<model>__<fmt>__<tune>/{adapter,merged,gguf,eval_hf.json,eval_gguf/}`.
Download a GGUF with `modal volume get hishab-runs /runs/<run>/gguf/<file> .`.

## Metrics

Per example, after lenient parsing (code fences stripped, `JSON:` marker optional):

| metric | meaning |
|---|---|
| valid JSON | a JSON array was parsed |
| strict schema | every item has the 4 keys and an allowed category / type |
| **exact** | multiset of (amount, category, type) equals gold – the headline number |
| tuple F1 | micro P/R/F1 over (amount, category, type) across the split |
| category / type acc | on amount-aligned pairs |
| amounts match | same multiset of amounts (ignores labels) |
| empty acc | gold `[]` rows answered with `[]` |

Descriptions are not scored (free text). Everything is reported overall and per language.

## Extra training data without an LLM

`ml/data/generate_banglish.py` builds sentences from the hand-authored lexicon in
`ml/data/lexicon/banglish.py` (about 580 surface forms over 14 categories with realistic BDT
ranges, spelling variants, income/expense verbs, time words, connectors, number slang such as
`4.5k`, `12 hajar`, `১,২০০`, `2 lakh`, and spelled-out numbers such as `ponero hajar`,
`আড়াই হাজার`, `der lakh`, `pachsho`). Labels are exact by construction. It is weighted toward
the weak spots of the v3 model: Banglish rent / loan / gift / medical / utilities, Bangla
mobile_recharge / other, and income verbs. `ml/datasets/synthetic_handwritten_v1.jsonl` adds 155
free-form human-written messages (113 general, then 42 aimed at the error classes in
`ml/results/ERROR_ANALYSIS.md`: medicine after a doctor visit, data packs, loan direction,
family remittances, education side-costs, spelled-out numbers, totals). Both go to the
**train split only** (`--extra-train`), so the v3 validation / test sets stay identical and
numbers remain comparable. Runs trained on the enlarged data carry the `--tag v4` suffix
(5000 programmatic + 113 hand-written) or `--tag v5` (number words + 42 more hand-written rows).

## Numbers data: Codex templates + the numwords engine (v6)

The v5 model got every spelled-out Banglish amount wrong ("ponero hajar" -> 12000) because
only ~1% of training rows had a number without digits and the test set had none. The fix
splits the work: an LLM writes the sentences, a deterministic engine owns the numbers.

- `ml/hishab/numwords.py`: any integer -> Banglish words (spelling variants), Bangla script,
  Indian grouping, fractions (der / arai / sare), digit-word mixes ("15 hajar 500", "1.5 lakh",
  "80k"); plus `parse()` for the inverse. `ml/tests/test_numwords.py` round-trips thousands of
  amounts.
- `ml/data/generate_llm_templates.py --lang banglish|bangla|mixed|english`: asks Codex
  (`codex exec`, read-only, ephemeral) for messages with `{A1}` placeholders instead of
  amounts, each slot labelled with description / category / type / size, plus an English
  translation. Validated and deduped; appends to `ml/datasets/llm_templates_v1.jsonl`.
- `ml/data/fill_templates.py`: fills the placeholders from the engine, checks each rendering
  parses back to the labelled amount, writes extra-train rows with the translation in
  `metadata.translation` (build_splits.py now carries it into the split for `two_phase`).
  `--digit-free` makes word-only rows for an eval set.
- `ml/data/augment_numbers.py`: rewrites the digit amounts of existing Gemini rows as words,
  labels unchanged (`synthetic_numwords_aug_v1.jsonl`).
- `ml/datasets/probes_numbers.jsonl` + `ml/scripts/probe_gguf.py`: 20 number probes run
  through kiln-cli on a GGUF, the same path as the phone.

## Why v3 splits

`ryuma007/hishab-synthetic-v2`'s validation split is a subset of its train split, so the
published 42% "semantic match" was measured on training rows. v3 dedupes on a normalised
instruction and holds out 10% test / 5% validation stratified by language, source and
empty-output rows.
