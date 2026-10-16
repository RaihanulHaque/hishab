# Hishab experiment log

One row per training version, newest last. Everything is Qwen3.5-0.8B, `json` output, LoRA r=16
unless noted. Full per-run tables (all models, all quants) are in `RESULTS.md`; this file is the
story: what data each version had, what it scored, and what we learned.

## Eval sets

| set | rows | what it tests |
|---|---|---|
| `test` | 518 | held-out Gemini-generated messages, all four languages. Gold labels are noisy (see ERROR_ANALYSIS.md): ceiling is about 90 |
| `numbers` | 357 | held-out digit-free messages where every amount is spelled out ("der hajar", "পনেরো হাজার"). Added in v6; never trained on |
| probes | 20 | short Banglish/Bangla number phrases run on the Q6_K GGUF through kiln-cli, the exact phone path |

exact = the multiset of (amount, category, type) equals gold. bf16 = HF transformers on Modal.
GGUF = llama.cpp at kiln's pinned commit, greedy.

## Scoreboard

| version | train rows | test bf16 | test Q6_K | numbers bf16 | numbers Q6_K | probes Q6_K | status |
|---|---|---|---|---|---|---|---|
| v1-v2 | ~5,200 | n/a | n/a | n/a | n/a | n/a | notebook era; validation leaked into train, numbers not comparable |
| v3 | 4,409 | 87.6 | 85.7 | n/a | n/a | n/a | first clean Modal pipeline |
| v4 | 9,522 | 87.8 | 85.9 | n/a | n/a | n/a | +5,000 programmatic rows |
| v5 | 9,564 | **88.2** | **86.5** | 13.4 | n/a | 4/20 | **shipped in hishab_offline** |
| v6 | 18,582 | 85.7 | n/a | **81.8** | 75.9 | **16/20** | number words fixed; best candidate so far |
| v7 | 20,333 | 86.7 | running | 77.6 | running | pending | epoch-4 weights after a preemption |
| v8 | 20,333 | running | | | | | full fine-tune, 3 epochs |

## Per version

### v1-v2: notebooks (Dec 2025 to Jan 2026)
- **Data:** 5,200 Gemini-generated rows (`synthetic_dataset.jsonl`).
- **Training:** TRL notebooks in `experiments/`. Published as `ryuma007/qwen3.5-0.8b-hishab-synthetic-v2`.
- **Result:** no trustworthy number. The published validation split was a subset of train.
- **Lesson:** build clean splits before measuring anything.

### v3: clean pipeline baseline
- **Data:** the 5,200 rows deduped (14 near-duplicates removed), stratified 4,409 / 259 / 518 split.
- **Training:** Modal, 4 models x 2 formats (`json`, `two_phase`), LoRA, 5 epochs.
- **Result:** Qwen json 87.6 bf16, 85.5 at Q4_K_M-imat. Qwen beat LFM2.5-350M (81.5), LFM2.5-1.2B (82.4) and Gemma-3-270M (81.7).
- **Lesson:** `json` beat `two_phase` for Qwen, and every later version confirmed it. Quantization below Q4 hurts a lot.

### v4: programmatic Banglish
- **Data:** +5,000 rows from `ml/data/generate_banglish.py` (lexicon-driven, exact labels) and +113 hand-written rows, train only.
- **Result:** 87.8 bf16. Doubling the data moved the test score by 0.2 points.
- **Lesson:** the remaining errors are mostly label disagreements in the Gemini gold, not model failures. See ERROR_ANALYSIS.md.

### v5: targeted fixes (shipped)
- **Data:** +42 hand-written rows for medicine after a doctor visit, data packs, loan direction and a few number words.
- **Result:** 88.2 bf16, 86.5 at Q6_K. Kiln desktop check: Q6_K 5/6 hard probes, Q4_K_M-imat 3/6, so the app ships Q6_K (630 MB).
- **Problem found later:** spelled-out amounts fail. Numbers slice 13.4%, only 71 of 357 amounts right, probes 4/20.

### v6: number words
- **Data:** 18,582 rows. Added `ml/hishab/numwords.py` (number words to int and back, round-trip tested), 3,594 Codex-written templates filled into 7,327 rows, and 1,691 rows where digits were rewritten as words.
- **Result:** numbers slice 13.4 to **81.8**, with 356 of 357 amounts right. Probes 16/20 at Q6_K. Test dropped to 85.7.
- **GGUF gap:** every GGUF precision scores about 76 on numbers, even F16 and BF16: F16 75.9, BF16 76.2, Q8_0 76.8, Q6_K 75.9, Q5_K_M 73.9, Q4_K_M-imat 73.1. The loss is in Bangla-script compounds dropping a zero ("নয় হাজার পাঁচশো" read as 950), so it is llama.cpp's Qwen3.5 inference path, not quantization. NFC normalisation was added to the app and the eval anyway. Banglish is unaffected at 147 of 149.
- **Lesson:** the test drop is mostly convention shift. Codex rows follow `docs/LABELLING_GUIDELINE.md`, which the Gemini gold does not always follow.

### v7: totals, noise, negatives
- **Data:** 20,333 rows. +740 rows that state a total that must not become an entry, +350 no-transaction rows labelled `[]`, typing noise on 20% of Codex rows, +1,447 Bangla-script word-amount rows.
- **Run:** Modal preempted the container at 83% of 5 epochs. The relaunch resumed from the saved epoch-4 checkpoint and stopped, so v7 is epoch 4, not the best epoch (epoch 2 had the lowest validation loss in both v6 and v7).
- **Result:** test 86.7, the best since v5. Category errors fell from 55 to 41. Numbers fell to 77.6: amounts are still 354 of 357, but rows with "chhoy" (six) or "ashi" (eighty) flip to income, a sign of overfitting. Count errors rose from 9 to 15.
- **Fix made:** `train.py` now commits checkpoints each epoch and resumes after a preemption.

### v8: full fine-tune (running)
- **Data:** same as v7.
- **Training:** all weights, 3 epochs, best epoch by validation loss.
- **Goal:** keep v7's test gain without the type flips. If it is not clearly better than v6 on both test and numbers, the next step is a clean LoRA v7 with 3 epochs, which needs your approval first.

## Open problems
1. **Label noise in `test`.** 128 disagreement rows need judging against the guideline. The Codex judge stalled on quota, so this is still open.
2. **GGUF gap on Bangla-script numbers.** About 6 points lost in llama.cpp at every precision. This is a kiln or llama.cpp issue, not a training one.
3. **No real user messages.** Every eval set is synthetic, and people will type messier Banglish than this.
