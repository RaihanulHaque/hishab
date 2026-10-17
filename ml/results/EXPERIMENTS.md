# Hishab experiment log

Every training version of the shipping model line: Qwen3.5-0.8B with `json` output. `RESULTS.md`
has the other base models (LFM2.5, Gemma) and the `two_phase` runs. `RELABEL.md` explains how the
cleaned eval sets were made.

Last updated 2026-10-17 (night).

## Where things stand

- **In the app today:** v5 Q6_K (630 MB). It reads spelled-out amounts wrong most of the time (numbers 13.4, probes 6/20).
- **Best model:** v10. Ship **v10 Q4_K_M-imat** (529 MB): test_clean 93.1, numbers_clean 79.2, numbers 85.4, probes 18/20, the same as Q6_K at 100 MB less, and 34 tok/s on the Pixel 8 against Q6_K's 27. Q4_0-imat is about 6% faster still but 1.3 points lower on numbers_clean. The 3-bit files are not faster (see "Speed on the Pixel 8").
- **Biggest fix along the way:** v9 found that transformers 5 tokenized Bengali differently from the phone. Since v9 the phone file scores the same as the full-precision model.
- **Next:** relabel the income and gift categories of the Codex-template training rows, which is what `numbers_clean` is waiting on.

## Timeline

| version | what changed | headline result |
|---|---|---|
| v1-v2 | Gemini data, TRL notebooks | no trustworthy score (validation leaked into train) |
| v3 | clean splits, Modal pipeline, LoRA | test 87.6 |
| v4 | +5,000 programmatic Banglish rows | test 87.8; more data barely helped |
| v5 | +42 targeted hand-written rows | test 88.2, but numbers 13.4; **shipped** |
| v6 | number-word engine + Codex templates | numbers 13.4 to 81.8; test 85.7 |
| v7 | stated totals, typing noise, no-transaction rows | test 86.7, numbers 77.6; preempted at epoch 4 |
| v8 | full fine-tune instead of LoRA | test 89.6, numbers 84.9; phone file lost Bangla numbers |
| v9 | faithful tokenizer | phone file now equals full precision |
| v10 | relabelled Gemini training rows | test_clean 92.2 bf16, 93.1 on the phone file |

## How to read the numbers

| set | rows | what it tests |
|---|---|---|
| `validation` | 259 | used for picking the best epoch; reported but not the headline |
| `test` | 518 | held-out Gemini-generated messages in four languages. The gold labels are noisy (see ERROR_ANALYSIS.md), so about 90 is the ceiling |
| `numbers` | 357 | held-out digit-free messages where every amount is spelled out, like "der hajar" or "পনেরো হাজার". Added in v6, never trained on |
| `test_clean` | 476 | the `test` rows where Gemini 3.8 Flash and Sonnet 5.5, labelling from scratch with the guideline, agree; their label is the gold (72 rows differ from `test`). Headline from v10 on |
| `numbers_clean` | 317 | the same for `numbers` (52 rows differ, all category) |
| probes | 20 | short number phrases run through kiln-cli on the Q6_K file, which is exactly what the phone runs |

- **exact:** the set of (amount, category, type) for the message equals gold. One wrong field fails the whole message.
- **F1:** micro F1 over the individual transactions, so partly-right messages get partial credit.
- **count / amount / category / type:** accuracy on that field alone.
- **bf16 / HF:** the full-precision model on Modal with HF transformers, greedy.
- **GGUF:** llama.cpp at kiln's pinned commit, CPU, greedy. This is the phone path.

Test language mix: Banglish 219, Bangla 125, English 99, mixed 75.
Numbers language mix: Banglish 149, Bangla 89, mixed 74, English 45.

## Scoreboard

### On the cleaned labels (the headline from v10 on)

| version | test_clean bf16 | test_clean Q6_K | test_clean Q4_K_M-imat | numbers_clean bf16 | numbers_clean Q6_K | numbers_clean Q4_K_M-imat |
|---|---|---|---|---|---|---|
| v8 | 84.5 | | | 77.9 | | |
| v9 | 83.6 | | | **79.5** | | |
| v10 | **92.2** | **93.1** | **93.1** | 78.9 | 79.2 | 79.2 |

### On the original labels

v10 scores lower here by design: it was trained to the guideline, which the old `test` labels break
on about 14% of rows.

| version | train rows | test bf16 | test Q6_K | numbers bf16 | numbers Q6_K | probes Q6_K | note |
|---|---|---|---|---|---|---|---|
| v1-v2 | ~5,200 | n/a | n/a | n/a | n/a | n/a | leaky split |
| v3 | 4,409 | 87.6 | 85.7 | n/a | n/a | n/a | |
| v4 | 9,522 | 87.8 | 85.9 | n/a | n/a | n/a | |
| v5 | 9,564 | 88.2 | 86.5 | 13.4 | n/a | 6/20 | in the app |
| v6 | 18,582 | 85.7 | 84.4 | 81.8 | 73.9 | 14/20 | |
| v7 | 20,333 | 86.7 | 86.1 | 77.6 | 73.9 | 17/20 | epoch-4 weights |
| v8 | 20,333 | **89.6** | **89.2** | 84.9 | 80.1 | **18/20** | mis-tokenized training |
| v9 | 20,333 | 89.2 | **89.2** | 83.2 | 83.8 | 17/20 | Q4_K_M-imat 89.6 / 84.0 |
| v10 | 20,057 | 80.3 | 81.1 | **85.2** | **85.2** | **18/20** | relabelled training |

## Per version

### v1-v2: notebooks (Dec 2025 to Jan 2026)
- **Data:** 5,200 Gemini-generated rows (`synthetic_dataset.jsonl`).
- **Training:** TRL notebooks in `experiments/`. Published as `ryuma007/qwen3.5-0.8b-hishab-synthetic-v2`.
- **Result:** no trustworthy number, because the published validation split was a subset of train.
- **Lesson:** build clean splits before measuring anything.

### v3: clean pipeline baseline
- **Data:** the 5,200 rows deduped (14 near-duplicates removed), stratified into 4,409 / 259 / 518.
- **Training:** Modal, 4 base models x 2 output formats, LoRA, 5 epochs.
- **Result:** 87.6 bf16, 85.7 at Q6_K. Qwen beat LFM2.5-1.2B (82.4), Gemma-3-270M (81.7) and LFM2.5-350M (81.5).
- **Lessons:** `json` beat `two_phase` for Qwen, and every later version confirmed it. Below Q4 the scores fall fast.

### v4: programmatic Banglish
- **Data:** +5,000 rows from `ml/data/generate_banglish.py` (lexicon-driven, exact labels) and +113 hand-written rows.
- **Result:** 87.8 bf16. Doubling the data moved the test score by 0.2 points.
- **Lesson:** the remaining errors are mostly label disagreements in the Gemini gold, not model failures.

### v5: targeted fixes (in the app)
- **Data:** +42 hand-written rows for medicine after a doctor visit, data packs, loan direction and a few number words.
- **Result:** 88.2 bf16 and 86.5 at Q6_K, the best original-label test scores until v8. A desktop kiln check of six hard probes gave Q6_K 5/6 and Q4_K_M-imat 3/6, so the app ships Q6_K.
- **Problem found later:** spelled-out amounts fail. Numbers set 13.4, with 233 of 357 messages getting the amount wrong. Probes 6/20.

### v6: number words
- **Data:** nearly doubled to 18,582. Added `ml/hishab/numwords.py`, which converts number words to integers and back and is round-trip tested. Codex wrote 3,594 templates that were filled into 7,327 rows, and 1,691 rows had their digits rewritten as words.
- **Result:** numbers set 13.4 to **81.8** bf16, with only 1 amount error in 357. Probes 6/20 to 14/20. Test fell 2.5 points to 85.7.
- **Why test fell:** category errors rose from 43 to 52, mostly gift, salary and food swaps. The Codex rows follow `docs/LABELLING_GUIDELINE.md`, and the Gemini gold often does not.
- **Found:** the phone file lost Bangla numbers (see "Phone path vs full precision"; explained in v9).

### v7: totals, typing noise, negatives
- **Data:** 20,333 rows. +740 rows that state a total that must not become an entry, +303 no-transaction rows labelled `[]`, typing noise on 20% of Codex rows, and +1,447 Bangla-script word-amount rows.
- **Run:** Modal preempted the container at step 5,287 of 6,355 (83%). The relaunch found the epoch-4 checkpoint and resumed past the end, so v7 is epoch 4, not the best epoch 2. Its scores are a lower bound for this data.
- **Result:** test 86.7 bf16 and 86.1 at Q6_K, the best since v5. Category errors fell from 52 to 41, and Bangla on test rose from 84.0 to 89.6. Probes 17/20.
- **Regressions:** numbers fell to 77.6, with type flips up from 7 to 20. Messages containing "chhoy" (six) or "ashi" (eighty) come out as income. Count errors on test rose from 9 to 15: it drops the middle item of some 3-item messages, and one probe returned nothing.
- **Fix made:** `train.py` now saves checkpoints every epoch and resumes after a preemption.

### v8: full fine-tune
- **Data:** same as v7.
- **Training:** all weights instead of LoRA, learning rate 3e-5, 3 epochs, best epoch 2 by validation loss. 3,813 steps in 104 min with no preemption.
- **Result:** the best original-label scores of any version. Test 89.6, up 1.4 on v5 and 2.9 on v7. Numbers 84.9, up 3.1 on v6. Wrong test messages fell to 54, the fewest of any version.
- **What improved:** Bangla on test rose to 92.0 and mixed to 97.3. v7's income/expense flips are gone, with type errors on numbers back down from 20 to 7. Count errors on test went back from 15 to 9.
- **Phone format:** Q6_K scores 89.2 on test, only 0.4 below bf16, and 80.1 on numbers. That is 3.1 and 6.2 points above any earlier GGUF. Q4_K_M-imat, at 529 MB, scores 88.6 and 80.1.
- **Phone probes:** 18/20 on both Q6_K and Q4_K_M-imat. Q6_K's two misses are "bish hajar pachsho" read as 25000 and "সাড়ে তিন হাজার" split into two entries. Q4_K_M-imat's misses are 25000 again and 21000 read as 12000.
- **Cost:** per step, the full fine-tune (1.64 s) ran about as fast as LoRA v7 (1.77 s). LoRA saves little on this model because a third of its weights are the 248k-token output layer, which costs the same either way. v8 was cheaper than v7 overall because it ran 3 epochs instead of 5 and was not preempted.
- **Lesson:** with this much data, updating all weights beats LoRA by a wide margin, even though its validation loss looked worse after epoch 1.
- **Published 2026-10-17:** GGUFs (Q6_K, Q5_K_M, Q4_K_M-imat) at `ryuma007/hishab-qwen3.5-0.8b-json-full-v8-GGUF`, bf16 weights at `ryuma007/hishab-qwen3.5-0.8b-json-full-v8`, and the training data at `ryuma007/hishab-synthetic-v4` (train, validation, test and numbers splits).
- **Verdict at the time:** ship candidate, beating v5 on test (89.2 vs 86.5 at Q6_K) and on spelled-out numbers (80.1 at Q6_K, against v5's 13.4 even at full precision). Superseded by v10.

### v9: the tokenizer fix
- **Data and training:** identical to v8 (20,333 rows, full fine-tune, 3 epochs, best epoch 2, val loss 0.1123).
- **What changed:** `hfeval.load_tokenizer` loads the Rust backend straight from `tokenizer.json` instead of letting transformers 5 rebuild it.
- **The bug:** transformers 5.10.2's `Qwen2Tokenizer` ignores the `[\p{L}\p{M}]+` letter-run rule in Qwen3.5's tokenizer.json and splits every Bengali word at its combining marks (nukta, vowel signs). "এয়ারপোর্টের ... পাইলাম" is 36 tokens for transformers, 26 for the `tokenizers` library and 26 for llama.cpp. v3 to v8 were all trained on the 36-token version, while the phone always saw the 26-token version. English and Banglish are unaffected (identical ids), which is why the gap was Bangla-only.
- **How it was found:** `ml/modal_app/gguf_diag.py` showed the gap is identical across every server setting (slots, ubatch, flash attention, KV type, threads) and on llama.cpp master. `gguf_diag_logits.py` then fed llama-server the exact HF token ids: next-token probabilities match HF to three decimals at every position, so the forward pass was never the problem, while `/apply-template` matched HF and `/tokenize` did not. `tok_check.py` pinned it on transformers. An earlier note that "tokenization is identical" had compared llama.cpp with the `tokenizers` library, which agree with each other; the odd one out was transformers.
- **Result:** bf16 test 89.2, numbers 83.2, within noise of v8's 89.6 / 84.9. v8 never paid for the defect at bf16 because it was evaluated with the same tokenizer it trained on.
- **Phone format:** test Q6_K 89.2, Q5_K_M 89.4, Q4_K_M-imat 89.6; numbers Q6_K **83.8**, Q5_K_M 84.0, Q4_K_M-imat 84.0. Bangla numbers: bf16 84.3, Q6_K 84.3, Q4_K_M-imat 86.5 (v8 Q6_K was 70.8). The phone path and the HF path now agree.
- **Q4_K_M-imat is now the best quant, not the worst.** Across v3 to v8, 4-bit lost 1 to 4 points to Q6_K. In v9, Q4_K_M-imat is 0.4 ahead on test and 0.2 ahead on numbers, within noise of Q6_K and of bf16 (one test row is 0.19 points, one numbers row 0.28). Two reasons: the imatrix is computed on rendered training text, which is now tokenized the same way the quantized model sees at inference, so its importance weights land on the right rows; and the Bangla token sequences are 30 to 40% shorter, so there is less low-precision drift per message. The old 4-bit penalty was partly the tokenizer defect compounding with quantization noise.
- **Phone probes:** on the published Hub files, Q6_K 17/20 and Q4_K_M-imat 17/20, with the same three misses: "sare tin hajar" as 3000, "der lakh ... abbu theke" as salary instead of gift, and "bish hajar pachsho" (26000 at Q6_K, 20000 at Q4_K_M-imat). 17/20 against v8's 18/20 is one probe, inside the noise of a 20-item set, while the 357-row numbers split rose from 80.1 to 83.8.
- **Published:** `ryuma007/hishab-qwen3.5-0.8b-json-full-v9` and `...-v9-GGUF` (Q6_K, Q5_K_M, Q4_K_M-imat); sha256 of the Hub files verified against the volume.
- **Lesson:** verify token ids end to end (training tokenizer vs `tokenizers` vs llama.cpp) before trusting any GGUF number. `load_tokenizer` now asserts it on a Bangla probe at every run.

### v10: guideline-consistent Gemini rows
- **Data:** the 4,409 Gemini-written train rows (`source` base / hard) relabelled from scratch by Gemini 3.8 Flash (agy) and Sonnet 5.5 with `docs/LABELLING_GUIDELINE.md`, the same way as the test set (RELABEL.md). `ml/data/apply_relabel.py` keeps a row only where the two agree: 3,285 agree with the old gold, 848 (20.5%) take the new label, and 276 where the labellers disagree are dropped. The other 15,924 rows are unchanged. 20,057 rows in `splits/train_v10.jsonl`.
- **What the relabel changed:** 596 categories, 203 transaction counts (plans with no entry, stated totals, merged or split items), 9 types, 5 amounts. Top swaps: shopping to tuition 112 (books, photocopies, exam fees), tuition to salary 95 (tutoring income), other to tuition 66, other to salary 61, gift to shopping 48.
- **Training:** identical to v9 (full fine-tune, lr 3e-5, 3 epochs, best epoch 2, val loss 0.1228). The validation split still has the old labels, so its loss is not comparable.
- **Result on the clean gold:** test_clean **92.2** bf16, **93.1** at Q6_K and Q4_K_M-imat, up from v9's 83.6. Wrong messages on test_clean: 37 of 476. Bangla on test_clean 97.4, English 97.0. On the 72 test rows where both labellers overrule the old gold, v10 gives the guideline answer 73.6% of the time (v9: 19.4%) and the old gold 13.9% (v9: 62.5%).
- **Result on the old gold:** test falls to 80.3, with count errors up from 7 to 36. That is the intended effect: those rows are where the old gold records plans, totals or merged items that the guideline does not. Numbers (old template gold) rises from 83.2 to **85.2**, the best of any version, and English numbers from 84.4 to 91.1.
- **numbers_clean did not move (79.5 to 78.9).** Every one of its 52 overrules is a category on a Codex-template row, and the 8,775 template rows in train were not relabelled. On those 52 the model still follows the template convention (63.5%) over the labellers (11.5%). Typical cases: "tuition er betn pailam" (template `tuition`, guideline `salary`), "sold rice and recharge, got cash" (template grocery / mobile_recharge, guideline `salary`), a panjabi bought for Abbu (template `gift`, guideline `shopping`). So the template rows do not fully follow the guideline on income and gift categories, contrary to what RELABEL.md assumed.
- **Phone format:** every quant from Q8_0 to IQ3_M-imat is within a point of bf16 on both clean sets. Q4_K_M-imat ties Q6_K (93.1 / 79.2) at 529 MB. Q5_K_M on the old splits: test 81.3, numbers 86.0.
- **Phone probes:** 18/20 on both Q6_K and Q4_K_M-imat (sha256 verified against the volume), with category and type right on all 20. "der lakh ... abbu theke" is `gift` again. The two misses are the same amount readings as before: "sare tin hajar" as 3000 and "bish hajar pachsho" as 25000.
- **Probes on every quant (2026-10-17, Mac kiln-cli):** Q8_0, Q6_K, Q5_K_M, Q4_K_M-imat, IQ4_XS-imat and Q3_K_M-imat 18/20; Q4_0-imat 19/20 (only "bish hajar pachsho" as 25000); IQ3_M-imat 18/20 but with a different miss, "pach hajar 2 sho" as 4200, which every other quant gets right. Q5_K_M and IQ3_M read "bish hajar pachsho" as 22000 instead of 25000. Category and type are right on all 20 for every quant.
- **Speed on the Pixel 8:** Q4_K_M-imat 34 tok/s, Q4_0-imat 36, Q6_K 27, Q8_0 24.5; the 3-bit files are no faster and IQ3_M-imat is the slowest of all (19 tok/s, 1.6 s to read the prompt). Full table under "Speed on the Pixel 8".
- **Published in full (2026-10-17):** all 11 quants, F16 to IQ3_M-imat, are now in `...-v10-GGUF`, with eval JSONs for each.
- **Published 2026-10-17:** `ryuma007/hishab-qwen3.5-0.8b-json-full-v10` (bf16) and `...-v10-GGUF` (Q6_K, Q5_K_M, Q4_K_M-imat; Hub sha256 verified against the volume), trained on `ryuma007/hishab-synthetic-v5`, which holds the 20,057-row train split and all five eval splits.
- **Next:** relabel the income and gift categories of the template rows (or regenerate them) the same way; that is what numbers_clean is waiting on.

## Open problems

1. **Label noise.** Fixed for the Gemini rows in v10 (test_clean 92.2). Still open for the Codex-template rows: their income and gift categories disagree with the guideline (numbers_clean, see v10). The 42 + 40 held-out rows and 276 train rows where the two labellers disagree need a human (`splits_clean/disagreements.jsonl`).
2. ~~GGUF Bangla gap~~ **Solved in v9**: it was transformers 5 mis-tokenizing Bengali during training, not llama.cpp (see v9). Worth reporting upstream to transformers.
3. **Dropped entries.** v7 sometimes returned `[]` or dropped a middle item, and v8 GGUF occasionally writes broken JSON that the app drops. Worth a dedicated probe set, since silent misses are the worst failure for a ledger app.
4. **No real user messages.** Every eval set is synthetic, and people will type messier Banglish than this.

---

# Detail tables

## Training setup

Shared by every version: batch 8 x 2 accumulation; max 512 tokens; cosine decay with 5% warmup;
loss on the completion only; Qwen3.5 thinking turned off; one A100-80GB on Modal; the best epoch
by validation loss is kept.

- **v3 to v7, LoRA:** r=16, alpha 32, dropout 0.05 on attention and MLP layers, learning rate 2e-4.
- **v8 to v10, full fine-tune:** all weights, learning rate 3e-5.

| version | epochs run | best epoch | best val loss | val loss by epoch | final train loss | train time |
|---|---|---|---|---|---|---|
| v3 | 5 | 3 | 0.0948 | .1055 / .0951 / .0948 / .1067 / .1162 | 0.014 | 32 min |
| v4 | 4 | 3 | 0.0992 | .1102 / .1026 / .0992 / .1063 | 0.013 | 61 min |
| v5 | 4 | 3 | 0.0954 | .1138 / .0982 / .0954 / .1064 | 0.014 | 43 min |
| v6 | 5 | 2 | 0.1089 | .1153 / .1089 / .1137 / .1256 / .1410 | 0.002 | 103 min |
| v7 | 4 of 5 | 2 | 0.1076 | .1121 / .1076 / .1119 / .1231 | 0.007 | preempted, see v7 |
| v8 (full) | 3 | 2 | 0.1112 | .1254 / .1112 / .1213 | 0.007 | 104 min |
| v9 (full, faithful tokenizer) | 3 | 2 | 0.1123 | .1223 / .1123 / .1235 | 0.007 | 85 min |
| v10 (as v9, relabelled train) | 3 | 2 | 0.1228 | .1308 / .1228 / .1311 | | 68 min |

Validation loss bottoms out at epoch 2 or 3 every time, then rises while train loss keeps
falling. More than 3 epochs is overfitting on this data.

## Training data

Which dataset each model version was trained and evaluated on:

| model version | train rows | local train file | published dataset (HF) | eval splits in it |
|---|---|---|---|---|
| v1-v2 | 5,200 | notebook data | `ryuma007/hishab-synthetic-v2` (validation leaks into train) | none trustworthy |
| v3 | 4,409 | the `base` / `hard` rows of `splits/train.jsonl` | `ryuma007/hishab-synthetic-v3` | validation, test |
| v4 | 9,522 | not saved | not published (v3 + programmatic + hand-written) | |
| v5 | 9,564 | not saved | not published (v4 + 42 hand-written) | |
| v6 | 18,582 | not saved | not published | |
| v7, v8, v9 | 20,333 | `splits/train.jsonl` | `ryuma007/hishab-synthetic-v4` | validation, test, numbers |
| v10 | 20,057 | `splits/train_v10.jsonl` | `ryuma007/hishab-synthetic-v5` | validation, test, test_clean, numbers, numbers_clean |

"Not saved" sets were built by `ml/data/build_splits.py` from the `ml/datasets/synthetic_*.jsonl`
sources as they stood then. Validation, test and numbers are the same rows in every dataset
version from v3 (numbers from v4 on). Dataset v5 adds the relabelled `test_clean` and
`numbers_clean` subsets and keeps the originals.

Train rows by source:

| source | v3 | v4 | v5 | v6 | v7 (= v8, v9) | v10 |
|---|---|---|---|---|---|---|
| Gemini base rows | 4,240 | 4,240 | 4,240 | 4,240 | 4,240 | 3,976 relabelled |
| Gemini hard rows | 169 | 169 | 169 | 169 | 169 | 157 relabelled |
| programmatic lexicon | | 5,000 | 5,000 | 5,000 | 5,000 | 5,000 |
| hand-written | | 113 | 155 | 155 | 155 | 155 |
| Codex templates | | | | 7,327 | 7,328 (refilled) | 7,328 |
| Codex Bangla word-amounts | | | | | 1,447 | 1,447 |
| digits rewritten as words | | | | 1,691 | 1,691 | 1,691 |
| no-transaction rows (`[]`) | | | | | 303 | 303 |
| **total** | **4,409** | **9,522** | **9,564** | **18,582** | **20,333** | **20,057** |

Train rows by language:

| version | Banglish | Bangla | mixed | English |
|---|---|---|---|---|
| v3 | 1,863 | 1,062 | 636 | 848 |
| v5 | 5,032 | 2,273 | 1,400 | 859 |
| v6 | 9,204 | 4,271 | 3,346 | 1,761 |
| v7 | 9,337 | 5,111 | 4,065 | 1,820 |
| v10 | 9,161 | 5,063 | 4,031 | 1,802 |

Transactions per message in train:

| version | 0 | 1 | 2 | 3 | 4+ |
|---|---|---|---|---|---|
| v3 | 11 | 1,449 | 2,387 | 494 | 68 |
| v5 | 47 | 3,529 | 4,201 | 1,400 | 387 |
| v6 | 47 | 7,692 | 8,279 | 1,927 | 637 |
| v7 | 350 | 8,445 | 8,861 | 1,995 | 682 |
| v10 | 458 | 8,373 | 8,618 | 1,932 | 676 |

## Test set, bf16

| version | exact | F1 | count | amount | category | type | Banglish | Bangla | English | mixed |
|---|---|---|---|---|---|---|---|---|---|---|
| v3 | 87.6 | 92.9 | 98.6 | 97.7 | 95.0 | 98.9 | 81.7 | 90.4 | 93.9 | 92.0 |
| v4 | 87.8 | 93.4 | 98.3 | 97.3 | 95.3 | 99.1 | 82.6 | 88.0 | 94.9 | 93.3 |
| v5 | 88.2 | 93.4 | 98.3 | 97.7 | 94.8 | 99.4 | **83.6** | 88.0 | 93.9 | 94.7 |
| v6 | 85.7 | 92.3 | 98.3 | 97.7 | 93.9 | 98.9 | 80.4 | 84.0 | 93.9 | 93.3 |
| v7 | 86.7 | 93.0 | 97.1 | 96.3 | 95.3 | 98.9 | 79.0 | 89.6 | 93.9 | 94.7 |
| v8 | **89.6** | **94.4** | 98.3 | **98.1** | 95.3 | **99.6** | 83.1 | **92.0** | **94.9** | **97.3** |
| v9 | 89.2 | 93.9 | **98.6** | 97.7 | **95.8** | 99.0 | **83.6** | 91.2 | **94.9** | 94.7 |
| v10 | 80.3 | 89.3 | 93.1 | 93.1 | 92.0 | 99.2 | 73.1 | 87.2 | 79.8 | 90.7 |
| v10 on `test_clean` (476) | 92.2 | 95.7 | 97.7 | 97.5 | 96.8 | 99.5 | 86.4 | 97.4 | 97.0 | 93.1 |

Wrong messages by first failing field, out of 518:

| version | total wrong | count | amount | type | category | most common category swaps (gold to model) |
|---|---|---|---|---|---|---|
| v3 | 64 | 7 | 5 | 10 | 42 | salary to other 4, other to tuition 3 |
| v4 | 63 | 9 | 5 | 8 | 41 | tuition to other 4, salary to other 4 |
| v5 | 61 | 9 | 3 | 6 | 43 | tuition to other 5, food to shopping 3 |
| v6 | 74 | 9 | 3 | 10 | 52 | gift to other 6, salary to other 6, food to grocery 5 |
| v7 | 69 | 15 | 4 | 9 | 41 | salary to other 6, grocery to shopping 4 |
| v8 | **54** | 9 | 1 | 4 | 40 | entertainment to other 4, tuition to other 4 |
| v9 | 56 | 7 | 5 | 8 | 36 | tuition to other 5, salary to other 4, other to tuition 3 |
| v10 | 102 | 36 | 0 | 5 | 61 | shopping to tuition 8, grocery to shopping 5, tuition to salary 5 |
| v10 on `test_clean` (of 476) | 37 | 11 | 1 | 4 | 21 | other to food 2, shopping to gift 2, other to shopping 2 |

Category is two-thirds of the errors up to v9, and most of those swaps are cases where the Gemini
gold disagrees with itself (ERROR_ANALYSIS.md). That is why test exact sat near 86 to 88
regardless of data, until v10 fixed the labels.

## Numbers set, bf16

| version | exact | F1 | count | amount | category | type | Banglish | Bangla | English | mixed |
|---|---|---|---|---|---|---|---|---|---|---|
| v5 | 13.4 | 24.2 | 85.2 | 19.9 | 80.3 | 89.9 | 10.7 | 9.0 | 17.8 | 21.6 |
| v6 | 81.8 | 87.2 | **100.0** | **99.7** | 88.1 | 98.8 | 79.2 | 76.4 | 88.9 | 89.2 |
| v7 | 77.6 | 84.1 | 99.4 | 99.2 | 86.8 | 96.5 | 75.8 | 74.2 | 82.2 | 82.4 |
| v8 | 84.9 | 89.2 | 99.7 | 98.9 | **90.8** | 98.8 | **81.9** | **86.5** | 88.9 | 86.5 |
| v9 | 83.2 | 87.4 | **100.0** | **99.7** | 88.8 | 98.8 | 80.5 | 84.3 | 84.4 | 86.5 |
| v10 | **85.2** | **89.3** | 99.7 | 99.2 | 90.6 | **99.3** | **81.9** | 83.1 | **91.1** | **90.5** |
| v10 on `numbers_clean` (317) | 78.9 | 84.5 | 99.7 | 99.1 | 85.2 | 99.6 | 74.4 | 82.5 | 85.7 | 78.8 |

Wrong messages by first failing field, out of 357:

| version | total wrong | count | amount | type | category | most common category swaps |
|---|---|---|---|---|---|---|
| v5 | 309 | 53 | 233 | 13 | 10 | amounts wrong almost everywhere |
| v6 | 65 | 0 | 1 | 7 | 57 | gift to shopping 9, gift to other 8 |
| v7 | 80 | 2 | 1 | 20 | 57 | gift to shopping 8, other to shopping 6, other to tuition 6 |
| v8 | 54 | 1 | 3 | 7 | 43 | gift to shopping 8, other to shopping 7 |
| v9 | 60 | 0 | 1 | 5 | 54 | gift to shopping 9, other to transport 7, other to shopping 6 |
| v10 | **53** | 1 | 2 | 4 | 46 | gift to shopping 6, other to transport 4, other to shopping 3 |
| v10 on `numbers_clean` (of 317) | 67 | 1 | 2 | 2 | 62 | salary to other 8, tuition to other 5, salary to transport 4 |

From v6 on, the model reads spelled-out amounts almost perfectly at full precision. What is left
is category convention, plus v7's 20 income/expense flips.

## Phone files (GGUF quantizations)

File sizes: F16 1,517 MB, Q8_0 812, Q6_K 630, Q5_K_M 578, Q4_K_M 529, Q4_0 502, IQ4_XS 504,
Q3_K_M 466, IQ3_M 454. The "-e8" variants of v6, which keep embeddings and output at Q8_0, add
40 to 60 MB and score within a point of the plain quants.

Test set, exact:

| version | F16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M-imat | Q4_K_M | Q4_0-imat | IQ4_XS-imat | Q3_K_M-imat | IQ3_M-imat |
|---|---|---|---|---|---|---|---|---|---|---|
| v3 | 85.5 | 85.5 | 85.7 | 84.2 | 85.5 | 83.4 | 84.9 | 83.2 | 80.1 | 80.3 |
| v4 | 86.5 | 85.5 | 85.9 | 86.7 | 85.1 | 82.2 | 84.0 | | | 81.5 |
| v5 | 86.7 | 86.1 | 86.5 | 85.1 | 84.6 | | 84.0 | | | |
| v6 | | 84.7 | 84.4 | 84.6 | 83.6 | | | | | |
| v7 | | | 86.1 | 84.6 | 84.6 | | | | | |
| v8 | | | **89.2** | 88.2 | 88.6 | | | | | |
| v9 | 89.0 | **89.6** | **89.2** | **89.4** | **89.6** | | | | | |
| v10 | | | 81.1 | 81.3 | 81.1 | | | | | |
| v10 on `test_clean` | 92.2 | 92.2 | 93.1 | 92.6 | 93.1 | 92.9 | 93.7 | 92.9 | 93.5 | 92.6 |

Numbers set, exact:

| version | BF16 | F16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M-imat |
|---|---|---|---|---|---|---|
| v6 | 76.2 | 75.9 | 76.8 | 73.9 | 73.9 | 73.1 |
| v7 | | | | 73.9 | 72.3 | 74.5 |
| v8 | | | | 80.1 | 80.7 | 80.1 |
| v9 | 83.2 (HF) | 83.2 | 82.9 | 83.8 | 84.0 | 84.0 |
| v10 | 85.2 (HF) | | | **85.2** | **86.0** | **85.4** |
| v10 on `numbers_clean` | 78.9 (HF) | 78.9 | 78.9 | 79.2 | 79.8 | 79.2 |

v10 on `numbers_clean`, the rest of the ladder: Q4_K_M 78.9, Q4_0 78.2, Q4_0-imat 77.9,
IQ4_XS-imat 77.9, Q3_K_M-imat 77.3, IQ3_M-imat 79.5. Every quant down to IQ3_M stays within a
point of bf16 on both clean sets.

### Phone path vs full precision, numbers set by language

| version | path | Banglish | Bangla | English | mixed |
|---|---|---|---|---|---|
| v6 | bf16 on HF | 79.2 | **76.4** | 88.9 | 89.2 |
| v6 | BF16 GGUF | 78.5 | 59.6 | 88.9 | 83.8 |
| v6 | Q6_K GGUF | 78.5 | **52.8** | 86.7 | 82.4 |
| v7 | bf16 on HF | 75.8 | 74.2 | 82.2 | 82.4 |
| v7 | Q6_K GGUF | 75.8 | 56.2 | 84.4 | 85.1 |
| v8 | bf16 on HF | 81.9 | **86.5** | 88.9 | 86.5 |
| v8 | Q6_K GGUF | 81.2 | **70.8** | 88.9 | 83.8 |
| v9 | bf16 on HF | 80.5 | 84.3 | 84.4 | 86.5 |
| v9 | Q6_K GGUF | 81.2 | **84.3** | 84.4 | 87.8 |
| v9 | Q4_K_M-imat GGUF | 80.5 | **86.5** | 84.4 | 87.8 |
| v10 | bf16 on HF | 81.9 | 83.1 | 91.1 | 90.5 |
| v10 | Q6_K GGUF | 81.9 | 83.1 | 91.1 | 90.5 |
| v10 | Q4_K_M-imat GGUF | 81.9 | 84.3 | 91.1 | 90.5 |

Through v8, Banglish was the same on both paths while Bangla dropped about 17 points even in
BF16 GGUF, which has the same weights as HF. The cause was the training tokenizer, not llama.cpp
(see v9). With the faithful tokenizer, v9's Bangla matches bf16 exactly at Q6_K and beats it at
Q4_K_M-imat.

### A GGUF-only output glitch

v8 Q6_K wrote invalid JSON on 3 of 357 numbers rows (v9: Q6_K 0, Q4_K_M-imat 1). In each one the
description contains a number word and the output jumps straight into digits, skipping the
`","amount":` part, for example `"description":"তেল বিক্রির চার000.0,`. The bf16 model never does
this. The app's lenient parser drops such entries, so the user would see a missing line.

## Speed on the Pixel 8 (v10)

kiln-bench on the Pixel 8 (Tensor G3), 2026-10-17, kiln main at 6b5d57b, android-arm64 CPU build.
Each format ran 3 times in each of two passes, the second pass in reverse order, waiting for full
CPU clocks and 20 s of rest before each format. The prompt is kiln-bench's ~65-token chat prompt;
the model stops on its own, so answer length differs by format (26 to 81 tokens). Q4_0-imat and
Q4_K_M-imat got 6 more runs each, alternating, because one Q4_0-imat run was disturbed (15.7 tok/s).

| format | file | generation tok/s, median (range) | prompt read, median | peak RAM | runs |
|---|---|---|---|---|---|
| Q8_0 | 812 MB | 24.5 (21.9 to 25.7) | 389 ms | 1,775 MB | 6 |
| Q6_K | 630 MB | 26.9 (23.5 to 30.4) | 542 ms | 1,411 MB | 6 |
| Q5_K_M | 578 MB | 29.3 (26.5 to 33.1) | 552 ms | 1,307 MB | 6 |
| **Q4_K_M-imat** | **529 MB** | **34.0** (30.7 to 36.7) | 460 ms | 1,210 MB | 12 |
| Q4_0-imat | 502 MB | **35.9** (15.7 to 40.1) | 409 ms | 1,148 MB | 12 |
| IQ4_XS-imat | 504 MB | 29.1 (27.6 to 32.7) | 789 ms | 957 MB | 6 |
| Q3_K_M-imat | 466 MB | 30.4 (27.5 to 33.6) | 723 ms | 964 MB | 6 |
| IQ3_M-imat | 454 MB | 19.4 (18.1 to 21.2) | 1,646 ms | 906 MB | 6 |

In the 6 alternating runs, Q4_0-imat ran 35.2 to 40.1 tok/s and Q4_K_M-imat 33.4 to 36.6.

What it says:
- **Down to 4 bits, smaller is faster,** because generation is limited by reading the weights from memory.
- **Below 4 bits it stops paying.** IQ4_XS and Q3_K_M are no faster than Q4_K_M and take 0.7 to 0.8 s to read the prompt instead of 0.46 s. IQ3_M is the slowest file of all: its codebook lookups cost more CPU than its 75 fewer MB save.
- **Q4_0-imat is the fastest,** about 6% ahead of Q4_K_M-imat (9% in the alternating runs), as kiln's QUANT.md found for LFM2.5. For a typical 45-token Hishab answer that is about 0.1 s per message. It scores 93.7 on test_clean but 77.9 on numbers_clean against Q4_K_M-imat's 79.2, so Q4_K_M-imat stays the pick.
- **RAM:** the K and I quants below 5 bits peak at about 0.9 to 1.2 GB; Q8_0 needs 1.8 GB.

## Phone probes, Q6_K through kiln-cli

v5 to v7 were rerun together on 2026-10-17, and v8 later the same day, with the same kiln-cli binary. Earlier runs recorded
v5 4/20 and v6 16/20; greedy decoding on CPU can flip a borderline token between builds and
thread counts. v9 and v10 were run on the published Hub files.

| probe | want | v5 | v6 | v7 | v8 | v9 Q6_K / Q4_K_M-imat | v10 (both) |
|---|---|---|---|---|---|---|---|
| ponero hajar taka salary pelam | 15000 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| sare tin hajar takar bazar korlam | 3500 | 3000 | 3000 | ✓ | ✓ | 3000 | 3000 |
| rickshaw vara ek hajar pachsho | 1500 | 1000 | ✓ | ✓ | ✓ | ✓ | ✓ |
| dui lakh bish hajar taka loan nilam | 220000 | 20000 | 210000 | ✓ | ✓ | ✓ | ✓ |
| der lakh taka pelam abbu theke | 150000 | ✓ | ✓ | ✓ | ✓ | 150000 salary | ✓ |
| cha khelam chollish taka | 40 | 400 | ✓ | ✓ | ✓ | ✓ | ✓ |
| lunch tin sho ponchash taka | 350 | 300 | ✓ | ✓ | ✓ | ✓ | ✓ |
| paytallish hajar taka salary | 45000 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| eksho bish taka recharge | 120 | 100 | ✓ | ✓ | ✓ | ✓ | ✓ |
| bish hajar pachsho takar phone kinlam | 20500 | 2000 | 25000 | 25000 | 25000 | 26000 / 20000 | 25000 |
| sat sho taka oushudh | 700 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| ekush hajar taka vara dilam | 21000 rent | 10000 | 21000 other | **no entry** | ✓ | ✓ | ✓ |
| 15 hajar 500 taka tuition fee | 15500 | 500 | ✓ | ✓ | ✓ | ✓ | ✓ |
| pach hajar 2 sho taka bazar | 5200 | 2000 | ✓ | ✓ | ✓ | ✓ | ✓ |
| পনেরো হাজার টাকা বেতন পেলাম | 15000 | 11000 | ✓ | ✓ | ✓ | ✓ | ✓ |
| আড়াই হাজার টাকার ওষুধ | 2500 | 18000 | 1500 | ✓ | ✓ | ✓ | ✓ |
| ১৫ হাজার ৫০০ টাকা বাসা ভাড়া | 15500 | 500 | 1500 | 15000 | ✓ | ✓ | ✓ |
| সাড়ে তিন হাজার টাকা বাজার | 3500 | 3000 | ✓ | ✓ | split in 2 | ✓ | ✓ |
| dersho taka cha nasta | 150 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| salary pelam 80k | 80000 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **exact** | | **6/20** | **14/20** | **17/20** | **18/20** | **17/20** (both) | **18/20** |

v7's miss on "ekush hajar taka vara dilam" returns nothing at all. That is a new failure type,
probably from the no-transaction rows added in v7, and it is worse for a user than a wrong
category because the entry silently disappears.
