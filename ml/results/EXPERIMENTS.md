# Hishab experiment log

Every training version of the shipping model line: Qwen3.5-0.8B, `json` output. For each version
this file records the data, the training setup, every score we have, the error breakdown and
what we learned. `RESULTS.md` has the other base models (LFM2.5, Gemma) and `two_phase` runs.

Last updated 2026-10-17 (evening). v9 closes the GGUF Bangla gap.

## How to read the numbers

| set | rows | what it tests |
|---|---|---|
| `validation` | 259 | used for picking the best epoch; reported but not the headline |
| `test` | 518 | held-out Gemini-generated messages in four languages. The gold labels are noisy (see ERROR_ANALYSIS.md), so about 90 is the ceiling |
| `numbers` | 357 | held-out digit-free messages where every amount is spelled out, like "der hajar" or "পনেরো হাজার". Added in v6, never trained on |
| probes | 20 | short number phrases run through kiln-cli on the Q6_K file, which is exactly what the phone runs |

- **exact:** the set of (amount, category, type) for the message equals gold. One wrong field fails the whole message.
- **F1:** micro F1 over the individual transactions, so partly-right messages get partial credit.
- **count / amount / category / type:** accuracy on that field alone.
- **bf16:** the full-precision model on Modal with HF transformers, greedy.
- **GGUF:** llama.cpp at kiln's pinned commit, CPU, greedy. This is the phone path.

Test language mix: Banglish 219, Bangla 125, English 99, mixed 75.
Numbers language mix: Banglish 149, Bangla 89, mixed 74, English 45.

## Scoreboard

| version | train rows | test bf16 | test Q6_K | numbers bf16 | numbers Q6_K | probes Q6_K | status |
|---|---|---|---|---|---|---|---|
| v1-v2 | ~5,200 | n/a | n/a | n/a | n/a | n/a | notebook era; leaky split, no trustworthy score |
| v3 | 4,409 | 87.6 | 85.7 | n/a | n/a | n/a | first clean pipeline |
| v4 | 9,522 | 87.8 | 85.9 | n/a | n/a | n/a | +5,000 programmatic rows |
| v5 | 9,564 | **88.2** | **86.5** | 13.4 | n/a | 6/20 | **shipped in hishab_offline** |
| v6 | 18,582 | 85.7 | 84.4 | **81.8** | 73.9 | 14/20 | number words learned |
| v7 | 20,333 | 86.7 | 86.1 | 77.6 | 73.9 | **17/20** | epoch-4 weights after a preemption |
| v8 | 20,333 | **89.6** | 89.2 | **84.9** | 80.1 | **18/20** | full fine-tune; trained with the mis-tokenizing transformers-5 tokenizer |
| v9 | 20,333 | 89.2 | **89.2** | 83.2 | **83.8** | 17/20 | **v8 recipe with the faithful tokenizer; GGUF now equals bf16; Q4_K_M-imat 89.6 / 84.0, ship candidate** |

The probe scores are one rerun of all three files on 2026-10-17 with the same kiln-cli binary.
Earlier runs recorded v5 4/20 and v6 16/20; greedy decoding on CPU can flip a borderline token
between builds and thread counts.

## Training setup

Shared by every version: LoRA r=16, alpha 32, dropout 0.05 on attention and MLP layers; learning
rate 2e-4 with cosine decay and 5% warmup; batch 8 x 2 accumulation; max 512 tokens; loss on the
completion only; Qwen3.5 thinking turned off. Training runs on one A100-80GB on Modal. The best
epoch by validation loss is kept.

| version | epochs run | best epoch | best val loss | val loss by epoch | final train loss | train time |
|---|---|---|---|---|---|---|
| v3 | 5 | 3 | 0.0948 | .1055 / .0951 / .0948 / .1067 / .1162 | 0.014 | 32 min |
| v4 | 4 | 3 | 0.0992 | .1102 / .1026 / .0992 / .1063 | 0.013 | 61 min |
| v5 | 4 | 3 | 0.0954 | .1138 / .0982 / .0954 / .1064 | 0.014 | 43 min |
| v6 | 5 | 2 | 0.1089 | .1153 / .1089 / .1137 / .1256 / .1410 | 0.002 | 103 min |
| v7 | 4 of 5 | 2 | 0.1076 | .1121 / .1076 / .1119 / .1231 | 0.007 | preempted, see v7 |
| v8 (full, lr 3e-5) | 3 | 2 | 0.1112 | .1254 / .1112 / .1213 | 0.007 | 104 min |
| v9 (full, lr 3e-5, faithful tokenizer) | 3 | 2 | 0.1123 | .1223 / .1123 / .1235 | 0.007 | 85 min |

Validation loss bottoms out at epoch 2 or 3 every time, then rises while train loss keeps
falling. More than 3 epochs is overfitting on this data.

## Training data by version

| source | v3 | v4 | v5 | v6 | v7 |
|---|---|---|---|---|---|
| Gemini base rows | 4,240 | 4,240 | 4,240 | 4,240 | 4,240 |
| Gemini hard rows | 169 | 169 | 169 | 169 | 169 |
| programmatic lexicon | | 5,000 | 5,000 | 5,000 | 5,000 |
| hand-written | | 113 | 155 | 155 | 155 |
| Codex templates | | | | 7,327 | 7,328 (refilled) |
| Codex Bangla word-amounts | | | | | 1,447 |
| digits rewritten as words | | | | 1,691 | 1,691 |
| no-transaction rows (`[]`) | | | | | 303 |
| **total** | **4,409** | **9,522** | **9,564** | **18,582** | **20,333** |

Train rows by language:

| version | Banglish | Bangla | mixed | English |
|---|---|---|---|---|
| v3 | 1,863 | 1,062 | 636 | 848 |
| v5 | 5,032 | 2,273 | 1,400 | 859 |
| v6 | 9,204 | 4,271 | 3,346 | 1,761 |
| v7 | 9,337 | 5,111 | 4,065 | 1,820 |

Transactions per message in train:

| version | 0 | 1 | 2 | 3 | 4+ |
|---|---|---|---|---|---|
| v3 | 11 | 1,449 | 2,387 | 494 | 68 |
| v5 | 47 | 3,529 | 4,201 | 1,400 | 387 |
| v6 | 47 | 7,692 | 8,279 | 1,927 | 637 |
| v7 | 350 | 8,445 | 8,861 | 1,995 | 682 |

## Test set, bf16, in detail

| version | exact | F1 | count | amount | category | type | Banglish | Bangla | English | mixed |
|---|---|---|---|---|---|---|---|---|---|---|
| v3 | 87.6 | 92.9 | 98.6 | 97.7 | 95.0 | 98.9 | 81.7 | 90.4 | 93.9 | 92.0 |
| v4 | 87.8 | 93.4 | 98.3 | 97.3 | 95.3 | 99.1 | 82.6 | 88.0 | 94.9 | 93.3 |
| v5 | **88.2** | **93.4** | 98.3 | 97.7 | 94.8 | **99.4** | **83.6** | 88.0 | 93.9 | **94.7** |
| v6 | 85.7 | 92.3 | 98.3 | 97.7 | 93.9 | 98.9 | 80.4 | 84.0 | 93.9 | 93.3 |
| v7 | 86.7 | 93.0 | 97.1 | 96.3 | 95.3 | 98.9 | 79.0 | 89.6 | 93.9 | 94.7 |
| v8 | **89.6** | **94.4** | 98.3 | **98.1** | **95.3** | **99.6** | 83.1 | **92.0** | **94.9** | **97.3** |
| v9 | 89.2 | 93.9 | **98.6** | 97.7 | **95.8** | 99.0 | **83.6** | 91.2 | **94.9** | 94.7 |

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

Category is two-thirds of every version's errors, and most of those swaps are cases where the
Gemini gold disagrees with itself (ERROR_ANALYSIS.md). That is why test exact sits near 86 to 88
regardless of data.

## Numbers set, bf16, in detail

| version | exact | F1 | count | amount | category | type | Banglish | Bangla | English | mixed |
|---|---|---|---|---|---|---|---|---|---|---|
| v5 | 13.4 | 24.2 | 85.2 | 19.9 | 80.3 | 89.9 | 10.7 | 9.0 | 17.8 | 21.6 |
| v6 | 81.8 | 87.2 | **100.0** | **99.7** | 88.1 | **98.8** | 79.2 | 76.4 | **88.9** | **89.2** |
| v7 | 77.6 | 84.1 | 99.4 | 99.2 | 86.8 | 96.5 | 75.8 | 74.2 | 82.2 | 82.4 |
| v8 | **84.9** | **89.2** | 99.7 | 98.9 | **90.8** | **98.8** | **81.9** | **86.5** | **88.9** | 86.5 |
| v9 | 83.2 | 87.4 | **100.0** | **99.7** | 88.8 | **98.8** | 80.5 | 84.3 | 84.4 | **86.5** |

Wrong messages by first failing field, out of 357:

| version | total wrong | count | amount | type | category | most common category swaps |
|---|---|---|---|---|---|---|
| v5 | 309 | 53 | 233 | 13 | 10 | amounts wrong almost everywhere |
| v6 | 65 | 0 | 1 | 7 | 57 | gift to shopping 9, gift to other 8 |
| v7 | 80 | 2 | 1 | 20 | 57 | gift to shopping 8, other to shopping 6, other to tuition 6 |
| v8 | **54** | 1 | 3 | 7 | 43 | gift to shopping 8, other to shopping 7 |
| v9 | 60 | 0 | 1 | 5 | 54 | gift to shopping 9, other to transport 7, other to shopping 6 |

From v6 on, the model reads spelled-out amounts almost perfectly at full precision. What is left
is category convention, plus v7's 20 income/expense flips.

## GGUF quantizations

Test set:

| version | F16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M-imat | Q4_K_M | Q4_0-imat | IQ4_XS-imat | Q3_K_M-imat | IQ3_M-imat |
|---|---|---|---|---|---|---|---|---|---|---|
| v3 | 85.5 | 85.5 | **85.7** | 84.2 | 85.5 | 83.4 | 84.9 | 83.2 | 80.1 | 80.3 |
| v4 | 86.5 | 85.5 | 85.9 | **86.7** | 85.1 | 82.2 | 84.0 | | | 81.5 |
| v5 | **86.7** | 86.1 | 86.5 | 85.1 | 84.6 | | 84.0 | | | |
| v6 | | 84.7 | 84.4 | 84.6 | 83.6 | | | | | |
| v7 | | | 86.1 | 84.6 | 84.6 | | | | | |
| v8 | | | **89.2** | **88.2** | **88.6** | | | | | |
| v9 | 89.0 | **89.6** | **89.2** | **89.4** | **89.6** | | | | | |

Numbers set:

| version | BF16 | F16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M-imat |
|---|---|---|---|---|---|---|
| v6 | 76.2 | 75.9 | **76.8** | 73.9 | 73.9 | 73.1 |
| v7 | | | | 73.9 | 72.3 | 74.5 |
| v8 | | | | **80.1** | **80.7** | **80.1** |
| v9 | 83.2 (HF) | 83.2 | 82.9 | **83.8** | **84.0** | **84.0** |

File sizes: F16 1,517 MB, Q8_0 812, Q6_K 630, Q5_K_M 578, Q4_K_M 529, Q4_0 502, IQ4_XS 504,
Q3_K_M 466, IQ3_M 454. The "-e8" variants of v6, which keep embeddings and output at Q8_0, add
40 to 60 MB and score within a point of the plain quants.

**The phone path loses Bangla-script numbers.** Same model, numbers set, by language:

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

Through v8, Banglish was the same on both paths while Bangla dropped about 17 points even in
BF16 GGUF, which has the same weights as HF. The cause was not llama.cpp: transformers 5.10.2's
`Qwen2Tokenizer` ignores the `[\p{L}\p{M}]+` rule in Qwen3.5's tokenizer.json and splits Bengali
words at their combining marks, so v3..v8 trained on a tokenization the phone never produces
(see v9). With the faithful tokenizer, v9's GGUF Bangla matches bf16 exactly at Q6_K and beats
it at Q4_K_M-imat. The earlier note that "tokenization is identical" compared llama.cpp with the
`tokenizers` library, which agree with each other; the odd one out was transformers.

**A GGUF-only output glitch.** v8 Q6_K wrote invalid JSON on 3 of 357 numbers rows (v9: Q6_K 0, Q4_K_M-imat 1). In each one
the description contains a number word and the output jumps straight into digits, skipping the
`","amount":` part, for example `"description":"তেল বিক্রির চার000.0,`. The bf16 model never
does this. The app's lenient parser drops such entries, so the user would see a missing line.

## Phone probes, Q6_K through kiln-cli

| probe | want | v5 | v6 | v7 | v8 | v9 Q6_K / Q4_K_M-imat |
|---|---|---|---|---|---|---|
| ponero hajar taka salary pelam | 15000 | ✓ | ✓ | ✓ | ✓ | ✓ |
| sare tin hajar takar bazar korlam | 3500 | 3000 | 3000 | ✓ | ✓ | 3000 |
| rickshaw vara ek hajar pachsho | 1500 | 1000 | ✓ | ✓ | ✓ | ✓ |
| dui lakh bish hajar taka loan nilam | 220000 | 20000 | 210000 | ✓ | ✓ | ✓ |
| der lakh taka pelam abbu theke | 150000 | ✓ | ✓ | ✓ | ✓ | 150000 salary |
| cha khelam chollish taka | 40 | 400 | ✓ | ✓ | ✓ | ✓ |
| lunch tin sho ponchash taka | 350 | 300 | ✓ | ✓ | ✓ | ✓ |
| paytallish hajar taka salary | 45000 | ✓ | ✓ | ✓ | ✓ | ✓ |
| eksho bish taka recharge | 120 | 100 | ✓ | ✓ | ✓ | ✓ |
| bish hajar pachsho takar phone kinlam | 20500 | 2000 | 25000 | 25000 | 25000 | 26000 / 20000 |
| sat sho taka oushudh | 700 | ✓ | ✓ | ✓ | ✓ | ✓ |
| ekush hajar taka vara dilam | 21000 rent | 10000 | 21000 other | **no entry** | ✓ | ✓ |
| 15 hajar 500 taka tuition fee | 15500 | 500 | ✓ | ✓ | ✓ | ✓ |
| pach hajar 2 sho taka bazar | 5200 | 2000 | ✓ | ✓ | ✓ | ✓ |
| পনেরো হাজার টাকা বেতন পেলাম | 15000 | 11000 | ✓ | ✓ | ✓ | ✓ |
| আড়াই হাজার টাকার ওষুধ | 2500 | 18000 | 1500 | ✓ | ✓ | ✓ |
| ১৫ হাজার ৫০০ টাকা বাসা ভাড়া | 15500 | 500 | 1500 | 15000 | ✓ | ✓ |
| সাড়ে তিন হাজার টাকা বাজার | 3500 | 3000 | ✓ | ✓ | split in 2 | ✓ |
| dersho taka cha nasta | 150 | ✓ | ✓ | ✓ | ✓ | ✓ |
| salary pelam 80k | 80000 | ✓ | ✓ | ✓ | ✓ | ✓ |
| **exact** | | **6/20** | **14/20** | **17/20** | **18/20** | **17/20** (both) |

v9 (both quants, on the published Hub files) misses the same three: "sare tin hajar" as 3000, "der lakh ... abbu theke" as salary instead of gift, and "bish hajar pachsho" (26000 at Q6_K, 20000 at Q4_K_M-imat); 17/20 against v8's 18/20 is one probe, inside the noise of a 20-item set, while the 357-row numbers split rose 80.1 to 83.8.

v7's miss on "ekush hajar taka vara dilam" returns nothing at all. That is a new failure type,
probably from the no-transaction rows added in v7, and it is worse for a user than a wrong
category because the entry silently disappears.

## Per version

### v1-v2: notebooks (Dec 2025 to Jan 2026)
- **Data:** 5,200 Gemini-generated rows (`synthetic_dataset.jsonl`).
- **Training:** TRL notebooks in `experiments/`. Published as `ryuma007/qwen3.5-0.8b-hishab-synthetic-v2`.
- **Result:** no trustworthy number, because the published validation split was a subset of train.
- **Lesson:** build clean splits before measuring anything.

### v3: clean pipeline baseline
- **Data:** the 5,200 rows deduped (14 near-duplicates removed), stratified into 4,409 / 259 / 518.
- **Training:** Modal, 4 base models x 2 output formats, 5 epochs.
- **Result:** 87.6 bf16, 85.7 at Q6_K. Qwen beat LFM2.5-1.2B (82.4), Gemma-3-270M (81.7) and LFM2.5-350M (81.5).
- **Lessons:** `json` beat `two_phase` for Qwen, and every later version confirmed it. Below Q4 the scores fall fast.

### v4: programmatic Banglish
- **Data:** +5,000 rows from `ml/data/generate_banglish.py` (lexicon-driven, exact labels) and +113 hand-written rows.
- **Result:** 87.8 bf16. Doubling the data moved the test score by 0.2 points.
- **Lesson:** the remaining errors are mostly label disagreements in the Gemini gold, not model failures.

### v5: targeted fixes, shipped
- **Data:** +42 hand-written rows for medicine after a doctor visit, data packs, loan direction and a few number words.
- **Result:** 88.2 bf16 and 86.5 at Q6_K, the best test scores of any version. A desktop kiln check of six hard probes gave Q6_K 5/6 and Q4_K_M-imat 3/6, so the app ships Q6_K.
- **Problem found later:** spelled-out amounts fail. Numbers set 13.4, with 233 of 357 messages getting the amount wrong. Probes 6/20.

### v6: number words
- **Data:** nearly doubled to 18,582. Added `ml/hishab/numwords.py`, which converts number words to integers and back and is round-trip tested. Codex wrote 3,594 templates that were filled into 7,327 rows, and 1,691 rows had their digits rewritten as words.
- **Result:** numbers set 13.4 to **81.8** bf16, with only 1 amount error in 357. Probes 6/20 to 14/20. Test fell 2.5 points to 85.7.
- **Why test fell:** category errors rose from 43 to 52, mostly gift, salary and food swaps. The Codex rows follow `docs/LABELLING_GUIDELINE.md`, and the Gemini gold often does not.
- **Found:** the GGUF Bangla gap described above.

### v7: totals, typing noise, negatives
- **Data:** 20,333 rows. +740 rows that state a total that must not become an entry, +303 no-transaction rows labelled `[]`, typing noise on 20% of Codex rows, and +1,447 Bangla-script word-amount rows.
- **Run:** Modal preempted the container at step 5,287 of 6,355 (83%). The relaunch found the epoch-4 checkpoint and resumed past the end, so v7 is epoch 4, not the best epoch 2. Its scores are a lower bound for this data.
- **Result:** test 86.7 bf16 and 86.1 at Q6_K, the best since v5. Category errors fell from 52 to 41, and Bangla on test rose from 84.0 to 89.6. Probes 17/20, the best so far.
- **Regressions:** numbers fell to 77.6, with type flips up from 7 to 20. Messages containing "chhoy" (six) or "ashi" (eighty) come out as income. Count errors on test rose from 9 to 15: it drops the middle item of some 3-item messages, and one probe returned nothing.
- **Fix made:** `train.py` now saves checkpoints every epoch and resumes after a preemption.

### v8: full fine-tune
- **Data:** same as v7.
- **Training:** all weights instead of LoRA, learning rate 3e-5, 3 epochs, best epoch 2 by validation loss. 3,813 steps in 104 min with no preemption.
- **Result at full precision:** the best version on every set. Test 89.6, up 1.4 on v5 and 2.9 on v7. Numbers 84.9, up 3.1 on v6. Wrong test messages fell to 54, the fewest of any version.
- **What improved:** Bangla on test rose to 92.0 and mixed to 97.3. v7's income/expense flips are gone, with type errors on numbers back down from 20 to 7. Count errors on test went back from 15 to 9.
- **Lesson:** with this much data, updating all weights beats LoRA by a wide margin, even though its validation loss looked worse after epoch 1.
- **Phone format:** Q6_K scores 89.2 on test, only 0.4 below bf16, and 80.1 on numbers. That is 3.1 and 6.2 points above any earlier GGUF. Q4_K_M-imat, at 529 MB, scores 88.6 and 80.1, so the smaller file is nearly as good.
- **Phone probes:** 18/20 on both Q6_K and Q4_K_M-imat, the best of any version. Q6_K's two misses are "bish hajar pachsho" read as 25000 and "সাড়ে তিন হাজার" split into two entries. Q4_K_M-imat's misses are 25000 again and 21000 read as 12000.
- **Published 2026-10-17:** GGUFs (Q6_K, Q5_K_M, Q4_K_M-imat) at `ryuma007/hishab-qwen3.5-0.8b-json-full-v8-GGUF`, bf16 weights at `ryuma007/hishab-qwen3.5-0.8b-json-full-v8`, and the training data at `ryuma007/hishab-synthetic-v4` (train, validation, test and numbers splits).
- **Cost:** per step, the full fine-tune (1.64 s) ran about as fast as LoRA v7 (1.77 s). LoRA saves little on this model because a third of its weights are the 248k-token output layer, which costs the same either way. v8 was cheaper than v7 overall because it ran 3 epochs instead of 5 and was not preempted.
- **Verdict:** v8 Q6_K is the ship candidate. It beats the shipped v5 on test (89.2 vs 86.5) and fixes spelled-out numbers (80.1 at Q6_K, against v5's 13.4 even at full precision).

### v9: the tokenizer fix
- **Data and training:** identical to v8 (20,333 rows, full fine-tune, 3 epochs, best epoch 2, val loss 0.1123).
- **What changed:** `hfeval.load_tokenizer` loads the Rust backend straight from `tokenizer.json`
  instead of letting transformers 5 rebuild it. transformers 5.10.2's `Qwen2Tokenizer` ignores the
  `[\p{L}\p{M}]+` letter-run rule in Qwen3.5's tokenizer.json and splits every Bengali word at its
  combining marks (nukta, vowel signs): "এয়ারপোর্টের ... পাইলাম" is 36 tokens for transformers,
  26 for the `tokenizers` library and 26 for llama.cpp. v3..v8 were all trained on the 36-token
  version; the phone always saw the 26-token version. English and Banglish are unaffected
  (identical ids), which is why the gap was Bangla-only.
- **How it was found:** `ml/modal_app/gguf_diag.py` showed the gap is identical across every
  server setting (slots, ubatch, flash attention, KV type, threads) and on llama.cpp master.
  `gguf_diag_logits.py` then fed llama-server the exact HF token ids: next-token probabilities
  match HF to three decimals at every position, so the forward pass was never the problem, while
  `/apply-template` matched HF and `/tokenize` did not. `tok_check.py` pinned it on transformers.
- **Result:** bf16 test 89.2, numbers 83.2 (within noise of v8's 89.6 / 84.9; v8 never paid for the
  defect at bf16 because it was evaluated with the same tokenizer it trained on). GGUF: test Q6_K
  89.2, Q5_K_M 89.4, Q4_K_M-imat 89.6; numbers Q6_K **83.8**, Q5_K_M 84.0, Q4_K_M-imat 84.0.
  Bangla numbers: bf16 84.3, Q6_K 84.3, Q4_K_M-imat 86.5 (v8 Q6_K was 70.8). The phone path and
  the HF path now agree.
- **Q4_K_M-imat is now the best quant, not the worst.** Across v3..v8, 4-bit lost 1 to 4 points to
  Q6_K; in v9 Q4_K_M-imat is 0.4 ahead on test and 0.2 ahead on numbers, within noise of Q6_K
  and of bf16 (one test row is 0.19 points, one numbers row 0.28). Two reasons: the imatrix is
  computed on rendered training text, which is now tokenized the same way the quantized model
  will see at inference, so its importance weights land on the right rows; and the Bangla
  token sequences are 30 to 40% shorter, so there is less low-precision drift per message. The
  old 4-bit penalty was partly the tokenizer defect compounding with quantization noise. Ship
  Q4_K_M-imat (529 MB) unless the phone probes say otherwise.
- **Published:** `ryuma007/hishab-qwen3.5-0.8b-json-full-v9` and `...-v9-GGUF` (Q6_K, Q5_K_M, Q4_K_M-imat); sha256 of the Hub files verified against the volume. Probes on the Hub files: Q6_K 17/20, Q4_K_M-imat 17/20 (same three misses).
- **Lesson:** verify token ids end to end (training tokenizer vs `tokenizers` vs llama.cpp) before
  trusting any GGUF number; `load_tokenizer` now asserts it on a Bangla probe at every run.

## Open problems

1. **Label noise in `test`.** Category is two-thirds of every version's errors. Relabelling in progress: Sonnet 5.5 (`ml/results/relabel/sonnet.jsonl`, all 875 rows, agrees with the old gold on 81% of test) and Gemini 3.8 Flash (`gemini_shard*.jsonl`); `ml/data/relabel_merge.py` keeps rows where both agree. Labellers misread some Banglish number words ("share" = sare, "shaat" = sixty), so on the `numbers` split trust the template gold for amounts.
2. ~~GGUF Bangla gap~~ **Solved in v9**: it was transformers 5 mis-tokenizing Bengali during training, not llama.cpp (see v9). Worth reporting upstream to transformers.
3. **Dropped entries.** v7 sometimes returned `[]` or dropped a middle item, and v8 GGUF occasionally writes broken JSON that the app drops. Worth a dedicated probe set, since silent misses are the worst failure for a ledger app.
4. **No real user messages.** Every eval set is synthetic, and people will type messier Banglish than this.
