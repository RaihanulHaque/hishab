# Hishab experiment log

The story of every model version: what we tried, why, and what we decided. The numbers live in
generated tables, so this file never has to be kept in sync by hand.

| file | what it holds | how it is updated |
|---|---|---|
| `RESULTS_TABLES.md` | every score table (status, leaderboard, per quant, per language, errors, training, data, legacy) | generated: `modal run ml/modal_app/collect_results.py` then `python ml/results/build_tables.py` |
| `versions.yaml` | one entry per model version (runs, data, published repos, what changed, status) | one new entry per version |
| `EXPERIMENTS.md` (this file) | the narrative per version, hand measurements (phone speed, probes), open problems | hand-written |
| `VOCAB_TRIM.md`, `RELABEL.md` | how the vocabulary trim and the two-labeller relabel work | hand-written |
| `docs/LABELLING_GUIDELINE.md` | the labelling rules (v2 from 2026-10-21) | owner's decisions |
| `archive/EXPERIMENTS_until_2026-10-21.md` | the previous all-in-one version of this file, tables included | frozen |

## Where things stand (2026-10-21)

- **In the app:** v12-trim38k Q4_K_M-imat (343 MB, app 1.2.0). **Best:** v13-trim38k, not yet published or in the app. Current scores: `RESULTS_TABLES.md`, "Current status".
- **Output format:** `pipe_signed` since v11 (`description|-800|category`, about half the tokens of JSON); vocabulary trimmed to 38k tokens since v11-trim38k (343 MB, ~45% faster on the phone).
- **Eval sets:** from v13 the headline sets are `test_clean_v2`, `numbers_clean_v2`, `small_amounts_v2`, labelled under guideline v2. The older sets stay in the legacy appendix; newer models can score lower there by following the new rules.
- **Next:** publish v13 and switch the app; then real user corrections (opt-in export of review-sheet edits) rather than more synthetic rounds.

## How to read the numbers

- **exact:** the message's set of (amount, category, type) equals the gold; one wrong field fails the whole message. Headline metric.
- **bf16** is the full-precision model on Modal (HF transformers, greedy); **GGUF** is llama.cpp at kiln's commit, the phone path. The app ships Q4_K_M-imat.
- **clean sets** keep only the held-out rows where two independent frontier labellers, labelling from scratch with the guideline, agree; their label is the gold (`RELABEL.md`). A message is about 0.2 points on the 480-row sets, and two training runs of the same recipe differ by about ±1 point.

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
- **Phone format:** every quant from Q8_0 to IQ3_M-imat is within a point of bf16 on both clean sets.

v11 on `numbers_clean`: Q4_0-imat 77.3 (v11-trim38k: 77.3). Every v11 quant built (F16, Q6_K, Q5_K_M, Q4_K_M-imat, Q4_0-imat) parses 100% on both clean sets. Q4_K_M-imat ties Q6_K (93.1 / 79.2) at 529 MB. Q5_K_M on the old splits: test 81.3, numbers 86.0.
- **Phone probes:** 18/20 on both Q6_K and Q4_K_M-imat (sha256 verified against the volume), with category and type right on all 20. "der lakh ... abbu theke" is `gift` again. The two misses are the same amount readings as before: "sare tin hajar" as 3000 and "bish hajar pachsho" as 25000.
- **Probes on every quant (2026-10-17, Mac kiln-cli):** Q8_0, Q6_K, Q5_K_M, Q4_K_M-imat, IQ4_XS-imat and Q3_K_M-imat 18/20; Q4_0-imat 19/20 (only "bish hajar pachsho" as 25000); IQ3_M-imat 18/20 but with a different miss, "pach hajar 2 sho" as 4200, which every other quant gets right. Q5_K_M and IQ3_M read "bish hajar pachsho" as 22000 instead of 25000. Category and type are right on all 20 for every quant.
- **Speed on the Pixel 8:** Q4_K_M-imat 34 tok/s, Q4_0-imat 36, Q6_K 27, Q8_0 24.5; the 3-bit files are no faster and IQ3_M-imat is the slowest of all (19 tok/s, 1.6 s to read the prompt). Full table under "Speed on the Pixel 8".
- **Published in full (2026-10-17):** all 11 quants, F16 to IQ3_M-imat, are now in `...-v10-GGUF`, with eval JSONs for each.
- **Published 2026-10-17:** `ryuma007/hishab-qwen3.5-0.8b-json-full-v10` (bf16) and `...-v10-GGUF` (Q6_K, Q5_K_M, Q4_K_M-imat; Hub sha256 verified against the volume), trained on `ryuma007/hishab-synthetic-v5`, which holds the 20,057-row train split and all five eval splits.
- **Next:** relabel the income and gift categories of the template rows (or regenerate them) the same way; that is what numbers_clean is waiting on.

### v11: pipe output instead of JSON
- **Why:** on the phone only generated tokens cost time (the system prompt is cached), and the JSON array spends more than half its tokens on keys, quotes and `.0`. Measured with the Qwen3.5 tokenizer on test_clean: JSON 45.3 tokens per message, YAML 45.9, TOML 58.1, array of arrays 26.9, pipe rows with a signed amount 20.7.
- **Format (`pipe_signed`):** one `description|amount|category` line per transaction; the amount's sign is the type (`-` expense, `+` income); whole amounts have no `.0`, decimals keep up to 2 places; `NONE` when there is no transaction. Rendered from the same labels as JSON (`schema.render_target`), so there is no separate dataset. `parse_output` detects the format and splits each line on its last two `|`. Every row of every split round-trips exactly.
- **Data and training:** identical to v10 (`train_v10.jsonl`, full fine-tune, lr 3e-5, 3 epochs, best epoch 2). Train time 97 min (v10: 68), so the shorter completions did not make training faster; the prompt dominates the sequence length. Validation loss (0.270) is per token and not comparable with JSON's.
- **Result:** test_clean 92.4 bf16, numbers_clean 79.2, test 80.7, numbers 85.4, all within half a point of v10. Parseable output 100% on every split. Output length 21 tokens on test_clean (v10 45).
- **Phone format (llama.cpp):** test_clean F16 92.4, Q6_K 92.4, Q5_K_M 92.4, Q4_K_M-imat 92.2, Q4_0-imat 92.6; numbers_clean F16 79.2, Q6_K 78.9, Q5_K_M 78.2, Q4_K_M-imat 78.9, Q4_0-imat 77.3.
- **Against v10 at Q4_K_M-imat:** test_clean 92.2 vs 93.1 is 4 messages; 15 rows newly wrong and 11 newly right, the usual churn between two runs. Type errors are 4 in both, so the one-character sign did not make income/expense worse there. On numbers_clean type errors rose from 2 to 6 (category errors 61 to 60), worth watching. The new misses are mostly borderline categories (school trip as entertainment, lunch box as food) and messages with a planned spend recorded as an entry.
- **Decimals:** the one decimal row in test_clean (`1250.50 tk`) comes out as `-1250.50`. Only 3 of the 20,057 training answers have a decimal (2,156 inputs contain `1.5k`-style shorthand, which becomes a whole number), so decimal amounts are not really tested yet.
- **On the Pixel 8 (2026-10-19, Q4_K_M-imat in the app with a pipe parser):** 4 of 4 messages right, including `+5000` income, a `1250.50` decimal, `NONE` for a message with no money in it, and "sare tin hajar" as 3500 (v10's probes read it as 3000). 7 to 33 output tokens per message; speed 18 to 32 tok/s while charging, so no clean speed comparison with v10 yet.
- **Published 2026-10-19:** `ryuma007/hishab-qwen3.5-0.8b-pipe-signed-full-v11` (bf16) and `...-v11-GGUF` (F16, Q6_K, Q5_K_M, Q4_K_M-imat, Q4_0-imat; Q4_K_M-imat sha256 on the Hub matches the file tested on the phone). The app (hishab_offline) now downloads v11.
- **Not done:** kiln probe set, a cool-phone speed comparison against v10.

### v11-trim38k: vocabulary trimming
Full write-up and how to redo it for a new version: `VOCAB_TRIM.md`.
- **Why:** Qwen3.5-0.8B ties its embedding to the output layer: one 248,320 x 1,024 table, 209 MB of the 529 MB Q4_K_M-imat file (stored at Q6_K) and about a third of the parameters. The output layer scores every vocabulary entry for every generated token, so on the phone that table is read once per token. Hishab's whole dataset (every message, both target formats, the prompts) uses 5,875 of the 248k tokens.
- **What was kept (`ml/hishab/vocab_trim.py`):** every token the data uses; all special tokens (including the 7 audio/vision markers declared only in `tokenizer_config.json`); every single-character token, so any text still encodes, byte by byte if need be; then the lowest-id (most frequent) tokens that are plain ASCII or Bangla script, until 38,000; plus the merge parents of every kept token. Merges whose parts or result were dropped are removed.
- **How (`ml/modal_app/trim_vocab.py`):** renumber the kept tokens in their original order, slice the one embedding tensor (`model.language_model.embed_tokens.weight`, which the output layer and the MTP layer share), remap the special ids in config and generation_config, and write a new run `qwen3.5-0.8b__pipe_signed__full__v11__trim38k`. No training. llama.cpp's converter recognises the tokenizer by hashing the ids of a test string, which changes with renumbering, so `convert_pinned.py` pins the pre-tokenizer to `qwen35` (the regex is unchanged).
- **Checks:** all 65,956 Hishab texts split into exactly the same pieces before and after. Other scripts still round-trip losslessly, just in more pieces (Cyrillic `Привет мир` 4 to 19 tokens, accented Latin `Héllo café — naïve résumé` 8 to 15); Bangla and Banglish probes keep their counts.
- **Size:** 0.75B to 0.54B parameters. F16 1,517 to 1,076 MB, Q6_K 630 to 443, Q4_K_M-imat 529 to **343**, Q4_0-imat 502 to 316.
- **Accuracy (llama.cpp), trimmed vs untrimmed v11:**

| split | F16 | Q6_K | Q4_K_M-imat | Q4_0-imat |
|---|---|---|---|---|
| test_clean | 92.4 / 92.4 | 92.4 / 92.4 | 92.0 / 92.2 | 92.9 / 92.6 |
| numbers_clean | 79.2 / 79.2 | 78.9 / 78.9 | 78.9 / 78.9 | 77.3 / 77.3 |
| test | 80.3 / | 80.5 / 80.7 | 80.3 / 80.5 | 80.1 / |
| numbers | 85.4 / | 85.2 / 85.4 | 85.2 / 84.6 | 84.3 / |

  F16 is identical on both clean sets, as expected when every eval text keeps its tokens; the quant cells move by one or two messages because the quantization and imatrix were redone. Parseable output 100% everywhere.
- **Speed on the Pixel 8 (2026-10-19, kiln-cli 0.5.0, CPU, Q4_K_M-imat, alternating, 15 s rest, 5 runs each):** generation **51.5 tok/s median (50.2 to 53.5)** trimmed against **35.4 (33.2 to 36.7)** untrimmed, about 45% faster. Prompt reading is faster too (about 120 vs 60 to 100 tok/s on an 11-token raw prompt). Resident memory 0.34 GB. The phone was charging, which slowed both equally since runs alternated.
- **Risk:** words outside the 38k tokens split into smaller pieces the model rarely saw in that form. None of the eval sets hits this, and real users mostly type the same English, Bangla and Banglish words, but rare spellings and other scripts are where to look if the trimmed model ever misreads something the untrimmed one did not.
- **Published 2026-10-19:** `ryuma007/hishab-qwen3.5-0.8b-pipe-signed-full-v11-trim38k` (bf16) and `...-trim38k-GGUF` (F16, Q6_K, Q4_K_M-imat, Q4_0-imat; Q4_K_M-imat sha256 on the Hub matches the benchmarked file). hishab_offline now downloads it (343 MB, about a minute on Wi-Fi).
- **In the app on the Pixel 8:** the same 4 test messages as v11 give the same answers, read in 1.1 s / 0.7 s / 0.7 s against v11's 1.8 / 1.3 / 1.6 s; warm-up generation 47.7 tok/s against 32.3. Prompt reading with the cached system prompt also dropped (284 vs 467 ms).

### v11-trim38k-L20: depth pruning, tried (2026-10-20)
- **Idea:** after the vocabulary trim, the 24 layers are 80% of the parameters, and on the phone generation time scales with bytes read per token. Remove the layers the task needs least, then heal the rest. `ml/modal_app/prune_layers.py`.
- **Which layers:** skip each layer in turn and measure top-1 agreement with the full model on test_clean answer tokens. Layer 0 is essential (3.6% agreement without it), the last layer (23, full attention) is next (90.5%); the middle linear-attention layers barely matter: 8 (99.3%), 9 (99.2%), 12 (99.2%), 13 (99.1%). Only linear-attention layers were candidates: the 6 full-attention layers do the copying extraction relies on. llama.cpp reads each layer's kind from the GGUF (`attention.recurrent_layers`), so an irregular pattern is fine on the phone.
- **Cut and heal:** layers 8, 9, 12, 13 removed (0.54B to 0.45B parameters). Agreement with v11-trim38k: 96.4% straight after cutting. Healed with all weights trainable on train_v10, 2 epochs, lr 2e-5, loss = KL to v11-trim38k on answer tokens + 0.2 x CE on the gold; 34 min on one A100. Agreement after: 98.2%.
- **Accuracy, bf16 (v11 in brackets):** test_clean **92.4** (92.4), test 80.7 (80.7), numbers 84.0 (85.4), numbers_clean **77.3** (79.2), validation 78.0 (79.2). Q4_K_M-imat (v11-trim38k in brackets): test_clean 91.8 (92.0), numbers_clean 76.3 (78.9). So everyday messages hold; spelled-out amounts lose about 2 points.
- **Size:** Q4_K_M-imat 343 to **288 MB** (−16%); kiln reports 0.29 GB resident (0.34).
- **Speed on the Pixel 8 (kiln-cli, CPU, raw prompt, 20 and 24 layers alternating with 15 s rest, 6 rounds):** 20 layers faster in every round, by 10% to 62%, median **+24%**. The phone throttled from round 3 (24-layer speed fell from 51 to 30 tok/s), so only the paired ratios are meaningful.
- **Verdict:** a real trade, not a free win like the vocabulary trim. Not in the app: it costs ~2.6 points on spelled-out amounts for ~24% speed. Worth reconsidering for slow phones (the Snapdragon 680 runs at ~10 tok/s), or after v12 data, healed again.

### MTP (multi-token prediction): tried, parked (2026-10-20)
- **Idea:** Qwen3.5-0.8B has one MTP layer (~25M parameters) that guesses the token after next from the main model's last hidden state. llama.cpp (`--spec-type draft-mtp`) lets the main model check the guesses in one pass, so greedy answers stay the same and only speed can change.
- **Found:** our fine-tuned runs had lost the layer (transformers' `Qwen3_5ForCausalLM` has no MTP module, so the merged models never saved `mtp.*`; the GGUFs say `nextn_predict_layers=1` but have no block 24). `ml/modal_app/mtp.py` grafts the base model's layer back and benchmarks; `ml/modal_app/train_mtp.py` trains only the MTP layer (main model frozen) to match v11's own next-token distribution (KL), 2 epochs on train_v10, 7.5 min on one A100.
- **Training worked:** top-1 agreement with the main model 68% to **97.3%** on test_clean (96.4% validation). The layer adds 13 MB (356 MB Q4_K_M-imat).
- **Modal (llama-server, CPU, all 476 test_clean, Q4_K_M-imat):** exact 92.0 in every setting. Trained MTP, 1 draft token: 98% accepted, +59% generation speed, 476/476 answers identical; 2 drafts: 86% accepted, **+87%**, 476/476 identical; 3 drafts: +113% but 469/476 identical (batched verification rounds differently; score unchanged). Untrained base layer: 68% accepted, +50% at 1 draft.
- **Pixel 8 (llama-server built from kiln's llama.cpp commit, pinned to the 5 big cores): no gain.** Plain 46.9 tok/s; trained MTP 45.1 (1 draft, 99% accepted) and 41.9 (2 drafts, 85%), answers identical. llama-bench shows verifying 2 tokens costs 1.3x one token and 3 tokens 1.9x, so ~1.4x was possible in theory; the per-step overhead (a second graph, recurrent-state snapshots for rollback, extra sampling) eats it on the phone. A first Android build without OpenMP ran MTP at 1 to 2 tok/s: ggml then starts threads for every graph, and MTP runs many tiny graphs (kiln builds with OpenMP for this reason).
- **Verdict:** parked. Accuracy is unaffected and acceptance is excellent, but it only pays where verification is cheap relative to step overhead (desktop). kiln has no speculative decoding; a lean implementation might recover part of the ~1.4x, unmeasured. Runs on the `hishab-runs` volume: `..._v11__trim38k__mtpbase` (grafted) and `..._v11__trim38k__mtp` (trained); not published.

### v12: amounts under 10 taka (2026-10-21)
- **Problem:** "5 taka rickshaw bhara" came out as `-500`. No training row had an amount under 10: the template filler's smallest size (`tiny`) was 10 to 100 taka, and a lone "5" in training almost always meant "5 hajar" or "5 sho". The eval sets could not see it (smallest amounts: test_clean 15, numbers_clean 10).
- **Data:** a `micro` size (1 to 9 taka) in `fill_templates.py`, and `generate_llm_templates.py --focus micro`, where every message has at least one 1 to 9 taka item (toffee, photocopy page, leguna, coin to a beggar, a few taka found or repaid), often next to a normal-sized one. The labelling guideline is now part of that prompt. Generated by Claude Sonnet through `claude -p` (`ml/data/gen_micro.sh`): 434 templates. Notes from the pilot: Haiku's messages were noticeably less natural; Codex timed out (15 min with nothing back per batch of 10 to 25, at default and medium reasoning effort); without the guideline in the prompt Sonnet labelled tutoring income as `tuition` and bought presents as `gift`; the first "English" batch came out entirely in Banglish (the examples were Banglish) and was relabelled, then regenerated with English examples. Amounts are filled by `numwords`, so labels are exact by construction (digits, Bangla digits, words: "panch", "পাঁচ", "৫").
- **Splits (`ml/data/build_micro.py`):** templates split before filling, 20% held out: `small_amounts.jsonl` (87 rows, every amount 1 to 9 present 6 to 13 times) and 3 rows per training template (15% with typing noise): +1,041 rows, `train_v12.jsonl` = 21,098 rows.
- **Before:** v11-trim38k Q4_K_M-imat scored **23.0** on small_amounts; 58 of its 90 small amounts were multiplied by 10, 100 or 1000 ("3 tk back" became +3000, "6 tk" became 60), 31 read right.
- **Training:** v11 recipe (pipe_signed, full fine-tune, lr 3e-5, 3 epochs, best epoch 2), then the vocabulary trim to 38k (6,037 tokens used by the data, all kept; every text tokenizes as before).
- **Result (Q4_K_M-imat, v11-trim38k in brackets):** small_amounts **81.6** (23.0), test_clean 91.8 (92.0), numbers_clean 78.2 (78.9), test 79.2 (80.3), numbers 84.3 (85.2). bf16: small_amounts 80.5, test_clean 92.4, numbers_clean 78.2, test 79.5, numbers 84.3. **All 90 small amounts read right.** The 16 small_amounts misses are categories (11), types (4) and one count: a lozenge as `medical` (gold `food`), a sold saree as `other` (gold `salary`), a gum as `shopping`. The other sets move by one to five messages, within run-to-run churn.
- **In the app on the Pixel 8 (2026-10-21):** "5 taka rickshaw bhara" −৳5, "photocopy 3 tk, rickshaw 30", "cha 8 tk ar singara 5 tk", "5 hajar taka basha vara dilam" −৳5,000 and "salary 45k pelam, electricity bill 1250.50" all right. Speed was 13 to 19 tok/s because the phone was charging and throttled (thermal status 1, CPU at 70 °C); the file has the same size and layers as v11-trim38k (51 tok/s cool).
- **Published 2026-10-21:** `ryuma007/hishab-qwen3.5-0.8b-pipe-signed-full-v12-trim38k` (bf16) and `...-GGUF` (Q4_K_M-imat; Hub sha256 matches the tested file). Training data and the `small_amounts` split published as `ryuma007/hishab-synthetic-v6` (train 21,098 + all v5 eval splits + small_amounts 87); the model cards link it.
- **Probes (Mac kiln-cli, Q4_K_M-imat, v11 → v12):** "5 taka rickshaw bhara" −500 → **−5**; "ভিক্ষুককে ২ টাকা দিলাম" −2000 → **−2**; "5 hajar taka basha vara" −5000 both; "5 sho taka diye juta" −500 both; "cha 8 tk ar singara 5 tk" right in both.

### v13: template and lexicon labels fixed, labelling guideline v2 (2026-10-21)
- **Problem:** v12 still lost most of its `numbers_clean` messages on categories (64 of 69 misses), mainly earned income not labelled `salary`. The 8,775 Codex-template rows and the 1,691 number-word copies had never been checked against the guideline; the copies even kept the pre-v10 Gemini labels of the rows they were made from.
- **Step 1, guideline v1, two labellers (`ml/data/relabel_templates.sh`):** the 10,466 template and number-word rows relabelled from scratch by Sonnet (`claude -p`) and a second labeller: Gemini 3.8 Flash ran out of agy credits after 520 rows, so Codex (7,041 rows, low reasoning effort, run with its own plugin-free `CODEX_HOME` because the user's Codex re-fetched two git plugin marketplaces on every call) and Opus for the rest (3,425). On the 121-row pilot each second labeller agreed with Sonnet 92 to 94%, and Codex/Gemini/Opus agree with each other 98%. Merge (`apply_relabel.py --amount-locked`, amounts are exact by construction so only categories/types may move): 7,890 confirmed, 1,276 relabelled, 1,210 dropped (labellers disagree), 90 dropped (agreed on other amounts) → `train_v13_v1.jsonl`.
- **Spot check and guideline v2:** the owner reviewed 40 relabelled rows and changed the rules rather than the labels (`docs/LABELLING_GUIDELINE.md` v2): money sent to or received from family follows its stated purpose (bazar → `grocery`, fees → `tuition` income), `gift` only without a purpose; wages I pay are `salary` expense; a driver's daily "joma" to the vehicle owner is `transport`; regular sales are `salary`, a one-off sale is `other`; scholarships stay `salary`, school trips `tuition`. A decision order (transaction? → direction → loans → stated purpose → family/earned) and small explicit tie-breaks (expected money is a plan, a late fee takes the bill's category, a present is labelled by what it is, an unspecified book is `tuition`) were added.
- **Step 2, guideline v2 (`ml/data/relabel_v2.sh`, `ml/data/build_v13.py`):** Opus relabelled all 20,795 training rows (all sources but the `[]` negatives) and Opus + Sonnet the 1,221 eval rows. Training: a v2 change is applied only when Sonnet, asked separately on the 1,658 changed rows, agrees and the amounts equal the current label (Opus misread a few Banglish amounts, e.g. "pochash hajar" as 25,000): 1,252 applied, 406 kept (mostly loan-direction flips on ambiguous wording, 48 of 57). Biggest applied changes: earned income from `other`/`tuition` to `salary` (378 + 83), wages paid to `salary` expense (121), prizes to `gift` (118), donations to `other` (90); 1,196 of the changed rows were programmatic-lexicon rows, never checked before. Eval: new `test_clean_v2` (482 of 518, 84 labels differ from the old gold), `numbers_clean_v2` (306 of 357, 58 differ), `small_amounts_v2` (76 of 87), `validation_v2` (agreed label where both agree, else the old one; used for epoch selection). The old clean sets are kept for comparison.
- **Data:** `train_v13.jsonl`, 19,798 rows (v12: 21,098; 1,300 ambiguous template rows dropped).
- **Training:** v12 recipe on `train_v13`, epoch picked on `validation_v2` (`train.py --val-split`), best epoch 2, 92 min; trimmed to 38k tokens (6,037 used by the data, same as v12).
- **Result, Q4_K_M-imat (v12-trim38k in brackets):** test_clean_v2 **94.2** (89.6), numbers_clean_v2 **89.2** (74.5), small_amounts_v2 **85.5** (82.9); on the old clean sets test_clean **94.3** (91.8), numbers_clean **90.2** (78.2), small_amounts **86.2** (81.6); test 79.0 (79.2); numbers 74.8 (84.3). bf16: test_clean 94.3, numbers_clean 90.9, small_amounts 88.5, test_clean_v2 94.4, numbers_clean_v2 89.5, small_amounts_v2 88.2. All 90 small amounts still read right. Wrong messages on test_clean (bf16): 27 (v12: 36), count 10, category 13; numbers_clean 29 (v12: 69).
- **The `numbers` drop is the relabel, checked:** of the 52 `numbers` messages v12 got right and v13 gets wrong, 29 are exactly the label both v2 labellers agreed on (the old template gold is what changed), 14 are rows the v2 labellers disagree on, 9 are real regressions. On `test` the same split is 9 / 4 / 4. The original-label sets now measure agreement with the old conventions.
- **Verdict:** clear win on every clean set, biggest where the labels were wrong (numbers_clean +12). Not published or in the app yet.

## Open problems

1. **Label noise.** Fixed for the Gemini rows in v10 (test_clean 92.2) and for the template, number-word and lexicon rows in v13 (numbers_clean 90.2). Was open for the Codex-template rows: their income and gift categories disagree with the guideline (numbers_clean, see v10). The 42 + 40 held-out rows and 276 train rows where the two labellers disagree need a human (`splits_clean/disagreements.jsonl`).
2. ~~GGUF Bangla gap~~ **Solved in v9**: it was transformers 5 mis-tokenizing Bengali during training, not llama.cpp (see v9). Worth reporting upstream to transformers.
3. **Dropped entries.** v7 sometimes returned `[]` or dropped a middle item, and v8 GGUF occasionally writes broken JSON that the app drops. Worth a dedicated probe set, since silent misses are the worst failure for a ledger app.
4. ~~**Amounts under 10 taka.**~~ **Fixed in v12** (small_amounts 23.0 to 81.6, every small amount read right). Was: "5 taka rickshaw bhara" came out as `-500` (v11-trim38k and L20 alike). No training row has an amount under 10, and a lone "5" in training almost always means "5 hajar" or "5 sho"; the eval sets cannot see it (smallest amounts: test_clean 15, numbers_clean 10). v12 needs a few hundred 1 to 9 taka rows ("2 tk cha", "৫ টাকা") and some in the eval sets.
5. **No real user messages.** Every eval set is synthetic, and people will type messier Banglish than this.

## Hand measurements

Measured outside the eval pipeline, so they are not in the generated tables.

### Speed on the Pixel 8

#### v10 quants (2026-10-17)

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

#### v11 and v11-trim38k (2026-10-19)

v11's file is the same size and architecture as v10's, so its tok/s matches; its saving is in tokens
per answer: 21 against 45 on test_clean (20 against 44 over the training targets). v11-trim38k then
makes every token cheaper, because the output layer scores 38k instead of 248k entries.

kiln-cli 0.5.0 on the Pixel 8 (CPU, charging), Q4_K_M-imat, raw 11-token prompt, the two files
alternating with 15 s rest, 5 runs each:

| file | size | generation tok/s, median (range) |
|---|---|---|
| v11 | 529 MB | 35.4 (33.2 to 36.7) |
| **v11-trim38k** | **343 MB** | **51.5** (50.2 to 53.5) |

kiln reports the trimmed model at 0.54B parameters and 0.34 GB resident.

In the app (system prompt cached, greedy), the same messages:

| message | v11 | v11-trim38k |
|---|---|---|
| salary 45k pelam, rickshaw 60, electricity bill 1250.50 (32 tokens out) | 1.8 s | 1.1 s |
| lunch -800, abbu dilo +5000 (21 tokens) | 1.3 s | 0.7 s |
| sare tin hajar tk bazar, cha 20 (17 tokens) | 1.6 s | 0.7 s |
| warm-up "chaa 10" (7 tokens) | 32.3 tok/s | 47.7 tok/s |

Against v10 (JSON, untrimmed, ~45 tokens at ~34 tok/s, about 1.3 s of generation per message), a
typical v11-trim38k answer of ~21 tokens at ~50 tok/s is about 0.4 s: roughly three times faster.

What it says:
- **Down to 4 bits, smaller is faster,** because generation is limited by reading the weights from memory.
- **Below 4 bits it stops paying.** IQ4_XS and Q3_K_M are no faster than Q4_K_M and take 0.7 to 0.8 s to read the prompt instead of 0.46 s. IQ3_M is the slowest file of all: its codebook lookups cost more CPU than its 75 fewer MB save.
- **Q4_0-imat is the fastest,** about 6% ahead of Q4_K_M-imat (9% in the alternating runs), as kiln's QUANT.md found for LFM2.5. For a typical 45-token Hishab answer that is about 0.1 s per message. It scores 93.7 on test_clean but 77.9 on numbers_clean against Q4_K_M-imat's 79.2, so Q4_K_M-imat stays the pick.
- **RAM:** the K and I quants below 5 bits peak at about 0.9 to 1.2 GB; Q8_0 needs 1.8 GB.

### Phone probes, Q6_K through kiln-cli

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

### A GGUF-only output glitch (v8)

v8 Q6_K wrote invalid JSON on 3 of 357 numbers rows (v9: Q6_K 0, Q4_K_M-imat 1). In each one the
description contains a number word and the output jumps straight into digits, skipping the
`","amount":` part, for example `"description":"তেল বিক্রির চার000.0,`. The bf16 model never does
this. The app's lenient parser drops such entries, so the user would see a missing line.
