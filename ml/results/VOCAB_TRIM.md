# Vocabulary trimming (2026-10-19)

Qwen3.5-0.8B carries a 248k-token vocabulary for 200+ languages. Hishab reads English, Bangla
and Banglish. Cutting the vocabulary to the 38k tokens those need makes the phone file 35%
smaller and generation about 45% faster, with no retraining and the same accuracy.

| | v11 | v11-trim38k |
|---|---|---|
| vocabulary | 248,320 | **38,000** |
| parameters | 0.75B | **0.54B** |
| Q4_K_M-imat file | 529 MB | **343 MB** |
| generation on a Pixel 8 | 35.4 tok/s | **51.5 tok/s** |
| test_clean, F16 / Q4_K_M-imat | 92.4 / 92.2 | 92.4 / 92.0 |
| numbers_clean, F16 / Q4_K_M-imat | 79.2 / 78.9 | 79.2 / 78.9 |

Published as `ryuma007/hishab-qwen3.5-0.8b-pipe-signed-full-v11-trim38k` (bf16) and
`...-trim38k-GGUF`. The app ships its Q4_K_M-imat.

## Why it works on this model

Qwen3.5-0.8B ties its input embedding to its output layer: one 248,320 x 1,024 table, about a
third of the parameters and 209 MB of the 529 MB Q4_K_M-imat file (llama.cpp stores it at Q6_K).
The output layer scores every vocabulary entry for every generated token, so on the phone that
whole table is read once per token. Generation on a phone is limited by memory reads, so a
smaller table means faster tokens. The 24 transformer layers, where the model's skill lives, are
not touched.

Hishab's whole dataset uses only 5,875 of the 248k tokens (every message, both target formats,
the system prompts and the chat markup). Everything else is other languages, code and rare
pieces.

The same trick works on any model whose vocabulary is a large share of its weights. It would
gain about as much on Gemma 3 270M (262k tokens, tied) and much less on LFM2.5 (65k tokens).

## What is kept

`ml/hishab/vocab_trim.py`, `select_tokens`, in this order:

1. every token the Hishab texts use;
2. every special token: the 26 in `tokenizer.json` plus the 7 audio and text-to-speech markers
   (`<|audio_start|>`, `<tts_pad>`, ...) that Qwen3.5 declares only in `tokenizer_config.json` (transformers adds them at load time, so the
   config refers to their ids);
3. every single-character token. In byte-level BPE these include the 256 byte tokens, so any
   text at all still encodes, in the worst case byte by byte;
4. the lowest-id tokens that are plain ASCII or Bangla script (with Bangla punctuation, ZWJ /
   ZWNJ and curly quotes), until the target size. Lower ids were merged earlier in BPE training,
   so they are the more frequent pieces;
5. the merge parents of every kept token, recursively. BPE builds each token from two smaller
   ones; without the parents a kept token could never be produced.

`trim_tokenizer_json` renumbers the kept tokens 0..N-1 in their original order, puts the special
tokens right after them, and drops every merge whose two parts or result were removed.

**Guarantee:** any text whose tokens were all kept splits into exactly the same pieces as
before. BPE only produces a token after producing its parents, and all of those are kept, so the
encoding path is unchanged. Text with removed tokens falls back to smaller kept pieces.

## Choosing the size

| target | file (Q4_K_M-imat) | what is kept |
|---|---|---|
| 7,248 | ~330 MB | only what the data uses, plus bytes, specials and parents |
| **38,000** (shipped) | 343 MB | the above plus the ~31k most frequent English and Bangla tokens |
| ~130,000 | ~430 MB | every ASCII or Bangla token (ASCII also covers French, Spanish, code) |

Above about 10k tokens the size barely changes: at 38k the table is 32 MB of the 343. The extra
English and Bangla tokens are cheap insurance for words real users type that the synthetic data
never had. 38k was chosen for that margin.

## Checks

- **Same tokens:** all 65,956 Hishab texts (every message, both target formats, prompts) split
  into identical pieces before and after. `trim_vocab.py` refuses to write the model otherwise.
- **Other text still round-trips losslessly**, in more pieces:

  | text | tokens before | after |
  |---|---|---|
  | নমস্কার, আজ ৫০০ টাকার মাছ কিনলাম 🐟 | 24 | 24 |
  | sare tin hajar tk dilam bKash e | 12 | 12 |
  | Héllo café — naïve résumé | 8 | 15 |
  | emoji 😶‍🌫️ ✅ 🦙 | 13 | 18 |
  | 我想在apple工作 1314151天 | 13 | 27 |
  | Привет мир | 4 | 19 |

- **Special tokens** stay single tokens. In the shipped tokenizer the 37,967 regular tokens come
  first, then the 33 special ones; `<|im_end|>` (end of answer) is id 37,969.
- **Accuracy** with llama.cpp, trimmed / untrimmed:

  | split | F16 | Q6_K | Q4_K_M-imat | Q4_0-imat |
  |---|---|---|---|---|
  | test_clean | 92.4 / 92.4 | 92.4 / 92.4 | 92.0 / 92.2 | 92.9 / 92.6 |
  | numbers_clean | 79.2 / 79.2 | 78.9 / 78.9 | 78.9 / 78.9 | 77.3 / 77.3 |
  | test | 80.3 / | 80.5 / 80.7 | 80.3 / 80.5 | 80.1 / |
  | numbers | 85.4 / | 85.2 / 85.4 | 85.2 / 84.6 | 84.3 / |

  F16 is identical, as the guarantee says it must be. The quantized files differ by one or two
  messages either way because the quantization and the imatrix were redone on the new table.
  Every file parses 100% of its answers.

- **Speed** on the Pixel 8 (kiln-cli 0.5.0, CPU, Q4_K_M-imat, the two files alternating with 15 s
  rest, 5 runs each): 51.5 tok/s median (50.2 to 53.5) against 35.4 (33.2 to 36.7).
  In the app the same messages took 1.1 / 0.7 / 0.7 s against 1.8 / 1.3 / 1.6 s.

## How to trim a new model version

```bash
# 1. trim a finished run (CPU container, a minute): writes /runs/<run>__trim38k
conda run -n ML modal run ml/modal_app/trim_vocab.py --run qwen3.5-0.8b__pipe_signed__full__v12 --target 38000

# 2. GGUFs + eval, exactly as for any run (gguf.py sees `vocab_trim` in eval_hf.json)
conda run -n ML modal run ml/modal_app/gguf.py --runs qwen3.5-0.8b__pipe_signed__full__v12__trim38k \
    --quants F16,Q6_K,Q4_K_M-imat,Q4_0-imat --split test_clean

# 3. publish
conda run -n ML modal run ml/modal_app/gguf.py --runs <trimmed run> --only-push \
    --push-only F16,Q6_K,Q4_K_M-imat,Q4_0-imat --dataset ryuma007/hishab-synthetic-v5
conda run -n ML modal run ml/modal_app/push_model.py --run <trimmed run> --dataset ryuma007/hishab-synthetic-v5
```

Then set `kModelFile`, `kModelUrl` and `kModelBytes` in the app's `lib/ai/model_manager.dart`.
Nothing else in the app changes: the vocabulary lives inside the GGUF, and kiln reads it from
there.

The kept set is recomputed from the data each time, so new training rows bring their tokens
with them. Always train on the full model and trim afterwards: training needs no change, and
the trim is free to redo with another size.

## Gotchas met on the way

- **llama.cpp's converter recognises a tokenizer by hashing the ids** it gives a fixed test
  string. Renumbering changes the hash, so conversion stops with "BPE pre-tokenizer was not
  recognized". The regex split is unchanged, so `ml/modal_app/convert_pinned.py` runs the
  converter with the type pinned to `qwen35`, the type untrimmed Qwen3.5 gets. `gguf.py` uses it
  for any run whose `eval_hf.json` has `vocab_trim`, and skips its usual restore of the base
  repo's tokenizer files, which would undo the trim.
- **Special tokens can live in two files.** Qwen3.5's `tokenizer_config.json` lists 7 more than
  `tokenizer.json`. They have to be carried into the trimmed tokenizer, or the config refers to
  ids that no longer exist.
- **Slice by shape, not by name.** `trim_vocab.py` slices every tensor whose first dimension is
  the vocabulary size. For Qwen3.5 that is only `model.language_model.embed_tokens.weight`; the
  output layer and the multi-token-prediction layer share it.
- **Don't pad the table.** Extra zero rows would give their tokens a logit of 0, which greedy
  decoding can pick when every real logit is negative.

## Risks and what to watch

- A word outside the 38k tokens splits into smaller pieces the model rarely saw in that form.
  No eval set hits this, and Bangla / Banglish / English probes keep their token counts, but
  rare spellings, names and other scripts are where to look if the trimmed model ever misreads
  something the untrimmed one reads right. The fix is a bigger target, not retraining.
- The model can no longer write a removed token. Hishab's answers copy words from the message
  and use fixed category names, so this has not mattered.

## Prior work

Vocabulary trimming of multilingual models is established for encoders and T5: Abdaoui et al.,
"Load What You Need: Smaller Versions of Multilingual BERT" (2020); David Dale's mT5 trimming
for Russian and English (2021); Ushio, Zhou and Camacho-Collados, "Efficient Multilingual
Language Model Compression through Vocabulary Trimming" (2023, the `vocabtrimmer` library). All
report large size savings with little or no loss. This is the same idea applied to a
decoder-only model with a tied output layer, served through llama.cpp, where the output layer
makes it a speed win as well as a size win.
