# v6 / v7 notes (overnight 2026-10-16 → 17)

Qwen3.5-0.8B, LoRA r=16, bf16 greedy on the held-out splits. `numbers` = 357 digit-free rows
with spelled-out amounts (`ml/datasets/splits/numbers.jsonl`), never trained on. Probes = the
20 Banglish/Bangla number phrases in `ml/datasets/probes_numbers.jsonl` run on the Q6_K GGUF
through kiln-cli (the phone path).

| run | train rows | test exact | numbers exact | numbers amounts right | probes amount | notes |
|---|---|---|---|---|---|---|
| v5 | 9,564 | **88.2** | 13.4 | 71/357 (20%) | 4/20 | shipped in the app; no number-word data |
| v6 | 18,582 | 85.7 | **81.8** | **356/357** | 16/20 | Codex templates + numwords; 5 epochs, best epoch 2 |
| v7 | 20,333 | 86.7 | 77.6 | 354/357 | – | epoch-4 weights of a preempted 5-epoch run (see below) |

GGUF of v6 on the numbers slice: F16 75.9, BF16 76.2, Q8_0 76.8, Q6_K 75.9, Q5_K_M 73.9,
Q4_K_M-imat 73.1; "-e8" variants (Q8_0 embeddings/output) change nothing. The ~6-point gap
to bf16 is therefore llama.cpp's inference path for this model, not quantization: the same
Bangla-script compounds drop a zero in every GGUF ("নয় হাজার পাঁচশো" → 950) while HF gets
them right. NFC normalisation (now applied in the app and the eval) is correct but was not the
cause. Banglish is unaffected (147/149 at Q6_K); Bangla is 74/89.

## What changed from v6 to v7 (data)

+740 rows that state a total which must not become an entry, +350 no-transaction rows
labelled `[]`, 20% of Codex rows with typing noise, +1,447 Bangla-script word-amount rows.

## v7 caveat

The v7 container was preempted at 83% of a 5-epoch run. The relaunch (3 epochs, with the new
checkpoint-resume code) found the preempted run's epoch-4 checkpoint on the volume and
resumed past the end, so "v7" is the epoch-4 weights, not the best epoch (epoch 2 had the
lowest validation loss in both v6 and v7). Treat its numbers as a lower bound for the data.

What the epoch-4 v7 shows anyway:
- test category errors fell 55 → 41 (the Codex conventions are more consistent than the
  Gemini gold), but count errors rose 9 → 15: it drops the middle item of some 3-item
  messages, and split one stated total into two invented entries.
- numbers slice: amounts still 354/357; the exact score fell on type (4 → 12). The type
  flips are spurious: rows containing "chhoy/chhoi/ছয়" (six) or "ashi" (eighty) come out as
  income. That is overfitting on a narrow augmentation, which is what the 4th epoch does.

## Next

v8 = full fine-tune (all weights), 3 epochs, best epoch by validation loss, same v7 data.
If v8 is not clearly better than v6 on test + numbers, re-run v7 cleanly (LoRA, 3 epochs)
before deciding; the data change and the epoch effect are confounded in the row above.
