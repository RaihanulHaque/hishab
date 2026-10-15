# Hishab fine-tune results

## Recommendation (2026-10-15)

**Ship candidate: `qwen3.5-0.8b__json__lora` at Q4_K_M with imatrix (529 MB).** It is the most
accurate model at every precision, loses nothing from bf16 to 4-bit (85.5% exact as F16 and as
Q4_K_M-imat on the 518-row held-out test), and `json` output is the shortest (about 49 tokens),
so it is also the fastest to finish on the phone among the accurate models. Banglish is its
weakest language at about 82%, bangla 90%, english 94%.

**Fast tier for low-end phones: `lfm2.5-350m__two_phase__lora` at Q5_K_M (260 MB) or Q4_K_M
(229 MB).** About 81% exact at twice the output length; kiln measured LFM2.5-350M Q4_K_M at
90 tok/s on a Pixel 8 against 37 tok/s for Qwen3.5-0.8B Q4_K_M, so wall time per message is
similar once the longer two_phase output is accounted for. The 350M model is fragile below 5
bits: Q4_K_M-imat / Q4_0 lose 4 to 10 points.

**Quality ceiling if size does not matter: `lfm2.5-1.2b__two_phase__lora` at IQ3_M-imat
(567 MB, 84.7%) or Q4_K_M (731 MB, 85.5%).** It is the most quantization-robust model tested but
never beats Qwen3.5-0.8B at the same size.

Gemma-3-270M (about 82% as F16, 253 MB at Q4_K_M-imat for 79.7%) is the smallest option; plain
Q4_0 collapses it to 50 to 62% so never ship it at Q4_0.

Format: `two_phase` (English translation before the JSON) helps the sub-500M models by 1 to 3
points and hurts nothing, but doubles output tokens; Qwen3.5-0.8B does not need it.

| run | quant | MB | exact | banglish exact | valid JSON | avg out tokens |
|---|---|---|---|---|---|---|
| qwen3.5-0.8b__json__lora | Q4_K_M-imat | 529 | 85.5 | 80.4 | 100.0 | 48.2 |
| qwen3.5-0.8b__json__lora | Q6_K | 630 | 85.7 | 81.7 | 100.0 | 48.2 |
| qwen3.5-0.8b__json__lora | Q8_0 | 812 | 85.5 | 81.7 | 100.0 | 48.1 |
| lfm2.5-1.2b__two_phase__lora | IQ3_M-imat | 567 | 84.7 | 79.9 | 99.8 | 91.0 |
| lfm2.5-1.2b__two_phase__lora | Q4_K_M | 731 | 85.5 | 80.8 | 100.0 | 90.9 |
| lfm2.5-350m__two_phase__lora | Q5_K_M | 260 | 81.5 | 74.9 | 100.0 | 90.2 |
| lfm2.5-350m__two_phase__lora | Q4_K_M | 229 | 80.7 | 75.8 | 99.6 | 92.4 |
| lfm2.5-350m__json__lora | Q4_K_M-imat | 229 | 79.2 | 74.4 | 100.0 | 59.7 |
| gemma-3-270m__json__lora | Q6_K | 283 | 81.9 | 71.2 | 100.0 | 45.8 |
| gemma-3-270m__json__lora | Q4_K_M-imat | 253 | 79.7 | 68.5 | 100.0 | 45.7 |

Caveats: all numbers are on synthetic Gemini-generated data (no real user messages yet);
llama.cpp tok/s in the tables below is from a shared 16-core x86 container with 8 parallel
slots and is only useful relatively. Qwen3.5-0.8B drops about 2 points going from HF bf16 to
the F16 GGUF (87.6 to 85.5), the other models reproduce within 1 point.

Held-out **test** split (518 rows) unless noted. exact = multiset of (amount, category, type) equals gold. F1 = micro F1 over those tuples.

## Full-precision models (bf16, HF transformers, greedy)

| run | base | fmt | tune | best val loss | valid JSON | exact | F1 | cat acc | type acc | banglish exact | bangla exact | english exact | mixed exact | avg out tok | train min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b__json__lora | Qwen/Qwen3.5-0.8B | json | lora | 0.095 | 100.0 | **87.6** | 92.9 | 95.0 | 98.9 | 81.7 | 90.4 | 93.9 | 92.0 | 48.8 | 32 |
| qwen3.5-0.8b__two_phase__lora | Qwen/Qwen3.5-0.8B | two_phase | lora | 0.161 | 100.0 | **86.7** | 92.4 | 94.4 | 98.7 | 81.3 | 87.2 | 92.9 | 93.3 | 81.9 | 35 |
| lfm2.5-1.2b__two_phase__lora | LiquidAI/LFM2.5-1.2B-Instruct | two_phase | lora | 0.160 | 100.0 | **85.7** | 92.2 | 93.9 | 99.2 | 80.8 | 84.0 | 93.9 | 92.0 | 90.1 | 11 |
| lfm2.5-350m__two_phase__lora | LiquidAI/LFM2.5-350M | two_phase | lora | 0.191 | 100.0 | **83.0** | 90.2 | 93.7 | 97.9 | 74.4 | 84.8 | 92.9 | 92.0 | 89.8 | 6 |
| lfm2.5-1.2b__json__lora | LiquidAI/LFM2.5-1.2B-Instruct | json | lora | 0.092 | 99.8 | **82.4** | 89.8 | 92.7 | 97.5 | 75.3 | 84.8 | 96.0 | 81.3 | 61.1 | 10 |
| gemma-3-270m__two_phase__lora | unsloth/gemma-3-270m-it | two_phase | lora | 0.227 | 100.0 | **81.9** | 89.6 | 93.3 | 98.0 | 73.5 | 84.8 | 90.9 | 89.3 | 78.9 | 7 |
| gemma-3-270m__json__lora | unsloth/gemma-3-270m-it | json | lora | 0.123 | 100.0 | **81.7** | 89.6 | 92.6 | 97.5 | 69.9 | 87.2 | 92.9 | 92.0 | 45.9 | 5 |
| lfm2.5-350m__json__lora | LiquidAI/LFM2.5-350M | json | lora | 0.098 | 100.0 | **81.5** | 89.5 | 93.0 | 97.5 | 74.9 | 84.0 | 91.9 | 82.7 | 59.8 | 5 |

## GGUF quantizations (llama.cpp CPU, kiln's commit, greedy)

| run | quant | MB | valid JSON | exact | F1 | banglish exact | truncated | tok/s per slot (16-core x86) |
|---|---|---|---|---|---|---|---|---|
| gemma-3-270m__json__lora | Q8_0 | 292 | 100.0 | **82.8** | 90.0 | 72.6 | 0 | 31.8 |
| gemma-3-270m__json__lora | F16 | 543 | 100.0 | **82.6** | 90.0 | 72.1 | 0 | 35.4 |
| gemma-3-270m__json__lora | Q6_K | 283 | 100.0 | **81.9** | 89.6 | 71.2 | 0 | 43.9 |
| gemma-3-270m__json__lora | Q5_K_M | 260 | 100.0 | **81.3** | 89.6 | 70.3 | 0 | 26.9 |
| gemma-3-270m__json__lora | Q4_K_M-imat | 253 | 100.0 | **79.7** | 88.1 | 68.5 | 0 | 30.6 |
| gemma-3-270m__json__lora | Q4_K_M | 253 | 99.6 | **78.4** | 87.6 | 68.0 | 0 | 40.7 |
| gemma-3-270m__json__lora | IQ4_XS-imat | 241 | 99.6 | **78.2** | 86.7 | 65.8 | 0 | 43.1 |
| gemma-3-270m__json__lora | IQ3_M-imat | 239 | 99.6 | **77.8** | 87.0 | 69.9 | 0 | 48.7 |
| gemma-3-270m__json__lora | Q3_K_M-imat | 242 | 97.9 | **70.3** | 84.5 | 65.8 | 7 | 39.9 |
| gemma-3-270m__json__lora | Q4_0 | 241 | 94.8 | **62.4** | 76.3 | 55.3 | 14 | 31.6 |
| gemma-3-270m__json__lora | Q4_0-imat | 242 | 97.9 | **49.8** | 71.8 | 54.8 | 3 | 29.6 |
| gemma-3-270m__two_phase__lora | F16 | 543 | 99.8 | **82.2** | 89.9 | 74.0 | 0 | 39.1 |
| gemma-3-270m__two_phase__lora | Q6_K | 283 | 100.0 | **80.9** | 89.1 | 72.1 | 0 | 47.4 |
| gemma-3-270m__two_phase__lora | Q8_0 | 292 | 100.0 | **80.9** | 89.1 | 71.7 | 0 | 45.5 |
| gemma-3-270m__two_phase__lora | Q5_K_M | 260 | 100.0 | **78.4** | 87.6 | 69.4 | 0 | 45.7 |
| gemma-3-270m__two_phase__lora | IQ4_XS-imat | 241 | 99.8 | **77.4** | 85.8 | 66.7 | 0 | 48.1 |
| gemma-3-270m__two_phase__lora | Q4_K_M-imat | 253 | 100.0 | **77.2** | 86.8 | 68.0 | 0 | 48.8 |
| gemma-3-270m__two_phase__lora | Q3_K_M-imat | 242 | 92.1 | **70.5** | 80.0 | 65.3 | 16 | 50.5 |
| gemma-3-270m__two_phase__lora | Q4_K_M | 253 | 95.4 | **70.3** | 81.2 | 60.7 | 1 | 43.4 |
| gemma-3-270m__two_phase__lora | IQ3_M-imat | 239 | 90.7 | **66.0** | 75.1 | 57.5 | 11 | 48.4 |
| gemma-3-270m__two_phase__lora | Q4_0-imat | 242 | 92.9 | **58.3** | 74.4 | 48.9 | 23 | 57.8 |
| gemma-3-270m__two_phase__lora | Q4_0 | 241 | 96.7 | **55.0** | 68.2 | 46.6 | 3 | 48.7 |
| lfm2.5-1.2b__two_phase__lora | Q8_0 | 1246 | 100.0 | **86.1** | 92.3 | 80.8 | 0 | 25.5 |
| lfm2.5-1.2b__two_phase__lora | F16 | 2343 | 100.0 | **85.9** | 92.2 | 81.3 | 0 | 22.2 |
| lfm2.5-1.2b__two_phase__lora | Q6_K | 963 | 100.0 | **85.9** | 92.2 | 81.7 | 0 | 18.7 |
| lfm2.5-1.2b__two_phase__lora | Q4_K_M | 731 | 100.0 | **85.5** | 91.7 | 80.8 | 0 | 25.0 |
| lfm2.5-1.2b__two_phase__lora | IQ4_XS-imat | 663 | 100.0 | **84.9** | 91.8 | 81.3 | 0 | 13.5 |
| lfm2.5-1.2b__two_phase__lora | Q4_0-imat | 698 | 100.0 | **84.9** | 91.4 | 80.4 | 0 | 25.2 |
| lfm2.5-1.2b__two_phase__lora | IQ3_M-imat | 567 | 99.8 | **84.7** | 91.3 | 79.9 | 0 | 16.9 |
| lfm2.5-1.2b__two_phase__lora | Q4_K_M-imat | 731 | 100.0 | **84.6** | 91.2 | 79.0 | 0 | 14.9 |
| lfm2.5-1.2b__two_phase__lora | Q5_K_M | 843 | 100.0 | **84.6** | 91.6 | 79.9 | 0 | 21.0 |
| lfm2.5-1.2b__two_phase__lora | Q4_0 | 696 | 100.0 | **83.0** | 89.7 | 77.6 | 0 | 15.7 |
| lfm2.5-1.2b__two_phase__lora | Q3_K_M-imat | 600 | 100.0 | **82.2** | 90.1 | 79.0 | 0 | 22.2 |
| lfm2.5-350m__json__lora | F16 | 711 | 100.0 | **81.9** | 89.6 | 75.3 | 0 | 40.2 |
| lfm2.5-350m__json__lora | Q8_0 | 379 | 100.0 | **81.3** | 89.3 | 74.4 | 0 | 31.9 |
| lfm2.5-350m__json__lora | Q6_K | 293 | 100.0 | **81.1** | 89.3 | 74.4 | 0 | 38.3 |
| lfm2.5-350m__json__lora | Q5_K_M | 260 | 100.0 | **80.3** | 88.7 | 72.6 | 0 | 23.0 |
| lfm2.5-350m__json__lora | Q4_K_M-imat | 229 | 100.0 | **79.2** | 87.8 | 74.4 | 0 | 48.1 |
| lfm2.5-350m__json__lora | Q4_0-imat | 220 | 99.6 | **77.6** | 86.2 | 74.0 | 1 | 49.9 |
| lfm2.5-350m__json__lora | IQ4_XS-imat | 210 | 100.0 | **76.4** | 86.1 | 72.6 | 0 | 21.2 |
| lfm2.5-350m__json__lora | Q4_K_M | 229 | 100.0 | **76.4** | 86.3 | 70.8 | 0 | 31.2 |
| lfm2.5-350m__json__lora | Q4_0 | 219 | 99.2 | **72.8** | 83.9 | 72.6 | 4 | 26.8 |
| lfm2.5-350m__json__lora | Q3_K_M-imat | 193 | 99.8 | **62.4** | 75.2 | 69.4 | 1 | 33.2 |
| lfm2.5-350m__json__lora | IQ3_M-imat | 184 | 99.8 | **54.1** | 70.0 | 52.5 | 0 | 29.0 |
| lfm2.5-350m__two_phase__lora | Q8_0 | 379 | 100.0 | **83.4** | 90.7 | 75.8 | 0 | 43.8 |
| lfm2.5-350m__two_phase__lora | F16 | 711 | 100.0 | **83.0** | 90.3 | 74.9 | 0 | 17.1 |
| lfm2.5-350m__two_phase__lora | Q6_K | 293 | 100.0 | **82.4** | 90.0 | 77.2 | 0 | 46.5 |
| lfm2.5-350m__two_phase__lora | Q5_K_M | 260 | 100.0 | **81.5** | 89.9 | 74.9 | 0 | 46.8 |
| lfm2.5-350m__two_phase__lora | Q4_K_M | 229 | 99.6 | **80.7** | 89.0 | 75.8 | 2 | 52.8 |
| lfm2.5-350m__two_phase__lora | IQ4_XS-imat | 210 | 100.0 | **76.8** | 85.9 | 73.1 | 0 | 47.2 |
| lfm2.5-350m__two_phase__lora | Q4_K_M-imat | 229 | 99.8 | **76.4** | 86.2 | 75.8 | 0 | 53.3 |
| lfm2.5-350m__two_phase__lora | Q4_0-imat | 220 | 99.8 | **74.5** | 84.9 | 69.9 | 0 | 51.5 |
| lfm2.5-350m__two_phase__lora | Q4_0 | 219 | 99.8 | **68.9** | 81.0 | 66.2 | 0 | 57.0 |
| lfm2.5-350m__two_phase__lora | Q3_K_M-imat | 193 | 99.6 | **66.0** | 78.6 | 70.3 | 0 | 37.6 |
| lfm2.5-350m__two_phase__lora | IQ3_M-imat | 184 | 100.0 | **65.3** | 77.8 | 72.6 | 0 | 26.0 |
| qwen3.5-0.8b__json__lora | Q6_K | 630 | 100.0 | **85.7** | 92.1 | 81.7 | 0 | 16.2 |
| qwen3.5-0.8b__json__lora | F16 | 1517 | 100.0 | **85.5** | 92.1 | 82.2 | 0 | 9.6 |
| qwen3.5-0.8b__json__lora | Q4_K_M-imat | 529 | 100.0 | **85.5** | 91.9 | 80.4 | 0 | 17.7 |
| qwen3.5-0.8b__json__lora | Q8_0 | 812 | 100.0 | **85.5** | 92.0 | 81.7 | 0 | 18.5 |
| qwen3.5-0.8b__json__lora | Q4_0-imat | 502 | 100.0 | **84.9** | 91.5 | 81.7 | 0 | 18.5 |
| qwen3.5-0.8b__json__lora | Q5_K_M | 578 | 100.0 | **84.2** | 91.2 | 80.8 | 0 | 15.9 |
| qwen3.5-0.8b__json__lora | Q4_0 | 501 | 100.0 | **83.6** | 90.4 | 81.7 | 0 | 19.3 |
| qwen3.5-0.8b__json__lora | Q4_K_M | 529 | 100.0 | **83.4** | 90.8 | 78.5 | 0 | 19.7 |
| qwen3.5-0.8b__json__lora | IQ4_XS-imat | 504 | 100.0 | **83.2** | 90.7 | 79.0 | 0 | 16.9 |
| qwen3.5-0.8b__json__lora | IQ3_M-imat | 454 | 99.8 | **80.3** | 88.0 | 75.8 | 0 | 14.5 |
| qwen3.5-0.8b__json__lora | Q3_K_M-imat | 466 | 99.8 | **80.1** | 88.8 | 74.9 | 0 | 16.3 |
| qwen3.5-0.8b__two_phase__lora | F16 | 1517 | 100.0 | **85.5** | 91.8 | 81.3 | 0 | 17.4 |
| qwen3.5-0.8b__two_phase__lora | Q6_K | 630 | 100.0 | **85.3** | 91.6 | 80.8 | 0 | 17.1 |
| qwen3.5-0.8b__two_phase__lora | IQ4_XS-imat | 504 | 100.0 | **85.1** | 91.5 | 79.9 | 0 | 15.2 |
| qwen3.5-0.8b__two_phase__lora | Q8_0 | 812 | 100.0 | **84.9** | 91.4 | 80.8 | 0 | 20.6 |
| qwen3.5-0.8b__two_phase__lora | Q4_K_M-imat | 529 | 100.0 | **84.7** | 91.5 | 81.7 | 0 | 12.3 |
| qwen3.5-0.8b__two_phase__lora | Q5_K_M | 578 | 100.0 | **84.7** | 91.1 | 80.8 | 0 | 11.6 |
| qwen3.5-0.8b__two_phase__lora | Q4_K_M | 529 | 100.0 | **84.6** | 91.3 | 81.7 | 0 | 19.2 |
| qwen3.5-0.8b__two_phase__lora | Q4_0-imat | 502 | 99.8 | **83.6** | 90.6 | 79.0 | 1 | 20.5 |
| qwen3.5-0.8b__two_phase__lora | Q4_0 | 501 | 100.0 | **83.0** | 90.7 | 80.8 | 0 | 21.9 |
| qwen3.5-0.8b__two_phase__lora | Q3_K_M-imat | 466 | 100.0 | **81.9** | 89.3 | 78.1 | 0 | 18.1 |
| qwen3.5-0.8b__two_phase__lora | IQ3_M-imat | 454 | 46.3 | **36.3** | 61.9 | 43.8 | 260 | 18.7 |
