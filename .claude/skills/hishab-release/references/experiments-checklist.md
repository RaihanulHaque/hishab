# EXPERIMENTS.md checklist

Every table below gets a row for each new version or experiment, next to the previous version's
row, in that table's column order. `ml/results/experiment_rows.py` prints most rows; the source of
each is given. Leave a cell blank when that eval was not run.

## Top of the file

| section | what to add | source |
|---|---|---|
| Where things stand | "In the app today" line; one bullet for the new result | judgement + numbers below |
| Timeline | `| vN | what changed | headline result |` | per-version write-up |
| Scoreboard: on the cleaned labels | test_clean bf16 / Q6_K / Q4_K_M-imat, numbers_clean bf16 / Q6_K / Q4_K_M-imat | eval_hf.json + eval_gguf |
| Scoreboard: on the original labels | train rows, test bf16 / Q6_K, numbers bf16 / Q6_K, probes, note | eval_hf.json + eval_gguf |

## Per version

A `### vN: <what>` section: problem, data, training, before/after per split (bf16 and phone
file), error breakdown of the targeted class, probes, published repos, verdict.

## Detail tables

| section | table | source (experiment_rows.py block) |
|---|---|---|
| Training setup | `| version | epochs run | best epoch | best val loss | val loss by epoch | final train loss | train time |` | "Training setup" |
| Training data | `| model version | train rows | local train file | published dataset (HF) | eval splits in it |` | hand: file and dataset names |
| Training data | train rows by source: add a column for vN and a row for any new source; update the total | "sources:" line |
| Training data | train rows by language | "train rows by language" |
| Training data | transactions per message in train | "transactions per message" |
| Test set, bf16 | `| version | exact | F1 | count | amount | category | type | Banglish | Bangla | English | mixed |` for test and test_clean | "Test set, bf16" |
| Test set, bf16 | wrong messages by first failing field, test and test_clean | "wrong messages" |
| Numbers set, bf16 | same two tables for numbers and numbers_clean | "Numbers set, bf16" |
| Any extra eval set (e.g. Small-amounts set) | its own table: bf16 and phone file, plus the targeted error count | "<split>, bf16" + phone line |
| Phone files | file sizes line | GGUF sizes |
| Phone files | test exact by quant; test_clean exact by quant | "Phone files" test / test_clean lines |
| Phone files | numbers exact (BF16, F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M-imat); numbers_clean the same | "Phone files" numbers lines |
| Phone path vs full precision | numbers set by language, bf16 and each quant scored | last block |
| Speed on the Pixel 8 | only if speed was measured: alternating pairs, conditions | phone benchmark |
| Open problems | mark fixed problems as fixed (strike + "Fixed in vN"); add new ones | judgement |

Final check:

```bash
grep -n "^| v<previous>" ml/results/EXPERIMENTS.md | cut -c1-60
grep -n "^| vN" ml/results/EXPERIMENTS.md | cut -c1-60
```

Every table that appears in the first list has to appear in the second.
