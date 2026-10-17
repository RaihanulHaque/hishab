# Relabelling the held-out sets (2026-10-17)

Two frontier models labelled every `test` (518) and `numbers` (357) row from scratch, with
`docs/LABELLING_GUIDELINE.md` in the prompt and without seeing the existing gold:
Gemini 3.8 Flash (through the `agy` CLI, `ml/data/relabel.py`) and Claude Sonnet 5.5 (Claude
Code subagents on the batch files the same script emits). `ml/data/relabel_merge.py` keeps the
rows where the two agree on the (amount, category, type) multiset; those are
`ml/datasets/splits_clean/{test,numbers}.jsonl` (also uploaded as `/splits/test_clean.jsonl` and
`/splits/numbers_clean.jsonl`, which `train.py` now evaluates on automatically).

| split | rows | labellers agree | of which old gold agrees too | both overrule old gold | labellers disagree |
|---|---|---|---|---|---|
| test | 518 | 476 (91.9%) | 404 | **72 (13.9%)**: category 47, count 24, type 1 | 42 (count 21, category 20, amount 1) |
| numbers | 357 | 317 (88.8%) | 265 | **52 (14.6%)**: all category | 40 (category 20, amount 15, count 2, type 3) |

Each labeller alone matches the old gold on about 81% of test rows (Gemini 81.7, Sonnet 81.3),
which is the same level the models reach: the models learned the old gold's conventions.

## What the overrules are

Typical corrections, all of them rules in the guideline that the Gemini gold did not follow:
plans are not transactions ("Set aside 5k for grocery" has no entry); photocopy, notes and
study books are `tuition`; business profit, freelance and tutoring income are `salary`; a
flower bought as a present is `shopping`, money given is `gift`; outings "যাতায়াত ও খাবার মোট"
with one total are `other`; rows whose gold had `amount: 0.0` get no entry.

On the `numbers` split every overrule is a category; the labellers never agree on an amount
that differs from the template gold, so the template amounts stand. Individually they do misread
Banglish number words ("share" for *sare* = and a half, "shaat" = sixty, "pochanno" = 55), which
is why single-labeller amounts must not be trusted on that split.

## What the models score on the clean gold

| model | old gold, all rows | old gold, agreed rows | **clean gold, agreed rows** | on the overruled rows (old → clean) |
|---|---|---|---|---|
| v8, test | 89.6 | 91.2 | **84.5** | 63.9 → 19.4 |
| v9, test | 89.2 | 90.1 | **83.6** | 62.5 → 19.4 |
| v8, numbers | 84.9 | 87.4 | **77.9** | 65.4 → 7.7 |
| v9, numbers | 83.2 | 87.1 | **79.5** | 61.5 → 15.4 |
| v10, test | 80.3 | | **92.2** | 13.9 → 73.6 |
| v10, numbers | 85.2 | | **78.9** | 63.5 → 11.5 |

Read: on the 72 test rows where both labellers say the gold is wrong, the model agrees with the
old gold about 63% of the time and with the guideline about 19%. It learned the noise. The
4,409 Gemini-written training rows (`source` = base / hard) carry the same conventions, so the
fix is to relabel them the same way and retrain; that is v10. The Codex-template rows (7,328 +
1,447), the programmatic rows and the hand-written rows were assumed to follow the guideline.

**v10 result (2026-10-17):** the train relabel (`apply_relabel.py`: 3,285 kept, 848 relabelled, 276 dropped)
moved test_clean from 83.6 to 92.2, and on the 72 overruled test rows the model now follows the
guideline 73.6% of the time instead of 19.4%. numbers_clean did not move: its 52 overrules are all
categories of Codex-template rows (tutoring income as `tuition`, sales income as the goods'
category, presents as `gift`), and the template train rows carry the same conventions. The
assumption above was wrong for income and gift categories; relabelling the template rows is next.

## Ceiling

If v10 fixes only the convention rows, the clean test score would be roughly
(404 + 72 × 0.9) / 476 ≈ 98%, with the remaining errors the genuine ones (amount parsing,
dropped items, rare categories). The 42 + 40 rows where even two frontier models disagree are
kept out of the clean sets (`splits_clean/disagreements.jsonl`); they are candidates for human
judgement, and a model cannot be blamed for them.
