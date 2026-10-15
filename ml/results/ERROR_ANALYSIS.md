# Error analysis: Qwen3.5-0.8B json, v3 vs v4 (test split, 518 rows)

Source: `preds_test.jsonl` of `qwen3.5-0.8b__json__lora` (v3, 4409 train rows) and
`qwen3.5-0.8b__json__lora__v4` (9522 train rows: v3 + 5000 programmatic + 113 hand-written).
A row is an error when the multiset of `(amount, category, type)` differs from gold.

| | v3 | v4 |
|---|---|---|
| test exact | 87.6 | 87.8 |
| test F1 | .929 | .934 |
| errors | 64 | 63 |
| shared errors | 49 | 49 |
| fixed by v4 | | 15 |
| newly broken by v4 | | 14 |
| banglish / bangla / english / mixed | 81.7 / 90.4 / 93.9 / 92.0 | 82.6 / 88.0 / 94.9 / 93.3 |

Doubling the training data moved the test score by 0.2 points. The 63 remaining errors break
down as: category 41, transaction count 9, income/expense type 8, amount 5.

## What the remaining errors actually are

Most are **label-convention disagreements**, not extraction failures. The Gemini-generated gold
is itself inconsistent on these, so the model cannot learn a rule and the test set cannot
reward one:

| ambiguity | examples from the error list | gold says |
|---|---|---|
| education side-costs (books, photocopy, printout, form fill-up, school trip) | "photocopy 500 tk" vs "photocopy 400 tk" in two different rows | `other` in one, `tuition` in the other |
| freelance / business / tuition-giving income | "Freelance kajer taka peyechi 18000" | `salary`; model says `other` |
| money sent by family, pocket money, salami | "bhaiya 500 taka pathaise pocket money" | `gift`; model says `other`; another gold row has pocket money as `other` |
| food vs grocery vs entertainment | fruit+sweets 250 (gold `grocery`), adda 800 (gold `food`), bbq party (gold `food`), popcorn (gold `entertainment` once, `food` once) | mixed |
| totals in the message | "Eklei 1700 tk" | gold drops the total; model sometimes emits it |
| direction of a loan | "loan শোধ করলো" (friend repaid me) | `income`; model says `expense` |
| rows whose gold invents an amount | "Paid for Scrivener subscription" gold 4500, "Monthly fee for Blinkist" gold 500 | not recoverable from the text |
| rows with amount 0.0 (7 in test, 43 in train) | "ধার নিয়েছি" with no number | gold emits `amount: 0.0`; model emits `[]` |

Genuine model errors, which the v5 data targets:

- **Bangla number words**: "পনেরো হাজার" read as 7000 (twice). The generator now emits
  spelled-out numbers (`ponero hajar`, `আড়াই হাজার`, `der lakh`, `pachsho`).
- **medicine bought after a doctor visit** labelled `grocery` (3 rows) → hand-written rows.
- **data / internet pack** labelled `utilities` instead of `mobile_recharge` → hand-written rows.
- **a transaction dropped when the sentence opens with "faltu khoroch"** (1 row).

## Implications

1. The 518-row test set has a label-noise ceiling around 90%; going from 87.8 to "very good"
   needs a relabelled test set with a written labelling guideline, not more model capacity.
   Suggested conventions (used in the hand-written rows): education side-costs → `tuition`;
   freelance/business/tutoring earnings → `salary`; money received from family or as salami →
   `gift` income; eating out of any kind → `food`; cinema/concert/game → `entertainment`;
   never emit a stated total as its own transaction.
2. Real user messages are the next test set. Synthetic Gemini text is cleaner than what people
   type into the app, and the model's weakest slice (Banglish, 82.6%) is exactly the one people
   will use most.
3. Because the task has a cheap verifiable reward (`ml/hishab/metrics.py`), GRPO or
   rejection-sampling fine-tuning can still help once the labels are consistent; before that it
   would just learn the noise.
