#!/usr/bin/env bash
# Guideline v2 relabel (2026-10-21). The guideline changed (stated purpose decides family money,
# wages paid are salary expense, ...), so every labelled row is relabelled against it:
#   training rows: Claude Opus, 6 shards (all sources but the [] negatives); changes are accepted by
#     apply_relabel_v2.py only where Opus's amounts equal the current label's (Opus misreads some
#     Banglish number words, e.g. "pochash hajar" as 25,000).
#   eval rows (validation, test, numbers, small_amounts): Opus and Sonnet independently; the clean
#     sets keep the rows where both agree.
#   PY=~/miniconda3/envs/ML/bin/python ./ml/data/relabel_v2.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
R=ml/results/relabel_v2
mkdir -p "$R/shards"
TRAIN_SRC=(base hard codex-template-v2 codex-template-v2-bangla-words numwords-aug-v1 programmatic-lexicon-v1
           micro-sonnet handwritten-v1 handwritten-v2)

train_shard() {
  "$PY" ml/data/relabel.py --splits train_v12 --sources "${TRAIN_SRC[@]}" --shard "$1/6" \
    --labeller claude --model opus --tag opus-g2 --timeout 600 \
    --out "$R/shards/opus_train_$1.jsonl" > "$R/shards/opus_train_$1.log" 2>&1
}
eval_run() {  # model tag
  "$PY" ml/data/relabel.py --splits validation test numbers small_amounts \
    --labeller claude --model "$1" --tag "$2" --timeout 600 \
    --out "$R/$2_eval.jsonl" > "$R/shards/$2_eval.log" 2>&1
}

for k in 0 1 2 3 4 5; do train_shard "$k" & done
eval_run opus opus-g2 &
eval_run sonnet sonnet-g2 &
wait

cat "$R"/shards/opus_train_*.jsonl > "$R/opus_train.jsonl"
wc -l "$R/opus_train.jsonl" "$R/opus-g2_eval.jsonl" "$R/sonnet-g2_eval.jsonl"
