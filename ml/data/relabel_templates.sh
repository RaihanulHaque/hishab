#!/usr/bin/env bash
# v13: relabel the Codex-template and number-word training rows with two independent labellers,
# in parallel shards. Second labeller: Codex (OpenAI, low reasoning effort); Gemini through agy
# ran out of credits after 520 rows on 2026-10-21 (set SECOND=gemini to use it). On the 121-row
# pilot Codex agreed with Sonnet 93% of the time (Gemini: 92%) and with Gemini 98%.
#   PY=~/miniconda3/envs/ML/bin/python ./ml/data/relabel_templates.sh            # both labellers
#   ONLY=codex PY=... ./ml/data/relabel_templates.sh                             # just the second one
# Then: python ml/data/apply_relabel.py --amount-locked ... (see apply_relabel.py).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
SECOND=${SECOND:-codex}
ONLY=${ONLY:-}
R=ml/results/relabel
SRC=(codex-template-v2 codex-template-v2-bangla-words numwords-aug-v1)
mkdir -p "$R/shards"

# Codex in its own minimal home: the user's ~/.codex has git-based plugin marketplaces that every
# `codex exec` re-fetches (one per batch: ~25 fetches a minute over a phone hotspot) and plugin and
# skill instructions that nearly double the tokens per call. Login is linked from ~/.codex.
export CODEX_HOME=${CODEX_HOME:-$HOME/.codex-hishab-labeller}
if [ ! -f "$CODEX_HOME/config.toml" ]; then
  mkdir -p "$CODEX_HOME" && ln -sf "$HOME/.codex/auth.json" "$CODEX_HOME/auth.json"
  printf 'model = "gpt-6.1-sol"\nmodel_reasoning_effort = "low"\n\n[features]\njs_repl = false\n' > "$CODEX_HOME/config.toml"
fi

second() {  # shard k of n
  if [ "$SECOND" = gemini ]; then
    "$PY" ml/data/relabel.py --splits train_v12 --sources "${SRC[@]}" --shard "$1/$2" \
      --labeller agy --model gemini-3.8-flash-high --tag gemini-3.8-flash --timeout 600 \
      --skip-from "$R/pilot_gemini_tpl.jsonl" --out "$R/shards/gemini_tpl_$1.jsonl" > "$R/shards/gemini_$1.log" 2>&1
  else
    "$PY" ml/data/relabel.py --splits train_v12 --sources "${SRC[@]}" --shard "$1/$2" \
      --labeller codex --model default --tag codex --timeout 600 \
      --skip-from "$R/pilot_codex_tpl.jsonl" --out "$R/shards/codex_tpl_$1.jsonl" > "$R/shards/codex_$1.log" 2>&1
  fi
}
sonnet() {
  "$PY" ml/data/relabel.py --splits train_v12 --sources "${SRC[@]}" --shard "$1/$2" \
    --labeller claude --model sonnet --tag sonnet --timeout 600 \
    --skip-from "$R/pilot_sonnet_tpl.jsonl" --out "$R/shards/sonnet_tpl_$1.jsonl" > "$R/shards/sonnet_$1.log" 2>&1
}

if [ "$ONLY" != sonnet ]; then for k in 0 1 2 3 4 5; do second "$k" 6 & done; fi
if [ "$ONLY" != "$SECOND" ] && [ "$ONLY" != codex ] && [ "$ONLY" != gemini ]; then for k in 0 1 2 3; do sonnet "$k" 4 & done; fi
wait

cat "$R/pilot_$SECOND"_tpl.jsonl "$R"/shards/"$SECOND"_tpl_*.jsonl > "$R/${SECOND}_tpl.jsonl"
cat "$R/pilot_sonnet_tpl.jsonl" "$R"/shards/sonnet_tpl_*.jsonl > "$R/sonnet_tpl.jsonl"
wc -l "$R/${SECOND}_tpl.jsonl" "$R/sonnet_tpl.jsonl"
