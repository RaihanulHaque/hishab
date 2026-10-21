---
name: hishab-release
description: End-to-end workflow for a new Hishab model version - new training data, Modal fine-tuning, vocabulary trimming, GGUF build and eval, writing it up completely in ml/results/EXPERIMENTS.md, publishing model and dataset to Hugging Face, and switching the hishab_offline app (Pixel test, release APK). Use it whenever a Hishab model is trained, retrained, trimmed, pruned, quantized, evaluated, compared, published or shipped, whenever EXPERIMENTS.md needs a new version or experiment, or when the user says things like "train v13", "publish it", "switch the app to the new model", "add it to the experiments", "build the release" - even if they name only one of these steps.
---

# Hishab model release

Hishab fine-tunes Qwen3.5-0.8B to turn Banglish / Bangla / English money messages into
`description|-800|category` lines, trims its vocabulary, and ships a 4-bit GGUF in the offline
Flutter app `~/Code/hishab_offline` (runs through kiln on the phone). This skill is the release
path, in order. Each step says what "done" looks like; don't skip ahead.

## Ground rules

- **Environment:** every Python/Modal command uses `~/miniconda3/envs/ML/bin/{python,modal}` with
  `MODAL_PROFILE=cortex-technologies`. The Mac has no GPU; training and evals run on Modal.
- **Ask before GPU spend.** Training is ~1.5 to 2 h on one A100 (~$5). State the estimate and wait
  for a yes. CPU work (trim, GGUF build, GGUF eval, dataset/model push) needs no extra approval
  once the user asked for the release.
- **`git pull` first** in both repos: the user also commits from an office PC.
- **zsh does not word-split variables.** `G="--a --b"; cmd $G` passes one argument. Spell
  arguments out, or put multi-command runs in a bash script (as `ml/data/gen_micro.sh` does).
  macOS has no `timeout` command.
- **`modal volume get` into a directory that exists**, or the file is saved under the directory's
  name. The `aclose(): asynchronous generator` tracebacks after a Modal run are client shutdown
  noise; check the result lines, not the exit noise.
- **Never eyeball-copy numbers.** Every number in docs comes from an eval JSON on the volume.

## 1. Data (only if the version adds data)

- New rows come from templates with amount placeholders: `ml/data/generate_llm_templates.py`
  (`--backend claude --model sonnet`; it includes `docs/LABELLING_GUIDELINE.md` in the prompt for
  `--focus micro`; extend that for new focuses). Sonnet gives natural text; Haiku was noticeably
  weaker; Codex timed out at 15 min per batch. Pilot 2 batches, read 10 samples per language, and
  check the labels against the guideline before scaling up. Check that each language batch is
  really in that language.
- Amounts are filled by `ml/hishab/numwords.py` through `fill_templates.py` (labels exact by
  construction). Split **templates** before filling so eval and train never share a pattern
  (`ml/data/build_micro.py` is the model).
- Write `ml/datasets/splits/train_vN.jsonl` (+ any new eval split) and upload:
  `modal volume put --force hishab-data ml/datasets/splits/<file> /splits/<file>`.
  Register a new eval split in `ml/modal_app/hfeval.py` (`eval_splits`) and in the `SPLITS`
  list of `ml/modal_app/trim_vocab.py`.
- Score the **current app model** on any new eval split before training, so the write-up has a
  "before" number: `modal run ml/modal_app/gguf.py --runs <current trimmed run> --quants Q4_K_M-imat --split <new split>`.

## 2. Train (ask first)

```bash
modal run --detach ml/modal_app/train.py --models qwen3.5-0.8b --fmts pipe_signed --tune full \
    --epochs 3 --train-split train_vN --tag vN
```

Run name: `qwen3.5-0.8b__pipe_signed__full__vN`. It evaluates bf16 on every split at the end.
Done when the log shows `exact=` for every split.

## 3. Trim the vocabulary

Always after fine-tuning, before GGUF (the trim keeps every token the data uses, and the
quantization should see the final table):

```bash
modal run ml/modal_app/trim_vocab.py --run qwen3.5-0.8b__pipe_signed__full__vN --target 38000
```

Done when it prints `kept: 38000` and no "tokenize differently" error. Details: `ml/results/VOCAB_TRIM.md`.

## 4. GGUF + phone-file scores

One split per invocation; the first call builds F16, imatrix and the quants:

```bash
for s in test_clean_v2 numbers_clean_v2 small_amounts_v2 test_clean numbers_clean small_amounts test numbers; do
  modal run ml/modal_app/gguf.py --runs qwen3.5-0.8b__pipe_signed__full__vN__trim38k \
      --quants F16,Q6_K,Q4_K_M-imat,Q4_0-imat --split $s
done
```

Always build and score all four (the user expects Q6_K in every version's tables; the app ships
Q4_K_M-imat). The `_v2` sets follow labelling guideline v2 and are the headline from v13 on; the
older sets stay for comparison with older models.

## 5. Compare and probe

- Compare per split against the version in the app (same quant, same split). Treat moves of one
  to five messages as run-to-run churn; say so rather than calling them wins or losses.
- Break the misses down (count / amount / type / category); a "fix" is only a fix if the targeted
  error class is gone, e.g. "all 90 small amounts read right", not just a higher exact score.
- Probe on the Mac with the exact app prompt (download the GGUF first):
  `~/Code/kiln/build/host/kiln-cli -m <gguf> --raw -p "<chat-templated prompt>" -n 48`
  (build the prompt from `hishab.prompts.system_prompt('pipe_signed')` + Qwen chat markup with an
  empty `<think>` block). Probe the target case and its contrasts ("5 taka" and "5 hajar").

## 6. Results docs: registry + generated tables + story

The numbers are generated; only the story is written by hand. Nothing can be half-updated as long
as the version is in the registry.

1. **Register the version** in `ml/results/versions.yaml` (label, date, run, gguf_run, fmt,
   train_split, dataset, hub, one-line change, status). When it replaces the app model or becomes
   the best, update `app:` / `best:` at the top and the old entry's `status`.
2. **Refresh and render:**
   ```bash
   modal run ml/modal_app/collect_results.py      # reads every registered run's evals -> results_cache.json
   python ml/results/build_tables.py              # writes RESULTS_TABLES.md
   ```
   Check the new version appears in the status, leaderboard, per-quant, per-language, errors and
   training tables. A blank cell means an eval wasn't run: go back to step 4, don't type a number.
3. **Write the story** in `ml/results/EXPERIMENTS.md` under "Per version": problem, data, training,
   the before/after on the current eval sets (copied from RESULTS_TABLES.md), the targeted error
   class, probes, verdict. Update "Where things stand" and "Open problems". Hand measurements (phone
   speed, probes) go under "Hand measurements", since they are not in the eval files.
4. **New eval set?** Add it to `current_eval` or `legacy_eval` in `versions.yaml` and a one-line
   description to `SET_DOC` in `build_tables.py`; re-score the older versions on it (bf16:
   `train.py --eval-only <runs>`; GGUF: `gguf.py --runs <runs> --quants F16,Q6_K,Q4_K_M-imat,Q4_0-imat --split <set>`)
   so the leaderboard compares like with like.

Experiments that are not shipped (pruning, MTP, ...) are registered too, with `status: experiment`.

## 7. Publish

```bash
modal run ml/modal_app/gguf.py --runs <trimmed run> --only-push --push-only Q4_K_M-imat --dataset ryuma007/hishab-synthetic-vD
modal run ml/modal_app/push_model.py --run <trimmed run> --dataset ryuma007/hishab-synthetic-vD
```

- New training data means a new dataset version: add `HISTORY`, `MODELS`, `SOURCES` (and
  `SPLIT_DOC` for new splits) entries in `ml/modal_app/push_dataset.py`, then
  `modal run ml/modal_app/push_dataset.py --repo ryuma007/hishab-synthetic-vD --train-split train_vN`.
  The model cards must link the dataset the model was trained on.
- Verify before touching the app: `curl -sIL <resolve URL> | grep -i x-linked` and compare size and
  sha256 with the file you tested (`shasum -a 256`). Check split sizes with
  `https://datasets-server.huggingface.co/size?dataset=<repo>`.

## 8. Ship in the app (`~/Code/hishab_offline`)

1. In `lib/ai/model_manager.dart` set `kModelFile` (`hishab-qwen3.5-0.8b-pipe-vN-trim38k-Q4_K_M-imat.gguf`;
   the `hishab-qwen3.5-0.8b-` prefix lets start-up delete the old file), `kModelUrl`, `kModelBytes`
   and the doc comment. If the output format or system prompt changed, update `kSystemPrompt`
   (must match `ml/hishab/prompts.py` character for character) and `lib/ai/parse.dart`.
2. Update `README.md` (model paragraph, sideload commands) and bump `version:` in `pubspec.yaml`.
3. `flutter analyze`, `flutter test`, then
   `flutter build apk --release --target-platform android-arm64` (39 MB; without the flag it's 75 MB
   of unused architectures).
4. Install on the Pixel (`adb -s 38041FDJH00B57 install -r ...`), launch, tap **Download model** and
   wait for the `hishab:` log line; check the file size on the phone.
5. Send test messages **only through `scripts/app_messages.sh`**, which checks the Home screen
   before every tap and closes review sheets without saving. Read `references/pixel-testing.md`
   before touching the phone. The user's ledger is real data.
6. Speed numbers on the phone are only meaningful as alternating pairs; check
   `dumpsys thermalservice` and say when the phone was charging or throttled.

## 9. Finish

- Commit both repos (attribution lines as the system says), push, and update the project memory
  file for model versions.
- Report: what changed, before/after per split, the targeted error class, file size, Hub links,
  what the app now downloads, and anything not done.
