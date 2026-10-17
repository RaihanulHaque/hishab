"""GRPO on top of an SFT run, with the exact-match metric as a verifiable reward.

  conda run -n ML modal run ml/modal_app/train_grpo.py --base qwen3.5-0.8b__json__full__v8 \
      [--steps 300] [--num-generations 8] [--lr 1e-6] [--tag grpo1] [--train-limit 4000]
      [--smoke]   # 10 steps, 48 eval rows

Starts from /runs/<base>/merged, samples `num_generations` completions per prompt from training
rows, and rewards each one with:
    -1.0          output is not a parseable JSON array of well-formed transactions
     tuple F1     over (amount, category, type) against gold (0..1)
    +1.0          bonus when the multiset matches exactly
    -0.5 * |len(pred) - len(gold)|   dropped or invented entries, capped at -1
so the reward is about the thing we ship, not the token likelihood. Writes the result as a new
run /runs/<base>__<tag>/ with merged/, eval_hf.json and preds_*.jsonl, same layout as train.py,
so gguf.py and report.py work on it unchanged.

The SFT model has memorised the train split (it gets 99% of train prompts right), so GRPO
samples prompts from a fresh pool instead: /splits/grpo_prompts.jsonl, made by
`ml/data/generate_banglish.py --seed 99` (labels exact by construction, never used for SFT).
Rows where the model is already always right carry no learning signal in GRPO (all rewards
equal), so --hard-only first scores the base model once per prompt and keeps only the prompts
it got wrong plus a random quarter of the rest.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modal_app.common import RUNS_DIR, VOLUMES, app, hf_secret, train_image  # noqa: E402

with train_image.imports():
    import torch


def _key(txs) -> list:
    return sorted((round(float(t["amount"]), 2), t["category"], t["type"]) for t in txs)


def reward_fn(completions, gold, **kwargs) -> list[float]:
    """TRL passes every extra dataset column as a kwarg (here `gold`, a JSON string per row)."""
    from hishab.schema import parse_output

    out = []
    for comp, g in zip(completions, gold):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        parsed = parse_output(text)
        if parsed["txs"] is None or not parsed["strict"]:
            out.append(-1.0)
            continue
        pred, gk = _key(parsed["txs"]), _key(json.loads(g))
        from collections import Counter

        pc, gc = Counter(pred), Counter(gk)
        tp = sum((pc & gc).values())
        prec = tp / max(len(pred), 1)
        rec = tp / max(len(gk), 1)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        r = f1 + (1.0 if pc == gc else 0.0) - min(1.0, 0.5 * abs(len(pred) - len(gk)))
        if not gk and not pred:  # correct empty answer
            r = 2.0
        out.append(r)
    return out


@app.function(image=train_image, gpu="A100-80GB", timeout=6 * 60 * 60, volumes=VOLUMES, secrets=[hf_secret])
def train_grpo(base: str, tag: str, steps: int, num_generations: int, lr: float, train_limit: int,
               hard_only: bool, eval_limit: int, seed: int, beta: float, batch_prompts: int, prompt_split: str) -> dict:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from hishab.prompts import build_messages
    from hishab.schema import parse_output
    # imported here, not at module level: train.py registers its own local_entrypoint on the shared app
    from modal_app.hfeval import eval_splits, generate_all, load_split, load_tokenizer, run_eval

    base_dir = Path(RUNS_DIR) / base
    meta = json.loads((base_dir / "eval_hf.json").read_text())
    fmt = meta["fmt"]
    name = f"{base}__{tag}"
    out_dir = Path(RUNS_DIR) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"### {name}: GRPO from {base_dir / 'merged'} fmt={fmt} steps={steps} G={num_generations} lr={lr}")

    rng = random.Random(seed)
    rows = load_split(prompt_split)
    rng.shuffle(rows)
    if train_limit:
        rows = rows[:train_limit]

    tokenizer = load_tokenizer(str(base_dir / "merged"))
    model = AutoModelForCausalLM.from_pretrained(str(base_dir / "merged"), dtype=torch.bfloat16, device_map="cuda")

    if hard_only:
        # one greedy pass with the base model; keep wrong prompts + 25% of right ones
        preds, _ = generate_all(model, tokenizer, rows, fmt, batch_size=32)
        keep = []
        for r, p in zip(rows, preds):
            t = parse_output(p)["txs"]
            wrong = t is None or _key(t) != _key(r["transactions"])
            if wrong or rng.random() < 0.25:
                keep.append(r)
        print(f"hard-only: {len(keep)} of {len(rows)} prompts kept ({sum(1 for r, p in zip(rows, preds) if parse_output(p)['txs'] is None or _key(parse_output(p)['txs']) != _key(r['transactions']))} wrong)")
        rows = keep
        model.train()

    ds = Dataset.from_list([{"prompt": build_messages(fmt, r["instruction"]), "gold": json.dumps(r["transactions"])} for r in rows])

    cfg = GRPOConfig(
        output_dir=str(out_dir / "checkpoints"),
        max_steps=steps,
        learning_rate=lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=max(1, steps // 20),
        per_device_train_batch_size=num_generations,     # one prompt's generations per micro-batch
        gradient_accumulation_steps=batch_prompts,       # prompts per optimizer step
        num_generations=num_generations,
        max_completion_length=192 if fmt == "json" else 320,
        temperature=1.0,
        beta=beta,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=seed,
        gradient_checkpointing=True,
    )
    trainer = GRPOTrainer(model=model, reward_funcs=reward_fn, args=cfg, train_dataset=ds, processing_class=tokenizer)
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    log = trainer.state.log_history
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=1))
    rewards = [h["reward"] for h in log if "reward" in h]
    print(f"reward first/last: {rewards[:2]} -> {rewards[-2:]}")

    merged = out_dir / "merged"
    trainer.model.save_pretrained(str(merged), safe_serialization=True)
    tokenizer.save_pretrained(str(merged))
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):  # keep the base's exact files
        src = base_dir / "merged" / f
        if src.exists():
            (merged / f).write_bytes(src.read_bytes())

    model = trainer.model
    model.eval()
    model.config.use_cache = True
    results = {**{k: meta[k] for k in ("model_key", "hf_id", "fmt", "tune", "epochs", "lr") if k in meta},
               "run": name, "tag": tag, "grpo": {"base": base, "steps": steps, "num_generations": num_generations, "lr": lr,
                                                 "beta": beta, "prompts": len(rows), "hard_only": hard_only,
                                                 "reward_first": rewards[:1], "reward_last": rewards[-1:]},
               "train_rows": len(rows), "train_time_s": round(train_time), "template": meta.get("template", {}), "splits": {}}
    run_eval(model, tokenizer, name, fmt, out_dir, results, eval_splits(eval_limit), batch_size=32)
    VOLUMES[RUNS_DIR].commit()
    return {k: v for k, v in results.items() if k != "template"}


@app.local_entrypoint()
def main(base: str = "qwen3.5-0.8b__json__full__v8", tag: str = "grpo1", steps: int = 300, num_generations: int = 8,
         lr: float = 1e-6, train_limit: int = 6000, hard_only: bool = True, eval_limit: int = 0, seed: int = 42,
         beta: float = 0.04, batch_prompts: int = 8, smoke: bool = False, prompt_split: str = "grpo_prompts"):
    if smoke:
        steps, train_limit, eval_limit = 10, 200, 48
    r = train_grpo.remote(base, tag, steps, num_generations, lr, train_limit, hard_only, eval_limit, seed, beta, batch_prompts, prompt_split)
    for split, s in r["splits"].items():
        a = s["all"]
        print(f"{r['run']} {split}: exact={a['exact_match']:.3f} f1={a['tx_f1']:.3f} valid={a['valid_json']:.3f}")
