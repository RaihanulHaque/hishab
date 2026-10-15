"""SFT one or more candidate models on Modal with TRL, then evaluate on validation + test.

  conda run -n ML modal run ml/modal_app/train.py \
      --models qwen3.5-0.8b,lfm2.5-350m --fmts json,two_phase [--epochs 5] [--tune lora|full]

Each (model, fmt, tune) runs in its own GPU container in parallel. Outputs per run under the
`hishab-runs` volume at /runs/<run_name>/:
  adapter/   LoRA weights (lora only)
  merged/    full HF model + tokenizer (what the GGUF step converts)
  eval_hf.json          metrics on validation and test (bf16, greedy)
  preds_{split}.jsonl   raw predictions for inspection
  train_log.json        trainer history
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ml/ for local imports
from modal_app.common import DATA_DIR, RUNS_DIR, VOLUMES, app, hf_secret, train_image  # noqa: E402

with train_image.imports():
    import torch

MAX_NEW_TOKENS = {"json": 192, "two_phase": 320}


def load_split(name: str) -> list[dict]:
    p = Path(DATA_DIR) / "splits" / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def to_prompt_completion(rows: list[dict], fmt: str) -> list[dict]:
    from hishab.prompts import build_messages
    from hishab.schema import render_target

    return [
        {
            "prompt": build_messages(fmt, r["instruction"]),
            "completion": [
                {"role": "assistant", "content": render_target(fmt, r["translation"], r["transactions"])}
            ],
        }
        for r in rows
    ]


def check_template(tokenizer, example: dict) -> dict:
    """The generation prompt must be a prefix of the fully rendered conversation, otherwise
    what we train on and what llama.cpp feeds the model at inference differ."""
    full = tokenizer.apply_chat_template(example["prompt"] + example["completion"], tokenize=False)
    gen = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True)
    ok = full.startswith(gen)
    print("=== chat template check:", "OK" if ok else "MISMATCH")
    print(repr(gen[-120:]))
    print(repr(full[len(gen) - 40 :][:220]))
    if not ok:
        raise RuntimeError("generation prompt is not a prefix of the training render")
    return {"generation_prompt_tail": gen[-80:], "completion_head": full[len(gen):][:80]}


@torch.inference_mode()
def generate_all(model, tokenizer, rows: list[dict], fmt: str, batch_size: int = 32) -> tuple[list[str], float]:
    from hishab.prompts import build_messages

    tokenizer.padding_side = "left"
    preds: list[str] = []
    n_tokens = 0
    t0 = time.time()
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        texts = [
            tokenizer.apply_chat_template(build_messages(fmt, r["instruction"]), tokenize=False, add_generation_prompt=True)
            for r in batch
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS[fmt],
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1] :]
        for g in gen:
            n_tokens += int((g != tokenizer.pad_token_id).sum())
        preds += [t.strip() for t in tokenizer.batch_decode(gen, skip_special_tokens=True)]
        print(f"  generated {min(i + batch_size, len(rows))}/{len(rows)}", flush=True)
    return preds, n_tokens / max(len(rows), 1)


@app.function(
    image=train_image,
    gpu="A100-80GB",
    timeout=4 * 60 * 60,
    volumes=VOLUMES,
    secrets=[hf_secret],
)
def train(
    model_key: str,
    fmt: str,
    tune: str = "lora",
    epochs: float = 5,
    lr: float | None = None,
    lora_r: int = 16,
    max_length: int = 512,
    seed: int = 42,
    max_steps: int = -1,
    eval_limit: int = 0,
    train_limit: int = 0,
) -> dict:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from hishab.metrics import evaluate
    from hishab.registry import MODELS, run_name

    spec = MODELS[model_key]
    name = run_name(model_key, fmt, tune)
    out_dir = Path(RUNS_DIR) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    lr = lr if lr is not None else (2e-4 if tune == "lora" else 3e-5)
    print(f"### {name}: {spec['hf_id']} fmt={fmt} tune={tune} epochs={epochs} lr={lr}")

    train_rows, val_rows, test_rows = load_split("train"), load_split("validation"), load_split("test")
    if train_limit:
        train_rows = train_rows[:train_limit]
    if eval_limit:
        val_rows, test_rows = val_rows[:eval_limit], test_rows[:eval_limit]
    train_ds = Dataset.from_list(to_prompt_completion(train_rows, fmt))
    val_ds = Dataset.from_list(to_prompt_completion(val_rows, fmt))

    tokenizer = AutoTokenizer.from_pretrained(spec["hf_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    template_info = check_template(tokenizer, train_ds[0])

    model = AutoModelForCausalLM.from_pretrained(spec["hf_id"], dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    print("model class:", type(model).__name__, "params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    peft_config = None
    if tune == "lora":
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=2 * lora_r,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=spec["lora_targets"],
        )

    bs = spec["batch_size"]
    args = SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        gradient_accumulation_steps=spec.get("grad_accum", 1),
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.0,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_length=max_length,
        packing=False,
        completion_only_loss=True,
        dataloader_num_workers=2,
        report_to="none",
        seed=seed,
        gradient_checkpointing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    (out_dir / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1))

    merged_dir = out_dir / "merged"
    if tune == "lora":
        trainer.model.save_pretrained(str(out_dir / "adapter"))
        model = trainer.model.merge_and_unload()
    else:
        model = trainer.model
    model.config.use_cache = True
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    del trainer
    torch.cuda.empty_cache()

    model.eval()
    results = {
        "run": name,
        "model_key": model_key,
        "hf_id": spec["hf_id"],
        "fmt": fmt,
        "tune": tune,
        "epochs": epochs,
        "lr": lr,
        "lora_r": lora_r if tune == "lora" else None,
        "train_rows": len(train_rows),
        "train_time_s": round(train_time),
        "best_eval_loss": min((h["eval_loss"] for h in trainer_state_eval(out_dir)), default=None),
        "template": template_info,
        "splits": {},
    }
    run_eval(model, tokenizer, name, fmt, out_dir, results, {"validation": val_rows, "test": test_rows},
             spec.get("eval_batch_size", 32))
    VOLUMES[RUNS_DIR].commit()
    return results


def run_eval(model, tokenizer, name, fmt, out_dir, results, splits, batch_size):
    from hishab.metrics import evaluate

    for split, rows in splits.items():
        print(f"--- generating on {split} ({len(rows)} rows, batch {batch_size})")
        preds, avg_tok = generate_all(model, tokenizer, rows, fmt, batch_size=batch_size)
        metrics = evaluate(rows, preds)
        metrics["avg_completion_tokens"] = round(avg_tok, 1)
        results["splits"][split] = metrics
        with (out_dir / f"preds_{split}.jsonl").open("w", encoding="utf-8") as f:
            for r, p in zip(rows, preds):
                f.write(json.dumps({"id": r["id"], "language": r["language"], "instruction": r["instruction"],
                                    "gold": r["transactions"], "pred": p}, ensure_ascii=False) + "\n")
        print(f"{name} {split}: exact={metrics['all']['exact_match']:.3f} f1={metrics['all']['tx_f1']:.3f} "
              f"valid={metrics['all']['valid_json']:.3f} avg_tok={avg_tok:.0f}")
    (out_dir / "eval_hf.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))


@app.function(image=train_image, gpu="A10G", timeout=2 * 60 * 60, volumes=VOLUMES, secrets=[hf_secret])
def eval_hf(run: str, eval_limit: int = 0) -> dict:
    """Re-score an existing merged run (bf16, greedy) without retraining."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hishab.registry import MODELS

    out_dir = Path(RUNS_DIR) / run
    results = json.loads((out_dir / "eval_hf.json").read_text())
    spec = MODELS[results["model_key"]]
    tokenizer = AutoTokenizer.from_pretrained(str(out_dir / "merged"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(out_dir / "merged"), dtype=torch.bfloat16, device_map="cuda").eval()
    val_rows, test_rows = load_split("validation"), load_split("test")
    if eval_limit:
        val_rows, test_rows = val_rows[:eval_limit], test_rows[:eval_limit]
    run_eval(model, tokenizer, run, results["fmt"], out_dir, results, {"validation": val_rows, "test": test_rows},
             spec.get("eval_batch_size", 32))
    VOLUMES[RUNS_DIR].commit()
    return results


def trainer_state_eval(out_dir: Path) -> list[dict]:
    log = json.loads((out_dir / "train_log.json").read_text())
    return [h for h in log if "eval_loss" in h]


@app.local_entrypoint()
def main(
    models: str = "qwen3.5-0.8b,lfm2.5-350m",
    fmts: str = "json,two_phase",
    tune: str = "lora",
    epochs: float = 5,
    lr: float = None,
    lora_r: int = 16,
    max_steps: int = -1,
    eval_limit: int = 0,
    train_limit: int = 0,
    eval_only: str = "",
):
    if eval_only:
        for r in eval_hf.map([x.strip() for x in eval_only.split(",") if x.strip()], kwargs={"eval_limit": eval_limit}):
            t = r["splits"]["test"]["all"]
            print(f"{r['run']} | test exact {t['exact_match']:.3f} | F1 {t['tx_f1']:.3f} | valid {t['valid_json']:.3f}")
        return
    jobs = [(m.strip(), f.strip()) for m in models.split(",") for f in fmts.split(",") if m.strip() and f.strip()]
    print(f"launching {len(jobs)} runs: {jobs}")
    handles = {
        (m, f): train.spawn(m, f, tune=tune, epochs=epochs, lr=lr, lora_r=lora_r, max_steps=max_steps,
                            eval_limit=eval_limit, train_limit=train_limit)
        for m, f in jobs
    }
    summary = []
    for (m, f), h in handles.items():
        try:
            r = h.get()
            t = r["splits"]["test"]["all"]
            summary.append((r["run"], t["exact_match"], t["tx_f1"], t["valid_json"], r["splits"]["test"]["avg_completion_tokens"]))
        except Exception as e:  # keep collecting the others
            summary.append((f"{m}__{f}__{tune}", None, None, None, str(e)[:200]))
    print("\nrun | test exact | test F1 | valid JSON | avg completion tokens")
    for row in summary:
        print(" | ".join(str(x if not isinstance(x, float) else round(x, 3)) for x in row))
