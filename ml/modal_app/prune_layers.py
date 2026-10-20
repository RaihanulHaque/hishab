"""Depth pruning for the Hishab Qwen3.5 model: find the layers the task needs least, remove them,
and heal the rest against the unpruned model.

  # 1. which layers matter? (skip each one, measure agreement with the full model on test_clean)
  conda run -n ML modal run ml/modal_app/prune_layers.py::probe --run qwen3.5-0.8b__pipe_signed__full__v11__trim38k
  # 2. remove 4 linear-attention layers and heal (KL to the full model, completion tokens only)
  conda run -n ML modal run ml/modal_app/prune_layers.py::prune --run <run> --drop 7,11,13,17 --epochs 2

Writes /runs/<run>__L<n_layers> with the healed merged model and an eval_hf.json (bf16 greedy on
every split, like train.py). Qwen3.5 is hybrid (linear attention + full attention); the full
attention layers do the copying that extraction lives on, so only linear layers are candidates.
llama.cpp reads the per-layer kind from the GGUF (`attention.recurrent_layers`), so any pattern is
fine on the phone.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modal_app.common import DATA_DIR, RUNS_DIR, VOLUMES, app, hf_secret, train_image  # noqa: E402

with train_image.imports():
    import torch
    import torch.nn.functional as F


def _rows(name: str) -> list[dict]:
    p = Path(DATA_DIR) / "splits" / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _encoder(tok, fmt):
    from hishab.prompts import build_messages
    from hishab.schema import render_target

    def encode(r: dict) -> tuple[list[int], int]:
        msgs = build_messages(fmt, r["instruction"])
        full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": render_target(fmt, r.get("translation"), r["transactions"])}], tokenize=False)
        gen = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        assert full.startswith(gen)
        ids = tok(full, add_special_tokens=False)["input_ids"]
        return ids, len(tok(gen, add_special_tokens=False)["input_ids"])

    return encode


def _batches(enc: list[tuple[list[int], int]], pad: int, bs: int, order: list[int]):
    for i in range(0, len(order), bs):
        chunk = [enc[j] for j in order[i : i + bs]]
        T = max(len(x) for x, _ in chunk)
        ids = torch.full((len(chunk), T), pad, dtype=torch.long)
        att = torch.zeros((len(chunk), T), dtype=torch.long)
        lm = torch.zeros((len(chunk), T - 1), dtype=torch.bool)  # logits at t predict x_{t+1}
        for b, (x, n_prompt) in enumerate(chunk):
            ids[b, : len(x)] = torch.tensor(x)
            att[b, : len(x)] = 1
            lm[b, n_prompt - 1 : len(x) - 1] = True  # completion tokens only
        yield ids.cuda(), att.cuda(), lm.cuda()


@app.function(image=train_image, gpu="A10G", timeout=60 * 60, volumes=VOLUMES, secrets=[hf_secret])
def probe_remote(run: str) -> dict:
    """For each layer: top-1 agreement with the full model on completion tokens when that layer is
    skipped (identity), plus the mean cosine distance between its input and output."""
    from transformers import AutoModelForCausalLM

    from modal_app.hfeval import load_tokenizer

    src = Path(RUNS_DIR) / run / "merged"
    meta = json.loads((Path(RUNS_DIR) / run / "eval_hf.json").read_text())
    tok = load_tokenizer(str(src))
    model = AutoModelForCausalLM.from_pretrained(str(src), dtype=torch.bfloat16, device_map="cuda").eval()
    layers = model.model.layers
    kinds = model.config.layer_types
    enc = [_encoder(tok, meta["fmt"])(r) for r in _rows("test_clean")]
    order = list(range(len(enc)))

    dist = [0.0] * len(layers)
    n_dist = 0
    hooks = []
    for i, layer in enumerate(layers):
        def hook(mod, args, out, i=i):
            x, y = args[0].float(), (out[0] if isinstance(out, tuple) else out).float()
            dist[i] += (1 - F.cosine_similarity(x, y, dim=-1)).mean().item()
        hooks.append(layer.register_forward_hook(hook))
    full = []
    with torch.no_grad():
        for ids, att, lm in _batches(enc, tok.pad_token_id, 16, order):
            full.append(model(input_ids=ids, attention_mask=att).logits[:, :-1].argmax(-1)[lm].cpu())
            n_dist += 1
    for h in hooks:
        h.remove()
    full_cat = torch.cat(full)

    agree = []
    with torch.no_grad():
        for i, layer in enumerate(layers):
            fwd = layer.forward
            layer.forward = lambda hidden_states, *a, **k: hidden_states
            preds = []
            for ids, att, lm in _batches(enc, tok.pad_token_id, 16, order):
                preds.append(model(input_ids=ids, attention_mask=att).logits[:, :-1].argmax(-1)[lm].cpu())
            layer.forward = fwd
            a = (torch.cat(preds) == full_cat).float().mean().item()
            agree.append(a)
            print(f"layer {i:2d} {kinds[i]:17s} skip-agreement {a:.4f}  in/out cosine distance {dist[i] / n_dist:.4f}", flush=True)
    out = {"run": run, "layer_types": kinds, "skip_agreement": agree, "cos_distance": [d / n_dist for d in dist]}
    (Path(RUNS_DIR) / run / "layer_probe.json").write_text(json.dumps(out, indent=1))
    VOLUMES[RUNS_DIR].commit()
    return out


@app.function(image=train_image, gpu="A100-80GB", timeout=4 * 60 * 60, volumes=VOLUMES, secrets=[hf_secret])
def prune_remote(run: str, drop: list[int], epochs: float = 2, lr: float = 2e-5, batch: int = 16,
                 train_split: str = "train_v10", limit: int = 0, ce_weight: float = 0.2) -> dict:
    import shutil

    from transformers import AutoModelForCausalLM

    from hishab.registry import MODELS
    from modal_app.hfeval import eval_splits, load_tokenizer, run_eval

    src = Path(RUNS_DIR) / run
    meta = json.loads((src / "eval_hf.json").read_text())
    fmt = meta["fmt"]
    tok = load_tokenizer(str(src / "merged"))
    teacher = AutoModelForCausalLM.from_pretrained(str(src / "merged"), dtype=torch.bfloat16, device_map="cuda").eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    kinds = teacher.config.layer_types
    for d in drop:
        if kinds[d] != "linear_attention":
            raise ValueError(f"layer {d} is {kinds[d]}; only linear_attention layers are dropped")

    # student: a copy of the teacher with the dropped layers removed and the rest renumbered
    student = AutoModelForCausalLM.from_pretrained(str(src / "merged"), dtype=torch.bfloat16, device_map="cuda")
    keep = [i for i in range(len(kinds)) if i not in set(drop)]
    tl = student.model.layers
    student.model.layers = torch.nn.ModuleList([tl[i] for i in keep])
    for new_i, layer in enumerate(student.model.layers):
        for mod in layer.modules():
            if hasattr(mod, "layer_idx"):
                mod.layer_idx = new_i
    cfg = student.config
    cfg.num_hidden_layers = len(keep)
    cfg.layer_types = [kinds[i] for i in keep]
    # full_attention_interval stays as it is: the GGUF converter writes it, llama.cpp ignores it when layer_types is present
    # transformers' cache mask uses config.layer_types too
    student.model.config = cfg
    student.train()
    for p in student.parameters():
        p.requires_grad_(True)
    n_params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"student: {len(keep)} layers, {n_params:.0f}M params; dropped {drop}", flush=True)

    enc_fn = _encoder(tok, fmt)
    train_rows = _rows(train_split)
    if limit:
        train_rows = train_rows[:limit]
    enc = [enc_fn(r) for r in train_rows]
    steps = int(epochs * ((len(enc) + batch - 1) // batch))
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05, anneal_strategy="cos")

    def agreement(rows):
        e = [enc_fn(r) for r in rows]
        hit = tot = 0
        student.eval()
        with torch.no_grad():
            for ids, att, lm in _batches(e, tok.pad_token_id, 16, list(range(len(e)))):
                a = student(input_ids=ids, attention_mask=att).logits[:, :-1].argmax(-1)
                b = teacher(input_ids=ids, attention_mask=att).logits[:, :-1].argmax(-1)
                hit += ((a == b) & lm).sum().item()
                tot += lm.sum().item()
        student.train()
        return hit / max(tot, 1)

    tc = _rows("test_clean")
    before = agreement(tc)
    print(f"agreement with teacher before healing: {before:.4f}", flush=True)

    import random
    step, t0, log, ep = 0, time.time(), [], 0
    while step < steps:
        order = list(range(len(enc)))
        random.Random(ep).shuffle(order)
        for ids, att, lm in _batches(enc, tok.pad_token_id, batch, order):
            if step >= steps:
                break
            with torch.no_grad():
                t_logits = teacher(input_ids=ids, attention_mask=att).logits[:, :-1][lm].float()
            s_logits = student(input_ids=ids, attention_mask=att).logits[:, :-1][lm].float()
            kl = F.kl_div(F.log_softmax(s_logits, -1), F.log_softmax(t_logits, -1), log_target=True, reduction="batchmean")
            ce = F.cross_entropy(s_logits, ids[:, 1:][lm])
            loss = kl + ce_weight * ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0 or step == steps:
                log.append({"step": step, "kl": round(kl.item(), 4), "ce": round(ce.item(), 4)})
                print(f"step {step}/{steps} kl {kl.item():.4f} ce {ce.item():.4f} {time.time() - t0:.0f}s", flush=True)
        ep += 1
    train_time = time.time() - t0
    after = agreement(tc)
    print(f"agreement with teacher after healing: {after:.4f}", flush=True)

    name = f"{run}__L{len(keep)}"
    dst = Path(RUNS_DIR) / name
    if dst.exists():
        shutil.rmtree(dst)
    merged = dst / "merged"
    student.eval()
    student.config.use_cache = True
    student.save_pretrained(str(merged), safe_serialization=True)
    for f in (src / "merged").iterdir():  # tokenizer files (trimmed vocab) as they are
        if f.suffix != ".safetensors" and f.name not in ("config.json", "generation_config.json", "model.safetensors.index.json"):
            shutil.copy(f, merged / f.name)
    del teacher
    torch.cuda.empty_cache()

    spec = MODELS[meta["model_key"]]
    results = {**meta, "run": name, "tag": f"{meta.get('tag', '')}+L{len(keep)}",
               "prune": {"source_run": run, "dropped_layers": drop, "n_layers": len(keep), "params_M": round(n_params),
                         "epochs": epochs, "lr": lr, "batch": batch, "steps": steps, "ce_weight": ce_weight,
                         "teacher_agreement_before": before, "teacher_agreement_after": after,
                         "train_time_s": round(train_time), "log": log},
               "splits": {}}
    run_eval(student, tok, name, fmt, dst, results, eval_splits(0), spec.get("eval_batch_size", 32))
    VOLUMES[RUNS_DIR].commit()
    return {"run": name, "before": before, "after": after,
            **{s: round(100 * results["splits"][s]["all"]["exact_match"], 1) for s in results["splits"]}}


@app.local_entrypoint()
def probe(run: str):
    out = probe_remote.remote(run)
    lin = [(a, i) for i, (a, k) in enumerate(zip(out["skip_agreement"], out["layer_types"])) if k == "linear_attention"]
    print("linear layers, least important first:", [i for _, i in sorted(lin, reverse=True)])


@app.local_entrypoint()
def prune(run: str, drop: str, epochs: float = 2, lr: float = 2e-5, batch: int = 16, limit: int = 0, ce_weight: float = 0.2):
    out = prune_remote.remote(run, [int(x) for x in drop.split(",")], epochs, lr, batch, limit=limit, ce_weight=ce_weight)
    print(json.dumps(out, indent=1))
