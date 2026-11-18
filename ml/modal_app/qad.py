"""Quantization-aware distillation (QAD) for the pruned Hishab model, aimed at llama.cpp's Q3_K.

  conda run -n ML modal run ml/modal_app/qad.py --run qwen3.5-0.8b__pipe_signed__full__v13__trim38k__L20 \
      --teacher qwen3.5-0.8b__pipe_signed__full__v13__trim38k --epochs 3

The student starts from <run> and trains with every decoder Linear weight passed through a fake Q3_K
quantizer (straight-through gradient), against the unpruned teacher (KL on answer tokens + ce_weight x
CE on the gold), exactly like prune_layers.py's heal. Latent (unquantized) weights are saved to
/runs/<run>__Q3 and then go through the normal gguf.py path (imatrix + llama-quantize Q3_K_M), so the
scores come from real llama.cpp files. The fake quantizer follows Q3_K: 256-weight super-blocks of 16
sub-blocks of 16, 3-bit values in [-4, 3], 6-bit sub-block scales times one super-block scale. It is
an approximation (llama.cpp searches the scale per sub-block; here it is absmax / -4), so the real file can differ a little.
Embeddings, lm_head and tensors whose width is not a multiple of 256 stay unquantized in training.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modal_app.common import RUNS_DIR, VOLUMES, app, hf_secret, train_image  # noqa: E402
from modal_app.prune_layers import _batches, _encoder, _rows  # noqa: E402

with train_image.imports():
    import torch
    import torch.nn.functional as F


def fake_q3k(w: "torch.Tensor") -> "torch.Tensor":
    """Q3_K round trip of a [out, in] weight (in a multiple of 256), returned in w's dtype."""
    shape, dtype = w.shape, w.dtype
    g = w.detach().float().reshape(-1, 16, 16)                 # super-blocks x sub-blocks x 16
    idx = g.abs().argmax(-1, keepdim=True)
    mx = g.gather(-1, idx)                                     # signed value of largest magnitude
    scale = mx / -4.0
    d = scale.abs().amax(dim=1, keepdim=True) / 32.0
    d = torch.where(d == 0, torch.ones_like(d), d)
    sc = (scale / d).round().clamp(-32, 31)
    s_hat = d * sc
    safe = torch.where(s_hat == 0, torch.ones_like(s_hat), s_hat)
    q = torch.where(s_hat == 0, torch.zeros_like(g), (g / safe).round().clamp(-4, 3))
    return (q * s_hat).reshape(shape).to(dtype)


class FakeQ3K(torch.nn.Module if "torch" in globals() else object):
    def forward(self, w):
        return w + (fake_q3k(w) - w).detach()                  # straight-through estimator


def _targets(model) -> tuple[list[tuple[str, "torch.nn.Linear"]], list[str]]:
    use, skip = [], []
    for name, m in model.named_modules():
        if not isinstance(m, torch.nn.Linear):
            continue
        if "lm_head" in name or m.in_features % 256 or m.out_features < 64:
            skip.append(name)
        else:
            use.append((name, m))
    return use, skip


@app.function(image=train_image, gpu="A100-80GB", timeout=4 * 60 * 60, volumes=VOLUMES, secrets=[hf_secret])
def qad_remote(run: str, teacher: str, epochs: float = 3, lr: float = 1e-5, batch: int = 16,
               train_split: str = "train_v13", ce_weight: float = 0.2, limit: int = 0, suffix: str = "Q3") -> dict:
    import shutil

    import torch.nn.utils.parametrize as P
    from transformers import AutoModelForCausalLM

    from hishab.registry import MODELS
    from modal_app.hfeval import eval_splits, load_tokenizer, run_eval

    src = Path(RUNS_DIR) / run
    meta = json.loads((src / "eval_hf.json").read_text())
    fmt = meta["fmt"]
    tok = load_tokenizer(str(src / "merged"))
    t = AutoModelForCausalLM.from_pretrained(str(Path(RUNS_DIR) / teacher / "merged"), dtype=torch.bfloat16, device_map="cuda").eval()
    for p in t.parameters():
        p.requires_grad_(False)
    student = AutoModelForCausalLM.from_pretrained(str(src / "merged"), dtype=torch.bfloat16, device_map="cuda")
    use, skip = _targets(student)
    print(f"fake Q3_K on {len(use)} Linear weights; left alone: {len(skip)} ({skip[:6]}...)", flush=True)
    for _, m in use:
        P.register_parametrization(m, "weight", FakeQ3K())
    student.train()
    for p in student.parameters():
        p.requires_grad_(True)

    enc_fn = _encoder(tok, fmt)
    rows = _rows(train_split)[: limit or None]
    enc = [enc_fn(r) for r in rows]
    steps = int(epochs * ((len(enc) + batch - 1) // batch))
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05, anneal_strategy="cos")

    def agreement(rs):
        e = [enc_fn(r) for r in rs]
        hit = tot = 0
        student.eval()
        with torch.no_grad():
            for ids, att, lm in _batches(e, tok.pad_token_id, 16, list(range(len(e)))):
                a = student(input_ids=ids, attention_mask=att).logits[:, :-1].argmax(-1)
                b = t(input_ids=ids, attention_mask=att).logits[:, :-1].argmax(-1)
                hit += ((a == b) & lm).sum().item()
                tot += lm.sum().item()
        student.train()
        return hit / max(tot, 1)

    tc = _rows("test_clean_v2")
    before = agreement(tc)
    print(f"agreement with teacher, fake-Q3 but untrained: {before:.4f}", flush=True)

    step, t0, log, ep = 0, time.time(), [], 0
    while step < steps:
        order = list(range(len(enc)))
        random.Random(ep).shuffle(order)
        for ids, att, lm in _batches(enc, tok.pad_token_id, batch, order):
            if step >= steps:
                break
            with torch.no_grad():
                tl = t(input_ids=ids, attention_mask=att).logits[:, :-1][lm].float()
            sl = student(input_ids=ids, attention_mask=att).logits[:, :-1][lm].float()
            kl = F.kl_div(F.log_softmax(sl, -1), F.log_softmax(tl, -1), log_target=True, reduction="batchmean")
            ce = F.cross_entropy(sl, ids[:, 1:][lm])
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
    print(f"agreement with teacher, fake-Q3 trained: {after:.4f}", flush=True)

    name = f"{run}__{suffix}"
    dst = Path(RUNS_DIR) / name
    if dst.exists():
        shutil.rmtree(dst)
    merged = dst / "merged"
    student.eval()
    for _, m in use:                                           # drop the fake quantizer, keep the latent weights
        P.remove_parametrizations(m, "weight", leave_parametrized=False)
    student.config.use_cache = True
    student.save_pretrained(str(merged), safe_serialization=True)
    for f in (src / "merged").iterdir():
        if f.suffix != ".safetensors" and f.name not in ("config.json", "generation_config.json", "model.safetensors.index.json"):
            shutil.copy(f, merged / f.name)
    results = {**meta, "run": name, "tag": f"{meta.get('tag', '')}+{suffix}",
               "qad": {"source_run": run, "teacher_run": teacher, "format": "Q3_K (fake, absmax/-4 scales)",
                       "quantized_tensors": len(use), "epochs": epochs, "lr": lr, "batch": batch, "steps": steps,
                       "ce_weight": ce_weight, "train_split": train_split, "teacher_agreement_untrained": before,
                       "teacher_agreement_trained": after, "train_time_s": round(train_time), "log": log},
               "splits": {}}
    # bf16 eval of the latent weights (the real scores are the GGUF ones); the fake-quantized eval is below
    del t
    torch.cuda.empty_cache()
    spec = MODELS[meta["model_key"]]
    for _, m in use:
        m.weight.data = fake_q3k(m.weight.data)                # bake the quantization in, evaluate that
    run_eval(student, tok, name, fmt, dst, results, eval_splits(0), spec.get("eval_batch_size", 32))
    VOLUMES[RUNS_DIR].commit()
    return {"run": name, "before": before, "after": after,
            **{s: round(100 * results["splits"][s]["all"]["exact_match"], 1) for s in results["splits"]}}


@app.local_entrypoint()
def main(run: str, teacher: str, epochs: float = 3, lr: float = 1e-5, batch: int = 16, limit: int = 0):
    print(json.dumps(qad_remote.remote(run, teacher, epochs, lr, batch, limit=limit), indent=1))
