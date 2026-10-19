"""Train the MTP (multi-token prediction) layer of a finished Hishab run against the run itself.

The main model stays frozen. For every position t the MTP layer sees what llama.cpp gives it at
inference (`draft-mtp`): the main model's final hidden state h_t (after the last norm) and the
embedding of the next token x_{t+1}; it must predict x_{t+2}. The target is the main model's own
distribution at t+1 (soft labels), not the gold text: a draft is accepted exactly when it matches
the main model's greedy choice, so imitating the main model is what raises acceptance.

  conda run -n ML modal run ml/modal_app/train_mtp.py --run qwen3.5-0.8b__pipe_signed__full__v11__trim38k

Starts from the base model's MTP layer, writes <run>__mtp with the trained `mtp.*` tensors next
to the unchanged main weights, and reports top-1 agreement (the acceptance a one-token draft
would get) on validation and test_clean before and after training.
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
    import torch.nn as nn
    import torch.nn.functional as F


def _rows(name: str) -> list[dict]:
    p = Path(DATA_DIR) / "splits" / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


@app.function(image=train_image, gpu="A100-80GB", timeout=3 * 60 * 60, volumes=VOLUMES, secrets=[hf_secret])
def train_mtp(run: str, base: str = "Qwen/Qwen3.5-0.8B", epochs: float = 2, lr: float = 3e-4,
              batch: int = 32, train_split: str = "train_v10", limit: int = 0, suffix: str = "mtp") -> dict:
    import shutil

    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm

    from hishab.prompts import build_messages
    from hishab.schema import render_target
    from modal_app.hfeval import load_tokenizer

    src = Path(RUNS_DIR) / run
    meta = json.loads((src / "eval_hf.json").read_text())
    fmt = meta["fmt"]
    tok = load_tokenizer(str(src / "merged"))
    model = AutoModelForCausalLM.from_pretrained(str(src / "merged"), dtype=torch.bfloat16, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tm = model.model  # Qwen3_5TextModel
    cfg = tm.config
    full_idx = cfg.layer_types.index("full_attention")

    class MTP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False)
            self.pre_fc_norm_embedding = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.pre_fc_norm_hidden = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.layers = nn.ModuleList([Qwen3_5DecoderLayer(cfg, layer_idx=full_idx)])
            self.norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    mtp = MTP()
    state = {}
    for fname in [f for f in list_repo_files(base) if f.endswith(".safetensors")]:
        with safe_open(hf_hub_download(base, fname), framework="pt") as sf:
            for k in sf.keys():
                if k.startswith("mtp."):
                    state[k[len("mtp."):]] = sf.get_tensor(k)
    missing, unexpected = mtp.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"MTP weights do not match: missing {missing}, unexpected {unexpected}")
    mtp = mtp.to("cuda", dtype=torch.float32)  # trained in fp32, saved in bf16

    def encode(r: dict) -> tuple[list[int], int]:
        msgs = build_messages(fmt, r["instruction"])
        full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": render_target(fmt, r.get("translation"), r["transactions"])}], tokenize=False)
        gen = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        assert full.startswith(gen)
        ids = tok(full, add_special_tokens=False)["input_ids"]
        n_prompt = len(tok(gen, add_special_tokens=False)["input_ids"])
        return ids, n_prompt

    def batches(rows: list[dict], bs: int, shuffle: bool, seed: int = 0):
        enc = [encode(r) for r in rows]
        order = list(range(len(enc)))
        if shuffle:
            import random
            random.Random(seed).shuffle(order)
        for i in range(0, len(order), bs):
            chunk = [enc[j] for j in order[i : i + bs]]
            T = max(len(ids) for ids, _ in chunk)
            ids = torch.full((len(chunk), T), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(chunk), T), dtype=torch.long)
            lossmask = torch.zeros((len(chunk), T - 1), dtype=torch.bool)
            for b, (x, n_prompt) in enumerate(chunk):
                ids[b, : len(x)] = torch.tensor(x)
                att[b, : len(x)] = 1
                # MTP output at t predicts x_{t+2}; count it when x_{t+2} is a completion token
                for t in range(max(n_prompt - 2, 0), len(x) - 2):
                    lossmask[b, t] = True
            yield ids.cuda(), att.cuda(), lossmask.cuda()

    def forward(ids, att):
        with torch.no_grad():
            H = tm(input_ids=ids, attention_mask=att).last_hidden_state  # final-normed, like h_nextn
            main_logits = model.lm_head(H).float()
            E = tm.embed_tokens(ids)
        B, T = ids.shape
        h = H[:, :-1].float()
        e = E[:, 1:].float()
        pos = torch.arange(1, T, device=ids.device)
        text_pos = pos.view(1, -1).expand(B, -1)
        rope_pos = pos.view(1, 1, -1).expand(3, B, -1)
        x = mtp.fc(torch.cat([mtp.pre_fc_norm_embedding(e), mtp.pre_fc_norm_hidden(h)], dim=-1))
        mask = create_causal_mask(config=cfg, inputs_embeds=x, attention_mask=att[:, 1:], past_key_values=None,
                                  position_ids=text_pos)
        pe = tm.rotary_emb(x, rope_pos)
        x = mtp.layers[0](x, position_embeddings=pe, attention_mask=mask, position_ids=text_pos)
        x = mtp.norm(x)
        mtp_logits = F.linear(x, model.lm_head.weight.float())
        return mtp_logits, main_logits[:, 1:]  # both predict x_{t+2} at index t

    @torch.no_grad()
    def agreement(rows: list[dict]) -> float:
        hit = tot = 0
        for ids, att, lm in batches(rows, 32, False):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a, b = forward(ids, att)
            hit += ((a.argmax(-1) == b.argmax(-1)) & lm).sum().item()
            tot += lm.sum().item()
        return hit / max(tot, 1)

    val, tc = _rows("validation"), _rows("test_clean")
    before = {"validation": agreement(val), "test_clean": agreement(tc)}
    print("top-1 agreement before:", before, flush=True)

    train_rows = _rows(train_split)
    if limit:
        train_rows = train_rows[:limit]
    steps = int(epochs * ((len(train_rows) + batch - 1) // batch))
    opt = torch.optim.AdamW(mtp.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05, anneal_strategy="cos")
    mtp.train()
    step, t0, log = 0, time.time(), []
    ep = 0
    while step < steps:
        for ids, att, lm in batches(train_rows, batch, True, seed=ep):
            if step >= steps:
                break
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a, b = forward(ids, att)
            a, b = a[lm].float(), b[lm].float()
            loss = F.kl_div(F.log_softmax(a, -1), F.log_softmax(b, -1), log_target=True, reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mtp.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0 or step == steps:
                acc = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
                log.append({"step": step, "loss": round(loss.item(), 4), "agree": round(acc, 4)})
                print(f"step {step}/{steps} loss {loss.item():.4f} agree {acc:.3f} {time.time() - t0:.0f}s", flush=True)
        ep += 1
    mtp.eval()
    after = {"validation": agreement(val), "test_clean": agreement(tc)}
    print("top-1 agreement after:", after, flush=True)

    # new run: main weights unchanged, trained mtp.* added
    name = f"{run}__{suffix}"
    dst = Path(RUNS_DIR) / name
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "merged").mkdir(parents=True)
    for f in (src / "merged").iterdir():
        if f.suffix != ".safetensors":
            shutil.copy(f, dst / "merged" / f.name)
    shards = sorted((src / "merged").glob("*.safetensors"))
    trained = {f"mtp.{k}": v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in mtp.state_dict().items()}
    for i, f in enumerate(shards):
        out = {}
        with safe_open(str(f), framework="pt") as sf:
            md = sf.metadata()
            for k in sf.keys():
                out[k] = sf.get_tensor(k)
        if i == len(shards) - 1:
            out.update(trained)
        save_file(out, str(dst / "merged" / f.name), metadata=md)
    info = {"source": base, "trained": True, "epochs": epochs, "lr": lr, "batch": batch, "steps": steps,
            "train_rows": len(train_rows), "target": "main model distribution at t+1 (KL)",
            "top1_agreement_before": before, "top1_agreement_after": after, "log": log,
            "train_time_s": round(time.time() - t0)}
    meta = {**meta, "run": name, "mtp": info}
    (dst / "eval_hf.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    VOLUMES[RUNS_DIR].commit()
    return {"run": name, **{k: info[k] for k in ("top1_agreement_before", "top1_agreement_after", "steps", "train_time_s")}}


@app.local_entrypoint()
def main(run: str, epochs: float = 2, lr: float = 3e-4, batch: int = 32, limit: int = 0):
    print(json.dumps(train_mtp.remote(run, epochs=epochs, lr=lr, batch=batch, limit=limit), indent=1))
