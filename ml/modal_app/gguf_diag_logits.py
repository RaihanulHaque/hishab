"""Where exactly does llama.cpp diverge from HF on a Bangla number message?

Teacher-forced comparison: take the HF model's greedy completion for a prompt, then at every
completion position ask both HF (fp32, CPU) and llama-server (F16 GGUF, same prompt tokens)
for the next-token distribution. Report, per position, HF's top token and probability, the
GGUF's probability for that same token and its own top token. The first position where the
GGUF's argmax differs is the divergence point; the trend of the probability gap along the
sequence tells whether it is a slow drift (precision accumulating in the recurrent layers) or a
sudden jump (a specific op).

  conda run -n ML modal run ml/modal_app/gguf_diag_logits.py --run qwen3.5-0.8b__json__full__v8 \
      --ids txn_xxx,txn_yyy [--quant F16]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modal_app.common import DATA_DIR, RUNS_DIR, VOLUMES, app, gguf_image  # noqa: E402

BIN = Path("/opt/llama.cpp/build/bin")


@app.function(image=gguf_image, cpu=16, memory=32768, timeout=60 * 60, volumes=VOLUMES)
def compare(run: str, quant: str, ids: list[str], split: str, max_new: int) -> list[dict]:
    import unicodedata

    import requests
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hishab.prompts import build_messages

    meta = json.loads((Path(RUNS_DIR) / run / "eval_hf.json").read_text())
    fmt = meta["fmt"]
    rows = {json.loads(l)["id"]: json.loads(l) for l in (Path(DATA_DIR) / "splits" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    merged = Path(RUNS_DIR) / run / "merged"
    tok = AutoTokenizer.from_pretrained(str(merged))
    model = AutoModelForCausalLM.from_pretrained(str(merged), dtype=torch.float32).eval()

    gguf = Path(RUNS_DIR) / run / "gguf" / f"{run}-{quant}.gguf"
    port = 8080
    proc = subprocess.Popen([str(BIN / "llama-server"), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port),
                             "-c", "2048", "-np", "1", "-t", str(os.cpu_count() or 8), "--no-warmup", "--log-disable"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            raise RuntimeError(proc.stderr.read()[-2000:])
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                break
        except Exception:
            pass
        if time.time() - t0 > 300:
            raise TimeoutError
        time.sleep(1)

    def gguf_next(token_ids: list[int]) -> dict:
        """next-token probs from llama-server given exact token ids (no tokenizer involved)."""
        r = requests.post(f"http://127.0.0.1:{port}/completion",
                          json={"prompt": token_ids, "n_predict": 1, "n_probs": 10, "temperature": 0,
                                "cache_prompt": False, "samplers": []}, timeout=600).json()
        probs = r["completion_probabilities"][0]
        tops = probs.get("top_logprobs") or probs.get("probs") or []
        out = {}
        for t in tops:
            tid = t.get("id")
            p = t.get("prob")
            if p is None and "logprob" in t:
                import math
                p = math.exp(t["logprob"])
            out[tid] = (p, t.get("token"))
        return out

    results = []
    try:
        for rid in ids:
            r = rows[rid]
            msgs = [{**m, "content": unicodedata.normalize("NFC", m["content"])} for m in build_messages(fmt, r["instruction"])]
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            prompt_ids = [int(t) for t in tok(text, add_special_tokens=False)["input_ids"]]
            # --- how does the server's own chat path build the prompt? ---
            path = {}
            try:
                rendered = requests.post(f"http://127.0.0.1:{port}/apply-template",
                                         json={"messages": msgs, "chat_template_kwargs": {"enable_thinking": False}}, timeout=60).json().get("prompt")
                path["template_matches_hf"] = rendered == text
                if rendered != text:
                    import difflib
                    path["template_diff"] = [d for d in difflib.ndiff(text.splitlines(), (rendered or "").splitlines()) if d[:1] in "+-"][:12]
                srv_ids = requests.post(f"http://127.0.0.1:{port}/tokenize", json={"content": rendered or text, "add_special": False, "with_pieces": True}, timeout=60).json()["tokens"]
                srv_ids_plain = [t["id"] if isinstance(t, dict) else t for t in srv_ids]
                path["tokens_match_hf"] = srv_ids_plain == prompt_ids
                if srv_ids_plain != prompt_ids:
                    k = next((j for j, (a, b) in enumerate(zip(srv_ids_plain, prompt_ids)) if a != b), min(len(srv_ids_plain), len(prompt_ids)))
                    path["first_token_diff_at"] = k
                    path["hf_tokens_around"] = [tok.decode([t]) for t in prompt_ids[max(0, k - 3): k + 6]]
                    path["srv_tokens_around"] = [(t["piece"] if isinstance(t, dict) else tok.decode([t])) for t in srv_ids[max(0, k - 3): k + 6]]
                    path["n_hf"], path["n_srv"] = len(prompt_ids), len(srv_ids_plain)
                chat = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     json={"messages": msgs, "max_tokens": max_new, "temperature": 0, "seed": 0,
                                           "chat_template_kwargs": {"enable_thinking": False}}, timeout=600).json()
                path["chat_completion"] = (chat["choices"][0]["message"].get("content") or "").strip()
                # same ids through /completion, greedy, full length
                comp_ids = requests.post(f"http://127.0.0.1:{port}/completion",
                                         json={"prompt": prompt_ids, "n_predict": max_new, "temperature": 0, "cache_prompt": False, "samplers": []}, timeout=600).json()
                path["completion_from_hf_ids"] = (comp_ids.get("content") or "").strip()
            except Exception as e:  # keep going; the teacher-forced part is independent
                path["error"] = repr(e)
            print(f"{rid} prompt-path: {json.dumps({k: v for k, v in path.items() if k not in ('chat_completion', 'completion_from_hf_ids')}, ensure_ascii=False)}")
            # HF greedy completion
            with torch.inference_mode():
                gen = model.generate(torch.tensor([prompt_ids]), max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            comp = gen[0, len(prompt_ids):].tolist()
            # strip trailing eos/pad
            while comp and comp[-1] in {tok.eos_token_id, tok.pad_token_id}:
                comp.pop()
            # HF distributions at every position in one pass
            with torch.inference_mode():
                logits = model(torch.tensor([prompt_ids + comp])).logits[0]
            probs = torch.softmax(logits.float(), -1)
            steps = []
            first_div = None
            for i in range(len(comp)):
                pos = len(prompt_ids) + i - 1  # logits at pos predict comp[i]
                hf_top = int(probs[pos].argmax())
                hf_p = float(probs[pos, comp[i]])
                g = gguf_next(prompt_ids + comp[:i])
                g_top = max(g.items(), key=lambda kv: kv[1][0])[0] if g else None
                g_p = g.get(comp[i], (0.0, None))[0]
                step = {"i": i, "tok": tok.decode([comp[i]]), "hf_p": round(hf_p, 4), "gguf_p": round(g_p or 0.0, 4),
                        "gguf_top": tok.decode([g_top]) if g_top is not None else None, "diverge": g_top != comp[i]}
                if step["diverge"] and first_div is None:
                    first_div = i
                steps.append(step)
            results.append({"id": rid, "language": r["language"], "instruction": r["instruction"], "n_prompt": len(prompt_ids), "prompt_path": path,
                            "hf_completion": tok.decode(comp), "gold": r["transactions"], "first_divergence": first_div, "steps": steps})
            print(f"{rid} n_prompt={len(prompt_ids)} n_comp={len(comp)} first_div={first_div}")
    finally:
        proc.terminate()
    return results


@app.local_entrypoint()
def main(run: str = "qwen3.5-0.8b__json__full__v8", quant: str = "F16", ids: str = "", split: str = "numbers",
         max_new: int = 160, out: str = "ml/results/gguf_diag_logits.json"):
    res = compare.remote(run, quant, [i for i in ids.split(",") if i], split, max_new)
    Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for r in res:
        print(f"\n### {r['id']} ({r['language']}, {r['n_prompt']} prompt tokens) first divergence at {r['first_divergence']}")
        print("   ", r["instruction"][:100])
        print("    HF:", r["hf_completion"][:160])
        pp = r.get("prompt_path", {})
        print("    server chat path:", {k: v for k, v in pp.items() if k not in ("chat_completion", "completion_from_hf_ids")})
        print("    chat endpoint :", pp.get("chat_completion", "")[:160])
        print("    HF-ids /compl :", pp.get("completion_from_hf_ids", "")[:160])
        print("    pos | token | HF p | GGUF p | GGUF top")
        for s in r["steps"]:
            flag = " <-- diverge" if s["diverge"] else ""
            if s["diverge"] or s["gguf_p"] < 0.5 or s["i"] % 8 == 0:
                print(f"    {s['i']:3d} | {s['tok']!r:16s} | {s['hf_p']:.3f} | {s['gguf_p']:.3f} | {s['gguf_top']!r}{flag}")
