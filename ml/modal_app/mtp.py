"""Multi-token prediction (MTP) for the Hishab Qwen3.5 models: graft an MTP layer, then measure
speculative decoding with llama.cpp (`--spec-type draft-mtp`).

Qwen3.5-0.8B ships one MTP layer (`mtp.*`, ~25M parameters): it reads the main model's last hidden
state plus the embedding of the token just chosen and guesses the token after that. The main model
then checks the guesses in one pass and keeps the ones it agrees with, so greedy output is unchanged
and only speed can move. transformers' Qwen3_5ForCausalLM has no MTP module, so fine-tuned runs
lose the layer (their GGUFs say nextn_predict_layers=1 but carry no block 24).

  # copy a run and add the base model's MTP layer (untrained for the fine-tune)
  conda run -n ML modal run ml/modal_app/mtp.py::graft --run qwen3.5-0.8b__pipe_signed__full__v11__trim38k
  # then build GGUFs of <run>__mtpbase with gguf.py as usual, and compare with and without MTP
  conda run -n ML modal run ml/modal_app/mtp.py::bench --run <run>__mtpbase --quant Q4_K_M-imat --split test_clean
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modal_app.common import DATA_DIR, RUNS_DIR, VOLUMES, app, gguf_image, hf_secret  # noqa: E402

LLAMA = Path("/opt/llama.cpp")
BIN = LLAMA / "build" / "bin"


@app.function(image=gguf_image, cpu=8, memory=16384, timeout=60 * 60, volumes=VOLUMES, secrets=[hf_secret])
def graft_remote(run: str, base: str = "Qwen/Qwen3.5-0.8B", suffix: str = "mtpbase") -> dict:
    import shutil

    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors import safe_open
    from safetensors.torch import save_file

    src = Path(RUNS_DIR) / run
    name = f"{run}__{suffix}"
    dst = Path(RUNS_DIR) / name
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "merged").mkdir(parents=True)
    for f in (src / "merged").iterdir():
        if f.suffix != ".safetensors":
            shutil.copy(f, dst / "merged" / f.name)

    mtp = {}
    for fname in [f for f in list_repo_files(base) if f.endswith(".safetensors")]:
        with safe_open(hf_hub_download(base, fname), framework="pt") as sf:
            for k in sf.keys():
                if k.startswith("mtp."):
                    mtp[k] = sf.get_tensor(k)
    if not mtp:
        raise RuntimeError(f"{base} has no mtp.* tensors")

    shards = sorted((src / "merged").glob("*.safetensors"))
    for i, f in enumerate(shards):
        out = {}
        with safe_open(str(f), framework="pt") as sf:
            md = sf.metadata()
            for k in sf.keys():
                out[k] = sf.get_tensor(k)
        if any(k.startswith("mtp.") for k in out):
            raise RuntimeError("run already has mtp tensors")
        if i == len(shards) - 1:
            out.update(mtp)
        save_file(out, str(dst / "merged" / f.name), metadata=md)
    idx = dst / "merged" / "model.safetensors.index.json"
    if idx.exists():  # sharded: register the new tensors in the last shard
        d = json.loads(idx.read_text())
        for k in mtp:
            d["weight_map"][k] = shards[-1].name
        idx.write_text(json.dumps(d, indent=2))

    cfg = json.loads((dst / "merged" / "config.json").read_text())
    assert cfg.get("mtp_num_hidden_layers", 0) >= 1, "config has no MTP layer"
    meta = json.loads((src / "eval_hf.json").read_text())
    meta = {**meta, "run": name, "mtp": {"source": base, "trained": False, "tensors": sorted(mtp)}}
    (dst / "eval_hf.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    VOLUMES[RUNS_DIR].commit()
    print(f"{name}: added {len(mtp)} mtp tensors from {base}")
    return {"run": name, "mtp_tensors": len(mtp)}


def _server(gguf: Path, extra: list[str], port: int):
    import subprocess

    import requests

    cmd = [str(BIN / "llama-server"), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port),
           "-c", "4096", "-np", "1", "-t", "8", "--jinja", "--no-warmup", "--log-disable", *extra]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    t0 = time.time()
    while time.time() - t0 < 300:
        if proc.poll() is not None:
            raise RuntimeError("llama-server died:\n" + (proc.stderr.read() or "")[-3000:])
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError("llama-server did not start")


@app.function(image=gguf_image, cpu=8, memory=16384, timeout=3 * 60 * 60, volumes=VOLUMES)
def bench_remote(run: str, quant: str = "Q4_K_M-imat", split: str = "test_clean", limit: int = 0,
                 draft_max: list[int] | None = None) -> dict:
    """Same rows, one slot, one request at a time: plain decoding, then MTP with each draft size.
    Reports exact match, whether every answer is identical to plain decoding, acceptance and tok/s."""
    import unicodedata

    import requests

    from hishab.metrics import evaluate
    from hishab.prompts import build_messages

    meta = json.loads((Path(RUNS_DIR) / run / "eval_hf.json").read_text())
    fmt = meta["fmt"]
    gguf = Path(RUNS_DIR) / run / "gguf" / f"{run}-{quant}.gguf"
    rows = [json.loads(l) for l in (Path(DATA_DIR) / "splits" / f"{split}.jsonl").read_text().splitlines() if l.strip()]
    if limit:
        rows = rows[:limit]
    configs = [("plain", [])] + [(f"mtp{n}", ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n)])
                                 for n in (draft_max or [1, 2, 3])]
    results = {}
    base_preds = None
    for i, (label, extra) in enumerate(configs):
        proc = _server(gguf, extra, 8090 + i)
        preds, n_out, pred_ms, d_n, d_acc = [], 0, 0.0, 0, 0
        try:
            for r in rows:
                msgs = [{**m, "content": unicodedata.normalize("NFC", m["content"])} for m in build_messages(fmt, r["instruction"])]
                resp = requests.post(f"http://127.0.0.1:{8090 + i}/v1/chat/completions", timeout=600, json={
                    "messages": msgs, "max_tokens": 128, "temperature": 0, "seed": 0,
                    "chat_template_kwargs": {"enable_thinking": False}}).json()
                preds.append((resp["choices"][0]["message"].get("content") or "").strip())
                t = resp.get("timings", {})
                n_out += t.get("predicted_n", 0)
                pred_ms += t.get("predicted_ms", 0.0)
                d_n += t.get("draft_n", 0) or 0
                d_acc += t.get("draft_n_accepted", 0) or 0
        finally:
            proc.terminate()
            proc.wait(10)
        m = evaluate(rows, preds)["all"]
        same = None if base_preds is None else sum(a == b for a, b in zip(base_preds, preds))
        if base_preds is None:
            base_preds = preds
        results[label] = {
            "exact": round(100 * m["exact_match"], 1),
            "identical_to_plain": same,
            "tok_per_s": round(n_out / (pred_ms / 1000), 1) if pred_ms else None,
            "gen_ms_per_answer": round(pred_ms / len(rows), 1),
            "draft_tokens": d_n, "accepted": d_acc,
            "acceptance": round(d_acc / d_n, 3) if d_n else None,
            "out_tokens": n_out,
        }
        print(label, json.dumps(results[label]), flush=True)
    out = {"run": run, "quant": quant, "split": split, "n": len(rows), "results": results}
    edir = Path(RUNS_DIR) / run / "eval_gguf"
    edir.mkdir(exist_ok=True)
    (edir / f"mtp_bench.{quant}.{split}.json").write_text(json.dumps(out, indent=1))
    VOLUMES[RUNS_DIR].commit()
    return out


@app.local_entrypoint()
def graft(run: str, base: str = "Qwen/Qwen3.5-0.8B", suffix: str = "mtpbase"):
    print(graft_remote.remote(run, base, suffix))


@app.local_entrypoint()
def bench(run: str, quant: str = "Q4_K_M-imat", split: str = "test_clean", limit: int = 0, draft_max: str = "1,2,3"):
    out = bench_remote.remote(run, quant, split, limit, [int(x) for x in draft_max.split(",")])
    print(json.dumps(out, indent=1))
