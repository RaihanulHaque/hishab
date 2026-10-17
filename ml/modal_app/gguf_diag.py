"""Isolate the llama.cpp accuracy gap on Bangla-script number messages.

The same weights score ~17 points lower on the Bangla slice of the `numbers` split through
llama.cpp (even as BF16/F16 GGUF) than through HF transformers. Tokenization is identical, so
the difference is in the inference path. This runs one slice through llama-server under several
server configurations and reports exact match per configuration:

  conda run -n ML modal run ml/modal_app/gguf_diag.py --run qwen3.5-0.8b__json__full__v8 --quant F16 \
      --language bangla [--configs baseline,np1,ub4,fa_off,kv_f32,t1]

Configs: baseline = the eval settings (8 parallel slots), np1 = one slot, ub1 = prompt processed
one token at a time (-b 1 -ub 1), fa_off = flash attention off, kv_f32 = f32 KV cache,
t1 = single thread. `--llama master` builds a second image at llama.cpp master to test whether
upstream already fixed it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modal_app.common import DATA_DIR, RUNS_DIR, VOLUMES, app, gguf_image  # noqa: E402

BIN = Path("/opt/llama.cpp/build/bin")
MAX_TOKENS = 192

CONFIGS: dict[str, list[str]] = {
    "baseline": ["-np", "8"],
    "np1": ["-np", "1"],
    "ub4": ["-np", "1", "-b", "4", "-ub", "4"],
    "fa_off": ["-np", "1", "-fa", "off"],
    "kv_f32": ["-np", "1", "-ctk", "f32", "-ctv", "f32"],
    "t1": ["-np", "1", "-t", "1"],
    "ub16": ["-np", "1", "-b", "16", "-ub", "16"],
}

# llama.cpp master, same build flags, for the "did upstream fix it" check
master_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev", "pkg-config")
    .run_commands(
        "git clone https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
        "cd /opt/llama.cpp && cmake -S . -B build -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON "
        "-DGGML_F16C=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF "
        "-DCMAKE_BUILD_TYPE=Release",
        "cd /opt/llama.cpp && cmake --build build --config Release -j --target llama-server",
        "pip install requests",
        "cd /opt/llama.cpp && git rev-parse HEAD > /opt/llama_commit.txt",
    )
    .add_local_dir(str(Path(__file__).resolve().parents[1] / "hishab"), remote_path="/root/hishab")
    .add_local_dir(str(Path(__file__).resolve().parents[1] / "modal_app"), remote_path="/root/modal_app")
)


def _rows(split: str, language: str | None, limit: int) -> list[dict]:
    rows = [json.loads(l) for l in (Path(DATA_DIR) / "splits" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if language:
        rows = [r for r in rows if r["language"] == language]
    return rows[:limit] if limit else rows


def _serve_and_eval(gguf: Path, extra: list[str], rows: list[dict], fmt: str, port: int = 8080) -> dict:
    import unicodedata

    import requests

    from hishab.metrics import evaluate
    from hishab.prompts import build_messages

    np_ = int(extra[extra.index("-np") + 1]) if "-np" in extra else 1
    cmd = [str(BIN / "llama-server"), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port),
           "-c", str(2048 * np_), "-t", str(os.cpu_count() or 8), "--jinja", "--no-warmup", "--log-disable", *extra]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            raise RuntimeError("llama-server died:\n" + (proc.stderr.read() or "")[-2000:])
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                break
        except Exception:
            pass
        if time.time() - t0 > 300:
            raise TimeoutError("server not healthy")
        time.sleep(1)

    def one(r: dict) -> str:
        msgs = [{**m, "content": unicodedata.normalize("NFC", m["content"])} for m in build_messages(fmt, r["instruction"])]
        resp = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                             json={"messages": msgs, "max_tokens": MAX_TOKENS, "temperature": 0, "seed": 0,
                                   "chat_template_kwargs": {"enable_thinking": False}}, timeout=900).json()
        return (resp["choices"][0]["message"].get("content") or "").strip()

    try:
        t0 = time.time()
        with ThreadPoolExecutor(np_) as ex:
            preds = list(ex.map(one, rows))
        wall = time.time() - t0
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
    m = evaluate(rows, preds)
    return {"exact": m["all"]["exact_match"], "amounts": m["all"].get("amounts_match"), "n": len(rows),
            "wall_s": round(wall, 1), "preds": preds}


@app.function(image=gguf_image, cpu=16, memory=16384, timeout=2 * 60 * 60, volumes=VOLUMES)
def diag_pinned(run: str, quant: str, config: str, split: str, language: str | None, limit: int) -> dict:
    meta = json.loads((Path(RUNS_DIR) / run / "eval_hf.json").read_text())
    gguf = Path(RUNS_DIR) / run / "gguf" / f"{run}-{quant}.gguf"
    rows = _rows(split, language, limit)
    out = _serve_and_eval(gguf, CONFIGS[config], rows, meta["fmt"])
    return {"llama": "pinned", "config": config, **out, "ids": [r["id"] for r in rows]}


@app.function(image=master_image, cpu=16, memory=16384, timeout=2 * 60 * 60, volumes=VOLUMES)
def diag_master(run: str, quant: str, config: str, split: str, language: str | None, limit: int) -> dict:
    meta = json.loads((Path(RUNS_DIR) / run / "eval_hf.json").read_text())
    gguf = Path(RUNS_DIR) / run / "gguf" / f"{run}-{quant}.gguf"
    rows = _rows(split, language, limit)
    out = _serve_and_eval(gguf, CONFIGS[config], rows, meta["fmt"])
    commit = Path("/opt/llama_commit.txt").read_text().strip()
    return {"llama": f"master@{commit[:9]}", "config": config, **out, "ids": [r["id"] for r in rows]}


@app.local_entrypoint()
def main(run: str = "qwen3.5-0.8b__json__full__v8", quant: str = "F16", split: str = "numbers",
         language: str = "bangla", configs: str = "baseline,np1,ub4,fa_off,kv_f32,t1", llama: str = "pinned",
         limit: int = 0, out: str = "ml/results/gguf_diag.json"):
    lang = language or None
    fn = diag_master if llama == "master" else diag_pinned
    cfgs = configs.split(",")
    results = list(fn.starmap([(run, quant, c, split, lang, limit) for c in cfgs]))
    # HF reference for the same rows
    hf_path = Path(out).parent / "hf_ref.json"
    print(f"\n{run} {quant} {split}/{language or 'all'} ({results[0]['n']} rows)")
    print("| llama | config | exact | amounts | wall s |")
    print("|---|---|---|---|---|")
    for r in results:
        print(f"| {r['llama']} | {r['config']} | {100 * r['exact']:.1f} | {100 * (r['amounts'] or 0):.1f} | {r['wall_s']} |")
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(p.read_text()) if p.exists() else []
    prev += [{k: v for k, v in r.items() if k != "preds"} | {"run": run, "quant": quant, "split": split, "language": language} for r in results]
    p.write_text(json.dumps(prev, indent=1, ensure_ascii=False))
    preds_dir = p.parent / "gguf_diag_preds"
    preds_dir.mkdir(exist_ok=True)
    for r in results:
        (preds_dir / f"{run}.{quant}.{r['llama'].replace('@', '_')}.{r['config']}.jsonl").write_text(
            "\n".join(json.dumps({"id": i, "pred": pr}, ensure_ascii=False) for i, pr in zip(r["ids"], r["preds"])), encoding="utf-8")
