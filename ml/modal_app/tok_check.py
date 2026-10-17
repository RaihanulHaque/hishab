"""Which tokenizer is faithful to Qwen3.5's tokenizer.json on Bangla text?

Tokenizes the same NFC-normalized strings with (a) transformers' AutoTokenizer (the training
path), (b) the raw `tokenizers` library loading tokenizer.json directly, and (c) llama-server's
/tokenize on the F16 GGUF (the phone path), and prints the pieces side by side.

  conda run -n ML modal run ml/modal_app/tok_check.py --run qwen3.5-0.8b__json__full__v8
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
from modal_app.common import RUNS_DIR, VOLUMES, app, gguf_image  # noqa: E402

BIN = Path("/opt/llama.cpp/build/bin")
SAMPLES = [
    "এয়ারপোর্টের ট্রিপের ভাড়া নগদে সাড়ে আট হাজার পাইলাম",
    "তানিয়া আগের ধার থেকে আড়াই হাজার টাকা ফেরত দিল",
    "নয় হাজার পাঁচশো টাকা",
    "basha vara dilam ponero hajar",
]


@app.function(image=gguf_image, cpu=4, timeout=30 * 60, volumes=VOLUMES)
def check(run: str, hf_id: str) -> dict:
    import unicodedata

    import requests
    import tokenizers
    import transformers
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    out = {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__, "samples": []}
    merged = Path(RUNS_DIR) / run / "merged"
    tj_base = hf_hub_download(hf_id, "tokenizer.json")
    t_auto_base = AutoTokenizer.from_pretrained(hf_id)
    t_auto_merged = AutoTokenizer.from_pretrained(str(merged))
    t_raw_base = Tokenizer.from_file(tj_base)
    t_raw_merged = Tokenizer.from_file(str(merged / "tokenizer.json")) if (merged / "tokenizer.json").exists() else None
    out["pre_tokenizer_base"] = json.loads(Path(tj_base).read_text())["pre_tokenizer"]
    if (merged / "tokenizer.json").exists():
        out["pre_tokenizer_merged"] = json.loads((merged / "tokenizer.json").read_text())["pre_tokenizer"]
    out["auto_base_class"] = type(t_auto_base).__name__
    out["auto_base_backend"] = repr(getattr(t_auto_base, "backend_tokenizer", None))[:120] if hasattr(t_auto_base, "backend_tokenizer") else None

    gguf = Path(RUNS_DIR) / run / "gguf" / f"{run}-F16.gguf"
    port = 8080
    proc = subprocess.Popen([str(BIN / "llama-server"), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port), "-c", "512",
                             "-t", str(os.cpu_count() or 4), "--no-warmup", "--log-disable"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
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
    try:
        for s in SAMPLES:
            s = unicodedata.normalize("NFC", s)
            row = {"text": s}
            row["auto_base"] = t_auto_base.tokenize(s)
            row["auto_merged"] = t_auto_merged.tokenize(s)
            row["raw_base"] = t_raw_base.encode(s, add_special_tokens=False).tokens
            if t_raw_merged:
                row["raw_merged"] = t_raw_merged.encode(s, add_special_tokens=False).tokens
            srv = requests.post(f"http://127.0.0.1:{port}/tokenize", json={"content": s, "add_special": False, "with_pieces": True}, timeout=60).json()["tokens"]
            row["llama_cpp"] = [t["piece"] for t in srv]
            row["ids_auto_base"] = t_auto_base(s, add_special_tokens=False)["input_ids"]
            row["ids_llama_cpp"] = [t["id"] for t in srv]
            out["samples"].append(row)
    finally:
        proc.terminate()
    return out


@app.local_entrypoint()
def main(run: str = "qwen3.5-0.8b__json__full__v8", hf_id: str = "Qwen/Qwen3.5-0.8B", out: str = "ml/results/tok_check.json"):
    r = check.remote(run, hf_id)
    Path(out).write_text(json.dumps(r, indent=1, ensure_ascii=False))
    print("transformers", r["transformers"], "tokenizers", r["tokenizers"], "class", r["auto_base_class"])
    print("pre_tokenizer base  :", json.dumps(r["pre_tokenizer_base"], ensure_ascii=False)[:400])
    if "pre_tokenizer_merged" in r:
        print("pre_tokenizer merged:", json.dumps(r["pre_tokenizer_merged"], ensure_ascii=False)[:400])
    for s in r["samples"]:
        print("\n", s["text"])
        for k in ("auto_base", "auto_merged", "raw_base", "raw_merged", "llama_cpp"):
            if k in s:
                print(f"  {k:12s} ({len(s[k]):3d}): {s[k]}")
        print("  ids equal (auto_base vs llama.cpp):", s["ids_auto_base"] == s["ids_llama_cpp"])
