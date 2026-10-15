"""Convert a trained run to GGUF, quantize it many ways (with and without an importance
matrix built from the training prompts) and evaluate every file with llama.cpp itself, the
same code and commit kiln runs on the phone.

  conda run -n ML modal run ml/modal_app/gguf.py --runs qwen3.5-0.8b__json__lora \
      [--quants F16,Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q4_K_M-imat,Q4_0,Q4_0-imat,IQ4_XS-imat,Q3_K_M-imat,IQ3_M-imat] \
      [--split test] [--push]

Outputs under /runs/<run>/gguf/: <run>-<quant>.gguf, imatrix.gguf, calib.txt and
/runs/<run>/eval_gguf/<quant>.json (metrics + tokens/s on the container CPU). --push uploads
every GGUF to the HF repo ryuma007/hishab-<run>-GGUF.
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
from modal_app.common import DATA_DIR, RUNS_DIR, VOLUMES, app, gguf_image, hf_secret  # noqa: E402

LLAMA = Path("/opt/llama.cpp")
BIN = LLAMA / "build" / "bin"
DEFAULT_QUANTS = "F16,Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q4_K_M-imat,Q4_0,Q4_0-imat,IQ4_XS-imat,Q3_K_M-imat,IQ3_M-imat"
MAX_TOKENS = {"json": 192, "two_phase": 320}
N_PARALLEL = 8


def _run(cmd: list[str], **kw) -> None:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def _load_split(name: str) -> list[dict]:
    p = Path(DATA_DIR) / "splits" / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _run_meta(run: str) -> dict:
    return json.loads((Path(RUNS_DIR) / run / "eval_hf.json").read_text())


@app.function(image=gguf_image, cpu=16, memory=32768, timeout=3 * 60 * 60, volumes=VOLUMES)
def make_ggufs(run: str, quants: list[str], force: bool = False) -> dict:
    """F16 conversion, imatrix on in-domain text, then every requested quant."""
    from hishab.prompts import build_messages
    from hishab.schema import render_target

    meta = _run_meta(run)
    fmt = meta["fmt"]
    run_dir = Path(RUNS_DIR) / run
    gdir = run_dir / "gguf"
    f16 = gdir / f"{run}-F16.gguf"
    merged_mtime = max(p.stat().st_mtime for p in (run_dir / "merged").glob("*.safetensors"))
    if f16.exists() and f16.stat().st_mtime < merged_mtime:
        print("F16 GGUF is older than the merged weights (run was retrained): rebuilding everything")
        force = True
    if force and gdir.exists():
        import shutil
        shutil.rmtree(gdir)
        edir = run_dir / "eval_gguf"
        if edir.exists():
            shutil.rmtree(edir)
    gdir.mkdir(exist_ok=True)
    f16 = gdir / f"{run}-F16.gguf"
    if not f16.exists():
        _restore_tokenizer_files(meta["hf_id"], run_dir / "merged")
        cmd = [sys.executable, LLAMA / "convert_hf_to_gguf.py", run_dir / "merged", "--outtype", "f16", "--outfile", f16]
        try:
            _run(cmd)
        except subprocess.CalledProcessError:
            # transformers-version mismatch fallback: the generic fast tokenizer class loads any tokenizer.json
            cfg = run_dir / "merged" / "tokenizer_config.json"
            d = json.loads(cfg.read_text())
            if d.get("tokenizer_class") == "TokenizersBackend":
                print("retrying with tokenizer_class=PreTrainedTokenizerFast")
                d["tokenizer_class"] = "PreTrainedTokenizerFast"
                cfg.write_text(json.dumps(d, indent=2, ensure_ascii=False))
                _run(cmd)
            else:
                raise

    calib = gdir / "calib.txt"
    if not calib.exists():
        # In-domain calibration: rendered chat turns from the training split (the phone will only
        # ever see this kind of text). Uses the tokenizer's template via transformers.
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(run_dir / "merged"))
        rows = _load_split("train")[:600]
        texts = [
            tok.apply_chat_template(
                build_messages(fmt, r["instruction"], render_target(fmt, r["translation"], r["transactions"])),
                tokenize=False,
            )
            for r in rows
        ]
        calib.write_text("\n".join(texts), encoding="utf-8")
    imatrix = gdir / "imatrix.gguf"
    if any(q.endswith("-imat") for q in quants) and not imatrix.exists():
        _run([BIN / "llama-imatrix", "-m", f16, "-f", calib, "-c", "512", "--chunks", "200", "-t", "16", "-o", imatrix])

    sizes = {"F16": f16.stat().st_size}
    for q in quants:
        if q == "F16":
            continue
        out = gdir / f"{run}-{q}.gguf"
        if not out.exists():
            base = q.removesuffix("-imat")
            cmd = [BIN / "llama-quantize"]
            if q.endswith("-imat"):
                cmd += ["--imatrix", imatrix]
            cmd += [f16, out, base, "16"]
            _run(cmd)
        sizes[q] = out.stat().st_size
    VOLUMES[RUNS_DIR].commit()
    print(json.dumps({k: f"{v / 1e6:.0f} MB" for k, v in sizes.items()}, indent=1))
    return sizes


TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "added_tokens.json",
                   "special_tokens_map.json", "chat_template.jinja", "chat_template.json")


def _restore_tokenizer_files(hf_id: str, merged: Path) -> None:
    """Overwrite the merged dir's tokenizer files with the base repo's originals.

    Fine-tuning never changes the tokenizer, but transformers 5 re-serialises it with an empty
    `added_tokens_decoder` and no `tokenizer.model`, and llama.cpp's converter then cannot tell
    that e.g. Gemma's <start_of_turn> is a control token (it got split into 7 text pieces and
    the model echoed the prompt). The base repo's files are what the stock GGUFs were made from."""
    from huggingface_hub import hf_hub_download, list_repo_files

    try:
        files = set(list_repo_files(hf_id))
    except Exception as e:  # offline / gated: keep going with what we have
        print("could not list base repo files:", e)
        return
    for name in TOKENIZER_FILES:
        if name in files:
            print("restoring", name, "from", hf_id)
            src = hf_hub_download(hf_id, name)
            (merged / name).write_bytes(Path(src).read_bytes())


class LlamaServer:
    def __init__(self, gguf: Path, ctx: int = 2048, port: int = 8080):
        self.port = port
        self.proc = subprocess.Popen(
            [str(BIN / "llama-server"), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port),
             "-c", str(ctx * N_PARALLEL), "-np", str(N_PARALLEL), "-t", str(os.cpu_count() or 8),
             "--jinja", "--no-warmup", "--log-disable"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )

    def wait(self, timeout: float = 300) -> None:
        import requests

        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server died:\n" + (self.proc.stderr.read() or "")[-3000:])
            try:
                if requests.get(f"http://127.0.0.1:{self.port}/health", timeout=2).status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1)
        raise TimeoutError("llama-server did not become healthy")

    def chat(self, messages: list[dict], max_tokens: int) -> dict:
        import requests

        r = requests.post(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            json={"messages": messages, "max_tokens": max_tokens, "temperature": 0, "seed": 0,
                  "chat_template_kwargs": {"enable_thinking": False}},
            timeout=600,
        )
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


@app.function(image=gguf_image, cpu=16, memory=16384, timeout=2 * 60 * 60, volumes=VOLUMES)
def eval_gguf(run: str, quant: str, split: str = "test", limit: int = 0) -> dict:
    from hishab.metrics import evaluate
    from hishab.prompts import build_messages

    meta = _run_meta(run)
    fmt = meta["fmt"]
    gguf = Path(RUNS_DIR) / run / "gguf" / f"{run}-{quant}.gguf"
    rows = _load_split(split)
    if limit:
        rows = rows[:limit]
    server = LlamaServer(gguf)
    server.wait()
    try:
        def one(r: dict) -> dict:
            resp = server.chat(build_messages(fmt, r["instruction"]), MAX_TOKENS[fmt])
            ch = resp["choices"][0]
            text = ch["message"].get("content") or ""
            timings = resp.get("timings", {})
            return {"pred": text.strip(), "finish": ch.get("finish_reason"),
                    "n_out": resp.get("usage", {}).get("completion_tokens", 0),
                    "n_in": resp.get("usage", {}).get("prompt_tokens", 0),
                    "pred_ms": timings.get("predicted_ms", 0.0), "prompt_ms": timings.get("prompt_ms", 0.0)}

        t0 = time.time()
        with ThreadPoolExecutor(N_PARALLEL) as ex:
            outs = list(ex.map(one, rows))
        wall = time.time() - t0
    finally:
        server.close()

    metrics = evaluate(rows, [o["pred"] for o in outs])
    n_out = sum(o["n_out"] for o in outs)
    pred_ms = sum(o["pred_ms"] for o in outs)
    result = {
        "run": run, "quant": quant, "split": split, "n": len(rows),
        "file_bytes": gguf.stat().st_size,
        "avg_completion_tokens": round(n_out / len(rows), 1),
        "avg_prompt_tokens": round(sum(o["n_in"] for o in outs) / len(rows), 1),
        "truncated": sum(o["finish"] == "length" for o in outs),
        "gen_tok_per_s_per_slot": round(n_out / (pred_ms / 1000), 1) if pred_ms else None,
        "wall_s": round(wall, 1),
        "metrics": metrics,
    }
    edir = Path(RUNS_DIR) / run / "eval_gguf"
    edir.mkdir(exist_ok=True)
    (edir / f"{quant}.{split}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    with (edir / f"{quant}.{split}.preds.jsonl").open("w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({"id": r["id"], "language": r["language"], "instruction": r["instruction"],
                                "gold": r["transactions"], **o}, ensure_ascii=False) + "\n")
    VOLUMES[RUNS_DIR].commit()
    a = metrics["all"]
    print(f"{run} {quant}: exact={a['exact_match']:.3f} f1={a['tx_f1']:.3f} valid={a['valid_json']:.3f} "
          f"size={result['file_bytes'] / 1e6:.0f}MB tok/s/slot={result['gen_tok_per_s_per_slot']}")
    return result


@app.function(image=gguf_image, cpu=4, timeout=60 * 60, volumes=VOLUMES, secrets=[hf_secret])
def push_ggufs(run: str, repo_prefix: str = "ryuma007/hishab-") -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    repo = f"{repo_prefix}{run.replace('__', '-')}-GGUF"
    api.create_repo(repo, repo_type="model", exist_ok=True)
    gdir = Path(RUNS_DIR) / run / "gguf"
    api.upload_folder(folder_path=str(gdir), repo_id=repo, allow_patterns=["*.gguf"], ignore_patterns=["imatrix.gguf"])
    edir = Path(RUNS_DIR) / run / "eval_gguf"
    if edir.exists():
        api.upload_folder(folder_path=str(edir), repo_id=repo, path_in_repo="eval", allow_patterns=["*.json"])
    return f"https://huggingface.co/{repo}"


@app.local_entrypoint()
def main(runs: str, quants: str = DEFAULT_QUANTS, split: str = "test", limit: int = 0, push: bool = False,
         skip_eval: bool = False, force: bool = False):
    qs = [q.strip() for q in quants.split(",") if q.strip()]
    run_list = [r.strip() for r in runs.split(",") if r.strip()]
    sizes = dict(zip(run_list, make_ggufs.map(run_list, [qs] * len(run_list), kwargs={"force": force})))
    if not skip_eval:
        jobs = [(r, q) for r in run_list for q in qs]
        results = list(eval_gguf.map([j[0] for j in jobs], [j[1] for j in jobs], kwargs={"split": split, "limit": limit}))
        print("\nrun | quant | MB | exact | F1 | valid | trunc | tok/s/slot")
        for res in results:
            a = res["metrics"]["all"]
            print(f"{res['run']} | {res['quant']} | {res['file_bytes'] / 1e6:.0f} | {a['exact_match']:.3f} | "
                  f"{a['tx_f1']:.3f} | {a['valid_json']:.3f} | {res['truncated']} | {res['gen_tok_per_s_per_slot']}")
    if push:
        for r in run_list:
            print("pushed:", push_ggufs.remote(r))
