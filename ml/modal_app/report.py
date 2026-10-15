"""Collect every eval json from the runs volume into ml/results/RESULTS.md and results.json.

  conda run -n ML modal run ml/modal_app/report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modal_app.common import REPO, RUNS_DIR, VOLUMES, app, gguf_image  # noqa: E402


@app.function(image=gguf_image, volumes=VOLUMES, timeout=600)
def collect() -> dict:
    out = {"hf": [], "gguf": []}
    for run_dir in sorted(Path(RUNS_DIR).iterdir()):
        f = run_dir / "eval_hf.json"
        if f.exists():
            out["hf"].append(json.loads(f.read_text()))
        for g in sorted((run_dir / "eval_gguf").glob("*.json")) if (run_dir / "eval_gguf").exists() else []:
            out["gguf"].append(json.loads(g.read_text()))
    return out


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def render(data: dict) -> str:
    lines = ["# Hishab fine-tune results", "", "Held-out **test** split (518 rows) unless noted. exact = multiset of "
             "(amount, category, type) equals gold. F1 = micro F1 over those tuples.", ""]
    lines += ["## Full-precision models (bf16, HF transformers, greedy)", "",
              "| run | base | fmt | tune | best val loss | valid JSON | exact | F1 | cat acc | type acc | banglish exact | bangla exact | english exact | mixed exact | avg out tok | train min |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(data["hf"], key=lambda r: -r["splits"]["test"]["all"]["exact_match"]):
        t = r["splits"]["test"]
        a, bl = t["all"], t["by_language"]
        lines.append(
            f"| {r['run']} | {r['hf_id']} | {r['fmt']} | {r['tune']} | {r['best_eval_loss']:.3f} | {pct(a['valid_json'])} | "
            f"**{pct(a['exact_match'])}** | {pct(a['tx_f1'])} | {pct(a['category_acc'])} | {pct(a['type_acc'])} | "
            f"{pct(bl.get('banglish', {}).get('exact_match'))} | {pct(bl.get('bangla', {}).get('exact_match'))} | "
            f"{pct(bl.get('english', {}).get('exact_match'))} | {pct(bl.get('mixed', {}).get('exact_match'))} | "
            f"{t['avg_completion_tokens']} | {r['train_time_s'] / 60:.0f} |"
        )
    if data["gguf"]:
        lines += ["", "## GGUF quantizations (llama.cpp CPU, kiln's commit, greedy)", "",
                  "| run | quant | MB | valid JSON | exact | F1 | banglish exact | truncated | tok/s per slot (16-core x86) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for g in sorted(data["gguf"], key=lambda g: (g["run"], -g["metrics"]["all"]["exact_match"])):
            a = g["metrics"]["all"]
            lines.append(
                f"| {g['run']} | {g['quant']} | {g['file_bytes'] / 1e6:.0f} | {pct(a['valid_json'])} | **{pct(a['exact_match'])}** | "
                f"{pct(a['tx_f1'])} | {pct(g['metrics']['by_language'].get('banglish', {}).get('exact_match'))} | "
                f"{g['truncated']} | {g['gen_tok_per_s_per_slot']} |"
            )
    return "\n".join(lines) + "\n"


@app.local_entrypoint()
def main():
    data = collect.remote()
    out = REPO / "ml" / "results"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))
    md = render(data)
    (out / "RESULTS.md").write_text(md)
    print(md)
