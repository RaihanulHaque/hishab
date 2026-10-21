"""Collect every registered version's scores from the `hishab-runs` volume in one call.

  conda run -n ML modal run ml/modal_app/collect_results.py

Reads ml/results/versions.yaml, and for each version returns: the bf16 eval (eval_hf.json) per split,
training facts (train_log.json), every phone-file eval (eval_gguf/<quant>.<split>.json), and an error
breakdown per split computed from the prediction files. Writes ml/results/results_cache.json, which
ml/results/build_tables.py renders into ml/results/RESULTS_TABLES.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modal_app.common import RUNS_DIR, VOLUMES, app, gguf_image  # noqa: E402



def _breakdown(lines: list[str]) -> dict:
    from collections import Counter

    from hishab.metrics import score_example
    from hishab.schema import parse_output

    kinds, swaps = Counter(), Counter()
    n = 0
    for line in lines:
        r = json.loads(line)
        n += 1
        s = score_example(r["pred"], r["gold"])
        if s["exact"]:
            continue
        kinds["count" if not s["count_ok"] else "amount" if not s["amounts_ok"] else
              "type" if s["type_hits"] < s["aligned"] else "category"] += 1
        pool = list(parse_output(r["pred"])["txs"] or [])
        for g in r["gold"]:
            for i, p in enumerate(pool):
                if round(p["amount"], 2) == round(float(g["amount"]), 2):
                    if p["category"] != g["category"]:
                        swaps[f'{g["category"]} to {p["category"]}'] += 1
                    pool.pop(i)
                    break
    return {"n": n, "wrong": sum(kinds.values()), **{k: kinds.get(k, 0) for k in ("count", "amount", "type", "category")},
            "swaps": swaps.most_common(3)}


@app.function(image=gguf_image, cpu=2, memory=4096, timeout=1800, volumes=VOLUMES)
def collect(registry: dict) -> dict:
    out = {}
    runs = {v["run"] for v in registry["versions"]} | {v["gguf_run"] for v in registry["versions"]}
    for run in sorted(runs):
        d = Path(RUNS_DIR) / run
        if not d.exists():
            out[run] = {"missing": True}
            continue
        rec: dict = {"bf16": {}, "gguf": {}, "errors_bf16": {}, "errors_gguf": {}}
        ev = d / "eval_hf.json"
        if ev.exists():
            e = json.loads(ev.read_text())
            rec["meta"] = {k: e.get(k) for k in ("fmt", "tune", "epochs", "lr", "train_rows", "train_time_s",
                                                  "best_eval_loss", "train_split", "val_split", "tag")}
            for split, m in e.get("splits", {}).items():
                rec["bf16"][split] = {"all": m["all"], "by_language": m["by_language"],
                                      "avg_tokens": m.get("avg_completion_tokens")}
        tl = d / "train_log.json"
        if tl.exists():
            log = json.loads(tl.read_text())
            rec["train_log"] = {"val_loss": [h["eval_loss"] for h in log if "eval_loss" in h],
                                "final_loss": next((h["loss"] for h in reversed(log) if "loss" in h), None)}
        for p in sorted(d.glob("preds_*.jsonl")):
            rec["errors_bf16"][p.stem[len("preds_"):]] = _breakdown(p.read_text(encoding="utf-8").splitlines())
        gd = d / "eval_gguf"
        if gd.exists():
            for p in sorted(gd.glob("*.json")):
                if p.name.startswith("mtp_bench"):
                    continue
                quant, split = p.name[: -len(".json")].rsplit(".", 1)
                g = json.loads(p.read_text())
                rec["gguf"][f"{quant}|{split}"] = {"all": g["metrics"]["all"], "by_language": g["metrics"]["by_language"],
                                                   "file_mb": round(g.get("file_bytes", 0) / 1e6)}
                pp = gd / f"{quant}.{split}.preds.jsonl"
                if pp.exists() and quant == "Q4_K_M-imat":
                    rec["errors_gguf"][split] = _breakdown(pp.read_text(encoding="utf-8").splitlines())
        out[run] = rec
    return out


@app.local_entrypoint()
def main():
    import yaml

    ROOT = Path(__file__).resolve().parents[2]  # local only: in the container the file sits at /root
    reg = yaml.safe_load((ROOT / "ml/results/versions.yaml").read_text())
    res = collect.remote(reg)
    (ROOT / "ml/results/results_cache.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"{len(res)} runs -> ml/results/results_cache.json")
