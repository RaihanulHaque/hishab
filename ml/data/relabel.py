"""Relabel held-out rows from scratch with a frontier model following docs/LABELLING_GUIDELINE.md.

The labeller never sees the existing gold, so its output is an independent second opinion.
Two labellers that agree with each other give a trustworthy test set; where they disagree, a
human (or a judge prompt) decides.

  # Gemini through the agy CLI (print mode, sandboxed, no tools):
  python ml/data/relabel.py --splits test numbers --labeller agy --model gemini-3.8-flash-high \
      --out ml/results/relabel/gemini.jsonl

  # Claude Sonnet through Claude Code print mode:
  python ml/data/relabel.py --splits train_v12 --sources codex-template-v2 --labeller claude --model sonnet \
      --tag sonnet --out ml/results/relabel/sonnet_templates.jsonl

  # Write the batch prompts to files so another model (e.g. a Claude Code subagent) can answer them:
  python ml/data/relabel.py --splits test numbers --emit-batches ml/results/relabel/batches

  # Merge the answer files such a model wrote back into one jsonl:
  python ml/data/relabel.py --collect ml/results/relabel/batches/answers --out ml/results/relabel/sonnet.jsonl

Output rows: {"id", "split", "instruction", "labels": [{description, amount, category, type}], "labeller"}.
Appends and skips ids already present in --out, so it can be re-run after a partial failure.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hishab.schema import CATEGORIES  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "ml/datasets/splits"
GUIDE = (ROOT / "docs/LABELLING_GUIDELINE.md").read_text()

PROMPT = """Answer directly from the text in this message. Do not run commands, do not read or write files, do not use any tools; just reply with the JSON.

You are labelling a Bangladeshi personal-finance dataset. Each message is something a user typed into
an expense-tracking app, in English, Bangla script, or Banglish (Bangla written in Latin letters), often mixed.
Extract every transaction strictly according to the guideline below. Amounts may be digits, Bangla digits,
or spelled out in words ("der hajar" = 1500, "sare tin hajar" = 3500, "পনেরো হাজার" = 15000, "2 lakh" = 200000,
"4.5k" = 4500). Amounts are in taka.

GUIDELINE
{guide}

For each item output an object: "id" (copied), "labels": a list of transactions, each
{{"description": short, in the message's own language/script, no amount; "amount": number;
"category": one of {cats}; "type": "expense" or "income"}}. Use [] when there is no transaction.

Output ONLY a JSON array with one object per item, in the same order as the items, no prose, no code fence.

ITEMS
{items}
"""


def load_rows(splits: list[str], sources: list[str] | None = None) -> list[dict]:
    rows = []
    for s in splits:
        for line in (SPLITS / f"{s}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if sources and r.get("source") not in sources:
                    continue
                rows.append({"id": r["id"], "split": s, "instruction": r["instruction"]})
    return rows


def prompt_for(batch: list[dict]) -> str:
    items = [{"id": r["id"], "message": r["instruction"]} for r in batch]
    return PROMPT.format(guide=GUIDE, cats=", ".join(CATEGORIES),
                         items=json.dumps(items, ensure_ascii=False, indent=0))


def parse_answer(text: str) -> dict[str, list[dict]]:
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return {}
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    out = {}
    for item in arr if isinstance(arr, list) else []:
        if not isinstance(item, dict) or "id" not in item or not isinstance(item.get("labels"), list):
            continue
        labels = []
        good = True
        for t in item["labels"]:
            try:
                labels.append({"description": str(t.get("description", "")), "amount": float(t["amount"]),
                               "category": t["category"], "type": t["type"]})
            except (KeyError, TypeError, ValueError):
                good = False
                break
            if labels[-1]["category"] not in CATEGORIES or labels[-1]["type"] not in ("income", "expense"):
                good = False
                break
        if good:
            out[item["id"]] = labels
    return out


def run_agy(prompt: str, model: str, timeout: int) -> str:
    cmd = ["agy", "--sandbox", "--model", model, "--output-format", "text", f"--print={prompt}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd="/tmp")
    except subprocess.TimeoutExpired:
        print(f"  agy timed out after {timeout}s", flush=True)
        return ""
    if r.returncode != 0:
        print(f"  agy exit {r.returncode}: {r.stderr[-300:]!r}", flush=True)
    return r.stdout


def run_claude(prompt: str, model: str, timeout: int) -> str:
    """Claude Code in print mode, no tools, a plain labeller system prompt (the default agent prompt
    treats data-labelling requests as suspicious instructions)."""
    cmd = ["claude", "-p", "--output-format", "text", "--tools", "", "--model", model, "--system-prompt",
           "You label a Bangladeshi personal-finance dataset. Reply with JSON only."]
    try:
        r = subprocess.run(cmd + [prompt], capture_output=True, text=True, timeout=timeout, cwd="/tmp")
    except subprocess.TimeoutExpired:
        print(f"  claude timed out after {timeout}s", flush=True)
        return ""
    if r.returncode != 0:
        print(f"  claude exit {r.returncode}: {r.stderr[-300:]!r}", flush=True)
    return r.stdout


def run_codex(prompt: str, model: str, timeout: int) -> str:
    """Codex (OpenAI) at low reasoning effort, read-only, no repo; the reply is the JSON array."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "out.txt"
        cmd = ["codex", "exec", "--ephemeral", "--skip-git-repo-check", "-s", "read-only", "-o", str(out),
               "-c", "model_reasoning_effort=low"]
        if model and model != "default":
            cmd += ["-m", model]
        try:
            subprocess.run(cmd + [prompt], capture_output=True, text=True, timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            print(f"  codex timed out after {timeout}s", flush=True)
            return ""
        return out.read_text() if out.exists() else ""


def already(out: Path) -> set[str]:
    if not out.exists():
        return set()
    return {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test", "numbers"])
    ap.add_argument("--sources", nargs="*", default=None, help="only rows whose `source` is in this list (e.g. base hard)")
    ap.add_argument("--labeller", choices=["agy", "claude", "codex"], default="agy")
    ap.add_argument("--model", default="gemini-3.8-flash-high")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="", help="k/n: label only every n-th row starting at k (parallel runs)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--emit-batches", type=Path, help="write batch prompts as files and exit")
    ap.add_argument("--collect", type=Path, help="directory of answer files (any name) to merge into --out")
    ap.add_argument("--tag", default="", help="labeller name written into rows (default: model)")
    ap.add_argument("--skip-from", type=Path, nargs="*", default=[], help="treat ids present in these jsonl files as already done")
    args = ap.parse_args()

    rows = load_rows(args.splits, args.sources)
    if args.limit:
        rows = rows[: args.limit]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        rows = rows[k::n]
    by_id = {r["id"]: r for r in rows}

    if args.emit_batches:
        args.emit_batches.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(rows), args.batch):
            (args.emit_batches / f"batch_{i // args.batch:03d}.txt").write_text(prompt_for(rows[i:i + args.batch]), encoding="utf-8")
        print(f"wrote {(len(rows) + args.batch - 1) // args.batch} batch prompts to {args.emit_batches}")
        return

    if not args.out:
        ap.error("--out is required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = already(args.out)
    for p in args.skip_from:
        done |= already(p)
    tag = args.tag or args.model

    if args.collect:
        n = 0
        with args.out.open("a", encoding="utf-8") as f:
            for p in sorted(args.collect.iterdir()):
                for rid, labels in parse_answer(p.read_text(encoding="utf-8")).items():
                    if rid in by_id and rid not in done:
                        f.write(json.dumps({**by_id[rid], "labels": labels, "labeller": tag}, ensure_ascii=False) + "\n")
                        done.add(rid)
                        n += 1
        print(f"collected {n} new rows -> {args.out} ({len(done)} total)")
        return

    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(todo)} rows to label with {args.model} ({len(done)} already done)")
    with args.out.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), args.batch):
            batch = todo[i:i + args.batch]
            t0 = time.time()
            runner = {"claude": run_claude, "codex": run_codex}.get(args.labeller, run_agy)
            answers = parse_answer(runner(prompt_for(batch), args.model, args.timeout))
            got = 0
            for r in batch:
                if r["id"] in answers:
                    f.write(json.dumps({**r, "labels": answers[r["id"]], "labeller": tag}, ensure_ascii=False) + "\n")
                    got += 1
            f.flush()
            print(f"batch {i // args.batch + 1}/{(len(todo) + args.batch - 1) // args.batch}: {got}/{len(batch)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
