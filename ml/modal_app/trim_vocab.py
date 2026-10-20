"""Trim a run's vocabulary to the tokens Hishab can use (see hishab/vocab_trim.py), as a new run.

  conda run -n ML modal run ml/modal_app/trim_vocab.py --run qwen3.5-0.8b__pipe_signed__full__v11 --target 38000

Writes /runs/<run>__trim<N>k/merged (sliced tied embedding, renumbered tokenizer, remapped special
ids) and eval_hf.json (the source run's metadata plus `vocab_trim`). No retraining: the
transformer layers are untouched. Then build and score GGUFs as usual with gguf.py, which pins
the tokenizer type for trimmed runs (the converter's fingerprint of the tokenizer changes).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modal_app.common import DATA_DIR, RUNS_DIR, VOLUMES, app, gguf_image, hf_secret  # noqa: E402

SPLITS = ["train_v10", "train_v12", "validation", "test", "test_clean", "numbers", "numbers_clean", "small_amounts"]


@app.function(image=gguf_image, cpu=8, memory=16384, timeout=60 * 60, volumes=VOLUMES, secrets=[hf_secret])
def trim(run: str, target: int = 38000) -> dict:
    import shutil

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from tokenizers import Tokenizer

    from hishab.prompts import SYSTEM_PROMPTS
    from hishab.schema import render_target
    from hishab.vocab_trim import select_tokens, trim_tokenizer_json

    src = Path(RUNS_DIR) / run
    meta = json.loads((src / "eval_hf.json").read_text())
    name = f"{run}__trim{round(target / 1000)}k"
    dst = Path(RUNS_DIR) / name
    if dst.exists():
        shutil.rmtree(dst)
    merged = dst / "merged"
    merged.mkdir(parents=True)
    for f in (src / "merged").iterdir():
        if f.suffix != ".safetensors":
            shutil.copy(f, merged / f.name)
    # the base repo's tokenizer files, as gguf.py's _restore_tokenizer_files uses (importing gguf.py
    # here would register its entrypoint a second time)
    from huggingface_hub import hf_hub_download, list_repo_files

    files = set(list_repo_files(meta["hf_id"]))
    for fname in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"):
        if fname in files:
            shutil.copy(hf_hub_download(meta["hf_id"], fname), merged / fname)

    # tokens used by every Hishab text (messages, both target formats, prompts, chat markup)
    tj = json.loads((merged / "tokenizer.json").read_text())
    # tokenizer_config can declare more special tokens (Qwen3.5: the audio markers, ids from 248070)
    # than tokenizer.json lists; transformers adds them at load time at those ids. Carry them over
    # so every id the config mentions still exists after renumbering.
    tc0 = json.loads((merged / "tokenizer_config.json").read_text())
    have = {t["id"] for t in tj["added_tokens"]}
    for k, v in sorted(tc0.get("added_tokens_decoder", {}).items(), key=lambda kv: int(kv[0])):
        if int(k) not in have:
            tj["added_tokens"].append({"id": int(k), "content": v["content"], "single_word": v.get("single_word", False),
                                       "lstrip": v.get("lstrip", False), "rstrip": v.get("rstrip", False),
                                       "normalized": v.get("normalized", False), "special": v.get("special", True)})
    tok = Tokenizer.from_str(json.dumps(tj))
    texts = list(SYSTEM_PROMPTS.values())
    texts.append("<|im_start|>system\nx<|im_end|>\n<|im_start|>user\ny<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    for s in SPLITS:
        p = Path(DATA_DIR) / "splits" / f"{s}.jsonl"
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            texts.append(r["instruction"])
            for fmt in ("json", "pipe_signed"):
                texts.append(render_target(fmt, r.get("translation"), r["transactions"]))
    used = set()
    for e in tok.encode_batch(texts, add_special_tokens=False):
        used.update(e.ids)
    keep = select_tokens(tj, used, target)
    new_tj, old_ids = trim_tokenizer_json(tj, keep)
    remap = {old: new for new, old in enumerate(old_ids)}

    # every Hishab text must split into the same pieces as before
    new_tok = Tokenizer.from_str(json.dumps(new_tj))
    bad = sum(
        o.tokens != n.tokens
        for o, n in zip(tok.encode_batch(texts, add_special_tokens=False), new_tok.encode_batch(texts, add_special_tokens=False))
    )
    if bad:
        raise RuntimeError(f"{bad} texts tokenize differently after trimming")
    (merged / "tokenizer.json").write_text(json.dumps(new_tj, ensure_ascii=False))

    # tokenizer_config: added_tokens_decoder is keyed by id
    tcp = merged / "tokenizer_config.json"
    tc = json.loads(tcp.read_text())
    if "added_tokens_decoder" in tc:
        tc["added_tokens_decoder"] = {str(remap[int(k)]): v for k, v in tc["added_tokens_decoder"].items()}
    tcp.write_text(json.dumps(tc, indent=2, ensure_ascii=False))

    # config / generation_config special ids and vocab size
    def fix_ids(d: dict) -> dict:
        for k in ("eos_token_id", "pad_token_id", "bos_token_id", "image_token_id", "video_token_id",
                  "vision_start_token_id", "vision_end_token_id"):
            v = d.get(k)
            if isinstance(v, int):
                d[k] = remap[v]
            elif isinstance(v, list):
                d[k] = [remap[x] for x in v]
        return d

    cfg = fix_ids(json.loads((merged / "config.json").read_text()))
    old_vocab = cfg["vocab_size"]
    cfg["vocab_size"] = len(old_ids)
    (merged / "config.json").write_text(json.dumps(cfg, indent=2))
    gcp = merged / "generation_config.json"
    if gcp.exists():
        gcp.write_text(json.dumps(fix_ids(json.loads(gcp.read_text())), indent=2))

    # weights: slice every tensor with a vocab-sized first dimension (the tied embedding)
    index = torch.tensor(old_ids, dtype=torch.long)
    sliced = []
    for f in sorted((src / "merged").glob("*.safetensors")):
        out = {}
        with safe_open(str(f), framework="pt") as sf:
            md = sf.metadata()
            for k in sf.keys():
                t = sf.get_tensor(k)
                if t.dim() >= 1 and t.shape[0] == old_vocab:
                    t = t.index_select(0, index).contiguous()
                    sliced.append(k)
                out[k] = t
        save_file(out, str(merged / f.name), metadata=md)
    if not sliced:
        raise RuntimeError("no vocab-sized tensor found")
    print("sliced:", sliced)

    info = {"source_run": run, "target": target, "kept": len(old_ids), "old_vocab": old_vocab,
            "used_by_data": len(used), "sliced": sliced}
    meta = {**meta, "run": name, "vocab_trim": info}
    (dst / "eval_hf.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    (dst / "vocab_trim_old_ids.json").write_text(json.dumps(old_ids))
    VOLUMES[RUNS_DIR].commit()
    print(json.dumps(info, indent=1))
    return info


@app.local_entrypoint()
def main(run: str, target: int = 38000):
    print(trim.remote(run, target))
