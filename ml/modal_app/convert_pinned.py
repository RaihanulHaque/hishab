"""Run llama.cpp's convert_hf_to_gguf.py with the BPE pre-tokenizer type given instead of detected.

  python convert_pinned.py <pre> <convert_hf_to_gguf args...>

The converter recognises a tokenizer by hashing the ids it gives a fixed test string. A trimmed
vocabulary renumbers the tokens, so the hash no longer matches even though the pre-tokenizer
(the regex split) is unchanged; pin the type the untrimmed model gets (Qwen3.5: "qwen35").
"""
import runpy
import sys

LLAMA = "/opt/llama.cpp"
sys.path.insert(0, LLAMA)
pre = sys.argv.pop(1)

import conversion.base as base  # noqa: E402

for obj in vars(base).values():
    if isinstance(obj, type) and "get_vocab_base_pre" in vars(obj):
        obj.get_vocab_base_pre = lambda self, tokenizer, _pre=pre: _pre

sys.argv[0] = f"{LLAMA}/convert_hf_to_gguf.py"
runpy.run_path(sys.argv[0], run_name="__main__")
