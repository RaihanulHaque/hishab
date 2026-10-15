"""Candidate base models. All of them are verified to load in kiln (llama.cpp CPU)."""

ATTN_MLP = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LFM2_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]

MODELS = {
    "qwen3.5-0.8b": {
        "hf_id": "Qwen/Qwen3.5-0.8B",
        "lora_targets": ATTN_MLP,
        "batch_size": 8,
        "grad_accum": 2,
        "notes": "248k vocab -> big logits; hybrid attention; thinking switch in template (trained with thinking off)",
    },
    "lfm2.5-350m": {
        "hf_id": "LiquidAI/LFM2.5-350M",
        "lora_targets": LFM2_TARGETS,
        "batch_size": 32,
        "notes": "fastest on phone (90 tok/s Q4_K_M on Pixel 8), fragile at 4 bits",
    },
    "lfm2.5-1.2b": {
        "hf_id": "LiquidAI/LFM2.5-1.2B-Instruct",
        "lora_targets": LFM2_TARGETS,
        "batch_size": 16,
        "notes": "quality ceiling candidate; ~700 MB at Q4",
    },
    "gemma-3-270m": {
        "hf_id": "unsloth/gemma-3-270m-it",
        "lora_targets": ATTN_MLP,
        "batch_size": 32,
        "eval_batch_size": 1,  # batched left-padded generation corrupts Gemma3 (sliding window)
        "notes": "ungated unsloth mirror of google/gemma-3-270m-it; no system role, folded into user turn",
    },
}


def run_name(model_key: str, fmt: str, tune: str = "lora") -> str:
    return f"{model_key}__{fmt}__{tune}"
