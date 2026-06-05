from pathlib import Path
from unsloth import FastLanguageModel
import torch
from collections import Counter
from tqdm import tqdm

model_name = "unsloth/gemma-4-E2B-it" # or "google/gemma-4-e2b"

# 1. Load Tokenizer and Model (in 4-bit to save your RAM)
model, processor = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048,
)

# 2. Token Frequency Analysis
CORPUS_FILE = str(Path(__file__).parent.parent / "ml" / "datasets" / "gemma_pruning_corpus_50mb.txt")
token_counts = Counter()

# 2. Extract the actual text tokenizer
# Gemma 4 uses a Processor; the tokenizer is stored at .tokenizer
tokenizer = processor.tokenizer

# 3. Frequency Analysis
token_counts = Counter()
# print(f"Step 2: Analyzing {corpus_path}...")

with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Counting Tokens"):
        clean_line = line.strip()
        if clean_line:
            # Direct call to the tokenizer's encode method
            tokens = tokenizer.encode(clean_line, add_special_tokens=False)
            token_counts.update(tokens)


# 4. Results
special_tokens = set(tokenizer.all_special_ids)
keep_tokens = set(token_counts.keys()) | special_tokens

print(f"\n--- Analysis Complete ---")
print(f"Original Vocab Size: {len(tokenizer)}")
print(f"Unique Tokens used (Bangla/Eng/Banglish): {len(token_counts)}")
print(f"Total to KEEP: {len(keep_tokens)}")
print(f"Total to DELETE: {len(tokenizer) - len(keep_tokens)}")

# Result
# --- Analysis Complete ---
# Original Vocab Size: 262144
# Unique Tokens used (Bangla/Eng/Banglish): 80248
# Total to KEEP: 80270
# Total to DELETE: 181874

import torch

def safe_prune(model, tokenizer, keep_tokens, save_path="./gemma-4-banglish-small"):
    # 1. Clean the Keep List
    # Ensure no ID exceeds the actual weight matrix dimensions
    max_valid_id = model.get_input_embeddings().weight.shape[0] - 1
    sorted_keep_ids = sorted([idx for idx in keep_tokens if idx <= max_valid_id])
    new_vocab_size = len(sorted_keep_ids)
    
    print(f"Pruning: Keeping {new_vocab_size} valid tokens. (Max ID in model: {max_valid_id})")

    # 2. Move weights to CPU to avoid CUDA assert crashes
    device = model.device
    print(f"Moving weights to CPU for surgery...")
    
    # Slice Input Embeddings
    old_embeddings = model.get_input_embeddings().weight.detach().cpu()
    new_embeddings = old_embeddings[sorted_keep_ids].clone()
    
    # Slice LM Head
    old_lm_head = model.get_output_embeddings().weight.detach().cpu()
    new_lm_head = old_lm_head[sorted_keep_ids].clone()

    # 3. Reconstruct Model Layers
    print("Rebuilding layers and moving back to device...")
    model.config.vocab_size = new_vocab_size
    
    # Re-wrap as Parameters and move back to original device (GPU)
    model.get_input_embeddings().weight = torch.nn.Parameter(new_embeddings.to(device))
    model.get_output_embeddings().weight = torch.nn.Parameter(new_lm_head.to(device))

    # 4. Save
    print(f"Saving to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print("Surgery successful!")
    return sorted_keep_ids

# Execute the safe version
sorted_ids = safe_prune(model, tokenizer, keep_tokens)

# Pruning: Keeping 80270 valid tokens. (Max ID in model: 262143)
# Moving weights to CPU for surgery...
# Rebuilding layers and moving back to device...
# Saving to ./gemma-4-banglish-small...
# Writing model shards: 100%
#  1/1 [00:12<00:00, 12.50s/it]
# Surgery successful!