from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import os
import random

OUTPUT_FILE = str(Path(__file__).parent.parent / "datasets" / "gemma_pruning_corpus_50mb.txt")
MAX_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

def build_compact_corpus():
    print(f"Creating compact corpus: {OUTPUT_FILE} (Target: 50MB)")
    
    current_size = 0
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # 1. Load Datasets
        print("Loading datasets...")
        nmt_ds = load_dataset("csebuetnlp/BanglaNMT", revision="refs/convert/parquet", split="train")
        banglish_ds = load_dataset("Ayon128/Banglish-English", split="train")

        # 2. Priority: Banglish (It's small, take all of it first)
        print("Processing Banglish dataset (Priority)...")
        for row in tqdm(banglish_ds, desc="Banglish"):
            text = f"{row.get('Banglish', '')} {row.get('English', '')}\n"
            f.write(text)
            current_size += len(text.encode('utf-8'))

        # 3. Fill remaining space with NMT (English/Bangla)
        # We shuffle a bit to get a diverse sample rather than just the first few rows
        print(f"\nFilling remaining space with NMT data...")
        indices = list(range(len(nmt_ds)))
        random.shuffle(indices) # Ensures we don't just get one category of text
        
        for i in tqdm(indices, desc="NMT Sampling"):
            if current_size >= MAX_SIZE_BYTES:
                break
                
            row = nmt_ds[i]
            text = f"{row.get('bn', '')} {row.get('en', '')}\n"
            encoded_text = text.encode('utf-8')
            
            f.write(text)
            current_size += len(encoded_text)

    print(f"\nSuccess! Final size: {os.path.getsize(OUTPUT_FILE) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    build_compact_corpus()