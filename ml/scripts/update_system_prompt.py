import json
import os
from pathlib import Path

_DATASETS = Path(__file__).parent.parent / "datasets"
jsonl_file = str(_DATASETS / "synthetic_dataset.jsonl")
chatml_file = str(_DATASETS / "synthetic_dataset_chatml.jsonl")

# Get the rows in the jsonl file
from typing import List
from pydantic import BaseModel
from pprint import pprint

new_system_prompt = "You are a financial assistant who understands English, Bengali and Banglish(Bengali written in english). Extract transactions from text into a JSON list."


print("=== PRODUCTION FORMAT (with metadata) ===")
# with open(jsonl_file, "r") as f:
#     count = 0
#     for line in f:
#         if line.strip():  # Skip empty lines
#             count += 1
    
#     print(f"Number of entries in production dataset: {count}")
    
#     # Show a sample
#     with open(jsonl_file, "r") as f2:
#         first_line = f2.readline()
#         if first_line:
#             print("\nFirst entry (production format):")
#             first_entry = json.loads(first_line)
#             print(f"ID: {first_entry['id']}")
#             print(f"Language: {first_entry['metadata']['lang']}")
#             print(f"Complexity: {first_entry['metadata']['complexity']}")
#             print(f"Input: {first_entry['messages'][1]['content']}")
#             print(f"Raw Transactions: {len(first_entry['raw_transactions'])}")

print(f"\n=== CHATML FORMAT (for fine-tuning) ===")
with open(chatml_file, "r") as f:
    count = 0
    # Change the system prompt in the ChatML file to the new one
    for line in f:
        # print(line)
        line_data = json.loads(line)
        if line_data["messages"][0]["role"] == "system":
            line_data["messages"][0]["content"] = new_system_prompt
        count += 1
    
    print(f"Number of entries in ChatML dataset: {count}")


# Read all lines, modify in memory, then write back
updated_lines = []
with open(chatml_file, "r") as f_read:
    for line in f_read:
        if line.strip():  # Skip empty lines
            line_data = json.loads(line)
            if line_data["messages"][0]["role"] == "system":
                line_data["messages"][0]["content"] = new_system_prompt
            updated_lines.append(json.dumps(line_data) + "\n")

# Write the updated data back to the file
with open(chatml_file, "w") as f:
    f.writelines(updated_lines)

print("System prompt updated successfully.")
    
# Show a sample
# with open(chatml_file, "r") as f2:
#     first_line = f2.readline()
#     if first_line:
#         print("\nFirst entry (ChatML format):")
#         first_entry = json.loads(first_line)
#         pprint(first_entry)