import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

# Constants for targeted generation
CATEGORIES = [
    "food", "grocery", "rent", "utilities", "mobile_recharge", 
    "transport", "tuition", "salary", "gift", "loan", 
    "shopping", "medical", "entertainment", "other"
]

SYSTEM_MESSAGE = (
    "You are a financial assistant who understands English, Bengali and "
    "Banglish(Bengali written in english). Extract transactions from text into a JSON list."
)

class Transaction(BaseModel):
    description: str
    amount: float
    category: Literal[
        "food", "grocery", "rent", "utilities", "mobile_recharge", 
        "transport", "tuition", "salary", "gift", "loan", 
        "shopping", "medical", "entertainment", "other"
    ]
    type: Literal["income", "expense"]

class DatasetExample(BaseModel):
    input: str
    output: list[Transaction]
    lang: Literal["bangla", "english", "banglish", "mixed"]

class GeneratedBatch(BaseModel):
    examples: list[DatasetExample]

HARD_PROMPT = """
# Role
Generate "Hard" training data for a Banglish/Bangla finance transaction extractor.
The model currently fails on complex sentences, direction (income vs expense), and double-meaning words.

# Output
Return exactly a JSON object with key "examples". Generate 10-12 unique examples.

# Focus Areas for this batch:
1. **Direction Clarity**: Use verbs that clearly define income vs expense in Banglish/Bangla.
   - Income: 'pelam', 'paisi', 'peyechi', 'ashlo', 'dhuklo', 'payechi', 'received', 'credited'.
   - Expense: 'dilam', 'disi', 'disilam', 'kinlam', 'khaisi', 'khoroch holo', 'spent', 'paid', 'geche'.
   - Challenge: "tuition theke peyechi 3k" (Income) vs "tuition fee dilam 3k" (Expense).

2. **Double Meaning Words**:
   - 'vat' (rice in Bengali, category: food) vs 'VAT' (tax in English, category: other).
   - 'gas' (cooking gas, category: utilities) vs 'gas' (CNG/car fuel, category: transport).
   - 'net' (internet, category: utilities) vs 'net' (mesh/other context).

3. **High Complexity**:
   - Long sentences with 3+ transactions joined by 'ar', 'kintu' (but), 'sathe', 'pore'.
   - Example: "ajke vat khelam 30 taka, dudh kinlam 30 ar tuition theke pelam 1400 taka kintu doi kinte 1.4k taka geche"

4. **Numeric Slang**:
   - Use '1.4k', '5k', '2.5 lakh', '10k' mixed with '1200 taka', '500 tk'.

# Category Schema: {categories}

# Realism:
- Use realistic Bangladeshi daily amounts.
- Banglish should be messy: 'ajke', 'tk', 'k', 'taka', 'bhara', 'khoros'.
"""

def main():
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found.")
        return

    _DATASETS = Path(__file__).parent.parent / "datasets"
    output_file = _DATASETS / "synthetic_dataset.jsonl"
    chatml_file = _DATASETS / "synthetic_dataset_chatml.jsonl"

    client = genai.Client(api_key=api_key)
    target = 200 # Target 200 hard examples
    appended = 0
    
    print(f"Generating {target} hard examples using gemini-2.5-flash-lite...")

    while appended < target:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite", 
                contents=HARD_PROMPT.format(categories=", ".join(CATEGORIES)),
                config={
                    "response_mime_type": "application/json",
                    "response_schema": GeneratedBatch.model_json_schema(),
                },
            )
            
            generated = GeneratedBatch.model_validate_json(response.text)
            
            new_prod_lines = []
            new_chat_lines = []
            
            for ex in generated.examples:
                if appended >= target: break
                
                entry_id = f"txn_hard_{uuid.uuid4().hex[:8]}"
                # Production Format
                prod_entry = {
                    "id": entry_id,
                    "messages": [
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": ex.input},
                        {"role": "assistant", "content": json.dumps([t.model_dump() for t in ex.output], ensure_ascii=False, separators=(",", ":"))}
                    ],
                    "metadata": {
                        "lang": ex.lang,
                        "complexity": "complex" if len(ex.output) >= 3 else "medium",
                        "generated_by": "gemini-2.5-flash-lite-hard-prompt",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    },
                    "raw_transactions": [t.model_dump() for t in ex.output]
                }
                new_prod_lines.append(json.dumps(prod_entry, ensure_ascii=False) + "\n")
                new_chat_lines.append(json.dumps({"messages": prod_entry["messages"]}, ensure_ascii=False) + "\n")
                appended += 1
            
            with open(output_file, "a", encoding="utf-8") as f_prod:
                f_prod.writelines(new_prod_lines)
            with open(chatml_file, "a", encoding="utf-8") as f_chat:
                f_chat.writelines(new_chat_lines)
                
            print(f"Progress: {appended}/{target}")
            
        except Exception as e:
            print(f"Error: {e}. Retrying in 5s...")
            time.sleep(5)

    print(f"Done. Appended {appended} hard examples to {output_file}")

if __name__ == "__main__":
    main()
