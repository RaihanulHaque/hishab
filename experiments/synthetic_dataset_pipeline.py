from pydantic import BaseModel, Field, field_validator, RootModel
from google import genai
from typing import List, Literal, Optional, Union
from datetime import datetime, timezone
import uuid
from pprint import pprint
from dotenv import load_dotenv
import os
import json

load_dotenv()

# --- 1. The Core Transaction Logic ---
class Transaction(BaseModel):
    description: str = Field(..., description="Short text identifying the item")
    amount: float = Field(..., ge=0, description="Amount in value, must be positive")
    
    # Expanded Categories for BD/Global Context
    category: Literal[
        "food",             # Restaurants, Snacks (Khabar)
        "grocery",          # Bazaar, Raw ingredients, Super shop
        "rent",             # Basha vara, Mess vara
        "utilities",        # Electricity, Gas, Water, Internet, ISP bill
        "mobile_recharge",  # Flexiload, Data pack (Very common in BD)
        "transport",        # Rickshaw, Bus, Uber, Fuel
        "tuition",          # Fees, Books, Photocopy (Education)
        "salary",           # Salary, Tuition income
        "gift",             # Salami, Gift from Abbu
        "loan",             # Dhar deya/neya
        "shopping",         # Clothes, Gadgets (Daraz/Amazon)
        "medical",          # Doctor, Medicine
        "entertainment",    # Movies, Netflix, Subscriptions
        "other"
    ]
    type: Literal["income", "expense"]

# --- 2. Metadata for Tracking Data Quality ---
class Metadata(BaseModel):
    lang: Literal["bangla", "english", "banglish", "mixed"]
    complexity: Literal["simple", "medium", "complex"] # simple=1 txn, medium=2-3, complex=4+
    generated_by: Literal["gemini-2.5-flash-lite"]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Quality Control Flags
    validated: bool = False
    validation_notes: Optional[str] = None
    
    # Cost tracking (Optional but good for production)
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None

# --- 3. The Dataset Entry with ChatML Export ---
class DatasetEntry(BaseModel):
    id: str = Field(default_factory=lambda: f"txn_{uuid.uuid4().hex[:8]}")
    input: str 
    output: List[Transaction] 
    metadata: Metadata

    def to_chatml_format(self):
        """
        Converts entry to the Standard Chat/ChatML format.
        Structure: {"messages": [{"role": "system", ...}, {"role": "user", ...}]}
        """
        # 1. Serialize the list of objects into a JSON string
        assistant_content = "[" + ", ".join([t.model_dump_json() for t in self.output]) + "]"

        # 2. Return the dictionary structure
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a financial assistant. Extract transactions from text into a JSON list."
                },
                {
                    "role": "user",
                    "content": self.input
                },
                {
                    "role": "assistant",
                    "content": assistant_content
                }
            ]
        }
    
    def to_production_format(self):
        """
        Converts entry to production format with metadata.
        Includes both ChatML messages and rich metadata for tracking.
        """
        return {
            "id": self.id,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a financial assistant. Extract transactions from text into a JSON list."
                },
                {
                    "role": "user",
                    "content": self.input
                },
                {
                    "role": "assistant",
                    "content": "[" + ", ".join([t.model_dump_json() for t in self.output]) + "]"
                }
            ],
            "metadata": self.metadata.model_dump(),
            "raw_transactions": [t.model_dump() for t in self.output]
        }
    
class ListOfDatasetEntries(RootModel):
    # A list wrapper for multiple DatasetEntry objects
    root: List[DatasetEntry]
    

system_prompt = """
# Role
You are the **Bangladeshi Financial Data Synthesizer**. Your task is to generate realistic, messy, and culturally accurate training data for a financial tracking AI.

# Goal
Generate a batch of **5 to 10 unique training examples**.
Each example must map a realistic User Input (in Bangla, English, or Banglish) to a strict JSON Output.

# 1. DISTRIBUTION RULES (CRITICAL)
Real life is random. You **MUST** vary the number of transactions in your generated examples according to this probability:
- **40% Single Transaction:** Short, quick updates. (e.g., "Lunch 120tk")
- **40% Multiple Transactions (2-3 items):** A daily summary. (e.g., "Rickshaw 50, Lunch 200, r recharge korlam 20tk")
- **20% Complex/Messy (4+ items):** A full day ledger or messy paragraph. (e.g., "Bazaar 2k, gari vara 100, tuition pelam 5000, but friend ke dhar dilam 500 ar ekta coke khelam 20 taka diye")

# 2. LINGUISTIC & NUMERIC RULES
Users use slang. You must resolve these to exact `float` numbers:
- **"k" / "K":** Thousand (e.g., "2.5k" → `2500.0`, "10k" → `10000.0`)
- **"L" / "Lakh":** Lakh (e.g., "1.5L" → `150000.0`)
- **"Hajar":** Thousand (e.g., "dui hajar" → `2000.0`)
- **"Cr":** Crore (10 Million).
- **Bangla Numerals:** Convert usually (e.g., "৫০০ টাকা" → `500.0`).
- **Currencies:** Ignore symbols like "Tk", "BDT", "$". Just extract the number.

# 3. CONTEXTUAL DISAMBIGUATION (The "Smart" Logic)
You must resolve ambiguous words based on the *intent* of the sentence:
- **"Vat" / "Bhat":**
    - IF context implies eating (kheyechi, bill, hotel, lunch): → `category: "food"` (Rice)
    - IF context implies government/official (tax, included): → `category: "other"` (Value Added Tax)
- **"Gas":**
    - IF cooking/home (cylinder, line gas, kitchen): → `category: "utilities"`
    - IF vehicle (CNG, octane, pump, gari): → `category: "transport"`
- **"Table":**
    - IF furniture (kinlam, broken): → `category: "shopping"`
    - IF restaurant (vara, booking): → `category: "entertainment"` or `food`
- **"Net":**
    - IF internet/wifi/data: → `category: "utilities"` or `mobile_recharge`

# 4. CATEGORY LIST
Use ONLY these categories:
[`food`, `grocery`, `rent`, `utilities`, `mobile_recharge`, `transport`, `tuition`, `salary`, `gift`, `loan`, `shopping`, `medical`, `entertainment`, `other`]

# 5. OUTPUT SCHEMA
Return a JSON Object with a key `examples` containing the list.
Each item must have:
- `input`: The simulated user text.
- `output`: A list of transaction objects (can be empty if no transaction found).

### EXAMPLES (Study these patterns)

**Case 1: Single Transaction (Banglish Short)**
Input: "ajke vat khelam 40 taka"
Output: `[{"description": "Vat (Rice)", "amount": 40.0, "category": "food", "type": "expense"}]`

**Case 2: Complex Multi-Transaction (Code-switching)**
Input: "Salary dhukse 25k! But onnek khoroch... basha vara dilam 10000, gas bill 950, ar ammu ke pathalam 5k."
Output: `[
  {"description": "Salary", "amount": 25000.0, "category": "salary", "type": "income"},
  {"description": "Basha vara", "amount": 10000.0, "category": "rent", "type": "expense"},
  {"description": "Gas bill", "amount": 950.0, "category": "utilities", "type": "expense"},
  {"description": "Sent to Ammu", "amount": 5000.0, "category": "gift", "type": "expense"}
]`

**Case 3: Messy/Noisy Input**
Input: "ajke mood valo na 500 taka harai felsi"
Output: `[{"description": "Lost money", "amount": 500.0, "category": "other", "type": "expense"}]`

**Case 4: Numeric Slang**
Input: "tuition theke 1.5k pelam"
Output: `[{"description": "Tuition", "amount": 1500.0, "category": "tuition", "type": "income"}]`
"""

# instructions = "Instruction: Generate a batch consisting mostly of **single short transactions**."
instruction_list = [
    # --- SECTION 1: PURE BANGLA SCRIPT (The "Desi" Context) ---
    # "Instruction: Generate **Pure Bangla** script inputs (about 30-35 inputs) focusing on daily **Bazaar/Grocery** (e.g., 'চাল ডাল কিনলাম', 'মাছ কিনলাম').",
    # "Instruction: Generate **Pure Bangla** script inputs (about 30-35 inputs) focusing on **Household Utilities** (e.g., 'বিদ্যুৎ বিল', 'গ্যাস বিল', 'ইন্টারনেট বিল').",
    # "Instruction: Generate **Pure Bangla** script inputs (about 30-35 inputs) focusing on **Transport** (e.g., 'রিকশা ভাড়া', 'বাস ভাড়া', 'সিএনজি').",
    # "Instruction: Generate **Pure Bangla** script inputs (about 30-35 inputs) focusing on **Income Sources** (e.g., 'বেতন পেলাম', 'টিউশনি ফি', 'আব্বু পাঠিয়েছে').",
    # "Instruction: Generate **Pure Bangla** script inputs (about 30-35 inputs) focusing on **Festivals & Gifts** (e.g., 'ঈদের কেনাকাটা', 'সালামি', 'উপহার').",
    # "Instruction: Generate **Pure Bangla** script inputs (about 30-35 inputs) focusing on **Medical Expenses** (e.g., 'ওষুধ', 'ডাক্তার দেখিয়েছি', 'টেস্ট করালাম').",
    # "Instruction: Generate **Pure Bangla** script inputs (about 30-35 inputs) focusing on **Lending & Borrowing** (e.g., 'ধার দিলাম', 'হাওলাত নিলাম').",
    # "Instruction: Generate **Pure Bangla** script inputs (about 30-35 inputs) focusing on **Student Life** (e.g., 'ফটোকপি', 'শিট কিনলাম', 'ফর্ম ফিলাপ').",

    # --- SECTION 2: PURE ENGLISH (Mixed Single & Multi) ---
    # Strategy: 3 out of 7 here are explicitly MULTIPLE.
    # "Instruction: Generate **Pure English** script inputs (about 30-35 inputs) focusing on **Single Salary/Income** events (e.g., 'Salary credited', 'Freelance payment').",
    # "Instruction: Generate **Pure English** script inputs (about 30-35 inputs) focusing on **Single Subscriptions** (e.g., 'Netflix subscription', 'Spotify premium').",
    # "Instruction: Generate **Pure English** script inputs (about 30-35 inputs) focusing on **Multiple Bill Payments** in one go (e.g., 'Paid electricity, gas, and internet bills together').", 
    # "Instruction: Generate **Pure English** script inputs (about 30-35 inputs) focusing on a **Weekend Outing** with multiple costs (e.g., 'Movie tickets AND Dinner AND Cab fare').",
    # "Instruction: Generate **Pure English** script inputs (about 30-35 inputs) focusing on **E-commerce Shopping** (e.g., 'Ordered headphones and a mouse from Amazon').",
    # "Instruction: Generate **Pure English** script inputs (about 30-35 inputs) focusing on **Travel Expenses** (e.g., 'Flight tickets + Hotel booking + Food').",
    # "Instruction: Generate **Pure English** script inputs (about 30-35 inputs) focusing on **Lending/Borrowing** (e.g., 'Lent 500 to friend').",

    # --- SECTION 3: BANGLISH (The "Messy" Reality) ---
    # Strategy: 5 out of 10 here are explicitly MULTIPLE.
    # "Instruction: Focus on **Banglish Number Slang** scripts input (about 35-40 inputs) scenario with a **Single Transaction** (e.g., '5k tuition fee').",
    # "Instruction: Focus on **'Vat' (Rice) vs 'VAT' (Tax)** scripts input (about 35-40 inputs) scenario with ambiguity in a single sentence.",
    # "Instruction: Focus on **'Gas' (Cooking) vs 'Gas/CNG' scripts input (about 35-40 inputs) scenario with (Transport)** ambiguity.",
    # "Instruction: Generate script inputs (about 30-35 inputs) **Banglish Day Summary** with **3 distinct transactions** (e.g., 'Sokal er nasta, Rickshaw vara, ar dupur er khabar').",
    # "Instruction: Generate script inputs (about 30-35 inputs) **Banglish Month-End Calculation** with **4+ transactions** (e.g., 'Rent dilam, Gas bill dilam, Wifi bill dilam, hat e matro 500 taka').",
    # "Instruction: Generate script inputs (about 30-35 inputs) **Banglish Shopping Spree** with **Multiple Items** (e.g., 'Panjabi kinlam 2k diye, ar ekta ghuri 500 taka').",
    # "Instruction: Generate script inputs (about 30-35 inputs) **Banglish Mixed Income & Expense** scenario (e.g., 'Salary pelam 20k kintu bondhur dhar shodh korlam 2k').",
    # "Instruction: Focus on **Mobile Financial Services** scripts input (about 35-40 inputs) (e.g., 'Bkash cash out charge' and 'Send money').",
    # "Instruction: Focus on **'Table/Box' Ambiguity** script inputs (about 30-35 inputs) (e.g., 'Table vara' vs 'Lunch box').",
    # "Instruction: Generate script inputs (about 30-35 inputs) **Banglish Student Life** scenario with **Multiple Fees** (e.g., 'Form fillup 2k, Sheet 100tk, ar friend er treat 500tk').",

    # --- SECTION 4: EXPLICIT STRUCTURE ENFORCERS (The Balancers) ---
    # Strategy: These strictly force the model to generate MULTIPLE items to balance Section 1.
    # "Instruction: Generate a **Pure Bangla** scripts input (about 35-40 inputs) scenario containing **Multiple Transactions** (e.g., 'বাজার ২k, রিকশা ৫০, নাস্তা ১০০').",
    # "Instruction: Generate a **Pure Bangla** scripts input (about 35-40 inputs) scenario containing **Income AND Expense** together (about 35-40 inputs) (e.g., 'বেতন পেলাম কিন্তু বাড়ি ভাড়া দিলাম').",
    # "Instruction: Generate a **Complex Banglish Narrative** scripts input (about 35-40 inputs) with **4 or more transactions** (about 35-40 inputs) (A full day's ledger).",
    # "Instruction: Generate a **Mixed Script (Code-switching)** scripts input (about 35-40 inputs) scenario with **Multiple items** (about 35-40 inputs) (e.g., 'Rice kinlam 50kg but payment korsi card e').",
    # "Instruction: Generate a **Math-Heavy** scripts input (about 35-40 inputs) scenario with **Multiple steps**  (e.g., '500x3 items plus 50 taka delivery charge').",
    # "Instruction: Generate a **Medium Length (3-4 items)** scripts input (about 35-40 inputs) scenario focusing on **Food and Transport** combined. (about 35-40 inputs)"

    # --- SECTION 5: MIXED BANGLA AND ENGLISH IN SAME SENTENCE ---
    "Instruction: Generate a **Mixed Bangla-English** scripts input (about 35-40 inputs) scenario with **Multiple Transactions** (e.g., 'আজকে market থেকে groceries কিনলাম ২k, তারপর rickshaw ভাড়া দিলাম ৫০ টাকা এবং Bought some snacks for 100 taka').",
    "Instruction: Generate a **Mixed Bangla-English** scripts input (about 35-40 inputs) scenario with **Income AND Expense** together (e.g., 'Received my salary of ২৫k টাকা but paid the electricity bill of ১২০০ টাকা').",
]

# --- 4. Example Usage ---
# Create a dummy entry
# txn1 = Transaction(description="Kacchi", amount=450, category="food", type="expense")
# txn2 = Transaction(description="Mobile Recharge", amount=200, category="mobile_recharge", type="expense")
# tx3 = Transaction(description="Salary", amount=50000, category="salary", type="income")
# meta = Metadata(lang="banglish", complexity="simple", generated_by="manual_fix")

# entry = DatasetEntry(
#     input="kacchi khelam 450 taka diye",
#     output=[txn1, txn2, tx3],
#     metadata=meta
# )

# # Export it
# pprint(entry.to_chatml_format())


# --- 5. LLM Integration For Synthetic Dataset Generation ---

client = genai.Client(
    api_key=os.environ["GOOGLE_API_KEY"]
)

# File paths
output_file = "synthetic_dataset.jsonl"
chatml_file = "synthetic_dataset_chatml.jsonl"

print(f"Starting dataset generation with {len(instruction_list)} batches...\n")

total_entries = 0

# Loop through each instruction
for idx, instruction in enumerate(instruction_list, 1):
    print(f"[{idx}/{len(instruction_list)}] Processing: {instruction[:60]}...")
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=system_prompt + "\n" + instruction,
            config={
                "response_mime_type": "application/json",
                "response_schema": ListOfDatasetEntries.model_json_schema(),
            }
        )
    except Exception as e:
        print(f"  ✗ Error during generation: {e}")
        continue
    
    try:
        entries = ListOfDatasetEntries.model_validate_json(response.text)
        print(f"  ✓ Generated {len(entries.root)} entries")
        
        # Save to production format (APPEND mode)
        with open(output_file, "a+") as f:
            for entry in entries.root:
                production_format = entry.to_production_format()
                f.write(json.dumps(production_format, ensure_ascii=False, default=str) + "\n")
        
        # Save to ChatML format (APPEND mode)
        with open(chatml_file, "a+") as f:
            for entry in entries.root:
                chatml_format = entry.to_chatml_format()
                f.write(json.dumps(chatml_format, ensure_ascii=False) + "\n")
        
        total_entries += len(entries.root)
        
    except Exception as e:
        print(f"  ✗ Validation failed: {e}")
        continue

print(f"\n{'='*60}")
print(f"✓ Dataset generation complete!")
print(f"  Total entries: {total_entries}")
print(f"  Production format: {output_file}")
print(f"  ChatML format: {chatml_file}")
print(f"{'='*60}")