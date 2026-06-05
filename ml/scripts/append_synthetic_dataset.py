import argparse
import json
import os
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field, RootModel


CATEGORIES = [
    "food",
    "grocery",
    "rent",
    "utilities",
    "mobile_recharge",
    "transport",
    "tuition",
    "salary",
    "gift",
    "loan",
    "shopping",
    "medical",
    "entertainment",
    "other",
]

SYSTEM_MESSAGE = (
    "You are a financial assistant who understands English, Bengali and "
    "Banglish(Bengali written in english). Extract transactions from text into a JSON list."
)


class Transaction(BaseModel):
    description: str = Field(..., description="Short text identifying the item")
    amount: float = Field(..., ge=0, description="Positive amount as a number")
    category: Literal[
        "food",
        "grocery",
        "rent",
        "utilities",
        "mobile_recharge",
        "transport",
        "tuition",
        "salary",
        "gift",
        "loan",
        "shopping",
        "medical",
        "entertainment",
        "other",
    ]
    type: Literal["income", "expense"]


class DatasetExample(BaseModel):
    input: str
    output: list[Transaction]
    lang: Literal["bangla", "english", "banglish", "mixed"]


class GeneratedBatch(BaseModel):
    examples: list[DatasetExample]


class Metadata(BaseModel):
    lang: Literal["bangla", "english", "banglish", "mixed"]
    complexity: Literal["simple", "medium", "complex"]
    generated_by: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    validated: bool = True
    validation_notes: str | None = "schema_validated_append_generation"
    tokens_input: int | None = None
    tokens_output: int | None = None


class ProductionEntry(BaseModel):
    id: str = Field(default_factory=lambda: f"txn_{uuid.uuid4().hex[:8]}")
    input: str
    output: list[Transaction]
    metadata: Metadata

    def assistant_content(self) -> str:
        return json.dumps([t.model_dump() for t in self.output], ensure_ascii=False, separators=(",", ":"))

    def to_chatml_format(self) -> dict:
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": self.input},
                {"role": "assistant", "content": self.assistant_content()},
            ]
        }

    def to_production_format(self) -> dict:
        return {
            "id": self.id,
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": self.input},
                {"role": "assistant", "content": self.assistant_content()},
            ],
            "metadata": self.metadata.model_dump(),
            "raw_transactions": [t.model_dump() for t in self.output],
        }


class ExistingInputs(RootModel):
    root: set[str]


PROMPT = """
# Role
You generate synthetic training data for a Bangladeshi personal finance transaction extractor.

# Output
Return exactly a JSON object with key "examples". Generate 8 to 12 unique examples.
Each example must contain:
- input: realistic user text
- output: list of transaction objects
- lang: one of bangla, english, banglish, mixed

# Required transaction schema
Each output transaction must contain:
- description: short natural description
- amount: exact float number
- category: one of {categories}
- type: income or expense

# Data distribution for this batch
{focus}

# Important realism rules
- Use realistic Bangladesh daily amounts. Avoid huge amounts unless salary, rent, device purchase, loan, or business income.
- Food/snacks/tea: usually 10 to 800.
- Rickshaw/bus/CNG/local transport: usually 10 to 800.
- Mobile recharge/data: usually 20 to 1500.
- Grocery/bazaar: usually 100 to 8000.
- Tuition fees/student fees: usually 100 to 30000.
- Rent/utilities: usually 300 to 50000.
- Salary/income: usually 1000 to 300000.
- Bkash/Nagad/Rocket charges should be small and separate only when explicitly mentioned.

# Language rules
- Banglish should look messy and natural: ajke, kalke, gotokal, taka, tk, k, dilam, disi, disilam, pelam, paisi, kinlam, kheyechi, khaisi, lagse, laglo, vara/bhara, rickshaw/riksha, cng, bazar/bajar, khoroch/khoros, tuition/tution.
- Mixed examples should genuinely mix Bangla script, English words, and Banglish in one sentence.
- Include common spelling variations and casual punctuation.
- Include some examples with multiple transactions and some with income plus expense.
- Include occasional "no transaction" examples where output is [] only if no amount/transaction is extractable.

# Numeric rules
- Convert k/K to thousand: 2.5k = 2500.0.
- Convert lakh/L to 100000: 1.5 lakh = 150000.0.
- Convert crore/Cr to 10000000.
- Convert Bangla digits to normal numeric amounts in the output.
- Ignore currency symbols and words like taka, tk, BDT.

# Categorization rules
- Eating meals/snacks/tea/restaurant bills -> food.
- Raw bazaar/grocery/super shop household food ingredients -> grocery.
- Internet/WiFi/electricity/gas/water bills -> utilities.
- Mobile recharge/data pack/flexiload -> mobile_recharge.
- Rickshaw/bus/CNG/Uber/fuel -> transport.
- Salary credited -> salary income.
- Tuition earned -> tuition income; tuition/school/college fees paid -> tuition expense.
- Gifts/salami/family support sent or received -> gift.
- Dhar/loan/hawlat given or repaid/received -> loan.

# Avoid
- Do not repeat any of these existing inputs:
{existing_inputs}
"""

FOCUSES = [
    "Mostly Banglish daily expenses. Use 50% single transaction, 35% 2-3 transactions, 15% 4+ transactions.",
    "Mixed Bangla-English-Banglish inputs with income and expense together. Use code-switching heavily.",
    "Banglish student life: tuition, form fillup, photocopy, books, transport, snacks, mobile recharge.",
    "Bangladeshi family/month-end expenses: rent, utility bills, grocery, family support, salary.",
    "Messy short phone-style notes with abbreviations, missing punctuation, and k/tk/lakh numeric slang.",
    "Pure Bangla plus mixed-script examples covering food, transport, grocery, medical, shopping, loan.",
    "Mobile financial services: bKash, Nagad, Rocket, cash out charge, send money, received money.",
    "Ambiguous words: gas home vs vehicle, vat rice vs VAT tax, net internet vs other contexts.",
]

LANG_FOCUSES = {
    "banglish": [
        "Banglish daily expenses. Use casual romanized Bengali with messy spelling and realistic BD amounts.",
        "Banglish student life: tuition, form fillup, photocopy, books, transport, snacks, mobile recharge.",
        "Banglish income plus expense: salary/tuition/freelance received, then bills, family support, transport.",
        "Short phone-style Banglish notes with k/tk/lakh numeric slang, missing punctuation, and spelling variants.",
    ],
    "bangla": [
        "Pure Bangla script examples covering grocery, food, transport, utilities, rent, salary, gift, loan.",
        "Pure Bangla month-end and daily-summary examples with 2-4 realistic transactions.",
        "Pure Bangla student/family examples: tuition fee, বাজার, ঔষধ, রিকশা ভাড়া, মোবাইল রিচার্জ.",
    ],
    "english": [
        "Pure English examples from a Bangladeshi finance context: salary, rent, bills, grocery, transport, tuition.",
        "Pure English short expense notes and daily summaries with realistic BD amounts and categories.",
        "Pure English income plus expense examples, including family support, loan repayment, subscriptions, medicine.",
    ],
    "mixed": [
        "Mixed Bangla-English-Banglish inputs with real code-switching in one sentence.",
        "Mixed-script examples using Bangla digits, English category words, and Banglish verbs like dilam/pelam/kinlam.",
        "Mixed inputs with income and expense together: salary/tuition received, bills paid, market/grocery/transport.",
    ],
}

BALANCE_FOR_5000 = {
    "banglish": 2000,
    "bangla": 1250,
    "english": 1000,
    "mixed": 750,
}


def complexity_for(output: list[Transaction]) -> Literal["simple", "medium", "complex"]:
    if len(output) <= 1:
        return "simple"
    if len(output) <= 3:
        return "medium"
    return "complex"


def read_existing_inputs(path: Path) -> set[str]:
    inputs: set[str] = set()
    if not path.exists():
        return inputs

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                for message in obj.get("messages", []):
                    if message.get("role") == "user":
                        inputs.add(message.get("content", "").strip())
                        break
            except json.JSONDecodeError:
                continue
    return inputs


def read_dataset_stats(path: Path) -> tuple[int, Counter[str]]:
    rows = 0
    langs: Counter[str] = Counter()
    if not path.exists():
        return rows, langs

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            lang = obj.get("metadata", {}).get("lang")
            if lang:
                langs[lang] += 1
    return rows, langs


def write_entries(output_file: Path, chatml_file: Path, entries: list[ProductionEntry]) -> None:
    with output_file.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry.to_production_format(), ensure_ascii=False, default=str) + "\n")

    with chatml_file.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry.to_chatml_format(), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Append Gemini-generated synthetic finance extraction data.")
    parser.add_argument("--target", type=int, default=200, help="Approximate number of new unique examples to append.")
    parser.add_argument(
        "--target-total",
        type=int,
        default=None,
        help="Append until the production dataset reaches this many rows.",
    )
    parser.add_argument(
        "--balanced-5000",
        action="store_true",
        help="When targeting 5000 rows, fill toward 40%% banglish, 25%% bangla, 20%% english, 15%% mixed.",
    )
    parser.add_argument("--model", default="gemini-2.5-flash-lite", help="Gemini model name.")
    _DATASETS = Path(__file__).parent.parent / "datasets"
    parser.add_argument("--output-file", default=str(_DATASETS / "synthetic_dataset.jsonl"))
    parser.add_argument("--chatml-file", default=str(_DATASETS / "synthetic_dataset_chatml.jsonl"))
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing. Add it to .env or export it in the shell.")

    output_file = Path(args.output_file)
    chatml_file = Path(args.chatml_file)
    existing_inputs = read_existing_inputs(output_file)
    current_rows, lang_counts = read_dataset_stats(output_file)

    if args.target_total is not None:
        args.target = max(0, args.target_total - current_rows)

    client = genai.Client(api_key=api_key)
    appended = 0
    batch_index = 0

    print(f"Existing rows: {current_rows}", flush=True)
    print(f"Existing unique inputs: {len(existing_inputs)}", flush=True)
    print(f"Existing language counts: {dict(lang_counts)}", flush=True)
    print(f"Appending up to {args.target} new examples using {args.model}", flush=True)

    while appended < args.target:
        if args.balanced_5000:
            deficits = {lang: BALANCE_FOR_5000[lang] - lang_counts.get(lang, 0) for lang in BALANCE_FOR_5000}
            target_lang = max(deficits, key=deficits.get)
            if deficits[target_lang] <= 0:
                focus = FOCUSES[batch_index % len(FOCUSES)]
                allowed_lang = "any of bangla, english, banglish, mixed"
            else:
                focus_options = LANG_FOCUSES[target_lang]
                focus = focus_options[batch_index % len(focus_options)]
                allowed_lang = target_lang
        else:
            focus = FOCUSES[batch_index % len(FOCUSES)]
            allowed_lang = "any of bangla, english, banglish, mixed"

        batch_index += 1
        sample_existing = sorted(existing_inputs)[-80:]
        prompt = PROMPT.format(
            categories=", ".join(CATEGORIES),
            focus=f"{focus}\n- All examples in this batch must use lang: {allowed_lang}.",
            existing_inputs=json.dumps(sample_existing, ensure_ascii=False),
        )

        response = None
        for attempt in range(1, 6):
            try:
                response = client.models.generate_content(
                    model=args.model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": GeneratedBatch.model_json_schema(),
                    },
                )
                break
            except Exception as exc:
                wait_seconds = min(90, 10 * attempt)
                print(
                    f"Batch {batch_index}: generation error on attempt {attempt}/5: {exc}. "
                    f"Waiting {wait_seconds}s.",
                    flush=True,
                )
                if attempt == 5:
                    raise
                time.sleep(wait_seconds)

        if response is None:
            raise RuntimeError("Generation failed without a response.")

        generated = GeneratedBatch.model_validate_json(response.text)
        new_entries: list[ProductionEntry] = []

        for example in generated.examples:
            user_input = example.input.strip()
            if not user_input or user_input in existing_inputs:
                continue
            if args.balanced_5000 and allowed_lang != "any of bangla, english, banglish, mixed":
                if example.lang != allowed_lang:
                    continue

            entry = ProductionEntry(
                input=user_input,
                output=example.output,
                metadata=Metadata(
                    lang=example.lang,
                    complexity=complexity_for(example.output),
                    generated_by=args.model,
                ),
            )
            new_entries.append(entry)
            existing_inputs.add(user_input)
            lang_counts[example.lang] += 1

        if new_entries:
            remaining = args.target - appended
            new_entries = new_entries[:remaining]
            write_entries(output_file, chatml_file, new_entries)
            appended += len(new_entries)

        print(f"Batch {batch_index}: accepted {len(new_entries)}; total appended {appended}/{args.target}", flush=True)

    print(f"Done. Appended {appended} examples.", flush=True)
    print(f"Production file: {output_file}", flush=True)
    print(f"ChatML file: {chatml_file}", flush=True)


if __name__ == "__main__":
    main()
