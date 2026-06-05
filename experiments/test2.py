from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List, Literal, Union
import os
from dotenv import load_dotenv
load_dotenv()

# Define schema
# class IncomeEntry(BaseModel):
#     source: Literal["Family", "Tuition", "Freelance/Job", "Gift", "Scholarship", "Other"]
#     amount: int
#     description: str

# class ExpenseEntry(BaseModel):
#     category: Literal["Food", "Transport", "Medical", "Clothing", "Academic", "Entertainment", "Gift", "Others"]
#     item: str
#     amount: int

# class FinanceResponse(BaseModel):
#     type: Literal["income", "expense"]
#     entries: Union[List[IncomeEntry], List[ExpenseEntry]]

add_income_entry_declaration = {
    "name": "add_income_entry",
    "description": "Add income to the user's account with source and amount.",
    "parameters": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "enum": ["Family", "Tuition", "Freelance/Job", "Gift", "Scholarship", "Other"],
                "description": "The source of the income.",
            },
            "amount": {
                "type": "integer",
                "description": "Amount of money received in Bangladeshi Taka (tk).",
            },
            "description": {
                "type": "string",
                "description": "Short description of the income.",
            },
        },
        "required": ["source", "amount", "description"],
    },
}

add_expense_entry_declaration = {
    "name": "add_expense_entry",
    "description": "Add an expense to the user's account categorized by purpose.",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["Food", "Transport", "Medical", "Clothing", "Academic", "Entertainment", "Gift", "Others"],
                "description": "Category of the expense.",
            },
            "item": {
                "type": "string",
                "description": "What the money was spent on.",
            },
            "amount": {
                "type": "integer",
                "description": "Amount of money spent in Bangladeshi Taka (tk).",
            },
        },
        "required": ["category", "item", "amount"],
    },
}


def add_income_entry(source: str, amount: int, description: str) -> dict:
    # You would insert into DB here
    print(f"Income added - Source: {source}, Amount: {amount}, Description: {description}")
    return {"status": "success", "type": "income", "source": source, "amount": amount, "description": description}

def add_expense_entry(category: str, item: str, amount: int) -> dict:
    # You would insert into DB here
    print(f"Expense added - Category: {category}, Item: {item}, Amount: {amount}")
    return {"status": "success", "type": "expense", "category": category, "item": item, "amount": amount}


tools = types.Tool(
    function_declarations=[
        add_income_entry_declaration,
        add_expense_entry_declaration,
    ],
)
config = types.GenerateContentConfig(
    system_instruction="""
    You are an intelligent financial assistant. Your purpose is to help the user track their income and expenses.
    When the user provides information about receiving income or incurring an expense, use the provided tools (functions) `add_income_entry` or `add_expense_entry` to record the details.
    Extract the necessary information such as source/category, amount, and description/item from the user's input, which may be in Banglish or English but make sure that the output must be in english.
    The currency is always Bangladeshi Taka (tk).
    Examples:
        - User: "abbu amake 4000 taka diyeche" -> Use `add_income_entry` function.
        - User: "dupurer khabarer jonno 200 taka khoroch hoyeche" -> Use `add_expense_entry` function.
    """, # <-- Removed the conflicting JSON output instruction
    tools=[tools],
    # response_mime_type="application/json", # Keep this commented out or remove it
)

# Define agent prompt
prompt = """
"dupurer khabarer jonno 200 taka khoroch hoyeche, auto 30tk, abbu 4000tk, boi kinechi 500 taka, amake aro 3000 taka diyeche khala, ghori er dam 50000 taka"
"""

# Call Gemini with schema
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents="input: " + prompt,
    config=config,
    # response_mime_type="application/json",
)

print(response)  # JSON string
# response_obj: FinanceResponse = response.parsed  # Pydantic object

found_function_call = False
if response.candidates and response.candidates[0].content:
    for part in response.candidates[0].content.parts:
        if part.function_call:
            found_function_call = True
            function_call = part.function_call
            print(f"Function to call: {function_call.name}")
            print(f"Arguments: {function_call.args}")
            #  In a real app, you would call your function here:
            try:
                if function_call.name == "add_income_entry":
                    result = add_income_entry(**function_call.args)
                elif function_call.name == "add_expense_entry":
                    result = add_expense_entry(**function_call.args)
                else:
                    print(f"Warning: Unknown function: {function_call.name}")
                print(f"Function call result: {result}")
            except Exception as e:
                print(f"Error calling function {function_call.name}: {e}")

        print("\n---\n")  # Separator for clarity

if not found_function_call:
    print("No function call found in the response.")
    # Print the response text only if no function call was found
    print(response.text)
    print(f"\n\n\n\n\n\n\n")
    # Print the response object
    print(response)