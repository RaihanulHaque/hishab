from google.genai import types


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