import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

def create_user(supabase: Client, email: str, password: str):
    """Create a new user in Supabase."""
    response = supabase.auth.sign_up({
        "email": email,
        "password": password
    })
    print(f"User creation response: {response}")

def sign_in_user(supabase: Client, email: str, password: str):
    """Sign in a user to Supabase."""
    response = supabase.auth.sign_in_with_password({
        "email": email,
        "password": password
    })
    print(f"User sign-in response: {response}")

def list_users(supabase: Client):
    """List all users in Supabase."""
    response = supabase.auth.admin.list_users(
        page=1,
        per_page=1000,
    )
    return response

def fetch_data(supabase: Client, table_name: str):
    response = supabase.table(table_name).select("*").execute()
    if response.error:
        print(f"Error fetching data from {table_name}: {response.error.message}")
    else:
        return response.data
    
def update_data(supabase: Client, table_name: str, user_id: str, data: dict):
    """Update data in a specified table for a given user ID."""
    response = supabase.table(table_name).update(data).eq("user_id", user_id).execute()

def update_income_entries(supabase: Client, user_id: str):
    # Get the income entries for the user
    response = supabase.table("income_entries").select("*").eq("user_id", user_id).execute()

    # Update each entry's amount
    if response.error:
        print(f"Error fetching income entries: {response.error.message}")
        return

    for entry in response.data:
        updated_amount = entry['amount'] * 100  # Example transformation
        update_response = supabase.table("income_entries").update({"amount": updated_amount}).eq("id", entry['id']).execute()
        
        if update_response.error:
            print(f"Error updating income entry {entry['id']}: {update_response.error.message}")
        else:
            print(f"Income entry {entry['id']} updated successfully.")

def update_expense_entries(supabase: Client, user_id: str):
    # Get the expense entries for the user
    response = supabase.table("expense_entries").select("*").eq("user_id", user_id).execute()

    # Update each entry's amount
    if response.error:
        print(f"Error fetching expense entries: {response.error.message}")
        return

    for entry in response.data:
        updated_amount = entry['amount'] * 100  # Example transformation
        update_response = supabase.table("expense_entries").update({"amount": updated_amount}).eq("id", entry['id']).execute()
        
        if update_response.error:
            print(f"Error updating expense entry {entry['id']}: {update_response.error.message}")
        else:
            print(f"Expense entry {entry['id']} updated successfully.")


# Initialize Supabase client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_API_KEY")
service_role_key: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

supabase: Client = Client(
    supabase_url=url,
    supabase_key=service_role_key
)


# Create a new user
# create_user(
#     supabase,
#     email="raihanulhaque404@gmail.com",
#     password="sa1tama_"
# )

# Sign in a user
# sign_in_user(
#     supabase,
#     email="raihanulhaque404@gmail.com",
#     password="sa1tama_"
# )
# print(supabase.auth.get_user().user.id)

# user_id = "d45a2ec6-cfb9-4025-aeac-32ad0fecbca2"
# limit = 5

# users = list_users(supabase)
# if users:
#     for user in users:
#         print(f"User ID: {user.id}, Email: {user.email}")

# print("User listing complete.")

# response = supabase.auth.sign_in_with_password(
#     {
#         "email": "rahiraihanulhaque@gmail.com",
#         "password": "sa1tama_",
#     }
# )
# print(f"Sign-in response: {response}")

# shihab AUTH id
auth_id = "580d2ad5-42e5-4a9c-a5bb-5a35dfc3f724"
# Rahi007 id
# auth_id = "d8e3052e-c16d-47b5-aaf1-a68a1e627e6c"
response = (
    supabase.table("expense_entries")
    .select("user_id, amount")
    # .eq("user_id", auth_id)
    .execute()
)
# print(f"Response: {response.data[0]}")

for entry in response.data:
    print(f"User ID: {entry['user_id']}, Amount: {entry['amount']}, Description: {entry.get('description', 'N/A')}")
    # updated_amount = entry['amount'] * 100
    # print(f"Updated Amount: {updated_amount}")

    # supabase.table("expense_entries").update({"amount": updated_amount}).eq("user_id", entry['user_id']).execute()