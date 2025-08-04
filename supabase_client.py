import os
from supabase import create_client, Client

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
    if response.error:
        print(f"Error listing users: {response.error.message}")
    else:
        return response.data

def fetch_data(supabase: Client, table_name: str):
    response = supabase.table(table_name).select("*").execute()
    if response.error:
        print(f"Error fetching data from {table_name}: {response.error.message}")
    else:
        return response.data

# Initialize Supabase client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_API_KEY")
supabase: Client = create_client(url, key)


# Create a new user
# create_user(
#     supabase,
#     email="raihanulhaque404@gmail.com",
#     password="sa1tama_"
# )

# Sign in a user
sign_in_user(
    supabase,
    email="raihanulhaque404@gmail.com",
    password="sa1tama_"
)
print(supabase.auth.get_user().user.id)

# user_id = "d45a2ec6-cfb9-4025-aeac-32ad0fecbca2"
# limit = 5

# response = supabase.table("expense_entries") \
#     .select("*") \
#     .eq("user_id", user_id) \
#     .order("created_at", desc=True) \
#     .limit(limit) \
#     .execute()


# print(f"Response: {response}")