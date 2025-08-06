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

def check_email_exists(supabase: Client, email: str) -> bool:
    response = supabase.table("users").select("id").execute()
    return len(response.data) > 0

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

users = list_users(supabase)
if users:
    for user in users:
        print(f"User ID: {user.id}, Email: {user.email}")

print("User listing complete.")