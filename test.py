from google import genai
from pydantic import BaseModel
import os
import dotenv
dotenv.load_dotenv()

class Recipe(BaseModel):
    recipe_name: str
    ingredients: list[str]

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY_MD"])
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="List a few popular cookie recipes, and include the amounts of ingredients.",
    config={
        "response_mime_type": "application/json",
        "response_schema": list[Recipe],
    },
)

response2 = client.models.generate_content(
    model="gemini-2.0-flash-lite",
    contents="5000 income. VAT 200 on food. Ice cream, 30. Also paid 150 for a new phone cover, which I totally needed. Oh, and petrol for the bike was 400. Then there was that 50 taka for the parking guy, no receipt there! Still need to figure out what's left after all this. It's a mess.",
)
# Use the response as a JSON string.
print(response2)
# print(response2.output_token_limit)
# Use instantiated objects.
