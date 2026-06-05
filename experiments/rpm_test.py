from google import genai
from pydantic import BaseModel
import os
import dotenv
import concurrent.futures
import time

dotenv.load_dotenv()

# Initialize Gemini Client
client = genai.Client(api_key=os.environ["2008011_GEMINI_API_KEY"])

# Prompt content
content = (
    "Hi, How are you"
)

# Single request function
def call_gemini(index: int):
    try:
        start_time = time.time()
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=content
        )
        elapsed = time.time() - start_time
        print(f"[{index}] ✅ Success | Time: {elapsed:.2f}s | Response: {str(response.text)[:50]}...")
        return response
    except Exception as e:
        print(f"[{index}] ❌ Error: {e}")
        return None

# Parallel test config
TOTAL_REQUESTS = 70  # You can change this to 30, 60, etc.
MAX_WORKERS = 50     # Threads in parallel

if __name__ == "__main__":
    print(f"🚀 Sending {TOTAL_REQUESTS} requests using {MAX_WORKERS} threads...\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(call_gemini, i) for i in range(TOTAL_REQUESTS)]
        for future in concurrent.futures.as_completed(futures):
            _ = future.result()