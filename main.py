from functions import tools, add_income_entry, add_expense_entry
from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Form
from sqlmodel import Session, select
from database import init_db, get_session
from models import Income, Expense
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
init_db()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY_404_HISHAB"))



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

@app.get("/", response_class=HTMLResponse)
def read_dashboard(request: Request, session: Session = Depends(get_session)):
    incomes = session.exec(select(Income)).all()
    expenses = session.exec(select(Expense)).all()
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "incomes": incomes,
        "expenses": expenses,
    })

@app.get("/analyze", response_class=HTMLResponse)
def analyze_page(request: Request):
    return templates.TemplateResponse("analyze.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_submit(
    request: Request,
    input_text: str = Form(...),
    session: Session = Depends(get_session)
):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="input: " + input_text,
        config=config,
    )

    success = False
    if response.candidates and response.candidates[0].content:
        for part in response.candidates[0].content.parts:
            if part.function_call:
                fn = part.function_call
                args = fn.args
                if fn.name == "add_income_entry":
                    session.add(Income(**args))
                    success = True
                elif fn.name == "add_expense_entry":
                    session.add(Expense(**args))
                    success = True
        session.commit()

    return templates.TemplateResponse("analyze.html", {
        "request": request,
        "success": success
    })



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0", port=8000)