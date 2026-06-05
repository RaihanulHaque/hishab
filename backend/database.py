from pathlib import Path
from sqlmodel import create_engine, Session, SQLModel
from models import Income, Expense

DATABASE_URL = f"sqlite:///{Path(__file__).parent / 'finance.db'}"
engine = create_engine(DATABASE_URL, echo=False)

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    return Session(engine)