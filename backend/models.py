from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class Income(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    source: str
    amount: int
    description: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Expense(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    category: str
    item: str
    amount: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)