from sqlmodel import SQLModel, Field, Relationship
from typing import List, Optional
from datetime import datetime

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str
    password_hash: str
    sessions: List["Session"] = Relationship(back_populates="user")

class Session(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    messages: List["Message"] = Relationship(back_populates="session")
    user: Optional[User] = Relationship(back_populates="sessions")

class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="session.id")
    sender: str  # 'user' or 'bot'
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session: Optional[Session] = Relationship(back_populates="messages")
