from __future__ import annotations
from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime

class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    username: str
    password_hash: str
    sessions: list[Session] = Relationship(back_populates="user")

class Session(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    messages: list[Message] = Relationship(back_populates="session")
    user: User | None = Relationship(back_populates="sessions")

class Message(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="session.id")
    sender: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session: Session | None = Relationship(back_populates="messages")