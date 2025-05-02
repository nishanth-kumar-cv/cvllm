from models import Message, Session as ChatSession
from sqlmodel import Session as DbSession, select
from datetime import datetime

def log_message(db: DbSession, user_id: int, content: str, sender: str):
    # Get or create session
    chat_session = db.exec(
        select(ChatSession).where(ChatSession.user_id == user_id).order_by(ChatSession.created_at.desc())
    ).first()
    if not chat_session:
        chat_session = ChatSession(user_id=user_id, created_at=datetime.utcnow())
        db.add(chat_session)
        db.commit()
        db.refresh(chat_session)

    message = Message(session_id=chat_session.id, sender=sender, content=content)
    db.add(message)
    db.commit()
