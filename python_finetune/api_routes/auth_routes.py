from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from python_finetune.models import User
from python_finetune.auth_jwt import get_password_hash, verify_password, create_access_token
from datetime import timedelta
from fastapi.security import OAuth2PasswordRequestForm

router = APIRouter()

@router.post("/register")
def register(username: str, password: str, db: Session = Depends()):
    if db.exec(select(User).where(User.username == username)).first():
        raise HTTPException(status_code=400, detail="User already exists")
    user = User(username=username, password_hash=get_password_hash(password))
    db.add(user)
    db.commit()
    return {"msg": "Registered"}

@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends()):
    user = db.exec(select(User).where(User.username == form_data.username)).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    token = create_access_token({"sub": user.username}, timedelta(minutes=60))
    return {"access_token": token, "token_type": "bearer"}