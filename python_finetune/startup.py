from fastapi import FastAPI
from sqlmodel import SQLModel, create_engine, Session
from routes import auth_routes
from infer_api import app as infer_app
from upload_endpoint import app as upload_app

DATABASE_URL = "postgresql://mistral:mistral123@db:5432/chat"
engine = create_engine(DATABASE_URL)

app = FastAPI()
app.include_router(auth_routes.router)
app.mount("/generate", infer_app)
app.mount("/upload", upload_app)

@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)

@app.get("/")
def root():
    return {"status": "Mistral LLM API running"}
