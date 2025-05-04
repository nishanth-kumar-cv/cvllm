from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import select, Session as DbSession
from models import ChartConfig
from auth_jwt import get_current_user
from startup import engine

router = APIRouter()

@router.post("/save_chart_config")
def save_config(config: ChartConfig, user=Depends(get_current_user)):
    with DbSession(engine) as db:
        config.user_id = user.id
        db.add(config)
        db.commit()
        return {"status": "saved"}

@router.get("/list_chart_configs")
def list_configs(user=Depends(get_current_user)):
    with DbSession(engine) as db:
        results = db.exec(select(ChartConfig).where(ChartConfig.user_id == user.id)).all()
        return results

@router.get("/generate_from_template/{config_id}")
def generate_chart(config_id: int, user=Depends(get_current_user)):
    with DbSession(engine) as db:
        config = db.get(ChartConfig, config_id)
        if not config or config.user_id != user.id:
            raise HTTPException(status_code=404, detail="Not found")
        # Could trigger plot_tool with stored query, return plot link...
        return {"status": "triggered", "query": config.query}
