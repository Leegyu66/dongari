from fastapi import APIRouter

from app.routers import health, predict

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health")
api_router.include_router(predict.router, prefix="/predict")
