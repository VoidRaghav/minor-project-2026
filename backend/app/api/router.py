from fastapi import APIRouter

from app.api.routes import domains, health, vision

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(domains.router)
api_router.include_router(vision.router)
