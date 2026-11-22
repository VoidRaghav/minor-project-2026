from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import CORS_ORIGINS

app = FastAPI(title="Explainable AI for Agriculture and Healthcare", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(api_router)
