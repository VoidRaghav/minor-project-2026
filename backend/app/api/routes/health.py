from fastapi import APIRouter

from app.vision.domains import VISION_DOMAINS
from app.vision.service import vision_store

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    return {"status": "ok", "image_models": {d: vision_store.available(d) for d in VISION_DOMAINS}}
