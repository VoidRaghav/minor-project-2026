from fastapi import APIRouter, Depends

from app.api.deps import get_store
from app.ml.store import ModelStore
from app.vision.domains import VISION_DOMAINS
from app.vision.service import vision_store

router = APIRouter(tags=["health"])


@router.get("/health")
def health(store: ModelStore = Depends(get_store)):
    return {
        "status": "ok",
        "models": sorted(store.models),
        "image_models": {d: vision_store.available(d) for d in VISION_DOMAINS},
    }
