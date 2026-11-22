from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.vision.domains import VISION_DOMAINS, VisionDomain
from app.vision.service import InvalidImage, ModelUnavailable, domain_summary, vision_store

router = APIRouter(prefix="/vision", tags=["vision"])


def get_vision_domain(domain_id: str) -> VisionDomain:
    domain = VISION_DOMAINS.get(domain_id)
    if domain is None:
        raise HTTPException(status_code=404, detail=f"Unknown image domain '{domain_id}'")
    return domain


@router.get("/domains")
def list_domains() -> List[dict]:
    return [domain_summary(d) for d in VISION_DOMAINS.values()]


@router.get("/domains/{domain_id}")
def read_domain(domain_id: str) -> dict:
    domain = get_vision_domain(domain_id)
    detail = domain_summary(domain)
    try:
        detail["samples"] = vision_store.samples(domain)
    except FileNotFoundError:
        detail["samples"] = []
    return detail


@router.get("/domains/{domain_id}/samples/{name}")
def read_sample(domain_id: str, name: str):
    path = vision_store.sample_path(get_vision_domain(domain_id), name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Unknown sample '{name}'")
    return FileResponse(path, media_type="image/jpeg")


@router.post("/domains/{domain_id}/predict")
def predict(domain_id: str, file: UploadFile = File(...), models: Optional[str] = Form(None)) -> dict:
    domain = get_vision_domain(domain_id)
    arch_ids = [m.strip() for m in models.split(",") if m.strip()] if models else None
    try:
        return vision_store.predict(domain, file.file.read(), arch_ids)
    except InvalidImage as error:
        raise HTTPException(status_code=422, detail=str(error))
    except ModelUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error))
