import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.vision.data import split_files
from app.vision.domains import VISION_DOMAINS
from app.vision.gradcam import GradCAM
from app.vision.models import ARCHS
from app.vision.service import vision_store

TRAINED = [(d, a) for d in VISION_DOMAINS for a in vision_store.available(d)]
needs_models = pytest.mark.skipif(len(TRAINED) < len(VISION_DOMAINS) * len(ARCHS), reason="image models not trained")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_gradcam_shape_and_range():
    for arch in ARCHS.values():
        model = arch.build(3, False).eval()
        probs, heat = GradCAM(model, arch.target_layer(model))(torch.randn(2, 3, 224, 224))
        assert probs.shape == (2, 3)
        assert torch.allclose(probs.sum(1), torch.ones(2), atol=1e-5)
        assert heat.shape == (2, 224, 224)
        assert float(heat.min()) >= 0 and float(heat.max()) <= 1


def test_unknown_image_domain(client):
    assert client.get("/api/vision/domains/unknown").status_code == 404


@needs_models
@pytest.mark.parametrize("domain_id", list(VISION_DOMAINS))
def test_every_model_predicts_from_test_images(client, domain_id):
    """Send one real held-out image per class through all three models."""
    domain = VISION_DOMAINS[domain_id]
    files = split_files(domain_id, "test")
    for label in range(len(domain.classes)):
        path = next(f for f, y in files if y == label)
        response = client.post(
            f"/api/vision/domains/{domain_id}/predict",
            files={"file": (path.name, path.read_bytes(), "image/jpeg")},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert {r["arch"] for r in body["results"]} == set(ARCHS)
        for result in body["results"]:
            assert result["prediction"] in domain.classes
            assert sum(p["probability"] for p in result["probabilities"]) == pytest.approx(1, abs=1e-4)
            assert result["heatmap"].startswith("data:image/png;base64,")


@needs_models
def test_accepts_png_of_any_size(client):
    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 500, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    response = client.post(
        "/api/vision/domains/leaf/predict",
        files={"file": ("leaf.png", buffer.getvalue(), "image/png")},
        data={"models": "mobilenet_v2"},
    )
    assert response.status_code == 200
    assert [r["arch"] for r in response.json()["results"]] == ["mobilenet_v2"]


def test_rejects_non_image(client):
    response = client.post(
        "/api/vision/domains/xray/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 422


@needs_models
def test_samples_are_served(client):
    detail = client.get("/api/vision/domains/xray").json()
    assert len(detail["models"]) == len(ARCHS)
    sample = detail["samples"][0]
    response = client.get(f"/api/vision/domains/xray/samples/{sample['name']}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
