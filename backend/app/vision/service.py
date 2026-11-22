import base64
import io
import json
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from app.core.config import ARTIFACTS_DIR
from app.vision.data import DATA_DIR, load_image, split_files
from app.vision.domains import VISION_DOMAINS, VisionDomain
from app.vision.gradcam import GradCAM, overlay
from app.vision.models import ARCHS, device, normalize

VISION_ARTIFACTS = ARTIFACTS_DIR / "vision"
SAMPLES_PER_CLASS = 4
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class InvalidImage(ValueError):
    pass


class ModelUnavailable(RuntimeError):
    pass


def _png_base64(array: np.ndarray) -> str:
    out = io.BytesIO()
    Image.fromarray(array).save(out, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def describe_heat(heat: np.ndarray) -> Tuple[str, float]:
    """Where the heatmap mass sits (e.g. 'lower left') and what share of it the top 20% of pixels hold."""
    total = heat.sum() + 1e-8
    ys, xs = np.mgrid[0:heat.shape[0], 0:heat.shape[1]]
    cy, cx = (heat * ys).sum() / total / heat.shape[0], (heat * xs).sum() / total / heat.shape[1]
    vertical = "upper" if cy < 0.4 else "lower" if cy > 0.6 else "middle"
    horizontal = "left" if cx < 0.4 else "right" if cx > 0.6 else "centre"
    region = "centre" if vertical == "middle" and horizontal == "centre" else f"{vertical} {horizontal}".replace("middle ", "")
    flat = np.sort(heat.ravel())[::-1]
    share = float(flat[: int(0.2 * flat.size)].sum() / total)
    return region, share


class VisionStore:
    def __init__(self):
        self.models: Dict[Tuple[str, str], Tuple[torch.nn.Module, GradCAM]] = {}
        self.lock = threading.Lock()
        self.device = device()
        self._metrics: Optional[Dict] = None

    @property
    def metrics(self) -> Dict:
        path = VISION_ARTIFACTS / "metrics.json"
        if self._metrics is None and path.exists():
            self._metrics = json.loads(path.read_text())
        return self._metrics or {}

    def available(self, domain_id: str) -> List[str]:
        return [a for a in ARCHS if (VISION_ARTIFACTS / f"{domain_id}_{a}.pt").exists()]

    def model(self, domain: VisionDomain, arch_id: str):
        key = (domain.id, arch_id)
        if key not in self.models:
            path = VISION_ARTIFACTS / f"{domain.id}_{arch_id}.pt"
            if not path.exists():
                raise ModelUnavailable(f"{ARCHS[arch_id].name} has not been trained for {domain.name}. Run: python -m app.vision.train")
            arch = ARCHS[arch_id]
            model = arch.build(len(domain.classes), False)
            state = torch.load(path, map_location="cpu")
            model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
            model.to(self.device).eval()
            self.models[key] = (model, GradCAM(model, arch.target_layer(model)))
        return self.models[key]

    def samples(self, domain: VisionDomain) -> List[Dict]:
        """A fixed, class-balanced set of held-out test images for the gallery."""
        files = split_files(domain.id, "test")
        rng = np.random.default_rng(7)
        chosen = []
        for label in range(len(domain.classes)):
            group = [f for f, y in files if y == label]
            for i in sorted(rng.choice(len(group), size=min(SAMPLES_PER_CLASS, len(group)), replace=False)):
                chosen.append({"name": group[i].stem, "label": domain.classes[label], "class_index": label})
        return chosen

    def sample_path(self, domain: VisionDomain, name: str):
        for path, _ in split_files(domain.id, "test"):
            if path.stem == name:
                return path
        return None

    def predict(self, domain: VisionDomain, data: bytes, arch_ids: Optional[List[str]] = None) -> Dict:
        if len(data) > MAX_UPLOAD_BYTES:
            raise InvalidImage("Image is larger than 10 MB.")
        try:
            image = load_image(io.BytesIO(data))
        except (UnidentifiedImageError, OSError):
            raise InvalidImage("Could not read the file as an image. Use a JPEG or PNG.")
        arch_ids = arch_ids or self.available(domain.id)
        unknown = [a for a in arch_ids if a not in ARCHS]
        if unknown:
            raise InvalidImage(f"Unknown models: {', '.join(unknown)}")
        if not arch_ids:
            raise ModelUnavailable("No image models are trained yet. Run: python -m app.vision.train")

        x = normalize(torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[None].to(self.device))
        results = []
        with self.lock:
            for arch_id in arch_ids:
                _, cam = self.model(domain, arch_id)
                start = time.perf_counter()
                probs, heat = cam(x)
                elapsed = (time.perf_counter() - start) * 1000
                probs, heat = probs[0].cpu().numpy(), heat[0].cpu().numpy()
                top = int(probs.argmax())
                region, share = describe_heat(heat)
                results.append({
                    "arch": arch_id,
                    "name": ARCHS[arch_id].name,
                    "prediction": domain.classes[top],
                    "confidence": float(probs[top]),
                    "probabilities": [{"label": c, "probability": float(p)} for c, p in zip(domain.classes, probs)],
                    "heatmap": _png_base64(overlay(image, heat)),
                    "focus_region": region,
                    "focus_share": share,
                    "latency_ms": elapsed,
                })

        votes = [r["prediction"] for r in results]
        consensus = max(set(votes), key=votes.count)
        agree = votes.count(consensus)
        summary = [
            f"{agree} of {len(results)} models predict {consensus}."
            if agree < len(results) else f"All {len(results)} models agree: {consensus}.",
        ]
        for r in results:
            summary.append(
                f"{r['name']}: {r['prediction']} ({r['confidence']:.0%}). Attention is mostly on the {r['focus_region']} "
                f"of the image; the hottest 20% of pixels hold {r['focus_share']:.0%} of the heatmap."
            )
        return {"domain": domain.id, "consensus": consensus, "agreement": agree, "results": results, "summary": summary}


vision_store = VisionStore()


def domain_summary(domain: VisionDomain) -> Dict:
    metrics = vision_store.metrics.get(domain.id, {})
    models = []
    for arch_id, arch in ARCHS.items():
        m = metrics.get(arch_id)
        if m:
            models.append({k: v for k, v in m.items() if k != "history"})
    return {
        "id": domain.id,
        "name": domain.name,
        "sector": domain.sector,
        "task": domain.task,
        "description": domain.description,
        "data_source": domain.data_source,
        "classes": list(domain.classes),
        "models": models,
    }


__all__ = ["vision_store", "domain_summary", "InvalidImage", "ModelUnavailable", "VISION_DOMAINS", "DATA_DIR"]
